"""Composite Loss Function Façade delegating to modular Loss Strategies."""

from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..detection_config import resolve_box_mode
from ..backbone_config import finite_scalar

from core.losses.strategies import (
    BaseLossStrategy,
    build_loss_strategy,
    get_available_loss_strategies,
)


class LossFunction(nn.Module):
    """Façade for modular detection loss strategies (baseline, uwag, oga).

    Delegates composite loss computation to the configured LossStrategy while
    providing 100% backward compatibility for:
    - checkpoint state dictionaries (both legacy flat keys and strategy-prefixed keys).
    - direct attribute access (e.g. criterion.log_scales, criterion.geometry_loss, criterion.weighting).
    - existing trainer and evaluation pipelines.
    """

    TASKS = ("cls", "offset", "size", "yaw")

    def __init__(self, cls_encoding: str, config: Dict[str, Any] = None, *, box_mode="bev"):
        super().__init__()
        self.cls_encoding = cls_encoding
        config = config or {}
        self.name = str(config.get("name", "baseline")).lower()
        self.box_mode = resolve_box_mode(box_mode)
        self.vertical_loss_weight = None
        if self.box_mode == "3d":
            if self.name not in ("baseline", "oga"):
                raise ValueError("3D supports only baseline or oga")
            self.vertical_loss_weight = finite_scalar(config.get("vertical_loss_weight"),
                                                     "vertical_loss_weight", positive=True)

        # Build active strategy via the registry
        self.strategy = build_loss_strategy(self.name, cls_encoding, config)

    @property
    def log_scales(self):
        if self.name == "baseline":
            return torch.empty(0, dtype=torch.float32)
        return getattr(self.strategy, "log_scales", None)

    @property
    def geometry_loss(self):
        return getattr(self.strategy, "geometry_loss", None)

    @property
    def weighting(self):
        return getattr(self.strategy, "weighting", None)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            strategy = self.__dict__.get("_modules", {}).get("strategy")
            if strategy is not None:
                return getattr(strategy, name)
            raise

    def forward(
        self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        self.validate_vertical_inputs(pred, target)
        loss_dict = self.strategy(pred, target)
        if self.box_mode == "3d":
            mask = target["reg_mask"].bool()
            # Select before arithmetic: unsupervised NaNs cannot contaminate loss.
            predicted = pred["vertical"].float().permute(0, 2, 3, 1)[mask]
            assigned = target["vertical"].float().permute(0, 2, 3, 1)[mask]
            vertical = (F.smooth_l1_loss(predicted, assigned, reduction="mean")
                        if predicted.numel() else predicted.sum() * 0.)
            loss_dict["vertical"] = vertical
            loss_dict["loss"] = loss_dict["loss"] + self.vertical_loss_weight * vertical
        if not torch.isfinite(loss_dict["loss"]):
            raise FloatingPointError("non-finite loss")
        return loss_dict

    def validate_vertical_inputs(self, pred, target):
        """Preflight every group before adaptive BEV criteria can update state."""
        if self.box_mode == "bev":
            if "vertical" in pred or "vertical" in target:
                raise ValueError("BEV vertical loss is disabled; explicitly select box_mode=3d")
            return
        expected = pred["offset"].shape
        if len(expected) != 4 or expected[1] != 2:
            raise ValueError("3D uses two-channel BEV branches plus a separate vertical branch")
        mask = target["reg_mask"]
        if mask.shape != (expected[0], *expected[2:]):
            raise ValueError("vertical reg_mask shape mismatch")
        if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
            raise ValueError("vertical reg_mask must contain finite binary values")
        for mapping in (pred, target):
            value = mapping.get("vertical")
            if not torch.is_tensor(value) or value.shape != expected or not value.is_floating_point():
                raise ValueError("vertical must have floating-point shape [B,2,H,W]")
            if value.device != mask.device or not torch.isfinite(value.permute(0, 2, 3, 1)[mask.bool()]).all():
                raise ValueError("vertical supervised values must be finite on the mask device")

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs,
    ):
        """Accept flat legacy loss keys during direct or parent-module loading."""
        for key in self.strategy.state_dict():
            legacy_key = f"{prefix}{key}"
            current_key = f"{prefix}strategy.{key}"
            if legacy_key in state_dict:
                if current_key not in state_dict:
                    state_dict[current_key] = state_dict[legacy_key]
                del state_dict[legacy_key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs,
        )


def build_loss_function(cls_encoding, config=None, *, task_groups=None, head_mode=None, box_mode="bev"):
    """Keep legacy construction unchanged; resolve explicit grouped metadata.

    Full data/model/loss agreement is validated by the pipeline before this
    objective-only factory. Training wiring switches to this factory in task36.
    """
    if head_mode is None:
        head_mode = "grouped" if task_groups is not None else "legacy_single"
    if not isinstance(head_mode, str) or head_mode.lower() not in ("legacy_single", "grouped"):
        raise ValueError("head_mode must be legacy_single or grouped")
    if head_mode.lower() == "legacy_single":
        if task_groups is not None or (config is not None and "group_weights" in config):
            raise ValueError("task_groups/group_weights cannot be ignored in legacy_single")
        return LossFunction(cls_encoding, config, box_mode=box_mode)
    if task_groups is None:
        raise ValueError("Grouped loss requires explicit resolved task_groups")
    from .grouped import GroupedLossFunction
    return GroupedLossFunction(cls_encoding, config, task_groups=task_groups, box_mode=box_mode)
