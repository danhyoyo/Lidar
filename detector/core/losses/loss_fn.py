"""Composite Loss Function Façade delegating to modular Loss Strategies."""

from typing import Any, Dict
import torch
import torch.nn as nn

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

    def __init__(self, cls_encoding: str, config: Dict[str, Any] = None):
        super().__init__()
        self.cls_encoding = cls_encoding
        config = config or {}
        self.name = str(config.get("name", "baseline")).lower()

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
        loss_dict = self.strategy(pred, target)
        if not torch.isfinite(loss_dict["loss"]):
            raise FloatingPointError("non-finite loss")
        return loss_dict

    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True):
        """Remap legacy checkpoint keys missing 'strategy.' prefix for backward compatibility."""
        remapped = {}
        target_keys = set(self.state_dict().keys())

        for k, v in state_dict.items():
            if k in target_keys:
                remapped[k] = v
            elif f"strategy.{k}" in target_keys:
                # Remap legacy un-prefixed key (e.g. 'log_scales' -> 'strategy.log_scales')
                remapped[f"strategy.{k}"] = v
            elif k.startswith("strategy.") and k[9:] in target_keys:
                remapped[k[9:]] = v
            else:
                remapped[k] = v

        return super().load_state_dict(remapped, strict=strict)
