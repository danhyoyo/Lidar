"""Base loss strategy interface and shared validation logic."""

from abc import ABC, abstractmethod
from typing import Any, Dict
import torch
import torch.nn as nn

from core.losses.focal_loss import modified_focal_loss, focal_loss


class BaseLossStrategy(nn.Module, ABC):
    """Abstract base class for all detection loss strategies."""

    def __init__(self, cls_encoding: str, config: Dict[str, Any] = None):
        super().__init__()
        self.cls_encoding = str(cls_encoding).lower()
        self.config = config or {}

    def _validate_inputs(self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]):
        """Validate shapes, channel counts, and required keys across predictions and targets."""
        for name in ("cls", "offset", "size", "yaw"):
            if name not in pred:
                raise KeyError(f"Missing required prediction head: {name!r}")
            if name not in target:
                raise KeyError(f"Missing required target head: {name!r}")

        if "reg_mask" not in target:
            raise KeyError("target dictionary must contain 'reg_mask'")

        for name in ("offset", "size", "yaw"):
            if pred[name].shape != target[name].shape:
                raise ValueError(
                    f"{name} prediction/target shape mismatch: "
                    f"{tuple(pred[name].shape)} != {tuple(target[name].shape)}"
                )

        expected_mask_shape = (
            pred["offset"].shape[0],
            pred["offset"].shape[2],
            pred["offset"].shape[3],
        )
        if target["reg_mask"].shape != expected_mask_shape:
            raise ValueError(
                f"reg_mask shape mismatch: {tuple(target['reg_mask'].shape)} != {expected_mask_shape}"
            )

        expected_cls_shape = (
            pred["cls"].shape[:1] + pred["cls"].shape[2:]
            if self.cls_encoding == "binary" else pred["cls"].shape
        )
        if target["cls"].shape != expected_cls_shape:
            raise ValueError(
                f"cls prediction/target shape mismatch: {tuple(pred['cls'].shape)} "
                f"is incompatible with {tuple(target['cls'].shape)}"
            )
        if pred["offset"].shape[1] not in {2, 3} or pred["size"].shape[1] != pred["offset"].shape[1]:
            raise ValueError("offset and size must both have 2 or 3 channels")
        if pred["yaw"].shape[1] != 2:
            raise ValueError("yaw must have 2 channels")

    def _compute_cls_loss(self, pred_cls: torch.Tensor, target_cls: torch.Tensor) -> torch.Tensor:
        """Compute binary or Gaussian focal loss based on cls_encoding."""
        if self.cls_encoding == "binary":
            return focal_loss(pred_cls, target_cls)
        return modified_focal_loss(pred_cls, target_cls)

    @abstractmethod
    def forward(
        self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Compute the composite objective and return a telemetry dictionary."""
        pass
