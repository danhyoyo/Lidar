"""UWAG (Uncertainty-Weighted Adaptive Geometry) loss strategy."""

from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.losses.l1_loss import l1_loss
from core.losses.strategies.base import BaseLossStrategy


class UwagLossStrategy(BaseLossStrategy):
    """Homoscedastic uncertainty weighting with axis-aligned footprint IoU."""

    TASKS = ("cls", "offset", "size", "yaw")

    def __init__(self, cls_encoding: str, config: Dict[str, Any] = None):
        super().__init__(cls_encoding, config)
        config = self.config

        self.geometric_weight = float(config.get("geometric_weight", 0.2))
        self.eps = float(config.get("epsilon", 1e-4))
        self.max_abs_log_size = float(config.get("max_abs_log_size", 10.0))

        if self.geometric_weight < 0:
            raise ValueError("geometric_weight must be non-negative")
        if self.eps <= 0:
            raise ValueError("epsilon must be positive")
        if self.max_abs_log_size <= 0:
            raise ValueError("max_abs_log_size must be positive")

        initial = config.get("initial_log_scales", [0.0] * len(self.TASKS))
        if len(initial) != len(self.TASKS):
            raise ValueError("initial_log_scales must contain four values")
        self.log_scales = nn.Parameter(torch.tensor(initial, dtype=torch.float32))

    def _yaw_aware_bev_iou_loss(
        self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Differentiable footprint IoU multiplied by doubled-yaw agreement."""
        mask = target["reg_mask"].float()
        pred_offset = pred["offset"][:, :2].float()
        target_offset = target["offset"][:, :2].float()
        pred_size = torch.exp(
            pred["size"][:, :2].float().clamp(
                -self.max_abs_log_size, self.max_abs_log_size
            )
        )
        target_size = torch.exp(
            target["size"][:, :2].float().clamp(
                -self.max_abs_log_size, self.max_abs_log_size
            )
        )

        pred_min = pred_offset - 0.5 * pred_size
        pred_max = pred_offset + 0.5 * pred_size
        target_min = target_offset - 0.5 * target_size
        target_max = target_offset + 0.5 * target_size
        intersection_size = (
            torch.minimum(pred_max, target_max)
            - torch.maximum(pred_min, target_min)
        ).clamp_min(0.0)
        intersection = intersection_size[:, 0] * intersection_size[:, 1]
        pred_area = pred_size[:, 0] * pred_size[:, 1]
        target_area = target_size[:, 0] * target_size[:, 1]
        bev_iou = intersection / (
            pred_area + target_area - intersection + self.eps
        )

        pred_yaw = F.normalize(pred["yaw"].float(), dim=1, eps=self.eps)
        target_yaw = F.normalize(target["yaw"].float(), dim=1, eps=self.eps)
        doubled_yaw_similarity = (
            pred_yaw * target_yaw
        ).sum(dim=1).clamp(-1.0, 1.0)
        yaw_factor = 0.5 * (1.0 + doubled_yaw_similarity)
        oriented_overlap = bev_iou * yaw_factor
        return self.geometric_weight * (
            ((1.0 - oriented_overlap) * mask).sum() / (mask.sum() + self.eps)
        )

    def forward(
        self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        self._validate_inputs(pred, target)

        cls_loss = self._compute_cls_loss(pred["cls"], target["cls"])
        offset_loss = l1_loss(pred["offset"], target["offset"], target["reg_mask"])
        size_loss = l1_loss(pred["size"], target["size"], target["reg_mask"])
        yaw_loss = l1_loss(pred["yaw"], target["yaw"], target["reg_mask"])

        components = torch.stack(
            [cls_loss.float(), offset_loss.float(), size_loss.float(), yaw_loss.float()]
        )
        task_loss = (
            torch.exp(-self.log_scales) * components + self.log_scales
        ).sum()
        geometric_loss = self._yaw_aware_bev_iou_loss(pred, target)
        loss = task_loss + geometric_loss

        loss_dict = {
            "loss": loss,
            "cls": cls_loss.detach(),
            "offset": offset_loss.detach(),
            "size": size_loss.detach(),
            "yaw": yaw_loss.detach(),
            "geo": geometric_loss.detach(),
        }
        for task, log_scale in zip(self.TASKS, self.log_scales):
            loss_dict[f"weight_{task}"] = torch.exp(-log_scale.detach())

        return loss_dict
