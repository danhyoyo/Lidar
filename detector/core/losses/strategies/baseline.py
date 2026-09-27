"""Baseline loss strategy: Focal loss for classification + unweighted L1 for box regression."""

from typing import Any, Dict
import torch

from core.losses.l1_loss import l1_loss
from core.losses.strategies.base import BaseLossStrategy


class BaselineLossStrategy(BaseLossStrategy):
    """Standard baseline detection loss."""

    def __init__(self, cls_encoding: str, config: Dict[str, Any] = None):
        super().__init__(cls_encoding, config)

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
        loss = components.sum()

        return {
            "loss": loss,
            "cls": cls_loss.detach(),
            "offset": offset_loss.detach(),
            "size": size_loss.detach(),
            "yaw": yaw_loss.detach(),
            "geo": components.new_zeros(()),
        }
