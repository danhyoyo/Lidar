"""Baseline loss strategy: Focal loss for classification + unweighted L1 for box regression."""

from typing import Any, Dict
import torch
import torch.nn.functional as F

from core.losses.l1_loss import l1_loss
from core.losses.strategies.base import BaseLossStrategy
from core.losses.iou_targets import compute_iou_targets


class BaselineLossStrategy(BaseLossStrategy):
    """Standard baseline detection loss with optional IoU head supervision."""

    def __init__(self, cls_encoding: str, config: Dict[str, Any] = None):
        super().__init__(cls_encoding, config)
        config = self.config
        self.use_iou = bool(config.get("use_iou", False))
        self.iou_target_type = str(config.get("iou_target_type", "mgiou"))
        self.iou_loss_weight = float(config.get("iou_loss_weight", 1.0))
        self.eps = float(config.get("epsilon", 1e-6))
        self.max_abs_log_size = float(config.get("max_abs_log_size", 10.0))

    def forward(
        self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        self._validate_inputs(pred, target)

        cls_loss = self._compute_cls_loss(pred["cls"], target["cls"])
        offset_loss = l1_loss(pred["offset"], target["offset"], target["reg_mask"])
        size_loss = l1_loss(pred["size"], target["size"], target["reg_mask"])
        yaw_loss = l1_loss(pred["yaw"], target["yaw"], target["reg_mask"])

        comp_list = [cls_loss.float(), offset_loss.float(), size_loss.float(), yaw_loss.float()]
        ret_dict = {
            "cls": cls_loss.detach(),
            "offset": offset_loss.detach(),
            "size": size_loss.detach(),
            "yaw": yaw_loss.detach(),
            "geo": cls_loss.new_zeros(()),
        }

        if self.use_iou:
            if "iou" not in pred:
                raise KeyError("Missing required prediction head: 'iou' when use_iou=True")
            pos_mask = target["reg_mask"].bool()
            if pos_mask.any():
                iou_target = compute_iou_targets(
                    pred,
                    target,
                    method=self.iou_target_type,
                    epsilon=self.eps,
                    max_abs_log_size=self.max_abs_log_size,
                )
                iou_loss = F.binary_cross_entropy_with_logits(
                    pred["iou"][:, 0][pos_mask],
                    iou_target[pos_mask],
                ) * self.iou_loss_weight
            else:
                iou_loss = pred["iou"].sum() * 0.0
            comp_list.append(iou_loss.float())
            ret_dict["iou"] = iou_loss.detach()

        loss = torch.stack(comp_list).sum()
        ret_dict["loss"] = loss
        return ret_dict
