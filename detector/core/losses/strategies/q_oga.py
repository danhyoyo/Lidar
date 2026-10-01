"""Q-OGA (Quality-Aligned & Range-Adaptive Oriented Geometric Alignment) loss strategy."""

from typing import Any, Dict
import torch
import torch.nn.functional as F

from core.losses.oriented_geometry_loss import box_corners, multiaxis_projection_giou
from core.losses.smooth_corner_loss import (
    smooth_pi_symmetric_corner_distance,
    compute_range_weights,
)
from core.losses.quality_focal_loss import quality_focal_loss
from core.losses.uncertainty_weighting import TemperatureSoftmaxUncertainty
from core.losses.strategies.base import BaseLossStrategy
from core.losses.strategies.registry import register_loss_strategy
from core.losses.iou_targets import compute_mgiou_targets


class QOgaLossStrategy(BaseLossStrategy):
    """Quality-Aligned, Range-Adaptive, and C^inf Smooth Soft-Min OGA Loss."""

    TASKS = ("cls", "offset", "size", "yaw", "geo")

    def __init__(self, cls_encoding: str, config: Dict[str, Any] = None):
        super().__init__(cls_encoding, config)
        config = self.config

        self.eps = float(config.get("epsilon", 1e-6))
        self.max_abs_log_size = float(config.get("max_abs_log_size", 10.0))
        self.corner_beta = float(config.get("corner_beta", 1.0))
        self.temperature = float(config.get("temperature", 2.0))
        self.clamp_bound = float(config.get("clamp_bound", 3.0))
        self.ema_momentum = float(config.get("ema_momentum", 0.99))
        self.soft_min_tau = float(config.get("soft_min_tau", 0.05))
        self.rda_gamma = float(config.get("rda_gamma", 1.5))
        self.rda_alpha = float(config.get("rda_alpha", 2.0))
        self.qcfa_beta = float(config.get("qcfa_beta", 1.0))

        if self.eps <= 0:
            raise ValueError("epsilon must be positive")
        if self.max_abs_log_size <= 0:
            raise ValueError("max_abs_log_size must be positive")
        if self.corner_beta < 0:
            raise ValueError("corner_beta must be non-negative")
        if not (0.0 < self.ema_momentum < 1.0):
            raise ValueError("ema_momentum must be in (0, 1)")

        self.weighting = TemperatureSoftmaxUncertainty(
            task_names=self.TASKS,
            temperature=self.temperature,
            clamp_bound=self.clamp_bound,
            ema_momentum=self.ema_momentum,
        )

    def forward(
        self, pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        self._validate_inputs(pred, target)

        positive = target["reg_mask"].reshape(-1).bool()
        device = pred["offset"].device

        # Compute dynamic MGIoU quality scores (detached)
        with torch.no_grad():
            mgiou_quality = compute_mgiou_targets(
                pred["offset"][:, :2],
                pred["size"][:, :2],
                pred["yaw"][:, :2],
                target["offset"][:, :2],
                target["size"][:, :2],
                target["yaw"][:, :2],
                target["reg_mask"],
                epsilon=self.eps,
                max_abs_log_size=self.max_abs_log_size,
            )

        # 1. Quality-coupled classification loss
        cls_loss = quality_focal_loss(
            pred["cls"], target["cls"], mgiou_quality, beta=self.qcfa_beta
        )

        if not positive.any():
            zero = (pred["offset"].sum() + pred["size"].sum() + pred["yaw"].sum()) * 0.0
            task_losses = {
                "cls": cls_loss,
                "offset": zero,
                "size": zero,
                "yaw": zero,
                "geo": zero,
            }
            loss, weights = self.weighting(task_losses)
            return {
                "loss": loss,
                "cls": cls_loss.detach(),
                "offset": zero.detach(),
                "size": zero.detach(),
                "yaw": zero.detach(),
                "geo": zero.detach(),
                "corner_dist": torch.zeros((), device=device),
                "proj_giou": torch.zeros((), device=device),
                "rda_mean_weight": torch.ones((), device=device),
                **{f"weight_{k}": w for k, w in weights.items()},
            }

        # 2. Coordinate losses with Range-Adaptive Reweighting
        def _pos(x):
            return x.permute(0, 2, 3, 1).reshape(-1, 2)[positive].float()

        tgt_off_pos = _pos(target["offset"])
        rda_weights = compute_range_weights(
            tgt_off_pos, r_max=70.4, gamma=self.rda_gamma, alpha=self.rda_alpha
        )

        # Weighted Smooth-L1 per coordinate task
        diff_off = F.smooth_l1_loss(_pos(pred["offset"]), tgt_off_pos, reduction="none").sum(dim=-1)
        offset_loss = (diff_off * rda_weights).mean()

        diff_sz = F.smooth_l1_loss(_pos(pred["size"]), _pos(target["size"]), reduction="none").sum(dim=-1)
        size_loss = (diff_sz * rda_weights).mean()

        diff_yaw = F.smooth_l1_loss(_pos(pred["yaw"]), _pos(target["yaw"]), reduction="none").sum(dim=-1)
        yaw_loss = (diff_yaw * rda_weights).mean()

        # 3. Geometry loss with C^inf Soft-Min Corner Distance
        with torch.autocast(device_type=device.type, enabled=False):
            pred_c, pred_ax, _, _ = box_corners(
                _pos(pred["offset"]), _pos(pred["size"]), _pos(pred["yaw"]), self.eps, self.max_abs_log_size
            )
            tgt_c, tgt_ax, _, _ = box_corners(
                tgt_off_pos, _pos(target["size"]), _pos(target["yaw"]), self.eps, self.max_abs_log_size
            )
            proj_loss = multiaxis_projection_giou(pred_c, pred_ax, tgt_c, tgt_ax, self.eps)
            corner_loss = smooth_pi_symmetric_corner_distance(
                pred_c, tgt_c, _pos(target["size"]), tau=self.soft_min_tau, epsilon=self.eps, max_abs_log_size=self.max_abs_log_size
            )
            geo_loss = proj_loss + self.corner_beta * corner_loss

        task_losses = {
            "cls": cls_loss,
            "offset": offset_loss,
            "size": size_loss,
            "yaw": yaw_loss,
            "geo": geo_loss,
        }
        loss, weights = self.weighting(task_losses)

        loss_dict = {
            "loss": loss,
            "cls": cls_loss.detach(),
            "offset": offset_loss.detach(),
            "size": size_loss.detach(),
            "yaw": yaw_loss.detach(),
            "geo": geo_loss.detach(),
            "corner_dist": corner_loss.detach(),
            "proj_giou": proj_loss.detach(),
            "rda_mean_weight": rda_weights.mean().detach(),
        }
        for task, w in weights.items():
            loss_dict[f"weight_{task}"] = w

        return loss_dict
