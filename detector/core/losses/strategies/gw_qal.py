"""GW-QAL (Gaussian-Wasserstein & Quality-Aligned Loss) strategy."""

from typing import Any, Dict
import torch
import torch.nn.functional as F

from core.losses.gaussian_wasserstein import (
    box_covariance_2d,
    closed_form_2d_gaussian_wasserstein,
    bounded_gwa_loss,
)
from core.losses.smooth_corner_loss import compute_range_weights
from core.losses.quality_focal_loss import quality_focal_loss
from core.losses.uncertainty_weighting import TemperatureSoftmaxUncertainty
from core.losses.strategies.base import BaseLossStrategy


class GwQalLossStrategy(BaseLossStrategy):
    """Flagship Gaussian-Wasserstein Alignment Loss with Dual-Stream & RDA."""

    TASKS = ("cls", "offset", "size", "yaw", "geo")

    def __init__(self, cls_encoding: str, config: Dict[str, Any] = None):
        super().__init__(cls_encoding, config)
        config = self.config

        self.eps = float(config.get("epsilon", 1e-6))
        self.max_abs_log_size = float(config.get("max_abs_log_size", 10.0))
        self.divisor = float(config.get("divisor", 12.0))
        self.tau_gwa = float(config.get("tau_gwa", 2.0))
        self.tau_sim = float(config.get("tau_sim", 2.0))
        self.beta_q = float(config.get("beta_q", 1.0))
        self.rda_gamma = float(config.get("rda_gamma", 1.5))
        self.rda_alpha = float(config.get("rda_alpha", 2.0))
        self.temperature = float(config.get("temperature", 2.0))
        self.clamp_bound = float(config.get("clamp_bound", 3.0))
        self.ema_momentum = float(config.get("ema_momentum", 0.99))

        if self.eps <= 0:
            raise ValueError("epsilon must be positive")
        if self.max_abs_log_size <= 0:
            raise ValueError("max_abs_log_size must be positive")
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
        B, _, H, W = pred["offset"].shape

        if not positive.any():
            zero = (pred["offset"].sum() + pred["size"].sum() + pred["yaw"].sum()) * 0.0
            cls_loss = self._compute_cls_loss(pred["cls"], target["cls"])
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
                "w2_mean_dist": torch.zeros((), device=device),
                "rda_mean_weight": torch.ones((), device=device),
                **{f"weight_{k}": w for k, w in weights.items()},
            }

        def _pos(x):
            return x.permute(0, 2, 3, 1).reshape(-1, 2)[positive].float()

        tgt_off_pos = _pos(target["offset"])
        tgt_sz_pos = _pos(target["size"])
        tgt_yaw_pos = _pos(target["yaw"])

        pred_off_pos = _pos(pred["offset"])
        pred_sz_pos = _pos(pred["size"])
        pred_yaw_pos = _pos(pred["yaw"])

        # 1. Closed-Form 2D Gaussian Wasserstein Computation
        with torch.autocast(device_type=device.type, enabled=False):
            pred_cov, _ = box_covariance_2d(pred_sz_pos, pred_yaw_pos, self.divisor, self.eps, self.max_abs_log_size)
            tgt_cov, _ = box_covariance_2d(tgt_sz_pos, tgt_yaw_pos, self.divisor, self.eps, self.max_abs_log_size)
            w2_sq = closed_form_2d_gaussian_wasserstein(pred_off_pos, pred_cov, tgt_off_pos, tgt_cov, self.eps)
            geo_loss = bounded_gwa_loss(w2_sq, tgt_sz_pos, self.tau_gwa, self.eps, self.max_abs_log_size)

        # 2. Dynamic Wasserstein Quality Alignment for Classification
        with torch.no_grad():
            clamped_tgt_sz = tgt_sz_pos.clamp(-self.max_abs_log_size, self.max_abs_log_size)
            diag = torch.sqrt(torch.exp(clamped_tgt_sz[:, 0]).square() + torch.exp(clamped_tgt_sz[:, 1]).square()).clamp_min(self.eps)
            wasserstein_similarity = torch.exp(-torch.sqrt(w2_sq.detach() + self.eps) / (self.tau_sim * diag)).clamp(0.0, 1.0)
            quality_map = torch.zeros((B, H, W), dtype=torch.float32, device=device)
            quality_map[target["reg_mask"].bool()] = wasserstein_similarity

        cls_loss = quality_focal_loss(pred["cls"], target["cls"], quality_map, beta=self.beta_q)

        # 3. RDA Spatial Reweighting on Coordinate Streams
        rda_weights = compute_range_weights(tgt_off_pos, r_max=70.4, gamma=self.rda_gamma, alpha=self.rda_alpha)

        diff_off = F.smooth_l1_loss(pred_off_pos, tgt_off_pos, reduction="none").sum(dim=-1)
        offset_loss = (diff_off * rda_weights).mean()

        diff_sz = F.smooth_l1_loss(pred_sz_pos, tgt_sz_pos, reduction="none").sum(dim=-1)
        size_loss = (diff_sz * rda_weights).mean()

        diff_yaw = F.smooth_l1_loss(pred_yaw_pos, tgt_yaw_pos, reduction="none").sum(dim=-1)
        yaw_loss = (diff_yaw * rda_weights).mean()

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
            "w2_mean_dist": torch.sqrt(w2_sq.detach() + self.eps).mean(),
            "rda_mean_weight": rda_weights.mean().detach(),
        }
        for task, w in weights.items():
            loss_dict[f"weight_{task}"] = w

        return loss_dict
