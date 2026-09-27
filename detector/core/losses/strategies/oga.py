"""OGA (Oriented Geometric Alignment & Adaptive) loss strategy."""

from typing import Any, Dict
import torch

from core.losses.l1_loss import smooth_l1_loss
from core.losses.oriented_geometry_loss import OrientedGeometryLoss
from core.losses.uncertainty_weighting import TemperatureSoftmaxUncertainty
from core.losses.strategies.base import BaseLossStrategy


class OgaLossStrategy(BaseLossStrategy):
    """Oriented Geometric Alignment & Adaptive loss combining dual-stream supervision,

    scale-normalized pi-symmetric corner distance, multi-axis projection GIoU,
    and temperature-softmax bounded uncertainty balancing (T-SBUW).
    """

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

        if self.eps <= 0:
            raise ValueError("epsilon must be positive")
        if self.max_abs_log_size <= 0:
            raise ValueError("max_abs_log_size must be positive")
        if self.corner_beta < 0:
            raise ValueError("corner_beta must be non-negative")
        if not (0.0 < self.ema_momentum < 1.0):
            raise ValueError("ema_momentum must be in (0, 1)")

        self.geometry_loss = OrientedGeometryLoss(
            beta=self.corner_beta,
            epsilon=self.eps,
            max_abs_log_size=self.max_abs_log_size,
        )
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

        cls_loss = self._compute_cls_loss(pred["cls"], target["cls"])
        offset_loss = smooth_l1_loss(pred["offset"], target["offset"], target["reg_mask"])
        size_loss = smooth_l1_loss(pred["size"], target["size"], target["reg_mask"])
        yaw_loss = smooth_l1_loss(pred["yaw"], target["yaw"], target["reg_mask"])

        geo_loss, geo_metrics = self.geometry_loss(
            pred["offset"][:, :2],
            pred["size"][:, :2],
            pred["yaw"][:, :2],
            target["offset"][:, :2],
            target["size"][:, :2],
            target["yaw"][:, :2],
            target["reg_mask"],
        )

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
            "corner_dist": geo_metrics["corner_dist"],
            "proj_giou": geo_metrics["proj_giou"],
            "clamp_count": geo_metrics.get("clamp_count", 0),
            "fallback_count": geo_metrics.get("fallback_count", 0),
        }
        for task, w in weights.items():
            loss_dict[f"weight_{task}"] = w

        return loss_dict
