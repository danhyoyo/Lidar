import torch
import torch.nn as nn
import torch.nn.functional as F

from core.losses.focal_loss import modified_focal_loss, focal_loss
from core.losses.l1_loss import l1_loss


class LossFunction(nn.Module):
    """Baseline loss or the thesis UWAG composite objective.

    UWAG combines learnable homoscedastic task weights, the dataset's
    object-size-adaptive Gaussian classification target, and a yaw-aware BEV
    overlap penalty. The one-argument constructor remains compatible with the
    legacy training agent and selects the baseline objective.
    """

    TASKS = ("cls", "offset", "size", "yaw")

    def __init__(self, cls_encoding, config=None):
        super(LossFunction, self).__init__()
        self.cls_encoding = cls_encoding
        config = config or {}
        self.name = str(config.get("name", "baseline")).lower()
        if self.name not in {"baseline", "uwag"}:
            raise ValueError(f"Unsupported loss name: {self.name!r}")

        self.geometric_weight = float(config.get("geometric_weight", 0.2))
        self.eps = float(config.get("epsilon", 1e-4))
        self.max_abs_log_size = float(config.get("max_abs_log_size", 10.0))
        if self.geometric_weight < 0:
            raise ValueError("geometric_weight must be non-negative")
        if self.eps <= 0:
            raise ValueError("epsilon must be positive")
        if self.max_abs_log_size <= 0:
            raise ValueError("max_abs_log_size must be positive")

        if self.name == "uwag":
            initial = config.get("initial_log_scales", [0.0] * len(self.TASKS))
            if len(initial) != len(self.TASKS):
                raise ValueError("initial_log_scales must contain four values")
            self.log_scales = nn.Parameter(torch.tensor(initial, dtype=torch.float32))
        else:
            self.register_buffer(
                "log_scales", torch.empty(0, dtype=torch.float32), persistent=False
            )

    def _yaw_aware_bev_iou_loss(self, pred, target):
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

    def forward(self, pred, target):
        for name in ("offset", "size", "yaw"):
            if pred[name].shape != target[name].shape:
                raise ValueError(
                    f"{name} prediction/target shape mismatch: "
                    f"{tuple(pred[name].shape)} != {tuple(target[name].shape)}"
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

        if self.cls_encoding == "binary":
            cls_loss = focal_loss(pred["cls"], target["cls"])
        else:
            cls_loss = modified_focal_loss(pred["cls"], target["cls"])
        offset_loss = l1_loss(pred["offset"], target["offset"], target["reg_mask"])
        size_loss = l1_loss(pred["size"], target["size"], target["reg_mask"])
        yaw_loss = l1_loss(pred["yaw"], target["yaw"], target["reg_mask"])

        components = torch.stack(
            [cls_loss.float(), offset_loss.float(), size_loss.float(), yaw_loss.float()]
        )
        if self.name == "uwag":
            task_loss = (
                torch.exp(-self.log_scales) * components + self.log_scales
            ).sum()
            geometric_loss = self._yaw_aware_bev_iou_loss(pred, target)
            loss = task_loss + geometric_loss
        else:
            geometric_loss = components.new_zeros(())
            loss = components.sum()
        if not torch.isfinite(torch.cat((components, loss.reshape(1)))).all():
            raise FloatingPointError("non-finite loss")

        loss_dict = {
            "loss": loss,
            # Keep telemetry on-device.  The trainer transfers the aggregated
            # scalar once per epoch instead of synchronizing for every batch.
            "cls": cls_loss.detach(),
            "offset": offset_loss.detach(),
            "size": size_loss.detach(),
            "yaw": yaw_loss.detach(),
            "geo": geometric_loss.detach(),
        }
        if self.name == "uwag":
            for task, log_scale in zip(self.TASKS, self.log_scales):
                loss_dict[f"weight_{task}"] = torch.exp(
                    -log_scale.detach()
                )

        return loss_dict
