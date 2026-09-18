import torch
import torch.nn as nn
import torch.nn.functional as F

from core.losses.focal_loss import modified_focal_loss, focal_loss
from core.losses.l1_loss import l1_loss


def heteroscedastic_nll(residual, log_var, mask, log_var_min, log_var_max):
    """Masked diagonal Gaussian NLL averaged over positive cells and channels."""
    if not (
        torch.isfinite(residual).all()
        and torch.isfinite(log_var).all()
        and torch.isfinite(mask).all()
    ):
        raise FloatingPointError("non-finite heteroscedastic NLL input")
    used = log_var.float().clamp(log_var_min, log_var_max)
    terms = 0.5 * (torch.exp(-used) * residual.float().square() + used)
    positive = mask.float().unsqueeze(1)
    return (terms * positive).sum() / (
        mask.sum().clamp_min(1.0) * terms.shape[1]
    )


def _footprint_covariance(size, yaw, max_abs_log_size, eps):
    width = torch.exp(size[:, 0].float().clamp(-max_abs_log_size, max_abs_log_size))
    length = torch.exp(size[:, 1].float().clamp(-max_abs_log_size, max_abs_log_size))
    unit_yaw = F.normalize(yaw.float(), dim=1, eps=eps)
    theta = 0.5 * torch.atan2(unit_yaw[:, 1], unit_yaw[:, 0])
    cosine, sine = torch.cos(theta), torch.sin(theta)
    length2, width2 = length.square() / 4, width.square() / 4
    return (
        length2 * cosine.square() + width2 * sine.square(),
        (length2 - width2) * cosine * sine,
        length2 * sine.square() + width2 * cosine.square(),
        length2 * width2,
    )


def gwd_footprint_loss(pred, target, eps=1e-4, max_abs_log_size=10.0):
    """Closed-form FP32 squared 2-D Gaussian Wasserstein footprint loss."""
    pa, pb, pd, det_p = _footprint_covariance(
        pred["size"], pred["yaw"], max_abs_log_size, eps
    )
    ta, tb, td, det_t = _footprint_covariance(
        target["size"], target["yaw"], max_abs_log_size, eps
    )
    trace_product = pa * ta + 2.0 * pb * tb + pd * td
    covariance_term = pa + pd + ta + td - 2.0 * torch.sqrt(
        (
            trace_product
            + 2.0 * torch.sqrt((det_p * det_t).clamp_min(eps * eps))
        ).clamp_min(eps)
    )
    center_term = (
        pred["offset"][:, :2].float() - target["offset"][:, :2].float()
    ).square().sum(1)
    distance2 = (center_term + covariance_term).clamp_min(0.0)
    values = 1.0 - 1.0 / (1.0 + torch.log1p(distance2))
    mask = target["reg_mask"].float()
    return (values * mask).sum() / (mask.sum() + eps)


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
        if self.name not in {
            "baseline",
            "uwag",
            "deterministic",
            "gwd",
            "heteroscedastic",
            "probgeo_uq",
        }:
            raise ValueError(f"Unsupported loss name: {self.name!r}")

        self.geometric_weight = float(config.get("geometric_weight", 0.2))
        self.gwd_weight = float(config.get("gwd_weight", 0.2))
        self.eps = float(config.get("epsilon", 1e-4))
        self.max_abs_log_size = float(config.get("max_abs_log_size", 10.0))
        self.log_var_min = float(config.get("log_var_min", -7.0))
        self.log_var_max = float(config.get("log_var_max", 4.0))
        if self.geometric_weight < 0 or self.gwd_weight < 0:
            raise ValueError("geometric_weight and gwd_weight must be non-negative")
        if self.eps <= 0:
            raise ValueError("epsilon must be positive")
        if self.max_abs_log_size <= 0:
            raise ValueError("max_abs_log_size must be positive")
        if self.log_var_min >= self.log_var_max:
            raise ValueError("log_var_min must be less than log_var_max")

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
        if not torch.isfinite(components).all():
            raise FloatingPointError("non-finite loss component")
        used_log_var = None
        if self.name in {"heteroscedastic", "probgeo_uq"}:
            expected_log_var = torch.cat((pred["offset"], pred["size"]), dim=1)
            if pred["offset"].shape[1] != 3 or pred.get(
                "log_var", torch.empty(0)
            ).shape != expected_log_var.shape:
                raise ValueError(
                    "heteroscedastic loss requires Center3D log_var with six channels"
                )
            residual = torch.cat(
                (
                    pred["offset"] - target["offset"],
                    pred["size"] - target["size"],
                ),
                dim=1,
            )
            reg_loss = heteroscedastic_nll(
                residual,
                pred["log_var"],
                target["reg_mask"],
                self.log_var_min,
                self.log_var_max,
            )
            offset_loss = heteroscedastic_nll(
                residual[:, :3],
                pred["log_var"][:, :3],
                target["reg_mask"],
                self.log_var_min,
                self.log_var_max,
            )
            size_loss = heteroscedastic_nll(
                residual[:, 3:],
                pred["log_var"][:, 3:],
                target["reg_mask"],
                self.log_var_min,
                self.log_var_max,
            )
            positive = target["reg_mask"].bool().unsqueeze(1).expand_as(
                pred["log_var"]
            )
            used_log_var = pred["log_var"].float()[positive]
        else:
            reg_loss = offset_loss + size_loss

        if self.name == "uwag":
            task_loss = (
                torch.exp(-self.log_scales) * components + self.log_scales
            ).sum()
            geometric_loss = self._yaw_aware_bev_iou_loss(pred, target)
            loss = task_loss + geometric_loss
        elif self.name in {"gwd", "probgeo_uq"}:
            if pred["offset"].shape[1] != 3:
                raise ValueError("GWD requires Center3D regression channels")
            geometric_loss = self.gwd_weight * gwd_footprint_loss(
                pred, target, self.eps, self.max_abs_log_size
            )
            loss = cls_loss.float() + reg_loss.float() + yaw_loss.float() + geometric_loss
        elif self.name == "heteroscedastic":
            geometric_loss = components.new_zeros(())
            loss = cls_loss.float() + reg_loss.float() + yaw_loss.float()
        else:
            geometric_loss = components.new_zeros(())
            loss = components.sum()
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite total loss")

        loss_dict = {
            "loss": loss,
            "cls": cls_loss.item(),
            "offset": offset_loss.item(),
            "size": size_loss.item(),
            "yaw": yaw_loss.item(),
            "geo": geometric_loss.item(),
        }
        if self.name == "uwag":
            for task, log_scale in zip(self.TASKS, self.log_scales):
                loss_dict[f"weight_{task}"] = torch.exp(
                    -log_scale.detach()
                ).item()
        if used_log_var is not None:
            loss_dict["log_var_saturation_min"] = (
                (used_log_var <= self.log_var_min).float().mean().item()
                if used_log_var.numel()
                else 0.0
            )
            loss_dict["log_var_saturation_max"] = (
                (used_log_var >= self.log_var_max).float().mean().item()
                if used_log_var.numel()
                else 0.0
            )

        return loss_dict
