"""Training-loss orchestration for the BEV detector.

``LossFunction`` owns classification, the legacy UWAG compatibility path, and
the fixed B-series reduction/weighting policy. Rotated-box mathematics lives
in dedicated ``nn.Module`` criteria in :mod:`gaussian_geometry_loss`; this
module deliberately does not duplicate covariance or matrix-solve code.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.losses.focal_loss import modified_focal_loss, focal_loss
from core.losses.gaussian_geometry_loss import KFIoULoss, KLDLoss, ProbIoULoss
from core.losses.l1_loss import l1_loss
from core.losses.mgiou_loss import MGIoULoss


class LossFunction(nn.Module):
    """Select the legacy objective or one fixed B-series regression objective.

    Configurations without ``regression`` use the historical schema exactly:
    ``name=baseline`` is focal classification plus three masked-L1 terms, and
    ``name=uwag`` additionally owns its four trainable uncertainty log-scales
    and yaw-aware auxiliary loss. This keeps old checkpoints' state dicts
    compatible.

    Configurations with ``regression`` use the B-series fixed schema. B1
    applies a fixed coefficient to masked L1; B2--B4 hold exactly one child
    geometry criterion (KFIoU, ProbIoU, KLD, or the explicitly authorized B5
    candidate MGIoU), so this orchestrator never reimplements geometry maths.
    """

    TASKS = ("cls", "offset", "size", "yaw")
    _FIXED_SCHEMA_KEYS = frozenset(
        {
            "regression",
            "weighting",
            "regression_weight",
            "covariance_epsilon",
            "max_abs_log_size",
            "probiou_switch_fraction",
            "kld_tau",
        }
    )

    def __init__(self, cls_encoding, config=None):
        super().__init__()
        self.cls_encoding = cls_encoding
        config = dict(config or {})

        # Presence of ``regression`` is the schema discriminator, rather than
        # a new default, so old JSON retains objective and state_dict layout.
        # Either fixed-schema selector must opt into strict fixed validation.
        # In particular, a malformed {"weighting": "fixed"} config may not
        # silently fall through to the historical baseline objective.
        self.is_fixed_schema = "regression" in config or "weighting" in config
        self.regression = None
        self.geometry_criterion = None
        if self.is_fixed_schema:
            self._initialize_fixed_schema(config)
        else:
            self._initialize_legacy_schema(config)

    @staticmethod
    def _finite_float(config, key, *, lower=None, upper=None):
        """Read one scalar config field and reject non-finite/out-of-range values."""
        try:
            value = float(config[key])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{key} must be a finite numeric value") from error
        if not math.isfinite(value):
            raise ValueError(f"{key} must be finite, got {value}")
        if lower is not None and value < lower:
            raise ValueError(f"{key} must be at least {lower}, got {value}")
        if upper is not None and value > upper:
            raise ValueError(f"{key} must be at most {upper}, got {value}")
        return value

    def _initialize_legacy_schema(self, config):
        """Preserve the pre-B-series baseline/UWAG behavior and state layout."""
        self.name = str(config.get("name", "baseline")).lower()
        if self.name not in {"baseline", "uwag"}:
            raise ValueError(f"Unsupported loss name: {self.name!r}")

        # Retain the old defaults and checks verbatim for historical configs.
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

    def _initialize_fixed_schema(self, config):
        """Validate B1--B5 and instantiate its single geometry child, if any."""
        unknown = set(config).difference(self._FIXED_SCHEMA_KEYS)
        if unknown:
            raise ValueError(
                "Unsupported fixed-loss config keys: " + ", ".join(sorted(unknown))
            )
        missing = self._FIXED_SCHEMA_KEYS.difference(config)
        if missing:
            raise ValueError(
                "Fixed-loss config is missing required keys: " + ", ".join(sorted(missing))
            )

        self.regression = str(config["regression"]).lower()
        if self.regression not in {"l1", "kfiou", "probiou", "kld", "mgiou"}:
            raise ValueError(f"Unsupported fixed regression loss: {self.regression!r}")
        if config["weighting"] != "fixed":
            raise ValueError("B-series regression weighting must be exactly 'fixed'")

        self.name = "fixed"
        self.regression_weight = self._finite_float(
            config, "regression_weight", lower=0.0
        )
        self.eps = self._finite_float(config, "covariance_epsilon")
        if self.eps <= 0.0:
            raise ValueError("covariance_epsilon must be greater than 0")
        self.max_abs_log_size = self._finite_float(config, "max_abs_log_size")
        if self.max_abs_log_size <= 0.0:
            raise ValueError("max_abs_log_size must be greater than 0")
        self.probiou_switch_fraction = self._finite_float(
            config, "probiou_switch_fraction", lower=0.0, upper=1.0
        )
        self.kld_tau = self._finite_float(config, "kld_tau", lower=1.0)

        # This is a registered child Module, not a function pointer/factory.
        if self.regression == "kfiou":
            self.geometry_criterion = KFIoULoss(
                epsilon=self.eps, max_abs_log_size=self.max_abs_log_size
            )
        elif self.regression == "probiou":
            self.geometry_criterion = ProbIoULoss(
                switch_fraction=self.probiou_switch_fraction,
                epsilon=self.eps,
                max_abs_log_size=self.max_abs_log_size,
            )
        elif self.regression == "kld":
            self.geometry_criterion = KLDLoss(
                tau=self.kld_tau,
                epsilon=self.eps,
                max_abs_log_size=self.max_abs_log_size,
            )
        elif self.regression == "mgiou":
            self.geometry_criterion = MGIoULoss(
                epsilon=self.eps, max_abs_log_size=self.max_abs_log_size
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
        bev_iou = intersection / (pred_area + target_area - intersection + self.eps)

        pred_yaw = F.normalize(pred["yaw"].float(), dim=1, eps=self.eps)
        target_yaw = F.normalize(target["yaw"].float(), dim=1, eps=self.eps)
        doubled_yaw_similarity = (pred_yaw * target_yaw).sum(dim=1).clamp(-1.0, 1.0)
        yaw_factor = 0.5 * (1.0 + doubled_yaw_similarity)
        oriented_overlap = bev_iou * yaw_factor
        return self.geometric_weight * (
            ((1.0 - oriented_overlap) * mask).sum() / (mask.sum() + self.eps)
        )

    def _validate_shapes(self, pred, target):
        for name in ("offset", "size", "yaw"):
            if pred[name].shape != target[name].shape:
                raise ValueError(
                    f"{name} prediction/target shape mismatch: "
                    f"{tuple(pred[name].shape)} != {tuple(target[name].shape)}"
                )
        expected_cls_shape = (
            pred["cls"].shape[:1] + pred["cls"].shape[2:]
            if self.cls_encoding == "binary"
            else pred["cls"].shape
        )
        if target["cls"].shape != expected_cls_shape:
            raise ValueError(
                f"cls prediction/target shape mismatch: {tuple(pred['cls'].shape)} "
                f"is incompatible with {tuple(target['cls'].shape)}"
            )
        if (
            pred["offset"].shape[1] not in {2, 3}
            or pred["size"].shape[1] != pred["offset"].shape[1]
        ):
            raise ValueError("offset and size must both have 2 or 3 channels")
        if pred["yaw"].shape[1] != 2:
            raise ValueError("yaw must have 2 channels")

    def _classification_loss(self, pred, target):
        if self.cls_encoding == "binary":
            return focal_loss(pred["cls"], target["cls"])
        return modified_focal_loss(pred["cls"], target["cls"])

    def _geometry_loss(self, pred, target, progress):
        """Call the selected geometry criterion on BEV channels only."""
        arguments = (
            pred["offset"][:, :2],
            pred["size"][:, :2],
            pred["yaw"][:, :2],
            target["offset"][:, :2],
            target["size"][:, :2],
            target["yaw"][:, :2],
            target["reg_mask"],
        )
        if self.regression == "probiou":
            return self.geometry_criterion(*arguments, progress=progress)
        return self.geometry_criterion(*arguments)

    def forward(self, pred, target, progress=0.0):
        self._validate_shapes(pred, target)
        cls_loss = self._classification_loss(pred, target)
        offset_loss = l1_loss(pred["offset"], target["offset"], target["reg_mask"])
        size_loss = l1_loss(pred["size"], target["size"], target["reg_mask"])
        yaw_loss = l1_loss(pred["yaw"], target["yaw"], target["reg_mask"])
        components = torch.stack(
            [cls_loss.float(), offset_loss.float(), size_loss.float(), yaw_loss.float()]
        )
        if not torch.isfinite(components).all():
            raise FloatingPointError("non-finite loss component")

        clamp_count = None
        yaw_fallback_count = None
        if not self.is_fixed_schema:
            if self.name == "uwag":
                task_loss = (
                    torch.exp(-self.log_scales) * components + self.log_scales
                ).sum()
                geometric_loss = self._yaw_aware_bev_iou_loss(pred, target)
                loss = task_loss + geometric_loss
            else:
                geometric_loss = components.new_zeros(())
                loss = components.sum()
        elif self.regression == "l1":
            geometric_loss = components.new_zeros(())
            loss = cls_loss + self.regression_weight * (
                offset_loss + size_loss + yaw_loss
            )
        else:
            try:
                progress_value = float(progress)
            except (TypeError, ValueError) as error:
                raise ValueError("progress must be a finite scalar") from error
            if not math.isfinite(progress_value):
                raise ValueError("progress must be finite")
            geometry_result = self._geometry_loss(pred, target, progress_value)
            if self.regression == "mgiou":
                geometry_value, clamp_count, yaw_fallback_count = geometry_result
            else:
                geometry_value, clamp_count = geometry_result
            if not torch.isfinite(geometry_value).all():
                raise FloatingPointError("non-finite geometry loss")
            # ``geo`` is the weighted contribution, matching the legacy
            # auxiliary diagnostic and keeping ``loss == cls + geo``.
            geometric_loss = self.regression_weight * geometry_value
            loss = cls_loss + geometric_loss

        if not torch.isfinite(loss).all():
            raise FloatingPointError("non-finite total loss")

        loss_dict = {
            "loss": loss,
            "cls": cls_loss.detach(),
            "offset": offset_loss.detach(),
            "size": size_loss.detach(),
            "yaw": yaw_loss.detach(),
            "geo": geometric_loss.detach(),
        }
        if clamp_count is not None:
            loss_dict["clamp_count"] = int(clamp_count.detach().cpu())
        if yaw_fallback_count is not None:
            loss_dict["yaw_fallback_count"] = int(yaw_fallback_count.detach().cpu())
        if not self.is_fixed_schema and self.name == "uwag":
            for task, log_scale in zip(self.TASKS, self.log_scales):
                loss_dict[f"weight_{task}"] = torch.exp(-log_scale.detach())
        return loss_dict
