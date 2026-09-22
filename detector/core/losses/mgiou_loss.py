"""Projection-based MGIoU loss specialized to rotated 2D BEV rectangles.

Dense detector heads use shape ``[B, 2, H, W]`` in prediction-first then target
order: offsets ``[x,y]``, log-sizes ``[log(width),log(length)]``, and doubled yaw
``[cos(2 theta),sin(2 theta)]``. Only positive ``reg_mask`` locations are
reduced. Geometry is explicitly FP32 under mixed precision; an empty mask
returns a graph-connected zero for every prediction head.
"""

import math

import torch
import torch.nn as nn


__all__ = ["MGIoULoss"]


def _validate_geometry_parameters(epsilon, max_abs_log_size):
    epsilon = float(epsilon)
    max_abs_log_size = float(max_abs_log_size)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")
    if not math.isfinite(max_abs_log_size) or max_abs_log_size <= 0.0:
        raise ValueError(
            "max_abs_log_size must be positive, "
            f"got {max_abs_log_size}"
        )
    return epsilon, max_abs_log_size


def _select_positive(values, positive):
    return values.permute(0, 2, 3, 1).reshape(-1, 2)[positive]


def _box_geometry(offset, log_size, doubled_yaw, epsilon, max_abs_log_size):
    """Return corners, local axes, clamp count, and invalid-yaw fallback count."""
    log_size = log_size.float()
    clamped = log_size.clamp(-max_abs_log_size, max_abs_log_size)
    width, length = torch.exp(clamped).unbind(dim=-1)

    # MGIoU needs explicit projection axes. Recover theta only here; the half
    # angle maps doubled-yaw encoding back to the rectangle's pi-periodic yaw.
    yaw = doubled_yaw.float()
    norm = torch.linalg.vector_norm(yaw, dim=-1, keepdim=True)
    valid = norm >= epsilon
    unit = yaw / norm.clamp_min(epsilon)
    cos2 = torch.where(valid, unit[..., :1], torch.ones_like(unit[..., :1]))
    sin2 = torch.where(valid, unit[..., 1:], torch.zeros_like(unit[..., 1:]))
    theta = 0.5 * torch.atan2(sin2, cos2).squeeze(-1)

    cosine, sine = torch.cos(theta), torch.sin(theta)
    length_axis = torch.stack((cosine, sine), dim=-1)
    width_axis = torch.stack((-sine, cosine), dim=-1)
    axes = torch.stack((length_axis, width_axis), dim=1)

    # theta=0 means length lies on +x and width on +y. These four signed
    # half-extent combinations are the rectangle corners in world coordinates.
    half_length = 0.5 * length.unsqueeze(-1) * length_axis
    half_width = 0.5 * width.unsqueeze(-1) * width_axis
    center = offset.float()
    corners = torch.stack(
        (
            center + half_length + half_width,
            center + half_length - half_width,
            center - half_length + half_width,
            center - half_length - half_width,
        ),
        dim=1,
    )
    return corners, axes, (clamped != log_size).sum(), (~valid).sum()


class MGIoULoss(nn.Module):
    r"""Marginalized GIoU over four rectangle-normal projection axes.

    Paper: https://ojs.aaai.org/index.php/AAAI/article/view/37505
    Official code: https://github.com/ldtho/MGIoU/blob/main/mgiou/losses.py

    This is the 2D rotated-BEV specialization only. It intentionally omits the
    upstream 3D/quadrangle wrappers, registries, and the optional fast signed-
    overlap approximation. For each prediction/target pair, both local axes
    from each box give four normals. Projecting all corners onto one normal
    produces intervals ``[p_min,p_max]`` and ``[t_min,t_max]``. The full 1D
    generalized IoU is

    ``GIoU = intersection/union - (hull-union)/hull``.

    Averaging the four marginal GIoUs yields the shape similarity, transformed
    to ``loss=(1-mean_GIoU)/2``. Unlike ordinary IoU, the hull term keeps a
    position gradient for disjoint intervals.

    Inputs follow prediction-first then target order and use ``[B,2,H,W]`` head
    maps plus ``[B,H,W]`` ``reg_mask``. The output is
    ``(positive_mean_loss, log_size_clamp_count, yaw_fallback_count)``. A zero
    doubled-yaw vector falls back to ``theta=0`` and increments the final count.
    """

    def __init__(self, epsilon=1e-6, max_abs_log_size=10.0):
        super().__init__()
        self.epsilon, self.max_abs_log_size = _validate_geometry_parameters(
            epsilon, max_abs_log_size
        )

    def forward(
        self,
        pred_offset,
        pred_size,
        pred_yaw,
        target_offset,
        target_size,
        target_yaw,
        reg_mask,
    ):
        positive = reg_mask.reshape(-1).bool()
        if not positive.any():
            zero = (pred_offset.sum() + pred_size.sum() + pred_yaw.sum()) * 0.0
            count = torch.zeros((), dtype=torch.int64, device=reg_mask.device)
            return zero, count, count.clone()

        with torch.autocast(device_type=pred_offset.device.type, enabled=False):
            pred_corners, pred_axes, pred_clamps, pred_fallbacks = _box_geometry(
                _select_positive(pred_offset, positive),
                _select_positive(pred_size, positive),
                _select_positive(pred_yaw, positive),
                self.epsilon,
                self.max_abs_log_size,
            )
            target_corners, target_axes, target_clamps, target_fallbacks = (
                _box_geometry(
                    _select_positive(target_offset, positive),
                    _select_positive(target_size, positive),
                    _select_positive(target_yaw, positive),
                    self.epsilon,
                    self.max_abs_log_size,
                )
            )

            # Project each box's four corners onto prediction and target axes.
            axes = torch.cat((pred_axes, target_axes), dim=1)
            pred_projection = torch.einsum("ncd,nad->nac", pred_corners, axes)
            target_projection = torch.einsum(
                "ncd,nad->nac", target_corners, axes
            )
            pred_min, pred_max = pred_projection.amin(-1), pred_projection.amax(-1)
            target_min = target_projection.amin(-1)
            target_max = target_projection.amax(-1)

            # Full 1D GIoU: standard non-negative intersection plus hull penalty.
            intersection = (
                torch.minimum(pred_max, target_max)
                - torch.maximum(pred_min, target_min)
            ).clamp_min(0.0)
            union = pred_max - pred_min + target_max - target_min - intersection
            hull = (
                torch.maximum(pred_max, target_max)
                - torch.minimum(pred_min, target_min)
            )
            giou = intersection / union.clamp_min(self.epsilon) - (
                hull - union
            ) / hull.clamp_min(self.epsilon)
            loss = (1.0 - giou.mean(dim=-1)) / 2.0
            return (
                loss.mean(),
                pred_clamps + target_clamps,
                pred_fallbacks + target_fallbacks,
            )
