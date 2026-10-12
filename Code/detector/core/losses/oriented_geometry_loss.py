"""Oriented BEV box geometry and Scale-Normalized Pi-Symmetric Corner Distance."""

import torch
import torch.nn as nn

__all__ = [
    "box_corners",
    "pi_symmetric_corner_distance",
    "multiaxis_projection_giou",
    "OrientedGeometryLoss",
]


def box_corners(offset, log_size, doubled_yaw, epsilon=1e-6, max_abs_log_size=10.0):
    """Compute 4 corners and local projection axes of rotated BEV bounding boxes.

    Inputs:
        offset: [N, 2] center (x, y) in metric coordinates.
        log_size: [N, 2] log-dimensions (log_width, log_length).
        doubled_yaw: [N, 2] doubled-angle vector (cos(2 theta), sin(2 theta)).
    Outputs:
        corners: [N, 4, 2] metric BEV corners in order [++, +-, -+, --].
        axes: [N, 2, 2] local projection axes (length_axis, width_axis).
        clamp_count: int scalar of clamped log_size elements.
        fallback_count: int scalar of zero-norm yaw fallback events.
    """
    log_size = log_size.float()
    clamped_log_size = log_size.clamp(-max_abs_log_size, max_abs_log_size)
    clamp_count = (clamped_log_size != log_size).sum()

    width = torch.exp(clamped_log_size[:, 0])
    length = torch.exp(clamped_log_size[:, 1])

    yaw = doubled_yaw.float()
    norm = torch.linalg.vector_norm(yaw, dim=-1, keepdim=True)
    valid = norm >= epsilon
    fallback_count = (~valid).sum()

    unit_yaw = yaw / norm.clamp_min(epsilon)
    cos2 = torch.where(valid, unit_yaw[..., :1], torch.ones_like(unit_yaw[..., :1]))
    sin2 = torch.where(valid, unit_yaw[..., 1:], torch.zeros_like(unit_yaw[..., 1:]))
    theta = 0.5 * torch.atan2(sin2, cos2).squeeze(-1)

    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    length_axis = torch.stack((cos_t, sin_t), dim=-1)   # [N, 2]
    width_axis = torch.stack((-sin_t, cos_t), dim=-1)   # [N, 2]
    axes = torch.stack((length_axis, width_axis), dim=1) # [N, 2, 2]

    half_l = 0.5 * length.unsqueeze(-1) * length_axis   # [N, 2]
    half_w = 0.5 * width.unsqueeze(-1) * width_axis     # [N, 2]
    center = offset.float()

    # 4 corners: [center + l/2 + w/2, center + l/2 - w/2, center - l/2 + w/2, center - l/2 - w/2]
    corners = torch.stack(
        (
            center + half_l + half_w,
            center + half_l - half_w,
            center - half_l + half_w,
            center - half_l - half_w,
        ),
        dim=1,
    )
    return corners, axes, clamp_count, fallback_count


def pi_symmetric_corner_distance(
    pred_corners, target_corners, target_log_size, epsilon=1e-6, max_abs_log_size=10.0
):
    """Compute scale-normalized pi-symmetric corner distance between oriented boxes.

    Inputs:
        pred_corners: [N, 4, 2] predicted corners.
        target_corners: [N, 4, 2] target corners.
        target_log_size: [N, 2] target log(width, length) for scale normalization.
        epsilon: numerical stability constant.
        max_abs_log_size: maximum log-dimension bound for scale normalization.
    """
    if pred_corners.shape[0] == 0:
        return pred_corners.sum() * 0.0

    # With indexing [++, +-, -+, --]:
    # Rotating by 180 degrees (pi) maps:
    # ++ (idx 0) to -- (idx 3)
    # +- (idx 1) to -+ (idx 2)
    # -+ (idx 2) to +- (idx 1)
    # -- (idx 3) to ++ (idx 0)
    # Direct alignment: [0, 1, 2, 3]
    # Pi-rotated alignment: [3, 2, 1, 0]
    dist_direct = torch.abs(pred_corners - target_corners).sum(dim=-1).mean(dim=-1) # [N]

    target_corners_pi = target_corners[:, [3, 2, 1, 0], :]
    dist_pi = torch.abs(pred_corners - target_corners_pi).sum(dim=-1).mean(dim=-1)   # [N]

    raw_distance = torch.minimum(dist_direct, dist_pi) # [N]

    # Normalize by the target bounding box diagonal to equalize scale sensitivity across classes
    clamped_target_size = target_log_size.float().clamp(-max_abs_log_size, max_abs_log_size)
    target_w = torch.exp(clamped_target_size[:, 0])
    target_l = torch.exp(clamped_target_size[:, 1])
    diagonal = torch.sqrt(target_w.square() + target_l.square()).clamp_min(epsilon)

    normalized_distance = raw_distance / diagonal
    return normalized_distance.mean()


def multiaxis_projection_giou(pred_corners, pred_axes, target_corners, target_axes, epsilon=1e-6):
    """Compute 1D Generalized IoU projected over 4 rectangle normal axes.

    Paper: AAAI 2026 MGIoU (Specialized 2D BEV projection).
    """
    if pred_corners.shape[0] == 0:
        return pred_corners.sum() * 0.0

    # 4 projection axes: 2 from prediction, 2 from target
    axes = torch.cat((pred_axes, target_axes), dim=1)  # [N, 4, 2]

    # Project 4 corners of each box onto the 4 projection axes
    pred_proj = torch.einsum("ncd,nad->nac", pred_corners, axes)      # [N, 4, 4]
    target_proj = torch.einsum("ncd,nad->nac", target_corners, axes)  # [N, 4, 4]

    pred_min, pred_max = pred_proj.amin(dim=-1), pred_proj.amax(dim=-1)        # [N, 4]
    target_min, target_max = target_proj.amin(dim=-1), target_proj.amax(dim=-1)  # [N, 4]

    intersection = (
        torch.minimum(pred_max, target_max) - torch.maximum(pred_min, target_min)
    ).clamp_min(0.0)
    union = pred_max - pred_min + target_max - target_min - intersection
    hull = torch.maximum(pred_max, target_max) - torch.minimum(pred_min, target_min)

    giou = intersection / union.clamp_min(epsilon) - (hull - union) / hull.clamp_min(epsilon)
    loss_1d = (1.0 - giou.mean(dim=-1)) / 2.0
    return loss_1d.mean()


class OrientedGeometryLoss(nn.Module):
    """Composite Rotated BEV Geometry Loss: Multi-Axis Projection GIoU + Normalized Corner Distance."""

    def __init__(self, beta=1.0, epsilon=1e-6, max_abs_log_size=10.0):
        super().__init__()
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.max_abs_log_size = float(max_abs_log_size)

        if self.beta < 0.0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")
        if self.epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        if self.max_abs_log_size <= 0.0:
            raise ValueError(f"max_abs_log_size must be positive, got {self.max_abs_log_size}")

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
            device = pred_offset.device
            return zero, {
                "proj_giou": torch.zeros((), device=device),
                "corner_dist": torch.zeros((), device=device),
                "clamp_count": torch.zeros((), dtype=torch.int64, device=device),
                "fallback_count": torch.zeros((), dtype=torch.int64, device=device),
            }

        def _select(x):
            return x.permute(0, 2, 3, 1).reshape(-1, 2)[positive].float()

        pred_off_pos = _select(pred_offset)
        pred_size_pos = _select(pred_size)
        pred_yaw_pos = _select(pred_yaw)

        tgt_off_pos = _select(target_offset)
        tgt_size_pos = _select(target_size)
        tgt_yaw_pos = _select(target_yaw)

        with torch.autocast(device_type=pred_offset.device.type, enabled=False):
            pred_c, pred_ax, pred_clamps, pred_fallbacks = box_corners(
                pred_off_pos, pred_size_pos, pred_yaw_pos, self.epsilon, self.max_abs_log_size
            )
            tgt_c, tgt_ax, tgt_clamps, tgt_fallbacks = box_corners(
                tgt_off_pos, tgt_size_pos, tgt_yaw_pos, self.epsilon, self.max_abs_log_size
            )

            proj_loss = multiaxis_projection_giou(pred_c, pred_ax, tgt_c, tgt_ax, self.epsilon)
            corner_loss = pi_symmetric_corner_distance(
                pred_c, tgt_c, tgt_size_pos, self.epsilon, self.max_abs_log_size
            )
            total_geo = proj_loss + self.beta * corner_loss

        return total_geo, {
            "proj_giou": proj_loss.detach(),
            "corner_dist": corner_loss.detach(),
            "clamp_count": (pred_clamps + tgt_clamps).detach(),
            "fallback_count": (pred_fallbacks + tgt_fallbacks).detach(),
        }

