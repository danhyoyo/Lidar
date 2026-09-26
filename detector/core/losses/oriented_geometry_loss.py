"""Oriented BEV box geometry and Scale-Normalized Pi-Symmetric Corner Distance."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["box_corners", "pi_symmetric_corner_distance"]


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


def pi_symmetric_corner_distance(pred_corners, target_corners, target_log_size, epsilon=1e-6):
    """Compute scale-normalized pi-symmetric corner distance between oriented boxes.

    Inputs:
        pred_corners: [N, 4, 2] predicted corners.
        target_corners: [N, 4, 2] target corners.
        target_log_size: [N, 2] target log(width, length) for scale normalization.
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
    dist_direct = torch.norm(pred_corners - target_corners, p=1, dim=-1).mean(dim=-1) # [N]

    target_corners_pi = target_corners[:, [3, 2, 1, 0], :]
    dist_pi = torch.norm(pred_corners - target_corners_pi, p=1, dim=-1).mean(dim=-1)   # [N]

    raw_distance = torch.minimum(dist_direct, dist_pi) # [N]

    # Normalize by the target bounding box diagonal to equalize scale sensitivity across classes
    target_w = torch.exp(target_log_size[:, 0].float())
    target_l = torch.exp(target_log_size[:, 1].float())
    diagonal = torch.sqrt(target_w.square() + target_l.square()).clamp_min(epsilon)

    normalized_distance = raw_distance / diagonal
    return normalized_distance.mean()
