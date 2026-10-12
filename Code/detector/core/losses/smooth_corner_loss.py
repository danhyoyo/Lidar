"""Smooth Log-Sum-Exp Soft-Min Pi-Symmetric Corner Distance and Spatial RDA."""

import torch

__all__ = ["smooth_pi_symmetric_corner_distance", "compute_range_weights"]


def smooth_pi_symmetric_corner_distance(
    pred_corners: torch.Tensor,
    target_corners: torch.Tensor,
    target_log_size: torch.Tensor,
    tau: float = 0.05,
    epsilon: float = 1e-6,
    max_abs_log_size: float = 10.0,
) -> torch.Tensor:
    """Compute C^inf-smooth scale-normalized pi-symmetric corner distance.

    Args:
        pred_corners: [N, 4, 2]
        target_corners: [N, 4, 2]
        target_log_size: [N, 2]
        tau: Soft-min temperature parameter.
        epsilon: Numerical stability floor.
        max_abs_log_size: Bound on log extents.
    """
    if pred_corners.shape[0] == 0:
        return pred_corners.sum() * 0.0

    # L1 corner distances
    dist_direct = torch.abs(pred_corners - target_corners).sum(dim=-1).mean(dim=-1)  # [N]
    target_corners_pi = target_corners[:, [3, 2, 1, 0], :]
    dist_pi = torch.abs(pred_corners - target_corners_pi).sum(dim=-1).mean(dim=-1)    # [N]

    # Log-Sum-Exp soft minimum: -tau * log(exp(-d1/tau) + exp(-d2/tau))
    stacked = torch.stack((-dist_direct / tau, -dist_pi / tau), dim=-1)
    smooth_min = -tau * torch.logsumexp(stacked, dim=-1)

    clamped_target_size = target_log_size.float().clamp(-max_abs_log_size, max_abs_log_size)
    target_w = torch.exp(clamped_target_size[:, 0])
    target_l = torch.exp(clamped_target_size[:, 1])
    diagonal = torch.sqrt(target_w.square() + target_l.square()).clamp_min(epsilon)

    return (smooth_min / diagonal).mean()


def compute_range_weights(
    pos_offsets: torch.Tensor,
    r_max: float = 70.4,
    gamma: float = 1.5,
    alpha: float = 2.0,
) -> torch.Tensor:
    """Compute Range & Density-Adaptive (RDA) regression weights per positive cell."""
    if pos_offsets.shape[0] == 0:
        return torch.empty(0, dtype=pos_offsets.dtype, device=pos_offsets.device)

    radial_dist = torch.linalg.vector_norm(pos_offsets[:, :2], dim=-1)
    normalized_r = (radial_dist / r_max).clamp(0.0, 1.5)
    return 1.0 + gamma * torch.pow(normalized_r, alpha)
