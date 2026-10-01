"""Analytic Inversion-Free 2D Gaussian Wasserstein Metric for Rotated BEV Boxes."""

import torch
import torch.nn.functional as F

__all__ = [
    "box_covariance_2d",
    "closed_form_2d_gaussian_wasserstein",
    "bounded_gwa_loss",
]


def box_covariance_2d(
    log_size: torch.Tensor,
    doubled_yaw: torch.Tensor,
    divisor: float = 12.0,
    epsilon: float = 1e-6,
    max_abs_log_size: float = 10.0,
):
    """Construct 2x2 spatial covariance matrices directly from doubled-angle yaw.

    Args:
        log_size: [N, 2] (log width, log length)
        doubled_yaw: [N, 2] (cos 2 theta, sin 2 theta)
        divisor: Local variance normalization divisor (default 12.0 for uniform rectangle).
        epsilon: Numerical stability constant.
        max_abs_log_size: Clamping bound for log sizes.
    Returns:
        covariance: [N, 2, 2] symmetric positive-definite covariance matrices.
        clamp_count: int scalar clamped log-size count.
    """
    log_size = log_size.float()
    doubled_yaw = doubled_yaw.float()
    clamped = log_size.clamp(-max_abs_log_size, max_abs_log_size)
    width, length = torch.exp(clamped).unbind(dim=-1)

    cos2, sin2 = F.normalize(doubled_yaw, dim=-1, eps=epsilon).unbind(dim=-1)

    parallel = length.square() / divisor
    perpendicular = width.square() / divisor
    mean = 0.5 * (parallel + perpendicular)
    delta = 0.5 * (parallel - perpendicular)

    covariance = torch.stack(
        (
            torch.stack((mean + delta * cos2, delta * sin2), dim=-1),
            torch.stack((delta * sin2, mean - delta * cos2), dim=-1),
        ),
        dim=-2,
    )
    scale = torch.maximum(parallel, perpendicular).clamp_min(1.0)
    eye = torch.eye(2, dtype=torch.float32, device=covariance.device)
    return covariance + (epsilon * scale)[..., None, None] * eye, (clamped != log_size).sum()


def closed_form_2d_gaussian_wasserstein(
    pred_offset: torch.Tensor,
    pred_cov: torch.Tensor,
    target_offset: torch.Tensor,
    target_cov: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Compute exact 2-Wasserstein distance between 2D Gaussians with zero matrix inversions.

    Formula:
        W_2^2 = ||mu_p - mu_t||^2 + Tr(Sigma_p) + Tr(Sigma_t) - 2 * sqrt(Tr(Sigma_p * Sigma_t) + 2 * sqrt(det(Sigma_p) * det(Sigma_t)))
    """
    if pred_offset.shape[0] == 0:
        return torch.empty(0, dtype=pred_offset.dtype, device=pred_offset.device)

    # 1. Center separation squared: ||mu_p - mu_t||^2
    diff_center = pred_offset - target_offset
    center_dist_sq = diff_center.square().sum(dim=-1)  # [N]

    # 2. Traces: Tr(Sigma) = Sigma_00 + Sigma_11
    tr_p = pred_cov[..., 0, 0] + pred_cov[..., 1, 1]
    tr_t = target_cov[..., 0, 0] + target_cov[..., 1, 1]

    # 3. Determinants: det(Sigma) = Sigma_00 * Sigma_11 - Sigma_01 * Sigma_10
    det_p = (pred_cov[..., 0, 0] * pred_cov[..., 1, 1] - pred_cov[..., 0, 1] * pred_cov[..., 1, 0]).clamp_min(epsilon)
    det_t = (target_cov[..., 0, 0] * target_cov[..., 1, 1] - target_cov[..., 0, 1] * target_cov[..., 1, 0]).clamp_min(epsilon)

    # 4. Product trace: Tr(Sigma_p * Sigma_t)
    tr_pt = (
        pred_cov[..., 0, 0] * target_cov[..., 0, 0]
        + 2.0 * pred_cov[..., 0, 1] * target_cov[..., 0, 1]
        + pred_cov[..., 1, 1] * target_cov[..., 1, 1]
    )

    # 5. Exact matrix square-root trace: sqrt(Tr(pt) + 2 * sqrt(det_p * det_t))
    inner = (tr_pt + 2.0 * torch.sqrt(det_p * det_t)).clamp_min(epsilon)
    trace_sqrt = 2.0 * torch.sqrt(inner)

    # 6. Combined 2-Wasserstein squared
    w2_sq = (center_dist_sq + tr_p + tr_t - trace_sqrt).clamp_min(0.0)
    return w2_sq


def bounded_gwa_loss(
    w2_sq: torch.Tensor,
    target_log_size: torch.Tensor,
    tau_gwa: float = 2.0,
    epsilon: float = 1e-6,
    max_abs_log_size: float = 10.0,
) -> torch.Tensor:
    """Normalize by target diagonal and apply bounded exponential mapping."""
    if w2_sq.shape[0] == 0:
        return w2_sq.sum() * 0.0

    clamped_target_size = target_log_size.float().clamp(-max_abs_log_size, max_abs_log_size)
    target_w = torch.exp(clamped_target_size[:, 0])
    target_l = torch.exp(clamped_target_size[:, 1])
    diagonal = torch.sqrt(target_w.square() + target_l.square()).clamp_min(epsilon)

    w_normalized = torch.sqrt(w2_sq + epsilon) / (tau_gwa * diagonal)
    return (1.0 - torch.exp(-w_normalized)).mean()
