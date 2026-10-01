"""Quality-Coupled Focal Loss for continuous overlap-guided classification."""

import torch

__all__ = ["quality_focal_loss"]


def quality_focal_loss(
    pred_logits: torch.Tensor,
    target_prob: torch.Tensor,
    quality_scores: torch.Tensor,
    beta: float = 1.0,
    gamma: float = 2.0,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Compute Quality-Coupled Focal Loss.

    Args:
        pred_logits: [B, C, H, W] raw classification logits.
        target_prob: [B, C, H, W] Gaussian classification targets in [0, 1].
        quality_scores: [B, H, W] detached geometric quality scores in [0, 1].
        beta: Quality target exponent.
        gamma: Modulating focal factor.
        epsilon: Numerical stability clamp bound.
    """
    pred_sig = pred_logits.float().sigmoid().clamp(min=epsilon, max=1.0 - epsilon)
    q = quality_scores.unsqueeze(1).clamp(0.0, 1.0)

    # Positive locations: modulated by quality^beta
    pos_mask = target_prob.ge(1.0)
    continuous_target = torch.where(pos_mask, target_prob * torch.pow(q, beta), target_prob)

    # Focal modulating weight |target - pred|^gamma
    focal_weight = torch.pow(torch.abs(continuous_target - pred_sig), gamma)

    # Negative log-likelihood (BCE with probabilities)
    bce = continuous_target * (-torch.log(pred_sig)) + (1.0 - continuous_target) * (-torch.log(1.0 - pred_sig))
    loss = focal_weight * bce

    num_pos = pos_mask.sum().clamp_min(1.0)
    return loss.sum() / num_pos
