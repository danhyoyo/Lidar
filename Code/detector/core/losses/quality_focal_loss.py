"""CenterNet-Compatible Quality-Coupled Focal Loss (CQFL)."""

import torch

__all__ = ["quality_focal_loss"]


def quality_focal_loss(
    pred_logits: torch.Tensor,
    target_prob: torch.Tensor,
    quality_scores: torch.Tensor,
    beta: float = 1.0,
    gamma: float = 2.0,
    alpha: float = 4.0,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Compute CenterNet-Compatible Quality-Coupled Focal Loss (CQFL).

    Resolves the Gaussian plateau false positive avalanche by strictly separating:
    1. Positive Peak Centers (target_prob == 1.0):
       Modulates classification target with geometric quality score: y* = q^beta.
       Loss = |y* - pred|^gamma * [y* * (-log(pred)) + (1 - y*) * (-log(1 - pred))]
       When q = 1.0, simplifies identically to CenterNet's (1 - pred)^gamma * (-log(pred)).
    2. Negative Background & Gaussian Neighbors (target_prob < 1.0):
       Preserves CenterNet's Dirac peak penalty discount with target = 0:
       Loss = (1 - target_prob)^alpha * (pred^gamma) * (-log(1 - pred))
       This forces predictions p -> 0 everywhere around the peak, eliminating false positive plateaus.

    Args:
        pred_logits: [B, C, H, W] raw classification logits.
        target_prob: [B, C, H, W] Gaussian classification targets in [0, 1].
        quality_scores: [B, H, W] detached geometric quality scores in [0, 1].
        beta: Quality target exponent.
        gamma: Modulating focal factor (default 2.0).
        alpha: Negative penalty reduction exponent (default 4.0).
        epsilon: Numerical stability clamp bound (default 1e-4).
    """
    pred_sig = pred_logits.float().sigmoid().clamp(min=epsilon, max=1.0 - epsilon)

    pos_mask = target_prob.ge(1.0)
    neg_mask = target_prob.lt(1.0)

    # 1. Positive peak loss modulated by geometric quality score
    pos_pred = pred_sig[pos_mask]
    if pos_pred.nelement() > 0:
        q = quality_scores.unsqueeze(1).clamp(0.0, 1.0)
        pos_target = torch.pow(q, beta).expand_as(target_prob)[pos_mask]
        pos_focal = torch.pow(torch.abs(pos_target - pos_pred), gamma)
        pos_bce = pos_target * (-torch.log(pos_pred)) + (1.0 - pos_target) * (-torch.log(1.0 - pos_pred))
        pos_loss = (pos_focal * pos_bce).sum()
    else:
        pos_loss = pred_logits.sum() * 0.0

    # 2. Negative neighbor & background loss (CenterNet penalty reduction)
    neg_pred = pred_sig[neg_mask]
    neg_target = target_prob[neg_mask]
    neg_weights = torch.pow(1.0 - neg_target, alpha)
    neg_focal = torch.pow(neg_pred, gamma)
    neg_loss = (neg_weights * neg_focal * (-torch.log(1.0 - neg_pred))).sum()

    num_pos = pos_mask.sum().clamp_min(1.0)
    return (pos_loss + neg_loss) / num_pos
