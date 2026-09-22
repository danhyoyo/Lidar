"""Gaussian geometry losses for rotated BEV detector heads.

The detector supplies dense maps with shape ``[B, 2, H, W]``. Offsets are
``[x, y]``, sizes are ``[log(width), log(length)]``, and yaw is represented by
``[cos(2 theta), sin(2 theta)]``. The doubled angle makes a rectangle invariant
to ``theta + pi`` and lets Gaussian losses construct covariance matrices without
recovering ``theta`` with ``atan2``.

Every loss below selects only locations enabled by ``reg_mask`` and returns the
mean over those positives. Geometry runs in FP32 with autocast disabled because
matrix solves and log determinants are fragile in BF16. An empty mask returns a
graph-connected zero, so all three prediction heads still receive zero gradients.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["box_covariance", "KFIoULoss", "ProbIoULoss", "KLDLoss"]


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


def _positive_mask(reg_mask):
    return reg_mask.reshape(-1).bool()


def _select_positive(values, positive):
    """Convert ``[B, 2, H, W]`` maps to positive-only ``[N, 2]`` FP32 rows."""
    return values.permute(0, 2, 3, 1).reshape(-1, 2)[positive].float()


def _empty_loss(pred_offset, pred_size, pred_yaw, reg_mask):
    zero = (pred_offset.sum() + pred_size.sum() + pred_yaw.sum()) * 0.0
    count = torch.zeros((), dtype=torch.int64, device=reg_mask.device)
    return zero, count


def box_covariance(log_size, doubled_yaw, divisor, epsilon, max_abs_log_size):
    r"""Build positive-definite BEV covariance matrices directly from head output.

    For local variances ``a = length^2 / divisor`` and
    ``b = width^2 / divisor``, rotation by ``theta`` gives

    ``Sigma = [[m + d cos(2theta), d sin(2theta)],``
    ``         [d sin(2theta), m - d cos(2theta)]]``

    where ``m=(a+b)/2`` and ``d=(a-b)/2``. Thus the detector's doubled-yaw
    encoding can be consumed directly: no angle reconstruction or periodic
    branch is needed. ``divisor=4`` models a box by its half extents for KFIoU
    and KLD, while ``divisor=12`` is the uniform-rectangle covariance used by
    ProbIoU. The returned count records clamped log-size scalar elements.

    The covariance is conditioned in the scale of each box. Let
    ``s=max(a,b,1)``; the returned matrix is ``Sigma + epsilon*s*I``. Below
    unit variance this is the familiar absolute ``epsilon`` floor, while above
    unit variance it is a relative floor that bounds the effective condition
    number by approximately ``1/epsilon``. A fixed ``epsilon*I`` is insufficient
    for very high aspect ratios in FP32: after rotation, cancellation in
    ``m +/- d*cos(2theta)`` can be larger than that fixed floor. Scaling the
    diagonal regularizer makes the floor representable at the covariance's
    magnitude without changing dtype or hiding invalid downstream results.
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

    # A scale-relative diagonal floor survives FP32 cancellation after rotation.
    # ``clamp_min(1)`` retains an absolute epsilon floor for sub-unit boxes.
    covariance_scale = torch.maximum(parallel, perpendicular).clamp_min(1.0)
    relative_jitter = epsilon * covariance_scale
    eye = torch.eye(2, dtype=torch.float32, device=covariance.device)
    return (
        covariance + relative_jitter[..., None, None] * eye,
        (clamped != log_size).sum(),
    )


def _positive_logdet(covariance, loss_name, covariance_name):
    sign, logdet = torch.linalg.slogdet(covariance)
    if not torch.all(sign > 0):
        invalid = int((sign <= 0).sum())
        raise ValueError(
            f"{loss_name} requires positive {covariance_name} determinants; "
            f"invalid count: {invalid}"
        )
    return logdet


class KFIoULoss(nn.Module):
    r"""Full KFIoU loss for positive rotated-BEV regression locations.

    Paper: https://arxiv.org/abs/2201.12558
    Reference implementation:
    https://github.com/open-mmlab/mmrotate/blob/main/mmrotate/models/losses/kf_iou_loss.py

    Each rectangle is mapped to ``N(mu, Sigma)`` with local variances
    ``[length^2/4, width^2/4]``. The Kalman product covariance is

    ``Sigma_KF = Sigma_p - K Sigma_p``,
    ``K = Sigma_p (Sigma_p + Sigma_t)^-1``.

    Only the fused volume is required, so the subtraction above is never
    materialized. Instead, the SPD parallel-sum determinant identity

    ``log det(Sigma_KF) = log det(Sigma_p) + log det(Sigma_t)``
    ``                       - log det(Sigma_p + Sigma_t)``

    avoids cancellation that can make the explicitly fused matrix indefinite
    in FP32 for mismatched, high-aspect rotated boxes. Gaussian volumes are
    ``V=4 sqrt(det(Sigma))`` and form
    ``KFIoU = V_KF / (V_p + V_t - V_KF)``. The shape term
    ``exp(1-KFIoU)-1`` is combined with the paper's center-aware term
    ``log(1 + delta^T Sigma_t^-1 delta)``. The latter preserves a useful
    position gradient when the two physical boxes do not overlap.

    Inputs follow prediction-first then target order, with every head tensor in
    ``[B, 2, H, W]`` and ``reg_mask`` in ``[B, H, W]``. Returns
    ``(positive_mean_loss, log_size_clamp_count)``.
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
        positive = _positive_mask(reg_mask)
        if not positive.any():
            return _empty_loss(pred_offset, pred_size, pred_yaw, reg_mask)

        with torch.autocast(device_type=pred_offset.device.type, enabled=False):
            offset = _select_positive(pred_offset, positive) - _select_positive(
                target_offset, positive
            )
            pred_covariance, pred_clamps = box_covariance(
                _select_positive(pred_size, positive),
                _select_positive(pred_yaw, positive),
                4.0,
                self.epsilon,
                self.max_abs_log_size,
            )
            target_covariance, target_clamps = box_covariance(
                _select_positive(target_size, positive),
                _select_positive(target_yaw, positive),
                4.0,
                self.epsilon,
                self.max_abs_log_size,
            )

            # Center term: log(1 + delta^T Sigma_t^-1 delta).
            solved_offset = torch.linalg.solve(
                target_covariance, offset.unsqueeze(-1)
            )
            center = torch.log1p(
                torch.matmul(offset.unsqueeze(-2), solved_offset).squeeze((-1, -2))
            )

            # The parallel-sum determinant identity computes the KF overlap
            # volume without forming Sp - Sp(Sp+St)^-1 Sp. That explicit
            # subtraction can destroy SPD through FP32 cancellation even when
            # Sp, St, and Sp+St are all valid SPD matrices.
            covariance_sum = pred_covariance + target_covariance
            pred_logdet = _positive_logdet(
                pred_covariance, "KFIoU", "prediction"
            )
            target_logdet = _positive_logdet(
                target_covariance, "KFIoU", "target"
            )
            sum_logdet = _positive_logdet(covariance_sum, "KFIoU", "sum")
            fused_logdet = pred_logdet + target_logdet - sum_logdet

            # Work in log-determinant space until the final volume conversion.
            pred_volume = 4.0 * torch.exp(0.5 * pred_logdet)
            target_volume = 4.0 * torch.exp(0.5 * target_logdet)
            fused_volume = 4.0 * torch.exp(0.5 * fused_logdet)
            overlap = fused_volume / (
                pred_volume + target_volume - fused_volume
            )
            loss = center + torch.exp(1.0 - overlap) - 1.0
            return loss.mean(), pred_clamps + target_clamps


class ProbIoULoss(nn.Module):
    r"""Scheduled Bhattacharyya/Hellinger ProbIoU loss.

    Paper: https://arxiv.org/abs/2106.06072
    Implementation note: this class follows the paper formulas directly; it
    does not claim an official reference repository.

    A uniform rectangle is represented by a Gaussian with local covariance
    ``diag(length^2/12, width^2/12)``. For means separated by ``delta`` and
    ``Sigma_m=(Sigma_p+Sigma_t)/2``, the Bhattacharyya distance is

    ``BD = 1/8 delta^T Sigma_m^-1 delta``
    ``     + 1/2 log(det(Sigma_m)/sqrt(det(Sigma_p)det(Sigma_t)))``.

    Before ``switch_fraction`` the raw BD supplies a strong long-range signal;
    afterwards the loss switches to Hellinger distance
    ``H=sqrt(1-exp(-BD))`` for bounded local refinement. Its derivative is
    singular at ``BD=0``, so a continuous linear branch near zero preserves a
    finite gradient for identical boxes.

    Inputs follow prediction-first then target order, with head tensors in
    ``[B, 2, H, W]``. Reduction is the mean over ``reg_mask`` positives only;
    FP32 geometry is enforced internally. Returns
    ``(positive_mean_loss, log_size_clamp_count)``.
    """

    def __init__(
        self,
        switch_fraction=0.5,
        epsilon=1e-6,
        max_abs_log_size=10.0,
    ):
        super().__init__()
        self.epsilon, self.max_abs_log_size = _validate_geometry_parameters(
            epsilon, max_abs_log_size
        )
        self.switch_fraction = float(switch_fraction)
        if not math.isfinite(self.switch_fraction) or not (
            0.0 <= self.switch_fraction <= 1.0
        ):
            raise ValueError(
                "switch_fraction must be in [0, 1], "
                f"got {self.switch_fraction}"
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
        progress=0.0,
    ):
        positive = _positive_mask(reg_mask)
        if not positive.any():
            return _empty_loss(pred_offset, pred_size, pred_yaw, reg_mask)

        with torch.autocast(device_type=pred_offset.device.type, enabled=False):
            offset = _select_positive(pred_offset, positive) - _select_positive(
                target_offset, positive
            )
            pred_covariance, pred_clamps = box_covariance(
                _select_positive(pred_size, positive),
                _select_positive(pred_yaw, positive),
                12.0,
                self.epsilon,
                self.max_abs_log_size,
            )
            target_covariance, target_clamps = box_covariance(
                _select_positive(target_size, positive),
                _select_positive(target_yaw, positive),
                12.0,
                self.epsilon,
                self.max_abs_log_size,
            )

            # Bhattacharyya mean-separation and covariance-shape components.
            mean_covariance = 0.5 * (pred_covariance + target_covariance)
            solved_offset = torch.linalg.solve(
                mean_covariance, offset.unsqueeze(-1)
            )
            mahalanobis = torch.matmul(
                offset.unsqueeze(-2), solved_offset
            ).squeeze((-1, -2))
            pred_logdet = _positive_logdet(
                pred_covariance, "ProbIoU", "prediction"
            )
            target_logdet = _positive_logdet(
                target_covariance, "ProbIoU", "target"
            )
            mean_logdet = _positive_logdet(mean_covariance, "ProbIoU", "mean")
            bd = 0.125 * mahalanobis + 0.5 * (
                mean_logdet - 0.5 * (pred_logdet + target_logdet)
            )
            bd = bd.clamp_min(0.0)

            if float(progress) < self.switch_fraction:
                values = bd
            else:
                # -expm1(-BD) is accurate for small BD. The linear branch
                # matches sqrt(x) at x=epsilon and avoids its infinite slope.
                hellinger_squared = (-torch.expm1(-bd)).clamp_min(0.0)
                values = torch.where(
                    hellinger_squared < self.epsilon,
                    hellinger_squared / (self.epsilon**0.5),
                    torch.sqrt(hellinger_squared.clamp_min(self.epsilon)),
                )
            return values.mean(), pred_clamps + target_clamps


class KLDLoss(nn.Module):
    r"""Directional Gaussian KLD comparator for rotated BEV boxes.

    Paper: https://papers.neurips.cc/paper/2021/hash/98f13708210194c475687be6106a3b84-Abstract.html
    Reference: https://github.com/open-mmlab/mmrotate/blob/main/mmrotate/models/losses/gaussian_dist_loss.py

    This implementation evaluates ``D_KL(pred || target)``:

    ``1/2 [tr(Sigma_t^-1 Sigma_p) + delta^T Sigma_t^-1 delta``
    ``     - 2 + log(det(Sigma_t)/det(Sigma_p))]``.

    The covariance divisor is four, matching the half-extent convention used by
    KLD-based oriented detection. Linear solves and ``slogdet`` replace explicit
    inverses and determinants. The final bounded transform is
    ``1 - 1/(tau + log(1 + D_KL))``, where ``tau >= 1``.

    Prediction tensors precede targets and all heads have shape ``[B,2,H,W]``.
    Only ``reg_mask`` positives contribute to the FP32 mean. Returns
    ``(positive_mean_loss, log_size_clamp_count)``.
    """

    def __init__(self, tau=1.0, epsilon=1e-6, max_abs_log_size=10.0):
        super().__init__()
        self.epsilon, self.max_abs_log_size = _validate_geometry_parameters(
            epsilon, max_abs_log_size
        )
        self.tau = float(tau)
        if not math.isfinite(self.tau) or self.tau < 1.0:
            raise ValueError(f"KLD tau must be finite and at least 1, got {self.tau}")

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
        positive = _positive_mask(reg_mask)
        if not positive.any():
            return _empty_loss(pred_offset, pred_size, pred_yaw, reg_mask)

        with torch.autocast(device_type=pred_offset.device.type, enabled=False):
            offset = _select_positive(pred_offset, positive) - _select_positive(
                target_offset, positive
            )
            pred_covariance, pred_clamps = box_covariance(
                _select_positive(pred_size, positive),
                _select_positive(pred_yaw, positive),
                4.0,
                self.epsilon,
                self.max_abs_log_size,
            )
            target_covariance, target_clamps = box_covariance(
                _select_positive(target_size, positive),
                _select_positive(target_yaw, positive),
                4.0,
                self.epsilon,
                self.max_abs_log_size,
            )

            # Directional D_KL(pred || target): the target covariance is the
            # metric for both the center and covariance-trace terms.
            pred_logdet = _positive_logdet(pred_covariance, "KLD", "prediction")
            target_logdet = _positive_logdet(
                target_covariance, "KLD", "target"
            )
            solved_offset = torch.linalg.solve(
                target_covariance, offset.unsqueeze(-1)
            )
            mahalanobis = torch.matmul(
                offset.unsqueeze(-2), solved_offset
            ).squeeze((-1, -2))
            solved_covariance = torch.linalg.solve(
                target_covariance, pred_covariance
            )
            trace = solved_covariance.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            divergence = (
                0.5 * mahalanobis
                + 0.5 * trace
                + 0.5 * (target_logdet - pred_logdet)
                - 1.0
            )

            loss = 1.0 - 1.0 / (self.tau + torch.log1p(divergence))
            return loss.mean(), pred_clamps + target_clamps
