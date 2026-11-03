"""Detached quality targets: exact rotated IoU and legacy geometry proxies."""

from typing import Dict
import torch
import torch.nn.functional as F

from core.losses.oriented_geometry_loss import box_corners
from core.losses.rotated_iou import aligned_rotated_iou


@torch.no_grad()
def compute_rotated_iou_targets(
    pred_offset: torch.Tensor,
    pred_size: torch.Tensor,
    pred_yaw: torch.Tensor,
    target_offset: torch.Tensor,
    target_size: torch.Tensor,
    target_yaw: torch.Tensor,
    reg_mask: torch.Tensor,
    epsilon: float = 1e-6,
    max_abs_log_size: float = 10.0,
) -> torch.Tensor:
    """Exact aligned footprint IoU in [B,H,W], zero outside selected cells.

    Offsets are metric residuals at the same grid cell, so its shared grid
    origin cancels. Width/length and doubled-yaw decoding match ``box_corners``
    and the BEV evaluator. Both prediction and target geometry are detached.
    """
    B, _, H, W = pred_offset.shape
    device = pred_offset.device
    result = torch.zeros((B, H, W), dtype=torch.float32, device=device)
    mask = reg_mask.bool()
    if not mask.any():
        return result

    with torch.autocast(device_type=device.type, enabled=False):
        def select(x):
            return x.detach().permute(0, 2, 3, 1)[mask].float()

        p_off, t_off = select(pred_offset), select(target_offset)
        # Recenter before constructing corners to avoid common world-coordinate
        # cancellation; no gradient is required for a supervision target.
        pred_c, _, _, _ = box_corners(
            p_off - t_off, select(pred_size), select(pred_yaw), epsilon, max_abs_log_size
        )
        target_c, _, _, _ = box_corners(
            torch.zeros_like(t_off), select(target_size), select(target_yaw), epsilon, max_abs_log_size
        )
        result[mask] = aligned_rotated_iou(pred_c, target_c)
    return result


@torch.no_grad()
def compute_mgiou_targets(
    pred_offset: torch.Tensor,
    pred_size: torch.Tensor,
    pred_yaw: torch.Tensor,
    target_offset: torch.Tensor,
    target_size: torch.Tensor,
    target_yaw: torch.Tensor,
    reg_mask: torch.Tensor,
    epsilon: float = 1e-6,
    max_abs_log_size: float = 10.0,
) -> torch.Tensor:
    """Compute per-cell legacy clamped MGIoU similarity (not polygon IoU).

    Predictions are detached to prevent circular regression gradients.
    Returns: [B, H, W] tensor in range [0.0, 1.0].
    """
    B, _, H, W = pred_offset.shape
    device = pred_offset.device
    target_iou = torch.zeros((B, H, W), dtype=torch.float32, device=device)

    pos_mask = reg_mask.bool()
    if not pos_mask.any():
        return target_iou

    with torch.autocast(device_type=device.type, enabled=False):
        # Select positive anchors: [N, 2] in FP32 to support mixed-precision training
        # IMPORTANT: pred inputs are detached to isolate gradients
        p_off = pred_offset.detach().permute(0, 2, 3, 1)[pos_mask].float()
        p_sz = pred_size.detach().permute(0, 2, 3, 1)[pos_mask].float()
        p_yaw = pred_yaw.detach().permute(0, 2, 3, 1)[pos_mask].float()

        t_off = target_offset.permute(0, 2, 3, 1)[pos_mask].float()
        t_sz = target_size.permute(0, 2, 3, 1)[pos_mask].float()
        t_yaw = target_yaw.permute(0, 2, 3, 1)[pos_mask].float()

        pred_c, pred_ax, _, _ = box_corners(p_off, p_sz, p_yaw, epsilon=epsilon, max_abs_log_size=max_abs_log_size)
        tgt_c, tgt_ax, _, _ = box_corners(t_off, t_sz, t_yaw, epsilon=epsilon, max_abs_log_size=max_abs_log_size)

        # 4 projection axes: [N, 4, 2]
        axes = torch.cat((pred_ax, tgt_ax), dim=1)

        # Project corners onto axes: [N, 4, 4]
        pred_proj = torch.einsum("ncd,nad->nac", pred_c, axes)
        target_proj = torch.einsum("ncd,nad->nac", tgt_c, axes)

        pred_min, pred_max = pred_proj.amin(dim=-1), pred_proj.amax(dim=-1)        # [N, 4]
        target_min, target_max = target_proj.amin(dim=-1), target_proj.amax(dim=-1)  # [N, 4]

        intersection = (
            torch.minimum(pred_max, target_max) - torch.maximum(pred_min, target_min)
        ).clamp_min(0.0)
        union = pred_max - pred_min + target_max - target_min - intersection
        hull = torch.maximum(pred_max, target_max) - torch.minimum(pred_min, target_min)

        giou = intersection / union.clamp_min(epsilon) - (hull - union) / hull.clamp_min(epsilon)
        mean_giou = giou.mean(dim=-1)  # [N]

        # Clamp the legacy similarity; this is not area IoU.
        cell_iou = mean_giou.clamp(0.0, 1.0)
        target_iou[pos_mask] = cell_iou.to(dtype=target_iou.dtype)
    return target_iou


@torch.no_grad()
def compute_yaw_footprint_targets(
    pred_offset: torch.Tensor,
    pred_size: torch.Tensor,
    pred_yaw: torch.Tensor,
    target_offset: torch.Tensor,
    target_size: torch.Tensor,
    target_yaw: torch.Tensor,
    reg_mask: torch.Tensor,
    epsilon: float = 1e-6,
    max_abs_log_size: float = 10.0,
) -> torch.Tensor:
    """Compute per-cell footprint IoU modulated by doubled-yaw agreement."""
    B, _, H, W = pred_offset.shape
    device = pred_offset.device
    target_iou = torch.zeros((B, H, W), dtype=torch.float32, device=device)

    pos_mask = reg_mask.bool()
    if not pos_mask.any():
        return target_iou

    with torch.autocast(device_type=device.type, enabled=False):
        p_off = pred_offset.detach().permute(0, 2, 3, 1)[pos_mask].float()
        p_sz = pred_size.detach().permute(0, 2, 3, 1)[pos_mask].float()
        p_yaw = pred_yaw.detach().permute(0, 2, 3, 1)[pos_mask].float()

        t_off = target_offset.permute(0, 2, 3, 1)[pos_mask].float()
        t_sz = target_size.permute(0, 2, 3, 1)[pos_mask].float()
        t_yaw = target_yaw.permute(0, 2, 3, 1)[pos_mask].float()

        # Footprint 2D axis-aligned IoU
        p_w = torch.exp(p_sz[:, 0].clamp(-max_abs_log_size, max_abs_log_size))
        p_l = torch.exp(p_sz[:, 1].clamp(-max_abs_log_size, max_abs_log_size))
        t_w = torch.exp(t_sz[:, 0].clamp(-max_abs_log_size, max_abs_log_size))
        t_l = torch.exp(t_sz[:, 1].clamp(-max_abs_log_size, max_abs_log_size))

        pred_min_x, pred_max_x = p_off[:, 0] - 0.5 * p_w, p_off[:, 0] + 0.5 * p_w
        pred_min_y, pred_max_y = p_off[:, 1] - 0.5 * p_l, p_off[:, 1] + 0.5 * p_l

        tgt_min_x, tgt_max_x = t_off[:, 0] - 0.5 * t_w, t_off[:, 0] + 0.5 * t_w
        tgt_min_y, tgt_max_y = t_off[:, 1] - 0.5 * t_l, t_off[:, 1] + 0.5 * t_l

        inter_x = (torch.minimum(pred_max_x, tgt_max_x) - torch.maximum(pred_min_x, tgt_min_x)).clamp_min(0.0)
        inter_y = (torch.minimum(pred_max_y, tgt_max_y) - torch.maximum(pred_min_y, tgt_min_y)).clamp_min(0.0)
        intersection = inter_x * inter_y

        pred_area = p_w * p_l
        tgt_area = t_w * t_l
        union = pred_area + tgt_area - intersection
        footprint_iou = intersection / union.clamp_min(epsilon)

        # Doubled-yaw cosine similarity
        p_yaw_u = F.normalize(p_yaw, dim=-1, eps=epsilon)
        t_yaw_u = F.normalize(t_yaw, dim=-1, eps=epsilon)
        yaw_cos = (p_yaw_u * t_yaw_u).sum(dim=-1).clamp(-1.0, 1.0)
        yaw_factor = 0.5 * (1.0 + yaw_cos)

        cell_iou = (footprint_iou * yaw_factor).clamp(0.0, 1.0)
        target_iou[pos_mask] = cell_iou.to(dtype=target_iou.dtype)
    return target_iou


def compute_iou_targets(
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    method: str = "mgiou",
    epsilon: float = 1e-6,
    max_abs_log_size: float = 10.0,
) -> torch.Tensor:
    """Unified dispatcher for IoU target computation."""
    reg_mask = target["reg_mask"]
    # Autocast can downcast the projection einsums even after inputs are
    # converted to float32. Keep target geometry in float32 throughout.
    with torch.autocast(device_type=pred["offset"].device.type, enabled=False):
        if method == "mgiou":
            return compute_mgiou_targets(
                pred["offset"][:, :2],
                pred["size"][:, :2],
                pred["yaw"][:, :2],
                target["offset"][:, :2],
                target["size"][:, :2],
                target["yaw"][:, :2],
                reg_mask,
                epsilon=epsilon,
                max_abs_log_size=max_abs_log_size,
            )
        elif method == "rotated_iou":
            return compute_rotated_iou_targets(
                pred["offset"][:, :2], pred["size"][:, :2], pred["yaw"][:, :2],
                target["offset"][:, :2], target["size"][:, :2], target["yaw"][:, :2],
                reg_mask, epsilon=epsilon, max_abs_log_size=max_abs_log_size,
            )
        elif method == "yaw_footprint":
            return compute_yaw_footprint_targets(
                pred["offset"][:, :2],
                pred["size"][:, :2],
                pred["yaw"][:, :2],
                target["offset"][:, :2],
                target["size"][:, :2],
                target["yaw"][:, :2],
                reg_mask,
                epsilon=epsilon,
                max_abs_log_size=max_abs_log_size,
            )
        else:
            raise ValueError(
                f"Unknown IoU target method: {method!r}. "
                "Expected 'mgiou', 'rotated_iou' or 'yaw_footprint'."
            )
