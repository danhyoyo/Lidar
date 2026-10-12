"""Aligned rotated rectangle IoU for detached quality supervision.

The intersection of two convex rectangles consists of contained corners and
edge intersections. All 24 candidates are processed in batches on the input
device; this target calculation needs no CPU polygon library or CUDA extension.
"""

import torch


def _cross(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _polygon_area(points):
    centered = points - points.mean(dim=1, keepdim=True)
    return .5 * _cross(centered, centered.roll(-1, dims=1)).sum(dim=1).abs()


@torch.no_grad()
def aligned_rotated_iou(pred_corners, target_corners):
    """Return [N] area IoU for aligned [N,4,2] rectangle corner tensors.

    Corner order is ``[++, +-, -+, --]``, as returned by ``box_corners``.
    FP32 and pair-local dimension normalization protect small boxes and AMP.
    This function intentionally supplies no regression gradients.
    """
    if pred_corners.shape != target_corners.shape or pred_corners.shape[1:] != (4, 2):
        raise ValueError("aligned rectangle corners must both have shape [N, 4, 2]")
    if pred_corners.shape[0] == 0:
        return pred_corners.new_empty((0,), dtype=torch.float32)

    with torch.autocast(device_type=pred_corners.device.type, enabled=False):
        order = [0, 1, 3, 2]  # Clockwise polygon traversal.
        origin = target_corners.float().mean(dim=1, keepdim=True)
        pred = pred_corners.float()[:, order] - origin
        target = target_corners.float()[:, order] - origin
        pred_edges = pred.roll(-1, dims=1) - pred
        target_edges = target.roll(-1, dims=1) - target
        scale = torch.cat((pred_edges, target_edges), dim=1).norm(dim=-1).amax(dim=1)
        tiny = torch.finfo(torch.float32).tiny
        scale = scale.clamp_min(tiny)[:, None, None]
        pred, target = pred / scale, target / scale
        pred_edges = pred.roll(-1, dims=1) - pred
        target_edges = target.roll(-1, dims=1) - target

        def inside(points, polygon, edges):
            relative = points[:, :, None, :] - polygon[:, None, :, :]
            return (_cross(edges[:, None, :, :], relative) <= 0).all(dim=-1)

        pred_inside = inside(pred, target, target_edges)
        target_inside = inside(target, pred, pred_edges)

        p, r = pred[:, :, None, :], pred_edges[:, :, None, :]
        q, s = target[:, None, :, :], target_edges[:, None, :, :]
        denominator = _cross(r, s)
        nonparallel = denominator.abs() > tiny
        safe_denominator = torch.where(nonparallel, denominator, torch.ones_like(denominator))
        relative = q - p
        t = _cross(relative, s) / safe_denominator
        u = _cross(relative, r) / safe_denominator
        crosses = nonparallel & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        intersections = (p + t[..., None] * r).flatten(1, 2)
        points = torch.cat((pred, target, intersections), dim=1)
        valid = torch.cat((pred_inside, target_inside, crosses.flatten(1)), dim=1)
        points = torch.where(valid[..., None], points, torch.zeros_like(points))
        count = valid.sum(dim=1)
        centroid = points.sum(dim=1) / count.clamp_min(1)[:, None]
        relative = points - centroid[:, None, :]
        angles = torch.atan2(relative[..., 1], relative[..., 0])
        angles = torch.where(valid, angles, torch.full_like(angles, 4 * torch.pi))
        indices = angles.argsort(dim=1, stable=True)
        ordered = relative.gather(1, indices[..., None].expand(-1, -1, 2))

        # Invalid candidates sort after the polygon; explicitly close the last
        # valid vertex to the first instead of including padded points.
        consecutive = _cross(ordered[:, :-1], ordered[:, 1:])
        edge_mask = torch.arange(points.shape[1] - 1, device=points.device)[None] < (count - 1)[:, None]
        last = ordered.gather(1, (count - 1).clamp_min(0)[:, None, None].expand(-1, 1, 2)).squeeze(1)
        twice_area = (consecutive * edge_mask).sum(dim=1) + _cross(last, ordered[:, 0])
        intersection = torch.where(count >= 3, .5 * twice_area.abs(), torch.zeros_like(twice_area))
        pred_area, target_area = _polygon_area(pred), _polygon_area(target)
        intersection = torch.minimum(intersection, torch.minimum(pred_area, target_area))
        union = pred_area + target_area - intersection
        return (intersection / union.clamp_min(tiny)).clamp(0., 1.)
