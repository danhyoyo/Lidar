"""Optional PCU-inspired scene heuristics over staged point-cloud chunks.

Chunks are (points, keep_mask); scene points are not concatenated per candidate.
Ground/LOS/shadow/density/intensity rules are heuristics, not a sensor simulator.
"""
from __future__ import annotations

import numpy as np
from .hybrid_geometry import EPS_M, points_in_boxes_union, ray_box_intervals


def ground_height(chunks, box, *, require_support=True):
    distance = np.hypot(box[4], box[5])
    radius = np.clip(2.5 + .03 * max(0., distance - 15.), 2.5, 4.)
    minimum = 6 if distance <= 15 else 4 if distance <= 30 else 2
    local = []
    for points, keep in chunks:
        mask = keep & ((points[:, 0] - box[4])**2 + (points[:, 1] - box[5])**2 <= radius**2)
        local.append(points[mask, 2])
    values = np.concatenate(local) if local else np.empty(0)
    if len(values) < minimum:
        return (not require_support), float(box[6])
    height = float(np.percentile(values, 5))
    supported = -2.5 <= height <= -.8 and int((values <= height + .35).sum()) >= minimum
    return (supported or not require_support), height if supported else float(box[6])


def static_obstacle_count(chunks, box, *, backend):
    expanded = box.copy()
    expanded[6] += .35
    expanded[1] = max(float(box[1]) + 1.5, 3.5) - .35
    return sum(int((keep & points_in_boxes_union(points, expanded[None], backend=backend)).sum())
               for points, keep in chunks)


def line_of_sight_blockers(chunks, box, *, backend):
    count = 0
    maximum_range = np.hypot(box[4], box[5]) + np.hypot(box[3], box[2]) / 2
    for points, keep in chunks:
        ranges = np.hypot(points[:, 0], points[:, 1])
        mask = keep & (ranges >= 2) & (ranges <= maximum_range) & (points[:, 2] > box[6] + .4)
        candidates = points[mask]
        hit, enter, _ = ray_box_intervals(candidates, box, backend=backend)
        lengths = np.linalg.norm(candidates[:, :3], axis=1)
        count += int((hit & (enter > 1) & ((enter - 1) * lengths > 1.5)).sum())
    return count


def shadow_mask(points, box, *, backend):
    hit, _, leave = ray_box_intervals(points, box, backend=backend)
    mask = np.zeros(len(points), bool)
    lengths = np.linalg.norm(points[:, :3], axis=1)
    # Endpoint t=1 lies beyond the box exit; points on its surface are retained.
    mask[hit] = (1 - leave[hit]) * lengths[hit] > EPS_M
    return mask


def transform_object(local_points, entry, box, rng, *, density=False, intensity=False,
                     minimum_points=5, gamma=1.7):
    """Database local points retain source yaw and use geometric-center Z."""
    sx, sy, _, _, _, _, source_yaw = entry["box3d_lidar"]
    source_range, target_range = np.hypot(sx, sy), np.hypot(box[4], box[5])
    local = local_points
    if density and target_range > source_range > 0:
        indices = np.flatnonzero(rng.random(len(local)) <= (source_range / target_range)**2)
        if len(indices) < minimum_points and len(local) >= minimum_points:
            indices = rng.choice(len(local), minimum_points, replace=False)
        local = local[indices]
    result = local.copy()
    angle = float(box[7]) - float(source_yaw)
    c, s = np.cos(angle), np.sin(angle)
    result[:, 0] = local[:, 0] * c - local[:, 1] * s + box[4]
    result[:, 1] = local[:, 0] * s + local[:, 1] * c + box[5]
    result[:, 2] += box[6] + box[1] / 2
    if intensity and result.shape[1] >= 4 and source_range > 0 and target_range > 0:
        result[:, 3] = np.clip(result[:, 3] * (source_range / target_range)**max(0., 2-gamma), 0., 1.)
    return result
