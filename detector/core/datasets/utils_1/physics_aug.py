import numpy as np
from core.datasets.utils_1.box_geometry import GEOMETRY_EPS_M, ray_box_intervals


def random_flip_3d(points: np.ndarray, boxes: np.ndarray, p: float = 0.5):
    """Random horizontal flip along the lateral Y-axis.

    points: (N, C) with [x, y, z, intensity, ...]
    boxes: (M, 7) or (M, 8) with [h, w, l, x, y, z, yaw] or [cls, h, w, l, x, y, z, yaw]
    """
    if np.random.random() <= p:
        if len(points) > 0:
            points[:, 1] = -points[:, 1]
        if len(boxes) > 0:
            if boxes.shape[1] == 7:
                boxes[:, 4] = -boxes[:, 4]  # y
                boxes[:, 6] = -boxes[:, 6]  # yaw
            elif boxes.shape[1] >= 8:
                boxes[:, 5] = -boxes[:, 5]  # y
                boxes[:, 7] = -boxes[:, 7]  # yaw
    return points, boxes


def distance_adaptive_subsample(
    points: np.ndarray, box: np.ndarray, r_origin: float, r_target: float
) -> np.ndarray:
    """Subsamples points proportionally to (r_origin / r_target)^2."""
    if r_target <= r_origin or len(points) == 0:
        return points
    ratio = min(1.0, (r_origin / r_target) ** 2)
    mask = np.random.random(len(points)) <= ratio
    if np.sum(mask) < 5 and len(points) >= 5:
        idx = np.random.choice(len(points), size=5, replace=False)
        return points[idx]
    return points[mask]


def radiometric_intensity_calibrate(
    points: np.ndarray, r_origin: float, r_target: float, gamma: float = 1.7
) -> np.ndarray:
    """Calibrates point intensity according to radar range attenuation equation."""
    if len(points) == 0 or points.shape[1] < 4 or r_origin <= 0 or r_target <= 0:
        return points
    points = points.copy()
    attenuation = (r_origin / r_target) ** max(0.0, 2.0 - gamma)
    points[:, 3] = np.clip(points[:, 3] * attenuation, 0.0, 1.0)
    return points


def shadow_point_mask(bg_points: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Mask rays that leave the solid oriented box before reaching the point.

    Entry/exit are exact box geometry; a 1e-8 meter distance tolerance keeps
    surface and interior points out of the shadow deletion mask.
    """
    hit, _, leave = ray_box_intervals(bg_points, box)
    lengths = np.linalg.norm(bg_points[:, :3], axis=1)
    mask = np.zeros(len(bg_points), dtype=bool)
    mask[hit] = (1.0 - leave[hit]) * lengths[hit] > GEOMETRY_EPS_M
    return mask


def mask_shadow_points(bg_points: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Remove points behind the oriented box, keeping point order/features."""
    return bg_points[~shadow_point_mask(bg_points, box)]


def check_line_of_sight_occlusion(
    box: np.ndarray,
    points: np.ndarray,
    max_blocking_points: int = 5,
    min_obstacle_height: float = 0.4,
) -> bool:
    """Heuristic foreground blocker count using exact ray-box intersections.

    Only rays entering the box beyond their foreground point can block it.
    Retain the 2m near-sensor exclusion, height above candidate ground, and
    1.5m entry-face buffer. Five blockers is a heuristic, not surface coverage.
    """
    hit, enter, _ = ray_box_intervals(points, box)
    b = np.asarray(box)[-7:]
    ranges = np.linalg.norm(points[:, :2], axis=1)
    lengths = np.linalg.norm(points[:, :3], axis=1)
    gap = np.zeros(len(points), dtype=float)
    gap[hit] = (enter[hit] - 1.0) * lengths[hit]
    blockers = (hit & (enter > 1.0) & (ranges >= 2.0)
                & (gap > 1.5)
                & (points[:, 2] > b[5] + min_obstacle_height))
    return bool(np.count_nonzero(blockers) >= max_blocking_points)
