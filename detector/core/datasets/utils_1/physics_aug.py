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
    surface and interior points out of the shadow deletion mask. Uses
    conservative range and azimuthal sector pre-filtering for acceleration.
    """
    if len(bg_points) == 0:
        return np.zeros(0, dtype=bool)

    b = np.asarray(box)[-7:]
    h, w, l, bx, by, bz, yaw = b
    r_center = np.hypot(bx, by)
    r_box = np.hypot(l * 0.5, w * 0.5)
    r_min = max(0.0, r_center - r_box - GEOMETRY_EPS_M)

    pt_ranges_sq = bg_points[:, 0] ** 2 + bg_points[:, 1] ** 2
    r_cand = pt_ranges_sq >= (r_min * r_min)
    if not np.any(r_cand):
        return np.zeros(len(bg_points), dtype=bool)

    if r_center > r_box:
        theta_box = np.arctan2(by, bx)
        sin_dtheta = min(1.0, r_box / r_center)
        dtheta = np.arcsin(sin_dtheta) + 0.05
        cand_indices = np.where(r_cand)[0]
        sub_pts = bg_points[cand_indices]
        theta_pts = np.arctan2(sub_pts[:, 1], sub_pts[:, 0])
        angle_diff = np.abs(np.arctan2(np.sin(theta_pts - theta_box), np.cos(theta_pts - theta_box)))
        angle_cand = angle_diff <= dtheta
        if not np.any(angle_cand):
            return np.zeros(len(bg_points), dtype=bool)
        full_cand = cand_indices[angle_cand]
    else:
        full_cand = np.where(r_cand)[0]

    sub_pts = bg_points[full_cand]
    hit, _, leave = ray_box_intervals(sub_pts, box)
    sub_lengths = np.sqrt(sub_pts[:, 0] ** 2 + sub_pts[:, 1] ** 2 + sub_pts[:, 2] ** 2)
    sub_mask = np.zeros(len(sub_pts), dtype=bool)
    sub_mask[hit] = (1.0 - leave[hit]) * sub_lengths[hit] > GEOMETRY_EPS_M

    out = np.zeros(len(bg_points), dtype=bool)
    out[full_cand] = sub_mask
    return out


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
    Retains the 2m near-sensor exclusion, height above candidate ground, and
    1.5m entry-face buffer. Pre-filters points outside bounding range and angle.
    """
    if len(points) == 0:
        return False

    b = np.asarray(box)[-7:]
    h, w, l, bx, by, bz, yaw = b
    r_center = np.hypot(bx, by)
    r_box = np.hypot(l * 0.5, w * 0.5)
    r_max = r_center + r_box + GEOMETRY_EPS_M

    pt_ranges_sq = points[:, 0] ** 2 + points[:, 1] ** 2
    cand = (
        (pt_ranges_sq >= 4.0)
        & (pt_ranges_sq <= r_max * r_max)
        & (points[:, 2] > bz + min_obstacle_height)
    )
    if np.count_nonzero(cand) < max_blocking_points:
        return False

    cand_indices = np.where(cand)[0]
    sub_pts = points[cand_indices]

    if r_center > r_box:
        theta_box = np.arctan2(by, bx)
        sin_dtheta = min(1.0, r_box / r_center)
        dtheta = np.arcsin(sin_dtheta) + 0.05
        theta_pts = np.arctan2(sub_pts[:, 1], sub_pts[:, 0])
        angle_diff = np.abs(np.arctan2(np.sin(theta_pts - theta_box), np.cos(theta_pts - theta_box)))
        angle_cand = angle_diff <= dtheta
        if np.count_nonzero(angle_cand) < max_blocking_points:
            return False
        sub_pts = sub_pts[angle_cand]

    hit, enter, _ = ray_box_intervals(sub_pts, box)
    sub_lengths = np.sqrt(sub_pts[:, 0] ** 2 + sub_pts[:, 1] ** 2 + sub_pts[:, 2] ** 2)
    gap = (enter - 1.0) * sub_lengths
    blockers = hit & (enter > 1.0) & (gap > 1.5)
    return bool(np.count_nonzero(blockers) >= max_blocking_points)
