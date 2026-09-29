import numpy as np


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


def mask_shadow_points(bg_points: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Removes background points lying in the line-of-sight shadow frustum behind box.

    box: 7 elements [h, w, l, bx, by, bz, yaw] (bz is BOTTOM center)
         or 8 elements [cls, h, w, l, bx, by, bz, yaw]
    """
    if len(bg_points) == 0:
        return bg_points

    b = box[1:] if len(box) >= 8 else box
    h, w, l, bx, by, bz, yaw = b[:7]
    r_box = np.sqrt(bx**2 + by**2)
    if r_box < 1.0:
        return bg_points

    # Exact angular extents from sensor origin
    half_diag = np.sqrt(l**2 + w**2) / 2.0
    delta_azimuth = np.arctan2(half_diag, r_box)
    azimuth_box = np.arctan2(by, bx)

    elev_min = np.arctan2(bz, r_box)
    elev_max = np.arctan2(bz + h, r_box)

    # Spherical coordinates of background points
    r_bg = np.sqrt(bg_points[:, 0] ** 2 + bg_points[:, 1] ** 2)
    azimuth_bg = np.arctan2(bg_points[:, 1], bg_points[:, 0])
    elev_bg = np.arctan2(bg_points[:, 2], np.maximum(r_bg, 1e-6))

    # Angular difference with wrap-around in [-pi, pi]
    az_diff = np.abs((azimuth_bg - azimuth_box + np.pi) % (2 * np.pi) - np.pi)

    in_range = r_bg > (r_box + half_diag)
    in_azimuth = az_diff <= delta_azimuth
    in_elev = (elev_bg >= elev_min) & (elev_bg <= elev_max)

    shadow_mask = in_range & in_azimuth & in_elev
    return bg_points[~shadow_mask]


def check_line_of_sight_occlusion(
    box: np.ndarray,
    points: np.ndarray,
    max_blocking_points: int = 5,
    min_obstacle_height: float = 0.4,
) -> bool:
    """Verifies that line-of-sight from Ego (0, 0, 0) to box is not blocked by foreground obstacles.

    Returns True if occluded (e.g. placed behind a building, wall, or another vehicle), False if line-of-sight is clear.
    """
    if len(points) == 0:
        return False

    b = box[1:] if len(box) >= 8 else box
    h, w, l, bx, by, bz, _ = b[:7]
    r_target = np.sqrt(bx**2 + by**2)
    if r_target < 3.0:
        return False

    # Angular wedge spanned by candidate box
    diag = np.sqrt(w**2 + l**2) / 2.0
    delta_azimuth = np.arctan2(diag, r_target)
    azimuth_target = np.arctan2(by, bx)

    # Spherical coordinates of background points
    r_pts = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)

    # Only consider foreground points between Ego and candidate box (with 1.5m buffer)
    fg_mask = (r_pts >= 2.0) & (r_pts < (r_target - 1.5))
    if not np.any(fg_mask):
        return False

    fg_pts = points[fg_mask]
    r_fg = r_pts[fg_mask]
    azimuth_fg = np.arctan2(fg_pts[:, 1], fg_pts[:, 0])

    # Angular difference with wrap-around in [-pi, pi]
    az_diff = np.abs((azimuth_fg - azimuth_target + np.pi) % (2 * np.pi) - np.pi)
    in_azimuth = az_diff <= (delta_azimuth * 0.95)

    # Height of ray from Ego (0,0,0) to box bottom at distance r_fg
    z_ray_bottom = r_fg * (bz / r_target)
    is_blocking_ray = fg_pts[:, 2] >= (z_ray_bottom - 0.15)
    is_tall = fg_pts[:, 2] > (bz + min_obstacle_height)

    blocking_count = np.sum(in_azimuth & is_blocking_ray & is_tall)
    return bool(blocking_count >= max_blocking_points)

