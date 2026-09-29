from typing import Optional, Tuple
import numpy as np
from shapely.geometry import Polygon


def get_box_2d_polygon(box: np.ndarray, margin: float = 0.0) -> Polygon:
    """Returns a Shapely 2D Polygon in BEV (x, y) with optional expansion margin.

    box: 7 elements [h, w, l, bx, by, bz, yaw]
         or 8 elements [cls, h, w, l, bx, by, bz, yaw]
    """
    b = box[1:] if len(box) >= 8 else box
    h, w, l, bx, by, bz, yaw = b[:7]
    w_m = w + 2.0 * margin
    l_m = l + 2.0 * margin

    # 4 corners in local box frame (x is length, y is width)
    corners_local = np.array(
        [
            [-l_m / 2.0, -w_m / 2.0],
            [-l_m / 2.0, w_m / 2.0],
            [l_m / 2.0, w_m / 2.0],
            [l_m / 2.0, -w_m / 2.0],
        ]
    )
    cos_y = np.cos(yaw)
    sin_y = np.sin(yaw)
    rot_mat = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    corners_world = np.dot(corners_local, rot_mat.T) + np.array([bx, by])
    return Polygon(corners_world)


def check_box_collision_2d(
    new_box: np.ndarray, existing_boxes: np.ndarray, min_margin: float = 0.3
) -> bool:
    """Checks if new_box collides with any existing_boxes in BEV (2D plane).

    Fast circular approximation reject followed by exact Shapely Polygon
    intersection.
    """
    if len(existing_boxes) == 0:
        return False

    b_new = new_box[1:] if len(new_box) >= 8 else new_box
    new_w, new_l, new_x, new_y = b_new[1], b_new[2], b_new[3], b_new[4]
    new_r = (
        np.sqrt((new_w + 2.0 * min_margin) ** 2 + (new_l + 2.0 * min_margin) ** 2)
        / 2.0
    )

    b_exist = (
        existing_boxes[:, 1:]
        if existing_boxes.shape[1] >= 8
        else existing_boxes
    )
    exist_w, exist_l = b_exist[:, 1], b_exist[:, 2]
    exist_x, exist_y = b_exist[:, 3], b_exist[:, 4]
    exist_r = np.sqrt(exist_w**2 + exist_l**2) / 2.0

    dist_sq = (new_x - exist_x) ** 2 + (new_y - exist_y) ** 2
    radius_sum_sq = (new_r + exist_r) ** 2
    potential_collision_indices = np.where(dist_sq < radius_sum_sq)[0]

    if len(potential_collision_indices) == 0:
        return False

    poly_new = get_box_2d_polygon(new_box, margin=min_margin)
    for idx in potential_collision_indices:
        poly_exist = get_box_2d_polygon(existing_boxes[idx], margin=0.0)
        if poly_new.intersects(poly_exist):
            return True

    return False


def estimate_local_ground_z(
    points: np.ndarray,
    center_x: float,
    center_y: float,
    radius: float = 3.0,
    default_z: float = -1.6,
) -> float:
    """Estimates local road surface height z using the 5th percentile of local points."""
    if len(points) == 0:
        return default_z
    dist_sq = (points[:, 0] - center_x) ** 2 + (points[:, 1] - center_y) ** 2
    local_pts = points[dist_sq <= radius**2]
    if len(local_pts) < 10:
        return default_z
    z_est = float(np.percentile(local_pts[:, 2], 5))
    if z_est > -0.8 or z_est < -2.5:
        return default_z
    return z_est


def snap_box_to_ground(box: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Snaps the bottom position of box to the local road surface.

    In MobilePIXOR, box[z] is bottom center.
    """
    box_snapped = box.copy()
    z_idx = 6 if len(box) >= 8 else 5
    x_idx = 4 if len(box) >= 8 else 3
    y_idx = 5 if len(box) >= 8 else 4

    z_ground = estimate_local_ground_z(
        points, box_snapped[x_idx], box_snapped[y_idx]
    )
    box_snapped[z_idx] = z_ground
    return box_snapped


def check_ground_support(
    box: np.ndarray,
    points: np.ndarray,
    min_points: Optional[int] = None,
    radius: Optional[float] = None,
) -> Tuple[bool, float]:
    """Verifies that a candidate location is supported by a real ground surface.

    Returns (has_support, ground_z). Rejects empty voids (e.g. behind buildings).
    Adapts search radius and point count to LiDAR beam divergence at range.
    """
    if len(points) == 0:
        return False, -1.6

    b = box[1:] if len(box) >= 8 else box
    bx, by = b[3], b[4]
    r_target = float(np.sqrt(bx**2 + by**2))

    if radius is None:
        radius = float(np.clip(2.5 + 0.03 * max(0.0, r_target - 15.0), 2.5, 4.0))

    if min_points is None:
        if r_target <= 15.0:
            min_pts = 6
        elif r_target <= 30.0:
            min_pts = 4
        else:
            min_pts = 2
    else:
        min_pts = min_points

    dist_sq = (points[:, 0] - bx) ** 2 + (points[:, 1] - by) ** 2
    local_pts = points[dist_sq <= radius**2]

    # Must have minimum points on road
    if len(local_pts) < min_pts:
        return False, -1.6

    # 5th percentile represents ground level
    z_est = float(np.percentile(local_pts[:, 2], 5))
    if not (-2.5 <= z_est <= -0.8):
        return False, -1.6

    # Ground height spread: road surface points must be reasonably consistent
    z_road = local_pts[local_pts[:, 2] <= z_est + 0.35, 2]
    if len(z_road) < min_pts:
        return False, -1.6

    return True, z_est


def check_static_obstacle_collision(
    box: np.ndarray,
    points: np.ndarray,
    max_obstacle_points: int = 3,
    min_height_above_ground: float = 0.35,
) -> bool:
    """Checks whether the 3D volume of box overlaps with elevated non-ground obstacle points.

    Returns True if collision detected (e.g. wall, pole, hedge, building), False if clear.
    """
    if len(points) == 0:
        return False

    b = box[1:] if len(box) >= 8 else box
    h, w, l, bx, by, bz, yaw = b[:7]

    # Broad-phase cylinder pre-filter
    diag = np.sqrt(w**2 + l**2) / 2.0
    dist_sq = (points[:, 0] - bx) ** 2 + (points[:, 1] - by) ** 2
    cand_mask = dist_sq <= (diag + 0.2) ** 2
    if not np.any(cand_mask):
        return False

    cand_pts = points[cand_mask]

    # Local box coordinates
    dx = cand_pts[:, 0] - bx
    dy = cand_pts[:, 1] - by
    cos_y = np.cos(-yaw)
    sin_y = np.sin(-yaw)
    x_rot = dx * cos_y - dy * sin_y
    y_rot = dx * sin_y + dy * cos_y
    z_rot = cand_pts[:, 2] - bz

    in_footprint = (np.abs(x_rot) <= l / 2.0) & (np.abs(y_rot) <= w / 2.0)
    is_elevated = (z_rot >= min_height_above_ground) & (
        z_rot <= max(h + 1.5, 3.5)
    )

    obstacle_count = np.sum(in_footprint & is_elevated)
    return bool(obstacle_count > max_obstacle_points)

