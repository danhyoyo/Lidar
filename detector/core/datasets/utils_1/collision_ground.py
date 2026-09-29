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
