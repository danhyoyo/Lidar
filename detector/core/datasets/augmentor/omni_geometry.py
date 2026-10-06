"""Vectorized 2D/3D geometry operations for Omni-PCU Augmentation."""

from __future__ import annotations
import numpy as np


def boxes_to_bev_corners(boxes: np.ndarray) -> np.ndarray:
    """Convert (N, 7) or (N, 8) boxes to (N, 4, 2) BEV corner coordinates.
    Box convention: [h, w, l, x, y, z, yaw] or [class, h, w, l, x, y, z, yaw].
    """
    if len(boxes) == 0:
        return np.empty((0, 4, 2), dtype=np.float32)
    b = boxes[:, 1:] if boxes.shape[1] >= 8 else boxes
    w, l, x, y, yaw = b[:, 1], b[:, 2], b[:, 3], b[:, 4], b[:, 6]

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # 4 local corners (x along length, y along width)
    # Order: Front-Left, Front-Right, Rear-Right, Rear-Left
    half_l = l * 0.5
    half_w = w * 0.5

    x_corners = np.stack([half_l, half_l, -half_l, -half_l], axis=1)  # (N, 4)
    y_corners = np.stack([half_w, -half_w, -half_w, half_w], axis=1)  # (N, 4)

    # Rotate and translate
    x_rot = x_corners * cos_yaw[:, None] - y_corners * sin_yaw[:, None] + x[:, None]
    y_rot = x_corners * sin_yaw[:, None] + y_corners * cos_yaw[:, None] + y[:, None]

    return np.stack([x_rot, y_rot], axis=-1)  # (N, 4, 2)


def check_collision_2d_vectorized(
    new_boxes: np.ndarray,
    existing_boxes: np.ndarray,
    min_margin: float = 0.0,
) -> np.ndarray:
    """Vectorized BEV collision detection using Broad-Phase Circle Reject
    followed by Narrow-Phase Separating Axis Theorem (SAT).
    
    Returns boolean array of shape (len(new_boxes),) indicating collision.
    """
    if len(existing_boxes) == 0 or len(new_boxes) == 0:
        return np.zeros(len(new_boxes), dtype=bool)

    b_new = new_boxes[:, 1:] if new_boxes.shape[1] >= 8 else new_boxes
    b_exist = existing_boxes[:, 1:] if existing_boxes.shape[1] >= 8 else existing_boxes

    # Broad-phase: Bounding circle radius
    r_new = np.hypot(b_new[:, 1] + 2.0 * min_margin, b_new[:, 2] + 2.0 * min_margin) * 0.5
    r_exist = np.hypot(b_exist[:, 1], b_exist[:, 2]) * 0.5

    # Pairwise center distances (N_new, N_exist)
    dx = b_new[:, 3:4] - b_exist[:, 3:4].T
    dy = b_new[:, 4:5] - b_exist[:, 4:5].T
    dist_sq = dx * dx + dy * dy
    radius_sum_sq = (r_new[:, None] + r_exist[None, :]) ** 2

    broad_phase_coll = dist_sq < radius_sum_sq  # (N_new, N_exist)
    if not np.any(broad_phase_coll):
        return np.zeros(len(new_boxes), dtype=bool)

    # Narrow-phase: SAT on candidate pairs
    new_corners = boxes_to_bev_corners(new_boxes)  # (N_new, 4, 2)
    exist_corners = boxes_to_bev_corners(existing_boxes)  # (N_exist, 4, 2)

    has_collision = np.zeros(len(new_boxes), dtype=bool)
    for i in range(len(new_boxes)):
        cand_indices = np.where(broad_phase_coll[i])[0]
        if len(cand_indices) == 0:
            continue
        c1 = new_corners[i]  # (4, 2)
        for j in cand_indices:
            c2 = exist_corners[j]  # (4, 2)
            if _sat_polygon_intersection(c1, c2, min_margin):
                has_collision[i] = True
                break

    return has_collision


def _sat_polygon_intersection(c1: np.ndarray, c2: np.ndarray, margin: float = 0.0) -> bool:
    """Check 2D oriented rectangle overlap via Separating Axis Theorem (SAT)."""
    # 4 potential separating axes: normals to edges of c1 and c2
    edges1 = np.roll(c1, -1, axis=0) - c1
    edges2 = np.roll(c2, -1, axis=0) - c2
    axes = np.concatenate([edges1[:2], edges2[:2]], axis=0)
    # Orthogonal normals: [-dy, dx]
    normals = np.stack([-axes[:, 1], axes[:, 0]], axis=-1)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-8)

    for normal in normals:
        proj1 = np.dot(c1, normal)
        proj2 = np.dot(c2, normal)
        min1, max1 = proj1.min() - margin, proj1.max() + margin
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 or max2 < min1:
            return False  # Separating axis found
    return True


def project_to_road_plane(
    x: np.ndarray | float,
    y: np.ndarray | float,
    plane: np.ndarray,
) -> np.ndarray | float:
    """Compute road elevation z from KITTI road plane equation: ax + by + cz + d = 0.
    
    Guards against degenerate or vertical planes (|c| < 1e-5).
    """
    a, b, c, d = plane[:4]
    if abs(c) < 1e-5:
        # Fallback to standard KITTI LiDAR height if normal is degenerate
        return np.full_like(x, -1.65, dtype=np.float64) if isinstance(x, np.ndarray) else -1.65
    return -(a * x + b * y + d) / c


def points_in_oriented_box_3d(
    points: np.ndarray,
    box: np.ndarray,
    extra_margin: np.ndarray | None = None,
) -> np.ndarray:
    """Return boolean mask of points inside a 3D bounding box.
    Box: [h, w, l, x, y, z_bottom, yaw] or [class, h, w, l, x, y, z_bottom, yaw].
    """
    if len(points) == 0:
        return np.zeros(0, dtype=bool)

    b = box[1:] if len(box) >= 8 else box
    h, w, l, bx, by, bz, yaw = b[:7]

    if extra_margin is not None:
        h = h + extra_margin[0]
        w = w + extra_margin[1]
        l = l + extra_margin[2]

    # Translate points
    pts = points[:, :3].copy()
    pts[:, 0] -= bx
    pts[:, 1] -= by
    pts[:, 2] -= (bz + h * 0.5)

    # Rotate into box coordinate frame
    cos_y = np.cos(-yaw)
    sin_y = np.sin(-yaw)
    x_rot = pts[:, 0] * cos_y - pts[:, 1] * sin_y
    y_rot = pts[:, 0] * sin_y + pts[:, 1] * cos_y
    z_rot = pts[:, 2]

    in_x = np.abs(x_rot) <= (l * 0.5)
    in_y = np.abs(y_rot) <= (w * 0.5)
    in_z = np.abs(z_rot) <= (h * 0.5)

    return in_x & in_y & in_z
