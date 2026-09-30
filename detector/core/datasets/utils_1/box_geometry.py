"""Sensor-origin ray intersections with bottom-centered oriented boxes.

The box is a solid geometric proxy, not a model of object surfaces or beams.
"""
import numpy as np

# Meters: tolerance for slab membership and for near-tangent intersections.
GEOMETRY_EPS_M = 1e-8


def _box_values(box):
    b = np.asarray(box, dtype=np.float64)
    if b.shape not in {(7,), (8,)} or not np.isfinite(b).all():
        raise ValueError('box must have 7 or 8 finite values')
    b = b[-7:]
    if np.any(b[:3] <= 0):
        raise ValueError('box dimensions must be positive')
    return b


def _local_points(points, b):
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] < 3 or not np.isfinite(xyz).all():
        raise ValueError('points must have shape (N, >=3) and finite values')
    c, s = np.cos(b[6]), np.sin(b[6])
    local = xyz[:, :3] - b[3:6]
    return np.column_stack((local[:, 0] * c + local[:, 1] * s,
                            -local[:, 0] * s + local[:, 1] * c, local[:, 2]))


def points_in_box(points, box):
    """Return inclusive volume membership, preserving overlapping memberships."""
    b = _box_values(box)
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3 or not np.isfinite(pts).all():
        raise ValueError('points must have shape (N, >=3) and finite values')
    if len(pts) == 0:
        return np.zeros(0, dtype=bool)

    h, w, l, bx, by, bz, yaw = b
    r_sq = (w * 0.5) ** 2 + (l * 0.5) ** 2 + GEOMETRY_EPS_M
    dx = pts[:, 0] - bx
    dy = pts[:, 1] - by
    dz = pts[:, 2] - bz
    cand = (dx * dx + dy * dy <= r_sq) & (dz >= -GEOMETRY_EPS_M) & (dz <= h + GEOMETRY_EPS_M)
    if not np.any(cand):
        return np.zeros(len(pts), dtype=bool)

    cand_indices = np.where(cand)[0]
    local = _local_points(pts[cand_indices], b)
    sub_mask = ((np.abs(local[:, 0]) <= l * 0.5 + GEOMETRY_EPS_M)
                & (np.abs(local[:, 1]) <= w * 0.5 + GEOMETRY_EPS_M)
                & (local[:, 2] >= -GEOMETRY_EPS_M)
                & (local[:, 2] <= h + GEOMETRY_EPS_M))
    out = np.zeros(len(pts), dtype=bool)
    out[cand_indices] = sub_mask
    return out


def ray_box_intervals(points, box):
    """Intersect rays ``t * points[:,:3]`` (t >= 0) with an oriented box.

    Points themselves are at t=1. Return (hit, t_enter, t_exit), with misses
    represented by (+inf, -inf). Entry is clipped to zero for sensors inside
    the box. Zero-length rays never hit. Parallel axes use slab membership;
    denominators are never clipped. Boundary tolerance is 1e-8 meters.
    """
    b = _box_values(box)
    xyz = np.asarray(points, dtype=np.float64)
    # Validate and obtain sensor position in the box frame.
    if xyz.ndim != 2 or xyz.shape[1] < 3 or not np.isfinite(xyz).all():
        raise ValueError("points must have shape (N, >=3) and finite values")
    origin = _local_points(np.zeros((1, 3)), b)[0]
    # Rotate directions directly to avoid subtracting large translated values.
    c, s = np.cos(b[6]), np.sin(b[6])
    directions = np.column_stack((xyz[:, 0] * c + xyz[:, 1] * s,
                                  -xyz[:, 0] * s + xyz[:, 1] * c, xyz[:, 2]))
    lower = np.array([-b[2] / 2, -b[1] / 2, 0.0])
    upper = np.array([b[2] / 2, b[1] / 2, b[0]])
    enter = np.zeros(len(xyz), dtype=np.float64)
    leave = np.full(len(xyz), np.inf)
    hit = np.any(directions != 0, axis=1)
    for axis in range(3):
        direction = directions[:, axis]
        parallel = direction == 0
        hit &= ~parallel | ((origin[axis] >= lower[axis] - GEOMETRY_EPS_M)
                            & (origin[axis] <= upper[axis] + GEOMETRY_EPS_M))
        moving = ~parallel
        ta = (lower[axis] - origin[axis]) / direction[moving]
        tb = (upper[axis] - origin[axis]) / direction[moving]
        enter[moving] = np.maximum(enter[moving], np.minimum(ta, tb))
        leave[moving] = np.minimum(leave[moving], np.maximum(ta, tb))
    # Express intersection tolerance as a distance along the sensor ray.
    ray_length = np.linalg.norm(directions, axis=1)
    moving_rays = ray_length > 0
    distance_gap = np.full(len(xyz), np.inf)
    distance_gap[moving_rays] = (enter[moving_rays] - leave[moving_rays]) * ray_length[moving_rays]
    hit &= (leave >= 0) & (distance_gap <= GEOMETRY_EPS_M)
    return hit, np.where(hit, enter, np.inf), np.where(hit, leave, -np.inf)
