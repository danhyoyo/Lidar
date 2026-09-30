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


try:
    import numba
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

if _HAS_NUMBA:
    @numba.njit(fastmath=True)
    def _ray_box_numba_kernel(xyz, lower, upper, orig, c, s, eps_m):
        N = xyz.shape[0]
        hit = np.ones(N, dtype=numba.boolean)
        enter = np.full(N, np.inf, dtype=numba.float64)
        leave = np.full(N, -np.inf, dtype=numba.float64)

        for i in range(N):
            dx = xyz[i, 0] * c + xyz[i, 1] * s
            dy = -xyz[i, 0] * s + xyz[i, 1] * c
            dz = xyz[i, 2]

            ray_len = np.sqrt(dx * dx + dy * dy + dz * dz)
            if ray_len == 0.0:
                hit[i] = False
                continue

            t_enter = 0.0
            t_leave = np.inf
            is_hit = True

            # Axis 0 (x / length)
            if dx == 0.0:
                if orig[0] < lower[0] - eps_m or orig[0] > upper[0] + eps_m:
                    is_hit = False
            else:
                inv_d = 1.0 / dx
                t1 = (lower[0] - orig[0]) * inv_d
                t2 = (upper[0] - orig[0]) * inv_d
                t_enter = max(t_enter, min(t1, t2))
                t_leave = min(t_leave, max(t1, t2))

            # Axis 1 (y / width)
            if is_hit:
                if dy == 0.0:
                    if orig[1] < lower[1] - eps_m or orig[1] > upper[1] + eps_m:
                        is_hit = False
                else:
                    inv_d = 1.0 / dy
                    t1 = (lower[1] - orig[1]) * inv_d
                    t2 = (upper[1] - orig[1]) * inv_d
                    t_enter = max(t_enter, min(t1, t2))
                    t_leave = min(t_leave, max(t1, t2))

            # Axis 2 (z / height)
            if is_hit:
                if dz == 0.0:
                    if orig[2] < lower[2] - eps_m or orig[2] > upper[2] + eps_m:
                        is_hit = False
                else:
                    inv_d = 1.0 / dz
                    t1 = (lower[2] - orig[2]) * inv_d
                    t2 = (upper[2] - orig[2]) * inv_d
                    t_enter = max(t_enter, min(t1, t2))
                    t_leave = min(t_leave, max(t1, t2))

            if is_hit and t_leave >= 0.0 and (t_enter - t_leave) * ray_len <= eps_m:
                hit[i] = True
                enter[i] = t_enter
                leave[i] = t_leave
            else:
                hit[i] = False

        return hit, enter, leave


def _ray_box_numpy_fallback(xyz, lower, upper, origin, c, s, eps_m):
    directions = np.column_stack((xyz[:, 0] * c + xyz[:, 1] * s,
                                  -xyz[:, 0] * s + xyz[:, 1] * c, xyz[:, 2]))
    enter = np.zeros(len(xyz), dtype=np.float64)
    leave = np.full(len(xyz), np.inf)
    hit = np.any(directions != 0, axis=1)
    for axis in range(3):
        direction = directions[:, axis]
        parallel = direction == 0
        hit &= ~parallel | ((origin[axis] >= lower[axis] - eps_m)
                            & (origin[axis] <= upper[axis] + eps_m))
        moving = ~parallel
        ta = (lower[axis] - origin[axis]) / direction[moving]
        tb = (upper[axis] - origin[axis]) / direction[moving]
        enter[moving] = np.maximum(enter[moving], np.minimum(ta, tb))
        leave[moving] = np.minimum(leave[moving], np.maximum(ta, tb))
    ray_length = np.sqrt(directions[:, 0] ** 2 + directions[:, 1] ** 2 + directions[:, 2] ** 2)
    moving_rays = ray_length > 0
    distance_gap = np.full(len(xyz), np.inf)
    distance_gap[moving_rays] = (enter[moving_rays] - leave[moving_rays]) * ray_length[moving_rays]
    hit &= (leave >= 0) & (distance_gap <= eps_m)
    return hit, np.where(hit, enter, np.inf), np.where(hit, leave, -np.inf)


def ray_box_intervals(points, box):
    """Intersect rays ``t * points[:,:3]`` (t >= 0) with an oriented box.

    Points themselves are at t=1. Return (hit, t_enter, t_exit), with misses
    represented by (+inf, -inf). Entry is clipped to zero for sensors inside
    the box. Zero-length rays never hit. Parallel axes use slab membership;
    denominators are never clipped. Boundary tolerance is 1e-8 meters.
    Accelerated with Numba JIT when available.
    """
    b = _box_values(box)
    xyz = np.asarray(points, dtype=np.float64)
    # Validate and obtain sensor position in the box frame.
    if xyz.ndim != 2 or xyz.shape[1] < 3 or not np.isfinite(xyz).all():
        raise ValueError("points must have shape (N, >=3) and finite values")
    if len(xyz) == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    origin = _local_points(np.zeros((1, 3)), b)[0]
    c, s = np.cos(b[6]), np.sin(b[6])
    lower = np.array([-b[2] / 2, -b[1] / 2, 0.0], dtype=np.float64)
    upper = np.array([b[2] / 2, b[1] / 2, b[0]], dtype=np.float64)
    xyz3 = np.ascontiguousarray(xyz[:, :3], dtype=np.float64)

    if _HAS_NUMBA:
        try:
            return _ray_box_numba_kernel(xyz3, lower, upper, origin, c, s, GEOMETRY_EPS_M)
        except Exception:
            pass
    return _ray_box_numpy_fallback(xyz3, lower, upper, origin, c, s, GEOMETRY_EPS_M)


if _HAS_NUMBA:
    try:
        # Pre-compile Numba kernel at import time so workers inherit compiled machine code
        _dummy_xyz = np.zeros((1, 3), dtype=np.float64)
        _dummy_lower = np.array([-1.0, -1.0, 0.0], dtype=np.float64)
        _dummy_upper = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        _dummy_orig = np.zeros(3, dtype=np.float64)
        _ray_box_numba_kernel(_dummy_xyz, _dummy_lower, _dummy_upper, _dummy_orig, 1.0, 0.0, GEOMETRY_EPS_M)
    except Exception:
        pass
