"""CPU geometry for hybrid sampling; bottom-centered [class,h,w,l,x,y,z,yaw].

Inspired by MMDetection3D's batched box collision / points-in-box utilities.
Implemented independently with SAT, streaming union masks and slab ray tests.
Numba is optional; no CUDA/MMCV/MMDetection3D runtime dependency is required.
"""
from __future__ import annotations

import math
import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None

EPS_M = 1e-8


def _jit(function):
    return njit(cache=True)(function) if njit is not None else function


def resolve_backend(backend):
    if backend not in ("auto", "numpy", "numba"):
        raise ValueError("GEOMETRY_BACKEND must be auto, numpy or numba")
    if backend == "numba" and njit is None:
        raise RuntimeError("Numba backend requested; install numba or use numpy/auto")
    return "numba" if backend != "numpy" and njit is not None else "numpy"


def _boxes(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 8:
        raise ValueError("boxes must have shape (M, 8)")
    if not np.isfinite(values).all() or (values[:, 1:4] <= 0).any():
        raise ValueError("boxes must be finite with positive dimensions")
    return np.ascontiguousarray(values)


def _points(values):
    values = np.asarray(values)
    if values.ndim != 2 or values.shape[1] < 3 or not np.isfinite(values).all():
        raise ValueError("points must have finite shape (N, >=3)")
    return values


def _rectangles(boxes, margin=0):
    out = np.empty((len(boxes), 8), np.float64)
    out[:, :2] = boxes[:, 4:6]
    out[:, 2] = boxes[:, 3] / 2 + margin
    out[:, 3] = boxes[:, 2] / 2 + margin
    out[:, 4] = np.cos(boxes[:, 7])
    out[:, 5] = np.sin(boxes[:, 7])
    out[:, 6] = np.abs(out[:, 4]) * out[:, 2] + np.abs(out[:, 5]) * out[:, 3]
    out[:, 7] = np.abs(out[:, 5]) * out[:, 2] + np.abs(out[:, 4]) * out[:, 3]
    return out


@_jit
def _collision_kernel(a, b):
    result = np.zeros((len(a), len(b)), np.bool_)
    for i in range(len(a)):
        for j in range(len(b)):
            dx, dy = b[j, 0] - a[i, 0], b[j, 1] - a[i, 1]
            if abs(dx) >= a[i, 6] + b[j, 6] - EPS_M or abs(dy) >= a[i, 7] + b[j, 7] - EPS_M:
                continue
            ca, sa, cb, sb = a[i, 4], a[i, 5], b[j, 4], b[j, 5]
            dot, cross = abs(ca * cb + sa * sb), abs(ca * sb - sa * cb)
            overlap = (
                abs(dx * ca + dy * sa) < a[i, 2] + b[j, 2] * dot + b[j, 3] * cross - EPS_M
                and abs(-dx * sa + dy * ca) < a[i, 3] + b[j, 2] * cross + b[j, 3] * dot - EPS_M
                and abs(dx * cb + dy * sb) < b[j, 2] + a[i, 2] * dot + a[i, 3] * cross - EPS_M
                and abs(-dx * sb + dy * cb) < b[j, 3] + a[i, 2] * cross + a[i, 3] * dot - EPS_M
            )
            result[i, j] = overlap
    return result


def collision_matrix(candidates, occupied, margin=0.0, *, backend="auto"):
    """Strict positive BEV overlap; touching boundaries are allowed. Margin expands candidates."""
    candidates, occupied = _boxes(candidates), _boxes(occupied)
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("collision margin must be finite and nonnegative")
    a, b = _rectangles(candidates, margin), _rectangles(occupied)
    if resolve_backend(backend) == "numba":
        return _collision_kernel(a, b)
    # Vectorized SAT: four separating axes plus a world-axis AABB broad phase.
    dx, dy = b[None, :, 0] - a[:, None, 0], b[None, :, 1] - a[:, None, 1]
    ca, sa = a[:, None, 4], a[:, None, 5]
    cb, sb = b[None, :, 4], b[None, :, 5]
    dot, cross = np.abs(ca * cb + sa * sb), np.abs(ca * sb - sa * cb)
    la, wa, lb, wb = a[:, None, 2], a[:, None, 3], b[None, :, 2], b[None, :, 3]
    return ((np.abs(dx) < a[:, None, 6] + b[None, :, 6] - EPS_M)
            & (np.abs(dy) < a[:, None, 7] + b[None, :, 7] - EPS_M)
            & (np.abs(dx * ca + dy * sa) < la + lb * dot + wb * cross - EPS_M)
            & (np.abs(-dx * sa + dy * ca) < wa + lb * cross + wb * dot - EPS_M)
            & (np.abs(dx * cb + dy * sb) < lb + la * dot + wa * cross - EPS_M)
            & (np.abs(-dx * sb + dy * cb) < wb + la * cross + wa * dot - EPS_M))


@_jit
def _union_kernel(points, boxes, extra):
    result = np.zeros(len(points), np.bool_)
    for b in boxes:
        cosine, sine = math.cos(b[7]), math.sin(b[7])
        hl, hw = b[3] / 2 + extra[0], b[2] / 2 + extra[1]
        ex, ey = abs(cosine) * hl + abs(sine) * hw, abs(sine) * hl + abs(cosine) * hw
        for i in range(len(points)):
            if result[i]:
                continue
            dx, dy, z = np.float64(points[i, 0]) - b[4], np.float64(points[i, 1]) - b[5], points[i, 2]
            if abs(dx) > ex or abs(dy) > ey or z < b[6] - extra[2] or z > b[6] + b[1] + extra[2]:
                continue
            if abs(dx * cosine + dy * sine) <= hl and abs(-dx * sine + dy * cosine) <= hw:
                result[i] = True
    return result


def points_in_boxes_union(points, boxes, extra=(0., 0., 0.), *, backend="auto"):
    """Return an N-element union mask without allocating an N x M matrix."""
    points, boxes = _points(points), _boxes(boxes)
    extra = np.asarray(extra, np.float64)
    if extra.shape != (3,) or not np.isfinite(extra).all() or (extra < 0).any():
        raise ValueError("extra must contain three finite nonnegative margins")
    if resolve_backend(backend) == "numba":
        return _union_kernel(points, boxes, extra)
    from .geometry import points_in_box
    mask = np.zeros(len(points), bool)
    for b in boxes:
        mask |= points_in_box(points, b, extra)
    return mask


@_jit
def _ray_kernel(points, b):
    hit = np.zeros(len(points), np.bool_)
    enter = np.full(len(points), np.inf)
    leave = np.full(len(points), -np.inf)
    c, s = math.cos(b[7]), math.sin(b[7])
    origin = np.array([-b[4] * c - b[5] * s, b[4] * s - b[5] * c, -b[6]])
    lower, upper = np.array([-b[3]/2, -b[2]/2, 0.]), np.array([b[3]/2, b[2]/2, b[1]])
    for i in range(len(points)):
        direction = np.array([points[i, 0]*c + points[i, 1]*s,
                              -points[i, 0]*s + points[i, 1]*c, np.float64(points[i, 2])])
        norm = math.sqrt(np.sum(direction * direction))
        if norm == 0:
            continue
        lo, hi, valid = 0., np.inf, True
        for axis in range(3):
            if direction[axis] == 0:
                if origin[axis] < lower[axis] - EPS_M or origin[axis] > upper[axis] + EPS_M:
                    valid = False
                    break
            else:
                ta = (lower[axis] - origin[axis]) / direction[axis]
                tb = (upper[axis] - origin[axis]) / direction[axis]
                lo, hi = max(lo, min(ta, tb)), min(hi, max(ta, tb))
        if valid and hi >= 0 and (lo - hi) * norm <= EPS_M:
            hit[i], enter[i], leave[i] = True, lo, hi
    return hit, enter, leave


def ray_box_intervals(points, box, *, backend="auto"):
    points, boxes = _points(points), _boxes(np.asarray(box)[None])
    if resolve_backend(backend) == "numba":
        return _ray_kernel(points, boxes[0])
    b = boxes[0]
    c, s = np.cos(b[7]), np.sin(b[7])
    origin = np.array([-b[4]*c - b[5]*s, b[4]*s - b[5]*c, -b[6]])
    directions = np.column_stack((points[:, 0].astype(float)*c + points[:, 1]*s,
                                  -points[:, 0].astype(float)*s + points[:, 1]*c, points[:, 2]))
    low, high = [-b[3]/2, -b[2]/2, 0], [b[3]/2, b[2]/2, b[1]]
    enter, leave, hit = np.zeros(len(points)), np.full(len(points), np.inf), np.any(directions != 0, axis=1)
    for axis in range(3):
        d = directions[:, axis]
        moving = d != 0
        hit &= moving | ((origin[axis] >= low[axis] - EPS_M) & (origin[axis] <= high[axis] + EPS_M))
        ta, tb = (low[axis] - origin[axis]) / d[moving], (high[axis] - origin[axis]) / d[moving]
        enter[moving] = np.maximum(enter[moving], np.minimum(ta, tb))
        leave[moving] = np.minimum(leave[moving], np.maximum(ta, tb))
    lengths = np.linalg.norm(directions, axis=1)
    gap = np.full(len(points), np.inf)
    np.multiply(enter - leave, lengths, out=gap, where=lengths > 0)
    hit &= (leave >= 0) & (gap <= EPS_M)
    return hit, np.where(hit, enter, np.inf), np.where(hit, leave, -np.inf)
