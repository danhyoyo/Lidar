"""Experimental BEV descriptors from pooled 3D moments in XY windows.

These are CANUPO-inspired descriptors, not spherical-neighborhood CANUPO.
No labels or camera data enter this encoder. Only occupied output cells receive
features; count/geometry validity remain explicit and Hist14 is unchanged.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "detector")]
from core.datasets.utils_1.preprocess import encode_bev, _grid_shape

HIST_NAMES = ["height_count_b0", "height_count_b1", "height_count_b2", "height_count_b3",
              "z_max", "z_mean", "z_span", "z_std", "intensity_max", "intensity_mean",
              "log_density", "occupancy", "point_offset_x", "point_offset_y"]
RICH_NAMES = ["height_occupancy_b0", "height_occupancy_b1", "height_occupancy_b2",
              "z_max", "z_mean", "intensity_max", "intensity_mean", "log_density"]
SCALES = (1., 2.)


def channel_names(scales=SCALES):
    support, shape = [], []
    for s in scales:
        support.extend([f"neighbor_log_count_{s:g}m", f"geometry_valid_{s:g}m"])
        shape.extend([f"dimensionality_1d_{s:g}m", f"dimensionality_2d_{s:g}m"])
    return {"rich8": RICH_NAMES, "hist14": HIST_NAMES,
            "hist18": HIST_NAMES + support, "hist22": HIST_NAMES + support + shape}


def box_sums_at_cells(integral, iy, ix, ry, rx):
    """Summed-area tables have an extra zero row/column; clip at real ROI borders."""
    height, width = integral.shape[1] - 1, integral.shape[2] - 1
    y0, y1 = np.maximum(0, iy - ry), np.minimum(height, iy + ry + 1)
    x0, x1 = np.maximum(0, ix - rx), np.minimum(width, ix + rx + 1)
    return (integral[:, y1, x1] - integral[:, y0, x1]
            - integral[:, y1, x0] + integral[:, y0, x0]).T


def covariance_from_moments(sums):
    counts = sums[:, 0]
    means = sums[:, 1:4] / np.maximum(counts[:, None], 1.)
    cov = np.empty((len(sums), 3, 3), np.float64)
    for col, (i, j) in enumerate([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)], 4):
        value = sums[:, col] / np.maximum(counts, 1.) - means[:, i] * means[:, j]
        cov[:, i, j] = value
        cov[:, j, i] = value
    return cov


def dimensionality(covariance, counts, min_neighbors=6):
    """Use CANUPO's normalized-variance barycentric dimensionality convention."""
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance)[:, ::-1], 0.)
    total = eigenvalues.sum(axis=1)
    valid = (counts >= min_neighbors) & (total > 1e-12)
    bary = np.zeros_like(eigenvalues)
    e = eigenvalues[valid] / total[valid, None]
    bary[valid, 0] = e[:, 0] - e[:, 1]
    bary[valid, 1] = 2 * (e[:, 1] - e[:, 2])
    bary[valid, 2] = 3 * e[:, 2]
    return np.clip(bary, 0., 1.).astype(np.float32), valid


def encode_maps(points, geometry, scales=SCALES, min_neighbors=6):
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 4 or points.dtype.kind not in "fiu":
        raise ValueError("Expected numeric XYZI points with shape (N, >=4)")
    if min_neighbors < 3 or not all(np.isfinite(s) and s > 0 for s in scales):
        raise ValueError("Invalid scale or minimum neighbor count")
    if len(scales) != 2:
        raise ValueError("This 18/22-channel experiment uses exactly two scales")
    width, height, _ = _grid_shape(geometry)
    start = time.perf_counter()
    hist = encode_bev(points, geometry, {"name": "hist14", "backend": "numpy"})
    hist_time = time.perf_counter() - start
    start = time.perf_counter()
    rich = encode_bev(points, geometry, {"name": "rich8"})
    rich_time = time.perf_counter() - start
    start = time.perf_counter()
    promoted_points = points[:, :4].astype(np.float64)
    eligible = np.isfinite(promoted_points).all(axis=1)
    for col, axis in enumerate("xyz"):
        eligible &= ((promoted_points[:, col] > geometry[f"{axis}_min"] + .001)
                     & (promoted_points[:, col] < geometry[f"{axis}_max"] - .001))
    # Float64 and a central reference reduce cancellation in second moments.
    pts = promoted_points[eligible, :3]
    ix = np.floor((pts[:, 0] - geometry["x_min"]) / geometry["x_res"]).astype(np.int64)
    iy = np.floor((pts[:, 1] - geometry["y_min"]) / geometry["y_res"]).astype(np.int64)
    flat = iy * width + ix
    xyz = pts - np.array([(geometry[f"{a}_min"] + geometry[f"{a}_max"]) / 2 for a in "xyz"])
    moments = np.empty((10, height, width), np.float64)
    moments[0] = np.bincount(flat, minlength=height * width).reshape(height, width)
    for j in range(3):
        moments[1 + j] = np.bincount(flat, weights=xyz[:, j], minlength=height * width).reshape(height, width)
    for col, (j, k) in enumerate([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)], 4):
        moments[col] = np.bincount(flat, weights=xyz[:, j] * xyz[:, k], minlength=height * width).reshape(height, width)
    occupied = moments[0] > 0
    oy, ox = np.where(occupied)
    integral = np.pad(moments, ((0, 0), (1, 0), (1, 0))).cumsum(axis=1).cumsum(axis=2)
    support = np.zeros((height, width, 4), np.float32)
    shape = np.zeros((height, width, 4), np.float32)
    counts = np.zeros((height, width, 2), np.int32)
    barycentric = np.zeros((height, width, 2, 3), np.float32)
    windows = []
    for j, scale in enumerate(scales):
        rx = int(np.floor(scale / (2 * geometry["x_res"]) + .5))
        ry = int(np.floor(scale / (2 * geometry["y_res"]) + .5))
        pooled = box_sums_at_cells(integral, oy, ox, ry, rx)
        count = np.rint(pooled[:, 0]).astype(np.int32)
        bary, valid = dimensionality(covariance_from_moments(pooled), count, min_neighbors)
        support[oy, ox, 2 * j] = np.log1p(count)
        support[oy, ox, 2 * j + 1] = valid
        shape[oy, ox, 2 * j:2 * j + 2] = bary[:, :2]
        counts[oy, ox, j] = count
        barycentric[oy, ox, j] = bary
        windows.append({"nominal_scale_m": float(scale), "cells_x": 2 * rx + 1,
                        "cells_y": 2 * ry + 1, "width_x_m": (2 * rx + 1) * geometry["x_res"],
                        "width_y_m": (2 * ry + 1) * geometry["y_res"]})
    hist18 = np.concatenate([hist, support], axis=-1)
    hist22 = np.concatenate([hist18, shape], axis=-1)
    geometry_time = time.perf_counter() - start
    return {"rich8": rich, "hist14": hist, "hist18": hist18, "hist22": hist22,
            "counts": counts, "barycentric": barycentric, "occupied": occupied,
            "windows": windows, "timing": {"rich8_seconds": rich_time, "hist14_seconds": hist_time,
                "geometry_seconds": geometry_time, "hist18_22_seconds": hist_time + geometry_time},
            "roi_points": len(pts)}
