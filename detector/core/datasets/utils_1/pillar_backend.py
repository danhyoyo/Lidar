"""Prepare uncapped occupied pillars; learning happens in the model, not NumPy."""

import numpy as np

try:
    from core.bev_encoding import resolve_bev_encoding
except ImportError:
    from detector.core.bev_encoding import resolve_bev_encoding


def prepare_pillars(points, geometry, encoding=None):
    """Return packed ten-dimensional point features and occupied-pillar indices.

    Every finite ROI point is retained. No fixed-size padding, point sampling or
    maximum-pillar cap is used. Coordinates are [batch, y, x], initially batch 0.
    """
    schema = resolve_bev_encoding(encoding or {"name": "pillar32"}, geometry)
    if not schema.is_packed:
        raise ValueError("prepare_pillars requires a packed pillar encoding")
    if points.ndim != 2 or points.shape[1] < 4:
        raise ValueError(f"{schema.name} expects points shaped (N, >=4)")
    keep = np.isfinite(points[:, :4]).all(axis=1)
    for column, axis in enumerate("xyz"):
        keep &= points[:, column] > geometry[f"{axis}_min"] + .001
        keep &= points[:, column] < geometry[f"{axis}_max"] - .001
    pts = np.array(points[keep, :4], dtype=np.float32, copy=True)
    if not len(pts):
        result = {"features": np.empty((0, 10), dtype=np.float32),
                  "pillar_indices": np.empty(0, dtype=np.int64),
                  "coords": np.empty((0, 3), dtype=np.int64), "batch_size": 1}
        if schema.is_rich_pillar:
            result["rich_features"] = np.empty((0, 8), dtype=np.float32)
        return result
    # Match rich8's float32 floor-division at exact cell boundaries so encoder
    # comparisons do not accidentally change XY membership as well.
    x = ((pts[:, 0] - geometry["x_min"]) // geometry["x_res"]).astype(np.int64)
    y = ((pts[:, 1] - geometry["y_min"]) // geometry["y_res"]).astype(np.int64)
    width = schema.grid_shape[0]
    occupied, inverse, counts = np.unique(y * width + x, return_inverse=True, return_counts=True)
    mean = np.column_stack([np.bincount(inverse, weights=pts[:, axis], minlength=len(occupied)) / counts
                            for axis in range(3)]).astype(np.float32)
    cluster = pts[:, :3] - mean[inverse]
    center = np.column_stack((
        pts[:, 0] - (geometry["x_min"] + (x + .5) * geometry["x_res"]),
        pts[:, 1] - (geometry["y_min"] + (y + .5) * geometry["y_res"]),
        pts[:, 2] - (geometry["z_min"] + geometry["z_max"]) / 2,
    )).astype(np.float32)
    pts[:, 3] = np.clip(pts[:, 3] * schema.intensity_scale, 0, 1)
    coords = np.column_stack((np.zeros(len(occupied), dtype=np.int64),
                              occupied // width, occupied % width))
    result = {"features": np.concatenate((pts, cluster, center), axis=1),
              "pillar_indices": inverse.astype(np.int64), "coords": coords,
              "batch_size": 1}
    if schema.is_rich_pillar:
        # Match the legacy rasterizer on occupied groups, without allocating
        # a separate dense rich8 BEV or applying intensity_scale twice.
        rich = np.zeros((len(occupied), 8), dtype=np.float32)
        count = counts.astype(np.float32)
        z = np.clip((pts[:, 2] - geometry["z_min"]) /
                    (geometry["z_max"] - geometry["z_min"]), 0, 1).astype(np.float32)
        rich[inverse, np.minimum((z * 3).astype(np.int32), 2)] = 1
        np.maximum.at(rich[:, 3], inverse, z)
        rich[:, 4] = np.bincount(inverse, weights=z, minlength=len(occupied)) / count
        np.maximum.at(rich[:, 5], inverse, pts[:, 3])
        rich[:, 6] = np.bincount(inverse, weights=pts[:, 3], minlength=len(occupied)) / count
        rich[:, 7] = np.minimum(1, np.log1p(count) / np.log1p(schema.density_norm))
        result["rich_features"] = rich
    return result
