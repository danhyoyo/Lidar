"""hist14 v1 rasterization with a NumPy reference and explicit backend choice."""

from __future__ import annotations

import math

import numpy as np

HIST14_IMPLEMENTATION_VERSION = 1
_numba_kernel = None


def encode_hist14(points, schema):
    """Select the explicitly requested backend; never silently substitute it."""
    if schema.backend == "numpy":
        return encode_hist14_numpy(points, schema)
    if schema.backend == "numba":
        return encode_hist14_numba(points, schema)
    raise ValueError(f"unsupported hist14 backend: {schema.backend!r}")


def _validated_points(points, schema):
    if schema.name != "hist14" or schema.version != 1 or schema.channels != 14:
        raise ValueError("hist14 rasterization requires a hist14 v1 schema")
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 4 or points.dtype.kind not in "fiu":
        raise ValueError("hist14 expects real numeric points shaped (N, >=4)")
    # Ignore additional features; both kernels start from identical float64 data.
    return np.ascontiguousarray(points[:, :4], dtype=np.float64)


def encode_hist14_numpy(points, schema):
    """Return contiguous float32 [Y,X,14] without modifying the source cloud."""
    points = _validated_points(points, schema)
    geometry = dict(schema.geometry)
    x_size, y_size, _ = schema.grid_shape
    cell_count = x_size * y_size
    output = np.zeros((cell_count, 14), dtype=np.float32)

    keep = np.isfinite(points).all(axis=1)
    for column, axis in enumerate("xyz"):
        keep &= points[:, column] > geometry[f"{axis}_min"] + 0.001
        keep &= points[:, column] < geometry[f"{axis}_max"] - 0.001
    points = points[keep]
    if len(points) == 0:
        return output.reshape(y_size, x_size, 14)

    ix = np.floor((points[:, 0] - geometry["x_min"]) / geometry["x_res"]).astype(np.int64)
    iy = np.floor((points[:, 1] - geometry["y_min"]) / geometry["y_res"]).astype(np.int64)
    if np.any((ix < 0) | (ix >= x_size) | (iy < 0) | (iy >= y_size)):
        raise ValueError("hist14 point lies outside the resolved grid; check geometry divisibility")
    flat = iy * x_size + ix
    height = (points[:, 2] - geometry["z_min"]) / (geometry["z_max"] - geometry["z_min"])
    band = np.minimum(np.floor(4 * height).astype(np.int64), 3)
    # Finite inputs can overflow during scaling; clipping defines saturation.
    with np.errstate(over="ignore"):
        intensity = np.clip(points[:, 3] * schema.intensity_scale, 0, 1)

    count = np.bincount(flat, minlength=cell_count)
    occupied = count > 0
    divisor = np.maximum(count, 1)
    bin_counts = np.bincount(flat * 4 + band, minlength=cell_count * 4).reshape(cell_count, 4)
    density_log = np.log1p(schema.density_norm)
    output[:, :4] = np.minimum(1, np.log1p(bin_counts) / density_log)

    height_max = np.zeros(cell_count, dtype=np.float64)
    height_min = np.ones(cell_count, dtype=np.float64)
    intensity_max = np.zeros(cell_count, dtype=np.float64)
    np.maximum.at(height_max, flat, height)
    np.minimum.at(height_min, flat, height)
    np.maximum.at(intensity_max, flat, intensity)
    mean_height = np.bincount(flat, weights=height, minlength=cell_count) / divisor
    mean_height_squared = np.bincount(flat, weights=height * height, minlength=cell_count) / divisor
    variance = np.maximum(0, mean_height_squared - mean_height * mean_height)
    output[:, 4] = height_max
    output[:, 5] = mean_height
    output[occupied, 6] = height_max[occupied] - height_min[occupied]
    output[:, 7] = np.where(count > 1, np.sqrt(variance), 0)
    output[:, 8] = intensity_max
    output[:, 9] = np.bincount(flat, weights=intensity, minlength=cell_count) / divisor
    output[:, 10] = np.minimum(1, np.log1p(count) / density_log)
    output[:, 11] = occupied

    # Summing point-to-cell residuals avoids subtracting large mean coordinates.
    for column, axis, index in ((0, "x", ix), (1, "y", iy)):
        center = geometry[f"{axis}_min"] + (index + 0.5) * geometry[f"{axis}_res"]
        residual = (points[:, column] - center) / geometry[f"{axis}_res"]
        mean_residual = np.bincount(flat, weights=residual, minlength=cell_count) / divisor
        output[:, 12 + column] = np.clip(mean_residual, -0.5, 0.5)
    return output.reshape(y_size, x_size, 14)


def _accumulate_and_normalize(
    points, x_size, y_size, x_min, x_max, x_res, y_min, y_max, y_res,
    z_min, z_max, density_norm, intensity_scale,
):
    """One point pass and one cell pass; all accumulator arithmetic is float64."""
    cell_count = x_size * y_size
    # Columns follow output features, with sums/minimum replacing mean/span.
    accum = np.zeros((cell_count, 14), dtype=np.float64)
    output = np.zeros((y_size, x_size, 14), dtype=np.float32)
    z_span = z_max - z_min
    density_log = math.log1p(density_norm)
    for index in range(points.shape[0]):
        x, y, z, intensity = points[index, 0], points[index, 1], points[index, 2], points[index, 3]
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z) and math.isfinite(intensity)):
            continue
        if not (x > x_min + .001 and x < x_max - .001 and
                y > y_min + .001 and y < y_max - .001 and
                z > z_min + .001 and z < z_max - .001):
            continue
        ix = int(math.floor((x - x_min) / x_res))
        iy = int(math.floor((y - y_min) / y_res))
        if ix < 0 or ix >= x_size or iy < 0 or iy >= y_size:
            raise ValueError("hist14 point lies outside the resolved grid; check geometry divisibility")
        flat = iy * x_size + ix
        height = (z - z_min) / z_span
        band = min(int(math.floor(4 * height)), 3)
        value = min(1.0, max(0.0, intensity * intensity_scale))
        if accum[flat, 10] == 0:
            accum[flat, 6] = height
        accum[flat, band] += 1
        accum[flat, 4] = max(accum[flat, 4], height)
        accum[flat, 5] += height
        accum[flat, 6] = min(accum[flat, 6], height)
        accum[flat, 7] += height * height
        accum[flat, 8] = max(accum[flat, 8], value)
        accum[flat, 9] += value
        accum[flat, 10] += 1
        accum[flat, 12] += (x - (x_min + (ix + .5) * x_res)) / x_res
        accum[flat, 13] += (y - (y_min + (iy + .5) * y_res)) / y_res

    for flat in range(cell_count):
        count = accum[flat, 10]
        if count == 0:
            continue
        iy, ix = flat // x_size, flat % x_size
        for band in range(4):
            output[iy, ix, band] = min(1.0, math.log1p(accum[flat, band]) / density_log)
        mean_height = accum[flat, 5] / count
        variance = max(0.0, accum[flat, 7] / count - mean_height * mean_height)
        output[iy, ix, 4] = accum[flat, 4]
        output[iy, ix, 5] = mean_height
        output[iy, ix, 6] = accum[flat, 4] - accum[flat, 6]
        output[iy, ix, 7] = math.sqrt(variance) if count > 1 else 0.0
        output[iy, ix, 8] = accum[flat, 8]
        output[iy, ix, 9] = accum[flat, 9] / count
        output[iy, ix, 10] = min(1.0, math.log1p(count) / density_log)
        output[iy, ix, 11] = 1
        output[iy, ix, 12] = min(.5, max(-.5, accum[flat, 12] / count))
        output[iy, ix, 13] = min(.5, max(-.5, accum[flat, 13] / count))
    return output


def _get_numba_kernel():
    global _numba_kernel
    if _numba_kernel is None:
        try:
            from numba import njit
        except ImportError as exc:
            raise ImportError(
                "hist14 backend=numba requires Numba. Install a compatible Numba "
                "version or select backend=numpy."
            ) from exc
        _numba_kernel = njit(cache=True, fastmath=False)(_accumulate_and_normalize)
    return _numba_kernel


def encode_hist14_numba(points, schema):
    """Run the compiled kernel; explicit requests also check dependency on empty input."""
    points = _validated_points(points, schema)
    kernel = _get_numba_kernel()
    geometry = dict(schema.geometry)
    return kernel(
        points, schema.grid_shape[0], schema.grid_shape[1],
        geometry["x_min"], geometry["x_max"], geometry["x_res"],
        geometry["y_min"], geometry["y_max"], geometry["y_res"],
        geometry["z_min"], geometry["z_max"], schema.density_norm, schema.intensity_scale,
    )
