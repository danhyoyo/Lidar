"""hist14 v1 rasterization with a NumPy reference and explicit backend choice."""

from __future__ import annotations

import math

import numpy as np

HIST14_IMPLEMENTATION_VERSION = 2
_numba_kernel = None


def encode_hist14(points, schema):
    """Select the explicitly requested backend; never silently substitute it."""
    if schema.backend == "numpy":
        return encode_hist14_numpy(points, schema)
    if schema.backend == "numba":
        return encode_hist14_numba(points, schema)
    raise ValueError(f"unsupported hist14 backend: {schema.backend!r}")


def encode_hist14_chw(points, schema):
    """Return independently owned contiguous [14,Y,X] model input.

    The public rasterizer keeps its HWC contract. Datasets use this entry point
    to avoid materializing HWC and copying it back to CHW before device transfer.
    """
    if schema.backend == "numpy":
        return _encode_hist14_numpy(points, schema, channels_first=True)
    if schema.backend == "numba":
        return _encode_hist14_numba(points, schema, channels_first=True)
    raise ValueError(f"unsupported hist14 backend: {schema.backend!r}")


def _validated_points(points, schema, *, promote=True):
    if schema.name != "hist14" or schema.version != 1 or schema.channels != 14:
        raise ValueError("hist14 rasterization requires a hist14 v1 schema")
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] < 4 or points.dtype.kind not in "fiu":
        raise ValueError("hist14 expects real numeric points shaped (N, >=4)")
    # The compiled kernel promotes each scalar before arithmetic, avoiding a
    # full-cloud conversion/copy for the common native float32/float64 inputs.
    if not promote and points.dtype in (np.dtype("float32"), np.dtype("float64")):
        return points[:, :4]
    return np.ascontiguousarray(points[:, :4], dtype=np.float64)


def encode_hist14_numpy(points, schema):
    """Return contiguous float32 [Y,X,14] without modifying the source cloud."""
    return _encode_hist14_numpy(points, schema)


def _encode_hist14_numpy(points, schema, *, channels_first=False):
    points = _validated_points(points, schema)
    geometry = dict(schema.geometry)
    x_size, y_size, _ = schema.grid_shape
    cell_count = x_size * y_size
    output = np.zeros((14, cell_count) if channels_first else (cell_count, 14), dtype=np.float32)
    shape = (14, y_size, x_size) if channels_first else (y_size, x_size, 14)

    keep = np.isfinite(points).all(axis=1)
    for column, axis in enumerate("xyz"):
        keep &= points[:, column] > geometry[f"{axis}_min"] + 0.001
        keep &= points[:, column] < geometry[f"{axis}_max"] - 0.001
    points = points[keep]
    if len(points) == 0:
        return output.reshape(shape)

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

    # Group through a grid lookup instead of sorting points. Statistics and
    # normalization touch occupied cells only; empty output stays exactly zero.
    dense_count = np.bincount(flat, minlength=cell_count)
    occupied = np.flatnonzero(dense_count)
    count = dense_count[occupied]
    occupied_count = len(occupied)
    lookup = np.empty(cell_count, dtype=np.int64)
    lookup[occupied] = np.arange(occupied_count)
    inverse = lookup[flat]
    features = np.zeros((occupied_count, 14), dtype=np.float32)
    bin_counts = np.bincount(inverse * 4 + band, minlength=occupied_count * 4).reshape(occupied_count, 4)
    density_log = np.log1p(schema.density_norm)
    features[:, :4] = np.minimum(1, np.log1p(bin_counts) / density_log)

    height_max = np.zeros(occupied_count, dtype=np.float64)
    height_min = np.ones(occupied_count, dtype=np.float64)
    intensity_max = np.zeros(occupied_count, dtype=np.float64)
    np.maximum.at(height_max, inverse, height)
    np.minimum.at(height_min, inverse, height)
    np.maximum.at(intensity_max, inverse, intensity)
    mean_height = np.bincount(inverse, weights=height, minlength=occupied_count) / count
    mean_height_squared = np.bincount(inverse, weights=height * height, minlength=occupied_count) / count
    variance = np.maximum(0, mean_height_squared - mean_height * mean_height)
    features[:, 4] = height_max
    features[:, 5] = mean_height
    features[:, 6] = height_max - height_min
    features[:, 7] = np.where(count > 1, np.sqrt(variance), 0)
    features[:, 8] = intensity_max
    features[:, 9] = np.bincount(inverse, weights=intensity, minlength=occupied_count) / count
    features[:, 10] = np.minimum(1, np.log1p(count) / density_log)
    features[:, 11] = 1

    # Summing point-to-cell residuals avoids subtracting large mean coordinates.
    for column, axis, index in ((0, "x", ix), (1, "y", iy)):
        center = geometry[f"{axis}_min"] + (index + 0.5) * geometry[f"{axis}_res"]
        residual = (points[:, column] - center) / geometry[f"{axis}_res"]
        mean_residual = np.bincount(inverse, weights=residual, minlength=occupied_count) / count
        features[:, 12 + column] = np.clip(mean_residual, -0.5, 0.5)
    if channels_first:
        output[:, occupied] = features.T
    else:
        output[occupied] = features
    return output.reshape(shape)


def _accumulate_and_normalize(
    points, x_size, y_size, x_min, x_max, x_res, y_min, y_max, y_res,
    z_min, z_max, density_norm, intensity_scale, channels_first=False,
):
    """One point pass and one occupied-cell pass, with float64 arithmetic.

    A zero-initialized grid maps each cell to a compact accumulator row plus one.
    Empty rows are never touched. Workspace and output are local to each call.
    """
    cell_count = x_size * y_size
    capacity = min(points.shape[0], cell_count)
    lookup = np.zeros(cell_count, dtype=np.int32)
    cells = np.empty(capacity, dtype=np.int64)
    # No accumulator for the constant occupancy feature; offsets use cols 11/12.
    accum = np.empty((capacity, 13), dtype=np.float64)
    occupied_count = 0
    output = np.zeros(cell_count * 14, dtype=np.float32)
    z_span = z_max - z_min
    density_log = math.log1p(density_norm)
    # Integer counts recur across cells/bins. Evaluate log1p once per small count
    # without changing the original double-precision expression or its rounding.
    table_size = min(points.shape[0] + 1, 256)
    density_table = np.empty(table_size, dtype=np.float64)
    for count in range(table_size):
        density_table[count] = min(1.0, math.log1p(count) / density_log)
    for index in range(points.shape[0]):
        x = np.float64(points[index, 0])
        y = np.float64(points[index, 1])
        z = np.float64(points[index, 2])
        intensity = np.float64(points[index, 3])
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
        row = lookup[flat] - 1
        if row < 0:
            row = occupied_count
            occupied_count += 1
            lookup[flat] = row + 1
            cells[row] = flat
            for column in range(13):
                accum[row, column] = 0.0
        height = (z - z_min) / z_span
        band = min(int(math.floor(4 * height)), 3)
        value = min(1.0, max(0.0, intensity * intensity_scale))
        if accum[row, 10] == 0:
            accum[row, 6] = height
        accum[row, band] += 1
        accum[row, 4] = max(accum[row, 4], height)
        accum[row, 5] += height
        accum[row, 6] = min(accum[row, 6], height)
        accum[row, 7] += height * height
        accum[row, 8] = max(accum[row, 8], value)
        accum[row, 9] += value
        accum[row, 10] += 1
        accum[row, 11] += (x - (x_min + (ix + .5) * x_res)) / x_res
        accum[row, 12] += (y - (y_min + (iy + .5) * y_res)) / y_res

    step = cell_count if channels_first else 1
    for row in range(occupied_count):
        flat = cells[row]
        base = flat if channels_first else flat * 14
        count = accum[row, 10]
        for band in range(4):
            bin_count = int(accum[row, band])
            output[base + band * step] = (density_table[bin_count] if bin_count < table_size
                                         else min(1.0, math.log1p(bin_count) / density_log))
        mean_height = accum[row, 5] / count
        variance = max(0.0, accum[row, 7] / count - mean_height * mean_height)
        output[base + 4 * step] = accum[row, 4]
        output[base + 5 * step] = mean_height
        output[base + 6 * step] = accum[row, 4] - accum[row, 6]
        output[base + 7 * step] = math.sqrt(variance) if count > 1 else 0.0
        output[base + 8 * step] = accum[row, 8]
        output[base + 9 * step] = accum[row, 9] / count
        total_count = int(count)
        output[base + 10 * step] = (density_table[total_count] if total_count < table_size
                                   else min(1.0, math.log1p(total_count) / density_log))
        output[base + 11 * step] = 1
        output[base + 12 * step] = min(.5, max(-.5, accum[row, 11] / count))
        output[base + 13 * step] = min(.5, max(-.5, accum[row, 12] / count))
    if channels_first:
        return output.reshape(14, y_size, x_size)
    return output.reshape(y_size, x_size, 14)


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
        _numba_kernel = njit(cache=True, fastmath=False, nogil=True)(_accumulate_and_normalize)
    return _numba_kernel


def encode_hist14_numba(points, schema):
    """Run the compiled kernel; explicit requests also check dependency on empty input."""
    return _encode_hist14_numba(points, schema)


def _encode_hist14_numba(points, schema, *, channels_first=False):
    points = _validated_points(points, schema, promote=False)
    kernel = _get_numba_kernel()
    geometry = dict(schema.geometry)
    return kernel(
        points, schema.grid_shape[0], schema.grid_shape[1],
        geometry["x_min"], geometry["x_max"], geometry["x_res"],
        geometry["y_min"], geometry["y_max"], geometry["y_res"],
        geometry["z_min"], geometry["z_max"], schema.density_norm, schema.intensity_scale,
        channels_first,
    )
