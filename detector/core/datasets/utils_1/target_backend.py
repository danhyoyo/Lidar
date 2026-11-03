"""Shared Python/Numba regression targets with explicit cell ownership."""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
except ImportError:  # Keep the normal training path free of a Numba dependency.
    njit = None


def _fill_regression_targets(
    boxes, radii, output_x, output_y, x_min_metric, y_min_metric, x_res, y_res,
    out_size_factor, offset_map, size_map, yaw_map, reg_mask, nearest_center,
    vertical_map=None,
):
    # Match the float32 grid arithmetic used by heatmap peak quantization.
    x_min_f32, y_min_f32 = np.float32(x_min_metric), np.float32(y_min_metric)
    x_res_f32, y_res_f32 = np.float32(x_res), np.float32(y_res)
    factor_f32 = np.float32(out_size_factor)
    centers = np.empty((boxes.shape[0], 2), dtype=np.float64)
    center_owner = np.full((output_x, output_y), -1, dtype=np.int32)
    center_distance = np.full((output_x, output_y), np.inf)
    owner_distance = np.full((output_x, output_y), np.inf)

    for index in range(boxes.shape[0]):
        coor_x = float((boxes[index, 4] - x_min_f32) / x_res_f32 / factor_f32)
        coor_y = float((boxes[index, 5] - y_min_f32) / y_res_f32 / factor_f32)
        centers[index, 0], centers[index, 1] = coor_x, coor_y
        if nearest_center and radii[index] > 0:
            p_x, p_y = int(coor_x), int(coor_y)
            if 0 <= p_x < output_x and 0 <= p_y < output_y:
                distance = (p_x - coor_x) ** 2 + (p_y - coor_y) ** 2
                # A shared regression head can retain only one box per cell.
                # Exact ties keep the first box in canonical order.
                if distance < center_distance[p_x, p_y]:
                    center_owner[p_x, p_y] = index
                    center_distance[p_x, p_y] = distance

    for index in range(boxes.shape[0]):
        box = boxes[index]
        radius = radii[index]
        coor_x, coor_y = centers[index]
        width, length, x, y, yaw = box[2], box[3], box[4], box[5], box[7]
        yaw2 = np.fmod(2.0 * np.float64(yaw), 2.0 * math.pi)
        start_x, end_x = max(0, int(coor_x - radius)), min(output_x, int(coor_x + radius))
        start_y, end_y = max(0, int(coor_y - radius)), min(output_y, int(coor_y + radius))
        if nearest_center and radius > 0:
            # Always supervise the quantized peak, including small-radius disks.
            end_x = min(output_x, max(end_x, int(coor_x) + 1))
            end_y = min(output_y, max(end_y, int(coor_y) + 1))

        for p_x in range(start_x, end_x):
            for p_y in range(start_y, end_y):
                distance = (p_x - coor_x) ** 2 + (p_y - coor_y) ** 2
                reserved = nearest_center and center_owner[p_x, p_y] == index
                if math.sqrt(distance) >= radius and not reserved:
                    continue
                if nearest_center:
                    if center_owner[p_x, p_y] >= 0 and not reserved:
                        continue
                    if not reserved and distance >= owner_distance[p_x, p_y]:
                        continue
                    owner_distance[p_x, p_y] = distance

                metric_x = np.float32(p_x * out_size_factor * x_res + x_min_metric)
                metric_y = np.float32(p_y * out_size_factor * y_res + y_min_metric)
                offset_map[p_x, p_y, 0] = x - metric_x
                offset_map[p_x, p_y, 1] = y - metric_y
                size_map[p_x, p_y, 0] = math.log(np.float64(width))
                size_map[p_x, p_y, 1] = math.log(np.float64(length))
                yaw_map[p_x, p_y, 0] = math.cos(yaw2)
                yaw_map[p_x, p_y, 1] = math.sin(yaw2)
                reg_mask[p_x, p_y] = 1.0
                if vertical_map is not None:
                    # Written inside the BEV ownership decision, never reassigned.
                    vertical_map[p_x, p_y, 0] = box[6]
                    vertical_map[p_x, p_y, 1] = math.log(np.float64(box[1]))


_compiled_fill = njit(cache=True)(_fill_regression_targets) if njit is not None else None


def _generate_targets(boxes, radii, output_shape, geometry, out_size_factor, backend, assignment,
                      box_mode="bev"):
    if assignment not in {"nearest_center", "legacy"}:
        raise ValueError("regression_assignment must be 'nearest_center' or 'legacy'")
    if backend == "numba" and _compiled_fill is None:
        raise RuntimeError(
            "target_backend='numba' requires Numba. Install the optional "
            "training dependency with `pip install numba`."
        )

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 8)
    radii = np.asarray(radii, dtype=np.float32)
    if box_mode == "3d":
        if not np.isfinite(boxes).all() or (boxes[:, 1:4] <= 0).any():
            raise ValueError("3D boxes must be finite with positive height, width and length")
        if radii.shape != (len(boxes),) or not np.isfinite(radii).all() or (radii < 0).any():
            raise ValueError("radii must contain one finite nonnegative value per box")
    if assignment == "nearest_center" and boxes.shape[0]:
        order = np.lexsort(tuple(boxes[:, i] for i in reversed(range(8))))
        boxes, radii = np.ascontiguousarray(boxes[order]), radii[order]
    output_x, output_y = output_shape
    offset_map = np.zeros((output_x, output_y, 2), dtype=np.float32)
    size_map = np.zeros((output_x, output_y, 2), dtype=np.float32)
    yaw_map = np.zeros((output_x, output_y, 2), dtype=np.float32)
    reg_mask = np.zeros((output_x, output_y), dtype=np.float32)
    vertical_map = np.zeros((output_x, output_y, 2), dtype=np.float32) if box_mode == "3d" else None
    kernel = _compiled_fill if backend == "numba" else _fill_regression_targets
    kernel(
        boxes,
        radii,
        output_x,
        output_y,
        float(geometry["x_min"]),
        float(geometry["y_min"]),
        float(geometry["x_res"]),
        float(geometry["y_res"]),
        int(out_size_factor),
        offset_map,
        size_map,
        yaw_map,
        reg_mask,
        assignment == "nearest_center",
        vertical_map,
    )
    if box_mode == "3d":
        return offset_map, size_map, yaw_map, reg_mask, vertical_map
    return offset_map, size_map, yaw_map, reg_mask


def fill_regression_targets_python(
    boxes, radii, output_shape, geometry, out_size_factor, assignment="nearest_center",
):
    """Generate dense maps using reserved peaks and nearest-center ownership."""
    return _generate_targets(boxes, radii, output_shape, geometry, out_size_factor, "python", assignment)


def fill_regression_targets_numba(
    boxes, radii, output_shape, geometry, out_size_factor, assignment="nearest_center",
):
    """Compile the same target kernel as Python; ``legacy`` keeps last-box-wins."""
    return _generate_targets(boxes, radii, output_shape, geometry, out_size_factor, "numba", assignment)


def fill_regression_targets_3d(
    boxes, radii, output_shape, geometry, out_size_factor, *, backend="python",
    assignment="nearest_center",
):
    """Return the four legacy BEV maps followed by bottom-z/log-height.

    Python and Numba execute the same ownership kernel; the legacy public APIs
    continue returning exactly four arrays.
    """
    if backend not in {"python", "numba"}:
        raise ValueError("backend must be python or numba")
    return _generate_targets(boxes, radii, output_shape, geometry, out_size_factor,
                             backend, assignment, box_mode="3d")
