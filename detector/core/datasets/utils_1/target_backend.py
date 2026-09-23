"""Optional CPU backends for generating detector regression targets.

The default dataset path remains pure Python/Torch.  Numba is intentionally
optional because target generation runs inside DataLoader workers and must not
change the model or target representation.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
except ImportError:  # Keep the normal training path free of a Numba dependency.
    njit = None


if njit is not None:

    @njit(cache=True)
    def _fill_regression_targets(
        boxes,
        radii,
        output_x,
        output_y,
        x_min_metric,
        y_min_metric,
        x_res,
        y_res,
        out_size_factor,
        offset_map,
        size_map,
        yaw_map,
        reg_mask,
    ):
        for index in range(boxes.shape[0]):
            box = boxes[index]
            radius = radii[index]
            width = box[2]
            length = box[3]
            x = box[4]
            y = box[5]
            yaw = box[7]
            # Numba supports NumPy's fmod ufunc in nopython mode, unlike
            # math.fmod in the Colab Python 3.13 runtime.  fmod preserves the
            # legacy sign convention for negative yaw values.
            yaw2 = np.fmod(2.0 * yaw, 2.0 * math.pi)

            coor_x = (x - x_min_metric) / x_res / out_size_factor
            coor_y = (y - y_min_metric) / y_res / out_size_factor

            start_x = max(0, int(coor_x - radius))
            end_x = min(output_x, int(coor_x + radius))
            start_y = max(0, int(coor_y - radius))
            end_y = min(output_y, int(coor_y + radius))

            for p_x in range(start_x, end_x):
                for p_y in range(start_y, end_y):
                    if math.sqrt(
                        (p_x - coor_x) ** 2 + (p_y - coor_y) ** 2
                    ) < radius:
                        metric_x = p_x * out_size_factor * x_res + x_min_metric
                        metric_y = p_y * out_size_factor * y_res + y_min_metric

                        offset_map[p_x, p_y, 0] = x - metric_x
                        offset_map[p_x, p_y, 1] = y - metric_y
                        size_map[p_x, p_y, 0] = math.log(width)
                        size_map[p_x, p_y, 1] = math.log(length)
                        yaw_map[p_x, p_y, 0] = math.cos(yaw2)
                        yaw_map[p_x, p_y, 1] = math.sin(yaw2)
                        reg_mask[p_x, p_y] = 1.0


def fill_regression_targets_numba(boxes, radii, output_shape, geometry, out_size_factor):
    """Return maps with the same layout and assignment order as ``update_reg_map``."""
    if njit is None:
        raise RuntimeError(
            "target_backend='numba' requires Numba. Install the optional "
            "training dependency with `pip install numba`."
        )

    output_x, output_y = output_shape
    offset_map = np.zeros((output_x, output_y, 2), dtype=np.float32)
    size_map = np.zeros((output_x, output_y, 2), dtype=np.float32)
    yaw_map = np.zeros((output_x, output_y, 2), dtype=np.float32)
    reg_mask = np.zeros((output_x, output_y), dtype=np.float32)
    _fill_regression_targets(
        boxes,
        np.asarray(radii, dtype=np.float32),
        output_x,
        output_y,
        # ``boxes`` is float32, as are the original Torch coordinate
        # calculations.  Preserve that arithmetic precision for grid bounds.
        np.float32(geometry["x_min"]),
        np.float32(geometry["y_min"]),
        np.float32(geometry["x_res"]),
        np.float32(geometry["y_res"]),
        np.float32(out_size_factor),
        offset_map,
        size_map,
        yaw_map,
        reg_mask,
    )
    return offset_map, size_map, yaw_map, reg_mask
