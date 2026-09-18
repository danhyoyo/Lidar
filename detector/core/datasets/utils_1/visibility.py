"""Sample measured LiDAR rays to mark locally observed free BEV cells.

Only the segment immediately before each return is needed: the Gaussian
propagation kernel has compact support around measured cells. A sampled cell
is free at the ray's interpolated height, and cells without a sampled ray
remain unknown.
"""

import numpy as np


def free_space_maps(points, geometry, *, height_ranges, ray_length_m=0.7,
                    step_m=0.05, range_margin_m=0.1,
                    sensor_origin=(0.0, 0.0, 0.0)):
    """Return HxWxK masks, with 1 at cells crossed by a free ray segment.

    Ray samples stop short of the measured return by range_margin_m. The
    caller clears cells containing a return after constructing the masks.
    """
    ranges = tuple(tuple(map(float, pair)) for pair in height_ranges)
    if not ranges or any(len(pair) != 2 or not np.isfinite(pair).all()
                         or pair[0] >= pair[1] for pair in ranges):
        raise ValueError("height_ranges must contain finite increasing pairs")
    if not (np.isfinite([ray_length_m, step_m, range_margin_m]).all()
            and ray_length_m > range_margin_m > 0 and step_m > 0):
        raise ValueError("require ray_length_m > range_margin_m > 0 and step_m > 0")
    origin = np.asarray(sensor_origin, dtype=np.float32)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("sensor_origin must contain three finite coordinates")
    x_size = round((geometry["x_max"] - geometry["x_min"]) / geometry["x_res"])
    y_size = round((geometry["y_max"] - geometry["y_min"]) / geometry["y_res"])
    result = np.zeros((y_size, x_size, len(ranges)), dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError("points must have shape (N, >=3)")
    xyz = np.asarray(points[np.isfinite(points[:, :3]).all(axis=1), :3],
                     dtype=np.float32)
    if not len(xyz):
        return result

    offsets = np.arange(range_margin_m, ray_length_m + step_m * 0.5,
                        step_m, dtype=np.float32)
    for start in range(0, len(xyz), 32768):
        returns = xyz[start:start + 32768]
        rays = returns - origin
        lengths = np.linalg.norm(rays, axis=1)
        keep = lengths > range_margin_m
        if not np.any(keep):
            continue
        returns, rays, lengths = returns[keep], rays[keep], lengths[keep]
        unit = rays / lengths[:, None]
        samples = returns[:, None, :] - offsets[None, :, None] * unit[:, None, :]
        valid = offsets[None, :] < lengths[:, None]
        valid &= (samples[..., 0] > geometry["x_min"])
        valid &= (samples[..., 0] < geometry["x_max"])
        valid &= (samples[..., 1] > geometry["y_min"])
        valid &= (samples[..., 1] < geometry["y_max"])
        x_index = np.floor((samples[..., 0] - geometry["x_min"])
                           / geometry["x_res"]).astype(np.int32)
        y_index = np.floor((samples[..., 1] - geometry["y_min"])
                           / geometry["y_res"]).astype(np.int32)
        for band, (low, high) in enumerate(ranges):
            selected = valid & (samples[..., 2] >= low) & (samples[..., 2] < high)
            if np.any(selected):
                result[y_index[selected], x_index[selected], band] = 1.0
    return result
