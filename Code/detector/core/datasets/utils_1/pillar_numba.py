"""Optional serial fused pillar statistics/features; strict reduction arithmetic."""

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def filter_points(points, bounds):
    """Fuse finite/ROI filtering and float32 copy, preserving input order."""
    output = np.empty((len(points), 4), dtype=np.float32)
    retained = 0
    for index in range(len(points)):
        valid = True
        for column in range(4):
            if not np.isfinite(points[index, column]):
                valid = False
        for column in range(3):
            if not (points[index, column] > bounds[column, 0] and points[index, column] < bounds[column, 1]):
                valid = False
        if valid:
            for column in range(4):
                output[retained, column] = points[index, column]
            retained += 1
    return output[:retained]


@njit(cache=True, fastmath=False)
def build_pillar_features(points, x, y, inverse, counts, z, center_z,
                          x_min, x_res, y_min, y_res, has_rich):
    pillars = len(counts)
    xyz_sum = np.zeros((pillars, 3), dtype=np.float64)
    rich_sum = np.zeros((pillars if has_rich else 0, 2), dtype=np.float64)
    rich = np.zeros((pillars if has_rich else 0, 8), dtype=np.float32)
    # Accumulate each pillar in original point order, matching bincount's
    # float64 weighted sums. Neither fastmath nor parallel reassociation is used.
    for index in range(len(points)):
        pillar = inverse[index]
        for column in range(3):
            xyz_sum[pillar, column] += np.float64(points[index, column])
        if has_rich:
            height, intensity = z[index], points[index, 3]
            band = min(int(np.float32(height * np.float32(3))), 2)
            rich[pillar, band] = 1
            # np.maximum.at retains the incoming value on equal values too,
            # which matters for the sign of zero after intensity clipping.
            if height >= rich[pillar, 3]:
                rich[pillar, 3] = height
            if intensity >= rich[pillar, 5]:
                rich[pillar, 5] = intensity
            rich_sum[pillar, 0] += np.float64(height)
            rich_sum[pillar, 1] += np.float64(intensity)
    means = np.empty((pillars, 3), dtype=np.float32)
    for pillar in range(pillars):
        for column in range(3):
            means[pillar, column] = np.float32(xyz_sum[pillar, column] / counts[pillar])
        if has_rich:
            divisor = np.float64(np.float32(counts[pillar]))
            rich[pillar, 4] = np.float32(rich_sum[pillar, 0] / divisor)
            rich[pillar, 6] = np.float32(rich_sum[pillar, 1] / divisor)
    features = np.empty((len(points), 10), dtype=np.float32)
    for index in range(len(points)):
        pillar = inverse[index]
        for column in range(4):
            features[index, column] = points[index, column]
        for column in range(3):
            features[index, 4 + column] = points[index, column] - means[pillar, column]
        # XY centers used float64 arithmetic in the NumPy reference; z center
        # stays precomputed with the reference's scalar-promotion rules.
        features[index, 7] = np.float32(np.float64(points[index, 0]) - (x_min + (x[index] + .5) * x_res))
        features[index, 8] = np.float32(np.float64(points[index, 1]) - (y_min + (y[index] + .5) * y_res))
        features[index, 9] = center_z[index]
    return features, rich
