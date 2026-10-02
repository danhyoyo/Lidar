import pytest
import torch
from detector.core.models.encoders.pillar_ops import group_and_sort_pillars

def test_group_and_sort_pillars_correctness():
    geometry = {
        "x_min": 0.0, "x_max": 70.4, "x_res": 0.1,
        "y_min": -40.0, "y_max": 40.0, "y_res": 0.1,
        "z_min": -2.5, "z_max": 1.0, "z_res": 0.1,
    }
    # Create points in two pillars:
    # Pillar A at (1.05, -38.05) has 3 points with unsorted z: 0.5, -1.2, -0.1
    # Pillar B at (3.05, -36.05) has 2 points: 0.2, -0.8
    points = torch.tensor([
        [1.05, -38.05, 0.5, 0.8],    # z = 0.5
        [1.05, -38.05, -1.2, 0.4],   # z = -1.2
        [1.05, -38.05, -0.1, 0.6],   # z = -0.1
        [3.05, -36.05, 0.2, 0.9],
        [3.05, -36.05, -0.8, 0.3],
    ], dtype=torch.float32)

    features, indices, num_pillars = group_and_sort_pillars(
        points, geometry, max_points_per_pillar=20, max_pillars=32000
    )
    assert num_pillars == 2
    assert features.shape == (2, 20, 8)
    assert indices.shape == (2, 2)

    # Check Z-sorting in both pillars:
    # First pillar has 3 points: z values at channel 5 should be strictly ascending
    p0_count = (features[0, :, 5] != 0).sum()
    assert p0_count == 3
    assert features[0, 0, 5] < features[0, 1, 5] < features[0, 2, 5]

    # Second pillar has 2 points: z values at channel 5 should be ascending
    p1_count = (features[1, :, 5] != 0).sum()
    assert p1_count == 2
    assert features[1, 0, 5] < features[1, 1, 5]

def test_group_and_sort_pillars_empty_and_oob():
    geometry = {
        "x_min": 0.0, "x_max": 70.4, "x_res": 0.1,
        "y_min": -40.0, "y_max": 40.0, "y_res": 0.1,
        "z_min": -2.5, "z_max": 1.0, "z_res": 0.1,
    }
    # Completely empty point cloud
    features, indices, num_pillars = group_and_sort_pillars(
        torch.zeros((0, 4)), geometry
    )
    assert num_pillars == 0
    assert features.shape[0] == 0

    # Points outside ROI
    oob_points = torch.tensor([[-10.0, -50.0, 10.0, 1.0]], dtype=torch.float32)
    features, indices, num_pillars = group_and_sort_pillars(
        oob_points, geometry
    )
    assert num_pillars == 0
