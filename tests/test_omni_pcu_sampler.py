"""Tests for Omni-PCU SOTA Sampler."""

from __future__ import annotations
import numpy as np
import pytest
from shapely.geometry import Polygon

from core.datasets.augmentor.omni_geometry import (
    boxes_to_bev_corners,
    check_collision_2d_vectorized,
    project_to_road_plane,
    points_in_oriented_box_3d,
)


def test_bev_corners_orientation_and_dimensions():
    # Box: [cls, h, w, l, x, y, z, yaw=0]
    box = np.array([[0, 2.0, 2.0, 4.0, 10.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    corners = boxes_to_bev_corners(box)[0]  # (4, 2)
    assert corners.shape == (4, 2)
    # Check width along Y and length along X for yaw=0
    x_span = corners[:, 0].max() - corners[:, 0].min()
    y_span = corners[:, 1].max() - corners[:, 1].min()
    assert np.isclose(x_span, 4.0, atol=1e-4)
    assert np.isclose(y_span, 2.0, atol=1e-4)


def test_sat_collision_matches_shapely():
    box1 = np.array([[0, 1.5, 1.8, 4.2, 10.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    # Overlapping box
    box2 = np.array([[0, 1.5, 1.8, 4.2, 11.0, 0.5, 0.0, np.pi / 4]], dtype=np.float32)
    # Non-overlapping box
    box3 = np.array([[0, 1.5, 1.8, 4.2, 25.0, 15.0, 0.0, 0.0]], dtype=np.float32)

    poly1 = Polygon(boxes_to_bev_corners(box1)[0])
    poly2 = Polygon(boxes_to_bev_corners(box2)[0])
    poly3 = Polygon(boxes_to_bev_corners(box3)[0])

    assert poly1.intersects(poly2)
    assert not poly1.intersects(poly3)

    coll_overlap = check_collision_2d_vectorized(box2, box1, min_margin=0.0)
    assert coll_overlap[0] == True

    coll_disjoint = check_collision_2d_vectorized(box3, box1, min_margin=0.0)
    assert coll_disjoint[0] == False


def test_road_plane_analytical_projection():
    # Plane: 0*x + (-0.05)*y + (-0.998)*z + (-1.65) = 0
    plane = np.array([0.0, -0.05, -0.998, -1.65], dtype=np.float64)
    x = np.array([10.0, 20.0, 30.0], dtype=np.float64)
    y = np.array([0.0, 2.0, -2.0], dtype=np.float64)
    z_ground = project_to_road_plane(x, y, plane)
    assert len(z_ground) == 3
    # Check equation: a*x + b*y + c*z + d == 0
    residuals = plane[0] * x + plane[1] * y + plane[2] * z_ground + plane[3]
    assert np.allclose(residuals, 0.0, atol=1e-6)


def test_bounded_point_cache_eviction(tmp_path):
    import os
    from core.datasets.augmentor.omni_cache import BoundedPointCache
    # Cache limit 1KB (tiny)
    cache = BoundedPointCache(max_size_mb=0.001)
    pts1 = np.ones((100, 4), dtype=np.float32)  # 1600 bytes -> exceeds 1KB

    cache.put("sample_1", pts1)
    assert cache.get("sample_1") is not None

    pts2 = np.zeros((100, 4), dtype=np.float32)
    cache.put("sample_2", pts2)
    # sample_1 should be evicted or cache size bounded
    assert cache.get("sample_2") is not None
    assert cache.current_bytes <= 1600


def test_cache_resets_on_pid_change():
    import os
    from core.datasets.augmentor.omni_cache import BoundedPointCache
    cache = BoundedPointCache(max_size_mb=10)
    cache.put("key1", np.ones((10, 4), dtype=np.float32))
    assert cache.get("key1") is not None

    # Simulate process fork by altering stored pid
    cache.pid = os.getpid() + 999
    assert cache.get("key1") is None
    assert len(cache) == 0

