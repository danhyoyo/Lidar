import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "detector"),
    str(ROOT / "detector" / "core" / "datasets"),
]

from core.datasets.utils_1.collision_ground import (
    check_box_collision_2d,
    check_ground_support,
    check_static_obstacle_collision,
    estimate_local_ground_z,
    snap_box_to_ground,
)


def test_box_collision_exact_polygons():
    # Box 1 at (10, 0), length=4.5, width=1.8 (covers x in [7.75, 12.25], y in [-0.9, 0.9])
    existing = np.array([[1.5, 1.8, 4.5, 10.0, 0.0, -1.0, 0.0]], dtype=np.float32)
    # Box 2 directly overlapping at (11, 0) -> COLLISION
    colliding = np.array([[1.5, 1.8, 4.5, 11.0, 0.0, -1.0, 0.0]], dtype=np.float32)[0]
    # Box 3 in adjacent lane at y=2.5 (distance between centers is 2.5m, gap between edges is 0.7m > margin 0.3) -> NO COLLISION
    adjacent = np.array([[1.5, 1.8, 4.5, 10.0, 2.5, -1.0, 0.0]], dtype=np.float32)[0]
    # Box 4 far away at (30, 0) -> NO COLLISION
    free_box = np.array([[1.5, 1.8, 4.5, 30.0, 0.0, -1.0, 0.0]], dtype=np.float32)[0]

    assert check_box_collision_2d(colliding, existing, min_margin=0.3) is True
    assert check_box_collision_2d(adjacent, existing, min_margin=0.3) is False
    assert check_box_collision_2d(free_box, existing, min_margin=0.3) is False


def test_box_collision_8col_support():
    existing = np.array(
        [[0.0, 1.5, 1.8, 4.5, 10.0, 0.0, -1.0, 0.0]], dtype=np.float32
    )
    colliding = np.array(
        [[1.0, 1.5, 1.8, 4.5, 11.0, 0.0, -1.0, 0.0]], dtype=np.float32
    )[0]
    free_box = np.array(
        [[1.0, 1.5, 1.8, 4.5, 30.0, 0.0, -1.0, 0.0]], dtype=np.float32
    )[0]
    assert check_box_collision_2d(colliding, existing, min_margin=0.3) is True
    assert check_box_collision_2d(free_box, existing, min_margin=0.3) is False


def test_box_collision_empty_existing():
    existing = np.zeros((0, 7), dtype=np.float32)
    box = np.array([1.5, 1.8, 4.5, 10.0, 0.0, -1.0, 0.0], dtype=np.float32)
    assert check_box_collision_2d(box, existing) is False


def test_ground_plane_snapping_bottom():
    ground_pts = np.zeros((100, 4), dtype=np.float32)
    ground_pts[:, 0] = np.random.uniform(8.0, 12.0, 100)
    ground_pts[:, 1] = np.random.uniform(-2.0, 2.0, 100)
    ground_pts[:, 2] = -1.6 + np.random.normal(0, 0.02, 100)

    z_ground = estimate_local_ground_z(ground_pts, center_x=10.0, center_y=0.0)
    assert np.isclose(z_ground, -1.6, atol=0.05)

    # 7-col box currently floating at z = 0.0
    floating_box_7 = np.array(
        [1.5, 1.8, 4.5, 10.0, 0.0, 0.0, 0.0], dtype=np.float32
    )
    snapped_7 = snap_box_to_ground(floating_box_7, ground_pts)
    assert np.isclose(snapped_7[5], z_ground, atol=1e-4)

    # 8-col box currently floating at z = 0.0
    floating_box_8 = np.array(
        [0.0, 1.5, 1.8, 4.5, 10.0, 0.0, 0.0, 0.0], dtype=np.float32
    )
    snapped_8 = snap_box_to_ground(floating_box_8, ground_pts)
    assert np.isclose(snapped_8[6], z_ground, atol=1e-4)


def test_estimate_local_ground_z_outlier_rejection():
    # Points on overhead structure at z = +1.0
    overhead_pts = np.zeros((50, 4), dtype=np.float32)
    overhead_pts[:, 0] = np.random.uniform(8.0, 12.0, 50)
    overhead_pts[:, 1] = np.random.uniform(-2.0, 2.0, 50)
    overhead_pts[:, 2] = 1.0

    z_fallback = estimate_local_ground_z(overhead_pts, center_x=10.0, center_y=0.0)
    assert z_fallback == -1.6  # Default fallback due to outlier rejection


def test_ground_support_valid_road():
    # Flat ground around (20, 0) with z = -1.6
    xs = np.linspace(18, 22, 10)
    ys = np.linspace(-2, 2, 10)
    xx, yy = np.meshgrid(xs, ys)
    ground_pts = np.column_stack(
        [xx.ravel(), yy.ravel(), np.full(100, -1.6), np.ones(100)]
    ).astype(np.float32)

    cand_box = np.array(
        [1.0, 1.5, 2.0, 4.5, 20.0, 0.0, -1.6, 0.0], dtype=np.float32
    )
    has_support, ground_z = check_ground_support(cand_box, ground_pts)
    assert has_support is True
    assert -1.7 <= ground_z <= -1.5


def test_ground_support_rejects_empty_void():
    empty_pts = np.zeros((0, 4), dtype=np.float32)
    cand_box = np.array(
        [1.0, 1.5, 2.0, 4.5, 35.0, 25.0, -1.6, 0.0], dtype=np.float32
    )
    has_support, _ = check_ground_support(cand_box, empty_pts)
    assert has_support is False


def test_obstacle_collision_rejects_wall():
    # Ground at z=-1.6 plus an elevated vertical wall cluster at (20, 0) with z in [-1.0, 1.5]
    wall_pts = np.array(
        [
            [20.0, 0.0, -0.5, 0.5],
            [20.1, 0.1, 0.0, 0.5],
            [19.9, -0.1, 0.5, 0.5],
            [20.0, 0.2, 1.0, 0.5],
        ],
        dtype=np.float32,
    )

    cand_box = np.array(
        [1.0, 1.5, 2.0, 4.5, 20.0, 0.0, -1.6, 0.0], dtype=np.float32
    )
    collides = check_static_obstacle_collision(cand_box, wall_pts)
    assert collides is True


def test_obstacle_collision_allows_clean_road():
    # Points only on the road surface (z = -1.6, local z <= 0.0)
    road_pts = np.array(
        [
            [20.0, 0.0, -1.6, 0.5],
            [20.1, 0.1, -1.58, 0.5],
            [19.9, -0.1, -1.62, 0.5],
        ],
        dtype=np.float32,
    )

    cand_box = np.array(
        [1.0, 1.5, 2.0, 4.5, 20.0, 0.0, -1.6, 0.0], dtype=np.float32
    )
    collides = check_static_obstacle_collision(cand_box, road_pts)
    assert collides is False



def test_ground_support_neighborhood_does_not_prove_full_footprint():
    # This preserved heuristic accepts support outside a narrow footprint.
    # It documents the limitation for future slope/curb/footprint work.
    box = np.array([2,.5,4,20,0,-1.6,0], np.float32)
    points = np.array([[20,1.5,-1.6,.2]] * 4, np.float32)
    support, height = check_ground_support(box, points)
    assert support and height == pytest.approx(-1.6)
