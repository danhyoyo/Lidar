import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "detector"),
    str(ROOT / "detector" / "core" / "datasets"),
]

from core.datasets.utils_1.physics_aug import (
    random_flip_3d,
    distance_adaptive_subsample,
    radiometric_intensity_calibrate,
    mask_shadow_points,
    check_line_of_sight_occlusion,
)


def test_random_flip_3d_7col_and_8col():
    # 7-col: [h, w, l, x, y, z, yaw]
    boxes_7 = np.array([[1.5, 1.8, 4.5, 10.0, 5.0, -1.0, 0.5]], dtype=np.float32)
    # 8-col: [cls, h, w, l, x, y, z, yaw]
    boxes_8 = np.array([[0.0, 1.5, 1.8, 4.5, 10.0, 5.0, -1.0, 0.5]], dtype=np.float32)
    points = np.array(
        [[10.0, 5.0, -0.5, 0.8], [10.0, 4.0, -0.5, 0.5]], dtype=np.float32
    )

    # Force flip
    p_flipped, b7_flipped = random_flip_3d(points.copy(), boxes_7.copy(), p=1.0)
    assert np.allclose(p_flipped[:, 1], -points[:, 1])
    assert np.allclose(b7_flipped[:, 4], -boxes_7[:, 4])  # y flipped
    assert np.allclose(b7_flipped[:, 6], -boxes_7[:, 6])  # yaw flipped
    assert np.allclose(b7_flipped[:, 3], boxes_7[:, 3])  # x untouched

    _, b8_flipped = random_flip_3d(points.copy(), boxes_8.copy(), p=1.0)
    assert np.allclose(b8_flipped[:, 5], -boxes_8[:, 5])  # y flipped
    assert np.allclose(b8_flipped[:, 7], -boxes_8[:, 7])  # yaw flipped
    assert np.allclose(b8_flipped[:, 0], boxes_8[:, 0])  # cls untouched
    assert np.allclose(b8_flipped[:, 4], boxes_8[:, 4])  # x untouched


def test_random_flip_3d_probability_zero():
    boxes = np.array([[1.5, 1.8, 4.5, 10.0, 5.0, -1.0, 0.5]], dtype=np.float32)
    points = np.array([[10.0, 5.0, -0.5, 0.8]], dtype=np.float32)
    p_unflipped, b_unflipped = random_flip_3d(points.copy(), boxes.copy(), p=0.0)
    assert np.allclose(p_unflipped, points)
    assert np.allclose(b_unflipped, boxes)


def test_distance_adaptive_subsample_ratio():
    np.random.seed(42)
    box = np.array([1.5, 1.8, 4.5, 30.0, 0.0, -1.0, 0.0], dtype=np.float32)
    points = np.random.uniform(-1.0, 1.0, size=(1000, 4)).astype(np.float32)
    # Move from 10m to 20m -> ratio (10/20)^2 = 0.25 -> ~250 points
    subsampled = distance_adaptive_subsample(points, box, r_origin=10.0, r_target=20.0)
    assert 200 <= len(subsampled) <= 300

    # Moving closer (r_target <= r_origin) should retain all points
    retained_all = distance_adaptive_subsample(
        points, box, r_origin=20.0, r_target=10.0
    )
    assert len(retained_all) == len(points)


def test_radiometric_intensity_calibrate():
    points = np.array([[10.0, 0.0, 0.0, 0.8]], dtype=np.float32)
    # Move further (10m -> 20m) -> intensity attenuates
    calibrated = radiometric_intensity_calibrate(
        points.copy(), r_origin=10.0, r_target=20.0, gamma=1.7
    )
    assert calibrated[0, 3] < points[0, 3]
    assert calibrated[0, 3] > 0.0


def test_mask_shadow_points_elevation_and_wrap():
    # Box at x=10, y=0, z=-1.0 (bottom), h=2.0 (spans z from -1.0 to 1.0, center z=0.0)
    box = np.array([2.0, 2.0, 4.0, 10.0, 0.0, -1.0, 0.0], dtype=np.float32)
    bg_points = np.array(
        [
            [5.0, 0.0, 0.0, 0.5],  # in front of box (x=5) -> KEEP
            [20.0, 0.0, 0.0, 0.5],  # directly behind in shadow (x=20, z=0) -> REMOVE
            [20.0, 10.0, 0.0, 0.5],  # to the side (y=10) -> KEEP
            [20.0, 0.0, 5.0, 0.5],  # above shadow cone (z=5) -> KEEP
        ],
        dtype=np.float32,
    )

    retained = mask_shadow_points(bg_points, box)
    assert len(retained) == 3
    assert np.allclose(retained[0, :3], [5.0, 0.0, 0.0])
    assert np.allclose(retained[1, :3], [20.0, 10.0, 0.0])
    assert np.allclose(retained[2, :3], [20.0, 0.0, 5.0])


def test_line_of_sight_occlusion_behind_wall():
    # Wall located at x=15, y=0, z in [-1.0, 2.0]
    wall_ys = np.linspace(-3, 3, 20)
    wall_zs = np.linspace(-1.0, 2.0, 10)
    wy, wz = np.meshgrid(wall_ys, wall_zs)
    wall_pts = np.column_stack(
        [np.full(200, 15.0), wy.ravel(), wz.ravel(), np.ones(200)]
    ).astype(np.float32)

    # Target car at x=30, y=0, z=-1.6 (directly behind the wall from Ego at (0,0))
    car_box = np.array(
        [1.0, 1.5, 2.0, 4.5, 30.0, 0.0, -1.6, 0.0], dtype=np.float32
    )
    is_occluded = check_line_of_sight_occlusion(car_box, wall_pts)
    assert is_occluded is True


def test_line_of_sight_occlusion_clear_path():
    # Ground surface only at z=-1.6, no elevated obstacle between Ego and car
    road_xs = np.linspace(5, 25, 20)
    road_pts = np.column_stack(
        [road_xs, np.zeros(20), np.full(20, -1.6), np.ones(20)]
    ).astype(np.float32)

    car_box = np.array(
        [1.0, 1.5, 2.0, 4.5, 30.0, 0.0, -1.6, 0.0], dtype=np.float32
    )
    is_occluded = check_line_of_sight_occlusion(car_box, road_pts)
    assert is_occluded is False

