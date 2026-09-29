import sys
from pathlib import Path
import pickle
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "detector"),
    str(ROOT / "detector" / "core" / "datasets"),
]

from core.datasets.utils_1.gt_sampler import GTSampler


def test_gt_sampler_with_8col_boxes(tmp_path):
    np.random.seed(42)
    db_file = tmp_path / "mock_db.pkl"
    mock_db = {
        "Car": [
            {
                "box": np.array(
                    [0.0, 1.5, 1.8, 4.5, 15.0, 0.0, -1.0, 0.0],
                    dtype=np.float32,
                ),
                "points": np.random.uniform(-0.5, 0.5, size=(50, 4)).astype(
                    np.float32
                ),
                "r_origin": 15.0,
                "num_points": 50,
            }
        ]
    }
    with open(db_file, "wb") as f:
        pickle.dump(mock_db, f)

    sampler = GTSampler(str(db_file), sample_counts={"Car": 1}, p=1.0)
    init_points = np.zeros((100, 4), dtype=np.float32)
    # Existing scene with 1 pedestrian (cls=1)
    init_boxes = np.array(
        [[1.0, 1.7, 0.6, 0.8, 10.0, 5.0, -1.0, 0.0]], dtype=np.float32
    )

    aug_points, aug_boxes = sampler(init_points, init_boxes)
    assert len(aug_boxes) == 2
    assert aug_boxes.shape[1] == 8
    assert aug_boxes[1, 0] == 0.0  # Inserted car has class id 0
    assert len(aug_points) > 100


def test_gt_sampler_probability_zero(tmp_path):
    db_file = tmp_path / "mock_db.pkl"
    mock_db = {
        "Car": [
            {
                "box": np.array(
                    [0.0, 1.5, 1.8, 4.5, 15.0, 0.0, -1.0, 0.0],
                    dtype=np.float32,
                ),
                "points": np.zeros((10, 4), dtype=np.float32),
                "r_origin": 15.0,
                "num_points": 10,
            }
        ]
    }
    with open(db_file, "wb") as f:
        pickle.dump(mock_db, f)

    sampler = GTSampler(str(db_file), sample_counts={"Car": 1}, p=0.0)
    init_points = np.zeros((50, 4), dtype=np.float32)
    init_boxes = np.zeros((0, 8), dtype=np.float32)

    aug_points, aug_boxes = sampler(init_points, init_boxes)
    assert len(aug_boxes) == 0
    assert len(aug_points) == 50


def test_gt_sampler_removes_interior_background_points(tmp_path):
    np.random.seed(42)
    db_file = tmp_path / "mock_db.pkl"
    # Canonical car points (centered at 0, bottom at 0)
    car_points = np.zeros((20, 4), dtype=np.float32)
    car_points[:, 0] = np.linspace(-1.0, 1.0, 20)  # within l=4.5
    car_points[:, 1] = np.linspace(-0.5, 0.5, 20)  # within w=1.8
    car_points[:, 2] = np.linspace(0.1, 1.4, 20)  # within h=1.5
    car_points[:, 3] = 0.9

    mock_db = {
        "Car": [
            {
                "box": np.array(
                    [0.0, 1.5, 1.8, 4.5, 15.0, 0.0, -1.6, 0.0],
                    dtype=np.float32,
                ),
                "points": car_points,
                "r_origin": 15.0,
                "num_points": 20,
            }
        ]
    }
    with open(db_file, "wb") as f:
        pickle.dump(mock_db, f)

    sampler = GTSampler(
        str(db_file), sample_counts={"Car": 1}, p=1.0, enable_physics=True
    )
    # Background scene with ground points and a cluster of noise points
    bg_pts = []
    for x in np.linspace(5, 40, 30):
        for y in np.linspace(-10, 10, 20):
            bg_pts.append([x, y, -1.6, 0.5])
    init_points = np.array(bg_pts, dtype=np.float32)
    init_boxes = np.zeros((0, 8), dtype=np.float32)

    aug_points, aug_boxes = sampler(init_points, init_boxes)
    assert len(aug_boxes) == 1
    # Verify placed car is within BEV bounds
    bx = aug_boxes[0, 4]
    by = aug_boxes[0, 5]
    assert 2.0 <= bx <= 65.0
    assert -35.0 <= by <= 35.0
    assert len(aug_points) > 0


def test_gt_sampler_rejects_placement_inside_wall(tmp_path):
    db_file = tmp_path / "mock_db.pkl"
    mock_db = {
        "Car": [
            {
                "box": np.array(
                    [0.0, 1.5, 1.8, 4.5, 15.0, 0.0, -1.6, 0.0],
                    dtype=np.float32,
                ),
                "points": np.ones((50, 4), dtype=np.float32),
                "r_origin": 15.0,
                "num_points": 50,
            }
        ]
    }
    with open(db_file, "wb") as f:
        pickle.dump(mock_db, f)

    sampler = GTSampler(
        str(db_file), sample_counts={"Car": 1}, p=1.0, enable_physics=True
    )

    # Empty scene with only a massive wall at x in [5, 65], y in [-35, 35], z in [-0.5, 3.0]
    # No valid flat ground anywhere
    xs = np.linspace(5, 65, 30)
    ys = np.linspace(-30, 30, 30)
    xx, yy = np.meshgrid(xs, ys)
    wall_pts = np.column_stack(
        [xx.ravel(), yy.ravel(), np.full(900, 1.0), np.ones(900)]
    ).astype(np.float32)

    aug_pts, aug_boxes = sampler(wall_pts, np.zeros((0, 8), dtype=np.float32))
    # Since only elevated obstacles exist and no road support exists, placement must be safely rejected
    assert len(aug_boxes) == 0

