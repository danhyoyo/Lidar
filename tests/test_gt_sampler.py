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
