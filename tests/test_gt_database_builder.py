import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "detector"),
    str(ROOT / "detector" / "core" / "datasets"),
    str(ROOT / "tools" / "dataset_converter"),
]

from tools.dataset_converter.create_gt_database import (
    extract_object_points,
    build_kitti_gt_database,
)


def test_extract_object_points_bottom_convention():
    # Box bottom center at (10, 0, -1.0), h=2.0 (spans z from -1.0 to 1.0)
    # 7-col box: [h, w, l, x, y, z, yaw]
    box = np.array([2.0, 2.0, 4.0, 10.0, 0.0, -1.0, 0.0], dtype=np.float32)
    points = np.array(
        [
            [10.0, 0.0, -0.5, 0.8],  # Inside (z = -0.5 is between -1.0 and 1.0)
            [11.0, 0.5, 0.5, 0.6],  # Inside
            [10.0, 0.0, 1.5, 0.9],  # Above top (z = 1.5 > 1.0) -> Outside
            [10.0, 0.0, -1.5, 0.2],  # Below bottom (z = -1.5 < -1.0) -> Outside
            [15.0, 0.0, 0.0, 0.9],  # Outside in x
        ],
        dtype=np.float32,
    )

    inside_pts = extract_object_points(points, box)
    assert len(inside_pts) == 2
    assert np.allclose(inside_pts[0, :3], [10.0, 0.0, -0.5])
    assert np.allclose(inside_pts[1, :3], [11.0, 0.5, 0.5])


def test_extract_object_points_8col():
    # 8-col box: [cls, h, w, l, x, y, z, yaw]
    box_8 = np.array(
        [0.0, 2.0, 2.0, 4.0, 10.0, 0.0, -1.0, 0.0], dtype=np.float32
    )
    points = np.array([[10.0, 0.0, -0.5, 0.8]], dtype=np.float32)
    inside_pts = extract_object_points(points, box_8)
    assert len(inside_pts) == 1


def test_build_kitti_gt_database_mock(tmp_path):
    import pickle

    # Create mock KITTI directory structure
    processed_dir = tmp_path / "processed"
    pointcloud_dir = processed_dir / "training" / "pointcloud"
    label_dir = processed_dir / "training" / "label"
    pointcloud_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    # Mock file 000001
    sample_id = "000001"
    # Points with 10 inside car box
    car_pts = np.zeros((10, 4), dtype=np.float32)
    car_pts[:, 0] = np.linspace(9.0, 11.0, 10)  # around x=10
    car_pts[:, 1] = np.linspace(-0.5, 0.5, 10)  # around y=0
    car_pts[:, 2] = np.linspace(-1.0, 0.5, 10)  # z in [-1, 0.5]
    car_pts[:, 3] = 0.5
    car_pts.tofile(str(pointcloud_dir / f"{sample_id}.bin"))

    # Label: Car h w l x y z yaw
    label_content = "Car 1.5 1.8 4.5 10.0 0.0 -1.0 0.0\n"
    (label_dir / f"{sample_id}.txt").write_text(
        label_content, encoding="utf-8"
    )

    train_ids_file = tmp_path / "train.txt"
    train_ids_file.write_text(f"{sample_id}\n", encoding="utf-8")

    out_pkl = tmp_path / "gt_database.pkl"
    build_kitti_gt_database(
        str(processed_dir), str(train_ids_file), str(out_pkl), min_points=5
    )

    assert out_pkl.exists()
    with open(out_pkl, "rb") as f:
        db = pickle.load(f)

    assert "Car" in db
    assert len(db["Car"]) == 1
    sample = db["Car"][0]
    assert sample["num_points"] == 10
    assert sample["box"].shape == (8,)
    assert np.isclose(sample["r_origin"], 10.0)
