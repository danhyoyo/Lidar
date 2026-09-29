import sys
from pathlib import Path
import pickle
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "detector"),
    str(ROOT / "detector" / "core" / "datasets"),
]

from core.datasets.dataset import Dataset
from utils_1.transform import OneOf


def create_mock_dataset_env(tmp_path):
    data_dir = tmp_path / "data"
    pointcloud_dir = data_dir / "pointcloud"
    label_dir = data_dir / "label"
    pointcloud_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    # 1 sample
    sid = "000001"
    pts = np.zeros((100, 4), dtype=np.float32)
    pts[:, 0] = np.linspace(5.0, 20.0, 100)
    pts[:, 1] = np.linspace(-3.0, 3.0, 100)
    pts[:, 2] = -1.6
    pts[:, 3] = 0.5
    pts.tofile(str(pointcloud_dir / f"{sid}.bin"))

    label_str = "Car 1.5 1.8 4.5 10.0 0.0 -1.6 0.0\n"
    (label_dir / f"{sid}.txt").write_text(label_str, encoding="utf-8")

    split_file = tmp_path / "train.txt"
    split_file.write_text(f"{sid};kitti\n", encoding="utf-8")

    db_file = tmp_path / "mock_db.pkl"
    mock_db = {
        "Car": [
            {
                "box": np.array(
                    [0.0, 1.5, 1.8, 4.5, 15.0, 0.0, -1.6, 0.0],
                    dtype=np.float32,
                ),
                "points": np.random.uniform(-0.5, 0.5, size=(40, 4)).astype(
                    np.float32
                ),
                "r_origin": 15.0,
                "num_points": 40,
            }
        ]
    }
    with open(db_file, "wb") as f:
        pickle.dump(mock_db, f)

    main_cfg = {
        "num_classes": 3,
        "out_size_factor": 4,
        "gaussian_overlap": 0.5,
        "min_radius": 2,
        "bev_encoding": {"name": "rich8"},
        "kitti": {
            "location": str(data_dir),
            "objects": {"Car": 0, "Pedestrian": 1, "Cyclist": 2},
            "geometry": {
                "x_min": 0.0,
                "x_max": 70.4,
                "x_res": 0.1,
                "y_min": -40.0,
                "y_max": 40.0,
                "y_res": 0.1,
                "z_min": -3.0,
                "z_max": 1.0,
                "z_res": 0.1,
            },
        },
    }
    return str(split_file), main_cfg, str(db_file)


def test_legacy_mode_toggle_off(tmp_path):
    split_file, main_cfg, _ = create_mock_dataset_env(tmp_path)
    aug_cfg_legacy = {
        "p": 0.5,
        "rotation": {"use": True, "limit_angle": 15.0, "p": 0.5},
        "scaling": {"use": True, "range": [0.95, 1.05], "p": 0.5},
        "translation": {"use": True, "scale": 0.1, "p": 0.5},
    }
    ds = Dataset(
        split_file,
        main_cfg,
        aug_cfg_legacy,
        cls_encoding="gaussian",
        task="train",
    )
    assert ds.use_pcu_aug is False
    assert isinstance(ds.augment, OneOf)
    item = ds[0]
    assert "voxel" in item
    assert item["voxel"].shape == (8, 800, 704)
    assert not torch.isnan(item["voxel"]).any()


def test_pcu_mode_toggle_on(tmp_path):
    split_file, main_cfg, db_file = create_mock_dataset_env(tmp_path)
    aug_cfg_pcu = {
        "use_pcu_aug": True,
        "pcu_aug": {
            "enable_gt_sampling": True,
            "gt_database_path": db_file,
            "sample_counts": {"Car": 1},
            "enable_shadow_masking": True,
            "enable_density_subsample": True,
            "enable_radiometric_calibration": True,
        },
        "flip_y": {"use": True, "p": 1.0},
        "rotation": {"use": True, "limit_angle": 15.0, "p": 0.5},
        "scaling": {"use": True, "range": [0.95, 1.05], "p": 0.5},
        "translation": {"use": True, "scale": 0.1, "p": 0.5},
    }
    ds = Dataset(
        split_file, main_cfg, aug_cfg_pcu, cls_encoding="gaussian", task="train"
    )
    assert ds.use_pcu_aug is True
    assert ds.gt_sampler is not None
    item = ds[0]
    assert "voxel" in item
    assert item["voxel"].shape == (8, 800, 704)
    assert not torch.isnan(item["voxel"]).any()
    assert not torch.isinf(item["voxel"]).any()
