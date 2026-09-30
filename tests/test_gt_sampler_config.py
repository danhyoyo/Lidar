import sys
from pathlib import Path
import json
import pickle
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "detector"),
    str(ROOT / "detector" / "core" / "datasets"),
]

from core.datasets.dataset import Dataset
from core.datasets.utils_1.gt_sampler import build_gt_sampler, PHYSICS_FLAGS

ROOT = Path(__file__).resolve().parents[1]
PCU_CONFIG_PATH = ROOT / "configs/kitti/physics_augmentation/kitti_mobilepixornext_litemla_oga_pcu.json"
CUMULATIVE_CONFIG_PATH = ROOT / "configs/kitti/cumulative/kitti_mobilepixornext_m1_m2_m4_m5_oga.json"


def _create_mock_env(tmp_path):
    data_dir = tmp_path / "data"
    pointcloud_dir = data_dir / "pointcloud"
    label_dir = data_dir / "label"
    pointcloud_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    # 1 sample with ground and foreground
    pts = np.zeros((100, 4), dtype=np.float32)
    pts[:, 0] = np.linspace(10.0, 30.0, 100)
    pts[:, 1] = np.linspace(-5.0, 5.0, 100)
    pts[:, 2] = -1.6
    pts[:, 3] = 0.5
    pts.tofile(str(pointcloud_dir / "000001.bin"))
    (label_dir / "000001.txt").write_text("", encoding="utf-8")

    split_file = tmp_path / "train.txt"
    split_file.write_text("000001;kitti\n", encoding="utf-8")

    db_file = tmp_path / "mock_db.pkl"
    mock_db = {
        "Car": [
            {
                "box": np.array([0.0, 1.5, 1.8, 4.5, 15.0, 0.0, -1.6, 0.0], dtype=np.float32),
                "points": np.c_[
                    np.random.uniform(-0.5, 0.5, size=(40, 3)),
                    np.full((40, 1), 0.9),
                ].astype(np.float32),
                "r_origin": 15.0,
                "num_points": 40,
            }
        ],
        "Pedestrian": [
            {
                "box": np.array([1.0, 1.7, 0.6, 0.8, 10.0, 2.0, -1.6, 0.0], dtype=np.float32),
                "points": np.c_[
                    np.random.uniform(-0.2, 0.2, size=(20, 3)),
                    np.full((20, 1), 0.8),
                ].astype(np.float32),
                "r_origin": 10.0,
                "num_points": 20,
            }
        ],
        "Cyclist": [
            {
                "box": np.array([2.0, 1.6, 0.7, 1.6, 12.0, -2.0, -1.6, 0.0], dtype=np.float32),
                "points": np.c_[
                    np.random.uniform(-0.3, 0.3, size=(25, 3)),
                    np.full((25, 1), 0.85),
                ].astype(np.float32),
                "r_origin": 12.0,
                "num_points": 25,
            }
        ],
    }
    with open(db_file, "wb") as f:
        pickle.dump(mock_db, f)

    return str(split_file), str(data_dir), str(db_file)


@pytest.mark.parametrize("config_path,expected_counts", [
    (PCU_CONFIG_PATH, {"Car": 3, "Pedestrian": 5, "Cyclist": 5}),
    (CUMULATIVE_CONFIG_PATH, {"Car": 8, "Pedestrian": 6, "Cyclist": 6}),
])
def test_real_configs_effective_settings(tmp_path, config_path, expected_counts):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    pcu_cfg = cfg["augmentation"]["pcu_aug"]
    assert pcu_cfg["enable_gt_sampling"] is True
    assert pcu_cfg["sample_counts"] == expected_counts
    for flag in PHYSICS_FLAGS:
        assert flag in pcu_cfg
        assert isinstance(pcu_cfg[flag], bool)
        assert pcu_cfg[flag] is True
    assert pcu_cfg["min_visible_points"] == 5
    assert pcu_cfg["min_visible_ratio"] == 0.5

    split_file, data_dir, db_file = _create_mock_env(tmp_path)
    cfg["data"]["kitti"]["location"] = data_dir
    pcu_cfg["gt_database_path"] = db_file

    # Training task loads and enables all flags
    ds_train = Dataset(
        split_file,
        cfg["data"],
        cfg["augmentation"],
        cls_encoding="gaussian",
        task="train",
    )
    assert ds_train.gt_sampler is not None
    assert ds_train.gt_sampler.sample_counts == expected_counts
    for flag in PHYSICS_FLAGS:
        assert getattr(ds_train.gt_sampler, flag) is True
    assert ds_train.gt_sampler.min_visible_points == 5
    assert ds_train.gt_sampler.min_visible_ratio == 0.5

    # Val and test tasks do not load sampler
    for non_train in ["val", "test"]:
        ds_nontrain = Dataset(
            split_file,
            cfg["data"],
            cfg["augmentation"],
            cls_encoding="gaussian",
            task=non_train,
        )
        assert ds_nontrain.gt_sampler is None


def test_explicit_flags_override_legacy_enable_physics(tmp_path):
    _, _, db_file = _create_mock_env(tmp_path)
    # enable_physics=False, but explicitly enable shadow masking
    pcu_cfg = {
        "gt_database_path": db_file,
        "sample_counts": {"Car": 1},
        "enable_physics": False,
        "enable_shadow_masking": True,
    }
    sampler = build_gt_sampler(pcu_cfg)
    assert sampler.enable_physics is False
    assert sampler.enable_shadow_masking is True
    assert sampler.enable_density_subsample is False
    assert sampler.enable_radiometric_calibration is False
    assert sampler.enable_ground_validation is False
    assert sampler.enable_static_collision is False
    assert sampler.enable_line_of_sight is False

    # enable_physics=True, but explicitly disable density subsample
    pcu_cfg2 = {
        "gt_database_path": db_file,
        "sample_counts": {"Car": 1},
        "enable_physics": True,
        "enable_density_subsample": False,
    }
    sampler2 = build_gt_sampler(pcu_cfg2)
    assert sampler2.enable_physics is True
    assert sampler2.enable_density_subsample is False
    assert sampler2.enable_shadow_masking is True
    assert sampler2.enable_radiometric_calibration is True


@pytest.mark.parametrize("flag_to_disable", list(PHYSICS_FLAGS))
def test_each_flag_has_isolated_effect(tmp_path, monkeypatch, flag_to_disable):
    _, _, db_file = _create_mock_env(tmp_path)
    pcu_cfg = {
        "gt_database_path": db_file,
        "sample_counts": {"Car": 1},
        "enable_physics": True,
        flag_to_disable: False,
    }
    sampler = build_gt_sampler(pcu_cfg)
    assert getattr(sampler, flag_to_disable) is False
    for other_flag in PHYSICS_FLAGS:
        if other_flag != flag_to_disable:
            assert getattr(sampler, other_flag) is True
