"""Tests for Omni-PCU SOTA Sampler."""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pytest
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "detector"),
    str(ROOT / "detector/core/datasets"),
    str(ROOT / "tools/kitti_training_pipeline"),
]

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


def test_build_database_includes_r_origin_and_density(tmp_path):
    import json
    import sys
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT / "tools/kitti_training_pipeline"))
    from build_gt_database import build_database
    from common import read_json

    proc_root = tmp_path / "processed"
    (proc_root / "pointcloud").mkdir(parents=True)
    (proc_root / "label").mkdir(parents=True)

    # Synthetic pointcloud: points tightly inside [x=20, y=5, z=-1.0]
    pts = np.random.uniform(-0.5, 0.5, size=(50, 4)).astype(np.float32)
    pts[:, 0] += 20.0
    pts[:, 1] += 5.0
    pts[:, 2] += -1.0
    pts.tofile(proc_root / "pointcloud" / "000001.bin")

    # Label: Car at [x=20, y=5, z=-1.75, l=4.0, w=2.0, h=1.5, yaw=0.0]
    (proc_root / "label" / "000001.txt").write_text(
        "Car 1.5 2.0 4.0 20.0 5.0 -1.75 0.0\n", encoding="utf-8"
    )
    manifest = tmp_path / "train.txt"
    manifest.write_text("000001\n", encoding="utf-8")

    out_dir = tmp_path / "gt_database"
    dest = build_database(proc_root, manifest, out_dir, min_points=1)
    meta = read_json(dest)

    entry = meta["db_infos"]["Car"][0]
    assert "r_origin" in entry
    assert np.isclose(entry["r_origin"], np.hypot(20.0, 5.0), atol=1e-3)
    assert "density" in entry
    assert entry["density"] > 0.0


def test_curriculum_difficulty_partitioning_and_scheduling(tmp_path):
    import json
    from core.datasets.augmentor.omni_sampler import OmniDataBaseSampler

    config = {
        "DB_INFO_PATH": ["gt_database/dbinfos_train.json"],
        "SAMPLE_GROUPS": ["Car:2"],
        "CURRICULUM": {
            "ENABLED": True,
            "WARMUP_EPOCHS": 10,
            "HARD_RATIO_BASE": 0.10,
            "HARD_RATIO_TARGET": 0.90,
            "DIFFICULTY_THRESHOLD": 0.40,
        },
    }
    meta = {
        "format": "lidar_gt_database_v1",
        "num_point_features": 4,
        "source_frame_ids": ["000001"],
        "db_infos": {
            "Car": [
                {
                    "name": "Car", "path": "car_easy.bin", "image_idx": "000001",
                    "box3d_lidar": [10.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
                    "num_points_in_gt": 500, "r_origin": 10.0, "density": 40.0,
                },
                {
                    "name": "Car", "path": "car_hard.bin", "image_idx": "000001",
                    "box3d_lidar": [60.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
                    "num_points_in_gt": 15, "r_origin": 60.0, "density": 1.2,
                },
            ]
        },
    }
    (tmp_path / "gt_database").mkdir()
    np.ones((500, 4), dtype=np.float32).tofile(tmp_path / "car_easy.bin")
    np.ones((15, 4), dtype=np.float32).tofile(tmp_path / "car_hard.bin")
    with (tmp_path / "gt_database" / "dbinfos_train.json").open("w") as f:
        json.dump(meta, f)

    sampler = OmniDataBaseSampler(tmp_path, config, {"Car": 0})
    # Verify pools partitioned
    assert len(sampler.easy_pools["Car"]) == 1
    assert len(sampler.hard_pools["Car"]) == 1

    # Verify probability schedule
    sampler.set_epoch(0)
    assert np.isclose(sampler.current_hard_ratio, 0.10)
    sampler.set_epoch(5)
    assert np.isclose(sampler.current_hard_ratio, 0.50)
    sampler.set_epoch(10)
    assert np.isclose(sampler.current_hard_ratio, 0.90)


def test_legacy_openpcdet_config_compatibility(tmp_path):
    import json
    from core.datasets.augmentor.omni_sampler import OmniDataBaseSampler

    # Legacy config: no CURRICULUM, no PLACEMENT, no PHYSICS, no r_origin in db
    legacy_config = {
        "DB_INFO_PATH": ["gt_database/dbinfos_train.json"],
        "SAMPLE_GROUPS": ["Car:1"],
        "PREPARE": {
            "filter_by_min_points": ["Car:5"],
            "filter_by_difficulty": [-1],
        },
    }
    meta = {
        "format": "lidar_gt_database_v1",
        "num_point_features": 4,
        "source_frame_ids": ["000001"],
        "db_infos": {
            "Car": [
                {
                    "name": "Car", "path": "car_legacy.bin", "image_idx": "000001",
                    "box3d_lidar": [15.0, 2.0, -1.0, 4.2, 1.8, 1.6, 0.0],
                    "num_points_in_gt": 40,
                    # Notice: NO r_origin or density!
                }
            ]
        },
    }
    (tmp_path / "gt_database").mkdir()
    np.ones((40, 4), dtype=np.float32).tofile(tmp_path / "car_legacy.bin")
    with (tmp_path / "gt_database" / "dbinfos_train.json").open("w") as f:
        json.dump(meta, f)

    sampler = OmniDataBaseSampler(tmp_path, legacy_config, {"Car": 0})
    assert len(sampler.db_infos["Car"]) == 1
    # Check that r_origin was automatically derived
    assert "r_origin" in sampler.db_infos["Car"][0]
    assert np.isclose(sampler.db_infos["Car"][0]["r_origin"], np.hypot(15.0, 2.0))

    # Test sampling runs cleanly without errors
    scene_pts = np.random.uniform(-10, 10, size=(100, 4)).astype(np.float32)
    scene_boxes = np.empty((0, 8), dtype=np.float32)
    pts_out, boxes_out = sampler(scene_pts, scene_boxes)
    assert len(boxes_out) <= 1


def test_density_subsample_enforces_five_point_floor(tmp_path):
    import json
    from core.datasets.augmentor.omni_sampler import OmniDataBaseSampler

    # Object at r=5m with 10 points
    config = {
        "DB_INFO_PATH": ["gt_database/dbinfos_train.json"],
        "SAMPLE_GROUPS": ["Car:1"],
        "PHYSICS": {"ENABLE_DENSITY_SUBSAMPLE": True},
        "PLACEMENT": {"RANGE_TIERS": [[60.0, 65.0]], "TIER_WEIGHTS": [1.0]},
    }
    meta = {
        "format": "lidar_gt_database_v1", "num_point_features": 4, "source_frame_ids": ["000001"],
        "db_infos": {
            "Car": [{
                "name": "Car", "path": "c.bin", "image_idx": "000001",
                "box3d_lidar": [5.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0],
                "num_points_in_gt": 10, "r_origin": 5.0,
            }]
        },
    }
    (tmp_path / "gt_database").mkdir()
    np.ones((10, 4), dtype=np.float32).tofile(tmp_path / "c.bin")
    with (tmp_path / "gt_database" / "dbinfos_train.json").open("w") as f:
        json.dump(meta, f)

    sampler = OmniDataBaseSampler(tmp_path, config, {"Car": 0})
    pts_out, boxes_out = sampler(np.empty((0, 4), dtype=np.float32), np.empty((0, 8), dtype=np.float32))
    if len(boxes_out) > 0:
        # Must retain at least 5 points due to the safety floor
        assert len(pts_out) >= 5


def test_anti_wall_rejects_elevated_obstacles(tmp_path):
    import json
    from core.datasets.augmentor.omni_sampler import OmniDataBaseSampler

    config = {
        "DB_INFO_PATH": ["gt_database/dbinfos_train.json"],
        "SAMPLE_GROUPS": ["Car:1"],
        "PHYSICS": {"ENABLE_ANTI_WALL": True, "ANTI_WALL_HEIGHT_THRESH": 0.35, "MAX_OBSTACLE_POINTS": 2},
        "PLACEMENT": {"RANGE_TIERS": [[15.0, 15.0]], "TIER_WEIGHTS": [1.0], "MAX_ATTEMPTS": 1},
    }
    # Scene with a dense horizontal wall of points spanning x in [13, 17], y in [-20, 20], z = -0.8
    # (ground is at -1.65m, curb is -1.30m, roof is -0.15m -> inside box and above curb)
    y_coords = np.linspace(-20, 20, 200, dtype=np.float32)
    wall_points = np.zeros((200, 4), dtype=np.float32)
    wall_points[:, 0] = 15.0
    wall_points[:, 1] = y_coords
    wall_points[:, 2] = -0.8

    meta = {
        "format": "lidar_gt_database_v1", "num_point_features": 4, "source_frame_ids": ["000001"],
        "db_infos": {"Car": [{"name": "Car", "path": "c.bin", "image_idx": "000001",
                              "box3d_lidar": [15, 0, 0, 4, 2, 1.5, 0], "num_points_in_gt": 20}]},
    }
    (tmp_path / "gt_database").mkdir()
    np.ones((20, 4), dtype=np.float32).tofile(tmp_path / "c.bin")
    with (tmp_path / "gt_database" / "dbinfos_train.json").open("w") as f:
        json.dump(meta, f)

    sampler = OmniDataBaseSampler(tmp_path, config, {"Car": 0})
    plane = np.array([0.0, 0.0, -1.0, -1.65], dtype=np.float64)  # z_ground = -1.65
    pts_out, boxes_out = sampler(wall_points, np.empty((0, 8), dtype=np.float32), road_plane=plane)
    # Candidate should be rejected by anti-wall check; no box inserted
    assert len(boxes_out) == 0


def test_transactional_rollback_preserves_original_visibility(tmp_path):
    import json
    from core.datasets.augmentor.omni_sampler import OmniDataBaseSampler

    config = {
        "DB_INFO_PATH": ["gt_database/dbinfos_train.json"],
        "SAMPLE_GROUPS": ["Car:1"],
        "PHYSICS": {"MIN_VISIBLE_RATIO": 0.60},
        "PLACEMENT": {"RANGE_TIERS": [[15.0, 15.0]], "TIER_WEIGHTS": [1.0], "MAX_ATTEMPTS": 1},
    }
    # Existing box with 10 points at x=15, y=0
    orig_box = np.array([[0, 1.5, 2.0, 4.0, 15.0, 0.0, -1.65, 0.0]], dtype=np.float32)
    orig_pts = np.zeros((10, 4), dtype=np.float32)
    orig_pts[:, 0] = 15.0
    orig_pts[:, 1] = 0.0
    orig_pts[:, 2] = -1.0  # inside orig_box

    meta = {
        "format": "lidar_gt_database_v1", "num_point_features": 4, "source_frame_ids": ["000001"],
        "db_infos": {"Car": [{"name": "Car", "path": "c.bin", "image_idx": "000001",
                              "box3d_lidar": [15, 0, 0, 4, 2, 1.5, 0], "num_points_in_gt": 20}]},
    }
    (tmp_path / "gt_database").mkdir()
    np.ones((20, 4), dtype=np.float32).tofile(tmp_path / "c.bin")
    with (tmp_path / "gt_database" / "dbinfos_train.json").open("w") as f:
        json.dump(meta, f)

    sampler = OmniDataBaseSampler(tmp_path, config, {"Car": 0})
    plane = np.array([0.0, 0.0, -1.0, -1.65], dtype=np.float64)
    pts_out, boxes_out = sampler(orig_pts, orig_box, road_plane=plane)
    # Rollback must preserve the original box and its 10 points
    assert len(boxes_out) == 1
    assert len(pts_out) == 10


def test_pickle_serialization_across_workers(tmp_path):
    import pickle
    import json
    from core.datasets.augmentor.omni_sampler import OmniDataBaseSampler

    config = {
        "DB_INFO_PATH": ["gt_database/dbinfos_train.json"],
        "SAMPLE_GROUPS": ["Car:1"],
    }
    meta = {
        "format": "lidar_gt_database_v1", "num_point_features": 4, "source_frame_ids": ["000001"],
        "db_infos": {"Car": [{"name": "Car", "path": "c.bin", "image_idx": "000001",
                              "box3d_lidar": [10, 0, 0, 4, 2, 1.5, 0], "num_points_in_gt": 5}]},
    }
    (tmp_path / "gt_database").mkdir()
    np.ones((5, 4), dtype=np.float32).tofile(tmp_path / "c.bin")
    with (tmp_path / "gt_database" / "dbinfos_train.json").open("w") as f:
        json.dump(meta, f)

    sampler = OmniDataBaseSampler(tmp_path, config, {"Car": 0})
    serialized = pickle.dumps(sampler)
    deserialized = pickle.loads(serialized)
    assert deserialized is not None
    assert deserialized.cache is not None
    assert len(deserialized.cache) == 0


def test_data_augmentor_loads_omni_gt_sampling(tmp_path):
    import json
    from core.datasets.augmentor.data_augmentor import DataAugmentor

    profile = {
        "AUG_CONFIG_LIST": [
            {
                "NAME": "omni_gt_sampling",
                "SAMPLE_GROUPS": ["Car:2"],
                "DB_INFO_PATH": ["gt_database/dbinfos_train.json"],
                "CURRICULUM": {"ENABLED": True, "WARMUP_EPOCHS": 5},
            }
        ]
    }
    meta = {
        "format": "lidar_gt_database_v1", "num_point_features": 4, "source_frame_ids": ["000001"],
        "db_infos": {"Car": []},
    }
    (tmp_path / "gt_database").mkdir()
    with (tmp_path / "gt_database" / "dbinfos_train.json").open("w") as f:
        json.dump(meta, f)

    aug = DataAugmentor(tmp_path, profile, {"Car": 0}, allowed_frame_ids=["000001"])
    assert aug is not None

    # Test set_epoch forwarding
    aug.set_epoch(3)
    _, _, omni_sampler = aug.queue[0]
    assert omni_sampler.epoch == 3

    # Test call runs cleanly
    pts = np.ones((50, 4), dtype=np.float32)
    boxes = np.array([[0, 1.5, 2.0, 4.0, 10.0, 0.0, -1.65, 0.0]], dtype=np.float32)
    out_pts, out_boxes = aug(pts, boxes)
    assert len(out_pts) == 50
    assert len(out_boxes) == 1


def test_omni_pcu_gt_profile_supports_model_forward_backward(tmp_path):
    import json
    import torch
    from pathlib import Path
    ROOT = Path(__file__).resolve().parents[1]
    import sys
    sys.path.insert(0, str(ROOT / "tools/kitti_training_pipeline"))
    from common import build_model, create_experiment_config
    from build_gt_database import build_database
    from core.datasets.dataset import Dataset
    from core.losses.loss_fn import LossFunction

    profile_path = ROOT / "configs/augmentation/omni_pcu_gt.json"
    assert profile_path.is_file(), "omni_pcu_gt.json configuration missing"

    # Set up mock processed dataset
    proc = tmp_path / "proc"
    (proc / "pointcloud").mkdir(parents=True)
    (proc / "label").mkdir(parents=True)
    (proc / "planes").mkdir(parents=True)

    # 100 points
    pts = np.random.uniform(-5, 5, size=(100, 4)).astype(np.float32)
    pts[:, 0] += 20.0
    pts[:, 1] += 5.0
    pts[:, 2] += -1.0
    pts.tofile(proc / "pointcloud" / "000001.bin")
    pts.tofile(proc / "pointcloud" / "000002.bin")

    # Label: Car at [x=20, y=5, z=-1.75, l=4.0, w=2.0, h=1.5, yaw=0.0]
    (proc / "label" / "000001.txt").write_text(
        "Car 1.5 2.0 4.0 20.0 5.0 -1.75 0.0\n", encoding="utf-8"
    )
    (proc / "label" / "000002.txt").write_text(
        "Car 1.5 2.0 4.0 20.0 5.0 -1.75 0.0\n", encoding="utf-8"
    )
    # Road plane
    (proc / "planes" / "000001.txt").write_text(
        "Plane 0.0 0.0 -1.0 -1.65\n", encoding="utf-8"
    )
    (proc / "planes" / "000002.txt").write_text(
        "Plane 0.0 0.0 -1.0 -1.65\n", encoding="utf-8"
    )
    manifest = tmp_path / "train.txt"
    manifest.write_text("000001;kitti\n000002;kitti\n", encoding="utf-8")

    # Build GT database
    db_dir = tmp_path / "gt_database"
    build_database(proc, manifest, db_dir, min_points=1)

    # Load config and override data paths
    base_cfg = json.loads((ROOT / "configs/config.json").read_text())
    aug_cfg = json.loads(profile_path.read_text())
    cfg = create_experiment_config(base_cfg, aug_cfg)
    cfg["data"]["kitti"]["location"] = str(proc)
    cfg["data"]["train_manifest"] = str(manifest)
    cfg["data"]["processed_data_path"] = str(proc)
    cfg["data"]["val_manifest"] = str(manifest)
    # Override db path to tmp_path
    cfg["augmentation"]["AUG_CONFIG_LIST"][0]["DB_INFO_PATH"] = [
        str(db_dir / "dbinfos_train.json")
    ]

    dataset = Dataset(str(manifest), cfg["data"], cfg["augmentation"], "gaussian", "train")
    assert len(dataset) == 2

    # Verify 2-worker DataLoader serialization
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=2)
    batches = list(loader)
    assert len(batches) == 1
    batch = batches[0]

    # Model forward and backward
    model = build_model(cfg)
    criterion = LossFunction("gaussian", cfg["loss"])
    pred = model(batch["voxel"])
    loss = criterion(pred, batch)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in model.parameters())






