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

    # Synthetic pointcloud: 100 points
    pts = np.random.uniform(-5, 5, size=(100, 4)).astype(np.float32)
    pts[:, :3] += np.array([20.0, 5.0, -1.0])
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



