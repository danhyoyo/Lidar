#!/usr/bin/env python3
"""Reproducible synthetic CPU benchmark; matched quotas, acceptance counts, warm JIT.

PCU is loaded read-only from this repository's git reference when available.
This measures current/PCU/hybrid implementations, not MMDetection3D runtime.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "detector"), str(ROOT / "tools/kitti_training_pipeline")]
from core.datasets.augmentor.database_sampler import DataBaseSampler
from core.datasets.augmentor.hybrid_sampler import HybridDataBaseSampler


def load_pcu(directory, reference):
    prefix = "core.datasets.utils_1."
    folder = directory / "pcu_source"
    folder.mkdir()
    for name in ("box_geometry", "collision_ground", "physics_aug", "gt_sampler"):
        result = subprocess.run(["git", "show", f"{reference}:detector/core/datasets/utils_1/{name}.py"],
                                cwd=ROOT, text=True, capture_output=True)
        if result.returncode:
            return None
        path = folder / f"{name}.py"
        path.write_text(result.stdout)
        spec = importlib.util.spec_from_file_location(prefix + name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[prefix + name] = module
        spec.loader.exec_module(module)
    return module.GTSampler


def synthetic_database(directory, n_points, quota):
    rng = np.random.default_rng(2026)
    points = np.column_stack([rng.uniform(0, 70.4, n_points), rng.uniform(-40, 40, n_points),
                              np.full(n_points, -1.6), rng.uniform(0, 1, n_points)]).astype(np.float32)
    original = np.array([[0, 2, 2, 4, x, -30, -1.6, .1] for x in range(10, 67, 8)], np.float32)
    db, pcu_entries = [], []
    for i, (x, y) in enumerate((x, y) for x in range(8, 65, 8) for y in (-10, 0, 10, 20)):
        local = rng.uniform([-1.8, -.8, -.9, .1], [1.8, .8, .9, .9], (500, 4)).astype(np.float32)
        yaw = .3
        c, s = np.cos(yaw), np.sin(yaw)
        canonical = local.copy()
        local[:, :2] = canonical[:, :2] @ np.array([[c, -s], [s, c]]).T
        path = directory / f"object_{i}.bin"
        local.tofile(path)
        box = [x, y, -.6, 4, 2, 2, yaw]
        db.append(dict(name="Car", path=path.name, image_idx="000001", gt_idx=i,
                       box3d_lidar=box, num_points_in_gt=len(local)))
        canonical[:, 2] += 1
        pcu_entries.append(dict(box=np.array([0, 2, 2, 4, x, y, -1.6, yaw], np.float32),
                                points=canonical, num_points=len(local), r_origin=float(np.hypot(x, y))))
    (directory / "dbinfos_train.json").write_text(json.dumps(dict(
        format="lidar_gt_database_v1", num_point_features=4,
        source_frame_ids=["000001"], db_infos={"Car": db})))
    pcu_path = directory / "pcu.pkl"
    with pcu_path.open("wb") as stream:
        pickle.dump({"Car": pcu_entries}, stream)
    config = dict(DB_INFO_PATH=["dbinfos_train.json"], NUM_POINT_FEATURES=4,
                  SAMPLE_GROUPS=[f"Car:{quota}"], LIMIT_WHOLE_SCENE=False,
                  REMOVE_EXTRA_WIDTH=[0, 0, 0])
    return points, original, config, pcu_path


def benchmark(*, repeats=20, n_points=120000, quota=10, pcu_reference="9b99d92"):
    with tempfile.TemporaryDirectory(prefix="lidar-hybrid-benchmark-") as temporary:
        folder = Path(temporary)
        points, boxes, base, pcu_path = synthetic_database(folder, n_points, quota)
        kwargs = dict(root_path=folder, class_names={"Car": 0},
                      geometry=dict(x_min=0, x_max=70.4, y_min=-40, y_max=40), allowed_frame_ids=["000001"])
        current = DataBaseSampler(config=base, **kwargs)
        basic_cfg = base | dict(GEOMETRY_BACKEND="numba", CACHE_SIZE_MB=64, PLACEMENT_MODE="source",
                                COLLISION_MARGIN=0, MAX_PLACEMENT_ATTEMPTS=1, CANDIDATE_MULTIPLIER=1,
                                ENABLE_VISIBILITY_PROTECTION=False)
        basic = HybridDataBaseSampler(config=basic_cfg, **kwargs)
        default_cfg = basic_cfg | dict(ENABLE_GROUND_VALIDATION=True, ENABLE_STATIC_COLLISION=True,
                                       ENABLE_DENSITY_SUBSAMPLE=True)
        conservative = HybridDataBaseSampler(config=default_cfg, **kwargs)
        full_cfg = default_cfg | dict(ENABLE_LINE_OF_SIGHT=True, ENABLE_SHADOW_MASKING=True,
                                      ENABLE_RADIOMETRIC_CALIBRATION=True, ENABLE_VISIBILITY_PROTECTION=True)
        full = HybridDataBaseSampler(config=full_cfg, **kwargs)
        jobs = {"current": lambda rng: current(points, boxes, rng),
                "hybrid_basic_cached": lambda rng: basic(points, boxes, rng),
                "hybrid_ground_static_density": lambda rng: conservative(points, boxes, rng),
                "hybrid_all_physics": lambda rng: full(points, boxes, rng)}
        pcu_class = load_pcu(folder, pcu_reference)
        if pcu_class:
            common = dict(database_path=str(pcu_path), sample_counts={"Car": quota}, p=1,
                          placement_mode="source_relative", range_scale=(1, 1), azimuth_jitter_deg=0)
            pcu_basic, pcu_full = pcu_class(enable_physics=False, **common), pcu_class(enable_physics=True, **common)
            jobs.update(pcu_basic=lambda rng: pcu_basic(points, boxes), pcu_all_physics=lambda rng: pcu_full(points, boxes))
        result = {"environment": dict(python=platform.python_version(), numpy=np.__version__,
                                      platform=platform.platform(), processor=platform.processor()),
                  "scene": dict(points=n_points, original_boxes=len(boxes), quota=quota,
                                database_objects=32, points_per_object=500, repeats=repeats),
                  "scope": "Synthetic CPU; warm JIT and caches; source placement; matched quota; no GPU/AP claims",
                  "pcu_reference": pcu_reference if pcu_class else None, "results": {}}
        for name, job in jobs.items():
            for seed in range(4):
                np.random.seed(seed)
                job(np.random.default_rng(seed))
            timings, counts = [], []
            for seed in range(repeats):
                np.random.seed(seed+100)
                start = time.perf_counter()
                _, output_boxes = job(np.random.default_rng(seed+100))
                timings.append((time.perf_counter()-start)*1000)
                counts.append(len(output_boxes)-len(boxes))
            result["results"][name] = dict(median_ms=float(np.median(timings)), p95_ms=float(np.percentile(timings, 95)),
                mean_inserted=float(np.mean(counts)), min_inserted=min(counts), max_inserted=max(counts))
        result["cache"] = basic.cache.stats()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--points", type=int, default=120000)
    parser.add_argument("--quota", type=int, default=10)
    parser.add_argument("--pcu-reference", default="9b99d92")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.points < 1 or not 1 <= args.quota <= 32:
        parser.error("positive repeats/points and quota in [1,32] required")
    result = benchmark(repeats=args.repeats, n_points=args.points, quota=args.quota, pcu_reference=args.pcu_reference)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
