"""Audit configured KITTI GT sampling before flips, jitter, BEV or target encoding.

Paths in the training config are interpreted from the current working directory,
as in training. Latency includes sampler diagnostics and metadata construction;
frame loading, database loading, hashing, plotting and JSON writing are excluded.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT, REPO_ROOT / "detector", REPO_ROOT / "detector/core/datasets"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def _read_manifest(path):
    frames = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(";")]
        frame_id = parts[0]
        if (
            len(parts) > 2
            or (len(parts) == 2 and parts[1] != "kitti")
            or not frame_id
            or frame_id in {".", ".."}
            or Path(frame_id).name != frame_id
            or "\\" in frame_id
            or any(char.isspace() for char in frame_id)
        ):
            raise ValueError(f"Invalid KITTI frame manifest {path}:{line_number}: {line!r}")
        frames.append(frame_id)
    if not frames:
        raise ValueError(f"Frame manifest {path} contains no frame IDs")
    return frames


def _load_frame(data_dir, frame_id):
    from tools.visualization.visualize_pcu_aug import load_kitti_frame

    root = Path(data_dir)
    layout = root if (root / "pointcloud").is_dir() else root / "training"
    label_file = layout / "label" / f"{frame_id}.txt"
    # Training requires annotations, including an empty label file for an empty scene.
    if not label_file.is_file():
        raise FileNotFoundError(f"Missing KITTI label file: {label_file}")
    return load_kitti_frame(root, frame_id)


def _identity(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _environment():
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git_commit": git_commit,
        "thread_environment": {
            name: os.environ.get(name)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
    }


def _save_example(path, frame_id, original_points, original_boxes, points, boxes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tools.visualization.visualize_pcu_aug import draw_bev_box

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for axis, cloud, labels, title in zip(
        axes, (original_points, points), (original_boxes, boxes), ("Original", "Measured sampler output")
    ):
        axis.scatter(cloud[:, 0], cloud[:, 1], s=1, color="gray")
        for box in labels:
            draw_bev_box(axis, box, color="#16853a")
        axis.set(title=f"{title}: {len(cloud)} points, {len(labels)} boxes", xlabel="X (m)", ylabel="Y (m)")
        axis.set_xlim(0, 70)
        axis.set_ylim(-35, 35)
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
    figure.suptitle(f"Frame {frame_id}: sampling before geometric augmentation")
    figure.tight_layout()
    try:
        figure.savefig(path, dpi=140)
    finally:
        plt.close(figure)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def run_audit(config_path, frames_path, seed, output_dir, *, warmup=3, max_examples=3):
    """Measure each manifest frame once and return the report written to audit.json."""
    for name, value in (("warmup", warmup), ("max_examples", max_examples)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    augmentation = config.get("augmentation", {})
    sampler_config = augmentation.get("pcu_aug", {})
    if augmentation.get("use_pcu_aug") is not True or sampler_config.get("enable_gt_sampling") is not True:
        raise ValueError("GT sampling must be enabled through use_pcu_aug and enable_gt_sampling for this audit")
    frame_ids = _read_manifest(frames_path)
    data_dir = config["data"]["kitti"]["location"]
    from core.datasets.utils_1.gt_sampler import build_gt_sampler

    sampler = build_gt_sampler(sampler_config)
    setting_names = (
        "sample_counts", "p", "enable_physics", "enable_ground_validation",
        "enable_static_collision", "enable_line_of_sight", "enable_shadow_masking",
        "enable_density_subsample", "enable_radiometric_calibration",
        "min_visible_points", "min_visible_ratio",
        "placement_mode", "range_scale", "azimuth_jitter_deg",
    )
    effective_settings = {name: getattr(sampler, name) for name in setting_names}
    identities = {
        "config": _identity(config_path),
        "manifest": _identity(frames_path),
        "database": _identity(sampler_config["gt_database_path"]),
    }
    # Warmup must not shift proposals used for the recorded seed.
    np.random.seed(seed ^ 0xA5A5A5A5)
    random.seed(seed ^ 0xA5A5A5A5)
    for index in range(warmup):
        points, boxes = _load_frame(data_dir, frame_ids[index % len(frame_ids)])
        sampler(points, boxes, return_diagnostics=True)
    np.random.seed(seed)
    random.seed(seed)

    frames, latencies, examples = [], [], []
    totals = {
        "attempted_by_class": {}, "accepted_by_class": {}, "rejected_by_reason": {},
        "removed_interior_points": 0, "removed_shadow_points": 0,
    }
    for frame_id in frame_ids:
        points, boxes = _load_frame(data_dir, frame_id)
        start = time.perf_counter()
        final_points, final_boxes, metadata = sampler(points, boxes, return_diagnostics=True)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        diagnostics = metadata["diagnostics"]
        for key in ("attempted_by_class", "accepted_by_class", "rejected_by_reason"):
            for name, count in diagnostics[key].items():
                totals[key][name] = totals[key].get(name, 0) + count
        for key in ("removed_interior_points", "removed_shadow_points"):
            totals[key] += diagnostics[key]
        record = {
            "frame_id": frame_id, "points_before": len(points), "points_after": len(final_points),
            "boxes_before": len(boxes), "boxes_after": len(final_boxes),
            "latency_ms": elapsed_ms, "diagnostics": diagnostics,
        }
        if len(examples) < max_examples:
            record["example_image"] = f"example_{len(frames):04d}_{frame_id}.png"
            examples.append((record["example_image"], frame_id, points, boxes, final_points, final_boxes))
        frames.append(record)
        latencies.append(elapsed_ms)

    report = {
        "schema_version": 1, "seed": seed, "warmup_calls": warmup,
        "data_directory": str(Path(data_dir).resolve()), "config": config,
        "effective_sampler_settings": effective_settings, "identities": identities,
        "environment": _environment(), "totals": totals, "frames": frames,
        "latency_ms": {
            "num_samples": len(latencies), "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
            "scope": "sampler with diagnostics and metadata; excludes loading, hashing, plotting and JSON writing",
        },
    }
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for filename, *example in examples:
        _save_example(output_path / filename, *example)
    (output_path / "audit.json").write_text(
        json.dumps(report, indent=2, default=_json_default, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--warmup", type=int, default=3, help="Untimed sampler calls before resetting the measured seed")
    parser.add_argument("--max-examples", type=int, default=3, help="Number of measured scenes to plot; 0 disables plots")
    args = parser.parse_args(argv)
    report = run_audit(args.config, args.frames, args.seed, args.output_dir, warmup=args.warmup, max_examples=args.max_examples)
    print(f"Audited {len(report['frames'])} frames; report: {args.output_dir / 'audit.json'}")


if __name__ == "__main__":
    main()
