#!/usr/bin/env python3
"""Create train-only object point crops from processed KITTI detector labels."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import sys

import numpy as np

from common import write_json
from prepare_kitti import read_ids

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "detector"))
from core.datasets.augmentor.geometry import points_in_box

CLASSES = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}


def build_database(processed_root, train_manifest, output_dir, *,
                   class_names=None, val_manifest=None, min_points=1):
    processed_root, train_manifest, output_dir = map(Path, (processed_root, train_manifest, output_dir))
    classes = dict(CLASSES if class_names is None else class_names)
    if min_points < 1:
        raise ValueError("min_points must be positive")
    identifiers = read_ids(train_manifest)
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("Training manifest must contain unique nonempty frame IDs")
    if val_manifest is not None and set(identifiers) & set(read_ids(Path(val_manifest))):
        raise ValueError("Train/validation overlap in GT database manifests")
    # Resolve every input before writing the first crop.
    for identifier in identifiers:
        if not identifier.isdigit():
            raise ValueError(f"Invalid KITTI frame ID: {identifier!r}")
        for folder, suffix in [("pointcloud", ".bin"), ("label", ".txt")]:
            path = processed_root / folder / (identifier + suffix)
            if not path.is_file():
                raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    db_infos = {name: [] for name in classes}
    for identifier in identifiers:
        points = np.fromfile(processed_root / "pointcloud" / f"{identifier}.bin", dtype=np.float32)
        if points.size % 4 or not np.isfinite(points).all():
            raise ValueError(f"Malformed point cloud: {identifier}")
        points = points.reshape(-1, 4)
        label_path = processed_root / "label" / f"{identifier}.txt"
        for index, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
            fields = line.split()
            if not fields or fields[0] not in classes:
                continue
            if len(fields) != 8:
                raise ValueError(f"Malformed detector label: {label_path}:{index + 1}")
            name = fields[0]
            box = np.array([classes[name], *map(float, fields[1:])], dtype=np.float32)
            if not np.isfinite(box).all() or (box[1:4] <= 0).any():
                raise ValueError(f"Invalid box: {label_path}:{index + 1}")
            local = points[points_in_box(points, box)].copy()
            if len(local) < min_points:
                continue
            height, width, length, x, y, z_bottom, yaw = map(float, box[1:])
            z_center = z_bottom + height/2
            local[:, :3] -= np.array([x, y, z_center], dtype=np.float32)
            point_path = output_dir / f"{identifier}_{name}_{index}.bin"
            local.tofile(point_path)
            # Store paths relative to processed root so a database can move with
            # the processed dataset; allow an explicitly external output folder.
            relative_path = Path(os.path.relpath(point_path.resolve(), processed_root.resolve())).as_posix()
            db_infos[name].append({
                "name": name, "path": relative_path, "image_idx": identifier,
                "gt_idx": index, "box3d_lidar": [x, y, z_center, length, width, height, yaw],
                "num_points_in_gt": len(local),
            })
    destination = output_dir / "dbinfos_train.json"
    write_json(destination, {
        "format": "lidar_gt_database_v1", "num_point_features": 4,
        "box_format": "x_y_z_center_length_width_height_yaw",
        "point_coordinates": "relative_to_box_center_without_yaw_rotation",
        "source_manifest_sha256": hashlib.sha256(train_manifest.read_bytes()).hexdigest(),
        "source_frame_ids": identifiers, "db_infos": db_infos,
        "counts": {name: len(entries) for name, entries in db_infos.items()},
    })
    return destination


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--processed-root", required=True, type=Path)
    value.add_argument("--train-split", type=Path, default=ROOT / "splits/kitti/train.txt")
    value.add_argument("--val-split", type=Path, default=ROOT / "splits/kitti/val.txt")
    value.add_argument("--output-dir", type=Path)
    value.add_argument("--min-points", type=int, default=1)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    destination = build_database(
        args.processed_root, args.train_split,
        args.output_dir or args.processed_root / "gt_database",
        val_manifest=args.val_split, min_points=args.min_points,
    )
    print(f"GT database: {destination.resolve()}", flush=True)


if __name__ == "__main__":
    main()
