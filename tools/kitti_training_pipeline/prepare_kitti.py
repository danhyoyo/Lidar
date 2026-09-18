#!/usr/bin/env python3
"""Convert official KITTI object data to the format used by this detector."""

from __future__ import annotations

import argparse
import math
import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from common import write_json


ACTIVE_CLASSES = ("Car", "Pedestrian", "Cyclist")


def multiply_3x3_vector(matrix: np.ndarray, vector) -> np.ndarray:
    """Multiply a small rotation matrix and vector without invoking BLAS."""
    return np.sum(matrix * np.asarray(vector)[None, :], axis=1)


def invert_affine_transform(transform: np.ndarray) -> np.ndarray:
    """Invert the 4x4 KITTI affine calibration without BLAS."""
    matrix = transform[:3, :3]
    cross0 = np.cross(matrix[1], matrix[2])
    cross1 = np.cross(matrix[2], matrix[0])
    cross2 = np.cross(matrix[0], matrix[1])
    determinant = float(np.sum(matrix[0] * cross0))
    if not math.isfinite(determinant) or abs(determinant) < 1e-12:
        raise ValueError("KITTI calibration transform is singular")
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = np.column_stack((cross0, cross1, cross2)) / determinant
    inverse[:3, 3] = -multiply_3x3_vector(
        inverse[:3, :3], transform[:3, 3])
    return inverse


def parse_calibration(path: Path) -> np.ndarray:
    values: Dict[str, np.ndarray] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in raw_line:
            continue
        name, payload = raw_line.split(":", 1)
        if payload.strip():
            values[name] = np.fromstring(payload, sep=" ", dtype=np.float64)
    rect_key = "R0_rect" if "R0_rect" in values else "R_rect"
    transform_key = "Tr_velo_to_cam" if "Tr_velo_to_cam" in values else "Tr_velo_cam"
    if rect_key not in values or transform_key not in values:
        raise ValueError(f"Missing R0_rect/Tr_velo_to_cam in {path}")
    rectification = values[rect_key].reshape(3, 3)
    velo_to_camera = values[transform_key].reshape(3, 4)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.sum(
        rectification[:, :, None] * velo_to_camera[None, :, :3], axis=1)
    result[:3, 3] = multiply_3x3_vector(rectification, velo_to_camera[:, 3])
    return result


def normalize_yaw(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def convert_labels(label_path: Path, calibration_path: Path) -> Tuple[List[str], Counter]:
    velo_to_rect = parse_calibration(calibration_path)
    rect_to_velo = invert_affine_transform(velo_to_rect)
    rect_rotation_to_velo = rect_to_velo[:3, :3]
    converted: List[str] = []
    counts: Counter = Counter()
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if not fields or fields[0] not in ACTIVE_CLASSES:
            continue
        if len(fields) < 15:
            raise ValueError(f"Malformed KITTI label {label_path}:{line_number}")
        object_type = fields[0]
        height, width, length = map(float, fields[8:11])
        camera_x, camera_y, camera_z = map(float, fields[11:14])
        rotation_y = float(fields[14])
        bottom_center_velo = (
            multiply_3x3_vector(rect_rotation_to_velo,
                                (camera_x, camera_y, camera_z))
            + rect_to_velo[:3, 3]
        )
        # Transform the object's length direction exactly from rect-camera to Velodyne.
        heading_rect = (math.cos(rotation_y), 0.0, -math.sin(rotation_y))
        heading_velo = multiply_3x3_vector(rect_rotation_to_velo, heading_rect)
        yaw_velo = normalize_yaw(math.atan2(heading_velo[1], heading_velo[0]))
        x, y, z = bottom_center_velo[:3]
        converted.append(
            f"{object_type} {height:.6f} {width:.6f} {length:.6f} "
            f"{x:.6f} {y:.6f} {z:.6f} {yaw_velo:.8f}"
        )
        counts[object_type] += 1
    return converted, counts


def read_ids(path: Path) -> List[str]:
    identifiers: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value:
            identifiers.append(value.split(";", 1)[0])
    return identifiers


def write_manifest(path: Path, identifiers: Iterable[str]) -> None:
    path.write_text("".join(f"{identifier};kitti\n" for identifier in identifiers), encoding="utf-8")


def make_split(all_ids: Sequence[str], train_count: int, seed: int,
               train_ids_path: Path | None, val_ids_path: Path | None):
    available = set(all_ids)
    if train_ids_path or val_ids_path:
        if not train_ids_path or not val_ids_path:
            raise ValueError("Provide both --train-ids and --val-ids")
        train_ids, val_ids = read_ids(train_ids_path), read_ids(val_ids_path)
        source = "provided manifests"
    else:
        shuffled = list(all_ids)
        random.Random(seed).shuffle(shuffled)
        train_ids, val_ids = sorted(shuffled[:train_count]), sorted(shuffled[train_count:])
        source = f"deterministic random split, seed={seed}"
    train_set, val_set = set(train_ids), set(val_ids)
    if len(train_ids) != len(train_set) or len(val_ids) != len(val_set):
        raise ValueError("Duplicate frame IDs found in a split manifest")
    overlap = train_set & val_set
    if overlap:
        raise ValueError(f"Train/validation overlap: {sorted(overlap)[:5]}")
    unknown = (train_set | val_set) - available
    if unknown:
        raise ValueError(f"Manifest references missing frames: {sorted(unknown)[:5]}")
    if train_set | val_set != available:
        missing = available - (train_set | val_set)
        raise ValueError(f"Split does not cover all frames: {sorted(missing)[:5]}")
    return train_ids, val_ids, source


def materialize(source: Path, destination: Path, mode: str, overwrite: bool) -> None:
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            return
        destination.unlink()
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "hardlink":
        os.link(source, destination)
    else:
        shutil.copy2(source, destination)


def generated_config(processed_root: Path, train_manifest: Path, val_manifest: Path):
    return {
        "date": "reproducible", "ver": 1, "note": "kitti_mobilepixor_baseline",
        "multi_gpu": False, "device": "cuda", "seed": 42,
        "data": {
            "num_classes": 3, "out_size_factor": 4, "gaussian_overlap": 0.1, "min_radius": 4,
            "kitti": {
                "location": str(processed_root.resolve()),
                "geometry": {
                    "x_min": 0.0, "x_max": 70.4, "y_min": -40.0, "y_max": 40.0,
                    "z_min": -2.5, "z_max": 1.0, "x_res": 0.1, "y_res": 0.1, "z_res": 0.1
                },
                "objects": {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
            }
        },
        # This checkout has mobilepixor.py but no mobilepixor_triplet.py.
        "model": {"backbone": "mobilepixor", "backbone_out_dim": 16, "cls_encoding": "gaussian"},
        "train": {
            "physical_batch_size": 2, "accumulation_steps": 2,
            "learning_rate": 0.0003, "epochs": 100, "momentum": 0.9,
            "weight_decay": 0.0005, "lr_decay_at": [40, 80], "save_every": 5,
            "data": str(train_manifest.resolve())
        },
        "val": {"physical_batch_size": 2, "data": str(val_manifest.resolve()), "val_every": 1},
        "augmentation": {
            "p": 0.5,
            "rotation": {"use": True, "limit_angle": 20, "p": 1},
            "scaling": {"use": True, "range": [0.95, 1.05], "p": 1},
            "translation": {"use": True, "scale": 0.4, "p": 1}
        }
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kitti-root", required=True, type=Path,
                        help="Directory containing training/velodyne, training/calib and training/label_2")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config-output", required=True, type=Path)
    parser.add_argument("--train-ids", type=Path)
    parser.add_argument("--val-ids", type=Path)
    parser.add_argument("--train-count", type=int, default=5984)
    parser.add_argument("--expected-total", type=int, default=7481)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pointcloud-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    training_root = args.kitti_root.expanduser().resolve() / "training"
    velodyne_root, calibration_root = training_root / "velodyne", training_root / "calib"
    source_label_root = training_root / "label_2"
    for path in (velodyne_root, calibration_root, source_label_root):
        if not path.is_dir():
            raise FileNotFoundError(f"Required KITTI directory not found: {path}")
    all_ids = sorted(path.stem for path in velodyne_root.glob("*.bin"))
    if len(all_ids) != args.expected_total:
        raise ValueError(f"Expected {args.expected_total} point clouds, found {len(all_ids)}")
    for identifier in all_ids:
        if not (calibration_root / f"{identifier}.txt").is_file():
            raise FileNotFoundError(calibration_root / f"{identifier}.txt")
        if not (source_label_root / f"{identifier}.txt").is_file():
            raise FileNotFoundError(source_label_root / f"{identifier}.txt")

    output_root = args.output_root.expanduser().resolve()
    pointcloud_root, target_label_root = output_root / "pointcloud", output_root / "label"
    pointcloud_root.mkdir(parents=True, exist_ok=True)
    target_label_root.mkdir(parents=True, exist_ok=True)
    total_counts: Counter = Counter()
    for index, identifier in enumerate(all_ids, start=1):
        materialize(velodyne_root / f"{identifier}.bin",
                    pointcloud_root / f"{identifier}.bin", args.pointcloud_mode, args.overwrite)
        labels, counts = convert_labels(source_label_root / f"{identifier}.txt",
                                        calibration_root / f"{identifier}.txt")
        total_counts.update(counts)
        target_label = target_label_root / f"{identifier}.txt"
        if args.overwrite or not target_label.exists():
            target_label.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
        if index % 500 == 0 or index == len(all_ids):
            print(f"Converted {index}/{len(all_ids)} frames")

    train_ids, val_ids, split_source = make_split(
        all_ids, args.train_count, args.seed, args.train_ids, args.val_ids)
    train_manifest, val_manifest = output_root / "train.txt", output_root / "val.txt"
    write_manifest(train_manifest, train_ids)
    write_manifest(val_manifest, val_ids)
    write_json(output_root / "conversion_manifest.json", {
        "source": str(training_root), "frames": len(all_ids),
        "train_frames": len(train_ids), "validation_frames": len(val_ids),
        "split_source": split_source, "active_label_counts": dict(total_counts),
        "pointcloud_mode": args.pointcloud_mode
    })
    write_json(args.config_output, generated_config(output_root, train_manifest, val_manifest))
    print(f"Processed KITTI: {output_root}")
    print(f"Training config: {args.config_output.expanduser().resolve()}")
    print(f"Split: train={len(train_ids)}, validation={len(val_ids)} ({split_source})")


if __name__ == "__main__":
    main()
