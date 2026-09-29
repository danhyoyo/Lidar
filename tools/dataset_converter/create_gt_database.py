import argparse
import os
from pathlib import Path
import pickle
from typing import Any, Dict, List
import numpy as np


def extract_object_points(points: np.ndarray, box: np.ndarray) -> np.ndarray:
    """Extracts LiDAR points lying strictly inside 3D oriented bounding box.

    box: [h, w, l, x, y, z, yaw] where z is BOTTOM center.
    """
    if len(points) == 0:
        return np.zeros((0, points.shape[1]), dtype=np.float32)

    b = box[1:] if len(box) >= 8 else box
    h, w, l, bx, by, bz, yaw = b[:7]

    pts_trans = points[:, :3] - np.array([bx, by, bz])
    cos_y = np.cos(-yaw)
    sin_y = np.sin(-yaw)

    x_rot = pts_trans[:, 0] * cos_y - pts_trans[:, 1] * sin_y
    y_rot = pts_trans[:, 0] * sin_y + pts_trans[:, 1] * cos_y
    z_rot = pts_trans[:, 2]

    # In MobilePIXOR, z is bottom center -> z_rot in [0, h]
    inside = (
        (np.abs(x_rot) <= l / 2.0)
        & (np.abs(y_rot) <= w / 2.0)
        & (z_rot >= 0.0)
        & (z_rot <= h)
    )
    return points[inside]


def build_kitti_gt_database(
    processed_dir: str,
    train_ids_file: str,
    output_file: str,
    min_points: int = 5,
):
    """Extracts objects directly from data/kitti/processed layout:

    training/pointcloud/{id}.bin and training/label/{id}.txt
    """
    processed_path = Path(processed_dir)
    pointcloud_dir = processed_path / "training" / "pointcloud"
    label_dir = processed_path / "training" / "label"

    with open(train_ids_file, "r", encoding="utf-8") as f:
        identifiers = [line.strip() for line in f if line.strip()]

    database: Dict[str, List[Dict[str, Any]]] = {
        "Car": [],
        "Pedestrian": [],
        "Cyclist": [],
    }
    class_map = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}

    for identifier in identifiers:
        bin_path = pointcloud_dir / f"{identifier}.bin"
        txt_path = label_dir / f"{identifier}.txt"
        if not bin_path.exists() or not txt_path.exists():
            continue

        points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)

        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts or parts[0] not in class_map:
                    continue
                cls_name = parts[0]
                cls_id = class_map[cls_name]
                h, w, l, x, y, z, yaw = map(float, parts[1:8])
                box_8 = np.array(
                    [cls_id, h, w, l, x, y, z, yaw], dtype=np.float32
                )

                obj_pts = extract_object_points(points, box_8)
                if len(obj_pts) < min_points:
                    continue

                # Canonical points relative to bottom center:
                # 1. Translate to bottom center (x, y, z)
                # 2. Rotate by -yaw so canonical box heading is along +x (yaw = 0)
                pts_trans = obj_pts[:, :3] - np.array([x, y, z])
                cos_y = np.cos(-yaw)
                sin_y = np.sin(-yaw)
                x_canonical = pts_trans[:, 0] * cos_y - pts_trans[:, 1] * sin_y
                y_canonical = pts_trans[:, 0] * sin_y + pts_trans[:, 1] * cos_y
                z_canonical = pts_trans[:, 2]

                canonical_pts = np.hstack(
                    [
                        np.stack(
                            [x_canonical, y_canonical, z_canonical], axis=1
                        ),
                        obj_pts[:, 3:],
                    ]
                ).astype(np.float32)

                database[cls_name].append(
                    {
                        "box": box_8,
                        "points": canonical_pts,
                        "r_origin": float(np.sqrt(x**2 + y**2)),
                        "num_points": len(obj_pts),
                    }
                )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(database, f)
    total_samples = sum(len(v) for v in database.values())
    print(f"Saved GT database to {output_file} with {total_samples} objects.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract Ground Truth objects from KITTI processed training set."
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/kitti/processed",
        help="Path to processed KITTI dataset containing training/pointcloud and training/label",
    )
    parser.add_argument(
        "--train-ids",
        type=str,
        default="splits/kitti/train.txt",
        help="Path to train IDs split file",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/kitti/kitti_gt_database.pkl",
        help="Output pickle file for GT database",
    )
    parser.add_argument(
        "--min-points",
        type=int,
        default=5,
        help="Minimum number of points inside box to include object",
    )
    args = parser.parse_args()

    build_kitti_gt_database(
        args.processed_dir, args.train_ids, args.output_file, args.min_points
    )


if __name__ == "__main__":
    main()
