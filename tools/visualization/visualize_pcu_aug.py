"""Visualization tool for PCU-Aug (Pillar 5 - Physics-Consistent 3D Augmentation).

Compares original LiDAR scenes with PCU-Aug augmented scenes in Bird's-Eye-View (BEV),
highlighting inserted objects, orientation headings, and ray-consistent shadow occlusions.
"""

import argparse
import os
from pathlib import Path
import random
import sys
from typing import Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "detector") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "detector"))
if str(REPO_ROOT / "detector" / "core" / "datasets") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "detector" / "core" / "datasets"))

from core.datasets.utils_1.collision_ground import get_box_2d_polygon
from core.datasets.utils_1.gt_sampler import GTSampler


CLASS_COLORS = {
    "Car": "#00FF66",        # Neon green
    "Pedestrian": "#FFCC00", # Warm yellow
    "Cyclist": "#00FFFF",    # Bright cyan
}
CLASS_MAP_INV = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}


def load_kitti_frame(
    data_dir: Union[str, Path], frame_id: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Loads raw point cloud and 8-col label boxes for a given KITTI frame ID."""
    data_path = Path(data_dir)
    # Check pointcloud layout
    if (data_path / "pointcloud").is_dir():
        pc_file = data_path / "pointcloud" / f"{frame_id}.bin"
        lbl_file = data_path / "label" / f"{frame_id}.txt"
    elif (data_path / "training" / "pointcloud").is_dir():
        pc_file = data_path / "training" / "pointcloud" / f"{frame_id}.bin"
        lbl_file = data_path / "training" / "label" / f"{frame_id}.txt"
    else:
        raise FileNotFoundError(f"Could not find pointcloud folder in {data_dir}")

    if not pc_file.is_file():
        raise FileNotFoundError(f"Missing point cloud file: {pc_file}")

    points = np.fromfile(pc_file, dtype=np.float32).reshape(-1, 4)

    boxes = []
    class_map = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
    if lbl_file.is_file():
        with open(lbl_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts or parts[0] not in class_map:
                    continue
                cls_id = class_map[parts[0]]
                h, w, l, x, y, z, yaw = map(float, parts[1:8])
                boxes.append([cls_id, h, w, l, x, y, z, yaw])

    boxes_arr = (
        np.array(boxes, dtype=np.float32)
        if len(boxes) > 0
        else np.zeros((0, 8), dtype=np.float32)
    )
    return points, boxes_arr


def draw_bev_box(
    ax,
    box: np.ndarray,
    color: str = "#00FF66",
    label_prefix: str = "",
    linestyle: str = "-",
    linewidth: float = 1.8,
    fill_alpha: float = 0.15,
):
    """Draws an oriented 3D bounding box projected on BEV (X forward, Y left)."""
    b = box[1:] if len(box) >= 8 else box
    cls_id = int(box[0]) if len(box) >= 8 else 0
    cls_name = CLASS_MAP_INV.get(cls_id, "Car")

    poly = get_box_2d_polygon(box, margin=0.0)
    xs, ys = poly.exterior.xy
    # Note: in BEV plot, X is forward (up or right). Here X is horizontal, Y is vertical.
    # We plot X on horizontal axis (0 to 70m) and Y on vertical axis (-40 to 40m).
    patch = MplPolygon(
        list(zip(xs, ys)),
        closed=True,
        edgecolor=color,
        facecolor=color,
        alpha=fill_alpha,
        linestyle=linestyle,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    # Draw outline with full opacity
    ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=linewidth)

    # Heading arrow from box center (bx, by) along (cos(yaw), sin(yaw))
    bx, by = b[3], b[4]
    yaw = b[6]
    arrow_len = min(b[2] * 0.45, 1.8)  # length along forward heading
    dx = arrow_len * np.cos(yaw)
    dy = arrow_len * np.sin(yaw)
    ax.arrow(
        bx,
        by,
        dx,
        dy,
        head_width=0.6,
        head_length=0.5,
        fc=color,
        ec=color,
        length_includes_head=True,
    )

    # Text annotation
    tag = f"{label_prefix}{cls_name}" if label_prefix else cls_name
    ax.text(
        bx,
        by,
        tag,
        color="white",
        fontsize=8,
        fontweight="bold",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.65, edgecolor="none"),
    )


def plot_pcu_augmentation_sample(
    data_dir: Union[str, Path],
    gt_database_path: Union[str, Path],
    frame_id: Optional[str] = None,
    sample_counts: Optional[Dict[str, int]] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (22, 10),
    pcu_config: Optional[Dict] = None,
) -> plt.Figure:
    """Generates a high-contrast side-by-side BEV comparison plot of PCU-Aug.

    Left Panel: Original LiDAR scene and ground truth annotations.
    Right Panel: Augmented scene with inserted 3D objects, ray-shadow occlusions, and annotations.
    """
    data_path = Path(data_dir)
    # Discover available frame IDs if not provided
    pc_folder = (
        data_path / "pointcloud"
        if (data_path / "pointcloud").is_dir()
        else data_path / "training" / "pointcloud"
    )
    available_frames = sorted([f.stem for f in pc_folder.glob("*.bin")])
    if not available_frames:
        raise FileNotFoundError(f"No .bin point clouds found in {pc_folder}")

    if frame_id is None:
        frame_id = random.choice(available_frames)

    orig_points, orig_boxes = load_kitti_frame(data_dir, frame_id)

    # A supplied config uses the same effective sampler settings as training.
    counts = {"Pedestrian": 6, "Cyclist": 5, "Car": 3} if sample_counts is None else sample_counts
    if pcu_config is None:
        sampler = GTSampler(database_path=str(gt_database_path), sample_counts=counts,
                            p=1.0, enable_physics=True)
    else:
        from core.datasets.utils_1.gt_sampler import build_gt_sampler
        settings = dict(pcu_config, gt_database_path=str(gt_database_path))
        sampler = build_gt_sampler(settings)

    aug_points, final_boxes, meta = sampler(
        orig_points.copy(), orig_boxes.copy(), return_metadata=True
    )

    # Identify inserted points: points inside inserted objects
    inserted_pts_list = meta.get("inserted_points", [])
    inserted_boxes = meta.get("inserted_boxes", [])

    # Style: Dark sleek theme
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, facecolor="#0e1117")

    for ax in (ax1, ax2):
        ax.set_facecolor("#161b22")
        ax.set_xlim(0, 70)
        ax.set_ylim(-35, 35)
        ax.set_xlabel("X (Forward, meters)", fontsize=12, color="#e6edf3")
        ax.set_ylabel("Y (Lateral Left/Right, meters)", fontsize=12, color="#e6edf3")
        ax.grid(True, linestyle="--", alpha=0.25, color="#8b949e")
        ax.set_aspect("equal", adjustable="box")
        # Draw ego vehicle indicator at (0, 0)
        ax.plot([0, 2.5, 0, 0], [-1.0, 0, 1.0, -1.0], color="#58a6ff", linewidth=2)
        ax.text(0, -2.5, "EGO", color="#58a6ff", fontsize=9, ha="center", fontweight="bold")

    # --- LEFT PANEL: Original Scene ---
    ax1.scatter(
        orig_points[:, 0],
        orig_points[:, 1],
        s=1.2,
        color="#8b949e",
        alpha=0.6,
        label=f"LiDAR Points ({len(orig_points):,})",
    )
    for b in orig_boxes:
        draw_bev_box(ax1, b, color="#2ecc71", label_prefix="")

    ax1.set_title(
        f"Original Scene [Frame: {frame_id}]\n"
        f"Objects: {len(orig_boxes)} | Points: {len(orig_points):,}",
        fontsize=14,
        color="#ffffff",
        pad=12,
        fontweight="bold",
    )
    ax1.legend(loc="upper right", framealpha=0.4, facecolor="#21262d")

    # --- RIGHT PANEL: PCU-Aug Augmented Scene ---
    # Find points in original cloud that were removed by shadow masking
    # (Fast distance approximation or KDTree check)
    aug_xy = aug_points[:, :2]
    # Retained background points
    ax2.scatter(
        aug_points[:, 0],
        aug_points[:, 1],
        s=1.2,
        color="#8b949e",
        alpha=0.4,
        label=f"Retained Background Points",
    )

    # Plot inserted object points in vivid glowing cyan
    if inserted_pts_list:
        all_inserted_pts = np.vstack(inserted_pts_list)
        ax2.scatter(
            all_inserted_pts[:, 0],
            all_inserted_pts[:, 1],
            s=2.5,
            color="#00FFFF",
            alpha=0.9,
            label=f"Inserted Object Points ({len(all_inserted_pts):,})",
        )

    # Draw original boxes (Green)
    for b in orig_boxes:
        draw_bev_box(ax2, b, color="#2ecc71", label_prefix="")

    # Draw newly inserted boxes (Magenta / Crimson)
    for idx, b in enumerate(inserted_boxes, start=1):
        draw_bev_box(
            ax2,
            b,
            color="#FF007F",
            label_prefix="PASTED ",
            linestyle="--",
            linewidth=2.2,
            fill_alpha=0.25,
        )

    num_inserted = len(inserted_boxes)
    ax2.set_title(
        f"Configured GT Sampling\n"
        f"Original: {len(orig_boxes)} + Pasted: {num_inserted} (Inserted: {num_inserted}) = Total {len(final_boxes)} objects",
        fontsize=14,
        color="#FF007F",
        pad=12,
        fontweight="bold",
    )
    ax2.legend(loc="upper right", framealpha=0.4, facecolor="#21262d")

    fig.suptitle(
        f"Configured GT Sampling Preview\n"
        f"Before global augmentation and BEV encoding",
        fontsize=16,
        color="#58a6ff",
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_p, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved PCU-Aug visualization to: {out_p}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visualize PCU-Aug 3D Ground Truth Augmentation in BEV."
    )
    parser.add_argument(
        "--data-dir",
        "--data_dir",
        dest="data_dir",
        type=str,
        default="data/kitti/processed",
        help="Path to processed KITTI dataset.",
    )
    parser.add_argument(
        "--gt-database",
        "--gt_database",
        dest="gt_database",
        type=str,
        default="data/kitti/kitti_gt_database.pkl",
        help="Path to kitti_gt_database.pkl.",
    )
    parser.add_argument(
        "--frame-id",
        "--frame_id",
        "--frame",
        dest="frame_id",
        type=str,
        default=None,
        help="Specific frame ID to visualize (e.g. 000001). If omitted, picks randomly.",
    )
    parser.add_argument(
        "--cars",
        "--car",
        dest="cars",
        type=int,
        default=3,
        help="Number of cars to sample.",
    )
    parser.add_argument(
        "--pedestrians",
        "--pedestrian",
        "--peds",
        dest="pedestrians",
        type=int,
        default=6,
        help="Number of pedestrians to sample.",
    )
    parser.add_argument(
        "--cyclists",
        "--cyclist",
        "--cycs",
        dest="cyclists",
        type=int,
        default=5,
        help="Number of cyclists to sample.",
    )
    parser.add_argument(
        "--output",
        "-o",
        dest="output",
        type=str,
        default="outputs/pcu_augmentation_visualization.png",
        help="Output image path for saving visualization.",
    )
    args = parser.parse_args()

    counts = {
        "Car": args.cars,
        "Pedestrian": args.pedestrians,
        "Cyclist": args.cyclists,
    }
    plot_pcu_augmentation_sample(
        data_dir=args.data_dir,
        gt_database_path=args.gt_database,
        frame_id=args.frame_id,
        sample_counts=counts,
        output_path=args.output,
    )
    print("Done generating visualization!")


if __name__ == "__main__":
    main()
