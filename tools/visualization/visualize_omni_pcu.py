"""Bird's-Eye-View (BEV) Visualization Tool for Omni-PCU Augmentation.

Compares raw LiDAR scenes with Omni-PCU augmented scenes, highlighting:
1. 3-tier stratified range placement (25m, 45m, 65m range rings)
2. Original vs newly inserted GT boxes (color-coded with heading vectors)
3. Physical density subsampling & anti-wall collision placement
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import numpy as np

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(REPO_ROOT),
    str(REPO_ROOT / "detector"),
    str(REPO_ROOT / "detector/core/datasets"),
    str(REPO_ROOT / "tools/kitti_training_pipeline"),
]

from core.datasets.augmentor.omni_geometry import boxes_to_bev_corners
from core.datasets.augmentor.omni_sampler import OmniDataBaseSampler


CLASS_COLORS = {
    0: {"name": "Car", "orig": "#00E5FF", "aug": "#FF1744"},         # Cyan vs Neon Red
    1: {"name": "Pedestrian", "orig": "#76FF03", "aug": "#FF9100"},  # Light Green vs Amber
    2: {"name": "Cyclist", "orig": "#E040FB", "aug": "#FFFF00"},     # Magenta vs Yellow
}
CLASS_MAP_INV = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}


def load_kitti_frame(
    data_dir: Path | str,
    frame_id: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Load LiDAR point cloud, bounding boxes, and optional road plane."""
    path = Path(data_dir)
    # Check directory structure
    for sub in ("", "training"):
        base = path / sub if sub else path
        pc_path = base / "pointcloud" / f"{frame_id}.bin"
        if pc_path.is_file():
            lbl_path = base / "label" / f"{frame_id}.txt"
            plane_path = base / "planes" / f"{frame_id}.txt"
            break
    else:
        raise FileNotFoundError(f"Could not find pointcloud for frame {frame_id} in {data_dir}")

    # Load point cloud
    points = np.fromfile(pc_path, dtype=np.float32).reshape(-1, 4)

    # Load boxes [class_id, h, w, l, x, y, z_bottom, yaw]
    boxes = []
    class_map = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
    if lbl_path.is_file():
        with lbl_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts or parts[0] not in class_map:
                    continue
                cls_id = class_map[parts[0]]
                # KITTI label format in processed directory:
                # name h w l x y z_bottom yaw
                try:
                    vals = [float(x) for x in parts[1:8]]
                    boxes.append([cls_id] + vals)
                except (ValueError, IndexError):
                    continue

    boxes_arr = np.array(boxes, dtype=np.float32) if boxes else np.empty((0, 8), dtype=np.float32)

    # Load road plane if present
    plane = None
    if plane_path.is_file():
        try:
            with plane_path.open("r", encoding="utf-8") as f:
                parts = f.read().strip().split()
                nums = [float(x) for x in parts if x.replace(".", "", 1).replace("-", "", 1).isdigit()]
                if len(nums) >= 4:
                    plane = np.array(nums[:4], dtype=np.float64)
        except Exception:
            plane = None

    return points, boxes_arr, plane


def draw_bev_scene(
    ax: plt.Axes,
    points: np.ndarray,
    boxes: np.ndarray,
    orig_box_count: int,
    title: str,
    x_range: Tuple[float, float] = (0.0, 70.0),
    y_range: Tuple[float, float] = (-35.0, 35.0),
) -> None:
    """Render a LiDAR point cloud and 3D bounding boxes in BEV coordinates."""
    ax.set_facecolor("#0F141C")  # Dark tech background

    # 1. Filter points within display range for fast rendering
    mask = (
        (points[:, 0] >= x_range[0])
        & (points[:, 0] <= x_range[1])
        & (points[:, 1] >= y_range[0])
        & (points[:, 1] <= y_range[1])
    )
    pts = points[mask]

    # 2. Scatter plot LiDAR points with height (Z) colormap
    z_vals = pts[:, 2]
    scatter = ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=0.25,
        c=z_vals,
        cmap="Blues_r",
        vmin=-2.5,
        vmax=1.0,
        alpha=0.6,
        rasterized=True,
    )

    # 3. Draw 3-tier Range Rings (PCU Stratified Range boundaries)
    theta = np.linspace(-np.pi / 4, np.pi / 4, 100)
    for radius, label in [(25.0, "25m (Tier 1)"), (45.0, "45m (Tier 2)"), (65.5, "65.5m (Tier 3)")]:
        arc_x = radius * np.cos(theta)
        arc_y = radius * np.sin(theta)
        ax.plot(arc_x, arc_y, color="#37474F", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.text(
            arc_x[-1] + 0.5,
            arc_y[-1],
            label,
            color="#78909C",
            fontsize=7,
            verticalalignment="center",
        )

    # 4. Draw Sensor Origin
    ax.plot([0], [0], marker="^", markersize=6, color="#FF5252", label="Ego LiDAR")

    # 5. Draw Bounding Boxes with heading arrows
    if len(boxes) > 0:
        corners = boxes_to_bev_corners(boxes)  # (N, 4, 2)
        for i, (b, poly_pts) in enumerate(zip(boxes, corners)):
            cls_id = int(b[0])
            is_inserted = i >= orig_box_count

            cfg = CLASS_COLORS.get(cls_id, {"name": "Obj", "orig": "#00E5FF", "aug": "#FF1744"})
            color = cfg["aug"] if is_inserted else cfg["orig"]
            linestyle = "-" if not is_inserted else "-"
            edge_width = 1.6 if is_inserted else 1.2

            # Box polygon
            polygon = MplPolygon(
                poly_pts,
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=edge_width,
                linestyle=linestyle,
                alpha=0.95,
            )
            ax.add_patch(polygon)

            # Heading direction vector (yaw)
            x, y, yaw, l = b[4], b[5], b[7], b[3]
            arrow_len = max(1.2, l * 0.45)
            dx = arrow_len * np.cos(yaw)
            dy = arrow_len * np.sin(yaw)
            ax.arrow(
                x,
                y,
                dx,
                dy,
                head_width=0.7,
                head_length=0.5,
                fc=color,
                ec=color,
                alpha=0.9,
                linewidth=0.8,
            )

            # Label text
            tag = f"[+{cfg['name']}]" if is_inserted else cfg["name"]
            ax.text(
                x,
                y - 0.9,
                tag,
                color=color,
                fontsize=7.5,
                weight="bold" if is_inserted else "normal",
                ha="center",
            )

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_xlabel("X (Forward, meters)", color="#ECEFF1", fontsize=9)
    ax.set_ylabel("Y (Lateral, meters)", color="#ECEFF1", fontsize=9)
    ax.set_title(title, color="#ECEFF1", fontsize=11, weight="bold", pad=10)
    ax.tick_params(colors="#90A4AE", labelsize=8)
    ax.grid(color="#263238", linestyle=":", linewidth=0.5, alpha=0.6)


def visualize_omni_pcu_comparison(
    points_orig: np.ndarray,
    boxes_orig: np.ndarray,
    points_aug: np.ndarray,
    boxes_aug: np.ndarray,
    frame_id: str = "sample",
    output_path: Optional[Path | str] = None,
    show: bool = False,
) -> plt.Figure:
    """Create side-by-side comparative BEV visualization figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9), facecolor="#0A0E14")

    inserted_count = max(0, len(boxes_aug) - len(boxes_orig))

    title_orig = f"Original Scene (Frame {frame_id})\nPoints: {len(points_orig):,} | GT Objects: {len(boxes_orig)}"
    title_aug = (
        f"Omni-PCU Augmented Scene (Frame {frame_id})\n"
        f"Points: {len(points_aug):,} | Objects: {len(boxes_aug)} "
        f"(+{inserted_count} physics-consistent inserted)"
    )

    draw_bev_scene(ax1, points_orig, boxes_orig, len(boxes_orig), title_orig)
    draw_bev_scene(ax2, points_aug, boxes_aug, len(boxes_orig), title_aug)

    # Global legend
    fig.text(
        0.5,
        0.02,
        "Cyan: Original Car  |  Light Green: Original Ped  |  Magenta: Original Cyc  |  "
        "Red / Orange / Yellow: Inserted Omni-PCU Objects  |  Dashed Rings: 25m/45m/65m Range Tiers",
        ha="center",
        color="#ECEFF1",
        fontsize=9,
        weight="medium",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved visualization to: {out.resolve()}")

    if show:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize Omni-PCU 3D LiDAR Augmentation")
    parser.add_argument("--data-dir", type=str, default="data/kitti/processed", help="Path to processed KITTI data")
    parser.add_argument("--frame-id", type=str, default=None, help="KITTI frame ID (e.g. 000008). Default: random")
    parser.add_argument("--config", type=str, default="configs/augmentation/omni_pcu_gt.json", help="Config file")
    parser.add_argument("--output", type=str, default="omni_pcu_vis.png", help="Output PNG path")
    parser.add_argument("--no-show", action="store_true", help="Do not invoke plt.show()")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        cfg_path = REPO_ROOT / args.config

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Determine frame id
    if args.frame_id is None:
        bin_files = list((data_dir / "pointcloud").glob("*.bin"))
        if not bin_files:
            bin_files = list((data_dir / "training" / "pointcloud").glob("*.bin"))
        if not bin_files:
            print(f"Error: No .bin files found in {data_dir}/pointcloud")
            sys.exit(1)
        frame_id = random.choice(bin_files).stem
    else:
        frame_id = args.frame_id

    print(f"Loading Frame {frame_id} from {data_dir}...")
    points, boxes, road_plane = load_kitti_frame(data_dir, frame_id)

    # Initialize Omni-PCU sampler
    aug_item = cfg["augmentation"]["AUG_CONFIG_LIST"][0]
    class_names = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
    sampler = OmniDataBaseSampler(data_dir, aug_item, class_names)

    # Execute sampling
    aug_points, aug_boxes = sampler(points.copy(), boxes.copy(), road_plane=road_plane)

    visualize_omni_pcu_comparison(
        points,
        boxes,
        aug_points,
        aug_boxes,
        frame_id=frame_id,
        output_path=args.output,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
