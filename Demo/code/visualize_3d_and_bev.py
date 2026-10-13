#!/usr/bin/env python3
"""Render KITTI LiDAR frame in dual contexts: 3D Perspective and BEV Top-Down.

Generates:
1. kitti_{frame}_3d_perspective.png: Oblique 3D viewpoint with 3D cuboid boxes
2. kitti_{frame}_bev_topdown.png: Top-down Bird's Eye View with range rings & yaw vectors
3. kitti_{frame}_dual_composite.png: Multi-view composite with zooms of Car, Cyclist, Pedestrian
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
import numpy as np

# Ensure Code directory is on sys.path
ROOT = Path(__file__).resolve().parents[2] / "Code"
sys.path.insert(0, str(ROOT))
from tools.visualization.export_kitti_overview import (
    CLASSES, CLASS_COLORS, EDGES, camera_basis, canvas_limits,
    points_in_object, read_points, read_objects, roi_vertices,
)

CLASS_PALETTE = {
    "Car": "#ff4b4b",         # Bright coral red
    "Cyclist": "#ffb703",     # Amber gold
    "Pedestrian": "#06d6a0",  # Neon emerald green
}


def project_points(points: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.asarray(points)[:, :3] @ basis


def render_3d_perspective(
    points: np.ndarray,
    objects: list,
    output_path: Path,
    *,
    roi: list[float] = [-2, 45, -24, 24, -2.8, 4.0],
    elevation: float = 22.0,
    azimuth: float = -45.0,
    dpi: int = 200,
    figsize: tuple[int, int] = (16, 10),
    title_suffix: str = "",
):
    """Render oblique 3D perspective projection with 3D wireframe cuboids."""
    lower, upper = np.asarray(roi[::2]), np.asarray(roi[1::2])
    mask = ((points[:, :3] >= lower) & (points[:, :3] <= upper)).all(axis=1)
    pts = points[mask]

    basis = camera_basis(elevation, azimuth)
    uvz = project_points(pts, basis)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor="#0b0f19")
    ax.set_facecolor="#0b0f19"

    # Depth sort points (painter's algorithm)
    order = np.argsort(uvz[:, 2], kind="stable")
    # Color points by height Z (normalized)
    norm_z = np.clip((pts[:, 2] + 2.5) / 5.0, 0.0, 1.0)
    colors = plt.cm.turbo(norm_z)

    ax.scatter(
        uvz[order, 0],
        uvz[order, 1],
        s=1.2,
        c=colors[order],
        alpha=0.75,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    # Render 3D Bounding Boxes
    for idx, obj in enumerate(objects):
        color = CLASS_PALETTE.get(obj.name, "#ffffff")
        corners_uvz = project_points(obj.corners, basis)

        # 12 edges
        for a, b in EDGES:
            p1, p2 = corners_uvz[a, :2], corners_uvz[b, :2]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=1.8, alpha=0.95, zorder=3)

        # Front face cross (corners 1, 2, 5, 6 in camera coords map to front)
        # To indicate heading direction in 3D:
        heading_dir = obj.corners[1] - obj.corners[0]
        front_center = (obj.corners[1] + obj.corners[2] + obj.corners[5] + obj.corners[6]) / 4.0
        front_uv = project_points([front_center], basis)[0, :2]
        center_uv = project_points([obj.center], basis)[0, :2]
        ax.plot([center_uv[0], front_uv[0]], [center_uv[1], front_uv[1]], color=color, linewidth=2.4, zorder=4)

        # Tag label at top center of box
        top_center = obj.corners[4:].mean(axis=0)
        label_uv = project_points([top_center], basis)[0, :2]
        ax.text(
            label_uv[0],
            label_uv[1] + 0.35,
            f"{obj.name} #{idx+1}",
            color="white",
            fontsize=7.5,
            fontweight="bold",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, edgecolor="none", alpha=0.85),
            zorder=5,
        )

    # Set viewport limits based on ROI
    limits = canvas_limits(roi_vertices(roi), basis, figsize[0] / figsize[1], margin=0.03)
    ax.set_xlim(limits[0], limits[1])
    ax.set_ylim(limits[2], limits[3])
    ax.axis("off")

    # Legend & Info text
    car_count = sum(o.name == "Car" for o in objects)
    cyc_count = sum(o.name == "Cyclist" for o in objects)
    ped_count = sum(o.name == "Pedestrian" for o in objects)

    title_text = f"KITTI LiDAR Scan 005663 — 3D Perspective View (Elevation 22°, Azimuth -45°){title_suffix}"
    fig.text(0.04, 0.95, title_text, color="white", fontsize=15, fontweight="bold")
    fig.text(
        0.04, 0.92,
        f"21 3D Objects:  {car_count} Cars  |  {cyc_count} Cyclists  |  {ped_count} Pedestrians  •  Points in ROI: {len(pts):,} / {len(points):,} raw",
        color="#a0aec0", fontsize=11
    )

    # Add custom legend handles
    handles = [
        plt.Line2D([0], [0], color=CLASS_PALETTE["Car"], lw=3, label=f"Car ({car_count})"),
        plt.Line2D([0], [0], color=CLASS_PALETTE["Cyclist"], lw=3, label=f"Cyclist ({cyc_count})"),
        plt.Line2D([0], [0], color=CLASS_PALETTE["Pedestrian"], lw=3, label=f"Pedestrian ({ped_count})"),
    ]
    leg = ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.96),
        facecolor="#161e2e",
        edgecolor="#2e384d",
        labelcolor="white",
        fontsize=10,
        framealpha=0.9,
    )
    leg.get_frame().set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 3D Perspective view to: {output_path}")


def render_bev_topdown(
    points: np.ndarray,
    objects: list,
    output_path: Path,
    *,
    x_range: tuple[float, float] = (-2.0, 45.0),
    y_range: tuple[float, float] = (-22.0, 22.0),
    dpi: int = 200,
    figsize: tuple[int, int] = (16, 12),
    title_suffix: str = "",
):
    """Render top-down Bird's Eye View (BEV: X forward, Y left) with range rings."""
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) &
        (points[:, 2] >= -2.8) & (points[:, 2] <= 4.0)
    )
    pts = points[mask]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, facecolor="#0b0f19")
    ax.set_facecolor="#0b0f19"

    # In LiDAR coordinates: X forward, Y left.
    # To display conventionally with Ego vehicle at bottom (X pointing upwards, Y pointing left/right):
    # Or X horizontal, Y vertical. In autonomous driving BEV:
    # Standard: X horizontal (0 to 45m forward), Y vertical (-22 to +22m left/right).
    norm_z = np.clip((pts[:, 2] + 2.5) / 5.0, 0.0, 1.0)
    colors = plt.cm.turbo(norm_z)

    # Plot point cloud in (X, Y)
    ax.scatter(pts[:, 0], pts[:, 1], s=1.0, c=colors, alpha=0.7, linewidths=0, rasterized=True, zorder=2)

    # Polar Range Rings from Ego Sensor (0, 0)
    theta = np.linspace(-np.pi / 2, np.pi / 2, 200)
    for r in [10, 20, 30, 40]:
        rx = r * np.cos(theta)
        ry = r * np.sin(theta)
        valid = (rx >= x_range[0]) & (rx <= x_range[1]) & (ry >= y_range[0]) & (ry <= y_range[1])
        ax.plot(rx[valid], ry[valid], color="#334155", linestyle="--", linewidth=1.0, zorder=1)
        # Ring distance label
        if r <= x_range[1]:
            ax.text(r, 0.4, f"{r}m", color="#64748b", fontsize=9, fontweight="semibold", zorder=3)

    # Coordinate Grid lines
    ax.axhline(0, color="#334155", linestyle=":", linewidth=1.0, zorder=1)
    ax.axvline(0, color="#334155", linestyle=":", linewidth=1.0, zorder=1)

    for y_line in [-15, -10, -5, 5, 10, 15]:
        ax.axhline(y_line, color="#1e293b", linestyle=":", linewidth=0.8, zorder=1)

    # Ego vehicle marker at (0, 0)
    ax.plot([0, 1.8, 0, -1.8, 0], [0, 0.9, -0.9, 0, 0], color="#38bdf8", lw=2, zorder=5)
    ax.scatter([0], [0], color="#38bdf8", s=60, zorder=6)
    ax.text(-1.0, 0.0, "Ego LiDAR", color="#38bdf8", fontsize=9, fontweight="bold", ha="right", va="center", zorder=6)

    # Render Object Bounding Box Footprints
    for idx, obj in enumerate(objects):
        color = CLASS_PALETTE.get(obj.name, "#ffffff")
        # In KITTI, bottom 4 corners are indices 0, 1, 2, 3
        # Let's take the XY of the 4 bottom corners for BEV polygon footprint
        corners_xy = obj.corners[:4, :2]

        # Draw filled footprint polygon
        poly = Polygon(corners_xy, closed=True, facecolor=color, edgecolor=color, alpha=0.3, linewidth=2.0, zorder=4)
        ax.add_patch(poly)

        # Draw heading vector from center
        front_center = (obj.corners[1, :2] + obj.corners[2, :2]) / 2.0
        center = obj.center[:2]
        heading_vec = (front_center - center)
        heading_norm = np.linalg.norm(heading_vec)
        if heading_norm > 1e-3:
            arrow_end = center + (heading_vec / heading_norm) * max(heading_norm * 1.3, 1.2)
            ax.annotate(
                "",
                xy=(arrow_end[0], arrow_end[1]),
                xytext=(center[0], center[1]),
                arrowprops=dict(arrowstyle="->", color=color, lw=2.2, mutation_scale=12),
                zorder=5,
            )

        # Distance from ego
        dist = np.linalg.norm(center)
        ax.text(
            center[0],
            center[1] + (0.9 if obj.name == "Car" else 0.7),
            f"{obj.name} #{idx+1} ({dist:.1f}m)",
            color="white",
            fontsize=7.5,
            fontweight="bold",
            ha="center",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor=color, edgecolor="none", alpha=0.85),
            zorder=6,
        )

    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_xlabel("Longitudinal Distance X (metres forward)", color="#94a3b8", fontsize=11, labelpad=8)
    ax.set_ylabel("Lateral Distance Y (metres left/right)", color="#94a3b8", fontsize=11, labelpad=8)
    ax.tick_params(colors="#94a3b8", which="both")
    for spine in ax.spines.values():
        spine.set_color("#334155")

    car_count = sum(o.name == "Car" for o in objects)
    cyc_count = sum(o.name == "Cyclist" for o in objects)
    ped_count = sum(o.name == "Pedestrian" for o in objects)

    title_text = f"KITTI LiDAR Scan 005663 — BEV Top-Down View (Bird's Eye View){title_suffix}"
    fig.text(0.04, 0.95, title_text, color="white", fontsize=15, fontweight="bold")
    fig.text(
        0.04, 0.92,
        f"21 Ground-Truth 3D Footprints  |  Range Rings at 10m, 20m, 30m, 40m  •  Arrows indicate Yaw Heading",
        color="#a0aec0", fontsize=11
    )

    handles = [
        plt.Line2D([0], [0], color=CLASS_PALETTE["Car"], lw=3, label=f"Car ({car_count})"),
        plt.Line2D([0], [0], color=CLASS_PALETTE["Cyclist"], lw=3, label=f"Cyclist ({cyc_count})"),
        plt.Line2D([0], [0], color=CLASS_PALETTE["Pedestrian"], lw=3, label=f"Pedestrian ({ped_count})"),
    ]
    leg = ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.96),
        facecolor="#161e2e",
        edgecolor="#2e384d",
        labelcolor="white",
        fontsize=10,
        framealpha=0.9,
    )
    leg.get_frame().set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved BEV Top-down view to: {output_path}")


def render_dual_composite(
    points: np.ndarray,
    objects: list,
    output_path: Path,
    *,
    roi: list[float] = [-2, 45, -24, 24, -2.8, 4.0],
    dpi: int = 200,
):
    """Render comprehensive side-by-side Dual-view (3D Perspective + BEV) with zoom insets."""
    fig = plt.figure(figsize=(24, 14), dpi=dpi, facecolor="#080c14")

    # Grid: Top 2/3 for 3D and BEV, bottom 1/3 for 3 class zooms
    gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1.0], hspace=0.18, wspace=0.12, left=0.03, right=0.97, top=0.92, bottom=0.04)

    ax_3d = fig.add_subplot(gs[0, :2])
    ax_bev = fig.add_subplot(gs[0, 2])
    ax_car = fig.add_subplot(gs[1, 0])
    ax_cyc = fig.add_subplot(gs[1, 1])
    ax_ped = fig.add_subplot(gs[1, 2])

    for ax in [ax_3d, ax_bev, ax_car, ax_cyc, ax_ped]:
        ax.set_facecolor("#0b0f19")

    # Filter ROI points
    lower, upper = np.asarray(roi[::2]), np.asarray(roi[1::2])
    mask = ((points[:, :3] >= lower) & (points[:, :3] <= upper)).all(axis=1)
    pts = points[mask]
    norm_z = np.clip((pts[:, 2] + 2.5) / 5.0, 0.0, 1.0)
    colors = plt.cm.turbo(norm_z)

    # 1. 3D Perspective in ax_3d
    basis = camera_basis(22.0, -45.0)
    uvz = project_points(pts, basis)
    order = np.argsort(uvz[:, 2], kind="stable")
    ax_3d.scatter(uvz[order, 0], uvz[order, 1], s=1.1, c=colors[order], alpha=0.75, linewidths=0, rasterized=True, zorder=1)

    for idx, obj in enumerate(objects):
        color = CLASS_PALETTE.get(obj.name, "#ffffff")
        corners_uvz = project_points(obj.corners, basis)
        for a, b in EDGES:
            p1, p2 = corners_uvz[a, :2], corners_uvz[b, :2]
            ax_3d.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=1.7, alpha=0.9, zorder=3)
        top_center = obj.corners[4:].mean(axis=0)
        label_uv = project_points([top_center], basis)[0, :2]
        ax_3d.text(label_uv[0], label_uv[1] + 0.35, f"{obj.name} #{idx+1}", color="white", fontsize=6.5,
                   fontweight="bold", ha="center", va="bottom",
                   bbox=dict(boxstyle="round,pad=0.15", facecolor=color, edgecolor="none", alpha=0.85), zorder=5)

    lims = canvas_limits(roi_vertices(roi), basis, 2.0 / 1.1, margin=0.03)
    ax_3d.set_xlim(lims[0], lims[1])
    ax_3d.set_ylim(lims[2], lims[3])
    ax_3d.axis("off")
    ax_3d.set_title("CONTEXT 1: 3D PERSPECTIVE VIEW (Elev 22°, Azim -45°)", color="#38bdf8", fontsize=13, fontweight="bold", pad=10)

    # 2. BEV Top-down in ax_bev
    ax_bev.scatter(pts[:, 0], pts[:, 1], s=0.9, c=colors, alpha=0.7, linewidths=0, rasterized=True, zorder=2)
    theta = np.linspace(-np.pi / 2, np.pi / 2, 150)
    for r in [10, 20, 30, 40]:
        rx, ry = r * np.cos(theta), r * np.sin(theta)
        v = (rx >= -2) & (rx <= 45) & (ry >= -22) & (ry <= 22)
        ax_bev.plot(rx[v], ry[v], color="#334155", linestyle="--", linewidth=0.9, zorder=1)
        ax_bev.text(r, 0.4, f"{r}m", color="#64748b", fontsize=8, fontweight="semibold", zorder=3)

    ax_bev.axhline(0, color="#334155", linestyle=":", linewidth=0.8, zorder=1)
    ax_bev.axvline(0, color="#334155", linestyle=":", linewidth=0.8, zorder=1)
    ax_bev.plot([0, 1.8, 0, -1.8, 0], [0, 0.9, -0.9, 0, 0], color="#38bdf8", lw=1.8, zorder=5)

    for idx, obj in enumerate(objects):
        color = CLASS_PALETTE.get(obj.name, "#ffffff")
        corners_xy = obj.corners[:4, :2]
        poly = Polygon(corners_xy, closed=True, facecolor=color, edgecolor=color, alpha=0.35, linewidth=1.6, zorder=4)
        ax_bev.add_patch(poly)
        center = obj.center[:2]
        ax_bev.text(center[0], center[1], f"#{idx+1}", color="white", fontsize=6.5, fontweight="bold",
                    ha="center", va="center", zorder=6)

    ax_bev.set_xlim(-2, 45)
    ax_bev.set_ylim(-22, 22)
    ax_bev.set_xlabel("Forward X (m)", color="#94a3b8", fontsize=9)
    ax_bev.set_ylabel("Lateral Y (m)", color="#94a3b8", fontsize=9)
    ax_bev.tick_params(colors="#94a3b8", labelsize=8)
    for sp in ax_bev.spines.values():
        sp.set_color("#334155")
    ax_bev.set_title("CONTEXT 2: BEV TOP-DOWN (X forward, Y left)", color="#38bdf8", fontsize=13, fontweight="bold", pad=10)

    # 3. Zooms for Car, Cyclist, Pedestrian
    # Find richest representative of each class
    car_objs = [o for o in objects if o.name == "Car"]
    cyc_objs = [o for o in objects if o.name == "Cyclist"]
    ped_objs = [o for o in objects if o.name == "Pedestrian"]

    # Choose best representative objects with high interior points
    def get_zoom_data(obj_list):
        best_obj = None
        best_pts = None
        max_cnt = -1
        for o in obj_list:
            inside = points_in_object(points, o, padding=0.6)
            cnt = np.sum(inside)
            if cnt > max_cnt:
                max_cnt = cnt
                best_obj = o
                best_pts = points[inside]
        return best_obj, best_pts

    target_classes = [("Car", car_objs, ax_car, CLASS_PALETTE["Car"]),
                      ("Cyclist", cyc_objs, ax_cyc, CLASS_PALETTE["Cyclist"]),
                      ("Pedestrian", ped_objs, ax_ped, CLASS_PALETTE["Pedestrian"])]

    for cname, olist, ax_z, ccolor in target_classes:
        obj, zpts = get_zoom_data(olist)
        if obj is not None and zpts is not None and len(zpts) > 0:
            # Orient camera along heading
            heading = obj.corners[1] - obj.corners[0]
            azimuth = -45.0 + np.rad2deg(np.arctan2(heading[1], heading[0]))
            z_basis = camera_basis(20.0, azimuth)
            z_uvz = project_points(zpts, z_basis)
            z_order = np.argsort(z_uvz[:, 2], kind="stable")
            z_norm = np.clip((zpts[:, 2] - obj.center[2] + 1.0) / 2.0, 0.0, 1.0)
            ax_z.scatter(z_uvz[z_order, 0], z_uvz[z_order, 1], s=5.0, c=plt.cm.turbo(z_norm[z_order]),
                         alpha=0.9, linewidths=0, zorder=2)

            # Box corners
            c_uvz = project_points(obj.corners, z_basis)
            for a, b in EDGES:
                p1, p2 = c_uvz[a, :2], c_uvz[b, :2]
                ax_z.plot([p1[0], p2[0]], [p1[1], p2[1]], color=ccolor, linewidth=2.0, zorder=3)

            z_lims = canvas_limits(obj.corners, z_basis, 1.0, margin=0.25)
            ax_z.set_xlim(z_lims[0], z_lims[1])
            ax_z.set_ylim(z_lims[2], z_lims[3])
            ax_z.axis("off")
            ax_z.set_title(f"ZOOM: {cname} ({len(zpts)} points)", color=ccolor, fontsize=11, fontweight="bold", pad=6)

    # Master Header banner
    fig.text(0.03, 0.965, "KITTI LiDAR Scan 005663 — Dual Context Visualization (3D Perspective & BEV Top-Down)",
             color="white", fontsize=18, fontweight="bold")
    fig.text(0.03, 0.935,
             f"Total: 21 Ground-Truth 3D Objects (4 Cars, 4 Cyclists, 13 Pedestrians)  •  123,465 LiDAR points (66,203 in Front ROI)",
             color="#94a3b8", fontsize=12)

    plt.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Dual Composite view to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Render KITTI LiDAR in 3D Perspective and BEV.")
    parser.add_argument("--frame", default="005663", help="KITTI Frame ID (default: 005663)")
    parser.add_argument("--dataset-root", default="Code/Kitti_Raw", help="Path to KITTI root")
    parser.add_argument("--output-dir", default="Demo/outputs/kitti_3d_bev_visualizations/005663", help="Output directory")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    velo_path = root / f"velodyne/training/velodyne/{args.frame}.bin"
    label_path = root / f"label_2/training/label_2/{args.frame}.txt"
    calib_path = root / f"calib/training/calib/{args.frame}.txt"

    for p in (velo_path, label_path, calib_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing required KITTI file: {p}")

    print(f"Loading KITTI frame {args.frame}...")
    points = read_points(velo_path)
    objects = read_objects(label_path, calib_path)
    print(f"Loaded {len(points):,} points and {len(objects)} 3D objects.")

    # 1. Standalone 3D Perspective
    render_3d_perspective(
        points, objects,
        out_dir / f"kitti_{args.frame}_3d_perspective.png",
    )

    # 2. Standalone BEV Top-down
    render_bev_topdown(
        points, objects,
        out_dir / f"kitti_{args.frame}_bev_topdown.png",
    )

    # 3. Dual-view Composite
    render_dual_composite(
        points, objects,
        out_dir / f"kitti_{args.frame}_dual_composite.png",
    )

    print("All visualizations rendered successfully!")


if __name__ == "__main__":
    main()
