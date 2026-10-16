#!/usr/bin/env python3
"""Compare and visualize GT database samplers on real KITTI point clouds.

Supports:
1. Raw Original Scene
2. OpenPCDet DataBaseSampler
3. HybridDataBaseSampler Default
4. HybridDataBaseSampler All Heuristics OFF ("Tắt hết")
5. HybridDataBaseSampler Ablation (toggle each heuristic flag ON/OFF)
6. HybridDataBaseSampler All Heuristics ON
7. Consolidated Comparison Matrix Grid Dashboard
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

# Ensure Code and detector modules are accessible
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "detector"), str(ROOT / "tools/kitti_training_pipeline")]

from core.datasets.augmentor.geometry import bev_polygon, points_in_box
from core.datasets.augmentor.database_sampler import DataBaseSampler
from core.datasets.augmentor.hybrid_sampler import HybridDataBaseSampler
from tools.visualization.export_kitti_overview import dataset_paths

CLASS_COLORS = {
    "Car": "#ff5252",         # Bright red / coral
    "Pedestrian": "#ffeb3b",  # Amber / yellow
    "Cyclist": "#ff9800",     # Vivid orange
    "original_box": "#4db8ff",# Cyan / blue for original
    "inserted_box": "#00ff66",# Bright neon green for inserted
}

ALL_FLAGS = [
    "ENABLE_GROUND_VALIDATION",
    "ENABLE_STATIC_COLLISION",
    "ENABLE_LINE_OF_SIGHT",
    "ENABLE_SHADOW_MASKING",
    "ENABLE_DENSITY_SUBSAMPLE",
    "ENABLE_RADIOMETRIC_CALIBRATION",
    "ENABLE_VISIBILITY_PROTECTION",
]


def load_scene_data(frame_id: str, kitti_dir: Path | str) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Load KITTI raw point cloud and converted bounding boxes for a frame."""
    paths = dataset_paths(kitti_dir)
    scan_path = paths.velodyne / f"{frame_id}.bin"
    label_path = paths.labels / f"{frame_id}.txt"
    calib_path = paths.calib / f"{frame_id}.txt"

    if not scan_path.is_file():
        raise FileNotFoundError(f"Missing velodyne scan: {scan_path}")
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing label file: {label_path}")
    if not calib_path.is_file():
        raise FileNotFoundError(f"Missing calib file: {calib_path}")

    points = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)

    # Use repo's standard conversion logic
    from tools.kitti_training_pipeline.prepare_kitti import convert_labels
    class_ids = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
    converted_lines, _ = convert_labels(label_path, calib_path)
    boxes = []
    for line in converted_lines:
        fields = line.split()
        if fields and fields[0] in class_ids:
            boxes.append([class_ids[fields[0]], *map(float, fields[1:])])
    boxes_arr = np.asarray(boxes, dtype=np.float32).reshape(-1, 8)
    return points, boxes_arr, class_ids


def get_base_configs(donor_db_path: str = "gt_database/dbinfos_train.json") -> Tuple[dict, dict]:
    """Load default augmentation configs for OpenPCDet and Hybrid samplers."""
    openpcdet_cfg_path = ROOT / "configs/augmentation/openpcdet_gt.json"
    hybrid_cfg_path = ROOT / "configs/augmentation/hybrid_gt.json"

    with open(openpcdet_cfg_path, "r", encoding="utf-8") as f:
        cfg_open = json.load(f)["augmentation"]["AUG_CONFIG_LIST"][0]
    with open(hybrid_cfg_path, "r", encoding="utf-8") as f:
        cfg_hybrid = json.load(f)["augmentation"]["AUG_CONFIG_LIST"][0]

    cfg_open["DB_INFO_PATH"] = [donor_db_path]
    cfg_hybrid["DB_INFO_PATH"] = [donor_db_path]
    return cfg_open, cfg_hybrid


def run_sampler_variant(
    variant_key: str,
    title: str,
    description: str,
    points: np.ndarray,
    boxes: np.ndarray,
    db_root: Path,
    class_ids: Dict[str, int],
    seed: int = 42,
    geometry: dict | None = None,
) -> Dict[str, Any]:
    """Execute a specific sampler variant and record provenance metadata."""
    if geometry is None:
        geometry = {"x_min": 0.0, "x_max": 70.4, "y_min": -40.0, "y_max": 40.0, "z_min": -2.5, "z_max": 1.0}

    cfg_open, cfg_hybrid = get_base_configs()
    flags_status = {f: False for f in ALL_FLAGS}
    sampler_type = "hybrid"

    # Variant definitions
    if variant_key == "raw_original":
        return {
            "key": variant_key,
            "title": title,
            "description": description,
            "sampler_type": "none",
            "flags": flags_status,
            "points": points.copy(),
            "boxes": boxes.copy(),
            "point_is_sampled": np.zeros(len(points), dtype=bool),
            "box_is_sampled": np.zeros(len(boxes), dtype=bool),
            "metadata": {
                "num_inserted": 0,
                "accepted_by_class": {c: 0 for c in class_ids},
                "removed_interior_points": 0,
                "removed_shadow_points": 0,
                "rejected_by_reason": {},
            },
        }

    if variant_key == "openpcdet":
        sampler_type = "openpcdet"
        # Match class quotas to hybrid for fair comparison if requested, or keep openpcdet groups
        cfg = copy.deepcopy(cfg_open)
        rng = np.random.default_rng(seed)
        sampler = DataBaseSampler(db_root, cfg, class_ids, geometry=geometry)
        p_out, b_out = sampler(points.copy(), boxes.copy(), rng)

        keep = np.ones(len(points), bool)
        for b in b_out[len(boxes):]:
            keep &= ~points_in_box(points, b, cfg.get("REMOVE_EXTRA_WIDTH", [0, 0, 0]))
        num_inserted = len(b_out) - len(boxes)
        point_is_sampled = np.arange(len(p_out)) >= int(keep.sum())
        box_is_sampled = np.arange(len(b_out)) >= len(boxes)

        # Count accepted per class
        inv_ids = {v: k for k, v in class_ids.items()}
        acc_by_class = {c: 0 for c in class_ids}
        for b in b_out[len(boxes):]:
            c_name = inv_ids.get(int(b[0]), "Unknown")
            if c_name in acc_by_class:
                acc_by_class[c_name] += 1

        return {
            "key": variant_key,
            "title": title,
            "description": description,
            "sampler_type": sampler_type,
            "flags": flags_status,
            "points": p_out,
            "boxes": b_out,
            "point_is_sampled": point_is_sampled,
            "box_is_sampled": box_is_sampled,
            "metadata": {
                "num_inserted": num_inserted,
                "accepted_by_class": acc_by_class,
                "removed_interior_points": int((~keep).sum()),
                "removed_shadow_points": 0,
                "rejected_by_reason": {},
            },
        }

    # Hybrid variants
    cfg = copy.deepcopy(cfg_hybrid)
    if variant_key == "hybrid_default":
        pass  # Uses defaults from hybrid_gt.json
    elif variant_key == "hybrid_all_off":
        for f in ALL_FLAGS:
            cfg[f] = False
    elif variant_key == "hybrid_all_on":
        for f in ALL_FLAGS:
            cfg[f] = True
    elif variant_key.startswith("hybrid_toggle_"):
        flag_to_toggle = variant_key.replace("hybrid_toggle_", "")
        if flag_to_toggle in ALL_FLAGS:
            current = cfg.get(flag_to_toggle, False)
            cfg[flag_to_toggle] = not current

    # Populate flags status
    for f in ALL_FLAGS:
        flags_status[f] = bool(cfg.get(f, False))

    rng = np.random.default_rng(seed)
    sampler = HybridDataBaseSampler(db_root, cfg, class_ids, geometry=geometry)
    p_out, b_out, meta = sampler(points.copy(), boxes.copy(), rng, return_metadata=True)

    return {
        "key": variant_key,
        "title": title,
        "description": description,
        "sampler_type": "hybrid",
        "flags": flags_status,
        "points": p_out,
        "boxes": b_out,
        "point_is_sampled": meta["point_is_sampled"],
        "box_is_sampled": meta["box_is_sampled"],
        "metadata": meta,
    }


def define_comparison_suite() -> List[Dict[str, str]]:
    """Define all comparison variants requested by user."""
    return [
        {
            "key": "raw_original",
            "file_prefix": "00_raw_original_scene",
            "title": "Raw Scene (Original Point Cloud)",
            "description": "Original KITTI LiDAR scan and GT annotations without any GT sampling.",
        },
        {
            "key": "openpcdet",
            "file_prefix": "01_openpcdet_gt_sampler",
            "title": "OpenPCDet GT Sampler",
            "description": "Standard OpenPCDet cut-and-paste augmentation with rotated box collisions.",
        },
        {
            "key": "hybrid_default",
            "file_prefix": "02_hybrid_gt_default",
            "title": "Hybrid GT Sampler (Default Heuristics)",
            "description": "Default recipe: Ground ON, Static ON, Density ON, Visibility ON; LOS/Shadow/Radiometric OFF.",
        },
        {
            "key": "hybrid_all_off",
            "file_prefix": "03_hybrid_all_heuristics_off",
            "title": "Hybrid GT Sampler (All Heuristics OFF - Tắt Hết)",
            "description": "All 7 physical/heuristic filters disabled. Evaluates pure geometric relocation & bounds.",
        },
        {
            "key": "hybrid_toggle_ENABLE_GROUND_VALIDATION",
            "file_prefix": "04_hybrid_ground_validation_off",
            "title": "Hybrid GT Sampler: Ground Validation OFF",
            "description": "Disables local 5th-percentile ground height snapping; uses source object relative height.",
        },
        {
            "key": "hybrid_toggle_ENABLE_STATIC_COLLISION",
            "file_prefix": "05_hybrid_static_collision_off",
            "title": "Hybrid GT Sampler: Static Collision OFF",
            "description": "Disables footprint elevated-point rejection. Candidates may overlap static background.",
        },
        {
            "key": "hybrid_toggle_ENABLE_LINE_OF_SIGHT",
            "file_prefix": "06_hybrid_line_of_sight_on",
            "title": "Hybrid GT Sampler: Line of Sight ON",
            "description": "Enables ray blocker counting between LiDAR sensor and candidate box center.",
        },
        {
            "key": "hybrid_toggle_ENABLE_SHADOW_MASKING",
            "file_prefix": "07_hybrid_shadow_masking_on",
            "title": "Hybrid GT Sampler: Shadow Masking ON",
            "description": "Enables LiDAR ray shadow projection; deletes occluded background points behind inserted boxes.",
        },
        {
            "key": "hybrid_toggle_ENABLE_DENSITY_SUBSAMPLE",
            "file_prefix": "08_hybrid_density_subsample_off",
            "title": "Hybrid GT Sampler: Density Subsample OFF",
            "description": "Disables point thinning when objects are relocated to larger sensor ranges.",
        },
        {
            "key": "hybrid_toggle_ENABLE_RADIOMETRIC_CALIBRATION",
            "file_prefix": "09_hybrid_radiometric_calibration_on",
            "title": "Hybrid GT Sampler: Radiometric Calibration ON",
            "description": "Enables distance-dependent LiDAR beam intensity attenuation model.",
        },
        {
            "key": "hybrid_toggle_ENABLE_VISIBILITY_PROTECTION",
            "file_prefix": "10_hybrid_visibility_protection_off",
            "title": "Hybrid GT Sampler: Visibility Protection OFF",
            "description": "Disables minimum surviving point/ratio threshold protection for pre-existing objects.",
        },
        {
            "key": "hybrid_all_on",
            "file_prefix": "11_hybrid_all_heuristics_on",
            "title": "Hybrid GT Sampler (All Heuristics ON - Bật Hết)",
            "description": "All 7 physical rules active simultaneously: ground, static, LOS, shadow, density, radiometric, visibility.",
        },
    ]


def render_bev_figure(
    result: Dict[str, Any],
    frame_id: str,
    geometry: dict,
    class_names: Dict[str, int],
    seed: int,
    output_path: Path,
    max_bg_points: int = 50000,
    dpi: int = 180,
) -> None:
    """Render a publication-quality BEV top-down image with HUD overlay."""
    fig = plt.figure(figsize=(14, 11), facecolor="#0a0e14")
    ax = fig.add_axes([0.08, 0.08, 0.62, 0.84], facecolor="#0f141c")

    points = result["points"]
    boxes = result["boxes"]
    pt_sampled = result["point_is_sampled"]
    bx_sampled = result["box_is_sampled"]
    meta = result["metadata"]

    # Filter visible points inside geometry bounds
    vis_mask = (
        (points[:, 0] >= geometry["x_min"]) & (points[:, 0] <= geometry["x_max"]) &
        (points[:, 1] >= geometry["y_min"]) & (points[:, 1] <= geometry["y_max"]) &
        (points[:, 2] >= geometry["z_min"]) & (points[:, 2] <= geometry["z_max"])
    )

    # 1. Background (original) points
    bg_idx = np.flatnonzero(vis_mask & (~pt_sampled))
    if len(bg_idx) > max_bg_points:
        rng = np.random.default_rng(0)
        bg_idx = rng.choice(bg_idx, max_bg_points, replace=False)
    ax.scatter(
        points[bg_idx, 1], points[bg_idx, 0],
        s=0.45, c="#4b6b94", alpha=0.45, linewidths=0, rasterized=True, label="Original LiDAR Points"
    )

    # 2. Inserted points
    fg_idx = np.flatnonzero(vis_mask & pt_sampled)
    if len(fg_idx) > 0:
        ax.scatter(
            points[fg_idx, 1], points[fg_idx, 0],
            s=1.8, c="#00ff66", alpha=0.95, linewidths=0, rasterized=True, label="Inserted Points (Provenance)"
        )

    # 3. Draw Bounding Boxes
    inv_class = {v: k for k, v in class_names.items()}
    for box, is_ins in zip(boxes, bx_sampled):
        c_name = inv_class.get(int(box[0]), "Object")
        color = CLASS_COLORS["inserted_box"] if is_ins else CLASS_COLORS["original_box"]
        lw = 1.6 if is_ins else 1.0
        poly = bev_polygon(box)
        coords = np.asarray(poly.exterior.coords)
        ax.plot(coords[:, 1], coords[:, 0], color=color, lw=lw, alpha=0.9)

        # Label tag
        ax.text(
            box[5], box[4], f"{c_name}",
            color=color, fontsize=6.5, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="#0a0e14", edgecolor=color, alpha=0.7, lw=0.5)
        )

    # Sensor origin
    ax.scatter(0, 0, marker="o", s=80, c="#ffffff", edgecolor="#ff0055", lw=2, zorder=10)
    ax.text(0, -1.8, "LiDAR (0,0)", color="#ffffff", fontsize=7.5, ha="center", va="top")

    # Range rings
    for r in [15, 30, 45, 60]:
        theta = np.linspace(-np.pi / 2, np.pi / 2, 100)
        ring_y = r * np.sin(theta)
        ring_x = r * np.cos(theta)
        valid_ring = (ring_y >= geometry["y_min"]) & (ring_y <= geometry["y_max"]) & (ring_x <= geometry["x_max"])
        ax.plot(ring_y[valid_ring], ring_x[valid_ring], color="#ffffff", lw=0.5, ls="--", alpha=0.18)
        ax.text(0, r + 0.4, f"{r}m", color="#ffffff", fontsize=6.5, alpha=0.35, ha="center")

    # Plot limits and orientation: Forward X is up, Lateral Y is left-right
    ax.set_xlim(geometry["y_max"], geometry["y_min"])
    ax.set_ylim(geometry["x_min"], geometry["x_max"])
    ax.set_aspect("equal")
    ax.set_xlabel("Lateral Y (m)  [Left ←   → Right]", color="#d0d7de", fontsize=10, labelpad=8)
    ax.set_ylabel("Forward X (m)  [Ahead ↑]", color="#d0d7de", fontsize=10, labelpad=8)
    ax.tick_params(colors="#8b949e", labelsize=8.5)
    ax.grid(True, color="#ffffff", alpha=0.08, ls=":")

    # Title
    fig.suptitle(f"{result['title']}", color="#ffffff", fontsize=14, fontweight="bold", y=0.96)
    ax.set_title(
        f"KITTI Frame: {frame_id} | Seed: {seed} | Sampler: {result['sampler_type'].upper()}",
        color="#8b949e", fontsize=9.5, pad=8
    )

    # 4. HUD / Information Panel on the right side
    hud_ax = fig.add_axes([0.72, 0.08, 0.25, 0.84], facecolor="#141a23")
    hud_ax.axis("off")

    hud_y = 0.97
    hud_ax.text(0.04, hud_y, "CONFIGURATION & HUD", color="#58a6ff", fontsize=11, fontweight="bold")
    hud_y -= 0.04

    hud_ax.text(0.04, hud_y, "Description:", color="#8b949e", fontsize=8, fontweight="bold")
    hud_y -= 0.025
    hud_ax.text(0.04, hud_y, result["description"], color="#c9d1d9", fontsize=7.5, wrap=True)
    hud_y -= 0.065

    # Statistics section
    hud_ax.text(0.04, hud_y, "OBJECT INSERTION STATS", color="#58a6ff", fontsize=9.5, fontweight="bold")
    hud_y -= 0.035

    num_orig = int((~bx_sampled).sum())
    num_ins = int(bx_sampled.sum())
    num_total = len(boxes)

    hud_ax.text(0.06, hud_y, f"• Original GT Boxes: {num_orig}", color="#c9d1d9", fontsize=8.5)
    hud_y -= 0.025
    hud_ax.text(0.06, hud_y, f"• Inserted GT Boxes: +{num_ins}", color="#7ee787", fontsize=8.5, fontweight="bold")
    hud_y -= 0.025
    hud_ax.text(0.06, hud_y, f"• Total Scene Boxes: {num_total}", color="#c9d1d9", fontsize=8.5)
    hud_y -= 0.035

    hud_ax.text(0.06, hud_y, "Class Breakdown Inserted:", color="#8b949e", fontsize=8)
    hud_y -= 0.025
    acc = meta.get("accepted_by_class", {})
    for c_name, c_cnt in acc.items():
        color_c = CLASS_COLORS.get(c_name, "#c9d1d9")
        hud_ax.text(0.10, hud_y, f"• {c_name}: +{c_cnt}", color=color_c, fontsize=8, fontweight="bold")
        hud_y -= 0.022
    hud_y -= 0.02

    # Points section
    hud_ax.text(0.04, hud_y, "POINT PROVENANCE STATS", color="#58a6ff", fontsize=9.5, fontweight="bold")
    hud_y -= 0.035
    num_pt_orig = int((~pt_sampled).sum())
    num_pt_ins = int(pt_sampled.sum())
    hud_ax.text(0.06, hud_y, f"• Surviving Orig Points: {num_pt_orig:,}", color="#c9d1d9", fontsize=8.5)
    hud_y -= 0.025
    hud_ax.text(0.06, hud_y, f"• Inserted Object Points: +{num_pt_ins:,}", color="#7ee787", fontsize=8.5, fontweight="bold")
    hud_y -= 0.025
    hud_ax.text(0.06, hud_y, f"• Total Points in Scene: {len(points):,}", color="#c9d1d9", fontsize=8.5)
    hud_y -= 0.025

    rem_int = meta.get("removed_interior_points", 0)
    rem_shd = meta.get("removed_shadow_points", 0)
    hud_ax.text(0.06, hud_y, f"• Deleted Interior: -{rem_int:,}", color="#ff7b72", fontsize=8)
    hud_y -= 0.022
    hud_ax.text(0.06, hud_y, f"• Deleted Shadow Rays: -{rem_shd:,}", color="#ff7b72", fontsize=8)
    hud_y -= 0.04

    # Heuristic Flags section
    hud_ax.text(0.04, hud_y, "HEURISTIC FLAGS STATUS", color="#58a6ff", fontsize=9.5, fontweight="bold")
    hud_y -= 0.032

    if result["sampler_type"] == "hybrid":
        for f in ALL_FLAGS:
            is_on = result["flags"].get(f, False)
            short_name = f.replace("ENABLE_", "")
            status_color = "#7ee787" if is_on else "#8b949e"
            tag = "[ON] " if is_on else "[OFF]"
            hud_ax.text(0.06, hud_y, f"{tag} {short_name}", color=status_color, fontsize=7.5, fontfamily="monospace")
            hud_y -= 0.021
    else:
        hud_ax.text(0.06, hud_y, "N/A (Standard Box Intersection Only)", color="#8b949e", fontsize=8)
        hud_y -= 0.04

    hud_y -= 0.025
    # Rejection Breakdown section
    hud_ax.text(0.04, hud_y, "REJECTION BREAKDOWN", color="#58a6ff", fontsize=9.5, fontweight="bold")
    hud_y -= 0.032
    rej = meta.get("rejected_by_reason", {})
    if rej:
        for r_reason, r_cnt in rej.items():
            hud_ax.text(0.06, hud_y, f"• {r_reason}: {r_cnt}", color="#d29922", fontsize=8)
            hud_y -= 0.022
    else:
        hud_ax.text(0.06, hud_y, "• None (all placements accepted)", color="#7ee787", fontsize=8)
        hud_y -= 0.025

    # Legend at bottom
    legend_elements = [
        Line2D([0], [0], color=CLASS_COLORS["original_box"], lw=1.5, label="Original GT Box"),
        Line2D([0], [0], color=CLASS_COLORS["inserted_box"], lw=1.8, label="Inserted GT Box"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4b6b94", markersize=5, label="Original Points"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#00ff66", markersize=6, label="Inserted Points"),
    ]
    ax.legend(
        handles=legend_elements, loc="lower right", facecolor="#0a0e14", edgecolor="#30363d",
        labelcolor="#c9d1d9", fontsize=8.5, framealpha=0.9
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def render_composite_matrix_grid(
    results_map: Dict[str, Dict[str, Any]],
    frame_id: str,
    geometry: dict,
    class_names: Dict[str, int],
    seed: int,
    output_path: Path,
    dpi: int = 160,
) -> None:
    """Render a consolidated 2x4 comparison grid showing key variants side-by-side."""
    selected_keys = [
        ("raw_original", "1. Original Raw Scene (No Aug)"),
        ("openpcdet", "2. OpenPCDet GT Sampler"),
        ("hybrid_all_off", "3. Hybrid All Heuristics OFF ('Tắt Hết')"),
        ("hybrid_default", "4. Hybrid GT Sampler (Default)"),
        ("hybrid_toggle_ENABLE_STATIC_COLLISION", "5. Hybrid Static Collision OFF"),
        ("hybrid_toggle_ENABLE_LINE_OF_SIGHT", "6. Hybrid Line of Sight ON"),
        ("hybrid_toggle_ENABLE_SHADOW_MASKING", "7. Hybrid Shadow Masking ON"),
        ("hybrid_all_on", "8. Hybrid All Heuristics ON ('Bật Hết')"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(26, 15), facecolor="#0a0e14")
    axes = axes.flatten()

    inv_class = {v: k for k, v in class_names.items()}

    for ax, (v_key, v_title) in zip(axes, selected_keys):
        res = results_map[v_key]
        points = res["points"]
        boxes = res["boxes"]
        pt_sampled = res["point_is_sampled"]
        bx_sampled = res["box_is_sampled"]
        meta = res["metadata"]

        ax.set_facecolor("#0f141c")

        vis_mask = (
            (points[:, 0] >= geometry["x_min"]) & (points[:, 0] <= geometry["x_max"]) &
            (points[:, 1] >= geometry["y_min"]) & (points[:, 1] <= geometry["y_max"]) &
            (points[:, 2] >= geometry["z_min"]) & (points[:, 2] <= geometry["z_max"])
        )

        bg_idx = np.flatnonzero(vis_mask & (~pt_sampled))
        if len(bg_idx) > 25000:
            rng = np.random.default_rng(0)
            bg_idx = rng.choice(bg_idx, 25000, replace=False)
        ax.scatter(points[bg_idx, 1], points[bg_idx, 0], s=0.35, c="#4b6b94", alpha=0.4, linewidths=0, rasterized=True)

        fg_idx = np.flatnonzero(vis_mask & pt_sampled)
        if len(fg_idx) > 0:
            ax.scatter(points[fg_idx, 1], points[fg_idx, 0], s=1.4, c="#00ff66", alpha=0.9, linewidths=0, rasterized=True)

        for box, is_ins in zip(boxes, bx_sampled):
            color = CLASS_COLORS["inserted_box"] if is_ins else CLASS_COLORS["original_box"]
            lw = 1.3 if is_ins else 0.8
            poly = bev_polygon(box)
            coords = np.asarray(poly.exterior.coords)
            ax.plot(coords[:, 1], coords[:, 0], color=color, lw=lw, alpha=0.85)

        # Range rings
        for r in [20, 40, 60]:
            theta = np.linspace(-np.pi / 2, np.pi / 2, 60)
            ax.plot(r * np.sin(theta), r * np.cos(theta), color="#ffffff", lw=0.4, ls="--", alpha=0.15)

        ax.set_xlim(geometry["y_max"], geometry["y_min"])
        ax.set_ylim(geometry["x_min"], geometry["x_max"])
        ax.set_aspect("equal")
        ax.tick_params(colors="#8b949e", labelsize=7)
        ax.grid(True, color="#ffffff", alpha=0.06, ls=":")

        num_ins = int(bx_sampled.sum())
        ax.set_title(
            f"{v_title}\nAdded: +{num_ins} objs | +{int(pt_sampled.sum()):,} pts",
            color="#ffffff", fontsize=9.5, fontweight="bold", pad=6
        )

    for i in range(4):
        axes[i + 4].set_xlabel("Lateral Y (m)", color="#d0d7de", fontsize=9)
    for i in (0, 4):
        axes[i].set_ylabel("Forward X (m)", color="#d0d7de", fontsize=9)

    fig.suptitle(
        f"3D LiDAR GT Sampler Comparison Matrix Dashboard | KITTI Frame {frame_id} | Seed {seed}",
        color="#ffffff", fontsize=16, fontweight="bold", y=0.98
    )
    legend_elements = [
        Line2D([0], [0], color=CLASS_COLORS["original_box"], lw=1.5, label="Original GT Box"),
        Line2D([0], [0], color=CLASS_COLORS["inserted_box"], lw=1.8, label="Inserted GT Box"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4b6b94", markersize=6, label="Original Points"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#00ff66", markersize=7, label="Inserted Points"),
    ]
    fig.legend(
        handles=legend_elements, loc="lower center", ncol=4, facecolor="#0a0e14",
        edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=10, framealpha=0.9
    )

    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.95])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)


def export_markdown_summary(
    results_map: Dict[str, Dict[str, Any]],
    output_dir: Path,
    frame_id: str,
    seed: int,
) -> None:
    """Generate Markdown comparison report for the output folder."""
    md_lines = [
        f"# Báo cáo So sánh Thuật toán GT Sampler & Hybrid Heuristics Ablation",
        f"",
        f"- **KITTI Frame**: `{frame_id}`",
        f"- **Random Seed**: `{seed}`",
        f"- **Thư mục xuất ảnh**: `{output_dir}`",
        f"",
        f"## 1. Bảng Tổng hợp Kết quả Thực nghiệm",
        f"",
        f"| STT | Tên Cấu hình / Biến thể | Thuật toán | Objs Thêm | Pts Thêm | Pts Xóa | Lý do Từ chối (Rejections) |",
        f"|---|---|---|---|---|---|---|",
    ]

    for idx, (v_key, res) in enumerate(results_map.items()):
        meta = res["metadata"]
        num_ins = meta.get("num_inserted", int(res["box_is_sampled"].sum()))
        pts_ins = int(res["point_is_sampled"].sum())
        pts_del = meta.get("removed_interior_points", 0) + meta.get("removed_shadow_points", 0)
        rej_dict = meta.get("rejected_by_reason", {})
        rej_str = ", ".join(f"`{k}: {v}`" for k, v in rej_dict.items()) if rej_dict else "None"
        md_lines.append(
            f"| {idx} | **{res['title']}** | `{res['sampler_type']}` | +{num_ins} | +{pts_ins:,} | -{pts_del:,} | {rej_str} |"
        )

    md_lines.extend([
        f"",
        f"## 2. Nhận xét Phân tích Thuật toán",
        f"",
        f"1. **OpenPCDet GT Sampler**:",
        f"   - Lấy mẫu ngẫu nhiên từ pool và kiểm tra va chạm hộp (bounding box collision).",
        f"   - Xóa các điểm bên trong hộp chèn, không tính đến địa hình mặt đất (ground support) hay chướng ngại vật tĩnh.",
        f"",
        f"2. **Hybrid GT Sampler (All Heuristics OFF - Tắt Hết)**:",
        f"   - Chỉ sử dụng cơ chế vị trí tương đối (`source_relative`) và khoảng đệm an toàn margin (0.1m).",
        f"   - Đạt tốc độ cao nhất trong họ Hybrid nhưng không lọc các trường hợp đối tượng bị chìm/bay hoặc chồng lấn cấu trúc tĩnh.",
        f"",
        f"3. **Hybrid GT Sampler (Default)**:",
        f"   - Kích hoạt `Ground Validation` và `Static Collision` giúp loại bỏ hoàn toàn các hộp bay lơ lửng hoặc đè lên tường/cây cối.",
        f"   - Kích hoạt `Density Subsample` và `Visibility Protection` bảo toàn tính tự nhiên của đám mây điểm.",
        f"",
        f"4. **Các tính năng Heuristic chuyên sâu**:",
        f"   - `Line of Sight (LOS)`: Kiểm tra độ che chắn góc nhìn từ cảm biến tới vật thể.",
        f"   - `Shadow Masking`: Mô phỏng tia quét bị cản, xóa các điểm nền nằm trong vùng khuất tia.",
        f"   - `Radiometric Calibration`: Mô hình hóa độ suy giảm năng lượng phản xạ theo bình phương khoảng cách.",
        f"",
        f"## 3. Danh sách File Ảnh Đã Tạo",
        f"",
    ])

    for v_key, res in results_map.items():
        md_lines.append(f"- `{res.get('file_name', v_key + '.png')}`: {res['title']}")
    md_lines.append(f"- `12_comparison_matrix_grid.png`: Bảng ghép trực quan tổng hợp 8 cấu hình tiêu biểu.")

    report_path = output_dir / "README.md"
    report_path.write_text("\n".join(md_lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-id", type=str, default="005663", help="KITTI Frame ID (e.g. 005663, 000004)")
    parser.add_argument("--kitti-dir", type=Path, default=ROOT / "Kitti_Raw", help="Path to KITTI raw root")
    parser.add_argument(
        "--donor-dir",
        type=Path,
        default=ROOT.parent / "Demo/outputs/pipeline/005663/donor_database",
        help="Path to prebuilt donor database directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT.parent / "Demo/outputs/gt_sampler_comparison",
        help="Target directory to write comparison images",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling")
    parser.add_argument("--dpi", type=int, default=180, help="Image DPI")
    args = parser.parse_args()

    print(f"=== LiDAR GT Sampler Comparison Visualization ===")
    print(f"Frame ID: {args.frame_id}")
    print(f"KITTI Directory: {args.kitti_dir}")
    print(f"Donor DB Directory: {args.donor_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Seed: {args.seed}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load scene data
    points, boxes, class_ids = load_scene_data(args.frame_id, args.kitti_dir)
    print(f"Loaded raw point cloud: {points.shape[0]:,} points, {boxes.shape[0]} GT boxes")

    geometry = {"x_min": 0.0, "x_max": 70.4, "y_min": -40.0, "y_max": 40.0, "z_min": -2.5, "z_max": 1.0}

    # 2. Define comparison suite
    suite = define_comparison_suite()
    results_map = {}
    summary_metrics = {}

    print(f"\nExecuting {len(suite)} sampler variants...")
    for item in suite:
        t0 = time.time()
        key = item["key"]
        file_prefix = item["file_prefix"]
        out_png = args.output_dir / f"{file_prefix}.png"

        res = run_sampler_variant(
            key,
            item["title"],
            item["description"],
            points,
            boxes,
            args.donor_dir,
            class_ids,
            seed=args.seed,
            geometry=geometry,
        )
        res["file_name"] = f"{file_prefix}.png"
        results_map[key] = res

        # Render individual BEV figure
        render_bev_figure(
            res,
            args.frame_id,
            geometry,
            class_names=class_ids,
            seed=args.seed,
            output_path=out_png,
            dpi=args.dpi,
        )
        dt = time.time() - t0
        num_ins = int(res["box_is_sampled"].sum())
        pts_ins = int(res["point_is_sampled"].sum())
        print(f"  [{file_prefix}] -> +{num_ins} boxes, +{pts_ins:,} points ({dt:.2f}s) => {out_png.name}")

        meta = res["metadata"]
        summary_metrics[key] = {
            "title": res["title"],
            "sampler_type": res["sampler_type"],
            "num_inserted": num_ins,
            "points_inserted": pts_ins,
            "flags": res["flags"],
            "accepted_by_class": meta.get("accepted_by_class", {}),
            "rejected_by_reason": meta.get("rejected_by_reason", {}),
            "removed_interior_points": meta.get("removed_interior_points", 0),
            "removed_shadow_points": meta.get("removed_shadow_points", 0),
        }

    # 3. Render Composite Matrix Grid
    print("\nRendering consolidated comparison dashboard grid...")
    grid_path = args.output_dir / "12_comparison_matrix_grid.png"
    render_composite_matrix_grid(
        results_map,
        args.frame_id,
        geometry,
        class_names=class_ids,
        seed=args.seed,
        output_path=grid_path,
        dpi=160,
    )
    print(f"  [12_comparison_matrix_grid] => {grid_path.name}")

    # 4. Save JSON summary & README
    json_path = args.output_dir / "summary_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)
    print(f"Saved metrics JSON to {json_path}")

    export_markdown_summary(results_map, args.output_dir, args.frame_id, args.seed)
    print(f"Saved Markdown report to {args.output_dir / 'README.md'}")
    print("\nAll visualizations completed successfully!")


if __name__ == "__main__":
    main()
