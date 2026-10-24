#!/usr/bin/env python3
"""Two GT BEVs, then paired detection/heatmap views for MobilePIXOR and Ours."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "Code"))
from tools.visualization.export_kitti_overview import read_objects, setup_matplotlib, sha256
from export_kitti_origin_mobilepixor_ours import load_verified_frame
from export_kitti_model_comparison import draw_cloud, draw_heatmap


def load_frame(root, ground_truth_root, frame):
    points, _, provenance = load_verified_frame(root, frame)
    label = ground_truth_root / "training/label_2" / f"{frame}.txt"
    calib = ground_truth_root / "training/calib" / f"{frame}.txt"
    ground_truth = read_objects(label, calib)
    results = {}
    for model_id in ("mobilepixor", "ours"):
        model_dir = root / frame / model_id
        metadata = json.loads((model_dir / "metadata.json").read_text())
        archive = model_dir / "classification_heatmaps.npz"
        if sha256(archive) != metadata["files"][archive.name]:
            raise ValueError(f"Heatmap hash mismatch: {frame}/{model_id}")
        with np.load(archive, allow_pickle=False) as stored:
            heatmaps = {key: stored[key].copy() for key in stored.files}
        import torch
        np.testing.assert_array_equal(heatmaps["probabilities"],
                                      torch.sigmoid(torch.from_numpy(heatmaps["logits"])).numpy())
        detections, lifted_corners = load_predictions(root, frame, model_id)
        from types import SimpleNamespace
        lifted = {index: SimpleNamespace(corners=corners) for index, corners in lifted_corners.items()}
        results[model_id] = {
            "detections": detections, "lifted": lifted, "heatmaps": heatmaps,
            "config": json.loads((model_dir / "config.inference.json").read_text()),
        }
        provenance["models"][model_id]["classification_heatmaps_sha256"] = sha256(archive)
    provenance["ground_truth"] = {
        "label_path": str(label), "label_sha256": sha256(label),
        "calibration_path": str(calib), "calibration_sha256": sha256(calib),
        "count": len(ground_truth), "class_counts": dict(Counter(obj.name for obj in ground_truth)),
        "coordinate_transform": "inverse(R0_rect @ Tr_velo_to_cam) of exact KITTI camera corners",
        "source_manifest": str(ground_truth_root / "source_manifest.json"),
    }
    return points, ground_truth, results, provenance


def load_predictions(root, frame, model_id):
    from tools.visualization.predict_kitti_overview import bev_detections
    directory = root / frame / model_id
    config = json.loads((directory / "config.inference.json").read_text())
    rows = np.load(directory / "bev_predictions.npy", allow_pickle=False)
    detections = bev_detections(rows, config["data"]["kitti"]["objects"])
    with np.load(directory / "lidar_lifted_cuboids.npz", allow_pickle=False) as stored:
        cuboids = {int(index): corners.copy() for index, corners in
                   zip(stored["prediction_indices"], stored["corners"])}
    return detections, cuboids


def draw_ground_truth(ax, points, objects, roi):
    x0, x1, y0, y1, z0, z1 = roi
    keep = ((points[:, :3] >= roi[::2]) & (points[:, :3] <= roi[1::2])).all(axis=1)
    scene = points[keep]
    ax.scatter(-scene[:, 1], scene[:, 0], s=.22, c="#00d79d", alpha=.9,
               linewidths=0, rasterized=True, zorder=1)
    shown = []
    for obj in objects:
        corners = obj.corners[:4, :2]
        if corners[:, 0].max() < x0 or corners[:, 0].min() > x1 or corners[:, 1].max() < y0 or corners[:, 1].min() > y1:
            continue
        shown.append(obj.label_index)
        closed = np.r_[corners, corners[:1]]
        ax.plot(-closed[:, 1], closed[:, 0], c="#ffdf40", lw=.9, zorder=3)
    ax.set_xlim(-y1, -y0)
    ax.set_ylim(x0, x1)
    ax.set_aspect("equal", adjustable="box")
    return shown


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT / "Demo/outputs/heatmap/model_comparison")
    parser.add_argument("--ground-truth-dir", type=Path, default=PROJECT / "Demo/data/kitti_groundtruth")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "Demo/outputs/heatmap/gt_mobilepixor_ours_pairs")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args(argv)
    if args.dpi < 300:
        parser.error("Use at least 300 DPI")
    frames = ["000134", "005663"]
    data = [load_frame(args.input_dir.resolve(), args.ground_truth_dir.resolve(), frame) for frame in frames]
    roi = np.array([0., 50., -26., 26., -2.8, 4.])
    width_mm, column_gap, pair_gap, label_band = 210., .8, .4, 8.
    column_width = (width_mm - column_gap) / 2
    small_width = (column_width - pair_gap) / 2
    # Native metric aspect ratio, with no outer margins or empty row padding.
    aspect = (roi[3] - roi[2]) / (roi[1] - roi[0])
    gt_height = column_width / aspect
    pair_height = small_width / aspect
    height_mm = gt_height + pair_height * 2 + label_band * 3
    plt = setup_matplotlib()
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4), dpi=args.dpi, facecolor="white")

    def panel(left, bottom, width, height):
        ax = fig.add_axes([left / width_mm, bottom / height_mm,
                           width / width_mm, height / height_mm], facecolor="black")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return ax

    row_top = height_mm
    gt_bottom = row_top - gt_height
    gt_shown, predictions_shown = {}, {}
    for column, (points, ground_truth, _, _) in enumerate(data):
        ax = panel(column * (column_width + column_gap), gt_bottom, column_width, gt_height)
        gt_shown[frames[column]] = draw_ground_truth(ax, points, ground_truth, roi)
    fig.text(.5, (gt_bottom - label_band / 2) / height_mm, "origin",
             fontsize=17, fontweight="bold", ha="center", va="center")
    row_top = gt_bottom - label_band
    for model_id, title in (("mobilepixor", "MobilePIXOR"), ("ours", "Ours")):
        bottom = row_top - pair_height
        for column, (points, _, results, _) in enumerate(data):
            left = column * (column_width + column_gap)
            ax = panel(left, bottom, small_width, pair_height)
            heat = panel(left + small_width + pair_gap, bottom, small_width, pair_height)
            shown = draw_cloud(ax, points, results[model_id], roi, "perspective")
            # Keep geometry/heatmap intact; remove only the old score annotations.
            for annotation in list(ax.texts):
                annotation.remove()
            draw_heatmap(heat, results[model_id], roi)
            predictions_shown[f"{model_id}/{frames[column]}"] = shown
        fig.text(.5, (bottom - label_band / 2) / height_mm, title,
                 fontsize=17, fontweight="bold", ha="center", va="center")
        row_top = bottom - label_band
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "gt_mobilepixor_ours_pairs_3x2"
    for suffix in ("png", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=args.dpi, facecolor="white", pad_inches=0)
    plt.close(fig)
    metadata = {
        "layout": {"row_1": "two LiDAR BEVs with official KITTI ground truth",
                   "row_2": "two MobilePIXOR pairs, each detection left / actual model heatmap right",
                   "row_3": "two Ours pairs, each detection left / actual model heatmap right"},
        "column_frames": frames, "row_labels": ["origin", "MobilePIXOR", "Ours"],
        "canvas_mm": [width_mm, height_mm], "dpi": args.dpi, "outer_margin_mm": 0,
        "column_gap_mm": column_gap, "pair_gap_mm": pair_gap, "roi_xyz": roi.tolist(),
        "heatmap": {"source": "verified sigmoid of original model classification logits",
                    "scale": [0, 1], "merge": "max across classes", "interpolation": "nearest",
                    "smoothing": False, "ground_truth_used": False},
        "scores_displayed": False, "extra_titles_displayed": False,
        "ground_truth_only_in_first_row": True, "ground_truth_used_for_predictions": False,
        "visible_ground_truth_indices": gt_shown, "visible_prediction_indices": predictions_shown,
        "sources": [item[3] for item in data], "script_sha256": sha256(Path(__file__)),
        "figures": {suffix: {"path": str(stem.with_suffix(f'.{suffix}')),
                             "sha256": sha256(stem.with_suffix(f'.{suffix}'))} for suffix in ("png", "svg")},
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(f"Exported {stem}.png and .svg; canvas {width_mm:g} x {height_mm:.2f} mm", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
