#!/usr/bin/env python3
"""Fresh checkpoint inference and the requested caption-only reference layout."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "Code"))
from tools.visualization.export_kitti_overview import read_objects, read_points, setup_matplotlib, sha256
from tools.visualization.predict_kitti_overview import BEVOverviewPredictor, bev_detections
from export_kitti_model_comparison import (
    draw_heatmap, load_cloud, validate_variant, write_json,
)

FRAMES = ("003736", "005663")
MODEL_IDS = ("mobilepixor", "ours")
CAPTIONS = ("A) Grouth truth", "B) MobilePIXOR", "C)Ours")
ROI = np.array([5., 38., -26., 20., -2.8, 4.])
BOX_COLOR = "#ff3030"


def load_comparison_cloud(frame):
    path = PROJECT / "Demo/data/kitti_groundtruth/training/velodyne" / f"{frame}.bin"
    if not path.is_file():
        return load_cloud(frame)
    source_path = path.with_suffix(".source.json")
    recorded = json.loads(source_path.read_text())
    expected = recorded["sha256"]
    if sha256(path) != expected:
        raise ValueError("Downloaded scan differs from its verified official KITTI source")
    points = read_points(path)
    source = {"format": "official KITTI binary XYZI", "path": str(path),
              "point_count": len(points), "points_xyzi_sha256": expected,
              "recorded_original_bin_sha256": expected, "original_bin_hash_matches": True,
              "split": recorded.get("split", "unknown"), "download_metadata": str(source_path),
              "download_metadata_sha256": sha256(source_path)}
    return points, source


def run_inference(manifest_path, output, device, threads, frames=FRAMES):
    manifest = json.loads(manifest_path.read_text())
    items = {item["id"]: item for item in manifest["models"]}
    clouds = {frame: load_comparison_cloud(frame) for frame in frames}
    results = {frame: {} for frame in frames}
    started = datetime.now(timezone.utc).isoformat()
    for model_id in MODEL_IDS:
        item = items[model_id]
        checkpoint = Path(manifest["artifacts_root"]) / item["checkpoint"]
        runner = BEVOverviewPredictor(checkpoint, device=device, threads=threads,
                                      score=.3, nms=.1, max_detections=500)
        variant = validate_variant(item, runner)
        print(f"Strictly loaded {model_id}: epoch {variant['epoch']}; {checkpoint}", flush=True)
        geometry = runner.config["data"]["kitti"]["geometry"]
        for axis, lo, hi in (("x", ROI[0], ROI[1]), ("y", ROI[2], ROI[3])):
            if lo < geometry[f"{axis}_min"] or hi > geometry[f"{axis}_max"]:
                raise ValueError("Display crop extends outside the trained model domain")
        for frame, (points, source) in clouds.items():
            directory = output / "inference" / frame / model_id
            directory.mkdir(parents=True, exist_ok=True)
            rows, inference, heatmaps = runner.predict(points, capture_heatmaps=True)
            if not all(np.isfinite(value).all() for value in
                       (rows, heatmaps["logits"], heatmaps["probabilities"])):
                raise ValueError("Nonfinite model outputs")
            import torch
            np.testing.assert_array_equal(heatmaps["probabilities"],
                                          torch.sigmoid(torch.from_numpy(heatmaps["logits"])).numpy())
            detections = bev_detections(rows, runner.config["data"]["kitti"]["objects"])
            np.save(directory / "bev_predictions.npy", rows)
            np.savez_compressed(directory / "classification_heatmaps.npz", **heatmaps)
            write_json(directory / "config.inference.json", runner.config)
            names = ("bev_predictions.npy", "classification_heatmaps.npz",
                     "config.inference.json")
            metadata = {"id": model_id, "variant": variant, "inference": inference,
                        "started_utc": started, "prediction_source": "fresh model forward",
                        "detection_count": len(rows), "class_counts": dict(Counter(d.name for d in detections)),
                        "box_mode_displayed": "bev", "vertical_geometry_estimated": False,
                        "source_xyzi_sha256": source["points_xyzi_sha256"],
                        "files": {name: sha256(directory / name) for name in names}}
            write_json(directory / "metadata.json", metadata)
            results[frame][model_id] = {"detections": detections,
                                        "config": runner.config, "heatmaps": heatmaps,
                                        "metadata": metadata}
            print(f"  Fresh forward {frame}: {len(rows)} BEV detections; "
                  f"heatmap {heatmaps['probabilities'].shape}", flush=True)
        if sha256(checkpoint) != runner.metadata["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed during export")
        del runner
        gc.collect()
    for frame, (points, source) in clouds.items():
        binary = output / "inference" / frame / f"{frame}_original.bin"
        points.tofile(binary)
        if sha256(binary) != source["points_xyzi_sha256"]:
            raise ValueError("Saved source pointcloud differs from original bytes")
        write_json(binary.parent / "source.json", source)
    write_json(output / "manifest.used.json", manifest)
    return clouds, results


def draw_scene(ax, points, roi, *, ground_truth=None, prediction=None):
    scene = points[((points[:, :3] >= roi[::2]) & (points[:, :3] <= roi[1::2])).all(axis=1)]
    ax.scatter(-scene[:, 1], scene[:, 0], s=.13, color="#00d79d", alpha=.9,
               linewidths=0, rasterized=True, zorder=1)
    shown = []
    if ground_truth is not None:
        boxes = [(obj.label_index, obj.center[:2], obj.corners[:4, :2]) for obj in ground_truth]
    else:
        boxes = []
        for obj in prediction["detections"]:
            boxes.append((obj.index, obj.row[2:4], obj.footprint))
    for index, center, corners in boxes:
        if not (roi[0] <= center[0] <= roi[1] and roi[2] <= center[1] <= roi[3]):
            continue
        shown.append(index)
        closed = np.r_[corners, corners[:1]]
        ax.plot(-closed[:, 1], closed[:, 0], color=BOX_COLOR, lw=.5, zorder=3)
    ax.set_xlim(-roi[3], -roi[2])
    ax.set_ylim(roi[0], roi[1])
    ax.set_aspect("equal", adjustable="box")
    return shown


def render(output, clouds, predictions, gt_root, dpi, frames=FRAMES, gt_height=48.):
    plt = setup_matplotlib()
    plt.rcParams.update({"font.family": "DejaVu Serif", "svg.fonttype": "none"})
    width, outer, column_gap, pair_gap, label_band = 280., .6, 1., .8, 7.5
    column_width = (width - 2 * outer - column_gap) / 2
    panel_size = (column_width - pair_gap) / 2
    metric_aspect = (ROI[3] - ROI[2]) / (ROI[1] - ROI[0])
    row_heights = [gt_height, panel_size / metric_aspect, panel_size / metric_aspect]
    gt_width = row_heights[0] * metric_aspect
    height = 2 * outer + sum(row_heights) + 3 * label_band
    fig = plt.figure(figsize=(width / 25.4, height / 25.4), dpi=dpi, facecolor="white")
    visible, truths, sources = {}, {}, []
    for frame in frames:
        label = gt_root / "training/label_2" / f"{frame}.txt"
        calib = gt_root / "training/calib" / f"{frame}.txt"
        truths[frame] = read_objects(label, calib)
        sources.append({"frame": frame, "pointcloud": clouds[frame][1],
                        "ground_truth": {"label": str(label), "sha256": sha256(label),
                                         "calibration": str(calib), "calibration_sha256": sha256(calib)},
                        "models": {key: predictions[frame][key]["metadata"] for key in MODEL_IDS}})

    def panel(left, bottom, panel_width, panel_height):
        ax = fig.add_axes([left / width, bottom / height, panel_width / width, panel_height / height],
                          facecolor="black")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return ax

    top = height - outer
    for row, caption in enumerate(CAPTIONS):
        row_height = row_heights[row]
        bottom = top - row_height
        for column, frame in enumerate(frames):
            left = outer + column * (column_width + column_gap)
            points = clouds[frame][0]
            if row == 0:
                ax = panel(left + (column_width - gt_width) / 2, bottom, gt_width, row_height)
                visible[f"gt/{frame}"] = draw_scene(ax, points, ROI, ground_truth=truths[frame])
            else:
                model_id = MODEL_IDS[row - 1]
                ax = panel(left, bottom, panel_size, row_height)
                heat = panel(left + panel_size + pair_gap, bottom, panel_size, row_height)
                result = predictions[frame][model_id]
                visible[f"{model_id}/{frame}"] = draw_scene(ax, points, ROI, prediction=result)
                draw_heatmap(heat, result, ROI)
        fig.text(.5, (bottom - label_band / 2) / height, caption,
                 fontsize=16., ha="center", va="center", color="black")
        top = bottom - label_band
    stem = output / f"gt_mobilepixor_ours_bev_{frames[0]}_{frames[1]}_3x2"
    for suffix in ("png", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=dpi, facecolor="white", pad_inches=0)
    plt.close(fig)
    metadata = {"row_labels": list(CAPTIONS), "column_frames": list(frames),
                "layout": "3 rows x 2 scenes; each model scene is detection left, heatmap right",
                "canvas_mm": [width, height], "dpi": dpi, "roi_xyz": ROI.tolist(),
                "view": "BEV; X up, +Y left", "outer_margin_mm": outer,
                "panel_padding_mm": 0, "row_heights_mm": row_heights,
                "ground_truth_panel_width_mm": gt_width, "model_panel_width_mm": panel_size,
                "column_1_restored": frames[0],
                "column_2_replaced": {"previous": "001225", "current": frames[1]},
                "box_color": BOX_COLOR,
                "ground_truth_geometry": "BEV footprints of calibrated KITTI ground truth",
                "prediction_geometry": "exact model BEV footprints; no height or cuboids",
                "vertical_geometry_estimated": False,
                "heatmap": {"source": "sigmoid of actual model logits", "range": [0, 1],
                            "merge": "max across classes", "interpolation": "nearest", "smoothing": False},
                "ground_truth_used_for_inference": False, "extra_titles": False, "scores_displayed": False,
                "visible_indices": visible, "sources": sources,
                "script_sha256": sha256(Path(__file__)),
                "figures": {suffix: {"path": str(stem.with_suffix(f'.{suffix}')),
                                     "sha256": sha256(stem.with_suffix(f'.{suffix}'))}
                            for suffix in ("png", "svg")}}
    write_json(output / "metadata.json", metadata)
    print(f"Exported {stem}.png and .svg", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT / "Demo/configs/reference_bev_zoom.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "Demo/outputs/heatmap/reference_bev_003736_005663_red")
    parser.add_argument("--ground-truth-dir", type=Path, default=PROJECT / "Demo/data/kitti_groundtruth")
    parser.add_argument("--frames", nargs=2, default=list(FRAMES))
    parser.add_argument("--ground-truth-height-mm", type=float, default=48.)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if args.threads < 1 or args.dpi < 300:
        parser.error("Use positive threads and at least 300 DPI")
    if any(len(frame) != 6 or not frame.isdecimal() for frame in args.frames):
        parser.error("Frame IDs must have exactly six decimal digits")
    if not np.isfinite(args.ground_truth_height_mm) or not 0 < args.ground_truth_height_mm <= 99:
        parser.error("Ground truth height must be finite, positive and at most 99 mm")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    clouds, results = run_inference(args.manifest, output, args.device, args.threads, args.frames)
    render(output, clouds, results, args.ground_truth_dir.resolve(), args.dpi,
           args.frames, args.ground_truth_height_mm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
