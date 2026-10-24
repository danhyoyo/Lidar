#!/usr/bin/env python3
"""Find a selected example with genuine Ours detection and heatmap advantages."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter, label, find_objects
import torch

from export_kitti_reference_comparison import PROJECT, ROI, load_comparison_cloud
from export_kitti_model_comparison import write_json, validate_variant
from select_kitti_qualitative_scene import ROOT as OURS_ROOT, ground_truth, measure
from tools.visualization.export_kitti_overview import sha256
from tools.visualization.predict_kitti_overview import BEVOverviewPredictor

ROOT = PROJECT / "Demo/outputs/heatmap/comparison_scene_search"
PEAK_THRESHOLD = .3
CENTER_RADIUS = 1.


def heatmap_measure(probabilities, config, objects):
    """Score the displayed merged raw probability map, without modifying it."""
    merged = probabilities.max(axis=0)
    geometry = config["data"]["kitti"]["geometry"]
    stride = config["data"]["out_size_factor"]
    dx, dy = geometry["x_res"] * stride, geometry["y_res"] * stride
    # Cell centers agree with imshow's native metric extent in the figure.
    x = geometry["x_min"] + (np.arange(merged.shape[1]) + .5) * dx
    y = geometry["y_min"] + (np.arange(merged.shape[0]) + .5) * dy
    xx, yy = np.meshgrid(x, y)
    roi = (xx >= ROI[0]) & (xx <= ROI[1]) & (yy >= ROI[2]) & (yy <= ROI[3])
    peaks = (merged == maximum_filter(merged, size=3, mode="constant", cval=-np.inf))
    peaks &= roi & (merged >= PEAK_THRESHOLD)
    components, count = label(peaks, structure=np.ones((3, 3), dtype=int))
    indices = []
    # Collapse equal neighboring maxima plateaus deterministically, without GT.
    for component, region in enumerate(find_objects(components), 1):
        if region is None:
            continue
        local = components[region] == component
        values = np.where(local, merged[region], -np.inf)
        row, column = np.unravel_index(np.argmax(values), values.shape)
        indices.append((row + region[0].start, column + region[1].start))
    indices.sort(key=lambda index: (-float(merged[index]), index))
    used, records = set(), []
    for row, column in indices:
        candidates = [(float(np.hypot(xx[row, column] - obj.center[0], yy[row, column] - obj.center[1])), index)
                      for index, obj in enumerate(objects) if index not in used]
        distance, index = min(candidates, default=(float("inf"), -1))
        match = distance <= CENTER_RADIUS
        if match:
            used.add(index)
        records.append({"cell_yx": [int(row), int(column)],
                        "xy": [float(xx[row, column]), float(yy[row, column])],
                        "probability": float(merged[row, column]), "true_positive": match,
                        "matched_gt_label_index": objects[index].label_index if match else None,
                        "center_distance_m": distance if np.isfinite(distance) else None})
    tp, fp, fn = len(used), len(indices) - len(used), len(objects) - len(used)
    foreground = np.zeros_like(roi)
    background = roi.copy()
    gt_peaks = []
    for obj in objects:
        distance = np.hypot(xx - obj.center[0], yy - obj.center[1])
        near = roi & (distance <= CENTER_RADIUS)
        foreground |= near
        background &= distance > 2 * CENTER_RADIUS
        gt_peaks.append(float(merged[near].max()) if near.any() else 0.)
    return {"gt": len(objects), "peaks": len(indices), "tp": tp, "fp": fp, "fn": fn,
            "errors": fp + fn, "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "mean_gt_peak_probability": float(np.mean(gt_peaks)) if gt_peaks else 0.,
            "foreground_probability_mass_fraction": float(merged[foreground].sum() / max(float(merged[roi].sum()), 1e-12)),
            "background_mean_probability": float(merged[background].mean()) if background.any() else 0.,
            "records": records}


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((PROJECT / "Demo/configs/reference_bev_zoom.json").read_text())
    items = {item["id"]: item for item in manifest["models"]}
    ours_checkpoint = Path(manifest["artifacts_root"]) / items["ours"]["checkpoint"]
    ours_hash = sha256(ours_checkpoint)
    ours_config = torch.load(ours_checkpoint, map_location="cpu", weights_only=True)["config"]
    mobile_checkpoint = Path(manifest["artifacts_root"]) / items["mobilepixor"]["checkpoint"]
    runner = BEVOverviewPredictor(mobile_checkpoint, device="cpu", threads=4, score=.3, nms=.1, max_detections=500)
    validate_variant(items["mobilepixor"], runner)
    write_json(ROOT / "ours.config.json", ours_config)
    write_json(ROOT / "mobilepixor.config.json", runner.config)
    candidates = json.loads((OURS_ROOT / "candidates.json").read_text())["frames"]
    results = []
    for item in candidates:
        frame = item["frame"]
        if frame == "003736":
            continue
        points, source = load_comparison_cloud(frame)
        objects = ground_truth(frame)
        previous = json.loads((OURS_ROOT / frame / "metrics.json").read_text())
        if previous["inference"]["checkpoint_sha256"] != ours_hash or previous["inference"]["checkpoint_epoch"] != 130:
            raise ValueError("Saved Ours outputs differ from the requested epoch 130 checkpoint")
        for filename, digest in previous["files"].items():
            if sha256(OURS_ROOT / frame / filename) != digest:
                raise ValueError("Saved actual Ours output hash mismatch")
        ours_rows = np.load(OURS_ROOT / frame / "ours_predictions.npy", allow_pickle=False)
        with np.load(OURS_ROOT / frame / "ours_heatmaps.npz", allow_pickle=False) as stored:
            ours_maps = {key: stored[key].copy() for key in stored.files}
        rows, inference, maps = runner.predict(points, capture_heatmaps=True)
        for tensors in (ours_maps, maps):
            if not np.isfinite(tensors["logits"]).all():
                raise ValueError("Nonfinite classification logits")
            np.testing.assert_array_equal(tensors["probabilities"],
                torch.sigmoid(torch.from_numpy(tensors["logits"])).numpy())
        if ours_maps["probabilities"].shape != maps["probabilities"].shape:
            raise ValueError("Models must have the same native output grid for this comparison")
        directory = ROOT / frame
        directory.mkdir(exist_ok=True)
        np.save(directory / "mobilepixor_predictions.npy", rows)
        np.savez_compressed(directory / "mobilepixor_heatmaps.npz", **maps)
        value = {"frame": frame, "source": source, "drive": item["drive"],
                 "ours_inference": previous["inference"], "mobilepixor_inference": inference,
                 "gt_class_count": len({obj.name for obj in objects}),
                 "ours": {"detection": measure(ours_rows, objects, ours_config["data"]["kitti"]["objects"]),
                          "heatmap": heatmap_measure(ours_maps["probabilities"], ours_config, objects)},
                 "mobilepixor": {"detection": measure(rows, objects, runner.config["data"]["kitti"]["objects"]),
                                "heatmap": heatmap_measure(maps["probabilities"], runner.config, objects)},
                 "files": {name: sha256(directory / name) for name in ("mobilepixor_predictions.npy", "mobilepixor_heatmaps.npz")}}
        od, md = value["ours"]["detection"], value["mobilepixor"]["detection"]
        oh, mh = value["ours"]["heatmap"], value["mobilepixor"]["heatmap"]
        value["detection_error_improvement"] = md["errors"] - od["errors"]
        value["heatmap_error_improvement"] = mh["errors"] - oh["errors"]
        value["joint_advantage"] = (od["errors"] < md["errors"] and oh["f1"] > mh["f1"] + 1e-9
                                     and oh["fp"] <= mh["fp"] and oh["fn"] <= mh["fn"])
        write_json(directory / "metrics.json", value)
        results.append(value)
        print(f"{frame}: detect ours {od['tp']}/{od['gt']} FP={od['fp']} vs mobile {md['tp']}/{md['gt']} FP={md['fp']}; "
              f"heat F1 {oh['f1']:.3f} vs {mh['f1']:.3f}; joint={value['joint_advantage']}", flush=True)
    results.sort(key=lambda value: (
        not value["joint_advantage"], value["ours"]["detection"]["errors"],
        value["ours"]["heatmap"]["errors"],
        -(value["detection_error_improvement"] + value["heatmap_error_improvement"]),
        -value["gt_class_count"], -value["ours"]["detection"]["gt"], value["frame"]))
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "candidate_count": len(results),
              "left_frame_excluded": "003736", "best_frame": results[0]["frame"],
              "joint_advantage_found": any(value["joint_advantage"] for value in results),
              "selection": "require fewer detection errors and higher heatmap F1 without more heatmap FP/FN; then minimize Ours detection/heatmap errors and maximize joint improvement",
              "scope": "selected qualitative comparison from tested validation scenes, not an unbiased benchmark",
              "score_threshold": .3, "nms_threshold": .1, "roi_xyz": ROI.tolist(),
              "heatmap_metric": {"map": "max across classes of unmodified sigmoid logits", "peak_threshold": PEAK_THRESHOLD,
                                 "local_max_window_cells": 3, "gt_center_match_radius_m": CENTER_RADIUS,
                                 "matching": "descending peak probability, closest unused GT center", "plateaus": "one deterministic peak per connected plateau",
                                 "cell_coordinates": "native imshow cell centers"},
              "rendering": {"gray_range": [0, 1], "interpolation": "nearest", "smoothing": False},
              "ranked": results, "script_sha256": sha256(Path(__file__))}
    write_json(ROOT / "ranking.json", report)
    with (ROOT / "ranking.csv").open("w") as stream:
        fields = ["frame", "joint_advantage", "ours_det_tp", "ours_det_fp", "ours_det_fn", "mobile_det_tp", "mobile_det_fp", "mobile_det_fn",
                  "ours_heat_tp", "ours_heat_fp", "ours_heat_fn", "mobile_heat_tp", "mobile_heat_fp", "mobile_heat_fn"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for value in results:
            row = {"frame": value["frame"], "joint_advantage": value["joint_advantage"]}
            for name, source in (("ours", "ours"), ("mobile", "mobilepixor")):
                for metric, key in (("det", "detection"), ("heat", "heatmap")):
                    for field in ("tp", "fp", "fn"):
                        row[f"{name}_{metric}_{field}"] = value[source][key][field]
            writer.writerow(row)
    print("BEST", results[0]["frame"], "JOINT ADVANTAGE", report["joint_advantage_found"], flush=True)


if __name__ == "__main__":
    main()
