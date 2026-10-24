#!/usr/bin/env python3
"""Render verified scans and saved real predictions in a borderless A4 layout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "Code"))
from tools.visualization.export_kitti_overview import (
    EDGES, camera_basis, canvas_limits, projected, read_points, setup_matplotlib, sha256,
)
from tools.visualization.predict_kitti_overview import bev_detections


def load_verified_frame(root, frame):
    directory = root / frame
    metadata = json.loads((directory / "metadata.json").read_text())
    binary = directory / f"{frame}_recovered_original.bin"
    if sha256(binary) != metadata["source"]["recorded_original_bin_sha256"]:
        raise ValueError(f"Original scan hash mismatch: {frame}")
    points = read_points(binary)
    outputs, provenance = {}, {}
    for model_id in ("mobilepixor", "ours"):
        model_dir = directory / model_id
        saved = json.loads((model_dir / "metadata.json").read_text())
        inference = saved["inference"]
        if not inference["strict_state_dict"] or not inference["checkpoint_identity_validated"]:
            raise ValueError("Predictions must come from validated strict checkpoint inference")
        if sha256(inference["checkpoint"]) != inference["checkpoint_sha256"]:
            raise ValueError("Source checkpoint has changed")
        for filename in ("bev_predictions.npy", "lidar_lifted_cuboids.npz", "config.inference.json"):
            if sha256(model_dir / filename) != saved["files"][filename]:
                raise ValueError(f"Prediction artifact hash mismatch: {frame}/{model_id}/{filename}")
        parent = next(item for item in metadata["models"] if item["id"] == model_id)
        if saved["files"] != parent["files"]:
            raise ValueError("Prediction artifacts disagree with the original frame export")
        config = json.loads((model_dir / "config.inference.json").read_text())
        rows = np.load(model_dir / "bev_predictions.npy", allow_pickle=False)
        detections = bev_detections(rows, config["data"]["kitti"]["objects"])
        with np.load(model_dir / "lidar_lifted_cuboids.npz", allow_pickle=False) as archive:
            cuboids = {int(index): corners.copy() for index, corners in
                       zip(archive["prediction_indices"], archive["corners"])}
        outputs[model_id] = (detections, cuboids)
        provenance[model_id] = {"inference": inference, "prediction_hashes": saved["files"],
                                "original_export": str(model_dir)}
    return points, outputs, {"frame": frame, "source": metadata["source"],
                              "recovered_binary": str(binary), "models": provenance}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=PROJECT / "Demo/outputs/heatmap/model_comparison")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT / "Demo/outputs/heatmap/origin_mobilepixor_ours_a4")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args(argv)
    if args.dpi < 300:
        parser.error("Use at least 300 DPI for the A4 figure")
    frames = ["000134", "005663"]
    data = [load_verified_frame(args.input_dir.resolve(), frame) for frame in frames]
    plt = setup_matplotlib()
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    width_mm, height_mm, gutter_mm, label_mm = 210., 297., .6, 8.
    panel_width = (width_mm - gutter_mm) / 2
    row_height = height_mm / 3
    panel_height = row_height - label_mm
    figure = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4),
                         dpi=args.dpi, facecolor="white")
    basis = camera_basis(54., 180.)
    roi = np.array([0., 45., -24., 24., -2.8, 4.])
    drawing = []
    for points, _, _ in data:
        scene = points[((points[:, :3] >= roi[::2]) & (points[:, :3] <= roi[1::2])).all(axis=1)]
        uv = projected(scene, basis)
        limits = canvas_limits(scene, basis, panel_width / panel_height, margin=.002)
        order = np.argsort(uv[:, 2], kind="stable")
        drawing.append((uv[order], limits))
    visible_boxes = {}
    for row, label in enumerate(("origin", "MobilePIXOR", "Ours")):
        top = height_mm - row * row_height
        bottom = top - panel_height
        for column, (points, predictions, _) in enumerate(data):
            left = column * (panel_width + gutter_mm)
            ax = figure.add_axes([left / width_mm, bottom / height_mm,
                                  panel_width / width_mm, panel_height / height_mm], facecolor="black")
            uv, limits = drawing[column]
            ax.scatter(uv[:, 0], uv[:, 1], s=.22, c="#00d79d", alpha=.9,
                       linewidths=0, rasterized=True, zorder=1)
            visible = []
            if row:
                model_id = "mobilepixor" if row == 1 else "ours"
                detections, cuboids = predictions[model_id]
                for obj in detections:
                    if not (roi[0] <= obj.row[2] <= roi[1] and roi[2] <= obj.row[3] <= roi[3]):
                        continue
                    visible.append(obj.index)
                    if obj.index in cuboids:
                        corners = cuboids[obj.index]
                        edges, style = EDGES, "-"
                    else:
                        corners = np.c_[obj.footprint, np.full(4, -1.7)]
                        edges, style = [(i, (i + 1) % 4) for i in range(4)], "--"
                    uv_box = projected(corners, basis)
                    for a, b in edges:
                        ax.plot(uv_box[[a, b], 0], uv_box[[a, b], 1],
                                color="#ffdf40", lw=.85, linestyle=style, zorder=3)
                visible_boxes[f"{model_id}/{frames[column]}"] = visible
            ax.set_xlim(limits[:2])
            ax.set_ylim(limits[2:])
            ax.set_aspect("equal", adjustable="box")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        figure.text(.5, (bottom - label_mm / 2) / height_mm, label,
                    fontsize=17, fontweight="bold", ha="center", va="center", color="black")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stem = output / "origin_mobilepixor_ours_3x2_a4"
    for suffix in ("png", "svg"):
        figure.savefig(stem.with_suffix(f".{suffix}"), dpi=args.dpi, facecolor="white", pad_inches=0)
    plt.close(figure)
    metadata = {"layout": "3 rows x 2 columns", "page_mm": [width_mm, height_mm],
                "dpi": args.dpi, "column_frames": frames, "row_labels": ["origin", "MobilePIXOR", "Ours"],
                "panel_size_mm": [panel_width, panel_height], "column_gap_mm": gutter_mm,
                "label_band_mm": label_mm, "outer_margin_mm": 0,
                "heatmaps_displayed": False, "scores_displayed": False, "titles_displayed": False,
                "camera": {"elevation": 54., "azimuth": 180.}, "roi_xyz": roi.tolist(),
                "predictions": "verified saved outputs from actual checkpoint inference",
                "vertical_geometry": "LiDAR-supported cuboids; dashed BEV footprints if unsupported",
                "visible_prediction_indices": visible_boxes, "sources": [item[2] for item in data],
                "script_sha256": sha256(Path(__file__)),
                "figures": {suffix: {"path": str(stem.with_suffix(f'.{suffix}')),
                                     "sha256": sha256(stem.with_suffix(f'.{suffix}'))}
                            for suffix in ("png", "svg")}}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(f"Exported {stem}.png and .svg", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
