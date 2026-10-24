#!/usr/bin/env python3
"""Render two model rows from verified real inference, preserving the GT figure."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from export_kitti_reference_comparison import PROJECT, MODEL_IDS, BOX_COLOR, draw_scene
from export_kitti_model_comparison import draw_heatmap, write_json
from tools.visualization.export_kitti_overview import read_points, setup_matplotlib, sha256
from tools.visualization.predict_kitti_overview import bev_detections


def load_verified(source):
    metadata = json.loads((source / "metadata.json").read_text())
    frames = metadata["column_frames"]
    if len(frames) != 2 or metadata["box_color"] != BOX_COLOR:
        raise ValueError("Expected two scenes and the same box color as the source figure")
    points, predictions = {}, {}
    for scene in metadata["sources"]:
        frame = scene["frame"]
        directory = source / "inference" / frame
        binary = directory / f"{frame}_original.bin"
        if sha256(binary) != scene["pointcloud"]["points_xyzi_sha256"]:
            raise ValueError(f"Pointcloud hash mismatch: {frame}")
        points[frame] = read_points(binary)
        predictions[frame] = {}
        for model_id in MODEL_IDS:
            stored = scene["models"][model_id]
            inference = stored["inference"]
            if not inference["strict_state_dict"] or not inference["checkpoint_identity_validated"]:
                raise ValueError("Expected actual strictly validated checkpoint inference")
            if sha256(inference["checkpoint"]) != inference["checkpoint_sha256"]:
                raise ValueError("Source checkpoint changed")
            if model_id == "ours" and inference["checkpoint_epoch"] != 130:
                raise ValueError("Ours must use the requested epoch 130")
            model_dir = directory / model_id
            for name, expected in stored["files"].items():
                if sha256(model_dir / name) != expected:
                    raise ValueError(f"Saved inference hash mismatch: {frame}/{model_id}/{name}")
            config = json.loads((model_dir / "config.inference.json").read_text())
            rows = np.load(model_dir / "bev_predictions.npy", allow_pickle=False)
            with np.load(model_dir / "classification_heatmaps.npz", allow_pickle=False) as archive:
                maps = {key: archive[key].copy() for key in archive.files}
            np.testing.assert_array_equal(maps["probabilities"], torch.sigmoid(torch.from_numpy(maps["logits"])).numpy())
            predictions[frame][model_id] = {
                "config": config, "heatmaps": maps,
                "detections": bev_detections(rows, config["data"]["kitti"]["objects"]),
            }
    return metadata, points, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=PROJECT / "Demo/outputs/heatmap/reference_bev_003736_005663_red")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT / "Demo/outputs/heatmap/models_only_003736_005663_red")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    if args.dpi < 300:
        parser.error("Use at least 300 DPI")
    source = args.input_dir.resolve()
    output = args.output_dir.resolve()
    if output == source:
        parser.error("Use a separate output directory to preserve the original figure")
    original, points, results = load_verified(source)
    frames = original["column_frames"]
    roi = np.asarray(original["roi_xyz"])
    captions = ("A) MobilePIXOR", "B) Ours")
    width, outer, column_gap, pair_gap, label_band = 280., .6, 1., .8, 7.5
    column_width = (width - 2 * outer - column_gap) / 2
    panel_width = (column_width - pair_gap) / 2
    panel_height = panel_width * (roi[1] - roi[0]) / (roi[3] - roi[2])
    height = 2 * outer + 2 * (panel_height + label_band)
    plt = setup_matplotlib()
    plt.rcParams.update({"font.family": "DejaVu Serif", "svg.fonttype": "none"})
    figure = plt.figure(figsize=(width / 25.4, height / 25.4), dpi=args.dpi, facecolor="white")
    shown = {}

    def panel(left, bottom):
        ax = figure.add_axes([left / width, bottom / height, panel_width / width,
                              panel_height / height], facecolor="black")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        return ax

    for row, (model_id, caption) in enumerate(zip(MODEL_IDS, captions)):
        bottom = height - outer - row * (panel_height + label_band) - panel_height
        for column, frame in enumerate(frames):
            left = outer + column * (column_width + column_gap)
            ax, heat = panel(left, bottom), panel(left + panel_width + pair_gap, bottom)
            prediction = results[frame][model_id]
            shown[f"{model_id}/{frame}"] = draw_scene(ax, points[frame], roi, prediction=prediction)
            draw_heatmap(heat, prediction, roi)
        figure.text(.5, (bottom - label_band / 2) / height, caption,
                    fontsize=16., ha="center", va="center", color="black")
    output.mkdir(parents=True, exist_ok=True)
    stem = output / f"mobilepixor_ours_bev_{frames[0]}_{frames[1]}_2x2"
    for extension in ("png", "svg"):
        figure.savefig(stem.with_suffix(f".{extension}"), dpi=args.dpi, facecolor="white", pad_inches=0)
    plt.close(figure)
    write_json(output / "metadata.json", {
        "layout": "2 model rows x 2 scenes; detection left, heatmap right in each scene",
        "row_labels": list(captions), "column_frames": frames,
        "ground_truth_displayed": False, "ground_truth_used_for_inference": False,
        "predictions": "unchanged verified outputs of original real checkpoint inference",
        "box_color": BOX_COLOR, "roi_xyz": roi.tolist(), "dpi": args.dpi,
        "canvas_mm": [width, height], "model_panel_mm": [panel_width, panel_height],
        "heatmap": original["heatmap"], "visible_indices": shown,
        "source_directory": str(source), "source_metadata_sha256": sha256(source / "metadata.json"),
        "models": {scene["frame"]: scene["models"] for scene in original["sources"]},
        "script_sha256": sha256(Path(__file__)),
        "figures": {extension: {"path": str(stem.with_suffix(f'.{extension}')),
                                "sha256": sha256(stem.with_suffix(f'.{extension}'))}
                    for extension in ("png", "svg")},
    })
    print(f"Exported {stem}.png and .svg", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
