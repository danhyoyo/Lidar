#!/usr/bin/env python3
"""Export four row_column folders, each containing detect.png and heatmap.png."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

import numpy as np

from export_kitti_models_only import load_verified
from export_kitti_reference_comparison import PROJECT, MODEL_IDS, BOX_COLOR, draw_scene
from export_kitti_model_comparison import draw_heatmap, write_json
from tools.visualization.export_kitti_overview import setup_matplotlib, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path,
                        default=PROJECT / "Demo/outputs/heatmap/models_only_003736_005663_red")
    parser.add_argument("--output-dir", type=Path,
                        default=PROJECT / "Demo/outputs/heatmap/individual_panels_003736_005663_red")
    args = parser.parse_args()
    source, output = args.input_dir.resolve(), args.output_dir.resolve()
    if source == output:
        parser.error("Use a separate directory for the individual panels")
    combined = json.loads((source / "metadata.json").read_text())
    inference_source = Path(combined["source_directory"])
    if sha256(inference_source / "metadata.json") != combined["source_metadata_sha256"]:
        raise ValueError("Source inference metadata changed")
    original, points, predictions = load_verified(inference_source)
    if original["column_frames"] != combined["column_frames"]:
        raise ValueError("Combined figure and inference disagree about the two scenes")
    frames = combined["column_frames"]
    roi = np.asarray(combined["roi_xyz"])
    width_mm, height_mm = combined["model_panel_mm"]
    dpi = combined["dpi"]
    plt = setup_matplotlib()
    output.mkdir(parents=True, exist_ok=True)
    folders = {}
    for row, model_id in enumerate(MODEL_IDS, 1):
        for column, frame in enumerate(frames, 1):
            name = f"{row}_{column}"
            directory = output / name
            directory.mkdir(exist_ok=True)
            files = {}
            for kind in ("detect", "heatmap"):
                figure = plt.figure(figsize=(width_mm / 25.4, height_mm / 25.4),
                                    dpi=dpi, facecolor="black")
                ax = figure.add_axes([0, 0, 1, 1], facecolor="black")
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                prediction = predictions[frame][model_id]
                if kind == "detect":
                    shown = draw_scene(ax, points[frame], roi, prediction=prediction)
                    if shown != combined["visible_indices"][f"{model_id}/{frame}"]:
                        raise ValueError("Individual panel changes the displayed predictions")
                else:
                    draw_heatmap(ax, prediction, roi)
                target = directory / f"{kind}.png"
                figure.savefig(target, dpi=dpi, facecolor="black", pad_inches=0)
                plt.close(figure)
                files[target.name] = {"sha256": sha256(target), "path": str(target)}
            folders[name] = {"row": row, "column": column, "model": model_id,
                             "frame": frame, "files": files,
                             "visible_prediction_indices": combined["visible_indices"][f"{model_id}/{frame}"]}
            print(f"{name}: {model_id}, frame {frame}; detect.png and heatmap.png", flush=True)
    write_json(output / "metadata.json", {
        "folder_naming": "row_column, one-based", "row_models": list(MODEL_IDS),
        "column_frames": frames, "folders": folders, "dpi": dpi,
        "panel_mm": [width_mm, height_mm], "box_color": BOX_COLOR,
        "roi_xyz": roi.tolist(), "heatmap": combined["heatmap"],
        "titles_or_captions_displayed": False, "outer_padding": 0,
        "inference_outputs_changed": False,
        "source_directory": str(source), "source_metadata_sha256": sha256(source / "metadata.json"),
        "script_sha256": sha256(Path(__file__)),
    })
    readme = (
        "Folder names use row_column from the two-model figure.\n\n"
        "1_1: MobilePIXOR, left scan 003736\n"
        "1_2: MobilePIXOR, right scan 005663\n"
        "2_1: Ours, left scan 003736\n"
        "2_2: Ours, right scan 005663\n\n"
        "Each folder contains detect.png and heatmap.png at 600 DPI.\n"
        "No text, captions or outer padding; red BEV footprints; raw grayscale probabilities.\n"
        "Predictions and heatmaps are unchanged verified model outputs.\n"
    )
    (output / "README.txt").write_text(readme)
    archive = output.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as saved:
        for directory in folders:
            for filename in ("detect.png", "heatmap.png"):
                saved.write(output / directory / filename, f"{directory}/{filename}")
        for filename in ("README.txt", "metadata.json"):
            saved.write(output / filename, filename)
    print(f"ZIP: {archive}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
