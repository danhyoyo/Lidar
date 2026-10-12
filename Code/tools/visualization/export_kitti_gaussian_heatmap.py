#!/usr/bin/env python3
"""Export KITTI Gaussian targets and the checkpoint's actual classification heatmaps.

Panels keep the original BEV cells: no blur, synthetic centers, or normalization
by image maxima. Gaussian targets reuse Dataset.update_cls_map unchanged.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.kitti_training_pipeline.prepare_kitti import convert_labels
from tools.visualization.export_kitti_overview import dataset_paths, read_points, setup_matplotlib, sha256
from tools.visualization.predict_kitti_overview import infer_frame


def configure_imports():
    sys.path.insert(0, str(ROOT / "tools/kitti_training_pipeline"))
    from common import configure_detector_imports
    configure_detector_imports(ROOT / "detector")


def grid_shape(config):
    """Return X,Y output dimensions using the same grid calculation as Dataset."""
    configure_imports()
    from core.bev_encoding import resolve_bev_encoding
    data = config["data"]
    geometry = data["kitti"]["geometry"]
    stride = data["out_size_factor"]
    encoding = data.get("bev_encoding", {})
    if encoding.get("name") == "hist14":
        schema = resolve_bev_encoding(encoding, geometry)
        return [size // stride for size in schema.grid_shape[:2]]
    return [int((geometry[f"{axis}_max"] - geometry[f"{axis}_min"])
                / geometry[f"{axis}_res"] / stride) for axis in ("x", "y")]


def build_gaussian_targets(boxes, config):
    """Build full [C,Y,X] targets directly with the training Gaussian writer."""
    import torch
    if config["model"].get("cls_encoding", "gaussian") != "gaussian":
        raise ValueError("Gaussian targets require cls_encoding=gaussian")
    configure_imports()
    from core.datasets.dataset import Dataset
    from core.task_groups import validate_objects
    boxes = np.asarray(boxes, dtype=np.float32)
    objects = config["data"]["kitti"]["objects"]
    validate_objects(objects)
    if (boxes.ndim != 2 or boxes.shape[1] != 8 or not np.isfinite(boxes).all()
            or (boxes[:, 1:4] <= 0).any() or (boxes[:, 0] != np.floor(boxes[:, 0])).any()
            or (boxes[:, 0] < 0).any() or (boxes[:, 0] >= len(objects)).any()):
        raise ValueError("Expected finite [class,h,w,l,x,y,z,yaw] boxes with positive dimensions")
    # Only the existing classification writer is needed; no augmentation or regression targets.
    dataset = Dataset.__new__(Dataset)
    dataset.config = config["data"]
    dataset.cls_encoding = "gaussian"
    dataset.out_size_factor = dataset.config["out_size_factor"]
    dataset.output_shape = grid_shape(config)
    accepted = dataset.filter_boxes(boxes, "kitti")
    internal = torch.zeros((len(objects), *dataset.output_shape), dtype=torch.float32)
    geometry = dataset.config["kitti"]["geometry"]
    for box in accepted:
        dataset.update_cls_map(internal, box, geometry)
    return internal.permute(0, 2, 1).numpy(), accepted.numpy()


def crop_heatmaps(heatmaps, config, roi):
    """Snap a metric ROI outward to existing cells, preserving [C,Y,X] orientation."""
    geometry = config["data"]["kitti"]["geometry"]
    stride = config["data"]["out_size_factor"]
    maps = np.asarray(heatmaps)
    if maps.ndim != 3:
        raise ValueError("Heatmaps must have shape [C,Y,X]")
    if len(roi) != 4 or not np.isfinite(roi).all() or roi[0] >= roi[1] or roi[2] >= roi[3]:
        raise ValueError("ROI must be finite XMIN XMAX YMIN YMAX with increasing bounds")
    bounds, extent = {}, []
    for axis, low, high, size in [("x", roi[0], roi[1], maps.shape[2]),
                                  ("y", roi[2], roi[3], maps.shape[1])]:
        origin = geometry[f"{axis}_min"]
        step = geometry[f"{axis}_res"] * stride
        first = max(0, int(np.floor((low - origin) / step + 1e-10)))
        last = min(size, int(np.ceil((high - origin) / step - 1e-10)))
        if first >= last:
            raise ValueError("ROI does not intersect the heatmap grid")
        bounds[axis] = [first, last]
        extent.extend([origin + first * step, origin + last * step])
    x0, x1 = bounds["x"]
    y0, y1 = bounds["y"]
    return maps[:, y0:y1, x0:x1], extent, bounds


def render_heatmap(matrix, path, *, extent, cmap, size, dpi):
    """Axes-free publication panel with the same fixed 0..1 scale for every map."""
    plt = setup_matplotlib()
    fig = plt.figure(figsize=(size / dpi, size / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(matrix, origin="lower", extent=extent, cmap=cmap, vmin=0, vmax=1,
              interpolation="nearest", aspect="auto")
    ax.set_axis_off()
    for suffix in ("png", "svg"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=dpi, pad_inches=0)
    plt.close(fig)


def render_preview(series, output, *, frame, extent, class_order, cmap):
    plt = setup_matplotlib()
    fig, axes = plt.subplots(len(series), len(class_order) + 1,
                             figsize=(15, 4.3 * len(series)), squeeze=False)
    columns = ["All classes (max)", *class_order]
    for row, (title, maps) in enumerate(series):
        matrices = [maps.max(axis=0), *maps]
        for column, (name, matrix) in enumerate(zip(columns, matrices)):
            ax = axes[row, column]
            image = ax.imshow(matrix, origin="lower", extent=extent, cmap=cmap,
                              vmin=0, vmax=1, interpolation="nearest", aspect="equal")
            ax.set_title(f"{title}\n{name}", fontsize=11)
            ax.set_xlabel("LiDAR x (m)")
            ax.set_ylabel("LiDAR y (m)")
            fig.colorbar(image, ax=ax, fraction=.035, pad=.03, ticks=[0, .5, 1])
    fig.suptitle(f"KITTI {frame} classification heatmaps — fixed scale [0, 1]", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(output / "heatmap_preview.png", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path,
                        default=ROOT.parent / "Result/artifacts/kitti_overview_model",
                        help="Previous inference export; supplies frame, checkpoint and source hashes")
    parser.add_argument("--checkpoint", type=Path, help="Override checkpoint file or selected directory")
    parser.add_argument("--frame", help="Default: frame recorded in prediction-dir")
    parser.add_argument("--kitti-root", type=Path, default=ROOT / "Kitti_Raw")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT.parent / "Result/artifacts/kitti_overview_heatmap")
    parser.add_argument("--source", choices=("both", "model", "target"), default="both")
    parser.add_argument("--roi", type=float, nargs=4, default=[0, 45, -24, 24],
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX"))
    parser.add_argument("--full-grid", action="store_true", help="Render the entire model XY grid")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap; viridis matches the reference")
    parser.add_argument("--size", type=int, default=1200)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if min(args.size, args.dpi, args.threads) <= 0:
        parser.error("size, dpi and threads must be positive")
    previous_path = args.prediction_dir.expanduser().resolve() / "metadata.json"
    previous = json.loads(previous_path.read_text()) if previous_path.is_file() else None
    if args.checkpoint is None and previous is None:
        parser.error("Provide --checkpoint or an existing --prediction-dir")
    checkpoint = (args.checkpoint.expanduser().resolve() if args.checkpoint else
                  Path(previous["inference"]["checkpoint"]))
    if checkpoint.is_dir():
        checkpoint /= "best.pt"
    frame = args.frame or (previous["frame_id"] if previous else "005663")
    if len(frame) != 6 or not frame.isdigit():
        parser.error("frame must be a six-digit KITTI ID")
    output = args.output_dir.expanduser().resolve()
    if output == args.prediction_dir.expanduser().resolve():
        parser.error("Use a separate output-dir to preserve the inference export")
    output.mkdir(parents=True, exist_ok=True)
    paths = dataset_paths(args.kitti_root)
    bin_path = paths.velodyne / f"{frame}.bin"
    source_bin = {"path": str(bin_path), "sha256": sha256(bin_path)}
    same_export = previous is not None and frame == previous["frame_id"] and args.checkpoint is None
    if same_export:
        if source_bin["sha256"] != previous["source_bin"]["sha256"]:
            raise ValueError("The LiDAR source differs from the previous inference export")
        if sha256(checkpoint) != previous["inference"]["checkpoint_sha256"]:
            raise ValueError("The checkpoint differs from the previous inference export")
    series, inference = [], None
    if args.source in ("both", "model"):
        print(f"Inferring actual classification heatmaps for KITTI {frame}", flush=True)
        settings = previous["inference"] if same_export else {}
        rows, config, inference = infer_frame(checkpoint, None, read_points(bin_path),
            device=args.device, threads=args.threads, score=settings.get("score_threshold", .3),
            nms=settings.get("nms_threshold", .1), max_detections=settings.get("max_detections", 500),
            heatmap_output=output / "model_heatmaps.npz")
        if same_export:
            if inference["config_sha256"] != previous["inference"]["config_sha256"]:
                raise ValueError("Embedded configuration differs from the previous inference export")
            old_rows = np.load(args.prediction_dir.expanduser().resolve() / f"{frame}_bev_predictions.npy")
            np.testing.assert_allclose(rows, old_rows, rtol=1e-5, atol=1e-4)
        with np.load(output / "model_heatmaps.npz") as archive:
            model_maps = archive["probabilities"]
        series.append(("Model prediction", model_maps, "gaussian_heatmap"))
    else:
        import torch
        from tools.visualization.predict_kitti_overview import inference_config
        config = inference_config(torch.load(checkpoint, map_location="cpu", weights_only=True))
    class_ids = config["data"]["kitti"]["objects"]
    class_order = sorted(class_ids, key=class_ids.get)
    target_sources, target_counts = None, None
    if args.source in ("both", "target"):
        label_path = paths.labels / f"{frame}.txt"
        calib_path = paths.calib / f"{frame}.txt"
        lines, _ = convert_labels(label_path, calib_path)
        boxes = np.asarray([[class_ids[fields[0]], *map(float, fields[1:])]
                            for fields in map(str.split, lines) if fields[0] in class_ids],
                           dtype=np.float32).reshape(-1, 8)
        targets, accepted = build_gaussian_targets(boxes, config)
        target_counts = dict(Counter(class_order[int(box[0])] for box in accepted))
        np.savez_compressed(output / "gaussian_targets.npz", heatmaps=targets, boxes=accepted,
                            class_order=np.asarray(class_order))
        target_sources = {name: {"path": str(path), "sha256": sha256(path)}
                          for name, path in [("labels", label_path), ("calibration", calib_path)]}
        series.append(("KITTI Gaussian target", targets, "gaussian_target"))
    geometry = config["data"]["kitti"]["geometry"]
    roi = ([geometry["x_min"], geometry["x_max"], geometry["y_min"], geometry["y_max"]]
           if args.full_grid else args.roi)
    preview_series, rendered = [], {}
    for title, maps, prefix in series:
        if maps.shape != (len(class_order), *reversed(grid_shape(config))):
            raise ValueError("Heatmap shape does not match the checkpoint output grid")
        cropped, extent, indices = crop_heatmaps(maps, config, roi)
        render_heatmap(cropped.max(axis=0), output / prefix, extent=extent,
                       cmap=args.cmap, size=args.size, dpi=args.dpi)
        for name, matrix in zip(class_order, cropped):
            render_heatmap(matrix, output / f"{prefix}_{name.lower()}", extent=extent,
                           cmap=args.cmap, size=args.size, dpi=args.dpi)
        np.savez_compressed(output / f"{prefix}_crop.npz", heatmaps=cropped,
                            merged=cropped.max(axis=0), extent=np.asarray(extent),
                            class_order=np.asarray(class_order))
        rendered[prefix] = {"source": title, "shape_cyx": list(maps.shape),
            "crop_shape_cyx": list(cropped.shape), "roi_cell_indices": indices,
            "render_extent_xy": extent, "max_per_class": maps.max(axis=(1, 2)).tolist()}
        preview_series.append((title, cropped))
    render_preview(preview_series, output, frame=frame, extent=extent, class_order=class_order, cmap=args.cmap)
    (output / "config.inference.json").write_text(json.dumps(config, indent=2) + "\n")
    metadata = {"frame_id": frame, "source_bin": source_bin, "class_order": class_order,
        "tensor_axes": ["class", "y", "x"], "origin": "lower", "colormap": args.cmap,
        "fixed_color_scale": [0, 1], "interpolation": "nearest", "merged_operator": "max across classes",
        "output_cell_size_xy_metres": [geometry[f"{axis}_res"] * config["data"]["out_size_factor"]
                                      for axis in ("x", "y")],
        "gaussian_overlap": config["data"]["gaussian_overlap"],
        "min_radius_cells": config["data"]["min_radius"],
        "gaussian_sigma": "(2 * radius + 1) / 6; radius from training gaussian_radius",
        "ground_truth_used_for_model_inference": False,
        "target_sources": target_sources, "target_counts_in_model_roi": target_counts,
        "target_augmentation_applied": False, "inference": inference, "panels": rendered,
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "matching_previous_detection_verified": bool(same_export and inference is not None),
        "previous_inference_metadata": str(previous_path) if same_export else None,
        "split_membership": previous.get("split_membership") if same_export else None,
        "reproduce_command": [sys.executable, *sys.argv]}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {len(series)} heatmap sets for frame {frame} to {output}", flush=True)
    print(f"Full [C,Y,X]: {rendered[next(iter(rendered))]['shape_cyx']}; GT counts: {target_counts}")


if __name__ == "__main__":
    main()
