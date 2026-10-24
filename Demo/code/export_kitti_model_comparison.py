#!/usr/bin/env python3
"""Run six real checkpoints and render three rows of LiDAR/heatmap pairs.

Backup inputs are accepted only if their complete float32 XYZI bytes match the
recorded original KITTI binary hash. No cached predictions or labels are read.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "Code"))
from tools.visualization.export_kitti_overview import (
    EDGES, camera_basis, canvas_limits, projected, read_points, roi_vertices,
    setup_matplotlib, sha256,
)
from tools.visualization.predict_kitti_overview import BEVOverviewPredictor, bev_detections, points_in_bev
from lift_kitti_bev_to_3d import lift_object, local_ground


BACKUPS = {
    "000134": ("Demo/outputs/backup/attention_placements/000134/tensors/input.npz",
               "points_xyzi", "Demo/outputs/backup/attention_placements/000134/metadata.json"),
    "005663": ("Demo/outputs/backup/pipeline/005663/tensors/pointcloud_stages.npz",
               "raw", "Demo/outputs/backup/pipeline/005663/metadata.json"),
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def load_cloud(frame, points_dir=None):
    if points_dir is not None:
        path = points_dir / f"{frame}.bin"
        points = read_points(path)
        provenance = {"format": "KITTI binary XYZI", "path": str(path.resolve())}
    else:
        if frame not in BACKUPS:
            raise ValueError(f"No verified backup for {frame}; provide --points-dir")
        archive, key, metadata = BACKUPS[frame]
        path, metadata_path = PROJECT / archive, PROJECT / metadata
        with np.load(path, allow_pickle=False) as saved:
            points = saved[key].copy()
        old = json.loads(metadata_path.read_text())
        expected = old.get("cloud_sha256") or old["source"]["bin_sha256"]
        actual = hashlib.sha256(points.tobytes(order="C")).hexdigest()
        if points.dtype != np.dtype("<f4") or actual != expected:
            raise ValueError("Backup point bytes differ from recorded original KITTI binary")
        provenance = {"format": "verified original XYZI recovered from NPZ",
                      "path": str(path), "npz_key": key, "archive_sha256": sha256(path),
                      "metadata_path": str(metadata_path), "metadata_sha256": sha256(metadata_path),
                      "recorded_original_bin_sha256": expected, "original_bin_hash_matches": True}
    if points.ndim != 2 or points.shape[1] != 4 or not len(points) or not np.isfinite(points).all():
        raise ValueError("Expected nonempty finite float32 XYZI pointcloud")
    provenance["point_count"] = len(points)
    provenance["points_xyzi_sha256"] = hashlib.sha256(points.tobytes(order="C")).hexdigest()
    ids = (PROJECT / "paper/research/val_ids.txt").read_text().splitlines()
    provenance["split"] = "validation" if frame in {line.split(";")[0] for line in ids} else "unknown"
    if frame == "005663":
        provenance["split"] = "training"
    return points, provenance


def validate_variant(item, runner):
    config = runner.config
    actual = {"backbone": config["model"]["backbone"], "encoding": runner.schema.name,
              "loss": config["loss"]["name"], "use_iou": runner.spec.use_iou,
              "epoch": runner.metadata["checkpoint_epoch"]}
    aug = config.get("augmentation", {})
    names = [entry.get("NAME") for entry in aug.get("AUG_CONFIG_LIST", [])]
    if "hybrid_gt_sampling" in names:
        actual["augmentation"] = "hybrid_gt_sampling"
    for key, value in item["expected"].items():
        if actual.get(key) != value:
            raise ValueError(f"{item['id']} expected {key}={value!r}, found {actual.get(key)!r}")
    if not runner.metadata["checkpoint_identity_validated"]:
        raise ValueError("Comparison requires saved checkpoint identity metadata")
    return actual


def lift_predictions(points, detections, z_bounds):
    """Estimate vertical geometry only where actual foreground supports it."""
    occupied = np.zeros(len(points), dtype=bool)
    for obj in detections:
        occupied |= points_in_bev(points, obj)
    lifted, reports = {}, []
    for obj in detections:
        try:
            plane, ground = local_ground(points, obj, occupied, z_bounds=z_bounds)
            result, report = lift_object(points, obj, plane, z_bounds=z_bounds)
            report.update(ground)
            if result is not None:
                lifted[obj.index] = result
        except ValueError as error:
            report = {"prediction_index": obj.index, "class": obj.name,
                      "status": "insufficient_ground_support", "reason": str(error)}
        reports.append(report)
    return lifted, reports


def draw_cloud(ax, points, result, roi, view):
    x0, x1, y0, y1, z0, z1 = roi
    lower, upper = np.asarray(roi[::2]), np.asarray(roi[1::2])
    scene = points[((points[:, :3] >= lower) & (points[:, :3] <= upper)).all(axis=1)]
    ax.set_facecolor("black")
    if view == "bev":
        uv = np.c_[-scene[:, 1], scene[:, 0], scene[:, 2]]
        limits = [-y1, -y0, x0, x1]
        transform = lambda corners: np.c_[-corners[:, 1], corners[:, 0]]
    else:
        basis = camera_basis(54., 180.)
        uv = projected(scene, basis)
        limits = canvas_limits(roi_vertices(roi), basis, (y1 - y0) / (x1 - x0), margin=.015)
        transform = lambda corners: projected(corners, basis)[:, :2]
    # Every plotted sample is an original measured LiDAR return.
    order = np.argsort(uv[:, 2], kind="stable")
    ax.scatter(uv[order, 0], uv[order, 1], s=.24, c="#00d79d", alpha=.85,
               linewidths=0, rasterized=True, zorder=1)
    visible = []
    for obj in result["detections"]:
        if not (x0 <= obj.row[2] <= x1 and y0 <= obj.row[3] <= y1):
            continue
        visible.append(obj.index)
        cuboid = result["lifted"].get(obj.index) if view != "bev" else None
        if cuboid is not None:
            corners = transform(cuboid.corners)
            edges = EDGES
        else:
            # Dashed outlines in the oblique panel have no supported height.
            corners = transform(np.c_[obj.footprint, np.full(4, -1.7)])
            edges = [(i, (i + 1) % 4) for i in range(4)]
        for a, b in edges:
            ax.plot(corners[[a, b], 0], corners[[a, b], 1], color="#ffdf40", lw=.65,
                    linestyle="--" if view != "bev" and cuboid is None else "-", zorder=3)
        center = corners.mean(axis=0)
        ax.annotate(f"{obj.score:.2f}", center, xytext=(0, 3), textcoords="offset points",
                    fontsize=4.4, color="#ff6860", ha="center", zorder=4)
    ax.set_xlim(limits[:2])
    ax.set_ylim(limits[2:])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return visible


def draw_heatmap(ax, result, roi):
    """Use native output cells, fixed probability scale, X up and +Y left."""
    config, maps = result["config"], result["heatmaps"]["probabilities"]
    g = config["data"]["kitti"]["geometry"]
    extent = [-g["y_max"], -g["y_min"], g["x_min"], g["x_max"]]
    ax.set_facecolor("black")
    image = ax.imshow(maps.max(axis=0).T[:, ::-1], origin="lower", extent=extent,
                      cmap="gray", vmin=0., vmax=1., interpolation="nearest", aspect="equal")
    ax.set_xlim(-roi[3], -roi[2])
    ax.set_ylim(roi[0], roi[1])
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def render_comparison(points, results, output, frame, split, roi, view):
    plt = setup_matplotlib()
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    fig = plt.figure(figsize=(20.8, 17.6), dpi=200, facecolor="white")
    grid = fig.add_gridspec(3, 4, left=.018, right=.982, bottom=.083, top=.948,
                           wspace=.035, hspace=.23, width_ratios=[1, 1, 1, 1])
    image = None
    for index, result in enumerate(results):
        row, pair = divmod(index, 2)
        ax = fig.add_subplot(grid[row, pair * 2])
        heat = fig.add_subplot(grid[row, pair * 2 + 1])
        visible = draw_cloud(ax, points, result, roi, view)
        image = draw_heatmap(heat, result, roi)
        ax.set_title("LiDAR + detections", fontsize=9, pad=7)
        heat.set_title("Classification heatmap", fontsize=9, pad=7)
        a, b = ax.get_position(), heat.get_position()
        center = (a.x0 + b.x1) / 2
        fig.text(center, min(a.y0, b.y0) - .019,
                 f"({chr(97 + index)}) {result['item']['title']}", ha="center", va="top", fontsize=13)
        counts = Counter(result["detections"][i].name for i in visible)
        text = "  /  ".join(f"{name}: {counts[name]}" for name in result["class_order"])
        fig.text(center, min(a.y0, b.y0) - .037, text, ha="center", va="top", fontsize=8, color="#535353")
        result["visible_prediction_indices"] = visible
    fig.text(.5, .986, f"KITTI {frame} ({split})", ha="center", va="top", fontsize=17)
    fig.text(.5, .967, "Same original LiDAR scan for all six checkpoints  |  score >= 0.30  |  rotated BEV NMS IoU = 0.10",
             ha="center", va="top", fontsize=9)
    color_ax = fig.add_axes([.38, .040, .24, .008])
    fig.colorbar(image, cax=color_ax, orientation="horizontal", ticks=[0., .25, .5, .75, 1.])
    color_ax.tick_params(labelsize=8, length=2)
    fig.text(.5, .061, "max sigmoid(class logits)  |  fixed probability scale [0, 1]  |  no smoothing",
             ha="center", fontsize=9)
    note = ("XY boxes and scores: checkpoint predictions. Cuboid Z/height: measured LiDAR; unsupported boxes: dashed BEV footprints."
            if view == "perspective" else "Predicted BEV footprints. X points up; +Y points left in both panels.")
    fig.text(.5, .014, note, ha="center", fontsize=8, color="#454545")
    stem = output / f"comparison_{frame}_3x2_{view}"
    for suffix in ("png", "svg"):
        fig.savefig(stem.with_suffix(f".{suffix}"), dpi=200, facecolor="white")
    plt.close(fig)
    # Render each pair independently so neighboring captions cannot enter it.
    for index, result in enumerate(results):
        pair_fig = plt.figure(figsize=(10.4, 5.8), dpi=240, facecolor="white")
        pair_grid = pair_fig.add_gridspec(1, 2, left=.01, right=.99, bottom=.15,
                                         top=.94, wspace=.035)
        ax = pair_fig.add_subplot(pair_grid[0, 0])
        heat = pair_fig.add_subplot(pair_grid[0, 1])
        visible = draw_cloud(ax, points, result, roi, view)
        draw_heatmap(heat, result, roi)
        ax.set_title("LiDAR + detections", fontsize=9)
        heat.set_title("Classification heatmap", fontsize=9)
        pair_fig.text(.5, .075, f"({chr(97 + index)}) {result['item']['title']}",
                      ha="center", fontsize=13)
        counts = Counter(result["detections"][i].name for i in visible)
        pair_fig.text(.5, .03, "  /  ".join(f"{name}: {counts[name]}" for name in result["class_order"]),
                      ha="center", fontsize=9, color="#535353")
        pair_fig.savefig(output / result["item"]["id"] / f"pair_{view}.png", dpi=240)
        plt.close(pair_fig)
    return {suffix: {"path": str(stem.with_suffix(f'.{suffix}')),
                     "sha256": sha256(stem.with_suffix(f'.{suffix}'))} for suffix in ("png", "svg")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT / "Demo/configs/model_comparison.json")
    parser.add_argument("--frames", nargs="+", default=["000134", "005663"])
    parser.add_argument("--points-dir", type=Path, help="Directory of original KITTI .bin files")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "Demo/outputs/heatmap/model_comparison")
    parser.add_argument("--device", choices=["cpu", "auto", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--roi", nargs=6, type=float, default=[0., 45., -24., 24., -2.8, 4.])
    args = parser.parse_args(argv)
    if args.threads <= 0 or not np.isfinite(args.roi).all() or any(a >= b for a, b in zip(args.roi[::2], args.roi[1::2])):
        parser.error("threads must be positive and ROI must have finite increasing bounds")
    if any(len(frame) != 6 or not frame.isdigit() for frame in args.frames):
        parser.error("Frame IDs must be six decimal digits")
    manifest = json.loads(args.manifest.read_text())
    if len(manifest["models"]) != 6 or len({m["id"] for m in manifest["models"]}) != 6:
        parser.error("Manifest must contain six distinct model IDs in layout order")
    clouds = {frame: load_cloud(frame, args.points_dir) for frame in args.frames}
    started = datetime.now(timezone.utc).isoformat()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    results = {frame: [] for frame in args.frames}
    for item in manifest["models"]:
        checkpoint = Path(manifest["artifacts_root"]) / item["checkpoint"]
        runner = BEVOverviewPredictor(checkpoint, device=args.device, threads=args.threads,
                                      score=.3, nms=.1, max_detections=500)
        actual = validate_variant(item, runner)
        print(f"Loaded {item['id']} strictly; epoch {actual['epoch']}", flush=True)
        for frame, (points, source) in clouds.items():
            output = root / frame / item["id"]
            output.mkdir(parents=True, exist_ok=True)
            rows, inference, heatmaps = runner.predict(points, capture_heatmaps=True)
            if not np.isfinite(rows).all() or not np.isfinite(heatmaps["logits"]).all():
                raise ValueError("Nonfinite real inference output")
            geometry = runner.config["data"]["kitti"]["geometry"]
            expected_shape = [len(runner.class_order), *reversed([
                n // runner.config["data"]["out_size_factor"] for n in runner.schema.grid_shape[:2]])]
            if list(heatmaps["probabilities"].shape) != expected_shape:
                raise ValueError("Actual output heatmap grid disagrees with checkpoint geometry")
            for axis, low, high in [("x", args.roi[0], args.roi[1]), ("y", args.roi[2], args.roi[3])]:
                if low < geometry[f"{axis}_min"] or high > geometry[f"{axis}_max"]:
                    raise ValueError("Display ROI must stay inside every model's XY sensing domain")
            detections = bev_detections(rows, runner.config["data"]["kitti"]["objects"])
            lifted, reports = lift_predictions(points, detections, args.roi[4:])
            np.save(output / "bev_predictions.npy", rows)
            np.savez_compressed(output / "classification_heatmaps.npz", **heatmaps)
            np.savez_compressed(output / "lidar_lifted_cuboids.npz",
                                prediction_indices=np.asarray(list(lifted), dtype=np.int64),
                                corners=np.asarray([obj.corners for obj in lifted.values()]).reshape(-1, 8, 3))
            write_json(output / "config.inference.json", runner.config)
            write_json(output / "vertical_support.json", reports)
            metadata = {"id": item["id"], "title": item["title"], "variant": actual,
                        "selection": item["selection"], "note": item.get("note"),
                        "inference": inference, "detection_count": len(rows),
                        "class_counts": dict(Counter(obj.name for obj in detections)),
                        "lidar_supported_cuboids": len(lifted), "prediction_source": "fresh model forward",
                        "model_config": runner.config["model"], "loss_config": runner.config["loss"],
                        "augmentation_config": runner.config.get("augmentation"),
                        "heatmap_shape_cyx": list(heatmaps["probabilities"].shape),
                        "files": {name: sha256(output / name) for name in ["bev_predictions.npy",
                            "classification_heatmaps.npz", "lidar_lifted_cuboids.npz", "config.inference.json",
                            "vertical_support.json"]}}
            results[frame].append({"item": item, "rows": rows, "config": runner.config,
                                   "class_order": runner.class_order, "detections": detections,
                                   "lifted": lifted, "heatmaps": heatmaps, "metadata": metadata})
            print(f"  {frame}: {len(rows)} detections; {len(lifted)} LiDAR-supported cuboids; "
                  f"heatmaps {heatmaps['probabilities'].shape}", flush=True)
        if sha256(checkpoint) != runner.metadata["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed during inference")
        del runner
        gc.collect()
    for frame, (points, source) in clouds.items():
        output = root / frame
        points.astype("<f4", copy=False).tofile(output / f"{frame}_recovered_original.bin")
        if sha256(output / f"{frame}_recovered_original.bin") != source["points_xyzi_sha256"]:
            raise ValueError("Recovered binary differs from source")
        figures = {view: render_comparison(points, results[frame], output, frame,
                                           source["split"], args.roi, view)
                   for view in ["perspective", "bev"]}
        for result in results[frame]:
            result["metadata"]["visible_prediction_indices"] = result["visible_prediction_indices"]
            write_json(output / result["item"]["id"] / "metadata.json", result["metadata"])
        write_json(output / "metadata.json", {
            "frame": frame, "started_utc": started, "source": source,
            "layout": "3 rows x 2 model pairs; LiDAR left and heatmap right in each pair",
            "model_order": [item["id"] for item in manifest["models"]],
            "manifest": str(args.manifest.resolve()), "manifest_sha256": sha256(args.manifest),
            "script_sha256": sha256(Path(__file__)), "roi_xyz": args.roi,
            "oblique_camera": {"elevation": 54., "azimuth": 180.},
            "heatmap": {"source": "sigmoid of actual classification logits before peak selection/NMS",
                        "merge": "max across classes", "tensor_axes": ["class", "y", "x"],
                        "display": "X up; +Y left", "colormap": "gray", "scale": [0, 1],
                        "interpolation": "nearest", "smoothing": False,
                        "iou_reranking_applied_to_heatmap": False},
            "ground_truth_used": False, "augmentation_applied_at_inference": False,
            "predicted_vertical_geometry": False, "vertical_geometry": "measured LiDAR support only",
            "figures": figures, "models": [result["metadata"] for result in results[frame]],
        })
        print(f"Rendered {output}", flush=True)
    write_json(root / "manifest.used.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
