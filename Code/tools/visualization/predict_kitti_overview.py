#!/usr/bin/env python3
"""Run a real Hist14 or binary-slice BEV checkpoint on KITTI LiDAR.

BEV checkpoints do not predict Z or height. The 3D scene shows planar footprints
on an explicitly declared display plane, never manufactured 3D detection boxes.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import json
import hashlib
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.visualization.export_kitti_overview import (
    CLASSES, CLASS_COLORS, camera_basis, canvas_limits, dataset_paths, projected,
    read_points, render_panel, roi_vertices, setup_matplotlib, sha256,
)


@dataclass
class BEVDetection:
    name: str
    index: int
    row: np.ndarray

    @property
    def score(self):
        return float(self.row[1])

    @property
    def footprint(self):
        _, _, x, y, length, width, yaw = self.row
        local = np.array([[-length/2, -width/2], [length/2, -width/2],
                          [length/2, width/2], [-length/2, width/2]])
        c, s = np.cos(yaw), np.sin(yaw)
        return local @ np.array([[c, s], [-s, c]]) + [x, y]


def bev_detections(rows, class_ids):
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != 7 or not np.isfinite(rows).all():
        raise ValueError("Expected finite BEV [class,score,x,y,length,width,yaw] rows, shape [N,7]")
    names = {identifier: name for name, identifier in class_ids.items()}
    result = []
    for index, row in enumerate(rows):
        if (row[0] != int(row[0]) or int(row[0]) not in names or not 0 <= row[1] <= 1
                or (row[4:6] <= 0).any()):
            raise ValueError("Invalid BEV class, score or dimensions")
        result.append(BEVDetection(names[int(row[0])], index, row.copy()))
    return result


def points_in_bev(points, obj):
    """Crop the predicted XY region only; there is no predicted vertical extent."""
    yaw = obj.row[6]
    c, s = np.cos(yaw), np.sin(yaw)
    local = (np.asarray(points)[:, :2] - obj.row[2:4]) @ np.array([[c, -s], [s, c]])
    return (np.abs(local) <= obj.row[4:6] / 2 + 1e-7).all(axis=1)


def inference_config(checkpoint, config_path=None):
    """Prefer the authoritative saved config; external overrides remain explicit."""
    if config_path is not None:
        return json.loads(Path(config_path).read_text())
    if not isinstance(checkpoint.get("config"), dict):
        raise ValueError("Checkpoint has no embedded config; provide --config explicitly")
    return copy.deepcopy(checkpoint["config"])


def classification_heatmaps(prediction, *, groups, num_classes):
    """Gather raw Gaussian head logits in global class order, before peak/NMS filtering."""
    import torch
    parts = []
    if groups:
        if set(prediction.get("groups", {})) != {group.name for group in groups}:
            raise ValueError("Classification groups must match the checkpoint")
        parts = [(prediction["groups"][group.name]["cls"], group.global_ids) for group in groups]
    else:
        parts = [(prediction["cls"], tuple(range(num_classes)))]
    channels = {}
    spatial_shape = None
    for tensor, ids in parts:
        if (tensor.ndim != 4 or tensor.shape[0] != 1 or tensor.shape[1] != len(ids)
                or not torch.isfinite(tensor).all()):
            raise ValueError("Expected finite classification logits of shape [1,C,Y,X]")
        if spatial_shape is not None and tensor.shape[-2:] != spatial_shape:
            raise ValueError("Classification heads must share their spatial shape")
        spatial_shape = tensor.shape[-2:]
        for local_id, global_id in enumerate(ids):
            if global_id in channels:
                raise ValueError("Duplicate global class ID in classification heads")
            channels[global_id] = tensor[0, local_id].detach().float().cpu()
    if set(channels) != set(range(num_classes)):
        raise ValueError("Classification heads must cover all global class IDs")
    logits = torch.stack([channels[index] for index in range(num_classes)])
    return logits.numpy(), torch.sigmoid(logits).numpy()


def encode_model_points(points, schema):
    """Use the unchanged training rasterizers and return contiguous [C,Y,X]."""
    if schema.name == "hist14":
        from core.datasets.utils_1.bev_backend import encode_hist14_chw
        return encode_hist14_chw(points, schema)
    if schema.name == "binary_slices":
        from core.datasets.utils_1.preprocess import voxelize
        return np.ascontiguousarray(voxelize(points, dict(schema.geometry)).transpose(2, 0, 1))
    raise ValueError(f"Unsupported comparison encoder: {schema.name}")


class BEVOverviewPredictor:
    """Load/validate the checkpoint once and reuse the exact decoder across frames."""

    def __init__(self, checkpoint_path, config_path=None, *, device="auto", threads=4,
                 score=.3, nms=.1, max_detections=500):
        import torch
        sys.path.insert(0, str(ROOT / "tools/kitti_training_pipeline"))
        from common import (configure_detector_imports, build_model, validate_evaluation_checkpoint,
                            normalize_state_dict, bev_encoding_spec, detection_spec)
        configure_detector_imports(ROOT / "detector")
        from postprocess import filter_pred

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.config = config = inference_config(checkpoint, config_path)
        self.spec = spec = detection_spec(config)
        if spec.box_mode != "bev":
            raise ValueError("This exporter handles BEV checkpoints only; use the 3D decoder for a 3D checkpoint")
        self.schema = schema = bev_encoding_spec(config)
        if schema.name not in {"hist14", "binary_slices"}:
            raise ValueError("This inference entry point supports hist14 and binary_slices encoding")
        device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu")
        torch.set_num_threads(threads)
        validate_evaluation_checkpoint(checkpoint, config)
        model = build_model(config)
        model.load_state_dict(normalize_state_dict(checkpoint), strict=True)
        self.model = model.to(device).eval()
        self.device, self.score, self.nms, self.max_detections = device, score, nms, max_detections
        self.encode, self.decode = encode_model_points, filter_pred
        self.class_order = sorted(config["data"]["kitti"]["objects"],
                                  key=config["data"]["kitti"]["objects"].get)
        os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/lidar-kitti-overview-numba")
        self.metadata = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
            "config_source": str(config_path) if config_path else "embedded checkpoint config",
            "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            "checkpoint_epoch": checkpoint.get("epoch"), "box_mode": spec.box_mode,
            "strict_state_dict": True,
            "checkpoint_identity_validated": checkpoint.get("checkpoint_identity") is not None,
            "vertical_prediction": False, "ground_truth_used_for_inference": False,
            "torch_version": torch.__version__, "device": device, "precision": "fp32",
            "encoding": schema.name, "encoding_backend": schema.backend,
            "score_threshold": score, "nms_threshold": nms, "max_detections": max_detections,
            "use_iou": spec.use_iou,
            "nms_alpha": config["data"]["kitti"].get("nms_alpha", .5) if spec.use_iou else 0.,
            "checkpoint_identity": checkpoint.get("checkpoint_identity")}

    def predict(self, points, *, capture_heatmaps=False):
        import torch
        started = time.perf_counter()
        tensor = torch.from_numpy(self.encode(points, self.schema)).unsqueeze(0).to(self.device)
        config, spec, heatmaps = self.config, self.spec, None
        with torch.inference_mode():
            prediction = self.model(tensor)
            if capture_heatmaps:
                if spec.cls_encoding != "gaussian":
                    raise ValueError("Gaussian heatmap export requires cls_encoding=gaussian")
                logits, probabilities = classification_heatmaps(prediction,
                    groups=spec.groups if spec.head_mode == "grouped" else (),
                    num_classes=len(self.class_order))
                heatmaps = {"logits": logits, "probabilities": probabilities,
                            "class_order": np.asarray(self.class_order)}
            rows = self.decode(prediction, config["data"]["kitti"], config["data"]["out_size_factor"],
                self.score, self.nms, task_groups=spec.groups if spec.head_mode == "grouped" else None,
                cls_encoding=spec.cls_encoding, use_iou=spec.use_iou, max_detections=self.max_detections,
                box_mode=spec.box_mode)
        metadata = {**self.metadata, "input_shape": list(tensor.shape),
                    "elapsed_encoding_forward_decode_seconds": time.perf_counter() - started}
        return rows, metadata, heatmaps


def infer_frame(checkpoint_path, config_path, points, *, device, threads, score, nms,
                max_detections, heatmap_output=None):
    runner = BEVOverviewPredictor(checkpoint_path, config_path, device=device, threads=threads,
                                 score=score, nms=nms, max_detections=max_detections)
    rows, metadata, heatmaps = runner.predict(points, capture_heatmaps=heatmap_output is not None)
    if heatmap_output is not None:
        np.savez_compressed(heatmap_output, **heatmaps)
        metadata["classification_heatmaps"] = {"path": str(Path(heatmap_output).resolve()),
            "sha256": sha256(heatmap_output), "shape_cyx": list(heatmaps["probabilities"].shape),
            "activation": "sigmoid", "threshold_applied": False, "nms_applied": False,
            "iou_reranking_applied": False, "class_order": runner.class_order}
    return rows, runner.config, metadata


def save_figure(fig, path, dpi):
    for suffix in ("png", "svg"):
        fig.savefig(path.with_suffix(f".{suffix}"), dpi=dpi, facecolor="black", pad_inches=0)
    setup_matplotlib().close(fig)


def draw_footprints(points, objects, *, output, basis, limits, size, dpi, display_z,
                    point_size=.28, labels=True, class_colors=False):
    plt = setup_matplotlib()
    fig = plt.figure(figsize=(size[0]/dpi, size[1]/dpi), dpi=dpi, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1], facecolor="black")
    uvz = projected(points, basis)
    order = np.argsort(uvz[:, 2], kind="stable")
    ax.scatter(uvz[order, 0], uvz[order, 1], s=point_size, c="#52d9e8",
               linewidths=0, alpha=.9, rasterized=True)
    for obj in objects:
        corners = np.c_[obj.footprint, np.full(4, display_z)]
        uv = projected(corners, basis)
        closed = np.r_[uv[:, :2], uv[:1, :2]]
        color = CLASS_COLORS[obj.name] if class_colors else "#ff554b"
        ax.plot(closed[:, 0], closed[:, 1], c=color, lw=.95, zorder=3)
        if labels:
            ax.annotate(f"{obj.score:.2f}", uv[:, :2].mean(axis=0), xytext=(0, 4),
                        textcoords="offset points", color="white", fontsize=5.5, ha="center",
                        bbox=dict(facecolor="black", edgecolor="none", alpha=.6, pad=.6), zorder=4)
    ax.set_xlim(limits[:2])
    ax.set_ylim(limits[2:])
    ax.set_aspect("equal")
    ax.set_axis_off()
    save_figure(fig, output, dpi)


def preview(output, frame, counts, epoch, score, zooms):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 32)
        small = ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        font = small = ImageFont.load_default()
    image = Image.new("RGB", (2400, 1900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((65, 25), f"KITTI {frame}  |  Model predictions  |  epoch {epoch}  |  score >= {score:g}",
              font=font, fill="#111827")
    draw.text((65, 80), "   ".join(f"{name}: {counts[name]}" for name in CLASSES) +
              "   |   BEV model: Z and height are not predicted", font=small, fill="#4b5563")
    for filename, title, x in (("lidar_pointcloud", "3D LiDAR input", 65),
                              ("prediction_footprints_3d", "Predicted XY footprints on LiDAR", 1230)):
        draw.text((x, 145), title, font=font, fill="#111827")
        with Image.open(output / f"{filename}.png") as panel:
            image.paste(panel.convert("RGB").resize((1105, 737), Image.Resampling.LANCZOS), (x, 195))
    for i, name in enumerate(CLASSES):
        x = 65 + i * 785
        if name not in zooms:
            draw.text((x, 990), f"{name}: no supported detection", font=small, fill="#4b5563")
            continue
        obj, indices = zooms[name]
        draw.text((x, 985), f"{name}  |  score {obj.score:.3f}", font=font, fill="#111827")
        draw.text((x, 1035), f"{len(indices)} real points in predicted XY region", font=small, fill="#4b5563")
        with Image.open(output / f"zoom_{name.lower()}.png") as panel:
            image.paste(panel.convert("RGB").resize((700, 700), Image.Resampling.LANCZOS), (x, 1090))
    draw.text((65, 1840), "Red outlines are planar BEV predictions. Vertical extent is not inferred or taken from GT.",
              font=small, fill="#4b5563")
    image.save(output / "overview_preview.png")
    image.crop((40, 960, 2370, 1815)).save(output / "object_zooms.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint file or selected directory")
    parser.add_argument("--config", type=Path, help="Explicit override; default uses embedded checkpoint config")
    parser.add_argument("--kitti-root", type=Path, default=ROOT / "Kitti_Raw")
    parser.add_argument("--frame", default="005663")
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "Result/artifacts/kitti_overview_model")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--score-threshold", type=float, default=.3)
    parser.add_argument("--nms-threshold", type=float, default=.1)
    parser.add_argument("--max-detections", type=int, default=500)
    parser.add_argument("--min-object-points", type=int, default=20)
    parser.add_argument("--roi", nargs=6, type=float, default=[-2, 45, -24, 24, -2.8, 4])
    parser.add_argument("--display-z", type=float, default=-1.7, help="Footprint display plane only; not predicted Z")
    parser.add_argument("--elevation", type=float, default=28.)
    parser.add_argument("--azimuth", type=float, default=-145.)
    parser.add_argument("--class-colors", action="store_true")
    args = parser.parse_args(argv)
    if not args.frame.isdigit() or len(args.frame) > 6:
        parser.error("Frame must be a KITTI numeric ID up to six digits")
    if min(args.threads, args.min_object_points, args.max_detections) < 1:
        parser.error("Threads and count parameters must be positive")
    if not 0 <= args.score_threshold <= 1 or not 0 <= args.nms_threshold <= 1:
        parser.error("Score and NMS thresholds must be finite values in [0,1]")
    if (not np.isfinite([*args.roi, args.display_z, args.elevation, args.azimuth]).all()
            or any(a >= b for a, b in zip(args.roi[::2], args.roi[1::2]))):
        parser.error("ROI must have ordered bounds and view coordinates must be finite")
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if checkpoint_path.is_dir():
        checkpoint_path /= "best.pt"
    config_path = args.config.expanduser().resolve() if args.config else None
    paths = dataset_paths(args.kitti_root)
    frame = args.frame.zfill(6)
    bin_path = paths.velodyne / f"{frame}.bin"
    points = read_points(bin_path)
    print(f"Inferring KITTI {frame} from {checkpoint_path}", flush=True)
    rows, config, inference = infer_frame(checkpoint_path, config_path, points, device=args.device,
        threads=args.threads, score=args.score_threshold, nms=args.nms_threshold, max_detections=args.max_detections)
    objects = bev_detections(rows, config["data"]["kitti"]["objects"])
    counts = {name: sum(obj.name == name for obj in objects) for name in CLASSES}
    print(f"Epoch {inference['checkpoint_epoch']}, {inference['device']}: {counts}", flush=True)
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.inference.json").write_text(json.dumps(config, indent=2) + "\n")
    np.save(output / f"{frame}_bev_predictions.npy", rows)
    np.savez_compressed(output / "bev_predictions.npz", **{frame: rows})
    with (output / "predictions.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["class", "score", "x", "y", "length", "width", "yaw"])
        writer.writerows([obj.name, *map(float, obj.row[1:])] for obj in objects)
    lower, upper = np.asarray(args.roi[::2]), np.asarray(args.roi[1::2])
    scene_mask = ((points[:, :3] >= lower) & (points[:, :3] <= upper)).all(axis=1)
    scene = points[scene_mask]
    if not len(scene):
        raise ValueError("No original points in the display ROI")
    visible = [obj for obj in objects if ((obj.footprint.max(axis=0) >= lower[:2])
               & (obj.footprint.min(axis=0) <= upper[:2])).all()]
    size, dpi = [1800, 1200], 300
    basis = camera_basis(args.elevation, args.azimuth)
    limits = canvas_limits(roi_vertices(args.roi), basis, size[0]/size[1], margin=.01)
    render_panel(scene, [], basis=basis, limits=limits, output=output / "lidar_pointcloud",
                 size=size, dpi=dpi, point_size=.28, raw=True)
    draw_footprints(scene, visible, output=output / "prediction_footprints_3d", basis=basis,
        limits=limits, size=size, dpi=dpi, display_z=args.display_z, class_colors=args.class_colors)
    bev_basis = camera_basis(90, -90)
    bev_limits = canvas_limits(roi_vertices(args.roi), bev_basis, 1., margin=.01)
    draw_footprints(scene, visible, output=output / "detection_bev", basis=bev_basis,
        limits=bev_limits, size=[1800, 1800], dpi=dpi, display_z=args.display_z,
        class_colors=args.class_colors)
    # Zoom selection and point crops use predictions + the measured binary only.
    zooms = {}
    zoom_metadata = {}
    z_mask = (points[:, 2] >= lower[2]) & (points[:, 2] <= upper[2])
    for name in CLASSES:
        candidates = []
        for obj in visible:
            if obj.name != name:
                continue
            indices = np.flatnonzero(points_in_bev(points, obj) & z_mask)
            if len(indices) >= args.min_object_points:
                candidates.append((len(indices), obj.score, -obj.index, obj, indices))
        if not candidates:
            zoom_metadata[name] = {"status": "no prediction with enough measured points"}
            continue
        _, _, _, obj, indices = max(candidates, key=lambda item: item[:3])
        zooms[name] = (obj, indices)
        zoom_basis = camera_basis(14, -150 + np.rad2deg(obj.row[6]))
        vertices = np.r_[points[indices, :3], np.c_[obj.footprint, np.full(4, args.display_z)]]
        zoom_limits = canvas_limits(vertices, zoom_basis, 1., margin=.15)
        draw_footprints(points[indices], [obj], output=output / f"zoom_{name.lower()}", basis=zoom_basis,
            limits=zoom_limits, size=[1000, 1000], dpi=dpi, display_z=args.display_z,
            point_size=1.8, labels=False, class_colors=args.class_colors)
        np.savez_compressed(output / f"zoom_{name.lower()}_points.npz", points=points[indices],
                            source_indices=indices, predicted_bev_row=obj.row)
        zoom_metadata[name] = {"prediction_index": obj.index, "score": obj.score, "point_count": len(indices),
            "crop": "predicted XY footprint intersected with display Z ROI; includes ground/context returns",
            "predicted_bev_row": obj.row.tolist()}
        print(f"  Zoom {name}: score {obj.score:.3f}, {len(indices)} original points in XY region", flush=True)
    metadata = {"frame_id": frame, "box_source": "fresh checkpoint inference", "counts": counts,
        "original_point_count": len(points), "scene_point_count": int(scene_mask.sum()),
        "source_bin": {"path": str(bin_path.resolve()), "sha256": sha256(bin_path)},
        "prediction_columns": ["class", "score", "x", "y", "length", "width", "yaw"],
        "inference": inference, "generated_points": 0, "ground_truth_boxes_used": False,
        "roi_xyz": args.roi, "scene_view": {"elevation": args.elevation, "azimuth": args.azimuth},
        "display_plane_z": args.display_z, "display_plane_is_model_prediction": False,
        "vertical_extent": "not predicted; no cuboids or inferred heights are drawn", "zoom_objects": zoom_metadata,
        "predictions": [{"class": obj.name, "bev_row": obj.row.tolist()} for obj in objects],
        "reproduce_command": [sys.executable, str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])]}
    metadata["split_membership"] = {}
    for split in ("train", "val"):
        manifest = ROOT / config.get(split, {}).get("data", "__missing_manifest__")
        if manifest.is_file():
            ids = {line.split(";", 1)[0].strip() for line in manifest.read_text().splitlines() if line.strip()}
            metadata["split_membership"][split] = {"contains_frame": frame in ids,
                "manifest": str(manifest.resolve()), "manifest_sha256": sha256(manifest)}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    preview(output, frame, counts, inference["checkpoint_epoch"], args.score_threshold, zooms)
    print(f"Saved real BEV prediction panels and point crops to {output}")
    return metadata


if __name__ == "__main__":
    main()
