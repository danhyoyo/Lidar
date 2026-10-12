#!/usr/bin/env python3
"""Render BEV detections as cuboids with LiDAR-estimated Z and height.

Class, score, XY, length, width and yaw come from saved model inference. Vertical
geometry is a measured-point postprocess, not a learned 3D checkpoint output.
Ground-truth labels and class height priors are never used.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.visualization.export_kitti_overview import (
    CLASSES, camera_basis, canvas_limits, points_in_object, read_points, render_panel,
    roi_vertices, sha256,
)
from tools.visualization.predict_kitti_overview import bev_detections, points_in_bev


@dataclass
class LiftedObject:
    name: str
    label_index: int
    row: np.ndarray
    corners: np.ndarray

    @property
    def score(self):
        return float(self.row[1])


def fit_ground_plane(points, *, cell_size=1., min_cells=6):
    """Robust z=a*x+b*y+c from low-return representatives in distinct XY cells.

    Return coefficients and original input indices of accepted ground samples.
    Representatives are actual measured points, not interpolated ground points.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 3 or not np.isfinite(points[:, :3]).all():
        raise ValueError("Invalid ground points")
    if len(points) < min_cells:
        raise ValueError("Insufficient ground support")
    cells = np.floor(points[:, :2] / cell_size).astype(np.int64)
    unique, inverse = np.unique(cells, axis=0, return_inverse=True)
    seeds = []
    for group in range(len(unique)):
        indices = np.flatnonzero(inverse == group)
        quantile = np.quantile(points[indices, 2], .15)
        seeds.append(indices[np.argmin(np.abs(points[indices, 2] - quantile))])
    seeds = np.asarray(seeds)
    if len(seeds) < min_cells:
        raise ValueError("Insufficient distinct ground cells")
    representatives = points[seeds]
    origin = representatives[:, :2].mean(axis=0)
    design = np.c_[representatives[:, :2] - origin, np.ones(len(seeds))]
    mask = representatives[:, 2] <= np.quantile(representatives[:, 2], .7) + .25
    for _ in range(8):
        if mask.sum() < min_cells or np.linalg.matrix_rank(design[mask]) < 3:
            raise ValueError("Degenerate ground support")
        plane = np.linalg.lstsq(design[mask], representatives[mask, 2], rcond=None)[0]
        residual = representatives[:, 2] - design @ plane
        median = np.median(residual[mask])
        mad = np.median(np.abs(residual[mask] - median))
        updated = np.abs(residual - median) <= max(.06, 2.5 * 1.4826 * mad)
        if np.array_equal(updated, mask):
            break
        mask = updated
    if mask.sum() < min_cells or np.linalg.matrix_rank(design[mask]) < 3:
        raise ValueError("Degenerate ground inliers")
    plane = np.linalg.lstsq(design[mask], representatives[mask, 2], rcond=None)[0]
    if np.linalg.norm(plane[:2]) > .3:
        raise ValueError("Unreliable ground slope")
    plane[2] -= plane[:2] @ origin
    return plane, seeds[mask]


def local_ground(points, obj, occupied, *, z_bounds):
    """Fit surrounding ground, excluding every model-predicted XY footprint."""
    xy = points[:, :2] - obj.row[2:4]
    c, s = np.cos(obj.row[6]), np.sin(obj.row[6])
    local = xy @ np.array([[c, -s], [s, c]])
    z_mask = (points[:, 2] >= z_bounds[0]) & (points[:, 2] <= z_bounds[1])
    errors = []
    for padding in (2., 4., 7.):
        mask = (np.abs(local) <= obj.row[4:6] / 2 + padding).all(axis=1)
        indices = np.flatnonzero(mask & ~occupied & z_mask)
        try:
            plane, accepted = fit_ground_plane(points[indices])
        except ValueError as error:
            errors.append(str(error))
            continue
        return plane, {"ground_plane": plane.tolist(), "ground_ring_padding_m": padding,
                       "ground_support_indices": indices[accepted].tolist(),
                       "ground_support_cells": len(accepted)}
    raise ValueError("No reliable local ground: " + "; ".join(errors))


def lift_object(points, obj, ground_plane, *, min_points=5, min_clearance=.2,
                height_gap=.45, top_quantile=.98, z_bounds=(-2.8, 4.)):
    """Retain the BEV box and estimate a supported upright vertical interval."""
    points = np.asarray(points)
    plane = np.asarray(ground_plane)
    indices = np.flatnonzero(points_in_bev(points, obj)
                            & (points[:, 2] >= z_bounds[0]) & (points[:, 2] <= z_bounds[1]))
    ground = np.c_[points[indices, :2], np.ones(len(indices))] @ plane
    heights = points[indices, 2] - ground
    above = heights > min_clearance
    support = indices[above]
    report = {"prediction_index": obj.index, "class": obj.name, "score": obj.score,
              "points_in_xy_region": len(indices), "above_ground_points": len(support),
              "status": "insufficient_foreground_support"}
    if len(support) < min_points:
        return None, report
    order = np.argsort(heights[above], kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(heights[above][order]) > height_gap) + 1)
    group = max(groups, key=lambda item: (len(item), -float(heights[above][item].min())))
    report["discarded_vertical_outliers"] = len(support) - len(group)
    support = support[group]
    report["foreground_support_points"] = len(support)
    if len(support) < min_points:
        return None, report
    bottom = float(np.r_[obj.row[2:4], 1.] @ plane)
    top = float(np.quantile(points[support, 2], top_quantile))
    if not np.isfinite([bottom, top]).all() or top - bottom < .3:
        report["status"] = "insufficient_vertical_extent"
        return None, report
    source = obj.row
    row = np.array([source[0], source[1], source[2], source[3], bottom,
                    source[4], source[5], top-bottom, source[6]], dtype=source.dtype)
    base = np.c_[obj.footprint, np.full(4, float(row[4]))]
    corners = np.r_[base, base + [0, 0, float(row[7])]]
    lifted = LiftedObject(obj.name, obj.index, row, corners)
    report.update(status="lifted", z_bottom=float(row[4]), height=float(row[7]),
                  support_indices=np.sort(support).tolist(), vertical_source="local ground + measured foreground Z")
    return lifted, report


def preview(output, source, counts, zooms, unsupported, outside_roi):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 32)
        small = ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        font = small = ImageFont.load_default()
    image = Image.new("RGB", (2400, 1900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((65, 25), f"KITTI {source['frame_id']}  |  BEV detections + LiDAR-derived vertical extents",
              font=font, fill="#111827")
    draw.text((65, 80), "   ".join(f"{name}: {counts[name]}" for name in CLASSES) +
              f"   |   {unsupported} unsupported, {outside_roi} outside display ROI", font=small, fill="#4b5563")
    for filename, title, x in (("lidar_pointcloud", "3D LiDAR input", 65),
                              ("detection_3d", "3D cuboids with estimated Z / height", 1230)):
        draw.text((x, 145), title, font=font, fill="#111827")
        with Image.open(output / f"{filename}.png") as panel:
            image.paste(panel.convert("RGB").resize((1105, 737), Image.Resampling.LANCZOS), (x, 195))
    for i, name in enumerate(CLASSES):
        x = 65 + i * 785
        if name not in zooms:
            draw.text((x, 985), f"{name}: insufficient support", font=small, fill="#4b5563")
            continue
        obj, indices = zooms[name]
        draw.text((x, 985), f"{name}  |  score {obj.score:.3f}", font=font, fill="#111827")
        draw.text((x, 1035), f"{len(indices)} real points  |  estimated height {obj.row[7]:.2f} m",
                  font=small, fill="#4b5563")
        with Image.open(output / f"zoom_{name.lower()}.png") as panel:
            image.paste(panel.convert("RGB").resize((700, 700), Image.Resampling.LANCZOS), (x, 1090))
    draw.text((65, 1840), "BEV geometry / scores: checkpoint. Vertical extents: point-cloud postprocess. No GT boxes used.",
              font=small, fill="#4b5563")
    image.save(output / "overview_preview.png")
    image.crop((40, 960, 2370, 1815)).save(output / "object_zooms.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True,
                        help="Output folder of predict_kitti_overview.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-support-points", type=int, default=5)
    parser.add_argument("--min-object-points", type=int, default=20)
    parser.add_argument("--min-clearance", type=float, default=.2)
    parser.add_argument("--height-gap", type=float, default=.45)
    parser.add_argument("--top-quantile", type=float, default=.98)
    parser.add_argument("--class-colors", action="store_true")
    args = parser.parse_args(argv)
    if min(args.min_support_points, args.min_object_points) < 1:
        parser.error("Point counts must be positive")
    if (not np.isfinite([args.min_clearance, args.height_gap, args.top_quantile]).all()
            or args.min_clearance < 0 or args.height_gap <= 0 or not .5 < args.top_quantile <= 1):
        parser.error("Clearance/gap must be finite, clearance >= 0, gap > 0 and top quantile in (.5,1]")
    source_dir = args.prediction_dir.expanduser().resolve()
    metadata_path = source_dir / "metadata.json"
    source = json.loads(metadata_path.read_text())
    if (source.get("box_source") != "fresh checkpoint inference" or source["inference"]["box_mode"] != "bev"
            or source.get("ground_truth_boxes_used", True)):
        raise ValueError("Expected verified checkpoint BEV inference metadata without GT boxes")
    bin_path = Path(source["source_bin"]["path"])
    if sha256(bin_path) != source["source_bin"]["sha256"]:
        raise ValueError("Original LiDAR binary does not match inference provenance")
    points = read_points(bin_path)
    frame = source["frame_id"]
    rows_path = source_dir / f"{frame}_bev_predictions.npy"
    rows = np.load(rows_path, allow_pickle=False)
    original_rows = np.asarray([obj["bev_row"] for obj in source["predictions"]], dtype=rows.dtype).reshape(-1, 7)
    if not np.array_equal(rows, original_rows):
        raise ValueError("Saved prediction rows do not match inference metadata")
    config = json.loads((source_dir / "config.inference.json").read_text())
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if config_hash != source["inference"]["config_sha256"]:
        raise ValueError("Saved config does not match inference provenance")
    detections = bev_detections(rows, config["data"]["kitti"]["objects"])
    output = args.output_dir.expanduser().resolve()
    if output == source_dir:
        parser.error("Use a separate output directory to preserve original BEV inference")
    output.mkdir(parents=True, exist_ok=True)
    occupied = np.zeros(len(points), dtype=bool)
    for obj in detections:
        occupied |= points_in_bev(points, obj)
    roi = source["roi_xyz"]
    z_bounds = (roi[4], roi[5])
    objects, reports = [], []
    for obj in detections:
        try:
            plane, ground_report = local_ground(points, obj, occupied, z_bounds=z_bounds)
        except ValueError as error:
            reports.append({"prediction_index": obj.index, "class": obj.name,
                            "status": "insufficient_ground_support", "reason": str(error)})
            continue
        lifted, report = lift_object(points, obj, plane, min_points=args.min_support_points,
            min_clearance=args.min_clearance, height_gap=args.height_gap,
            top_quantile=args.top_quantile, z_bounds=z_bounds)
        report.update(ground_report)
        reports.append(report)
        if lifted is not None:
            objects.append(lifted)
        print(f"{obj.index:02d} {obj.name}: {report['status']}" +
              (f", Z={lifted.row[4]:.2f}, h={lifted.row[7]:.2f} m" if lifted else ""), flush=True)
    counts = {name: sum(obj.name == name for obj in objects) for name in CLASSES}
    lifted_rows = np.asarray([obj.row for obj in objects], dtype=rows.dtype).reshape(-1, 9)
    source_indices = np.asarray([obj.label_index for obj in objects], dtype=np.int64)
    np.savez_compressed(output / "lifted_detections_3d.npz", rows=lifted_rows,
                        source_prediction_indices=source_indices, source_bev_rows=rows)
    lower, upper = np.asarray(roi[::2]), np.asarray(roi[1::2])
    mask = ((points[:, :3] >= lower) & (points[:, :3] <= upper)).all(axis=1)
    scene = points[mask]
    visible_objects = [obj for obj in objects if ((obj.corners.max(axis=0) >= lower)
                       & (obj.corners.min(axis=0) <= upper)).all()]
    visible_counts = {name: sum(obj.name == name for obj in visible_objects) for name in CLASSES}
    view = source["scene_view"]
    basis = camera_basis(view["elevation"], view["azimuth"])
    size, dpi = [1800, 1200], 300
    limits = canvas_limits(roi_vertices(roi), basis, size[0]/size[1], margin=.01)
    render_panel(scene, [], basis=basis, limits=limits, size=size, dpi=dpi, point_size=.28,
                 output=output / "lidar_pointcloud", raw=True)
    render_panel(scene, visible_objects, basis=basis, limits=limits, size=size, dpi=dpi, point_size=.28,
                 output=output / "detection_3d", labels=True, class_colors=args.class_colors)
    zooms, zoom_metadata = {}, {}
    for name in CLASSES:
        candidates = []
        for obj in visible_objects:
            if obj.name != name:
                continue
            indices = np.flatnonzero(points_in_object(points, obj))
            if len(indices) >= args.min_object_points:
                candidates.append((len(indices), obj.score, -obj.label_index, obj, indices))
        if not candidates:
            zoom_metadata[name] = {"status": "insufficient measured point support"}
            continue
        _, _, _, obj, indices = max(candidates, key=lambda item: item[:3])
        zooms[name] = (obj, indices)
        zoom_basis = camera_basis(14, -150 + np.rad2deg(obj.row[8]))
        zoom_limits = canvas_limits(obj.corners, zoom_basis, 1., margin=.15)
        render_panel(points[indices], [obj], basis=zoom_basis, limits=zoom_limits,
                     size=[1000, 1000], dpi=dpi, point_size=1.8, output=output / f"zoom_{name.lower()}",
                     class_colors=args.class_colors)
        np.savez_compressed(output / f"zoom_{name.lower()}_points.npz", points=points[indices],
                            source_indices=indices, lifted_row=obj.row, box_corners=obj.corners)
        zoom_metadata[name] = {"prediction_index": obj.label_index, "score": obj.score,
                               "point_count": len(indices), "height_estimated_m": float(obj.row[7]),
                               "z_bottom_estimated_m": float(obj.row[4])}
    report = {"frame_id": frame, "box_source": "checkpoint BEV predictions + LiDAR vertical estimation",
        "upstream_inference_metadata": {"path": str(metadata_path), "sha256": sha256(metadata_path)},
        "upstream_prediction_rows": {"path": str(rows_path), "sha256": sha256(rows_path)},
        "source_bin": source["source_bin"], "inference": source["inference"],
        "split_membership": source.get("split_membership", {}), "original_bev_counts": source["counts"],
        "lifted_counts": counts, "unsupported_prediction_count": len(detections)-len(objects),
        "visible_cuboid_counts": visible_counts, "outside_display_roi": len(objects)-len(visible_objects),
        "ground_truth_boxes_used": False, "generated_points": 0, "learned_vertical_prediction": False,
        "vertical_geometry_source": "robust local ground plane and measured above-ground Z support",
        "estimated_rows_columns": ["class", "score", "x", "y", "z_bottom", "length", "width", "height", "yaw"],
        "original_point_count": len(points), "scene_point_count": len(scene), "roi_xyz": roi,
        "estimator_settings": {"cell_size_m": 1., "min_ground_cells": 6,
            "ground_ring_padding_m": [2., 4., 7.], "min_support_points": args.min_support_points,
            "min_clearance_m": args.min_clearance, "vertical_component_gap_m": args.height_gap,
            "top_quantile": args.top_quantile, "minimum_supported_height_m": .3},
        "vertical_estimates": reports, "zoom_objects": zoom_metadata,
        "limitation": "Estimated visible vertical extent; sparse/occluded returns can underestimate actual height.",
        "reproduce_command": [sys.executable, str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])]}
    (output / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    preview(output, source, visible_counts, zooms, len(detections)-len(objects), len(objects)-len(visible_objects))
    print(f"Saved {len(objects)}/{len(detections)} supported cuboids and class zooms to {output}")
    return report


if __name__ == "__main__":
    main()
