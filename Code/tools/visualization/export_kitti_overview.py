#!/usr/bin/env python3
"""Export real KITTI LiDAR scenes, calibrated boxes and per-class object zooms.

Default boxes are ground truth, never model predictions. No points are synthesized.
The renderer projects actual 3D coordinates orthographically onto a headless canvas.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.kitti_training_pipeline.prepare_kitti import parse_calibration
from tools.kitti_training_pipeline.kitti_box_conversion import camera_box_corners

CLASSES = ("Car", "Cyclist", "Pedestrian")
CLASS_COLORS = {"Car": "#ff6257", "Cyclist": "#ffca55", "Pedestrian": "#88eb93"}
EDGES = [(i, (i + 1) % 4) for i in range(4)]
EDGES += [(i + 4, (i + 1) % 4 + 4) for i in range(4)]
EDGES += [(i, i + 4) for i in range(4)]


@dataclass
class Object3D:
    name: str
    label_index: int
    corners: np.ndarray
    truncated: float
    occluded: int
    score: float | None = None

    @property
    def center(self):
        return self.corners.mean(axis=0)


@dataclass
class DatasetPaths:
    velodyne: Path
    labels: Path
    calib: Path


def dataset_paths(root):
    """Support official training/, flattened training/, and separate ZIP extracts."""
    root = Path(root).expanduser().resolve()
    candidates = [
        DatasetPaths(root / "training/velodyne", root / "training/label_2", root / "training/calib"),
        DatasetPaths(root / "velodyne", root / "label_2", root / "calib"),
        DatasetPaths(root / "velodyne/training/velodyne", root / "label_2/training/label_2",
                     root / "calib/training/calib"),
    ]
    for candidate in candidates:
        if (all(path.is_dir() for path in (candidate.velodyne, candidate.labels, candidate.calib))
                and next(candidate.velodyne.glob("*.bin"), None) is not None):
            return candidate
    raise FileNotFoundError(f"Cannot find KITTI training velodyne, label_2 and calib under {root}")


def read_points(path):
    path = Path(path)
    if path.stat().st_size == 0 or path.stat().st_size % 16:
        raise ValueError(f"KITTI binary must contain nonempty 16-byte XYZI records: {path}")
    points = np.fromfile(path, dtype="<f4").reshape(-1, 4)
    if not np.isfinite(points).all():
        raise ValueError(f"Non-finite point coordinates/intensity in {path}")
    return points


def read_objects(label_path, calibration_path, *, predictions=False):
    """Preserve all eight corners, including calibration pitch/roll, in LiDAR XYZ."""
    inverse = np.linalg.inv(parse_calibration(Path(calibration_path)))
    objects = []
    for index, line in enumerate(Path(label_path).read_text().splitlines()):
        fields = line.split()
        if not fields or fields[0] not in CLASSES:
            continue
        if len(fields) != (16 if predictions else 15):
            suffix = " and a score" if predictions else ""
            raise ValueError(f"Expected official 15-field KITTI label{suffix}: {label_path}:{index + 1}")
        values = np.asarray([float(value) for value in fields[1:]])
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite KITTI label: {label_path}:{index + 1}")
        height, width, length = map(float, fields[8:11])
        camera_box = np.array([*map(float, fields[11:14]), length, width, height, float(fields[14])])
        camera_corners = camera_box_corners(camera_box)
        corners = (np.c_[camera_corners, np.ones(8)] @ inverse.T)[:, :3]
        score = float(fields[15]) if predictions else None
        if score is not None and not 0 <= score <= 1:
            raise ValueError(f"Prediction score must be in [0,1]: {label_path}:{index + 1}")
        objects.append(Object3D(fields[0], index, corners, float(fields[1]), int(fields[2]), score))
    return objects


def points_in_object(points, obj, *, padding=0.0):
    """Exact oriented volume test, with optional physical padding in metres."""
    if not np.isfinite(padding) or padding < 0:
        raise ValueError("padding must be finite and nonnegative")
    corners = obj.corners
    basis = np.stack([corners[1] - corners[0], corners[3] - corners[0],
                      corners[4] - corners[0]], axis=1)
    coordinates = (np.asarray(points)[:, :3] - corners[0]) @ np.linalg.inv(basis).T
    margin = padding / np.linalg.norm(basis, axis=0)
    return ((coordinates >= -margin - 1e-8) & (coordinates <= 1 + margin + 1e-8)).all(axis=1)


def is_usable(obj, max_distance):
    return (obj.center[0] > 0 and np.linalg.norm(obj.center[:2]) <= max_distance
            and (obj.score is not None or (0 <= obj.occluded <= 1 and 0 <= obj.truncated <= .3)))


def candidate_key(row):
    usable = row["usable_counts"]
    return (sum(usable[name] > 0 for name in CLASSES), min(usable.values()),
            sum(usable.values()), sum(row["counts"].values()))


def select_candidate(rows, *, min_points):
    supported = [row for row in rows if all(row["usable_counts"][name] > 0
                 and row["best_point_counts"][name] >= min_points for name in CLASSES)]
    if not supported:
        raise ValueError("No frame supports all three classes with the requested point count; "
                         "try --scan-top-k with a larger value or lower --min-object-points")
    return max(supported, key=lambda row: (candidate_key(row),
               min(row["best_point_counts"].values()), -int(row["frame_id"])))


def scan_candidates(paths, *, top_k, max_distance, min_points):
    """Rank all labels cheaply, then inspect point support in the top candidates."""
    rows = []
    for path in sorted(paths.labels.glob("*.txt")):
        if not ((paths.velodyne / f"{path.stem}.bin").is_file()
                and (paths.calib / path.name).is_file()):
            continue
        counts = Counter()
        usable = Counter()
        for line in path.read_text().splitlines():
            fields = line.split()
            if not fields or fields[0] not in CLASSES:
                continue
            if len(fields) != 15:
                raise ValueError(f"Expected official KITTI labels: {path}")
            counts[fields[0]] += 1
            x, z = float(fields[11]), float(fields[13])
            if (0 < z and np.hypot(x, z) <= max_distance and 0 <= int(fields[2]) <= 1
                    and 0 <= float(fields[1]) <= .3):
                usable[fields[0]] += 1
        if all(counts[name] > 0 for name in CLASSES):
            rows.append({"frame_id": path.stem, "counts": {name: counts[name] for name in CLASSES},
                         "usable_counts": {name: usable[name] for name in CLASSES}})
    rows.sort(key=candidate_key, reverse=True)
    inspected = rows[:top_k]
    for row in inspected:
        frame = row["frame_id"]
        points = read_points(paths.velodyne / f"{frame}.bin")
        objects = read_objects(paths.labels / f"{frame}.txt", paths.calib / f"{frame}.txt")
        best = dict.fromkeys(CLASSES, 0)
        for obj in objects:
            if is_usable(obj, max_distance):
                best[obj.name] = max(best[obj.name], int(points_in_object(points, obj).sum()))
        row["best_point_counts"] = best
    return select_candidate(inspected, min_points=min_points), inspected, len(rows)


def choose_zooms(points, objects, *, max_distance, min_points):
    selected = {}
    for name in CLASSES:
        candidates = []
        for obj in objects:
            if obj.name != name or not is_usable(obj, max_distance):
                continue
            indices = np.flatnonzero(points_in_object(points, obj))
            if len(indices) >= min_points:
                candidates.append((len(indices), 0. if obj.score is None else obj.score,
                                   -obj.occluded, -obj.label_index, obj, indices))
        if not candidates:
            raise ValueError(f"No usable {name} zoom with >= {min_points} points in this frame")
        best = max(candidates, key=lambda item: item[:4])
        selected[name] = (best[4], best[5])
    return selected


def camera_basis(elevation, azimuth):
    elevation, azimuth = np.deg2rad([elevation, azimuth])
    direction = np.array([np.cos(elevation) * np.cos(azimuth),
                          np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.])
    up = np.cross(direction, right)
    return np.stack([right, up, direction], axis=1)


def projected(points, basis):
    return np.asarray(points)[:, :3] @ basis


def canvas_limits(vertices, basis, aspect, *, margin=.06):
    uv = projected(vertices, basis)[:, :2]
    center = (uv.min(axis=0) + uv.max(axis=0)) / 2
    span = np.maximum(np.ptp(uv, axis=0) * (1 + 2 * margin), .1)
    span[0] = max(span[0], span[1] * aspect)
    span[1] = max(span[1], span[0] / aspect)
    return (*[center[0] - span[0] / 2, center[0] + span[0] / 2],
            *[center[1] - span[1] / 2, center[1] + span[1] / 2])


def roi_vertices(roi):
    x0, x1, y0, y1, z0, z1 = roi
    return np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])


def setup_matplotlib():
    # Avoid writes to a read-only home directory in headless/sandbox runs.
    if "MPLCONFIGDIR" not in os.environ:
        cache = Path(tempfile.gettempdir()) / f"kitti-overview-mpl-{os.getuid()}"
        cache.mkdir(exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(cache)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    return plt


def render_panel(points, objects, *, basis, limits, output, size, dpi, point_size,
                 raw=False, selected_indices=(), class_colors=False, labels=False):
    plt = setup_matplotlib()
    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi, facecolor="black")
    ax = fig.add_axes([0, 0, 1, 1], facecolor="black")
    uvz = projected(points, basis)
    # Back-to-front drawing preserves a consistent 3D view without altering samples.
    order = np.argsort(uvz[:, 2], kind="stable")
    colors = plt.get_cmap("plasma")(np.clip((points[:, 2] + 2.5) / 5., 0, 1)) if raw else "#52d9e8"
    ax.scatter(uvz[order, 0], uvz[order, 1], s=point_size,
               c=colors[order] if raw else colors, linewidths=0, alpha=.9,
               rasterized=True, zorder=1)
    selected_indices = set(selected_indices)
    for obj in objects:
        color = CLASS_COLORS[obj.name] if class_colors else "#ff554b"
        corners = projected(obj.corners, basis)
        segments = [corners[[a, b], :2] for a, b in EDGES]
        from matplotlib.collections import LineCollection
        ax.add_collection(LineCollection(segments, colors=color, linewidths=.95,
                                        alpha=.96, zorder=3))
        if labels:
            anchor = corners[4:, :2].mean(axis=0)
            text = f"{obj.score:.2f}" if obj.score is not None else f"{obj.label_index:02d}"
            if obj.label_index in selected_indices:
                text = f"{obj.name} {text}"
            ax.annotate(text, anchor, xytext=(0, 4), textcoords="offset points",
                        color="#f3f7fa", fontsize=5.5, ha="center", va="bottom",
                        bbox=dict(facecolor="black", edgecolor="none", alpha=.65, pad=.7), zorder=4)
    ax.set_xlim(limits[:2])
    ax.set_ylim(limits[2:])
    ax.set_aspect("equal")
    ax.set_axis_off()
    for suffix in ("png", "svg"):
        fig.savefig(output.with_suffix(f".{suffix}"), dpi=dpi, facecolor="black", pad_inches=0)
    plt.close(fig)


def composite_preview(output_dir, frame, counts, box_source, zooms):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 32)
        small = ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        font = small = ImageFont.load_default()
    image = Image.new("RGB", (2400, 1900), "white")
    draw = ImageDraw.Draw(image)
    draw.text((65, 30), f"KITTI {frame}  |  " + "   ".join(f"{name}: {counts[name]}" for name in CLASSES),
              font=font, fill="#111827")
    draw.text((65, 85), f"Real Velodyne returns  |  {box_source}", font=small, fill="#4b5563")
    for name, title, x in (("lidar_pointcloud", "3D LiDAR point cloud", 65),
                            ("detection_3d", "3D boxes", 1230)):
        draw.text((x, 145), title, font=font, fill="#111827")
        with Image.open(output_dir / f"{name}.png") as panel:
            panel = panel.convert("RGB").resize((1105, 737), Image.Resampling.LANCZOS)
            image.paste(panel, (x, 195))
    for i, name in enumerate(CLASSES):
        x = 65 + i * 785
        obj, indices = zooms[name]
        draw.text((x, 985), f"{name}  |  object {obj.label_index:02d}", font=font, fill="#111827")
        draw.text((x, 1035), f"{len(indices)} measured points inside box", font=small, fill="#4b5563")
        with Image.open(output_dir / f"zoom_{name.lower()}.png") as panel:
            panel = panel.convert("RGB").resize((700, 700), Image.Resampling.LANCZOS)
            image.paste(panel, (x, 1090))
    draw.text((65, 1840), "No generated points, interpolation, object insertion or fabricated confidence scores.",
              font=small, fill="#4b5563")
    image.save(output_dir / "overview_preview.png")
    # A separate three-object strip for direct insertion into an overview figure.
    strip = image.crop((40, 960, 2370, 1815))
    strip.save(output_dir / "object_zooms.png")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--kitti-root", type=Path, default=ROOT / "Kitti_Raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "Result/artifacts/kitti_overview")
    parser.add_argument("--frame", default="auto", help="Six-digit frame ID or auto")
    parser.add_argument("--predictions-dir", type=Path, help="Real KITTI 16-field prediction TXT files")
    parser.add_argument("--score-threshold", type=float, default=.3)
    parser.add_argument("--scan-top-k", type=int, default=30)
    parser.add_argument("--max-object-distance", type=float, default=45.)
    parser.add_argument("--min-object-points", type=int, default=20)
    parser.add_argument("--roi", nargs=6, type=float, default=[-2, 45, -24, 24, -2.8, 4],
                        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"))
    parser.add_argument("--elevation", type=float, default=28.)
    parser.add_argument("--azimuth", type=float, default=-145.)
    parser.add_argument("--zoom-elevation", type=float, default=14.)
    parser.add_argument("--zoom-azimuth", type=float, default=-150.,
                        help="Angle relative to object's length direction in LiDAR")
    parser.add_argument("--zoom-padding", type=float, default=.15,
                        help="Context outside the exact box (m); zero isolates interior returns")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--scene-size", nargs=2, type=int, default=[1800, 1200], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--zoom-size", type=int, default=1000)
    parser.add_argument("--point-size", type=float, default=.28, help="Scene marker area in points squared")
    parser.add_argument("--zoom-point-size", type=float, default=1.8)
    parser.add_argument("--class-colors", action="store_true", help="Use a different box color per class")
    parser.add_argument("--labels", action="store_true", help="Show object indices or genuine prediction scores")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    positive = [args.scan_top_k, args.max_object_distance, args.min_object_points, args.dpi,
                *args.scene_size, args.zoom_size, args.point_size, args.zoom_point_size]
    if not np.isfinite(positive).all() or min(positive) <= 0:
        parser.error("Counts, distances, image dimensions and marker sizes must be positive")
    if not np.isfinite(args.roi).all() or any(a >= b for a, b in zip(args.roi[::2], args.roi[1::2])):
        parser.error("ROI must contain finite ordered min/max pairs")
    if not np.isfinite([args.elevation, args.azimuth, args.zoom_elevation, args.zoom_azimuth,
                        args.zoom_padding, args.score_threshold]).all() or args.zoom_padding < 0:
        parser.error("View/padding parameters must be finite and zoom padding nonnegative")
    if not 0 <= args.score_threshold <= 1:
        parser.error("Score threshold must be in [0,1]")
    if args.frame != "auto" and (not args.frame.isdigit() or len(args.frame) > 6):
        parser.error("Frame must be a numeric KITTI ID (up to six digits) or auto")
    if args.predictions_dir and args.frame == "auto":
        parser.error("Use --frame with --predictions-dir to select a frame with actual predictions")

    paths = dataset_paths(args.kitti_root)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.frame == "auto":
        row, candidates, total = scan_candidates(paths, top_k=args.scan_top_k,
            max_distance=args.max_object_distance, min_points=args.min_object_points)
        frame = row["frame_id"]
        report = {"frames_with_three_classes": total, "selected_frame": frame, "candidates": candidates,
                  "ranking": "class diversity, minimum usable count per class, total usable count, total count; "
                             "require minimum point support in all three classes"}
        (output_dir / "frame_candidates.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Found {total} frames with all three classes; inspected {len(candidates)} candidates.", flush=True)
        for candidate in candidates[:5]:
            print(f"  {candidate['frame_id']}: {candidate['counts']}; best interior points "
                  f"{candidate['best_point_counts']}", flush=True)
    else:
        frame = args.frame.zfill(6)
    bin_path = paths.velodyne / f"{frame}.bin"
    cal_path = paths.calib / f"{frame}.txt"
    label_path = paths.labels / f"{frame}.txt"
    points = read_points(bin_path)
    gt_objects = read_objects(label_path, cal_path)
    prediction_path = args.predictions_dir / f"{frame}.txt" if args.predictions_dir else None
    objects = (read_objects(prediction_path, cal_path, predictions=True)
               if prediction_path else gt_objects)
    if prediction_path:
        objects = [obj for obj in objects if obj.score >= args.score_threshold]
    box_source = "KITTI predictions" if prediction_path else "KITTI ground truth"
    zooms = choose_zooms(points, objects, max_distance=args.max_object_distance, min_points=args.min_object_points)
    lower, upper = np.asarray(args.roi[::2]), np.asarray(args.roi[1::2])
    visible_indices = np.flatnonzero(((points[:, :3] >= lower) & (points[:, :3] <= upper)).all(axis=1))
    if not len(visible_indices):
        raise ValueError("No measured points fall inside the requested ROI")
    visible_objects = [obj for obj in objects if ((obj.corners.max(axis=0) >= lower)
                       & (obj.corners.min(axis=0) <= upper)).all()]
    basis = camera_basis(args.elevation, args.azimuth)
    limits = canvas_limits(roi_vertices(args.roi), basis, args.scene_size[0] / args.scene_size[1], margin=.01)
    common = dict(basis=basis, limits=limits, size=args.scene_size, dpi=args.dpi, point_size=args.point_size)
    print(f"Rendering frame {frame}: {len(points)} original points, {len(visible_indices)} in ROI; {box_source}.", flush=True)
    render_panel(points[visible_indices], [], output=output_dir / "lidar_pointcloud", raw=True, **common)
    render_panel(points[visible_indices], visible_objects, output=output_dir / "detection_3d",
                 selected_indices=[obj.label_index for obj, _ in zooms.values()],
                 class_colors=args.class_colors, labels=args.labels, **common)
    zoom_metadata = {}
    for name, (obj, inside_indices) in zooms.items():
        crop_indices = np.flatnonzero(points_in_object(points, obj, padding=args.zoom_padding))
        heading = obj.corners[1] - obj.corners[0]
        azimuth = args.zoom_azimuth + np.rad2deg(np.arctan2(heading[1], heading[0]))
        zoom_basis = camera_basis(args.zoom_elevation, azimuth)
        zoom_limits = canvas_limits(obj.corners, zoom_basis, 1., margin=.15)
        render_panel(points[crop_indices], [obj], basis=zoom_basis, limits=zoom_limits,
                     output=output_dir / f"zoom_{name.lower()}", size=[args.zoom_size] * 2,
                     dpi=args.dpi, point_size=args.zoom_point_size, class_colors=args.class_colors)
        np.savez_compressed(output_dir / f"zoom_{name.lower()}_points.npz",
                            points=points[crop_indices], source_indices=crop_indices,
                            inside_indices=inside_indices, box_corners=obj.corners)
        zoom_metadata[name] = {"label_index_zero_based": obj.label_index,
            "points_inside_exact_box": len(inside_indices), "points_rendered_with_padding": len(crop_indices),
            "padding_m": args.zoom_padding, "occluded": obj.occluded, "truncated": obj.truncated,
            "score": obj.score, "corners_lidar_xyz": obj.corners.tolist(),
            "view": {"elevation": args.zoom_elevation, "azimuth": float(azimuth)}}
        print(f"  {name}: label {obj.label_index:02d}, {len(inside_indices)} interior returns, "
              f"{len(crop_indices)} rendered with context.", flush=True)
    counts = {name: sum(obj.name == name for obj in objects) for name in CLASSES}
    source_files = {"velodyne": bin_path, "calibration": cal_path, "ground_truth_labels": label_path}
    if prediction_path:
        source_files["prediction_labels"] = prediction_path
    metadata = {"frame_id": frame, "box_source": box_source, "counts": counts,
        "visible_box_counts": {name: sum(obj.name == name for obj in visible_objects) for name in CLASSES},
        "source_files": {key: {"path": str(path.resolve()), "sha256": sha256(path)}
                         for key, path in source_files.items()},
        "coordinate_system": "Velodyne XYZ in metres: x forward, y left, z up",
        "box_transform": "inverse(R0_rect @ Tr_velo_to_cam) applied to exact KITTI camera corners",
        "original_point_count": len(points), "scene_point_count": len(visible_indices), "roi_xyz": args.roi,
        "projection": "3D orthographic", "point_sampling": "all original samples within ROI/crop; no subsampling",
        "generated_points": 0, "zoom_objects": zoom_metadata,
        "render_settings": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "reproduce_command": [sys.executable, str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])]}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    composite_preview(output_dir, frame, counts, box_source, zooms)
    print(f"Saved PNG + SVG panels, zoom point subsets, preview and metadata to {output_dir}")
    return metadata


if __name__ == "__main__":
    main()
