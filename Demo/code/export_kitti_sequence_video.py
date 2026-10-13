#!/usr/bin/env python3
"""Render a verified KITTI Object temporal sequence using a real BEV checkpoint."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / "Code"
sys.path.insert(0, str(ROOT))
from tools.visualization.export_kitti_overview import (
    CLASS_COLORS, camera_basis, canvas_limits, dataset_paths, projected,
    read_points, roi_vertices, setup_matplotlib, sha256,
)
from tools.visualization.export_kitti_gaussian_heatmap import crop_heatmaps
from tools.visualization.predict_kitti_overview import BEVOverviewPredictor, bev_detections

DEFAULT_MAPPING = ROOT.parent / "Result/artifacts/kitti_temporal_mapping"


def load_sequence(ids_path, mapping_dir, velodyne_dir):
    """Resolve Object IDs using KITTI's 1-based permutation, requiring consecutive frames."""
    identifiers = Path(ids_path).read_text().split()
    if (not identifiers or len(set(identifiers)) != len(identifiers)
            or any(len(value) != 6 or not value.isdigit() for value in identifiers)):
        raise ValueError("Sequence must contain distinct six-digit Object IDs")
    mapping_dir, velodyne_dir = Path(mapping_dir), Path(velodyne_dir)
    permutation = [int(value) for value in (mapping_dir / "train_rand.txt").read_text().strip().split(",")]
    mapping = [line.split() for line in (mapping_dir / "train_mapping.txt").read_text().splitlines()
               if line.strip()]
    if sorted(permutation) != list(range(1, len(mapping) + 1)):
        raise ValueError("Invalid official KITTI permutation")
    entries = []
    for identifier in identifiers:
        object_id = int(identifier)
        if object_id >= len(permutation):
            raise ValueError(f"Object ID {identifier} is outside the official mapping")
        fields = mapping[permutation[object_id] - 1]
        if len(fields) != 3 or not fields[2].isdigit():
            raise ValueError(f"Malformed KITTI mapping row for {identifier}")
        date, drive, raw_frame = fields
        if entries and (drive != entries[-1]["drive"] or date != entries[-1]["date"]
                        or int(raw_frame) != entries[-1]["raw_frame"] + 1):
            raise ValueError("Sequence must be consecutive in raw-frame order within one drive")
        source = velodyne_dir / f"{identifier}.bin"
        if not source.is_file():
            raise FileNotFoundError(source)
        entries.append({"object_id": identifier, "date": date, "drive": drive,
                        "raw_frame": int(raw_frame), "source_bin": str(source.resolve())})
    return entries


class SequenceRenderer:
    """Reuse artists with fixed projection, limits and colour scales across frames."""

    def __init__(self, *, roi, width, height, elevation, azimuth, heatmap_extent,
                 title, score_threshold, camera_image_size=None, right_camera_image_size=None):
        self.plt = plt = setup_matplotlib()
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Rectangle
        self.basis = camera_basis(elevation, azimuth)
        self.fig = plt.figure(figsize=(width / 100, height / 100), dpi=100, facecolor="#080d15")
        fig = self.fig
        scale = width / 1920
        with_camera = camera_image_size is not None
        if right_camera_image_size is not None and not with_camera:
            raise ValueError("Right camera requires a left camera")
        self.camera_artist = self.right_camera_artist = None
        heading = ("KITTI cameras · LiDAR · BEV detection · Classification heatmap" if with_camera
                   else "KITTI LiDAR · BEV detection · Classification heatmap")
        fig.text(.025, .955, heading,
                 color="white", fontsize=22 * scale, weight="bold", va="top")
        fig.text(.025, .90, title, color="#9dadc2", fontsize=12 * scale)
        self.status = fig.text(.975, .90, "", color="#dfe7f2", fontsize=12 * scale, ha="right")
        if with_camera:
            stereo = right_camera_image_size is not None
            for left, size, label, attribute in (
                (.025, camera_image_size, "Left colour camera / image_2", "camera_artist"),
                (.52, right_camera_image_size, "Right colour camera / image_3", "right_camera_artist"),
            ):
                if size is None:
                    continue
                image_width, image_height = size
                ax = fig.add_axes([left, .625, .455 if stereo else .95, .235], facecolor="black")
                ax.set_axis_off()
                artist = ax.imshow(np.zeros((image_height, image_width, 3), dtype=np.uint8),
                                   interpolation="nearest", aspect="equal")
                setattr(self, attribute, artist)
                fig.text(left, .875, label, color="white", fontsize=15 * scale, weight="bold")
        panel_bottom, panel_height = (.18, .365) if with_camera else (.18, .65)
        self.axes = [fig.add_axes([left, panel_bottom, .295, panel_height], facecolor="black")
                     for left in (.025, .353, .681)]
        for ax in self.axes:
            ax.set_axis_off()
        for left, text in zip((.025, .353, .681),
                              ("Measured LiDAR / 3D view", "Model prediction / BEV", "Model heatmap / max classes")):
            fig.text(left, .56 if with_camera else .845, text,
                     color="white", fontsize=17 * scale, weight="bold")
        scene_ax, bev_ax, heatmap_ax = self.axes
        limits = canvas_limits(roi_vertices(roi), self.basis,
                               aspect=(.295 * width) / (panel_height * height), margin=.04)
        scene_ax.set_xlim(limits[:2])
        scene_ax.set_ylim(limits[2:])
        scene_ax.set_aspect("equal")
        self.scene_points = scene_ax.scatter([], [], s=.38 * scale ** 2, linewidths=0, alpha=.95)
        bev_ax.set_xlim(heatmap_extent[:2])
        bev_ax.set_ylim(heatmap_extent[2:])
        self.bev_extent = heatmap_extent
        bev_ax.set_aspect("equal")
        self.bev_points = bev_ax.scatter([], [], s=.25 * scale ** 2, linewidths=0,
                                        color="#52d9e8", alpha=.8)
        self.box_artist = LineCollection([], linewidths=1.8 * scale, zorder=3)
        bev_ax.add_collection(self.box_artist)
        self.box_labels = []
        self.heatmap_artist = heatmap_ax.imshow(np.zeros((2, 2)), origin="lower",
            extent=heatmap_extent, cmap="viridis", vmin=0, vmax=1,
            interpolation="nearest", aspect="equal")
        colorbar_ax = fig.add_axes([.977, .23, .006, .27] if with_camera else [.977, .25, .006, .46])
        colorbar = fig.colorbar(self.heatmap_artist, cax=colorbar_ax, ticks=[0, .5, 1])
        colorbar.ax.tick_params(colors="#c1ccdd", labelsize=10 * scale, length=2)
        colorbar.outline.set_edgecolor("#24334b")
        self.plasma = plt.get_cmap("plasma")
        self.counts_text = fig.text(.353, .13, "", color="#dfe7f2", fontsize=12 * scale)
        fig.text(.025, .13, "Original XYZ points · height colours fixed", color="#9dadc2", fontsize=11 * scale)
        fig.text(.681, .13, "sigmoid(cls logits) · fixed scale 0–1", color="#9dadc2", fontsize=11 * scale)
        handles = [Rectangle((0, 0), 1, 1, facecolor=CLASS_COLORS[name])
                   for name in ("Car", "Pedestrian", "Cyclist")]
        fig.legend(handles, ["Car", "Pedestrian", "Cyclist"], loc="lower left",
                   bbox_to_anchor=(.345, .063), ncol=3, frameon=False,
                   labelcolor="white", fontsize=11 * scale)
        fig.text(.025, .063, f"BEV checkpoint · score ≥ {score_threshold:g} · independent frame inference",
                 color="#9dadc2", fontsize=10 * scale)
        fig.text(.681, .063, "x → forward    y ↑ left    No temporal smoothing",
                 color="#9dadc2", fontsize=10 * scale)
        self.timeline = Rectangle((.025, .027), 0, .006, transform=fig.transFigure,
                                  facecolor="#52d9e8", edgecolor="none")
        fig.add_artist(self.timeline)
        self.label_size = 10 * scale

    def render(self, points, rows, probabilities, *, class_ids, index, total, entry, fps, output,
               camera_image=None, right_camera_image=None):
        for artist, rgb in ((self.camera_artist, camera_image),
                            (self.right_camera_artist, right_camera_image)):
            if artist is not None:
                if rgb is None or rgb.dtype != np.uint8 or rgb.shape != artist.get_array().shape:
                    raise ValueError("Provide matching uint8 RGB data for each camera panel")
                artist.set_data(rgb)
            elif rgb is not None:
                raise ValueError("Camera data provided without a camera panel")
        uvz = projected(points, self.basis)
        order = np.argsort(uvz[:, 2], kind="stable")
        self.scene_points.set_offsets(uvz[order, :2])
        self.scene_points.set_facecolors(self.plasma(np.clip((points[order, 2] + 2.5) / 5., 0, 1)))
        self.bev_points.set_offsets(points[:, :2])
        objects = []
        for obj in bev_detections(rows, class_ids):
            xy_min, xy_max = obj.footprint.min(axis=0), obj.footprint.max(axis=0)
            x0, x1, y0, y1 = self.bev_extent
            if xy_min[0] <= x1 and xy_max[0] >= x0 and xy_min[1] <= y1 and xy_max[1] >= y0:
                objects.append(obj)
        segments, colours = [], []
        for label in self.box_labels:
            label.remove()
        self.box_labels.clear()
        for obj in objects:
            footprint = obj.footprint
            segments.append(np.r_[footprint, footprint[:1]])
            colours.append(CLASS_COLORS.get(obj.name, "#ff6257"))
            x, y = obj.row[2:4]
            self.box_labels.append(self.axes[1].text(x, y + obj.row[5] / 2 + .45,
                f"{obj.score:.2f}", color="white", fontsize=self.label_size, ha="center",
                va="bottom", clip_on=True,
                bbox=dict(facecolor="black", edgecolor="none", alpha=.6, pad=.5), zorder=4))
        self.box_artist.set_segments(segments)
        self.box_artist.set_colors(colours)
        self.heatmap_artist.set_data(probabilities.max(axis=0))
        counts = Counter(obj.name for obj in objects)
        self.counts_text.set_text("Visible: " + " · ".join(f"{name}: {counts[name]}" for name in ("Car", "Pedestrian", "Cyclist")))
        self.status.set_text(f"{index + 1}/{total}  |  Object {entry['object_id']}  |  Raw {entry['raw_frame']:04d}  |  {index / fps:.1f}s")
        self.timeline.set_width(.95 * (index + 1) / total)
        self.fig.savefig(output, dpi=100, facecolor=self.fig.get_facecolor())
        return dict(counts)

    def close(self):
        self.plt.close(self.fig)


def encode_video(frames_dir, output, *, fps, count, threads):
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(fps),
               "-start_number", "0", "-i", str(frames_dir / "frame_%06d.png"),
               "-frames:v", str(count), "-c:v", "libx264", "-preset", "medium", "-crf", "18",
               "-threads", str(threads), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output)]
    subprocess.run(command, check=True)
    probe = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_read_frames,duration",
        "-of", "json", str(output)], capture_output=True, text=True, check=True)
    stream = json.loads(probe.stdout)["streams"][0]
    if int(stream["nb_read_frames"]) != count:
        raise RuntimeError(f"Encoded frame count mismatch: {stream}")
    return command, stream


def create_contact_sheet(frames_dir, output, count, *, cell_height=420):
    from PIL import Image
    selected = sorted(set(np.linspace(0, count - 1, min(6, count), dtype=int).tolist()))
    sheet = Image.new("RGB", (1920, cell_height * ((len(selected) + 1) // 2)), "#080d15")
    for cell, index in enumerate(selected):
        with Image.open(frames_dir / f"frame_{index:06d}.png") as frame:
            frame = frame.convert("RGB")
            frame.thumbnail((950, cell_height - 25))
            left, top = (cell % 2) * 960, (cell // 2) * cell_height
            sheet.paste(frame, (left + (960 - frame.width) // 2, top + 20))
    sheet.save(output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping-dir", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--frame-list", type=Path, default=DEFAULT_MAPPING / "longest_consecutive_object_ids.txt")
    parser.add_argument("--reference-inference-dir", type=Path, default=ROOT.parent / "Result/artifacts/kitti_overview_model")
    parser.add_argument("--checkpoint", type=Path, help="Default: checkpoint used for the previous single-frame overview")
    parser.add_argument("--kitti-root", type=Path, default=ROOT / "Kitti_Raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT.parent / "Result/artifacts/kitti_sequence_video")
    parser.add_argument("--limit", type=int, help="Render the first N consecutive frames for a smoke run")
    parser.add_argument("--fps", type=float, default=10)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--roi", type=float, nargs=6, default=[-2, 45, -24, 24, -2.8, 4])
    parser.add_argument("--elevation", type=float, default=28)
    parser.add_argument("--azimuth", type=float, default=-145)
    parser.add_argument("--score-threshold", type=float, default=.3)
    parser.add_argument("--nms-threshold", type=float, default=.1)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    if (not np.isfinite(args.fps) or args.fps <= 0 or args.threads <= 0
            or args.width < 960 or args.height < 480 or args.width % 2 or args.height % 2):
        parser.error("Use positive fps/threads and even video dimensions, at least 960x480")
    if args.limit is not None and args.limit <= 0:
        parser.error("limit must be positive")
    roi = np.asarray(args.roi, dtype=float)
    if not np.isfinite(roi).all() or (roi[::2] >= roi[1::2]).any():
        parser.error("ROI bounds must be finite and increasing")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        parser.error("ffmpeg and ffprobe must be installed")
    paths = dataset_paths(args.kitti_root)
    entries = load_sequence(args.frame_list, args.mapping_dir, paths.velodyne)
    if args.limit:
        entries = entries[:args.limit]
    reference_path = args.reference_inference_dir / "metadata.json"
    reference = json.loads(reference_path.read_text()) if reference_path.is_file() else None
    if args.checkpoint:
        checkpoint = args.checkpoint.expanduser().resolve()
        if checkpoint.is_dir():
            checkpoint /= "best.pt"
    elif reference:
        checkpoint = Path(reference["inference"]["checkpoint"])
        if sha256(checkpoint) != reference["inference"]["checkpoint_sha256"]:
            raise ValueError("Checkpoint changed since the previous overview export")
    else:
        parser.error("Provide --checkpoint or an existing --reference-inference-dir")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    frames_dir, heatmap_dir = output / "frames", output / "heatmaps"
    frames_dir.mkdir(exist_ok=True)
    heatmap_dir.mkdir(exist_ok=True)
    print(f"Loading checkpoint once; {len(entries)} consecutive frames from {entries[0]['drive']}", flush=True)
    runner = BEVOverviewPredictor(checkpoint, device=args.device, threads=args.threads,
                                 score=args.score_threshold, nms=args.nms_threshold)
    config = runner.config
    class_ids = config["data"]["kitti"]["objects"]
    from tools.visualization.export_kitti_gaussian_heatmap import grid_shape
    dummy = np.zeros((len(class_ids), *reversed(grid_shape(config))), dtype=np.float32)
    _, extent, cell_indices = crop_heatmaps(dummy, config, roi[:4])
    # Use heatmap cell boundaries for matching XY in all three panels.
    display_roi = [extent[0], extent[1], extent[2], extent[3], roi[4], roi[5]]
    renderer = SequenceRenderer(roi=display_roi, width=args.width, height=args.height,
        elevation=args.elevation, azimuth=args.azimuth, heatmap_extent=extent,
        title=f"{entries[0]['drive']}  ·  raw {entries[0]['raw_frame']}–{entries[-1]['raw_frame']}  ·  {args.fps:g} FPS",
        score_threshold=args.score_threshold)
    predictions, reports = {}, []
    started = time.perf_counter()
    try:
        for index, entry in enumerate(entries):
            points = read_points(Path(entry["source_bin"]))
            rows, inference, heatmaps = runner.predict(points, capture_heatmaps=True)
            cropped, actual_extent, _ = crop_heatmaps(heatmaps["probabilities"], config, roi[:4])
            np.testing.assert_array_equal(actual_extent, extent)
            scene_indices = np.flatnonzero(((points[:, :3] >= np.asarray(display_roi)[::2])
                & (points[:, :3] <= np.asarray(display_roi)[1::2])).all(axis=1))
            visible_counts = renderer.render(points[scene_indices], rows, cropped, class_ids=class_ids,
                index=index, total=len(entries), entry=entry, fps=args.fps,
                output=frames_dir / f"frame_{index:06d}.png")
            if index == 0:
                shutil.copyfile(frames_dir / "frame_000000.png", output / "preview.png")
            heatmap_path = heatmap_dir / f"{entry['object_id']}.npz"
            np.savez_compressed(heatmap_path, **heatmaps, scene_source_indices=scene_indices)
            predictions[entry["object_id"]] = rows
            counts = Counter(obj.name for obj in bev_detections(rows, class_ids))
            reports.append({**entry, "video_index": index, "video_time_seconds": index / args.fps,
                "nominal_source_time_seconds": (entry["raw_frame"] - entries[0]["raw_frame"]) / 10,
                "source_bin_sha256": sha256(Path(entry["source_bin"])),
                "original_point_count": len(points), "scene_point_count": len(scene_indices),
                "detection_counts": dict(counts), "visible_detection_counts": visible_counts,
                "heatmap_sha256": sha256(heatmap_path),
                "inference_seconds": inference["elapsed_encoding_forward_decode_seconds"]})
            if index == 0 or (index + 1) % 10 == 0 or index + 1 == len(entries):
                print(f"{index + 1}/{len(entries)} · Object {entry['object_id']} · raw {entry['raw_frame']} · "
                      f"{dict(counts)} · elapsed {time.perf_counter() - started:.1f}s", flush=True)
    finally:
        renderer.close()
    np.savez_compressed(output / "bev_predictions.npz", **predictions)
    (output / "frame_manifest.json").write_text(json.dumps(reports, indent=2) + "\n")
    (output / "config.inference.json").write_text(json.dumps(config, indent=2) + "\n")
    create_contact_sheet(frames_dir, output / "contact_sheet.png", len(entries))
    print("Encoding H.264 MP4 and verifying decoded frame count", flush=True)
    video_path = output / "visualization.mp4"
    ffmpeg_command, stream = encode_video(frames_dir, video_path, fps=args.fps,
                                         count=len(entries), threads=args.threads)
    split_membership = {}
    for split in ("train", "val"):
        manifest = ROOT / config.get(split, {}).get("data", "__missing_manifest__")
        if manifest.is_file():
            members = {line.split(";", 1)[0].strip() for line in manifest.read_text().splitlines()}
            split_membership[split] = {"frame_count": sum(e["object_id"] in members for e in entries),
                                      "manifest": str(manifest), "sha256": sha256(manifest)}
    metadata = {"drive": entries[0]["drive"], "frame_count": len(entries),
        "raw_frame_range": [entries[0]["raw_frame"], entries[-1]["raw_frame"]],
        "consecutive_raw_frames_verified": True, "source_nominal_hz": 10,
        "absolute_timestamps_available": False, "fps": args.fps,
        "duration_seconds": len(entries) / args.fps, "video": stream,
        "video_sha256": sha256(video_path), "inference": runner.metadata,
        "model_load_count": 1, "ground_truth_used": False,
        "temporal_smoothing_applied": False, "generated_points": False,
        "panels": ["Measured XYZ LiDAR orthographic 3D view", "Predicted BEV boxes", "Predicted classification heatmap"],
        "requested_roi": args.roi, "display_roi": display_roi,
        "camera": {"elevation": args.elevation, "azimuth": args.azimuth},
        "fixed_height_colour_range_metres": [-2.5, 2.5], "heatmap_colour_range": [0, 1],
        "heatmap_axes": ["class", "y", "x"], "class_order": runner.class_order,
        "heatmap_crop_cell_indices": cell_indices, "heatmap_merged_operator": "max across classes",
        "mapping_sources": {name: {"path": str((args.mapping_dir / name).resolve()),
                                   "sha256": sha256(args.mapping_dir / name)}
                            for name in ("train_rand.txt", "train_mapping.txt")},
        "frame_list": {"path": str(args.frame_list.resolve()), "sha256": sha256(args.frame_list)},
        "split_membership": split_membership, "ffmpeg_command": ffmpeg_command,
        "elapsed_seconds": time.perf_counter() - started,
        "reproduce_command": [sys.executable, *sys.argv]}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved {video_path}: {len(entries)} frames, {len(entries) / args.fps:.1f}s", flush=True)


if __name__ == "__main__":
    main()
