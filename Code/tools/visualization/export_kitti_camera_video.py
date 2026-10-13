#!/usr/bin/env python3
"""Pair KITTI camera PNGs with the saved LiDAR, BEV predictions and heatmaps."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import time
from zipfile import ZipFile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.visualization.export_kitti_overview import read_points, sha256
from tools.visualization.export_kitti_gaussian_heatmap import crop_heatmaps, grid_shape
from tools.visualization.export_kitti_sequence_video import (
    SequenceRenderer, create_contact_sheet, encode_video, load_sequence,
)

ARTIFACTS = ROOT.parent / "Result/artifacts"


def camera_paths(entries, image_dir):
    """Match the original Object filename, never the video index or raw-frame ID."""
    paths = [Path(image_dir) / f"{entry['object_id']}.png" for entry in entries]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} matching camera images:\n" + "\n".join(missing[:20]))
    return paths


def extract_camera_zip(source, destination, entries):
    """Extract only exact training camera members after checking all source PNGs."""
    source, destination = Path(source), Path(destination)
    ids = [entry["object_id"] for entry in entries]
    members = [f"training/{camera}/{identifier}.png"
               for identifier in ids for camera in ("image_2", "image_3")]
    with ZipFile(source) as archive:
        names = archive.namelist()
        missing = [name for name in members if name not in names]
        if missing:
            raise FileNotFoundError("Camera ZIP is missing exact training members:\n" + "\n".join(missing[:20]))
        if any(names.count(name) != 1 for name in members):
            raise ValueError("Duplicate camera members in ZIP")
        if "frame_ids.txt" in names:
            if archive.read("frame_ids.txt").decode().split() != ids:
                raise ValueError("ZIP frame_ids.txt does not match the saved video order")
        image_size = None
        for name in members:
            with Image.open(io.BytesIO(archive.read(name))) as image:
                image.load()
                if image.mode != "RGB" or (image_size is not None and image.size != image_size):
                    raise ValueError(f"Expected consistently sized RGB images: {name}")
                image_size = image.size
        for name in members:
            output = destination / name
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(archive.read(name))
    report = {"source_zip": str(source.resolve()), "source_zip_sha256": sha256(source),
              "image_count": len(members), "image_size": list(image_size),
              "split": "training", "frame_ids": ids}
    (destination / "extraction_metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def read_rgb(path):
    with Image.open(path) as image:
        if image.mode != "RGB":
            raise ValueError(f"Expected RGB image: {path}")
        return np.array(image)


def verify_cached_frame(entry, cache_dir, config, display_roi, class_order):
    """Require the original scan and the exact saved full-resolution model maps."""
    source = Path(entry["source_bin"])
    if sha256(source) != entry["source_bin_sha256"]:
        raise ValueError(f"LiDAR source changed: {source}")
    points = read_points(source)
    if len(points) != entry["original_point_count"]:
        raise ValueError("Original point count changed")
    path = cache_dir / "heatmaps" / f"{entry['object_id']}.npz"
    if sha256(path) != entry["heatmap_sha256"]:
        raise ValueError(f"Cached heatmap changed: {path}")
    with np.load(path, allow_pickle=False) as heatmaps:
        probabilities = heatmaps["probabilities"]
        if (probabilities.shape != (len(class_order), *reversed(grid_shape(config)))
                or heatmaps["class_order"].tolist() != class_order
                or not np.isfinite(probabilities).all()):
            raise ValueError("Cached heatmap shape/classes/values do not match the model")
        np.testing.assert_allclose(probabilities, 1 / (1 + np.exp(-heatmaps["logits"])),
                                   rtol=2e-6, atol=1e-7)
        indices = heatmaps["scene_source_indices"]
        expected = np.flatnonzero(((points[:, :3] >= np.asarray(display_roi)[::2])
            & (points[:, :3] <= np.asarray(display_roi)[1::2])).all(axis=1))
        np.testing.assert_array_equal(indices, expected)
        if len(indices) != entry["scene_point_count"]:
            raise ValueError("Scene point count changed")
        cropped, extent, _ = crop_heatmaps(probabilities, config, display_roi[:4])
    np.testing.assert_array_equal(extent, display_roi[:4])
    return points[indices], cropped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=ARTIFACTS / "kitti_sequence_video")
    parser.add_argument("--images-zip", type=Path,
        default=Path.home() / "Downloads/kitti_training_174frames_images.zip")
    parser.add_argument("--image-dir", type=Path,
        help="Existing image root containing training/image_2 and training/image_3; bypass ZIP extraction")
    parser.add_argument("--output-dir", type=Path, default=ARTIFACTS / "kitti_sequence_video_camera")
    parser.add_argument("--camera-mode", choices=("stereo", "left"), default="stereo")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, help="Render the first N frames for a smoke check")
    args = parser.parse_args(argv)
    if (args.width < 960 or args.height < 540 or args.width % 2 or args.height % 2
            or args.threads < 1 or (args.limit is not None and args.limit < 1)):
        parser.error("Use even dimensions >=960x540, positive threads/limit")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        parser.error("ffmpeg and ffprobe are required")
    cache_dir = args.cache_dir.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output == cache_dir:
        parser.error("Use a separate output directory for the camera video")
    metadata = json.loads((cache_dir / "metadata.json").read_text())
    entries = json.loads((cache_dir / "frame_manifest.json").read_text())
    config = json.loads((cache_dir / "config.inference.json").read_text())
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if config_hash != metadata["inference"]["config_sha256"]:
        raise ValueError("Cached model configuration changed")
    if metadata["ground_truth_used"] or metadata["generated_points"]:
        raise ValueError("Expected saved model inference on measured points")
    if len(entries) != metadata["frame_count"]:
        raise ValueError("Cached frame count changed")
    for reference in (*metadata["mapping_sources"].values(), metadata["frame_list"]):
        if sha256(Path(reference["path"])) != reference["sha256"]:
            raise ValueError("Official mapping or selected frame list changed")
    mapping_dir = Path(metadata["mapping_sources"]["train_mapping.txt"]["path"]).parent
    official = load_sequence(metadata["frame_list"]["path"], mapping_dir,
                             Path(entries[0]["source_bin"]).parent)
    for index, (saved, mapped) in enumerate(zip(entries, official)):
        if (saved["video_index"] != index or any(saved[key] != mapped[key]
                for key in ("object_id", "date", "drive", "raw_frame", "source_bin"))):
            raise ValueError("Saved frames do not match the verified official timeline")
    if args.image_dir:
        image_dir = args.image_dir.expanduser().resolve()
        extraction = None
    else:
        image_dir = ARTIFACTS / "kitti_sequence_images"
        print("Validating and extracting exact left/right camera pairs from the Colab ZIP", flush=True)
        extraction = extract_camera_zip(args.images_zip.expanduser().resolve(), image_dir, entries)
    left_paths = camera_paths(entries, image_dir / "training/image_2")
    right_paths = (camera_paths(entries, image_dir / "training/image_3")
                   if args.camera_mode == "stereo" else None)
    if args.limit:
        entries = entries[:args.limit]
    output.mkdir(parents=True, exist_ok=True)
    frames_dir = output / "frames"
    frames_dir.mkdir(exist_ok=True)
    left_size = read_rgb(left_paths[0]).shape[1::-1]
    right_size = read_rgb(right_paths[0]).shape[1::-1] if right_paths else None
    renderer = SequenceRenderer(roi=metadata["display_roi"], width=args.width, height=args.height,
        elevation=metadata["camera"]["elevation"], azimuth=metadata["camera"]["azimuth"],
        heatmap_extent=metadata["display_roi"][:4],
        title=f"{metadata['drive']}  ·  raw {entries[0]['raw_frame']}–{entries[-1]['raw_frame']}  ·  {metadata['fps']:g} FPS",
        score_threshold=metadata["inference"]["score_threshold"],
        camera_image_size=left_size, right_camera_image_size=right_size)
    started, reports = time.perf_counter(), []
    class_ids = config["data"]["kitti"]["objects"]
    predictions_path = cache_dir / "bev_predictions.npz"
    try:
        with np.load(predictions_path, allow_pickle=False) as predictions:
            if set(predictions.files) != {entry["object_id"] for entry in official}:
                raise ValueError("Cached prediction IDs do not match the original sequence")
            for index, entry in enumerate(entries):
                points, heatmaps = verify_cached_frame(entry, cache_dir, config,
                    metadata["display_roi"], metadata["class_order"])
                visible_counts = renderer.render(points, predictions[entry["object_id"]], heatmaps,
                    class_ids=class_ids, index=index, total=len(entries), entry=entry, fps=metadata["fps"],
                    camera_image=read_rgb(left_paths[index]),
                    right_camera_image=read_rgb(right_paths[index]) if right_paths else None,
                    output=frames_dir / f"frame_{index:06d}.png")
                if visible_counts != entry["visible_detection_counts"]:
                    raise ValueError("Displayed detection counts changed during replay")
                reports.append({**entry, "left_image": str(left_paths[index]),
                    "left_image_sha256": sha256(left_paths[index]),
                    "right_image": str(right_paths[index]) if right_paths else None,
                    "right_image_sha256": sha256(right_paths[index]) if right_paths else None})
                if index == 0 or (index + 1) % 10 == 0 or index + 1 == len(entries):
                    print(f"{index + 1}/{len(entries)} · Object {entry['object_id']} · raw {entry['raw_frame']} · "
                          f"elapsed {time.perf_counter() - started:.1f}s", flush=True)
    finally:
        renderer.close()
    shutil.copyfile(frames_dir / "frame_000000.png", output / "preview.png")
    create_contact_sheet(frames_dir, output / "contact_sheet.png", len(entries), cell_height=560)
    print("Encoding and verifying the camera MP4", flush=True)
    video_path = output / "visualization.mp4"
    command, stream = encode_video(frames_dir, video_path, fps=metadata["fps"],
        count=len(entries), threads=args.threads)
    (output / "frame_manifest.json").write_text(json.dumps(reports, indent=2) + "\n")
    report = {"drive": metadata["drive"], "frame_count": len(entries),
        "raw_frame_range": [entries[0]["raw_frame"], entries[-1]["raw_frame"]],
        "consecutive_raw_frames_verified": True, "fps": metadata["fps"],
        "duration_seconds": len(entries) / metadata["fps"], "video": stream,
        "video_sha256": sha256(video_path), "camera_mode": args.camera_mode,
        "left_image_size": list(left_size), "right_image_size": list(right_size) if right_size else None,
        "camera_images_used_for_inference": False, "camera_box_overlay_applied": False,
        "inference": metadata["inference"], "inference_reused_from": str(cache_dir),
        "new_model_forward_count": 0, "ground_truth_used": False, "generated_points": False,
        "temporal_smoothing_applied": False, "camera_zip": extraction,
        "panels": ["Left colour camera / image_2"]
            + (["Right colour camera / image_3"] if right_paths else [])
            + metadata["panels"],
        "cache_sources": {name: {"path": str(cache_dir / name), "sha256": sha256(cache_dir / name)}
            for name in ("metadata.json", "frame_manifest.json", "config.inference.json", "bev_predictions.npz")},
        "split_membership": metadata["split_membership"],
        "display_roi": metadata["display_roi"], "heatmap_colour_range": [0, 1],
        "class_order": metadata["class_order"], "ffmpeg_command": command,
        "elapsed_seconds": time.perf_counter() - started,
        "reproduce_command": [sys.executable, *sys.argv]}
    (output / "metadata.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {video_path}: {len(entries)} frames, {len(entries) / metadata['fps']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
