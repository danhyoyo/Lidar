#!/usr/bin/env python3
"""Select an Ours qualitative example by measured, fixed-threshold BEV errors."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import sys
import zipfile
import zlib

import numpy as np
from shapely.geometry import Polygon

from export_kitti_reference_comparison import (
    PROJECT, ROI, load_comparison_cloud,
)
from export_kitti_model_comparison import write_json, validate_variant
from tools.visualization.export_kitti_overview import read_objects, sha256
from tools.visualization.predict_kitti_overview import BEVOverviewPredictor, bev_detections

ROOT = PROJECT / "Demo/outputs/heatmap/ours_scene_search"
DATA = PROJECT / "Demo/data/kitti_groundtruth"
THRESHOLDS = {"Car": .7, "Pedestrian": .5, "Cyclist": .5}
URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip"


def in_roi(center):
    return ROI[0] <= center[0] <= ROI[1] and ROI[2] <= center[1] <= ROI[3]


def ground_truth(frame):
    return [obj for obj in read_objects(DATA / f"training/label_2/{frame}.txt",
                                       DATA / f"training/calib/{frame}.txt") if in_roi(obj.center)]


def measure(rows, objects, class_ids):
    """Count all displayed GT, with standard score-ordered class-aware matching."""
    predictions = [obj for obj in bev_detections(rows, class_ids) if in_roi(obj.row[2:4])]
    used, records = set(), []
    for prediction in sorted(predictions, key=lambda obj: (-obj.score, obj.index)):
        first = Polygon(prediction.footprint)
        candidates = []
        for index, obj in enumerate(objects):
            if index in used or obj.name != prediction.name:
                continue
            second = Polygon(obj.corners[:4, :2])
            value = first.intersection(second).area / first.union(second).area
            candidates.append((float(value), index))
        overlap, index = max(candidates, default=(0., -1))
        matched = overlap >= THRESHOLDS[prediction.name]
        if matched:
            used.add(index)
        records.append({"prediction_index": prediction.index, "class": prediction.name,
                        "score": prediction.score, "true_positive": matched,
                        "matched_gt_label_index": objects[index].label_index if matched else None,
                        "best_unmatched_same_class_iou": overlap})
    tp = len(used)
    fp, fn = len(predictions) - tp, len(objects) - tp
    return {"gt": len(objects), "predictions": len(predictions), "tp": tp, "fp": fp, "fn": fn,
            "errors": fp + fn, "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn), "f1": 2 * tp / max(1, 2 * tp + fp + fn),
            "unmatched_gt_label_indices": [obj.label_index for index, obj in enumerate(objects) if index not in used],
            "matches": records}


def prepare(count):
    ROOT.mkdir(parents=True, exist_ok=True)
    val = {line.split(";")[0] for line in (PROJECT / "paper/research/val_ids.txt").read_text().splitlines()}
    mapping = {row["object_id"]: row for row in csv.DictReader(
        (PROJECT / "Demo/data/temporal_mapping/object_to_raw_order.csv").open())}
    labels = zipfile.ZipFile("/tmp/kitti_groundtruth_label_2.zip")
    calib = zipfile.ZipFile("/tmp/kitti_groundtruth_calib.zip")
    by_drive = defaultdict(list)
    for frame in sorted(val):
        if frame in {"000008", "000134", "004338", "005663"} or frame not in mapping:
            continue
        text = labels.read(f"training/label_2/{frame}.txt").decode()
        fields = [line.split() for line in text.splitlines()]
        approx = [f for f in fields if f[0] in THRESHOLDS
                  and ROI[0] <= float(f[13]) <= ROI[1] and -ROI[3] <= float(f[11]) <= -ROI[2]]
        if not 5 <= len(approx) <= 12:
            continue
        for name, archive in (("label_2", labels), ("calib", calib)):
            path = DATA / f"training/{name}/{frame}.txt"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(f"training/{name}/{frame}.txt"))
        objects = ground_truth(frame)
        if not 5 <= len(objects) <= 12:
            continue
        easy = sum(obj.occluded == 0 and obj.truncated <= .15 for obj in objects)
        # Candidate ordering uses GT visibility only, never checkpoint outputs.
        by_drive[mapping[frame]["drive"]].append((frame, easy / len(objects), len(objects)))
    for drive in by_drive:
        by_drive[drive].sort(key=lambda row: (-row[1], -row[2], row[0]))
    chosen = []
    # Round-robin across drives; avoid near-duplicate scans within each drive.
    while len(chosen) < count and any(by_drive.values()):
        for drive in sorted(by_drive):
            if not by_drive[drive] or len(chosen) >= count:
                continue
            frame, visibility, gt_count = by_drive[drive].pop(0)
            raw = int(mapping[frame]["raw_frame"])
            chosen.append({"frame": frame, "drive": drive, "raw_frame": raw,
                           "easy_gt_fraction": visibility, "gt_in_roi": gt_count})
            by_drive[drive] = [item for item in by_drive[drive]
                               if abs(int(mapping[item[0]]["raw_frame"]) - raw) >= 30]
    sparse = Path("/tmp/kitti_search_remote_index.zip")
    with sparse.open("wb") as stream:
        stream.truncate(28750710812)
        stream.seek(28748613660)
        stream.write(Path("/tmp/kitti_velodyne_directory_tail.zip").read_bytes())
    archive = zipfile.ZipFile(sparse)
    config = []
    for index, item in enumerate(chosen):
        frame = item["frame"]
        member = archive.getinfo(f"training/velodyne/{frame}.bin")
        item.update({"member": member.filename, "offset": member.header_offset,
                     "compressed_size": member.compress_size, "size": member.file_size,
                     "crc32": member.CRC, "compression": member.compress_type,
                     "download": str(ROOT / f"{frame}.member")})
        if index:
            config.append("next")
        config.extend([f'url = "{URL}"', f'range = "{member.header_offset}-{member.header_offset + member.compress_size + 1023}"',
                       f'output = "{item["download"]}"', 'fail', 'silent', 'show-error', 'max-time = 60', 'retry = 2'])
    archive.close()
    sparse.unlink()
    write_json(ROOT / "candidates.json", {"frames": chosen, "roi_xyz": ROI.tolist(),
               "selection_before_inference": "validation; 5-12 displayed GT; visible objects; diverse drives; temporal separation >=30 frames"})
    (ROOT / "downloads.curl").write_text("\n".join(config) + "\n")
    print(f"Prepared {len(chosen)} validation scans across {len({x['drive'] for x in chosen})} drives", flush=True)
    for item in chosen:
        print(item["frame"], item["drive"], item["gt_in_roi"], flush=True)


def unpack():
    for item in json.loads((ROOT / "candidates.json").read_text())["frames"]:
        archive = Path(item["download"]).read_bytes()
        header = struct.unpack("<4s5H3I2H", archive[:30])
        assert header[0] == b"PK\x03\x04"
        assert archive[30:30 + header[-2]].decode() == item["member"]
        start = 30 + header[-2] + header[-1]
        compressed = archive[start:start + item["compressed_size"]]
        data = compressed if item["compression"] == 0 else zlib.decompress(compressed, -15)
        if len(data) != item["size"] or zlib.crc32(data) != item["crc32"]:
            raise ValueError(f"Official ZIP member CRC/size mismatch: {item['frame']}")
        target = DATA / item["member"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        write_json(target.with_suffix(".source.json"), {
            **{key: item[key] for key in ("member", "offset", "compressed_size", "size", "crc32", "compression")},
            "url": URL, "sha256": hashlib.sha256(data).hexdigest(), "zip_crc32_verified": True, "split": "validation"})
    print("All downloaded official KITTI members verified by size and CRC32", flush=True)


def evaluate():
    manifest = json.loads((PROJECT / "Demo/configs/reference_bev_zoom.json").read_text())
    item = next(item for item in manifest["models"] if item["id"] == "ours")
    checkpoint = Path(manifest["artifacts_root"]) / item["checkpoint"]
    runner = BEVOverviewPredictor(checkpoint, device="cpu", threads=4, score=.3, nms=.1, max_detections=500)
    validate_variant(item, runner)
    candidates = json.loads((ROOT / "candidates.json").read_text())["frames"]
    scores = []
    for item in candidates:
        frame = item["frame"]
        points, source = load_comparison_cloud(frame)
        rows, metadata, maps = runner.predict(points, capture_heatmaps=True)
        if not all(np.isfinite(value).all() for value in (rows, maps["logits"], maps["probabilities"])):
            raise ValueError("Nonfinite actual inference output")
        directory = ROOT / frame
        directory.mkdir(exist_ok=True)
        np.save(directory / "ours_predictions.npy", rows)
        np.savez_compressed(directory / "ours_heatmaps.npz", **maps)
        result = {"frame": frame, "drive": item["drive"], "source": source,
                  "inference": metadata,
                  "ground_truth_files": {name: sha256(DATA / f"training/{name}/{frame}.txt")
                                         for name in ("label_2", "calib")},
                  **measure(rows, ground_truth(frame), runner.config["data"]["kitti"]["objects"])}
        result["files"] = {name: sha256(directory / name) for name in ("ours_predictions.npy", "ours_heatmaps.npz")}
        write_json(directory / "metrics.json", result)
        scores.append(result)
        print(f"{frame}: GT={result['gt']} TP={result['tp']} FP={result['fp']} FN={result['fn']} F1={result['f1']:.3f}", flush=True)
    scores.sort(key=lambda row: (row["errors"], -row["f1"], -row["tp"], row["frame"]))
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "selection": "minimum FP+FN, then maximum F1, then maximum TP; Ours only",
              "scope": "selected qualitative example; best among tested candidates, not an unbiased benchmark",
              "score_threshold": .3, "nms_threshold": .1, "roi_xyz": ROI.tolist(),
              "iou_thresholds": THRESHOLDS, "matching": "same class, descending prediction score, one-to-one greedy IoU",
              "all_displayed_gt_counted": True, "candidate_count": len(scores),
              "best_frame": scores[0]["frame"], "ranked": scores,
              "script_sha256": sha256(Path(__file__))}
    write_json(ROOT / "ranking.json", report)
    print("BEST", scores[0]["frame"], "TP", scores[0]["tp"], "FP", scores[0]["fp"], "FN", scores[0]["fn"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "unpack", "evaluate"))
    parser.add_argument("--count", type=int, default=32)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(args.count)
    elif args.mode == "unpack":
        unpack()
    else:
        evaluate()
