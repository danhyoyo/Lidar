#!/usr/bin/env python3
"""Select a saved checkpoint by locked KITTI BEV validation metrics."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from common import read_json, sha256, write_json
from evaluate_kitti_bev import run_evaluation


def select_best(records):
    usable = [record for record in records if record["map_moderate_percent"] is not None]
    if not usable:
        raise ValueError("No checkpoint produced a Moderate mAP")
    return max(
        usable,
        key=lambda record: (
            record["map_moderate_percent"],
            record["mean_ap_9_percent"],
            -record["epoch"],
        ),
    )


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint-dir", required=True, type=Path)
    value.add_argument("--config", required=True, type=Path)
    value.add_argument("--detector-root", required=True, type=Path)
    value.add_argument("--kitti-root", required=True, type=Path)
    value.add_argument("--split", required=True, type=Path)
    value.add_argument("--output-dir", required=True, type=Path)
    value.add_argument("--device", default="cuda")
    value.add_argument("--score-threshold", type=float, default=0.05)
    value.add_argument("--nms-threshold", type=float, default=0.10)
    value.add_argument("--max-detections", type=int, default=500)
    value.add_argument("--warmup-frames", type=int, default=10)
    value.add_argument("--max-frames", type=int)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    config = read_json(args.config)
    checkpoints = []
    for path in args.checkpoint_dir.glob("*epoch.pt"):
        match = re.fullmatch(r"(\d+)epoch\.pt", path.name)
        if match:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort()
    if not checkpoints:
        raise FileNotFoundError(f"No *epoch.pt checkpoints in {args.checkpoint_dir}")

    evaluations_dir = args.output_dir / "evaluations"
    evaluations_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for epoch, checkpoint in checkpoints:
        evaluation_path = evaluations_dir / f"epoch_{epoch:03d}.json"
        result = run_evaluation(
            name=f"epoch_{epoch:03d}", backend="pytorch", model_path=checkpoint,
            config_path=args.config, detector_root=args.detector_root,
            kitti_root=args.kitti_root, split_path=args.split,
            output_path=evaluation_path, device=args.device,
            score_threshold=args.score_threshold, nms_threshold=args.nms_threshold,
            max_detections=args.max_detections, warmup_frames=args.warmup_frames,
            max_frames=args.max_frames,
        )
        accuracy = result["accuracy"]
        records.append({
            "epoch": epoch,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256(checkpoint),
            "evaluation": str(evaluation_path.resolve()),
            "map_moderate_percent": accuracy["map_moderate_percent"],
            "mean_ap_9_percent": accuracy["mean_ap_9_percent"],
        })

    selected = select_best(records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "best.pt"
    shutil.copy2(selected["checkpoint"], destination)
    write_json(args.output_dir / "selection.json", {
        "criterion": "max Moderate BEV mAP, then mean BEV AP-9, then earlier epoch",
        "config_sha256": sha256(args.config),
        "split_sha256": sha256(args.split),
        "selected": {**selected, "copied_checkpoint": str(destination.resolve())},
        "evaluated": records,
    })
    print(f"Selected epoch {selected['epoch']}: {destination.resolve()}")


if __name__ == "__main__":
    main()
