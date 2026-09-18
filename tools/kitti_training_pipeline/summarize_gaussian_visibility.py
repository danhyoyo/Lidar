#!/usr/bin/env python3
"""Summarize held-out KITTI 3D AP, errors, and latency across ablations."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


CLASSES = ("Car", "Pedestrian", "Cyclist")


def extract(report):
    accuracy = report["accuracy"]["3d"]
    row = {
        "map_3d_moderate_percent": accuracy["map_moderate_percent"],
        "mean_3d_ap_9_percent": accuracy["mean_ap_9_percent"],
        "input_to_detections_ms": report["latency"]["input_to_detections"]["mean_ms"],
        "preprocess_ms": report["latency"]["preprocess"]["mean_ms"],
        "model_ms": report["latency"]["model"]["mean_ms"],
        "frames": report["data"]["frames"],
        "split_sha256": report["data"]["split_sha256"],
        "score_threshold": report["protocol"]["score_threshold"],
        "backbone": report["data"]["backbone"],
        "loss": report["data"]["loss"],
    }
    for class_name in CLASSES:
        prefix = class_name.lower()
        metric = accuracy["per_class"][class_name]["difficulties"]["Moderate"]
        row[f"{prefix}_ap_3d_moderate_percent"] = metric["ap_r40_percent"]
        row[f"{prefix}_precision"] = metric["precision_at_score_threshold"]
        row[f"{prefix}_recall"] = metric["max_recall"]
        row[f"{prefix}_false_positives_per_frame"] = metric["false_positives_per_frame"]
        row[f"{prefix}_observed_free_false_positives_per_frame"] = metric["observed_free_false_positives_per_frame"]
        for band, value in accuracy["per_class"][class_name].get(
                "moderate_distance_bands", {}).items():
            row[f"{prefix}_ap_3d_moderate_{band}_percent"] = value["ap_r40_percent"]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("artifacts/kitti"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/kitti/gaussian_visibility_irb_summary.csv"))
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    rows, missing = [], []
    for variant in ("a0", "a1", "a2", "a3", "a4"):
        for seed in args.seeds:
            path = args.results_root / f"gaussian_visibility_irb_{variant}_seed{seed}" / "report.json"
            if not path.is_file():
                missing.append(str(path))
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("status") != "ok" or report["data"]["frames"] != 997:
                raise ValueError(f"Expected complete 997-frame report: {path}")
            if not report["protocol"]["observed_free_diagnostic"]["enabled"]:
                raise ValueError(f"Missing observed-free diagnostic: {path}")
            rows.append({"variant": variant, "seed": seed, **extract(report)})
    if missing and not args.allow_incomplete:
        raise FileNotFoundError("Missing reports:\n  " + "\n  ".join(missing))
    if not rows:
        raise ValueError("No complete evaluation reports found")
    if len({row["split_sha256"] for row in rows}) != 1:
        raise ValueError("Reports use different splits")
    if len({row["score_threshold"] for row in rows}) != 1:
        raise ValueError("Reports use different score thresholds")
    if {row["backbone"] for row in rows} != {"mobilepixor"}:
        raise ValueError("Expected IRB MobilePIXOR reports")
    if {row["loss"] for row in rows} != {"baseline"}:
        raise ValueError("Expected standard focal/L1 loss reports")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)
    for variant in ("a0", "a1", "a2", "a3", "a4"):
        values = [row["map_3d_moderate_percent"] for row in rows
                  if row["variant"] == variant and row["map_3d_moderate_percent"] is not None]
        if values:
            spread = statistics.stdev(values) if len(values) > 1 else 0.0
            print(f"{variant}: 3D Moderate mAP R40 = {statistics.mean(values):.3f}"
                  f" +/- {spread:.3f} pp (n={len(values)})")


if __name__ == "__main__":
    main()
