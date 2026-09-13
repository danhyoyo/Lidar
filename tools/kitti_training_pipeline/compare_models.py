#!/usr/bin/env python3
"""Evaluate multiple KITTI BEV models with one protocol and compare them."""
from __future__ import annotations

import argparse
import csv
import gc
import re
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from common import write_json
from evaluate_kitti_bev import CLASSES, DIFFICULTIES, run_evaluation


def parse_model(value: str) -> Tuple[str, str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("model must be NAME=BACKEND:PATH")
    name, remainder = value.split("=", 1)
    if ":" not in remainder:
        raise argparse.ArgumentTypeError("model must be NAME=BACKEND:PATH")
    backend, path = remainder.split(":", 1)
    if not name.strip() or backend not in {"pytorch", "tensorrt"} or not path:
        raise argparse.ArgumentTypeError("backend must be pytorch or tensorrt")
    return name.strip(), backend, Path(path)


def safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return result or "model"


def nested(value: Dict[str, Any], *keys, default=None):
    current: Any = value
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def flatten(result: Dict[str, Any]) -> Dict[str, Any]:
    accuracy = result.get("accuracy", {})
    primary = accuracy.get("3d", {})
    bev = accuracy.get("bev", accuracy)
    row = {
        "name": result.get("name"),
        "status": result.get("status"),
        "backend": nested(result, "model", "backend"),
        "path": nested(result, "model", "path"),
        "bytes": nested(result, "model", "bytes"),
        "sha256": nested(result, "model", "sha256"),
        "frames": nested(result, "data", "frames"),
        "mean_3d_ap_9_percent": primary.get("mean_ap_9_percent"),
        "map_3d_moderate_percent": primary.get("map_moderate_percent"),
        "mean_bev_ap_9_percent": bev.get("mean_ap_9_percent"),
        "map_bev_moderate_percent": bev.get("map_moderate_percent"),
        "detections": nested(result, "counts", "detections"),
        "torch_peak_memory_mb": nested(result, "runtime", "torch_peak_memory_mb"),
        "elapsed_seconds": nested(result, "runtime", "elapsed_seconds"),
        "error_type": result.get("error_type"),
        "error": result.get("error"),
    }
    for class_name in CLASSES:
        for difficulty in DIFFICULTIES:
            row[f"{class_name.lower()}_{difficulty.lower()}_3d_ap_r40_percent"] = nested(
                primary, "per_class", class_name, "difficulties",
                difficulty, "ap_r40_percent")
    for stage in ("preprocess", "host_to_device", "model", "decode_nms",
                  "input_to_detections"):
        for metric in ("mean_ms", "p50_ms", "p95_ms", "p99_ms", "fps"):
            row[f"{stage}_{metric}"] = nested(result, "latency", stage, metric)
    return row


def fmt(value, decimals=2) -> str:
    return "n/a" if value is None else f"{value:.{decimals}f}"


def markdown_table(results) -> str:
    lines = [
        "# KITTI MobileBEV model comparison",
        "",
        "All successful models used the same frame IDs, decoder, score threshold, NMS, ROI, IoU thresholds and AP R40 implementation.",
        "",
        "| Model | Status | 3D mAP Moderate (%) | Mean 3D AP-9 (%) | Model mean (ms) | Model p95 (ms) | Model FPS | E2E mean (ms) | E2E p95 (ms) | E2E FPS | Peak MiB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        if result.get("status") != "ok":
            lines.append(f"| {result.get('name')} | error: {result.get('error')} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {result['name']} | ok | "
            f"{fmt(nested(result, 'accuracy', '3d', 'map_moderate_percent'))} | "
            f"{fmt(nested(result, 'accuracy', '3d', 'mean_ap_9_percent'))} | "
            f"{fmt(nested(result, 'latency', 'model', 'mean_ms'))} | "
            f"{fmt(nested(result, 'latency', 'model', 'p95_ms'))} | "
            f"{fmt(nested(result, 'latency', 'model', 'fps'))} | "
            f"{fmt(nested(result, 'latency', 'input_to_detections', 'mean_ms'))} | "
            f"{fmt(nested(result, 'latency', 'input_to_detections', 'p95_ms'))} | "
            f"{fmt(nested(result, 'latency', 'input_to_detections', 'fps'))} | "
            f"{fmt(nested(result, 'runtime', 'torch_peak_memory_mb'), 1)} |"
        )
    lines.extend([
        "",
        "AP values are percentages. E2E here is the offline input-to-detections boundary and excludes ROS transport, tracking, rendering and HMI.",
        "",
    ])
    return "\n".join(lines)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--model", action="append", required=True, type=parse_model,
                       metavar="NAME=BACKEND:PATH")
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
    value.add_argument("--progress-every", type=int, default=50)
    value.add_argument("--fail-on-model-error", action="store_true")
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    names = [model[0] for model in args.model]
    if len(names) != len(set(names)):
        raise ValueError("Every --model NAME must be unique")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results = []
    for name, backend, path in args.model:
        output = args.output_dir / f"{safe_name(name)}.json"
        print(f"\n=== Evaluating {name} ({backend}) ===", flush=True)
        try:
            result = run_evaluation(
                name=name, backend=backend, model_path=path,
                config_path=args.config, detector_root=args.detector_root,
                kitti_root=args.kitti_root, split_path=args.split,
                output_path=output, device=args.device,
                score_threshold=args.score_threshold,
                nms_threshold=args.nms_threshold,
                max_detections=args.max_detections,
                warmup_frames=args.warmup_frames,
                max_frames=args.max_frames,
                progress_every=args.progress_every)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as error:
            result = {
                "status": "error", "name": name,
                "model": {"backend": backend, "path": str(path.resolve())},
                "error_type": type(error).__name__, "error": str(error),
            }
            write_json(output, result)
            print(f"[{name}] ERROR: {type(error).__name__}: {error}", flush=True)
        results.append(result)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    comparison = {
        "status": "partial" if any(x["status"] != "ok" for x in results) else "ok",
        "created_unix": time.time(), "elapsed_seconds": time.time() - started,
        "models_requested": len(results),
        "models_succeeded": sum(x["status"] == "ok" for x in results),
        "models_failed": sum(x["status"] != "ok" for x in results),
        "models": results,
    }
    json_path = args.output_dir / "comparison.json"
    csv_path = args.output_dir / "comparison.csv"
    md_path = args.output_dir / "comparison.md"
    write_json(json_path, comparison)
    rows = [flatten(result) for result in results]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md_path.write_text(markdown_table(results), encoding="utf-8")
    print(f"\nWrote {json_path.resolve()}")
    print(f"Wrote {csv_path.resolve()}")
    print(f"Wrote {md_path.resolve()}")
    print(markdown_table(results))
    if args.fail_on_model_error and comparison["models_failed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
