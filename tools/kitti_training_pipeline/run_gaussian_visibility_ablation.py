#!/usr/bin/env python3
"""Train, select, and report one Gaussian-visibility KITTI ablation run."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from common import read_json, write_json


ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / "tools" / "kitti_training_pipeline"
CONFIGS = ROOT / "configs" / "kitti" / "gaussian_visibility"
VARIANTS = {
    "a0": "a0_rich8_baseline.json",
    "a1": "a1_gaussian_plain.json",
    "a2": "a2_gaussian_free.json",
    "a3": "a3_gaussian_height_free.json",
    "a4": "a4_uniform_plain.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kitti-root", required=True, type=Path)
    parser.add_argument("--processed-root", type=Path,
                        help="Prepared KITTI directory with pointcloud/ and label/")
    parser.add_argument("--output-root", type=Path, default=ROOT / "artifacts" / "kitti")
    parser.add_argument("--detector-root", type=Path, default=ROOT / "detector")
    parser.add_argument("--stage", choices=("all", "train", "select", "report"),
                        default="all")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--physical-batch-size", type=int)
    parser.add_argument("--accumulation-steps", type=int)
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"))
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config_source = CONFIGS / VARIANTS[args.variant]
    config = config_source
    # Keep the run name stable across train/select/report stages.  Existing
    # training artifacts use this canonical name without an extra prefix.
    run_name = f"gaussian_visibility_{args.variant}_seed{args.seed}"
    run_dir = args.output_root / run_name
    selected = run_dir / "selected_3d"
    eval_config = run_dir / "config.resolved.json"
    split_root = ROOT / "splits" / "kitti" / "gaussian_visibility"
    processed = args.processed_root
    if processed is not None:
        processed = processed.expanduser().resolve()
        for subdir in ("pointcloud", "label"):
            if not (processed / subdir).is_dir():
                parser.error(f"Prepared KITTI directory missing: {processed / subdir}")

    if args.stage in ("all", "select", "report"):
        for subdir in ("label_2", "calib"):
            path = args.kitti_root / "training" / subdir
            if not path.is_dir():
                parser.error(f"KITTI raw annotation directory missing: {path}")
    if args.resume and args.stage not in ("all", "train"):
        parser.error("--resume applies only to the training stage")

    def invoke(script, *arguments):
        command = [sys.executable, str(TOOLS / script), *map(str, arguments)]
        print("Running:", " ".join(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    if args.stage in ("all", "train"):
        source = read_json(config_source)
        location = processed or Path(source["data"]["kitti"]["location"])
        if not location.is_absolute():
            location = ROOT / location
        if not (location / "pointcloud").is_dir() or not (location / "label").is_dir():
            parser.error(f"Prepared KITTI missing at {location}; use --processed-root")
        source["data"]["kitti"]["location"] = str(location.resolve())
        config = run_dir / "config.input.json"
        write_json(config, source)
        command = [
            "--config", config, "--detector-root", args.detector_root,
            "--output-root", args.output_root, "--run-name", run_name,
            "--seed", args.seed, "--num-workers", args.num_workers,
            "--device", args.device,
        ]
        for option, value in (
            ("--epochs", args.epochs),
            ("--physical-batch-size", args.physical_batch_size),
            ("--accumulation-steps", args.accumulation_steps),
            ("--precision", args.precision),
        ):
            if value is not None:
                command.extend((option, value))
        if args.resume:
            command.extend(("--resume", args.resume))
        invoke("train.py", *command)

    if args.stage in ("all", "select", "report") and processed is not None:
        if not eval_config.is_file():
            parser.error(f"Training config missing: {eval_config}")
        resolved = read_json(eval_config)
        resolved["data"]["kitti"]["location"] = str(processed)
        eval_config = run_dir / "config.evaluation.json"
        write_json(eval_config, resolved)

    if args.stage in ("all", "select"):
        invoke(
            "select_checkpoint.py",
            "--checkpoint-dir", run_dir / "checkpoints",
            "--config", eval_config,
            "--detector-root", args.detector_root,
            "--kitti-root", args.kitti_root,
            "--split", split_root / "select.txt",
            "--output-dir", selected,
            "--device", args.device,
            "--score-threshold", args.score_threshold,
        )

    if args.stage in ("all", "report"):
        invoke(
            "evaluate_kitti_bev.py",
            "--name", run_name,
            "--backend", "pytorch",
            "--model", selected / "best.pt",
            "--config", eval_config,
            "--detector-root", args.detector_root,
            "--kitti-root", args.kitti_root,
            "--split", split_root / "report.txt",
            "--output", run_dir / "report.json",
            "--device", args.device,
            "--score-threshold", args.score_threshold,
            "--visibility-diagnostic",
        )
        print("Report:", run_dir / "report.json", flush=True)


if __name__ == "__main__":
    main()
