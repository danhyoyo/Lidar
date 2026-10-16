"""Prepare one frozen Hist14/Hist22 pair, or evaluate its verified selected winner.

Preparation never launches training. Run the printed train commands explicitly;
the evaluate subcommand checks complete checkpoint selection history first.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

try:
    from .common import bev_encoding_spec, checkpoint_identity
    from .checkpoint_selection import selection_settings
    from .notebook_config import validate_notebook_config
    from .notebook_workflow import evaluation_command, selected_run
    from .paired_initialization import MODE, paired_policy
except ImportError:
    from common import bev_encoding_spec, checkpoint_identity
    from checkpoint_selection import selection_settings
    from notebook_config import validate_notebook_config
    from notebook_workflow import evaluation_command, selected_run
    from paired_initialization import MODE, paired_policy

ROOT = Path(__file__).resolve().parents[2]
ALLOWED_DIFFERENCES = ("data.bev_encoding.name", "data.bev_encoding.out_channels", "experiment.name")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def leaf_differences(first, second, prefix=""):
    """Exact differing leaf paths, also retaining additions/removals."""
    if isinstance(first, dict) and isinstance(second, dict):
        result = []
        for key in sorted(first.keys() | second.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in first or key not in second:
                result.append(path)
            else:
                result.extend(leaf_differences(first[key], second[key], path))
        return result
    return [] if first == second else [prefix]


def _assert_fresh_source(config):
    # An initialization policy describes construction; provenance describes a
    # model already constructed/trained. Never launder the latter into a new run.
    forbidden = {"initialization_provenance", "warm_start", "warm_start_checkpoint",
                 "resume", "resume_checkpoint", "pretrained", "pretrained_checkpoint", "checkpoint"}
    def inspect(value, prefix=""):
        if isinstance(value, dict):
            for key, child in value.items():
                path = f"{prefix}.{key}" if prefix else key
                if key.replace("-", "_") in forbidden:
                    raise ValueError(f"Fresh untrained baseline required: remove trained initialization/resume source {path} explicitly")
                inspect(child, path)
        elif isinstance(value, list):
            for child in value: inspect(child, prefix)
    inspect(config)
    if config.get("initialization") is not None and paired_policy(config) is None:
        raise ValueError("Fresh baseline requires no initialization or the same paired random policy")


def _shell(command, root):
    return f"cd {shlex.quote(str(root))} && {shlex.join(command)}"


def selected_evaluation_command(run_dir, output, *, repo_root=ROOT, device="cuda"):
    """Resolve and verify the configured AP/loss winner; never fall back to last.pt."""
    run = selected_run(run_dir)
    config = run["config"]
    settings = selection_settings(config)
    raw = config.get("evaluation", {}).get("kitti_root")
    if not raw:
        raise ValueError("Selected evaluation requires evaluation.kitti_root")
    return evaluation_command(run, settings["metric_mode"], raw, Path(output).resolve(),
                              repo_root=repo_root, device=device)


def prepare_pair(baseline_path, output_root, *, repo_root=ROOT, device="cuda"):
    """Validate then prepare exactly two fresh runs, refusing every output collision."""
    source_path = Path(baseline_path).expanduser().resolve()
    output = Path(output_root).expanduser().absolute()
    root = Path(repo_root).expanduser().resolve()
    for relative in ("tools/kitti_training_pipeline/train.py", "tools/kitti_training_pipeline/hist22_ablation.py", "detector"):
        if not (root / relative).exists():
            raise FileNotFoundError(f"Hist22 checkout is incomplete: {root / relative}")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Pair output already exists: {output}; choose a new directory")
    for relative in ("run.json", "metrics.jsonl", "checkpoints", "selected"):
        if (source_path.parent / relative).exists():
            raise ValueError("Baseline belongs to an existing training run; supply an explicitly fresh resolved source snapshot")
    source_bytes = source_path.read_bytes()
    source = json.loads(source_bytes)
    _canonical(source)
    _assert_fresh_source(source)
    schema = bev_encoding_spec(source)
    if schema.name != "hist14":
        raise ValueError("The resolved user baseline must use Hist14")
    if str(source["model"].get("backbone", "")).lower() != "mobilepixornext":
        raise ValueError("Hist22 paired ablation requires MobilePixorNeXt")
    geometry = dict(schema.geometry)
    if geometry["x_res"] != .1 or geometry["y_res"] != .1:
        raise ValueError("Hist22 pair requires x_res=y_res=0.1 m for both runs")
    validate_notebook_config(copy.deepcopy(source))
    seed = source.get("seed", 42)
    policy = {"mode": MODE, "version": 1, "seed": seed}
    configs = []
    for encoding, width, tag in (("hist14", 14, "base"), ("hist22", 22, "geom")):
        config = copy.deepcopy(source)
        config["seed"] = seed
        config["data"]["bev_encoding"] = {"name": encoding, "version": 1,
            "out_channels": width, "backend": schema.backend,
            "density_norm": schema.density_norm, "intensity_scale": schema.intensity_scale}
        config["initialization"] = copy.deepcopy(policy)
        config.setdefault("experiment", {})["name"] = f"{encoding}_{tag}_s{seed}"
        paired_policy(config)
        validate_notebook_config(copy.deepcopy(config))
        checkpoint_identity(config)
        configs.append(config)
    differences = leaf_differences(*configs)
    if differences != sorted(ALLOWED_DIFFERENCES):
        raise ValueError(f"Pair differs outside the exact allowed encoding/experiment leaves: {differences}")
    # Fingerprint the complete canonical configuration, including budget, split,
    # augmentation, backend and selector. Directory location is not the identity.
    runs = []
    for config in configs:
        digest = hashlib.sha256(_canonical(config)).hexdigest()
        name = config["experiment"]["name"] + "_" + digest[:20]
        config_path = output / "configs" / (name + ".json")
        run_dir = output / "runs" / name
        # training_command reads its config; build argv from validated settings
        # here so *all* preflight runs before writing the first artifact.
        training = config["train"]
        command = [sys.executable, str(root / "tools/kitti_training_pipeline/train.py"),
            "--config", str(config_path), "--detector-root", str(root / "detector"),
            "--output-root", str(output / "runs"), "--run-name", name, "--device", device,
            "--num-workers", str(training["num_workers"]), "--target-backend", training["target_backend"],
            "--precision", training["precision"]]
        if training.get("compile_model", False): command.append("--compile-model")
        evaluation = [sys.executable, str(root / "tools/kitti_training_pipeline/hist22_ablation.py"),
            "evaluate", "--run-dir", str(run_dir), "--output", str(output / "reports" / (name + ".json")),
            "--repo-root", str(root), "--device", device]
        runs.append({"label": "A" if len(runs) == 0 else "B", "encoding": bev_encoding_spec(config).name,
            "config_sha256": digest, "schema_sha256": bev_encoding_spec(config).semantic_hash,
            "backend": schema.backend, "run_name": name, "config_path": str(config_path), "run_dir": str(run_dir),
            "checkpoint_selection": selection_settings(config),
            "train_argv": command, "train_command": _shell(command, root),
            "evaluation_argv": evaluation, "evaluation_command": _shell(evaluation, root)})
    manifest = {"version": 1, "status": "prepared_only", "source_path": str(source_path),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(), "source_snapshot": str(output / "baseline.source.json"),
        "repo_root": str(root), "cwd": str(root), "initialization": policy,
        "allowed_difference_paths": sorted(ALLOWED_DIFFERENCES), "config_differences": differences,
        "runs": runs, "training_executed": False,
        "data_readiness": "not audited by preparation; validate KITTI assets, train-only GT database and GPU/precision before training"}
    pending = {"status": "pending_full_training", "metric": selection_settings(configs[0])["metric_mode"],
        "sampling": "R40", "checkpoint_primary": selection_settings(configs[0])["primary"],
        "runs": {r["label"]: {"run_name": r["run_name"], "checkpoint_epoch": None,
            "per_class_ap_percent": {name: {difficulty: None for difficulty in ("Easy", "Moderate", "Hard")}
                for name in ("Car", "Pedestrian", "Cyclist")}, "map_moderate_percent": None} for r in runs},
        "delta_B_minus_A_map_moderate_percent": None,
        "note": "Fill only after full paired detector training and selected-checkpoint evaluation. No Point AP or smoke/probe substitute."}
    # Every semantic and collision check above precedes any directory/write.
    # Exclusive mkdir reserves this pair; cleanup affects only our new directory.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    try:
        (output / "configs").mkdir()
        (output / "baseline.source.json").write_bytes(source_bytes)
        for run, config in zip(runs, configs):
            Path(run["config_path"]).write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        (output / "ap_report.pending.json").write_text(json.dumps(pending, indent=2, allow_nan=False) + "\n")
        commands = ["# Prepared only. Train explicitly; evaluate only after training completes."]
        for run in runs:
            commands.extend([f"# {run['label']}: {run['encoding']}", run["train_command"], run["evaluation_command"]])
        (output / "commands.sh").write_text("\n".join(commands) + "\n")
    except BaseException:
        shutil.rmtree(output)
        raise
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prepare = modes.add_parser("prepare", help="Prepare only: never start training")
    prepare.add_argument("--baseline", required=True, type=Path)
    prepare.add_argument("--output-root", required=True, type=Path)
    evaluate = modes.add_parser("evaluate", help="Verify the policy-selected checkpoint, then evaluate it")
    evaluate.add_argument("--run-dir", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    for mode in (prepare, evaluate):
        mode.add_argument("--repo-root", type=Path, default=ROOT)
        mode.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.mode == "prepare":
        manifest = prepare_pair(args.baseline, args.output_root, repo_root=args.repo_root, device=args.device)
        print(f"Prepared exactly two runs: {args.output_root.resolve() / 'manifest.json'}")
        for run in manifest["runs"]:
            print(run["train_command"])
            print(run["evaluation_command"])
    else:
        command = selected_evaluation_command(args.run_dir, args.output, repo_root=args.repo_root, device=args.device)
        subprocess.run(command, cwd=args.repo_root.resolve(), check=True)


if __name__ == "__main__":
    main()
