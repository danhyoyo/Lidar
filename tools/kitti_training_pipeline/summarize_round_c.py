#!/usr/bin/env python3
"""Summarize the locked C0/C1/C2 protocol and paired frame bootstrap."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from common import read_json, sha256, write_json


PROFILES = ("C0", "C1", "C2")
SEEDS = (42, 43, 44)
ROOT = Path(__file__).resolve().parents[2]
ROUND_DIR = ROOT / "results/kitti/probabilistic_geometry_uq/round_c"


def describe(values):
    values = [float(value) for value in values]
    if not values or not all(math.isfinite(value) for value in values):
        raise ValueError("summary values must be non-empty and finite")
    return {
        "mean": float(np.mean(values)),
        "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "values": values,
    }


def _moderate_map(predictions, labels):
    from evaluate_kitti_bev import evaluate_accuracy

    value = evaluate_accuracy(predictions, labels, space="3d")
    return float(value["map_moderate_percent"])


def paired_bootstrap(profile_predictions, labels, *, samples=2000,
                     rng_seed=42, metric=None):
    """Use one frame resample for every profile/seed in each iteration."""
    metric = metric or _moderate_map
    if set(profile_predictions) != set(PROFILES):
        raise ValueError(f"predictions must contain exactly {PROFILES}")
    for profile in PROFILES:
        if set(profile_predictions[profile]) != set(SEEDS):
            raise ValueError(f"{profile} must contain seeds {SEEDS}")
    frame_ids = sorted(labels)
    if not frame_ids:
        raise ValueError("bootstrap labels are empty")
    for profile in PROFILES:
        for seed in SEEDS:
            if set(profile_predictions[profile][seed]) != set(frame_ids):
                raise ValueError(f"{profile} seed {seed} prediction IDs differ")

    full = {
        profile: {
            seed: metric(profile_predictions[profile][seed], labels)
            for seed in SEEDS
        }
        for profile in PROFILES
    }
    comparisons = {
        "c1_minus_c0": ("C1", "C0"),
        "c2_minus_c1": ("C2", "C1"),
    }
    draws = {name: [] for name in comparisons}
    rng = np.random.default_rng(rng_seed)
    for _ in range(samples):
        sampled_ids = rng.choice(frame_ids, size=len(frame_ids), replace=True)
        sampled_labels = {
            str(index): labels[frame_id]
            for index, frame_id in enumerate(sampled_ids)
        }
        sampled_scores = {profile: {} for profile in PROFILES}
        for profile in PROFILES:
            for seed in SEEDS:
                sampled_predictions = {
                    str(index): profile_predictions[profile][seed][frame_id]
                    for index, frame_id in enumerate(sampled_ids)
                }
                sampled_scores[profile][seed] = metric(
                    sampled_predictions, sampled_labels
                )
        for name, (candidate, reference) in comparisons.items():
            draws[name].append(float(np.mean([
                sampled_scores[candidate][seed]
                - sampled_scores[reference][seed]
                for seed in SEEDS
            ])))

    result = {}
    for name, (candidate, reference) in comparisons.items():
        values = np.asarray(draws[name], dtype=np.float64)
        result[name] = {
            "estimate": float(np.mean([
                full[candidate][seed] - full[reference][seed]
                for seed in SEEDS
            ])),
            "ci95_lower": float(np.quantile(values, 0.025)),
            "ci95_upper": float(np.quantile(values, 0.975)),
            "samples": int(samples),
            "rng_seed": int(rng_seed),
        }
    return result


def _path(base, entry, key, default):
    value = Path(entry.get(key, default))
    return value if value.is_absolute() else base / value


def _load_entry(entry, base, calibration_hash, test_hash):
    profile, seed = str(entry["profile"]).upper(), int(entry["seed"])
    run_dir = _path(base, entry, "run_dir", ".")
    paths = {
        "run": _path(run_dir, entry, "run", "run.json"),
        "config": _path(run_dir, entry, "config", "config.resolved.json"),
        "selection": _path(run_dir, entry, "selection", "selected_3d/selection.json"),
        "checkpoint": _path(run_dir, entry, "checkpoint", "selected_3d/best.pt"),
        "evaluation": _path(run_dir, entry, "evaluation", "evaluation_test.json"),
        "predictions": _path(
            run_dir, entry, "predictions", "evaluation_test.predictions.npz"
        ),
        "calibration_predictions": _path(
            run_dir, entry, "calibration_predictions",
            "evaluation_calibration.predictions.npz",
        ),
        "uncertainty": _path(run_dir, entry, "uncertainty", "uncertainty_test.json"),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Round C artifacts:\n  " + "\n  ".join(missing))

    run = read_json(paths["run"])
    config = read_json(paths["config"])
    selection = read_json(paths["selection"])
    evaluation = read_json(paths["evaluation"])
    uncertainty = read_json(paths["uncertainty"])
    expected_run = {
        "seed": seed,
        "precision": "bf16",
        "epochs": 50,
        "physical_batch_size": 16,
        "accumulation_steps": 1,
    }
    for key, expected in expected_run.items():
        if run.get(key) != expected:
            raise ValueError(
                f"{profile} seed {seed}: run.json {key}={run.get(key)!r}, "
                f"expected {expected!r}"
            )
    allowed_variants = {profile}
    if profile == "C0" and seed == 42:
        allowed_variants.add("B3")
    if str(run.get("variant", "")).upper() not in allowed_variants:
        raise ValueError(f"{profile} seed {seed}: incompatible run variant")
    train = config.get("train", {})
    for key in ("precision", "epochs", "physical_batch_size", "accumulation_steps"):
        if train.get(key) != expected_run[key]:
            raise ValueError(f"{profile} seed {seed}: resolved config {key} mismatch")
    if config.get("seed") != seed:
        raise ValueError(f"{profile} seed {seed}: resolved seed mismatch")
    expected_architecture = {
        "C0": ("binary_slices", False),
        "C1": ("binary_slices", True),
        "C2": ("rich8", True),
    }[profile]
    architecture = (
        config.get("data", {}).get("bev_encoding", {}).get("name"),
        config.get("model", {}).get("scale_gated_fpn", False),
    )
    if architecture != expected_architecture:
        raise ValueError(f"{profile} seed {seed}: architecture mismatch")
    if (
        config.get("model", {}).get("predict_log_variance") is not True
        or config.get("loss", {}).get("name") != "probgeo_uq"
    ):
        raise ValueError(f"{profile} seed {seed}: ProbGeo-UQ config mismatch")
    config_hash = sha256(paths["config"])
    if selection.get("config_sha256") != config_hash:
        raise ValueError(f"{profile} seed {seed}: selection config hash mismatch")
    if evaluation.get("data", {}).get("config_sha256") != config_hash:
        raise ValueError(f"{profile} seed {seed}: evaluation config hash mismatch")
    if evaluation.get("data", {}).get("seed") != seed:
        raise ValueError(f"{profile} seed {seed}: evaluation seed mismatch")
    if selection.get("split_sha256") != calibration_hash:
        raise ValueError(f"{profile} seed {seed}: selection did not use calibration split")
    if evaluation.get("data", {}).get("split_sha256") != test_hash:
        raise ValueError(f"{profile} seed {seed}: evaluation did not use test split")
    if evaluation.get("model", {}).get("sha256") != sha256(paths["checkpoint"]):
        raise ValueError(f"{profile} seed {seed}: checkpoint hash mismatch")
    if evaluation.get("predictions", {}).get("sha256") != sha256(paths["predictions"]):
        raise ValueError(f"{profile} seed {seed}: prediction hash mismatch")
    checkpoint_hash = sha256(paths["checkpoint"])
    if selection.get("selected", {}).get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError(f"{profile} seed {seed}: selected checkpoint hash mismatch")
    protocol = evaluation.get("protocol", {})
    expected_protocol = {
        "score_threshold": 0.05,
        "nms_threshold": 0.10,
        "max_detections_per_frame": 500,
    }
    for key, expected in expected_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"{profile} seed {seed}: evaluation {key} mismatch")
    if (
        evaluation.get("data", {}).get("frames") != 997
        or evaluation.get("data", {}).get("full_split_frames") != 997
    ):
        raise ValueError(f"{profile} seed {seed}: incomplete test evaluation")
    uq_provenance = uncertainty.get("provenance", {})
    expected_uq_hashes = {
        "config_sha256": config_hash,
        "test_split_sha256": test_hash,
        "test_predictions_sha256": sha256(paths["predictions"]),
        "calibration_split_sha256": calibration_hash,
        "calibration_predictions_sha256": sha256(paths["calibration_predictions"]),
    }
    for key, expected in expected_uq_hashes.items():
        if uq_provenance.get(key) != expected:
            raise ValueError(f"{profile} seed {seed}: uncertainty {key} mismatch")

    with np.load(paths["predictions"]) as archive:
        predictions = {name: archive[name] for name in archive.files}
    return {
        "profile": profile,
        "seed": seed,
        "paths": {key: str(value.resolve()) for key, value in paths.items()},
        "hashes": {key: sha256(value) for key, value in paths.items()},
        "config": config,
        "evaluation": evaluation,
        "uncertainty": uncertainty,
        "predictions": predictions,
    }


def _profile_summary(runs):
    def evaluation_value(run, *keys):
        value = run["evaluation"]
        for key in keys:
            value = value[key]
        return value

    def uq_value(run, *keys):
        value = run["uncertainty"]
        for key in keys:
            value = value[key]
        return value

    return {
        "seeds": [run["seed"] for run in runs],
        "parameters": describe([
            evaluation_value(run, "model", "parameters") for run in runs
        ]),
        "map_3d_moderate_percent": describe([
            evaluation_value(run, "accuracy", "3d", "map_moderate_percent")
            for run in runs
        ]),
        "e2e_mean_ms": describe([
            evaluation_value(run, "latency", "input_to_detections", "mean_ms")
            for run in runs
        ]),
        "input_bytes_fp32": describe([
            evaluation_value(run, "data", "input_bytes_fp32") for run in runs
        ]),
        "gpu": sorted({evaluation_value(run, "runtime", "gpu") for run in runs}),
        "uq": {
            "calibrated_nll": describe([
                uq_value(run, "calibrated", "nll") for run in runs
            ]),
            "constant_nll": describe([
                uq_value(run, "baselines", "constant", "nll") for run in runs
            ]),
            "range_point_count_nll": describe([
                uq_value(run, "baselines", "range_point_count", "nll")
                for run in runs
            ]),
            "calibrated_aurc": describe([
                uq_value(run, "calibrated", "aurc") for run in runs
            ]),
            "score_only_aurc": describe([
                uq_value(run, "ranking_controls", "score_only_aurc")
                for run in runs
            ]),
            "shuffled_aurc": describe([
                uq_value(run, "ranking_controls", "shuffled_learned_aurc")
                for run in runs
            ]),
            "coverage_1sigma_gap": describe([
                float(np.mean(uq_value(
                    run, "calibrated", "coverage_1sigma_gap_per_dim"
                )))
                for run in runs
            ]),
        },
    }


def _uq_gate(profile):
    uq = profile["uq"]
    checks = {
        "nll_below_constant": (
            uq["calibrated_nll"]["mean"] < uq["constant_nll"]["mean"]
        ),
        "nll_below_range_point_count": (
            uq["calibrated_nll"]["mean"]
            < uq["range_point_count_nll"]["mean"]
        ),
        "aurc_below_score": (
            uq["calibrated_aurc"]["mean"] < uq["score_only_aurc"]["mean"]
        ),
        "aurc_below_shuffled": (
            uq["calibrated_aurc"]["mean"] < uq["shuffled_aurc"]["mean"]
        ),
        "coverage_gap_at_most_0_05": (
            uq["coverage_1sigma_gap"]["mean"] <= 0.05
        ),
    }
    return {"passed": all(checks.values()), "checks": checks}


def build_report(runs, bootstrap):
    profiles = {
        profile: _profile_summary(runs[profile]) for profile in PROFILES
    }
    gpu_values = [
        run["evaluation"].get("runtime", {}).get("gpu")
        for profile_runs in runs.values() for run in profile_runs
    ]
    latency_comparable = (
        len(gpu_values) == 9
        and all(isinstance(gpu, str) and "L4" in gpu for gpu in gpu_values)
        and len(set(gpu_values)) == 1
    )
    c1_latency = profiles["C1"]["e2e_mean_ms"]["mean"]
    c2_latency = profiles["C2"]["e2e_mean_ms"]["mean"]
    c1_bytes = profiles["C1"]["input_bytes_fp32"]["mean"]
    c2_bytes = profiles["C2"]["input_bytes_fp32"]["mean"]
    speedup = 1.0 - c2_latency / c1_latency
    byte_reduction = 1.0 - c2_bytes / c1_bytes
    uq_gates = {profile: _uq_gate(profiles[profile]) for profile in ("C1", "C2")}
    c1_accuracy = bootstrap["c1_minus_c0"]["ci95_lower"] > 0.0
    c2_accuracy = bootstrap["c2_minus_c1"]["ci95_lower"] > -1.0
    deployment_checks = {
        "same_nvidia_l4": latency_comparable,
        "e2e_at_least_20_percent_faster": latency_comparable and speedup >= 0.20,
        "input_bytes_at_least_50_percent_lower": byte_reduction >= 0.50,
    }
    gates = {
        "c1_accuracy": {
            "passed": c1_accuracy,
            "criterion": "95% CI lower bound of C1-C0 3D mAP Moderate > 0",
        },
        "c2_deployment": {
            "passed": c2_accuracy and all(deployment_checks.values()),
            "accuracy_passed": c2_accuracy,
            "speedup_fraction": speedup,
            "input_byte_reduction_fraction": byte_reduction,
            "checks": deployment_checks,
        },
        "probgeo_uq": uq_gates,
    }
    gates["locked_profiles"] = {
        "C1_accuracy": c1_accuracy and uq_gates["C1"]["passed"],
        "C2_deployment": (
            gates["c2_deployment"]["passed"] and uq_gates["C2"]["passed"]
        ),
    }
    return {
        "schema_version": 1,
        "protocol": {
            "seeds": list(SEEDS),
            "bootstrap_samples": 2000,
            "bootstrap_rng_seed": 42,
            "bootstrap_unit": "frame; shared draw across profiles and seeds",
        },
        "profiles": profiles,
        "paired_bootstrap": bootstrap,
        "gates": gates,
        "provenance": {
            profile: [
                {"seed": run["seed"], "paths": run["paths"], "hashes": run["hashes"]}
                for run in runs[profile]
            ]
            for profile in PROFILES
        },
    }


def write_outputs(report, json_path, markdown_path):
    write_json(json_path, report)
    lines = ["# Round C summary", ""]
    for profile in PROFILES:
        if profile not in report.get("profiles", {}):
            continue
        value = report["profiles"][profile]
        metric = value.get("map_3d_moderate_percent", value.get("accuracy"))
        lines.append(
            f"- {profile}: 3D mAP Moderate {metric['mean']:.4f} ± "
            f"{metric['sd']:.4f}"
        )
    lines.extend(["", "## Gates", ""])
    for name, gate in report.get("gates", {}).items():
        if isinstance(gate, dict) and "passed" in gate:
            lines.append(f"- {name}: {'PASS' if gate['passed'] else 'FAIL'}")
    Path(markdown_path).parent.mkdir(parents=True, exist_ok=True)
    Path(markdown_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", type=Path, default=ROUND_DIR / "runs.json")
    value.add_argument("--kitti-root", type=Path)
    value.add_argument("--output-dir", type=Path, default=ROUND_DIR)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    manifest = read_json(args.manifest)
    kitti_root = args.kitti_root or manifest.get("kitti_root")
    if not kitti_root:
        raise ValueError("--kitti-root or manifest.kitti_root is required")
    kitti_root = Path(kitti_root)
    calibration_split = ROOT / "splits/kitti/uq_calibration.txt"
    test_split = ROOT / "splits/kitti/uq_test.txt"
    calibration_hash, test_hash = sha256(calibration_split), sha256(test_split)
    base = args.manifest.resolve().parent
    indexed = {}
    for entry in manifest.get("runs", []):
        run = _load_entry(entry, base, calibration_hash, test_hash)
        key = (run["profile"], run["seed"])
        if key in indexed:
            raise ValueError(f"duplicate run {key}")
        indexed[key] = run
    expected = {(profile, seed) for profile in PROFILES for seed in SEEDS}
    if set(indexed) != expected:
        raise ValueError(f"manifest runs differ: missing={sorted(expected - set(indexed))}")
    runs = {
        profile: [indexed[(profile, seed)] for seed in SEEDS]
        for profile in PROFILES
    }

    from evaluate_kitti_bev import load_ground_truth, read_ids

    frame_ids = read_ids(test_split)
    geometry = runs["C0"][0]["config"]["data"]["kitti"]["geometry"]
    labels = {
        frame_id: load_ground_truth(frame_id, kitti_root, geometry)
        for frame_id in frame_ids
    }
    predictions = {
        profile: {run["seed"]: run["predictions"] for run in profile_runs}
        for profile, profile_runs in runs.items()
    }
    bootstrap = paired_bootstrap(
        predictions, labels, samples=2000, rng_seed=42
    )
    report = build_report(runs, bootstrap)
    write_outputs(
        report, args.output_dir / "summary.json", args.output_dir / "summary.md"
    )
    print(f"Wrote {(args.output_dir / 'summary.json').resolve()}")


if __name__ == "__main__":
    main()
