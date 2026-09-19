#!/usr/bin/env python3
"""Create active-sampling manifests and evaluate the label-informed oracle."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PIPELINE = ROOT / "tools" / "kitti_training_pipeline"
if str(PIPELINE) not in sys.path:
    sys.path.insert(0, str(PIPELINE))

from common import read_json, sha256, write_json


RATES = np.array([0.25, 0.50, 0.75, 1.00], dtype=np.float64)
TIMING_STATS = ("mean_ms", "p50_ms", "p95_ms", "p99_ms")


def split_ids(frame_ids, seed=42, dev_count=1200):
    unique = set(frame_ids)
    if len(unique) != len(frame_ids):
        raise ValueError("duplicate frame IDs")
    if not 0 < dev_count < len(frame_ids):
        raise ValueError("dev_count must leave both splits non-empty")
    ordered = sorted(unique, key=lambda value: hashlib.sha256(
        f"{seed}:{value}".encode("utf-8")).digest())
    return ordered[dev_count:], ordered[:dev_count]


def choose_actions(qualities, rates, beta):
    utilities = qualities - beta * rates[None, :]
    return np.asarray([
        max(range(len(rates)), key=lambda index: (row[index], -rates[index]))
        for row in utilities
    ], dtype=np.int64)


def read_ids(path):
    values = [line.split(";", 1)[0].strip()
              for line in Path(path).read_text(encoding="utf-8").splitlines()
              if line.strip()]
    if not all(values) or len(values) != len(set(values)):
        raise ValueError(f"invalid or duplicate frame IDs in {path}")
    return values


def write_split(path, frame_ids):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(f"{frame_id};kitti\n" for frame_id in frame_ids), encoding="utf-8"
    )
    os.replace(temporary, path)


def snapshot_baseline(args):
    source = Path(args.source).resolve()
    result = read_json(source)
    result["checkpoint_epoch"] = args.epoch
    result["baseline_source"] = {"path": str(source), "sha256": sha256(source)}
    write_json(args.output, result)


def create_splits(args):
    policy_train, controller_dev = split_ids(read_ids(args.source), args.seed, args.dev_count)
    write_split(args.policy_train, policy_train)
    write_split(args.controller_dev, controller_dev)


def load_prediction_archive(path, frame_ids):
    with np.load(path, allow_pickle=False) as archive:
        if list(archive.files) != list(frame_ids):
            raise ValueError(f"prediction frame IDs do not match controller split: {path}")
        return {frame_id: np.array(archive[frame_id], copy=True) for frame_id in frame_ids}


def same_predictions(first, second):
    return all(
        first[key].dtype == second[key].dtype
        and first[key].shape == second[key].shape
        and np.array_equal(first[key], second[key])
        for key in first
    )


def median_latency(results):
    stages = set(results[0]["latency"])
    if any(set(result["latency"]) != stages for result in results[1:]):
        raise ValueError("latency stages differ across repeats")
    return {
        stage: {
            stat: float(np.median([result["latency"][stage][stat] for result in results]))
            for stat in TIMING_STATS
        }
        for stage in sorted(stages)
    }


def validate_results(paths, frame_ids, config_path, split_path):
    grouped = {float(rate): [] for rate in RATES}
    expected_config = sha256(config_path)
    expected_split = sha256(split_path)
    shared = None
    for path in paths:
        result = read_json(path)
        sampling = result.get("sampling")
        if result.get("status") != "ok" or not isinstance(sampling, dict):
            raise ValueError(f"missing successful sampling result: {path}")
        if sampling.get("contract_version") != 1:
            raise ValueError(f"unknown sampling contract: {path}")
        rate = float(sampling.get("rate", -1))
        if rate not in grouped:
            raise ValueError(f"unknown sampling rate {rate}: {path}")
        per_frame = sampling.get("per_frame")
        if not isinstance(per_frame, dict) or list(per_frame) != frame_ids:
            raise ValueError(f"sampling frame IDs do not match controller split: {path}")
        data = result.get("data", {})
        model = result.get("model", {})
        signature = (
            data.get("config_sha256"), data.get("split_sha256"),
            model.get("sha256"), sampling.get("seed"), tuple(per_frame),
        )
        if data.get("config_sha256") != expected_config or data.get("split_sha256") != expected_split:
            raise ValueError(f"config or split hash mismatch: {path}")
        if shared is None:
            shared = signature
        elif signature != shared:
            raise ValueError(f"result provenance mismatch: {path}")
        grouped[rate].append((Path(path), result))

    if any(len(values) != 3 for values in grouped.values()):
        counts = {rate: len(values) for rate, values in grouped.items()}
        raise ValueError(f"expected exactly three results per rate, got {counts}")

    predictions = {}
    for rate, values in grouped.items():
        reference_counts = values[0][1]["sampling"]["per_frame"]
        if any(value[1]["sampling"]["per_frame"] != reference_counts for value in values[1:]):
            raise ValueError(f"point counts differ across repeats at rate {rate}")
        first = load_prediction_archive(values[0][1]["predictions"]["path"], frame_ids)
        for _, result in values[1:]:
            repeated = load_prediction_archive(result["predictions"]["path"], frame_ids)
            if not same_predictions(first, repeated):
                raise ValueError(f"predictions differ across repeats at rate {rate}")
        predictions[rate] = first
    return grouped, predictions, shared


def moderate_metric(accuracy, class_name):
    return accuracy["3d"]["per_class"][class_name]["difficulties"]["Moderate"]


def accuracy_guard(accuracy, dense):
    classes = {}
    passed = dense["3d"]["map_moderate_percent"] - accuracy["3d"]["map_moderate_percent"] <= 0.8
    for class_name in ("Car", "Pedestrian", "Cyclist"):
        current = moderate_metric(accuracy, class_name)
        baseline = moderate_metric(dense, class_name)
        ap_drop = baseline["ap_r40_percent"] - current["ap_r40_percent"]
        recall_drop = baseline["max_recall"] - current["max_recall"]
        class_pass = ap_drop <= 1.5 and recall_drop <= 0.03
        passed &= class_pass
        entry = {"ap_drop_points": ap_drop, "max_recall_drop": recall_drop,
                 "passes": bool(class_pass)}
        if class_name in {"Pedestrian", "Cyclist"}:
            entry["distance_bands"] = {}
            current_bands = accuracy["3d"]["per_class"][class_name]["moderate_distance_bands"]
            dense_bands = dense["3d"]["per_class"][class_name]["moderate_distance_bands"]
            for name, baseline_band in dense_bands.items():
                band = current_bands[name]
                recall_drop = baseline_band["max_recall"] - band["max_recall"]
                hard_guard = baseline_band["ground_truth"] >= 30
                band_pass = not hard_guard or recall_drop <= 0.05
                passed &= band_pass
                entry["distance_bands"][name] = {
                    "ground_truth": baseline_band["ground_truth"],
                    "max_recall_drop": recall_drop,
                    "hard_guard": hard_guard,
                    "passes": bool(band_pass),
                }
        classes[class_name] = entry
    return {
        "passes": bool(passed),
        "map_moderate_drop_points": (
            dense["3d"]["map_moderate_percent"]
            - accuracy["3d"]["map_moderate_percent"]
        ),
        "classes": classes,
    }


def static_points(grouped):
    points = []
    for rate in RATES:
        results = [item[1] for item in grouped[float(rate)]]
        point = {
            "rate": float(rate),
            "accuracy": results[0]["accuracy"],
            "latency": median_latency(results),
            "mean_raw_points": results[0]["sampling"]["mean_raw_points"],
            "mean_selected_points": results[0]["sampling"]["mean_selected_points"],
            "results": [str(item[0].resolve()) for item in grouped[float(rate)]],
        }
        points.append(point)
    dense = next(point for point in points if point["rate"] == 1.0)
    for point in points:
        guard = accuracy_guard(point["accuracy"], dense["accuracy"])
        preprocess = point["latency"]["preprocess"]["mean_ms"]
        end_to_end = point["latency"]["input_to_detections"]["mean_ms"]
        preprocess_gain = 1.0 - preprocess / dense["latency"]["preprocess"]["mean_ms"]
        end_to_end_gain = 1.0 - end_to_end / dense["latency"]["input_to_detections"]["mean_ms"]
        point["gate"] = {
            **guard,
            "preprocess_improvement": preprocess_gain,
            "input_to_detections_improvement": end_to_end_gain,
            "passes": bool(
                point["rate"] < 1.0 and guard["passes"]
                and preprocess_gain >= 0.35 and end_to_end_gain >= 0.15
            ),
        }
    passing = [point["rate"] for point in points if point["gate"]["passes"]]
    return points, {
        "passes": bool(passing),
        "passing_rates": passing,
        "reason": (f"passing non-dense rates: {passing}" if passing
                   else "no non-dense rate met all accuracy and latency thresholds"),
    }


def oracle_analysis(args):
    import evaluate_kitti_bev as evaluator

    frame_ids = read_ids(args.controller_dev)
    grouped, predictions, signature = validate_results(
        args.results, frame_ids, args.config, args.controller_dev
    )
    config = read_json(args.config)
    geometry = config["data"]["kitti"]["geometry"]
    labels = {
        frame_id: evaluator.load_ground_truth(frame_id, args.kitti_root, geometry)
        for frame_id in frame_ids
    }
    qualities = np.asarray([
        [evaluator.frame_quality(predictions[float(rate)][frame_id], labels[frame_id])
         for rate in RATES]
        for frame_id in frame_ids
    ], dtype=np.float32)
    betas = np.linspace(0.0, 1.0, 101)
    all_actions = np.asarray([choose_actions(qualities, RATES, beta) for beta in betas])
    points, by_actions = [], {}
    for beta, actions in zip(betas, all_actions):
        key = actions.tobytes()
        if key in by_actions:
            points[by_actions[key]]["betas"].append(float(beta))
            continue
        selected = {
            frame_id: predictions[float(RATES[action])][frame_id]
            for frame_id, action in zip(frame_ids, actions)
        }
        point = {
            "id": len(points),
            "betas": [float(beta)],
            "mean_rate": float(np.mean(RATES[actions])),
            "action_counts": {
                str(float(rate)): int(np.count_nonzero(actions == index))
                for index, rate in enumerate(RATES)
            },
            "accuracy": {
                "bev": evaluator.evaluate_accuracy(selected, labels, "bev", True),
                "3d": evaluator.evaluate_accuracy(selected, labels, "3d", True),
            },
        }
        by_actions[key] = len(points)
        points.append(point)

    static, static_gate = static_points(grouped)
    dense = next(point for point in static if point["rate"] == 1.0)
    eligible = [point for point in static if point["gate"]["passes"]]
    best_static = max(
        eligible,
        key=lambda point: (point["accuracy"]["3d"]["map_moderate_percent"], -point["rate"]),
        default=None,
    )
    comparisons = []
    winning = []
    for point in points:
        point["safety"] = accuracy_guard(point["accuracy"], dense["accuracy"])
        oracle_map = point["accuracy"]["3d"]["map_moderate_percent"]
        for fixed in static:
            fixed_map = fixed["accuracy"]["3d"]["map_moderate_percent"]
            comparisons.append({
                "oracle_id": point["id"], "static_rate": fixed["rate"],
                "map_gain_points": oracle_map - fixed_map,
                "rate_reduction": fixed["rate"] - point["mean_rate"],
            })
        if best_static is None or not point["safety"]["passes"]:
            continue
        map_gain = oracle_map - best_static["accuracy"]["3d"]["map_moderate_percent"]
        rate_reduction = best_static["rate"] - point["mean_rate"]
        matched_rate = abs(point["mean_rate"] - best_static["rate"]) <= 0.025
        if (matched_rate and map_gain >= 0.3) or (rate_reduction >= 0.10 and map_gain >= -0.2):
            winning.append((point, map_gain, rate_reduction))

    output = Path(args.output)
    actions_path = output.with_suffix(".actions.npz")
    actions_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        actions_path,
        frame_ids=np.asarray(frame_ids), qualities=qualities, rates=RATES,
        betas=betas, actions=all_actions,
    )
    if winning:
        winner, map_gain, rate_reduction = max(
            winning, key=lambda item: (item[1], item[2])
        )
        reason = (
            f"oracle {winner['id']} beats best static rate {best_static['rate']:.2f}: "
            f"mAP margin {map_gain:+.3f} points, rate reduction {rate_reduction:+.3f}"
        )
    elif best_static is None:
        reason = "oracle gate not evaluated because no static rate passed S0"
    else:
        best_map = max(
            (point["accuracy"]["3d"]["map_moderate_percent"]
             - best_static["accuracy"]["3d"]["map_moderate_percent"])
            for point in points if point["safety"]["passes"]
        ) if any(point["safety"]["passes"] for point in points) else float("-inf")
        best_rate = max(
            (best_static["rate"] - point["mean_rate"]
             for point in points if point["safety"]["passes"]),
            default=float("-inf"),
        )
        reason = (
            f"no safe oracle met the margin; best mAP margin {best_map:+.3f} points, "
            f"best rate reduction {best_rate:+.3f}"
        )
    report = {
        "static_points": static,
        "static_gate": static_gate,
        "oracle_points": points,
        "oracle_gate": {
            "passes": bool(winning),
            "best_static_rate": best_static["rate"] if best_static else None,
            "reason": reason,
            "comparisons": comparisons,
        },
        "provenance": {
            "config": str(Path(args.config).resolve()),
            "config_sha256": sha256(args.config),
            "controller_dev": str(Path(args.controller_dev).resolve()),
            "controller_dev_sha256": sha256(args.controller_dev),
            "checkpoint_sha256": signature[2],
            "sampling_seed": signature[3],
            "results": [str(Path(path).resolve()) for path in args.results],
            "actions": str(actions_path.resolve()),
            "actions_sha256": sha256(actions_path),
        },
    }
    write_json(output, report)


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)
    baseline = commands.add_parser("baseline")
    baseline.add_argument("--source", type=Path, required=True)
    baseline.add_argument("--epoch", type=int, required=True)
    baseline.add_argument("--output", type=Path, required=True)
    baseline.set_defaults(handler=snapshot_baseline)
    split = commands.add_parser("split")
    split.add_argument("--source", type=Path, required=True)
    split.add_argument("--policy-train", type=Path, required=True)
    split.add_argument("--controller-dev", type=Path, required=True)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--dev-count", type=int, default=1200)
    split.set_defaults(handler=create_splits)
    oracle = commands.add_parser("oracle")
    oracle.add_argument("--results", type=Path, nargs="+", required=True)
    oracle.add_argument("--config", type=Path, required=True)
    oracle.add_argument("--kitti-root", type=Path, required=True)
    oracle.add_argument("--controller-dev", type=Path, required=True)
    oracle.add_argument("--output", type=Path, required=True)
    oracle.set_defaults(handler=oracle_analysis)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
