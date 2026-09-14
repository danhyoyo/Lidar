#!/usr/bin/env python3
"""Detection-level ProbGeo-UQ metrics for 15-column Center3D prediction NPZ files."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from common import read_json, write_json
from evaluate_kitti_bev import (CLASS_IDS, IOU_THRESHOLDS, box_iou,
                                load_ground_truth, prediction_box, read_ids)


def fit_variance_scale(errors, log_vars, log_var_min=-7.0, log_var_max=4.0):
    """Analytic per-dimension variance multiplier fit on calibration only."""
    errors, log_vars = np.asarray(errors, dtype=np.float64), np.asarray(log_vars, dtype=np.float64)
    if errors.ndim != 2 or errors.shape != log_vars.shape or not len(errors) or not np.isfinite(errors).all() or not np.isfinite(log_vars).all():
        raise ValueError("variance fitting requires non-empty finite equal-shaped arrays")
    return np.maximum(np.mean(errors * errors / np.exp(np.clip(log_vars, log_var_min, log_var_max)), axis=0), 1e-12)


def _rank(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_values)) + 1]
    ends = np.r_[starts[1:], len(values)]
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + end - 1) / 2.0
    return ranks


def _spearman(first, second):
    if len(first) < 2 or np.std(first) == 0 or np.std(second) == 0:
        return None
    return float(np.corrcoef(_rank(first), _rank(second))[0, 1])


def _aurc(errors, uncertainty):
    risk = np.mean(np.abs(errors), axis=1)[np.argsort(uncertainty, kind="mergesort")]
    return float(np.mean(np.cumsum(risk) / np.arange(1, len(risk) + 1)))


def binary_auroc(scores, labels):
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int8)
    positives, negatives = int(labels.sum()), int(len(labels) - labels.sum())
    if not positives or not negatives:
        return None
    return float((_rank(scores)[labels.astype(bool)].sum() - positives * (positives - 1) / 2) / (positives * negatives))


def evaluate_arrays(errors, log_vars, log_var_min=-7.0, log_var_max=4.0):
    """Metric core usable in tests and after matching; errors/log_vars are N x 6."""
    errors, log_vars = np.asarray(errors, dtype=np.float64), np.asarray(log_vars, dtype=np.float64)
    if errors.ndim != 2 or errors.shape[1] != 6 or errors.shape != log_vars.shape:
        raise ValueError("errors and log_vars must both be N x 6")
    if not len(errors) or not np.isfinite(errors).all() or not np.isfinite(log_vars).all():
        raise ValueError("metrics require non-empty finite matched samples")
    used_log_vars = np.clip(log_vars, log_var_min, log_var_max)
    variance = np.exp(used_log_vars)
    nll = 0.5 * (errors * errors / variance + used_log_vars)
    std = np.sqrt(variance)
    coverage1 = np.mean(np.abs(errors) <= std, axis=0)
    coverage196 = np.mean(np.abs(errors) <= 1.96 * std, axis=0)
    ence = []
    for dimension in range(6):
        order = np.argsort(std[:, dimension], kind="mergesort")
        bins = np.array_split(order, 10)
        values = []
        for indices in bins:
            if len(indices) >= 30:
                rmse, rmsu = np.sqrt(np.mean(errors[indices, dimension] ** 2)), np.sqrt(np.mean(variance[indices, dimension]))
                values.append(abs(rmse - rmsu) / max(rmsu, 1e-12))
        ence.append(float(np.mean(values)) if values else None)
    uncertainty = np.mean(std, axis=1)
    return {
        "samples": int(len(errors)), "nll": float(np.mean(nll)), "nll_per_dim": np.mean(nll, axis=0).tolist(),
        "coverage_1sigma_per_dim": coverage1.tolist(), "coverage_1sigma_gap_per_dim": np.abs(coverage1 - 0.6827).tolist(),
        "coverage_196sigma_per_dim": coverage196.tolist(), "coverage_196sigma_gap_per_dim": np.abs(coverage196 - 0.95).tolist(),
        "ence_per_dim": ence, "sharpness_mean_std_per_dim": np.mean(std, axis=0).tolist(),
        "sharpness_median_std_per_dim": np.median(std, axis=0).tolist(),
        "spearman_std_abs_error_per_dim": [_spearman(std[:, d], np.abs(errors[:, d])) for d in range(6)],
        "aurc": _aurc(errors, uncertainty),
        "saturation_rate_min_per_dim": np.mean(log_vars <= log_var_min, axis=0).tolist(),
        "saturation_rate_max_per_dim": np.mean(log_vars >= log_var_max, axis=0).tolist(),
    }


def fit_baselines(errors, ranges, point_counts=None):
    """Fit calibration-only constant and range+point-count log-variance baselines."""
    errors, ranges = np.asarray(errors, dtype=np.float64), np.asarray(ranges, dtype=np.float64)
    constant = np.log(np.maximum(np.mean(errors ** 2, axis=0), 1e-12))
    result = {"constant_log_var": constant}
    if point_counts is not None and np.isfinite(point_counts).all():
        design = np.column_stack((np.ones(len(errors)), np.log1p(ranges), np.log1p(point_counts)))
        result["range_point_count_coefficients"] = np.linalg.lstsq(
            design, np.log(np.maximum(errors ** 2, 1e-12)), rcond=None
        )[0]
    return result


def count_points_in_box(points, box):
    """Count LiDAR points inside an upright, yaw-rotated GT box."""
    points = np.asarray(points, dtype=np.float32).reshape(-1, 4)
    dx, dy = points[:, 0] - box.x, points[:, 1] - box.y
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    local_length, local_width = c * dx + s * dy, -s * dx + c * dy
    inside = ((np.abs(local_length) <= box.length / 2) &
              (np.abs(local_width) <= box.width / 2) &
              (np.abs(points[:, 2] - box.z_center) <= box.height / 2))
    return int(np.count_nonzero(inside))


def load_frame_points(frame_id, kitti_root):
    path = Path(kitti_root) / "training" / "velodyne" / f"{frame_id}.bin"
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def load_prediction_npz(path, frame_ids):
    with np.load(path) as data:
        missing = sorted(set(frame_ids).difference(data.files))
        if missing:
            raise ValueError(f"prediction NPZ is missing split IDs: {missing[:5]}")
        return {frame_id: data[frame_id] for frame_id in frame_ids}


def conditional_metrics(errors, log_vars, metadata, log_var_min, log_var_max):
    result = {}
    buckets = {
        "class": (lambda value: value[0]),
        "range": (lambda value: "0_30m" if value[1] < 30 else "30_50m" if value[1] < 50 else "50_70.4m" if value[1] <= 70.4 else None),
        "point_count": (lambda value: "0_5" if value[2] <= 5 else "6_20" if value[2] <= 20 else ">20"),
    }
    for name, select in buckets.items():
        groups = {}
        for index, value in enumerate(metadata):
            group = select(value)
            if group is not None:
                groups.setdefault(group, []).append(index)
        result[name] = {
            group: {"samples": len(indices), "metrics": evaluate_arrays(errors[indices], log_vars[indices], log_var_min, log_var_max) if len(indices) >= 100 else None}
            for group, indices in groups.items()
        }
    return result


def matched_residuals(predictions, labels, kitti_root=None, log_var_min=-7.0, log_var_max=4.0):
    """Greedy score-ranked Moderate 3-D TPs, matching the local AP protocol."""
    errors, log_vars, scores, failures, ranges, matched_points, metadata = [], [], [], [], [], [], []
    id_to_class = {value: key for key, value in CLASS_IDS.items()}
    for frame_id, rows in predictions.items():
        frame_points = load_frame_points(frame_id, kitti_root) if kitti_root is not None else None
        rows = np.asarray(rows, dtype=np.float64).reshape(-1, rows.shape[-1])
        for class_id, class_name in id_to_class.items():
            valid = [box for box in labels[frame_id] if box.name == class_name and box.passes("Moderate")]
            used = set()
            for row in sorted((row for row in rows if int(row[0]) == class_id), key=lambda row: -row[1]):
                box = prediction_box(row, "3d")
                candidates = [(box_iou(box, target, "3d"), index) for index, target in enumerate(valid) if index not in used]
                best_iou, index = max(candidates, default=(0.0, -1))
                failed = best_iou < IOU_THRESHOLDS[class_name]
                if len(row) >= 15:
                    uncertainty = np.mean(np.exp(0.5 * np.clip(
                        row[9:15], log_var_min, log_var_max
                    )))
                    failures.append((float(uncertainty), float(failed)))
                if failed or len(row) < 15:
                    continue
                used.add(index)
                target = valid[index]
                errors.append([box.x - target.x, box.y - target.y, box.z_center - target.z_center,
                               math.log(box.width) - math.log(target.width), math.log(box.length) - math.log(target.length), math.log(box.height) - math.log(target.height)])
                log_vars.append(row[9:15])
                scores.append(row[1])
                ranges.append(math.hypot(target.x, target.y))
                points = np.nan if frame_points is None else count_points_in_box(frame_points, target)
                matched_points.append(points)
                metadata.append((class_name, ranges[-1], points))
    return (np.asarray(errors), np.asarray(log_vars), np.asarray(scores), np.asarray(failures),
            np.asarray(ranges), np.asarray(matched_points), metadata)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--kitti-root", required=True, type=Path)
    parser.add_argument("--split", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--calibration-predictions", type=Path)
    parser.add_argument("--calibration-split", type=Path)
    args = parser.parse_args(argv)
    config, ids = read_json(args.config), read_ids(args.split)
    labels = {frame_id: load_ground_truth(frame_id, args.kitti_root, config["data"]["kitti"]["geometry"]) for frame_id in ids}
    predictions = load_prediction_npz(args.predictions, ids)
    errors, log_vars, scores, failures, ranges, point_counts, metadata = matched_residuals(predictions, labels, args.kitti_root)
    log_var_min, log_var_max = config.get("loss", {}).get("log_var_min", -7.0), config.get("loss", {}).get("log_var_max", 4.0)
    result = {"raw": evaluate_arrays(errors, log_vars, log_var_min, log_var_max), "matched_samples": int(len(errors)),
              "failure_detection_samples": int(len(failures)),
              "conditional": conditional_metrics(errors, log_vars, metadata, log_var_min, log_var_max)}
    if len(failures) > 1 and len(np.unique(failures[:, 1])) == 2:
        result["failure_detection_auroc"] = binary_auroc(failures[:, 0], failures[:, 1])
        result["failure_detection_aurc"] = _aurc(failures[:, 1:2], failures[:, 0])
    if args.calibration_predictions:
        if not args.calibration_split:
            raise ValueError("--calibration-split is required with --calibration-predictions")
        calibration_ids = read_ids(args.calibration_split)
        if set(calibration_ids).intersection(ids):
            raise ValueError("calibration and test splits must be disjoint")
        calibration = load_prediction_npz(args.calibration_predictions, calibration_ids)
        calibration_labels = {frame_id: load_ground_truth(frame_id, args.kitti_root, config["data"]["kitti"]["geometry"])
                              for frame_id in calibration_ids}
        calibration_errors, calibration_vars, _, _, calibration_ranges, calibration_points, _ = matched_residuals(calibration, calibration_labels, args.kitti_root, log_var_min, log_var_max)
        alpha = fit_variance_scale(calibration_errors, calibration_vars, log_var_min, log_var_max)
        result["calibrator"] = {"variance_scale": alpha.tolist()}
        result["calibrated"] = evaluate_arrays(errors, log_vars + np.log(alpha), log_var_min, log_var_max)
        baselines = fit_baselines(calibration_errors, calibration_ranges, calibration_points)
        result["baselines"] = {"constant": evaluate_arrays(errors, np.broadcast_to(baselines["constant_log_var"], log_vars.shape), log_var_min, log_var_max)}
        if "range_point_count_coefficients" in baselines and np.isfinite(point_counts).all():
            design = np.column_stack((np.ones(len(errors)), np.log1p(ranges), np.log1p(point_counts)))
            result["baselines"]["range_point_count"] = evaluate_arrays(errors, design @ baselines["range_point_count_coefficients"], log_var_min, log_var_max)
        else:
            result["baselines"]["range_point_count"] = {"status": "unavailable: current prediction NPZ schema has no point counts"}
        result["ranking_controls"] = {
            "score_only_aurc": _aurc(errors, -scores),
            "shuffled_learned_aurc": _aurc(errors, np.mean(np.sqrt(np.exp(np.clip(log_vars[np.random.default_rng(42).permutation(len(log_vars))], log_var_min, log_var_max))), axis=1)),
            "oracle_abs_error_aurc": _aurc(errors, np.mean(np.abs(errors), axis=1)),
        }
    write_json(args.output, result)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
