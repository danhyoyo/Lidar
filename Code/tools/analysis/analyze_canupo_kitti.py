#!/usr/bin/env python3
"""Run the author's CANUPO binaries on KITTI with box-derived weak labels.

This is a point-feature feasibility study, not a detector AP benchmark.
All fitting and operating-threshold selection use the training split only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "detector")]
from core.datasets.utils_1.preprocess import encode_bev
from tools.kitti_training_pipeline.prepare_kitti import parse_calibration
from tools.visualization.export_kitti_overview import dataset_paths, read_points

CLASSES = ("Outside boxes", "Car", "Pedestrian", "Cyclist")
DIAMETERS = np.array([4., 2., 1., .5], dtype=np.float32)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_split(path):
    return {line.strip().split(";", 1)[0] for line in Path(path).read_text().splitlines() if line.strip()}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def msc_dtype(n_scales, point_fields):
    return np.dtype([("point", "<f4", point_fields),
                     ("ab", "<f4", (n_scales, 2)), ("count", "<i4", n_scales)])


def read_msc(path):
    """Read the original float32 CANUPO format; preserve unordered OpenMP rows."""
    with Path(path).open("rb") as stream:
        n, ns = struct.unpack("<ii", stream.read(8))
        if n < 0 or not 0 < ns < 1000:
            raise ValueError("Invalid MSC header")
        scales = np.frombuffer(stream.read(4 * ns), dtype="<f4").copy()
        nf, = struct.unpack("<i", stream.read(4))
        if nf < 3 or nf > 100:
            raise ValueError("Invalid MSC point fields")
        rows = np.frombuffer(stream.read(), dtype=msc_dtype(ns, nf)).copy()
    if len(rows) != n:
        raise ValueError(f"MSC row count mismatch: {path}")
    return scales, rows


def write_msc(path, scales, rows):
    ns = len(scales)
    nf = rows.dtype["point"].shape[0]
    with Path(path).open("wb") as stream:
        stream.write(struct.pack("<ii", len(rows), ns))
        stream.write(np.asarray(scales, dtype="<f4").tobytes())
        stream.write(struct.pack("<i", nf))
        stream.write(rows.tobytes())


def align_msc(rows, points):
    """The 4th core field carries an exact integer ID through parallel output."""
    ids = rows["point"][:, 3].astype(np.int64)
    if not np.array_equal(np.sort(ids), np.arange(len(points))):
        raise ValueError("CANUPO core IDs are not a permutation of input rows")
    aligned = rows[np.argsort(ids)]
    # Legacy text parsing can shift float32 coordinates by one ULP at KITTI ranges.
    if not np.allclose(aligned["point"][:, :3], points[:, :3], atol=1e-5, rtol=0):
        raise ValueError("CANUPO output is not aligned with input core coordinates")
    if not np.isfinite(aligned["ab"]).all():
        raise ValueError("Non-finite CANUPO descriptor")
    return aligned


def box_membership(camera_points, fields, padding=0.):
    h, w, length = map(float, fields[8:11])
    center = np.array(list(map(float, fields[11:14])))
    yaw = float(fields[14])
    delta = camera_points - center
    local_x = np.cos(yaw) * delta[:, 0] - np.sin(yaw) * delta[:, 2]
    local_z = np.sin(yaw) * delta[:, 0] + np.cos(yaw) * delta[:, 2]
    return ((np.abs(local_x) <= length / 2 + padding)
            & (np.abs(local_z) <= w / 2 + padding)
            & (delta[:, 1] >= -h - padding) & (delta[:, 1] <= padding))


def weak_labels(points, calib_path, label_path, image_size):
    transform = parse_calibration(calib_path)
    camera = points[:, :3] @ transform[:3, :3].T + transform[:3, 3]
    calibration = {}
    for line in Path(calib_path).read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        calibration[key] = np.fromstring(value, sep=" ")
    projected = np.c_[camera, np.ones(len(camera))] @ calibration["P2"].reshape(3, 4).T
    uv = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-9)
    eligible = ((projected[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < image_size["width"])
                & (uv[:, 1] >= 0) & (uv[:, 1] < image_size["height"]))
    labels = np.zeros(len(points), dtype=np.int8)
    object_ids = np.full(len(points), -1, dtype=np.int32)
    blocked = np.zeros(len(points), bool)
    hits = np.zeros(len(points), np.int16)
    objects = []
    for line in Path(label_path).read_text().splitlines():
        f = line.split()
        if f[0] == "DontCare":
            x1, y1, x2, y2 = map(float, f[4:8])
            blocked |= ((uv[:, 0] >= x1) & (uv[:, 0] <= x2)
                        & (uv[:, 1] >= y1) & (uv[:, 1] <= y2))
            continue
        if min(map(float, f[8:11])) <= 0:
            continue
        inside = box_membership(camera, f)
        halo = box_membership(camera, f, padding=.15)
        if f[0] not in CLASSES[1:]:
            blocked |= halo
            continue
        # Uncertain box boundaries are not used as negative training examples.
        blocked |= halo & ~inside
        oid = len(objects)
        labels[inside] = CLASSES.index(f[0])
        object_ids[inside] = oid
        hits += inside
        center_lidar = np.linalg.solve(transform, np.array([*map(float, f[11:14]), 1.]))
        objects.append({"id": oid, "class": f[0], "distance": float(np.linalg.norm(center_lidar[:2])),
                        "occluded": int(f[2]), "truncated": float(f[1])})
    eligible &= ~blocked & (hits <= 1)
    labels[~eligible] = -1
    return labels, object_ids, objects


def choose_frames(ids, labels_dir, n_random, n_enriched, rng):
    pool = np.array(sorted(ids))
    random_ids = rng.choice(pool, n_random, replace=False).tolist()
    remaining = sorted(set(ids) - set(random_ids))
    enriched = []
    for name in ("Pedestrian", "Cyclist"):
        candidates = [i for i in remaining if i not in enriched and any(
            line.startswith(name + " ") for line in (labels_dir / f"{i}.txt").read_text().splitlines())]
        amount = n_enriched // 2
        if len(candidates) < amount:
            raise ValueError(f"Insufficient {name} frames")
        enriched.extend(rng.choice(candidates, amount, replace=False).tolist())
    return [(i, "random") for i in random_ids] + [(i, "enriched") for i in enriched]


def run_binary(cmd, env, log_path, timeout=180):
    start = time.perf_counter()
    result = subprocess.run([str(x) for x in cmd], capture_output=True, text=True,
                            env=env, timeout=timeout)
    elapsed = time.perf_counter() - start
    Path(log_path).write_text("COMMAND " + repr([str(x) for x in cmd]) + "\n"
                              + result.stdout + "\n" + result.stderr)
    if result.returncode:
        raise RuntimeError(f"CANUPO failed ({result.returncode}); see {log_path}")
    return elapsed


def prepare_frame(frame, selection, partition, args, paths, geometry, image_sizes):
    target = args.output / "frames" / frame
    target.mkdir(parents=True, exist_ok=True)
    cache = target / "features.npz"
    if cache.exists():
        metadata = json.loads((target / "metadata.json").read_text())
        if metadata.get("hist14_sampling") != "float64_floor_v2":
            raise ValueError("Cached Hist14 lookup predates the boundary-index fix; choose a fresh output directory")
        if metadata["partition"] != partition or metadata["selection"] != selection:
            raise ValueError("Cached sampling differs; choose a fresh output directory")
        current_hashes = {k: digest(p) for k, p in {
            "scan": paths.velodyne / f"{frame}.bin", "labels": paths.labels / f"{frame}.txt",
            "calibration": paths.calib / f"{frame}.txt"}.items()}
        if current_hashes != metadata["hashes"]:
            raise ValueError("Cached KITTI inputs changed; choose a fresh output directory")
        return metadata
    points = read_points(paths.velodyne / f"{frame}.bin")
    labels, object_ids, objects = weak_labels(points, paths.calib / f"{frame}.txt",
                                             paths.labels / f"{frame}.txt", image_sizes[frame])
    roi = np.ones(len(points), bool)
    for col, axis in enumerate("xyz"):
        roi &= ((points[:, col] > geometry[f"{axis}_min"] + .001)
                & (points[:, col] < geometry[f"{axis}_max"] - .001))
    eligible = np.flatnonzero(roi & (labels >= 0))
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, int(frame)]))
    uniform = rng.choice(eligible, min(args.cores, len(eligible)), replace=False)
    extras = []
    for obj in objects:
        indices = np.flatnonzero(roi & (labels > 0) & (object_ids == obj["id"]))
        obj["roi_points"] = len(indices)
        extras.extend(rng.choice(indices, min(64, len(indices)), replace=False).tolist())
    # First rows preserve the uniform sample; extra points are diagnostic only at validation.
    extra = np.array(sorted(set(extras) - set(uniform.tolist())), dtype=np.int64)
    indices = np.r_[uniform, extra]
    core = points[indices]
    padding = float(max(DIAMETERS) / 2 + .01)
    source_mask = np.ones(len(points), bool)
    for col, axis in enumerate("xyz"):
        source_mask &= ((points[:, col] > geometry[f"{axis}_min"] - padding)
                        & (points[:, col] < geometry[f"{axis}_max"] + padding))
    source = points[source_mask]
    start = time.perf_counter()
    np.savetxt(target / "source.xyz", source[:, :3], fmt="%.8f")
    np.savetxt(target / "core.xyz", np.c_[core[:, :3], np.arange(len(core))], fmt="%.8f")
    write_seconds = time.perf_counter() - start
    elapsed = run_binary([args.binary_dir / "canupo", *DIAMETERS, ":", target / "source.xyz",
                          target / "core.xyz", target / "core.msc"], args.env, target / "canupo.log")
    scales, rows = read_msc(target / "core.msc")
    np.testing.assert_array_equal(scales, DIAMETERS)
    aligned = align_msc(rows, core)
    start = time.perf_counter()
    bev = encode_bev(points, geometry, {"name": "hist14", "backend": "numpy"})
    hist_seconds = time.perf_counter() - start
    promoted_core = core[:, :3].astype(np.float64)
    ix = np.floor((promoted_core[:, 0] - geometry["x_min"]) / geometry["x_res"]).astype(int)
    iy = np.floor((promoted_core[:, 1] - geometry["y_min"]) / geometry["y_res"]).astype(int)
    hist = bev[iy, ix]
    local = np.c_[hist, core[:, 2:4], np.linalg.norm(core[:, :2], axis=1)].astype(np.float32)
    np.savez_compressed(cache, local=local, geometry=aligned["ab"].reshape(len(core), -1),
                        counts=aligned["count"], xyz=core[:, :3], labels=labels[indices],
                        object_ids=object_ids[indices], uniform=np.arange(len(core)) < len(uniform))
    metadata = {"frame": frame, "partition": partition, "selection": selection,
                "hist14_sampling": "float64_floor_v2",
                "total_points": len(points), "source_points": len(source), "roi_points": int(roi.sum()),
                "eligible_points": len(eligible), "core_points": len(core), "uniform_points": len(uniform),
                "objects": objects, "canupo_seconds": elapsed, "ascii_write_seconds": write_seconds,
                "hist14_numpy_seconds": hist_seconds,
                "hashes": {k: digest(p) for k, p in {
                    "scan": paths.velodyne / f"{frame}.bin", "labels": paths.labels / f"{frame}.txt",
                    "calibration": paths.calib / f"{frame}.txt"}.items()}}
    write_json(target / "metadata.json", metadata)
    return metadata


def load_partition(output, metadata, partition, random_only=False):
    result = {}
    for m in metadata:
        if m["partition"] != partition or (random_only and m["selection"] != "random"):
            continue
        with np.load(output / "frames" / m["frame"] / "features.npz") as data:
            for key in data.files:
                result.setdefault(key, []).append(data[key])
            result.setdefault("frame", []).append(np.full(len(data["labels"]), m["frame"]))
    return {key: np.concatenate(value) for key, value in result.items()}


def feature_sets(data):
    geom = data["geometry"]
    support = np.c_[np.log1p(data["counts"]), (data["counts"] >= 6).astype(float)]
    local = data["local"]
    result = {"hist14_local": local, "canupo_geometry": geom,
              "canupo_geometry_support": np.c_[geom, support],
              "hist14_plus_support": np.c_[local, support],
              "hist14_plus_canupo": np.c_[local, support, geom]}
    for j, diameter in enumerate(DIAMETERS):
        result[f"single_d{diameter:g}"] = geom[:, j * 2:j * 2 + 2]
    return result


def operating_threshold(labels, scores, recall=.95):
    from sklearn.metrics import precision_recall_curve
    precision, recalls, thresholds = precision_recall_curve(labels, scores)
    valid = np.flatnonzero(recalls[:-1] >= recall)
    return float(thresholds[valid[-1]])


def binary_metrics(labels, scores, threshold):
    from sklearn.metrics import average_precision_score, roc_auc_score
    labels = np.asarray(labels, bool)
    kept = scores >= threshold
    return {"n": len(labels), "positive": int(labels.sum()),
            "prevalence": float(labels.mean()),
            "auroc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
            "threshold": threshold, "recall": float(kept[labels].mean()),
            "outside_box_rejection": float((~kept[~labels]).mean()),
            "precision": float(labels[kept].mean()) if kept.any() else 0.}


def retention(data, scores, threshold):
    rows = []
    distance = np.linalg.norm(data["xyz"][:, :2], axis=1)
    for class_id, name in enumerate(CLASSES[1:], 1):
        for lo, hi in [(0, 20), (20, 40), (40, 80), (0, 80)]:
            mask = ((data["labels"] == class_id) & (distance >= lo) & (distance < hi))
            if mask.any():
                rows.append({"class": name, "distance": f"{lo}-{hi}", "points": int(mask.sum()),
                             "retained": float((scores[mask] >= threshold).mean())})
    return rows


def object_retention(data, scores, threshold):
    rows = []
    for frame in np.unique(data["frame"]):
        in_frame = data["frame"] == frame
        for oid in np.unique(data["object_ids"][in_frame & (data["labels"] > 0)]):
            mask = in_frame & (data["object_ids"] == oid) & (data["labels"] > 0)
            rows.append({"frame": str(frame), "object_id": int(oid),
                         "class": CLASSES[int(data["labels"][mask][0])],
                         "sampled_points": int(mask.sum()),
                         "retained": float((scores[mask] >= threshold).mean())})
    return rows


def original_classifier(args, metadata, train, validation):
    """Fit the original LDA -> SVG -> PRM -> classify path, without SVG editing."""
    rng = np.random.default_rng(args.seed)
    all_rows = []
    all_labels = []
    for m in metadata:
        if m["partition"] != "fit":
            continue
        path = args.output / "frames" / m["frame"]
        scales, rows = read_msc(path / "core.msc")
        data = np.load(path / "features.npz")
        core = data["xyz"]
        all_rows.append(align_msc(rows, core))
        all_labels.append(data["labels"] > 0)
    all_rows = np.concatenate(all_rows)
    all_labels = np.concatenate(all_labels)
    for label, name in [(False, "outside"), (True, "foreground")]:
        indices = np.flatnonzero(all_labels == label)
        chosen = rng.choice(indices, min(8000, len(indices)), replace=False)
        write_msc(args.output / f"train_{name}.msc", DIAMETERS, all_rows[chosen])
    run_binary([args.binary_dir / "suggest_classifier_lda", args.output / "classifier.svg", ":",
                args.output / "train_outside.msc", "-", args.output / "train_foreground.msc"],
               args.env, args.output / "train_original.log")
    run_binary([args.binary_dir / "validate_classifier", args.output / "classifier.svg",
                args.output / "classifier.prm", "1", "2"],
               args.env, args.output / "validate_original.log")
    scores = {}
    for partition, data in [("tune", train), ("validation", validation)]:
        score_list = []
        for m in metadata:
            if m["partition"] != partition:
                continue
            path = args.output / "frames" / m["frame"]
            run_binary([args.binary_dir / "classify", args.output / "classifier.prm", path / "core.xyz",
                        path / "core.msc", path / "classified.xyz"],
                       args.env, path / "classify.log")
            classified = np.loadtxt(path / "classified.xyz")
            # Original output: xyz, class, confidence, Nmin, Nmax, original 4th scalar.
            if classified.shape[1] != 8:
                raise ValueError(f"Unexpected original classify output shape: {classified.shape}")
            core = np.load(path / "features.npz")["xyz"]
            if not np.allclose(classified[:, :3], core, atol=1e-4, rtol=1e-5):
                raise ValueError("Original classify changed point order")
            np.testing.assert_array_equal(classified[:, 7], np.arange(len(core)))
            class_id, confidence = classified[:, 3], classified[:, 4]
            if not np.isin(class_id, [1, 2]).all():
                raise ValueError("Original classifier returned an unknown class")
            score_list.append(np.where(class_id == 2, confidence, 1 - confidence))
        scores[partition] = np.concatenate(score_list)
        if len(scores[partition]) != len(data["labels"]):
            raise ValueError("Original classifier partition alignment failed")
    return scores


def evaluate(args, metadata):
    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    fit = load_partition(args.output, metadata, "fit")
    tune = load_partition(args.output, metadata, "tune")
    val = load_partition(args.output, metadata, "validation")
    fit_features, tune_features, val_features = map(feature_sets, (fit, tune, val))
    yfit = fit["labels"] > 0
    tune_uniform = tune["uniform"]
    # Primary metrics use only uniformly sampled cores from randomly selected validation frames.
    random_frames = {m["frame"] for m in metadata if m["partition"] == "validation" and m["selection"] == "random"}
    primary = val["uniform"] & np.isin(val["frame"], list(random_frames))
    metrics, saved_scores = {}, {}
    for name, features in fit_features.items():
        model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, learning_rate=.08,
                                               l2_regularization=1., early_stopping=False,
                                               random_state=args.seed)
        start = time.perf_counter()
        model.fit(features, yfit, sample_weight=compute_sample_weight("balanced", yfit))
        tune_scores = model.predict_proba(tune_features[name])[:, 1]
        scores = model.predict_proba(val_features[name])[:, 1]
        threshold = operating_threshold(tune["labels"][tune_uniform] > 0, tune_scores[tune_uniform])
        metrics[name] = binary_metrics(val["labels"][primary] > 0, scores[primary], threshold)
        metrics[name]["features"] = features.shape[1]
        metrics[name]["fit_predict_seconds"] = time.perf_counter() - start
        metrics[name]["retention_all_validation_cores"] = retention(val, scores, threshold)
        metrics[name]["object_retention_diagnostics"] = object_retention(val, scores, threshold)
        saved_scores[name] = scores
        print("PROBE", name, json.dumps({k: v for k, v in metrics[name].items()
                                         if k not in {"retention_all_validation_cores", "object_retention_diagnostics"}}), flush=True)
    original_scores = original_classifier(args, metadata, tune, val)
    threshold = operating_threshold(tune["labels"][tune_uniform] > 0, original_scores["tune"][tune_uniform])
    scores = original_scores["validation"]
    metrics["original_canupo_lda"] = binary_metrics(val["labels"][primary] > 0, scores[primary], threshold)
    metrics["original_canupo_lda"]["retention_all_validation_cores"] = retention(val, scores, threshold)
    metrics["original_canupo_lda"]["object_retention_diagnostics"] = object_retention(val, scores, threshold)
    saved_scores["original_canupo_lda"] = scores
    support = []
    distances = np.linalg.norm(val["xyz"][:, :2], axis=1)
    for j, diameter in enumerate(DIAMETERS):
        for class_id, name in enumerate(CLASSES):
            for lo, hi in [(0, 20), (20, 40), (40, 80), (0, 80)]:
                mask = ((val["labels"] == class_id) & (distances >= lo) & (distances < hi))
                if mask.any():
                    counts = val["counts"][mask, j]
                    support.append({"diameter": float(diameter), "class": name, "distance": f"{lo}-{hi}",
                                    "points": int(mask.sum()), "at_least_3": float((counts >= 3).mean()),
                                    "at_least_6": float((counts >= 6).mean()), "median_neighbors": float(np.median(counts))})
    # Frame bootstrap expresses sample uncertainty, not variation across training seeds.
    from sklearn.metrics import average_precision_score
    bootstrap, bootstrap_support = [], []
    rng = np.random.default_rng(args.seed + 1)
    frames = sorted(random_frames)
    groups = [np.flatnonzero(primary & (val["frame"] == frame)) for frame in frames]
    for _ in range(200):
        indices = np.concatenate([groups[j] for j in rng.integers(len(groups), size=len(groups))])
        label = val["labels"][indices] > 0
        combined_ap = average_precision_score(label, saved_scores["hist14_plus_canupo"][indices])
        bootstrap.append(combined_ap - average_precision_score(label, saved_scores["hist14_local"][indices]))
        bootstrap_support.append(combined_ap - average_precision_score(label, saved_scores["hist14_plus_support"][indices]))
    summary = {"metrics": metrics, "support": support,
               "primary_validation": {"frames": len(random_frames), "cores": int(primary.sum())},
               "delta_point_ap_bootstrap": {"replicates": 200, "seed": args.seed + 1,
                                            "percentile_95": np.quantile(bootstrap, [.025, .975]).tolist(),
                                            "versus_support_percentile_95": np.quantile(bootstrap_support, [.025, .975]).tolist()},
               "sklearn_version": sklearn.__version__}
    np.savez_compressed(args.output / "validation_predictions.npz", **saved_scores,
                        labels=val["labels"], primary=primary, xyz=val["xyz"], frame=val["frame"],
                        object_ids=val["object_ids"], uniform=val["uniform"], counts=val["counts"],
                        geometry=val["geometry"])
    write_json(args.output / "metrics.json", summary)
    make_plots(args.output, summary, val, saved_scores, primary, metadata)
    return summary


def benchmark_runtime(args, metadata, paths, geometry):
    """Three real frames, all ROI cores, repeated legacy CLI wall-clock timings."""
    candidates = sorted([m for m in metadata if m["partition"] == "validation" and m["selection"] == "random"],
                        key=lambda m: m["roi_points"])
    selected = [candidates[int(q * (len(candidates) - 1))] for q in (.25, .5, .75)]
    results = []
    for m in selected:
        folder = args.output / "runtime" / m["frame"]
        folder.mkdir(parents=True, exist_ok=True)
        points = read_points(paths.velodyne / f"{m['frame']}.bin")
        roi = np.ones(len(points), bool)
        for col, axis in enumerate("xyz"):
            roi &= ((points[:, col] > geometry[f"{axis}_min"] + .001)
                    & (points[:, col] < geometry[f"{axis}_max"] - .001))
        core = points[roi]
        start = time.perf_counter()
        np.savetxt(folder / "all_roi_core.xyz", np.c_[core[:, :3], np.arange(len(core))], fmt="%.8f")
        write_seconds = time.perf_counter() - start
        source = args.output / "frames" / m["frame"] / "source.xyz"
        canupo_times, classify_times, hist_times = [], [], []
        for repeat in range(3):
            canupo_times.append(run_binary([args.binary_dir / "canupo", *DIAMETERS, ":", source,
                                            folder / "all_roi_core.xyz", folder / "all_roi_core.msc"],
                                           args.env, folder / f"descriptor_{repeat}.log"))
            classify_times.append(run_binary([args.binary_dir / "classify", args.output / "classifier.prm",
                                              folder / "all_roi_core.xyz", folder / "all_roi_core.msc",
                                              folder / "classified.xyz"], args.env, folder / f"classify_{repeat}.log"))
            start = time.perf_counter()
            encode_bev(points, geometry, {"name": "hist14", "backend": "numpy"})
            hist_times.append(time.perf_counter() - start)
        results.append({"frame": m["frame"], "roi_cores": len(core), "source_points": m["source_points"],
                        "canupo_descriptor_seconds": canupo_times, "canupo_classify_seconds": classify_times,
                        "hist14_numpy_seconds": hist_times, "core_ascii_write_seconds": write_seconds})
        print("RUNTIME", m["frame"], "cores", len(core), "descriptor_ms", np.median(canupo_times) * 1000, flush=True)
    summary = {"threads": args.threads, "repeats": 3, "frames": results,
               "includes": "CLI startup + ASCII read + neighborhood/PCA + descriptor write; classify measured separately; source/core ASCII preparation excluded from descriptor time",
               "descriptor_median_ms": float(np.median([x for r in results for x in r["canupo_descriptor_seconds"]]) * 1000),
               "classify_median_ms": float(np.median([x for r in results for x in r["canupo_classify_seconds"]]) * 1000),
               "hist14_numpy_median_ms": float(np.median([x for r in results for x in r["hist14_numpy_seconds"]]) * 1000)}
    write_json(args.output / "runtime.json", summary)


def make_plots(output, summary, val, scores, primary, metadata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    names = ["hist14_local", "canupo_geometry", "hist14_plus_support", "hist14_plus_canupo", "original_canupo_lda"]
    for name in names:
        p, r, _ = precision_recall_curve(val["labels"][primary] > 0, scores[name][primary])
        ap = summary["metrics"][name]["average_precision"]
        axes[0].plot(r, p, label=f"{name} ({ap:.3f})")
    axes[0].set(xlabel="Recall of points inside target boxes", ylabel="Precision", title="Random held-out frames: weak-label point PR")
    axes[0].legend(fontsize=7)
    for name in CLASSES[1:]:
        data = [next(x for x in summary["support"] if x["diameter"] == d and x["class"] == name and x["distance"] == "0-80") for d in DIAMETERS[::-1]]
        axes[1].plot(DIAMETERS[::-1], [x["at_least_6"] for x in data], "o-", label=name)
    axes[1].set(xlabel="CANUPO scale diameter (m)", ylabel="Fraction with at least 6 neighbors", ylim=(0, 1.03), title="Geometry support on validation object cores")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "feature_comparison.png", dpi=180)
    fig.savefig(output / "feature_comparison.svg")
    plt.close(fig)
    candidates = [m for m in metadata if m["partition"] == "validation"]
    showcase = max(candidates, key=lambda m: sum(any(o["class"] == c and o["roi_points"] >= 5 for o in m["objects"]) for c in CLASSES[1:]))
    mask = val["frame"] == showcase["frame"]
    xyz = val["xyz"][mask]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True, sharey=True)
    for ax, values, title, cmap in [
        (axes[0], val["labels"][mask], "Box-derived weak classes", "tab10"),
        (axes[1], scores["original_canupo_lda"][mask], "Original CANUPO foreground score", "viridis"),
        (axes[2], scores["hist14_plus_canupo"][mask], "Hist14 + CANUPO probe score", "viridis")]:
        image = ax.scatter(xyz[:, 1], xyz[:, 0], c=values, s=2, cmap=cmap, vmin=0, vmax=3 if ax is axes[0] else 1)
        ax.set(title=title, xlabel="LiDAR y (m)", ylabel="LiDAR x (m)", aspect="equal")
        fig.colorbar(image, ax=ax, shrink=.7)
    fig.suptitle(f"Real KITTI frame {showcase['frame']} — sampled cores, ground truth boxes used only for labels")
    fig.tight_layout()
    fig.savefig(output / "scene_comparison.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kitti-root", type=Path, default=ROOT / "Kitti_Raw")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path, default=Path("/tmp/canupo_linux_static_64bits"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cores", type=int, default=4096)
    parser.add_argument("--random-fit", type=int, default=48)
    parser.add_argument("--enriched-fit", type=int, default=16)
    parser.add_argument("--tune", type=int, default=16)
    parser.add_argument("--random-val", type=int, default=48)
    parser.add_argument("--enriched-val", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.binary_dir = args.binary_dir.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    # The 2013 static glibc crashes when font scanning dlopens modern gconv modules.
    # A font-free config skips only SVG typography; numerical training is unchanged.
    fontconfig = args.output / "canupo-fontconfig.xml"
    fontconfig.write_text('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd">'
                          f'<fontconfig><cachedir>{args.output / "font-cache"}</cachedir></fontconfig>')
    args.env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS="1",
                    MKL_NUM_THREADS="1", FONTCONFIG_FILE=str(fontconfig))
    paths = dataset_paths(args.kitti_root)
    geometry = json.loads((ROOT / "configs/config.json").read_text())["data"]["kitti"]["geometry"]
    image_sizes = json.loads((args.kitti_root / "kitti_image_sizes.json").read_text())["frames"]
    train_path, val_path = ROOT / "splits/kitti/train.txt", ROOT / "splits/kitti/val.txt"
    train_ids, val_ids = read_split(train_path), read_split(val_path)
    if train_ids & val_ids:
        raise ValueError("Train/validation split overlaps")
    available = {p.stem for p in paths.labels.glob("*.txt") if (paths.velodyne / f"{p.stem}.bin").exists() and (paths.calib / p.name).exists()}
    rng = np.random.default_rng(args.seed)
    fitting = choose_frames(train_ids & available, paths.labels, args.random_fit, args.enriched_fit, rng)
    tune_ids = rng.choice(sorted((train_ids & available) - {i for i, _ in fitting}), args.tune, replace=False).tolist()
    validation = choose_frames(val_ids & available, paths.labels, args.random_val, args.enriched_val, rng)
    selected = [(frame, selection, "fit") for frame, selection in fitting]
    selected += [(frame, "random", "tune") for frame in tune_ids]
    selected += [(frame, selection, "validation") for frame, selection in validation]
    if len({i for i, _, _ in selected}) != len(selected):
        raise ValueError("Selected partitions overlap")
    manifest = {"seed": args.seed, "diameters_m": DIAMETERS.tolist(), "radii_m": (DIAMETERS / 2).tolist(),
                "threads": args.threads, "cores_per_frame": args.cores,
                "source": "Author's original Linux 64-bit CANUPO standalone binaries (2013 package)",
                "source_url": "https://nicolas.brodu.net/en/recherche/canupo/",
                "legacy_compatibility": "Font-free FONTCONFIG_FILE avoids static-glibc/gconv crash during SVG typography; original numerical binaries unchanged.",
                "binary_sha256": {p.name: digest(p) for p in args.binary_dir.iterdir() if p.is_file()},
                "script_sha256": digest(__file__), "python": sys.version,
                "platform": platform.platform(), "cpu": platform.processor(),
                "split": {"train": len(train_ids), "validation": len(val_ids), "available_labeled": len(available),
                          "train_sha256": digest(train_path), "validation_sha256": digest(val_path)},
                "geometry": geometry, "selected": [{"frame": f, "selection": s, "partition": p} for f, s, p in selected],
                "weak_label_definition": "Inside calibrated Car/Pedestrian/Cyclist 3D boxes vs outside annotated boxes; camera FOV only; DontCare, other classes and 0.15m outside-box halos excluded.",
                "command": sys.argv}
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in ("seed", "diameters_m", "threads", "cores_per_frame", "binary_sha256", "split", "geometry", "selected"):
            if previous.get(key) != manifest[key]:
                raise ValueError(f"Cached protocol changed ({key}); choose a fresh output directory")
    write_json(manifest_path, manifest)
    metadata = []
    for j, (frame, selection, partition) in enumerate(selected):
        result = prepare_frame(frame, selection, partition, args, paths, geometry, image_sizes)
        metadata.append(result)
        print(f"FRAME {j + 1}/{len(selected)} {partition} {frame} cores={result['core_points']} canupo={result['canupo_seconds']:.3f}s", flush=True)
    write_json(args.output / "frames.json", metadata)
    if not args.extract_only:
        evaluate(args, metadata)
        benchmark_runtime(args, metadata, paths, geometry)
        print("COMPLETE", args.output / "metrics.json", flush=True)


if __name__ == "__main__":
    main()
