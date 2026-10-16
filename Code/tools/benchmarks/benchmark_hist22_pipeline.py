#!/usr/bin/env python3
"""Reproducible Hist14/Hist22 CPU benchmark; never reports detector AP.

Use --mode encoding for paired preloaded real-cloud timings, --mode pipeline for
actual configured model stages and training/restore smoke, or --mode all.
Outputs are exclusive-create. Full pipeline artifacts live beside the JSON.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "detector"), str(ROOT / "detector/core/datasets"),
               str(ROOT / "tools/kitti_training_pipeline")]

from core.bev_encoding import resolve_bev_encoding
from utils_1.bev_backend import encode_hist14_chw
from utils_1.hist22_backend import encode_hist22_chw

SHOWCASE = ("000804", "004425", "004567", "001801")
THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")
SOURCE_PATHS = (
    "tools/benchmarks/benchmark_hist22_pipeline.py", "tests/test_hist22_pipeline_benchmark.py",
    "tools/benchmarks/profile_detector.py", "detector/core/bev_encoding.py",
    "detector/core/datasets/utils_1/bev_backend.py", "detector/core/datasets/utils_1/hist22_backend.py",
    "detector/core/datasets/dataset.py", "detector/postprocess.py",
    "tools/kitti_training_pipeline/common.py", "tools/kitti_training_pipeline/paired_initialization.py",
    "tools/kitti_training_pipeline/train.py", "tools/kitti_training_pipeline/prepare_kitti.py",
    "tools/kitti_training_pipeline/hist22_ablation.py",
    "detector/core/datasets/utils_1/preprocess.py", "detector/core/datasets/utils_1/target_backend.py",
)
SOURCE_PATHS += tuple(str(path.relative_to(ROOT)) for directory in
                     (ROOT / "detector/core/models", ROOT / "detector/core/losses")
                     for path in sorted(directory.rglob("*.py")))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rss():
    """Linux process RSS and native peak, including native/JIT allocator scratch."""
    status = Path("/proc/self/status").read_text()
    current = next(int(line.split()[1]) for line in status.splitlines() if line.startswith("VmRSS:"))
    return {"current_mib": current / 1024,
            "process_lifetime_peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}


def summarize(samples):
    return {"samples_ms": samples, "p50_ms": float(np.percentile(samples, 50)),
            "p95_ms": float(np.percentile(samples, 95)), "mean_ms": float(np.mean(samples))}


def timed(function):
    start = time.perf_counter_ns()
    value = function()
    return value, (time.perf_counter_ns() - start) / 1e6


def select_frames(train_path, val_path, *, per_split=8, seed=42):
    if per_split < 1:
        raise ValueError("per_split must be positive")
    result = [{"id": name, "selection": "showcase"} for name in SHOWCASE]
    seen = set(SHOWCASE)
    rng = random.Random(seed)
    for name, path in (("train", train_path), ("val", val_path)):
        ids = sorted({line.split(";")[0].strip() for line in Path(path).read_text().splitlines() if line.strip()})
        candidates = [frame for frame in ids if frame not in seen]
        if len(candidates) < per_split:
            raise ValueError(f"not enough {name} frames for requested sample")
        for frame in rng.sample(candidates, per_split):
            seen.add(frame)
            result.append({"id": frame, "selection": f"{name}_random"})
    return result


def run_encoding_benchmark(frames, geometry, *, warmup, iterations, seed, reference_count=4):
    if warmup < 0 or iterations < 1 or reference_count < 0:
        raise ValueError("invalid warmup/iterations/reference_count")
    if not frames:
        raise ValueError("at least one preloaded frame is required")
    schemas = {name: resolve_bev_encoding({"name": name, "backend": "numba"}, geometry)
               for name in ("hist14", "hist22")}
    encoders = {"hist14": encode_hist14_chw, "hist22": encode_hist22_chw}
    first, warmed, layouts = {}, {}, {}
    # Separate true first invocation from all subsequent frame warmups. This
    # includes dependency/JIT/cache startup; it is not isolated compiler time.
    for name in schemas:
        value, elapsed = timed(lambda: encoders[name](frames[0]["points"], schemas[name]))
        first[name] = elapsed
        if value.dtype != np.float32 or not value.flags.c_contiguous or not np.isfinite(value).all():
            raise AssertionError("encoder must produce finite contiguous float32 CHW")
        layouts[name] = {"shape": list(value.shape), "payload_bytes": value.nbytes,
                         "contiguous": bool(value.flags.c_contiguous), "dtype": str(value.dtype)}
        del value
    for frame in frames:
        for name in schemas:
            samples = []
            for _ in range(warmup):
                value, elapsed = timed(lambda: encoders[name](frame["points"], schemas[name]))
                samples.append(elapsed)
                del value
            warmed[(frame["id"], name)] = samples
    results = {frame["id"]: {name: [] for name in schemas} for frame in frames}
    rng, order = random.Random(seed), []
    for iteration in range(iterations):
        work = frames.copy()
        rng.shuffle(work)
        row_order = []
        for frame in work:
            pair = list(schemas)
            rng.shuffle(pair)
            for name in pair:
                value, elapsed = timed(lambda: encoders[name](frame["points"], schemas[name]))
                results[frame["id"]][name].append(elapsed)
                del value  # Deallocation excluded, allocation/zeroing included.
            row_order.append({"id": frame["id"], "order": pair})
        order.append(row_order)
    rows = []
    ref_schema = resolve_bev_encoding({"name": "hist22", "backend": "numpy"}, geometry)
    for index, frame in enumerate(frames):
        row = {key: value for key, value in frame.items() if key != "points"}
        row.update(point_count=len(frame["points"]), results={})
        for name, schema in schemas.items():
            row["results"][name] = dict(summarize(results[frame["id"]][name]),
                warmup_ms=warmed[(frame["id"], name)], **layouts[name], semantic_hash=schema.semantic_hash)
        a, b = (row["results"][name] for name in schemas)
        row["comparison"] = {"p50_ratio": b["p50_ms"] / a["p50_ms"],
                             "p50_overhead_ms": b["p50_ms"] - a["p50_ms"],
                             "p95_ratio": b["p95_ms"] / a["p95_ms"]}
        if index < reference_count:
            reference, reference_ms = timed(lambda: encode_hist22_chw(frame["points"], ref_schema))
            actual = encode_hist22_chw(frame["points"], schemas["hist22"])
            np.testing.assert_allclose(actual[:14], reference[:14], atol=1e-6, rtol=1e-5)
            np.testing.assert_allclose(actual[14:], reference[14:], atol=2e-6, rtol=1e-5)
            row["reference_parity"] = {"passed": True, "numpy_single_ms": reference_ms,
                "max_abs_error": float(np.max(np.abs(actual - reference))),
                "base_atol": 1e-6, "geometry_atol": 2e-6, "rtol": 1e-5}
            del actual, reference
        rows.append(row)
    aggregate = {name: summarize([sample for frame in frames for sample in results[frame["id"]][name]])
                 for name in schemas}
    a, b = (aggregate[name] for name in schemas)
    return {"scope": "Preloaded float32 clouds to independently owned contiguous CHW; no I/O/targets/model. Allocation included; deallocation excluded.",
            "warmup": warmup, "iterations_per_frame": iterations, "layout": "CHW",
            "first_call_ms": first,
            "first_call_scope": "First encoding invocation including dependency import/JIT compilation or disk cache loading; cache state uncontrolled, not isolated compiler time",
            "frames": rows, "aggregate": aggregate, "execution_order": order,
            "comparison": {"pooled_p50_ratio": b["p50_ms"] / a["p50_ms"],
                           "pooled_p50_overhead_ms": b["p50_ms"] - a["p50_ms"],
                           "pooled_p95_ratio": b["p95_ms"] / a["p95_ms"]},
            "rss": rss(), "rss_scope": "Whole encoding-process lifetime peak, including imports, all preloaded clouds, both encoders, JIT/cache and NumPy reference scratch; not per-encoder attribution"}


def tensor_leaves(value, prefix=""):
    if isinstance(value, dict):
        for key in sorted(value):
            yield from tensor_leaves(value[key], prefix + "/" + key)
    elif value is not None:
        yield prefix, value


def output_parity(left, right, *, exact=False):
    import torch
    a, b = dict(tensor_leaves(left)), dict(tensor_leaves(right))
    if a.keys() != b.keys():
        raise AssertionError("prediction keys differ")
    error = 0.
    for key in a:
        torch.testing.assert_close(a[key], b[key], atol=0 if exact else 1e-6,
                                   rtol=0 if exact else 1e-5)
        error = max(error, (a[key] - b[key]).abs().max().item())
    return {"passed": True, "max_abs_error": error, "atol": 0 if exact else 1e-6,
            "rtol": 0 if exact else 1e-5}


def run_pipeline_benchmark(configs, cloud_root, label_root, calib_root, frame_ids, artifact_root,
                           *, warmup, iterations, training_iterations, seed, training_frame_id="000000"):
    """CPU batch1 verification using the real Dataset/criterion/train checkpoint API.

    The dataset task is validation to bypass train-only GT database creation.
    No augmentation is applied, and throughput measures an already prepared
    batch separately from serial I/O/encoding/targets. It is not loader throughput.
    """
    import torch
    from core.datasets.dataset import Dataset, collate_detector_batch
    from detector.postprocess import filter_pred
    from tools.benchmarks.profile_detector import profile_detector
    from tools.kitti_training_pipeline.common import detection_spec
    from tools.kitti_training_pipeline.paired_initialization import build_initialized_model
    from tools.kitti_training_pipeline.prepare_kitti import convert_labels
    from tools.kitti_training_pipeline import train

    if warmup < 0 or min(iterations, training_iterations) < 1:
        raise ValueError("invalid pipeline iteration counts")
    if set(configs) != {"hist14", "hist22"}:
        raise ValueError("pipeline requires exactly hist14 and hist22 configs")
    if training_frame_id in frame_ids:
        raise ValueError("smoke train/validation manifests must be disjoint")
    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=False)
    processed = artifact_root / "processed"
    (processed / "pointcloud").mkdir(parents=True)
    (processed / "label").mkdir()
    sources, label_counts = {}, {}
    for frame_id in dict.fromkeys([*frame_ids, training_frame_id]):
        paths = {"cloud": Path(cloud_root) / f"{frame_id}.bin",
                 "label": Path(label_root) / f"{frame_id}.txt",
                 "calibration": Path(calib_root) / f"{frame_id}.txt"}
        converted, counts = convert_labels(paths["label"], paths["calibration"])
        (processed / "pointcloud" / f"{frame_id}.bin").symlink_to(paths["cloud"].resolve())
        (processed / "label" / f"{frame_id}.txt").write_text("\n".join(converted) + "\n")
        sources[frame_id] = {key: {"path": str(path.resolve()), "sha256": sha256(path)}
                             for key, path in paths.items()}
        label_counts[frame_id] = dict(counts)
    manifest = artifact_root / "val.txt"
    manifest.write_text("\n".join(frame_ids) + "\n")
    train_manifest = artifact_root / "train.txt"
    train_manifest.write_text(training_frame_id + "\n")
    report = {"device": "cpu", "precision": "fp32", "batch_size": 1,
        "scope": "Serial CPU stages and compute-step throughput on fixed validation-task samples; no train augmentation/GT DB, DataLoader workers/prefetch or AP",
        "frame_ids": frame_ids, "training_frame_id": training_frame_id,
        "sources": sources, "converted_label_counts": label_counts,
        "models": {}, "timing_schedule": [], "warmup": warmup, "iterations": iterations,
        "transfer_scope": "CPU tensor .to(cpu) traversal; no host-to-GPU transfer measured",
        "total_scope": "Totals sum separately timed stage calls; encode_to_detections excludes targets/I/O, serial_io_to_detections includes both",
        "io_scope": "Read cloud from OS page cache and copy constructor-cached converted labels; label/calibration conversion excluded"}
    live = {}
    for name, original in configs.items():
        config = copy.deepcopy(original)
        before = copy.deepcopy(config)
        config["device"] = "cpu"
        config["data"]["kitti"]["location"] = str(processed.resolve())
        config["train"].update(precision="fp32", physical_batch_size=1, num_workers=0,
            epochs=1, warmup_epochs=0, checkpoint_selection={"primary": "loss"})
        config["val"].update(physical_batch_size=1)
        config["train"]["data"] = str(train_manifest.resolve())
        config["val"]["data"] = str(manifest.resolve())
        train.seed_everything(seed)
        spec = detection_spec(config)
        dataset = Dataset(str(manifest), config["data"], config["augmentation"], spec.cls_encoding,
                          task="validation", target_backend=config["train"].get("target_backend", "numba"),
                          task_groups=spec.groups if spec.head_mode == "grouped" else None)
        first_batch, dataset_first_ms = timed(lambda: collate_detector_batch([dataset[0]]))
        training_dataset = Dataset(str(train_manifest), config["data"], config["augmentation"], spec.cls_encoding,
                          task="validation", target_backend=config["train"].get("target_backend", "numba"),
                          task_groups=spec.groups if spec.head_mode == "grouped" else None)
        training_batch, training_prepare_ms = timed(lambda: collate_detector_batch([training_dataset[0]]))
        model, provenance = build_initialized_model(config)
        config["initialization_provenance"] = provenance
        model.eval()
        with torch.no_grad():
            first_output, model_first_ms = timed(lambda: model(first_batch["voxel"]))
        stages = {stage: [] for stage in ("io", "encoding", "targets_collate", "transfer",
                                           "forward", "decode_nms", "encode_to_detections", "serial_io_to_detections")}
        live[name] = dict(config=config, dataset=dataset, model=model, first=first_output,
                          batch=first_batch, training_batch=training_batch, spec=spec, stages=stages)
        report["models"][name] = {"smoke_overrides": config_diff(before, config, exclude={"initialization_provenance"}),
            "dataset_task": "validation", "initialization_provenance": provenance,
            "dataset_first_ms": dataset_first_ms, "training_batch_prepare_ms": training_prepare_ms,
            "model_first_forward_ms": model_first_ms,
            "input_shape": list(first_batch["voxel"].shape), "input_payload_bytes": first_batch["voxel"].numel() * 4,
            "counts": profile_detector(original, device="meta"), "detections": []}
        (artifact_root / f"{name}.smoke.config.json").write_text(json.dumps(config, indent=2) + "\n")
    report["initial_output_parity"] = output_parity(live["hist14"]["first"], live["hist22"]["first"])
    # Both members see the same frame index per iteration; randomize model order.
    rng = random.Random(seed)
    for iteration in range(warmup + iterations):
        order = list(live)
        rng.shuffle(order)
        report["timing_schedule"].append(dict(warmup=iteration < warmup, order=order,
                                               frame_id=frame_ids[iteration % len(frame_ids)]))
        for name in order:
            state = live[name]
            dataset, model, config, spec = (state[key] for key in ("dataset", "model", "config", "spec"))
            index = iteration % len(frame_ids)
            def read_sample():
                points = dataset.read_points(str(processed / "pointcloud" / f"{frame_ids[index]}.bin"))
                return points, dataset.filter_boxes(dataset.get_boxes(index), "kitti")
            (points, boxes), io_ms = timed(read_sample)
            geometry = config["data"]["kitti"]["geometry"]
            voxel, encoding_ms = timed(lambda: dataset.encode_input(points, geometry))
            def labels_collate():
                labels = dataset.get_label(boxes, geometry)
                labels["voxel"] = voxel
                return collate_detector_batch([labels])
            batch, targets_ms = timed(labels_collate)
            batch, transfer_ms = timed(lambda: train.move_tensor_batch(batch, torch.device("cpu")))
            with torch.no_grad():
                output, forward_ms = timed(lambda: model(batch["voxel"]))
                detections, decode_ms = timed(lambda: filter_pred(output, config["data"]["kitti"],
                    config["data"]["out_size_factor"], config["evaluation"]["score_threshold"],
                    config["evaluation"]["nms_threshold"], task_groups=spec.groups if spec.head_mode == "grouped" else None,
                    cls_encoding=spec.cls_encoding, use_iou=config["model"].get("header_use_iou"),
                    max_detections=config["evaluation"].get("max_detections"), box_mode=spec.box_mode))
            if iteration >= warmup:
                values = dict(io=io_ms, encoding=encoding_ms, targets_collate=targets_ms,
                    transfer=transfer_ms, forward=forward_ms, decode_nms=decode_ms,
                    encode_to_detections=encoding_ms + transfer_ms + forward_ms + decode_ms,
                    serial_io_to_detections=io_ms + encoding_ms + targets_ms + transfer_ms + forward_ms + decode_ms)
                for key, value in values.items():
                    state["stages"][key].append(value)
                report["models"][name]["detections"].append(len(detections))
            state["batch"] = batch
    # Training starts from matched T0; eval does not update BN. Train the same
    # explicit training-split sample for both members; no loader throughput claim.
    for name in live:
        state = live[name]
        config, model, batch = (state[key] for key in ("config", "model", "training_batch"))
        criterion = train.build_training_criterion(config, torch.device("cpu"))
        optimizer = train.build_optimizer(model, criterion, config)
        scheduler = train.build_scheduler(optimizer, config, 1)
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        generator = torch.Generator().manual_seed(seed)
        train.seed_everything(seed)
        model.train()
        criterion.train()
        train.set_loss_epoch(criterion, 0)
        stem = dict(model.named_parameters())[config["initialization_provenance"]["input_weight_key"]]
        before = stem.detach().clone()
        losses, compute, forward, loss_times, backward, steps = [], [], [], [], [], []
        grad_nonzero, grad_finite = [], []
        for step_index in range(training_iterations):
            start = time.perf_counter_ns()
            optimizer.zero_grad(set_to_none=True)
            output, elapsed = timed(lambda: model(batch["voxel"]))
            forward.append(elapsed)
            loss, elapsed = timed(lambda: criterion(output, batch))
            loss_times.append(elapsed)
            if not torch.isfinite(loss["loss"]):
                raise AssertionError("nonfinite training loss")
            _, elapsed = timed(lambda: loss["loss"].backward())
            backward.append(elapsed)
            if name == "hist22":
                gradients = stem.grad[:, 14:]
                grad_finite = torch.isfinite(gradients).flatten(2).all(2).all(0).tolist()
                grad_nonzero = (gradients.abs().sum((0, 2, 3)) > 0).tolist()
                if not all(grad_finite) or not all(grad_nonzero):
                    raise AssertionError("an extra input slice has invalid/zero gradient")
            def update():
                clip = config["train"].get("grad_clip_norm", 0.)
                if clip:
                    torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(criterion.parameters()), clip,
                                                   error_if_nonfinite=True)
                scaler.step(optimizer)
                scaler.update()
            _, elapsed = timed(update)
            steps.append(elapsed)
            compute.append((time.perf_counter_ns() - start) / 1e6)
            losses.append(loss["loss"].detach().item())
            del output, loss
        scheduler.step()
        changed = ((stem.detach()[:, 14:] - before[:, 14:]).abs().sum((0, 2, 3)) > 0).tolist() if name == "hist22" else []
        if name == "hist22" and not all(changed):
            raise AssertionError("an extra input slice did not update")
        positive = sum(int(tensor.count_nonzero()) for key, tensor in tensor_leaves(batch)
                       if key.endswith("/reg_mask"))
        report["models"][name]["stages"] = {key: summarize(value) for key, value in state["stages"].items()}
        report["models"][name]["training"] = dict(
            scope="Prepared batch CPU FP32 forward + criterion + backward + clip + optimizer; excludes I/O/encoding/targets. First step retained, no training warmup.",
            iterations=training_iterations, losses=losses, compute_step=summarize(compute),
            forward=summarize(forward), criterion=summarize(loss_times), backward=summarize(backward),
            optimizer_clip_step=summarize(steps), samples_per_second=1000 / float(np.mean(compute)),
            positive_target_cells=positive, extra_slice_gradient_finite=grad_finite,
            extra_slice_gradient_nonzero=grad_nonzero, extra_slice_updated=changed)
        model.eval()
        with torch.no_grad():
            expected = model(batch["voxel"])
        optimizer.zero_grad(set_to_none=True)
        payload = train.checkpoint_payload(model, criterion, optimizer, scheduler, scaler, 1,
                                          {"loss": losses[-1]}, losses[-1], config, loader_generator=generator)
        checkpoint = artifact_root / f"{name}.smoke.checkpoint.pth"
        torch.save(payload, checkpoint)
        # Reconstruct every stateful object, read the saved file, and use the
        # normal strict resume API. No permissive weight-only restoration.
        restored_model, _ = build_initialized_model(config)
        restored_criterion = train.build_training_criterion(config, torch.device("cpu"))
        restored_optimizer = train.build_optimizer(restored_model, restored_criterion, config)
        restored_scheduler = train.build_scheduler(restored_optimizer, config, 1)
        restored_scaler = torch.amp.GradScaler("cpu", enabled=False)
        restored = train.restore_checkpoint(torch.load(checkpoint, map_location="cpu", weights_only=False),
            config, restored_model, restored_criterion, restored_optimizer, restored_scheduler,
            restored_scaler, loader_generator=torch.Generator().manual_seed(seed))
        restored_model.eval()
        with torch.no_grad():
            identity = output_parity(expected, restored_model(batch["voxel"]), exact=True)
        report["models"][name]["checkpoint"] = dict(path=str(checkpoint.resolve()),
            sha256=sha256(checkpoint), bytes=checkpoint.stat().st_size, restore_result=restored,
            strict_output_identity=identity["passed"], max_abs_error=identity["max_abs_error"])
        report["models"][name]["rss_after_training_restore"] = rss()
        del restored_model, restored_criterion, restored_optimizer, payload, expected, criterion, optimizer
        gc.collect()
    report["rss"] = rss()
    report["rss_scope"] = "Whole process lifetime peak including two paired models, source/prepared inputs, metadata profiler, JIT, autograd, optimizer and checkpoint restore copies; not isolated training peak or per-model attribution"
    report["decode_limit"] = "Paired random weights; empty/few detections do not measure trained-model NMS workload"
    return report


def config_diff(left, right, prefix="", exclude=frozenset()):
    rows = []
    for key in sorted(set(left) | set(right)):
        if key in exclude:
            continue
        path = f"{prefix}.{key}" if prefix else key
        a, b = left.get(key), right.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            rows.extend(config_diff(a, b, path))
        elif a != b:
            rows.append({"path": path, "original": a, "smoke": b})
    return rows


def environment():
    cpu = Path("/proc/cpuinfo").read_text()
    libraries = {}
    for name in ("numpy", "numba", "torch", "scipy"):
        try:
            libraries[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            libraries[name] = None
    return {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
            "cpu_model": next(line.split(":", 1)[1].strip() for line in cpu.splitlines() if line.startswith("model name")),
            "logical_cpu_count": os.cpu_count(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "libraries": libraries, "threads": {key: os.environ.get(key) for key in THREAD_KEYS},
            "sources_sha256": {path: sha256(ROOT / path) for path in SOURCE_PATHS}}


def load_verified_pair(manifest_path, seed):
    """Reject config drift before touching benchmark outputs or input datasets."""
    from tools.kitti_training_pipeline.hist22_ablation import ALLOWED_DIFFERENCES, leaf_differences, _canonical
    from tools.kitti_training_pipeline.paired_initialization import paired_policy
    manifest = json.loads(Path(manifest_path).read_text())
    runs = manifest.get("runs", [])
    if len(runs) != 2 or {run.get("encoding") for run in runs} != {"hist14", "hist22"}:
        raise ValueError("manifest must contain exactly two unique Hist14/Hist22 pair roles")
    if {run.get("label") for run in runs} != {"A", "B"} or len({run.get("run_name") for run in runs}) != 2:
        raise ValueError("manifest pair labels and run names must be unique")
    configs = {}
    for run in runs:
        name = run["encoding"]
        if run["label"] != ("A" if name == "hist14" else "B"):
            raise ValueError("manifest pair labels must map A=Hist14 and B=Hist22")
        config = json.loads(Path(run["config_path"]).read_text())
        digest = hashlib.sha256(_canonical(config)).hexdigest()
        if digest != run.get("config_sha256"):
            raise ValueError(f"manifest config digest mismatch: {name}")
        if run["run_name"] != config["experiment"]["name"] + "_" + digest[:20]:
            raise ValueError(f"manifest run name differs from canonical config identity: {name}")
        schema = resolve_bev_encoding(config["data"]["bev_encoding"], config["data"]["kitti"]["geometry"])
        if schema.semantic_hash != run.get("schema_sha256"):
            raise ValueError(f"manifest schema hash mismatch: {name}")
        if schema.name != name or schema.backend != "numba" or tuple(schema.output_shape[:2]) != (800, 704):
            raise ValueError("CLI requires the actual full-grid 800x704 Numba pair")
        policy = paired_policy(config)
        if policy is None or policy["seed"] != seed:
            raise ValueError("benchmark pair requires strict paired initialization matching seed")
        configs[name] = config
    if leaf_differences(configs["hist14"], configs["hist22"]) != sorted(ALLOWED_DIFFERENCES):
        raise ValueError("paired configs differ outside the exact three allowed encoding/experiment leaves")
    return manifest, configs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("all", "encoding", "pipeline"), default="all")
    parser.add_argument("--pair-manifest", type=Path)
    parser.add_argument("--cloud-root", type=Path, default=ROOT / "Kitti_Raw/velodyne/training/velodyne")
    parser.add_argument("--label-root", type=Path, default=ROOT / "Kitti_Raw/label_2/training/label_2")
    parser.add_argument("--calib-root", type=Path, default=ROOT / "Kitti_Raw/calib/training/calib")
    parser.add_argument("--train-split", type=Path, default=ROOT / "splits/kitti/train.txt")
    parser.add_argument("--val-split", type=Path, default=ROOT / "splits/kitti/val.txt")
    parser.add_argument("--per-split", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--pipeline-iterations", type=int, default=6)
    parser.add_argument("--training-iterations", type=int, default=2)
    parser.add_argument("--reference-count", type=int, default=4)
    parser.add_argument("--smoke-frames", nargs="+", default=["004425", "000804"])
    parser.add_argument("--training-frame", default="000000")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.warmup < 0 or min(args.iterations, args.pipeline_iterations, args.training_iterations, args.per_split) < 1 or args.reference_count < 0:
        parser.error("warmup/reference-count must be nonnegative; all iteration/sample counts must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.pair_manifest:
        parser.error("--pair-manifest is required")
    manifest, configs = load_verified_pair(args.pair_manifest, args.seed)
    report = {"environment": environment(), "argv": sys.argv if argv is None else argv,
              "pair_manifest": str(args.pair_manifest.resolve()), "pair_manifest_sha256": sha256(args.pair_manifest),
              "config_files": {run["encoding"]: {"path": run["config_path"], "sha256": sha256(run["config_path"])} for run in manifest["runs"]},
              "seed": args.seed, "detection_ap": "pending: no full detector training/evaluation performed"}
    if args.mode in ("encoding", "all"):
        frames = select_frames(args.train_split, args.val_split, per_split=args.per_split, seed=args.seed)
        for frame in frames:
            path = args.cloud_root / f"{frame['id']}.bin"
            frame.update(path=str(path.resolve()), source_sha256=sha256(path),
                         points=np.fromfile(path, dtype=np.float32).reshape(-1, 4))
        report["split_sources"] = {str(path.resolve()): sha256(path) for path in (args.train_split, args.val_split)}
        report["encoding"] = run_encoding_benchmark(frames, configs["hist14"]["data"]["kitti"]["geometry"],
            warmup=args.warmup, iterations=args.iterations, seed=args.seed, reference_count=args.reference_count)
        del frames
        gc.collect()
    if args.mode in ("pipeline", "all"):
        split_ids = lambda path: {line.split(";")[0].strip() for line in path.read_text().splitlines() if line.strip()}
        train_ids, val_ids = split_ids(args.train_split), split_ids(args.val_split)
        if train_ids & val_ids or args.training_frame not in train_ids or not set(args.smoke_frames) <= val_ids:
            raise ValueError("pipeline requires disjoint original splits, a training-split training frame and validation-split inference frames")
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        report["environment"].update(torch_num_threads=torch.get_num_threads(),
            torch_num_interop_threads=torch.get_num_interop_threads(), cuda_available=torch.cuda.is_available())
        report["pipeline"] = run_pipeline_benchmark(configs, args.cloud_root, args.label_root, args.calib_root,
            args.smoke_frames, args.output.with_suffix(".artifacts"), warmup=args.warmup,
            iterations=args.pipeline_iterations, training_iterations=args.training_iterations, seed=args.seed,
            training_frame_id=args.training_frame)
    report["final_source_sha256"] = {path: sha256(ROOT / path) for path in SOURCE_PATHS}
    if report["environment"]["sources_sha256"] != report["final_source_sha256"]:
        raise RuntimeError("source files changed during benchmark; discard mixed-source measurements")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "mode": args.mode,
        "encoding_comparison": report.get("encoding", {}).get("comparison"),
        "initial_output_parity": report.get("pipeline", {}).get("initial_output_parity"),
        "rss": rss()}, indent=2))
    return report


if __name__ == "__main__":
    main()
