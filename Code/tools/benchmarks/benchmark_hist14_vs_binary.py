#!/usr/bin/env python3
"""Matched, interleaved CPU Hist14/Binary comparison on real KITTI clouds.

Both encoders use their existing production implementations. Raster-only and
contiguous CHW boundaries are reported separately. No GPU or PyTorch is required.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "detector"), str(ROOT / "detector/core/datasets")]
from core.bev_encoding import resolve_bev_encoding
from utils_1 import bev_backend
from utils_1.preprocess import encode_bev


def compare(paths, geometry, *, warmup=5, iterations=30, seed=42, baseline_backend=None):
    """Read each cloud once, warm methods, then randomize every timed iteration."""
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be nonnegative and iterations must be positive")
    paths = [Path(path).resolve() for path in paths]
    if not paths:
        raise ValueError("at least one pointcloud is required")
    # Validate files before resolving geometry so malformed payloads are actionable.
    for path in paths:
        if path.stat().st_size % 16:
            raise ValueError(f"{path}: KITTI pointcloud must contain groups of four float32 values")
    schema = resolve_bev_encoding({"name": "hist14", "backend": "numba"}, geometry)
    original = None
    if baseline_backend is not None:
        baseline_backend = Path(baseline_backend).resolve()
        module_name = "hist14_comparison_baseline_" + hashlib.sha256(
            str(baseline_backend).encode()).hexdigest()[:12]
        spec = importlib.util.spec_from_file_location(module_name, baseline_backend)
        original = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = original  # Numba's cache imports its module.
        spec.loader.exec_module(original)

    import numba

    cpu_model = platform.processor() or platform.machine()
    cpu_info = Path("/proc/cpuinfo")
    if cpu_info.exists():
        for line in cpu_info.read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    source_paths = [Path(__file__), Path(bev_backend.__file__),
                    ROOT / "detector/core/datasets/utils_1/preprocess.py",
                    ROOT / "detector/core/bev_encoding.py"]
    if original is not None:
        source_paths.append(baseline_backend)
    report = {
        "scope": {"device": "CPU", "file_io": False, "augmentation": False,
                  "h2d": False, "model": False, "decode": False, "ap": False,
                  "raster": "native rasterizer output; Binary may be noncontiguous",
                  "chw": "rasterization plus independently owned contiguous float32 CHW model input",
                  "caveat": "local CPU benchmark; does not update archived GPU E2E speed or AP"},
        "protocol": {"order": "randomized_interleaving_per_frame", "seed": seed,
                     "warmup": warmup, "iterations_per_frame": iterations,
                     "first_call": "includes dependency/JIT/cache initialization where applicable",
                     "aggregation": "equal-weight mean across frame means"},
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "numba": numba.__version__, "platform": platform.platform(),
                        "cpu_model": cpu_model, "logical_cpus": os.cpu_count(),
                        "numba_threads": numba.get_num_threads(),
                        "kernel_parallel": False,
                        "thread_environment": {key: os.environ.get(key) for key in
                            ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}},
        "hist14_implementation_version": bev_backend.HIST14_IMPLEMENTATION_VERSION,
        "hist14_semantic_hash": schema.semantic_hash,
        "geometry": geometry,
        "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_paths},
        "frames": [],
    }
    rng = random.Random(seed)
    for path in paths:
        raw = path.read_bytes()
        points = np.frombuffer(raw, dtype=np.float32).reshape(-1, 4).copy()
        methods = {
            "binary_slices_raster": lambda: encode_bev(points, geometry),
            "binary_slices_chw": lambda: np.ascontiguousarray(
                encode_bev(points, geometry).transpose(2, 0, 1)),
            "hist14_numba_raster": lambda: bev_backend.encode_hist14_numba(points, schema),
            "hist14_numba_chw": lambda: bev_backend.encode_hist14_chw(points, schema),
        }
        if original is not None:
            methods["hist14_before_chw"] = lambda: original.encode_hist14_chw(points, schema)
        frame = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                 "point_count": len(points), "dtype": str(points.dtype),
                 "iteration_order": [], "results": {}}
        for name, encode in methods.items():
            start = time.perf_counter_ns()
            value = encode()
            first_ms = (time.perf_counter_ns() - start) / 1e6
            frame["results"][name] = {
                "first_call_ms": first_ms, "shape": list(value.shape),
                "payload_bytes": value.nbytes, "dtype": str(value.dtype),
                "contiguous": bool(value.flags.c_contiguous), "steady_ms": [],
            }
            del value
        for _ in range(warmup):
            names = list(methods)
            rng.shuffle(names)
            for name in names:
                value = methods[name]()
                del value
        for _ in range(iterations):
            names = list(methods)
            rng.shuffle(names)
            frame["iteration_order"].append(names)
            for name in names:
                start = time.perf_counter_ns()
                value = methods[name]()
                elapsed = (time.perf_counter_ns() - start) / 1e6
                frame["results"][name]["steady_ms"].append(elapsed)
                del value
        # Audit outside the timing window, one pair at a time to bound memory.
        actual = methods["hist14_numba_chw"]()
        reference = methods["hist14_numba_raster"]().transpose(2, 0, 1)
        frame["hist14_layout_bit_mismatches"] = int(np.count_nonzero(
            actual.view(np.uint32) != reference.view(np.uint32)))
        if frame["hist14_layout_bit_mismatches"]:
            raise AssertionError(f"{path}: Hist14 HWC/CHW output mismatch")
        del reference
        if original is not None:
            reference = methods["hist14_before_chw"]()
            frame["hist14_before_bit_mismatches"] = int(np.count_nonzero(
                actual.view(np.uint32) != reference.view(np.uint32)))
            if frame["hist14_before_bit_mismatches"]:
                raise AssertionError(f"{path}: Hist14 output differs from baseline")
            del reference
        del actual
        for values in frame["results"].values():
            samples = values["steady_ms"]
            values.update(mean_ms=float(np.mean(samples)),
                          p50_ms=float(np.percentile(samples, 50)),
                          p95_ms=float(np.percentile(samples, 95)))
        report["frames"].append(frame)
    report["mean_ms"] = {
        name: float(np.mean([frame["results"][name]["mean_ms"]
                             for frame in report["frames"]]))
        for name in report["frames"][0]["results"]}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointcloud", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/config.json")
    parser.add_argument("--baseline-backend", type=Path,
                        help="optional trusted original bev_backend.py snapshot for bitwise comparison")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    config_bytes = args.config.read_bytes()
    report = compare(args.pointcloud, json.loads(config_bytes)["data"]["kitti"]["geometry"],
                     warmup=args.warmup, iterations=args.iterations, seed=args.seed,
                     baseline_backend=args.baseline_backend)
    report["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "frames": len(report["frames"]),
                      "mean_ms": report["mean_ms"]}, indent=2))
    return report


if __name__ == "__main__":
    main()
