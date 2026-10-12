#!/usr/bin/env python3
"""CPU encoding benchmark: separate initialization, warmup and steady state."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "detector"), str(ROOT / "detector/core/datasets")]
from core.bev_encoding import resolve_bev_encoding
from utils_1.preprocess import encode_bev
from utils_1.bev_backend import HIST14_IMPLEMENTATION_VERSION, encode_hist14_chw

ENCODINGS = ("binary_slices", "rich8",
             "hist14_numpy", "hist14_numba")


def benchmark_encodings(points, geometry, *, encodings=ENCODINGS, warmup=3, iterations=20, layout="hwc"):
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be nonnegative and iterations must be positive")
    if layout not in ("hwc", "chw"):
        raise ValueError("layout must be hwc or chw")
    report = {
        "scope": ("CPU rasterization plus contiguous CHW model input; excludes file I/O, augmentation, H2D, detector inference, decode/NMS and AP"
                  if layout == "chw" else
                  "CPU rasterization only; excludes file I/O, augmentation, detector inference and AP"),
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "platform": platform.platform(), "processor": platform.processor() or platform.machine(),
                        "cpu_count": os.cpu_count(), "thread_environment": {
                            key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")}},
        "input": {"point_count": len(points), "dtype": str(points.dtype), "geometry": geometry},
        "results": {},
    }
    if "hist14_numba" in encodings:
        try:
            report["environment"]["numba"] = importlib.metadata.version("numba")
        except importlib.metadata.PackageNotFoundError:
            report["environment"]["numba"] = None
    for label in encodings:
        if label not in ENCODINGS:
            raise ValueError(f"unknown encoding benchmark: {label!r}")
        options = {"name": "hist14", "backend": label.split("_")[1]} if label.startswith("hist14_") else {"name": label}
        schema = resolve_bev_encoding(options, geometry)
        def encode():
            if layout == "chw" and schema.name == "hist14":
                return encode_hist14_chw(points, schema)
            value = encode_bev(points, geometry, options)
            return np.ascontiguousarray(value.transpose(2, 0, 1)) if layout == "chw" else value
        start = time.perf_counter()
        result = encode()
        first_call = (time.perf_counter() - start) * 1000
        shape, payload, contiguous = list(result.shape), result.nbytes, bool(result.flags.c_contiguous)
        del result
        warmup_ms, steady_ms = [], []
        for samples, repeats in ((warmup_ms, warmup), (steady_ms, iterations)):
            for _ in range(repeats):
                start = time.perf_counter()
                result = encode()
                samples.append((time.perf_counter() - start) * 1000)
                del result
        report["results"][label] = {
            "name": schema.name, "version": schema.version, "backend": schema.backend,
            "semantic_hash": schema.semantic_hash, "schema": schema.semantic_metadata(),
            "implementation_version": HIST14_IMPLEMENTATION_VERSION if schema.name == "hist14" else "legacy",
            "shape_yxc": list(schema.output_shape), "result_shape": shape,
            "output_layout": layout, "output_contiguous": contiguous,
            "output_dtype": "float32", "payload_bytes": payload,
            "first_call_ms": first_call,
            "first_call_scope": "First rasterization plus dependency import/JIT compilation or disk-cache loading for Numba; not isolated compiler time",
            "warmup_ms": warmup_ms, "steady_state_ms": steady_ms,
            "mean_ms": float(np.mean(steady_ms)), "p50_ms": float(np.percentile(steady_ms, 50)),
            "p95_ms": float(np.percentile(steady_ms, 95)),
        }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true")
    source.add_argument("--pointcloud", type=Path, help="Raw float32 KITTI [x,y,z,intensity] file")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/config.json")
    parser.add_argument("--points", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--encodings", nargs="+", choices=ENCODINGS, default=list(ENCODINGS))
    parser.add_argument("--layout", choices=("hwc", "chw"), default="hwc",
                        help="chw includes contiguous model input preparation; hist14 writes CHW directly")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.points < 0 or args.warmup < 0 or args.iterations < 1:
        parser.error("points/warmup must be nonnegative and iterations must be positive")
    config_bytes = args.config.read_bytes()
    geometry = json.loads(config_bytes)["data"]["kitti"]["geometry"]
    if args.synthetic:
        rng = np.random.default_rng(args.seed)
        lower = [geometry[f"{axis}_min"] for axis in "xyz"] + [0.]
        upper = [geometry[f"{axis}_max"] for axis in "xyz"] + [1.]
        points = rng.uniform(lower, upper, (args.points, 4)).astype(np.float32)
    else:
        raw = np.fromfile(args.pointcloud, dtype=np.float32)
        if len(raw) % 4:
            parser.error("pointcloud length must be divisible by four float32 features")
        points = raw.reshape(-1, 4)
    report = benchmark_encodings(points, geometry, encodings=args.encodings,
                                 warmup=args.warmup, iterations=args.iterations, layout=args.layout)
    report["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    report["input"].update(source="synthetic" if args.synthetic else str(args.pointcloud), seed=args.seed if args.synthetic else None)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
