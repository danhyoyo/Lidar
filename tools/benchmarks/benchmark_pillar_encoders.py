#!/usr/bin/env python3
"""Matched batch-one random-weight rich8 / pillar inference timing, without AP."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "detector"),
               str(ROOT / "detector/core/datasets"), str(ROOT / "tools/kitti_training_pipeline")]
from core.bev_encoding import resolve_bev_encoding
from core.datasets.utils_1.pillar_backend import prepare_pillars
from core.datasets.utils_1.preprocess import encode_bev
from common import build_model, model_parameter_report


def summary(values):
    return {"samples": len(values), "mean_ms": float(np.mean(values)),
            "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95))}


def prepare_input(points, geometry, encoding):
    if encoding["name"] == "pillar_rich":
        arrays = prepare_pillars(points, geometry, encoding)
        return {key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                for key, value in arrays.items()}
    return torch.from_numpy(encode_bev(points, geometry, encoding)).permute(2, 0, 1)


def benchmark_models(config, points, *, device="cuda", precision="fp32", warmup=10,
                     iterations=50, seed=42):
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be nonnegative and iterations positive")
    if precision not in ("fp32", "bf16"):
        raise ValueError("precision must be fp32 or bf16")
    if points.ndim != 2 or points.shape[1] != 4:
        raise ValueError("points must be shaped [N,4]")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("benchmark supports cpu or cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run on a GPU runtime or select --device cpu")
    if device.type == "cuda" and precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("selected CUDA device does not support BF16")
    geometry = config["data"]["kitti"]["geometry"]
    encodings = {
        "rich8": {"name": "rich8", "density_norm": 32, "intensity_scale": 1},
        "pillar_rich_max": {"name": "pillar_rich", "out_channels": 32, "backend": "torch",
                            "density_norm": 32, "intensity_scale": 1},
        "pillar_rich_max_mean_eca": {"name": "pillar_rich", "out_channels": 32, "backend": "torch",
                                     "density_norm": 32, "intensity_scale": 1,
                                     "pooling": "max_mean_eca", "eca_kernel_size": 3},
    }
    report = {
        "scope": "Batch-one random-weight eval: CPU preparation + contiguous input/H2D + complete detector forward. Excludes file I/O, decode/NMS, AP, training and ROS.",
        "timing": "Synchronized wall-clock timings include Python dispatch; model_only excludes preparation/transfer. Warmup excluded; same supplied point cloud repeated.",
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "cuda": torch.version.cuda, "device": str(device), "precision": precision,
                        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                        "cpu_threads": torch.get_num_threads(), "platform": platform.platform()},
        "input": {"points": len(points), "dtype": str(points.dtype), "geometry": geometry,
                  "sha256": hashlib.sha256(points.tobytes()).hexdigest()},
        "seed": seed, "warmup": warmup, "iterations": iterations, "results": {},
    }

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def transfer(value):
        if isinstance(value, dict):
            return {key: tensor.contiguous().to(device) if torch.is_tensor(tensor) else tensor
                    for key, tensor in value.items()}
        return value.unsqueeze(0).contiguous().to(device)

    def forward(model, value):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"):
            return model(value)

    with torch.inference_mode():
        for label, encoding in encodings.items():
            local_config = copy.deepcopy(config)
            local_config["data"]["bev_encoding"] = encoding
            torch.manual_seed(seed)
            model = build_model(local_config).to(device).eval()
            host_input = prepare_input(points, geometry, encoding)
            input_bytes = (sum(v.numel() * v.element_size() for v in host_input.values() if torch.is_tensor(v))
                           if isinstance(host_input, dict) else host_input.numel() * host_input.element_size())
            fixed_input = transfer(host_input)
            for _ in range(warmup):
                forward(model, fixed_input)
            synchronize()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            isolated = []
            for _ in range(iterations):
                start = time.perf_counter()
                forward(model, fixed_input)
                synchronize()
                isolated.append((time.perf_counter() - start) * 1000)
            samples = {key: [] for key in ("preprocess", "transfer", "model", "input_to_outputs")}
            for index in range(warmup + iterations):
                synchronize()
                start = part = time.perf_counter()
                value = prepare_input(points, geometry, encoding)
                preprocess_ms = (time.perf_counter() - part) * 1000
                part = time.perf_counter()
                value = transfer(value)
                synchronize()
                transfer_ms = (time.perf_counter() - part) * 1000
                part = time.perf_counter()
                prediction = forward(model, value)
                synchronize()
                model_ms = (time.perf_counter() - part) * 1000
                total_ms = (time.perf_counter() - start) * 1000
                if index >= warmup:
                    for key, elapsed in zip(samples, (preprocess_ms, transfer_ms, model_ms, total_ms)):
                        samples[key].append(elapsed)
                if index == warmup:
                    outputs = prediction["groups"].values() if "groups" in prediction else [prediction]
                    if not all(torch.isfinite(tensor).all() for output in outputs for tensor in output.values()
                               if torch.is_tensor(tensor)):
                        raise RuntimeError(f"nonfinite predictions in {label}")
                del prediction, value
            report["results"][label] = {
                "encoding": resolve_bev_encoding(encoding, geometry).semantic_metadata(),
                "parameter_counts": model_parameter_report(model)["parameter_counts"],
                "input_bytes": input_bytes, "model_only": summary(isolated),
                "latency": {key: summary(value) for key, value in samples.items()},
                "cuda_peak_allocated_mib": (torch.cuda.max_memory_allocated(device) / 1048576
                                            if device.type == "cuda" else None),
            }
            del model, fixed_input, host_input
    reference = report["results"]["rich8"]
    for result in report["results"].values():
        result["model_ratio_to_rich8"] = result["model_only"]["mean_ms"] / reference["model_only"]["mean_ms"]
        result["input_to_outputs_ratio_to_rich8"] = result["latency"]["input_to_outputs"]["mean_ms"] / reference["latency"]["input_to_outputs"]["mean_ms"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--synthetic", action="store_true")
    source.add_argument("--pointcloud", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/encoders/rich8.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--points", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads; use the same setting for all runs")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.points < 0 or args.threads < 1:
        parser.error("points must be nonnegative and threads positive")
    torch.set_num_threads(args.threads)
    config = json.loads(args.config.read_text())
    if args.synthetic:
        geometry = config["data"]["kitti"]["geometry"]
        lower = [geometry[f"{axis}_min"] for axis in "xyz"] + [0.]
        upper = [geometry[f"{axis}_max"] for axis in "xyz"] + [1.]
        points = np.random.default_rng(args.seed).uniform(lower, upper, (args.points, 4)).astype(np.float32)
    else:
        raw = np.fromfile(args.pointcloud, dtype=np.float32)
        if len(raw) % 4:
            parser.error("pointcloud must contain complete float32 [x,y,z,intensity] records")
        points = raw.reshape(-1, 4)
    result = benchmark_models(config, points, device=args.device, precision=args.precision,
                              warmup=args.warmup, iterations=args.iterations, seed=args.seed)
    result["input"]["source"] = "synthetic" if args.synthetic else str(args.pointcloud)
    result["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    for label, values in result["results"].items():
        print(f"{label}: model={values['model_only']['mean_ms']:.3f} ms, "
              f"input-to-outputs={values['latency']['input_to_outputs']['mean_ms']:.3f} ms, "
              f"model/rich8={values['model_ratio_to_rich8']:.3f}, "
              f"pipeline/rich8={values['input_to_outputs_ratio_to_rich8']:.3f}")
    if args.output:
        print(f"Report: {args.output}")
    return result


if __name__ == "__main__":
    main()
