#!/usr/bin/env python3
"""Matched NumPy/Numba pillar CPU preparation and optional CUDA forward timing."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "detector"), str(ROOT / "detector/core/datasets"),
               str(ROOT / "tools/kitti_training_pipeline")]
from common import build_model
from core.datasets.utils_1.pillar_backend import prepare_pillars
from tools.benchmarks.benchmark_pillar_encoders import summary


def benchmark_preparation(config, clouds, *, device="cpu", precision="bf16", warmup=10, iterations=50):
    if warmup < 0 or iterations < 1:
        raise ValueError("warmup must be nonnegative and iterations positive")
    if device not in {"cpu", "cuda"} or precision not in {"fp32", "bf16"}:
        raise ValueError("device must be cpu/cuda and precision fp32/bf16")
    geometry, encoding = config["data"]["kitti"]["geometry"], config["data"]["bev_encoding"]
    if not clouds:
        raise ValueError("at least one point cloud is required")
    report = {"scope": "Same fixed encoder/model and inputs; no architecture or AP experiment. "
                       "CPU timing excludes I/O and JIT first-call. CUDA pipeline includes CPU prep, "
                       "contiguous input/H2D and detector forward, excludes I/O/decode/NMS and DataLoader overlap.",
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "torch": torch.__version__, "numba": importlib.metadata.version("numba"),
                              "cpu_threads": torch.get_num_threads(),
                              "platform": platform.platform(), "device": device, "precision": precision},
              "warmup": warmup, "iterations": iterations, "frames": []}
    model = None
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA BF16 is unavailable")
        report["environment"].update(gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda)
        torch.manual_seed(42)
        model = build_model(config).cuda().eval()
        # Exercise a non-identity gate during output-parity checks too.
        if hasattr(model.point_encoder, "gate"):
            with torch.no_grad():
                model.point_encoder.gate[2].weight.fill_(.01)

    def prep(points, backend):
        return prepare_pillars(points, geometry, encoding, cpu_backend=backend)

    def forward(value):
        value = {key: torch.from_numpy(array).contiguous().cuda() if isinstance(array, np.ndarray) else array
                 for key, array in value.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            return model(value)

    first_points = next(iter(clouds.values()))
    start = time.perf_counter()
    prep(first_points, "numba")
    report["numba_first_call_ms"] = (time.perf_counter() - start) * 1000
    report["first_call_note"] = "Includes import/cache loading or compilation, depending on existing Numba disk cache."
    for name, points in clouds.items():
        baseline, optimized = prep(points, "numpy"), prep(points, "numba")
        for key in baseline:
            np.testing.assert_array_equal(baseline[key], optimized[key], err_msg=f"{name}/{key}")
            if isinstance(baseline[key], np.ndarray):
                assert baseline[key].dtype == optimized[key].dtype, f"{name}/{key}: dtype mismatch"
                assert baseline[key].tobytes() == optimized[key].tobytes(), f"{name}/{key}: bit mismatch"
        frame = {"name": name, "points": len(points), "retained_points": len(baseline["features"]),
                 "occupied_pillars": len(baseline["coords"]),
                 "sha256": hashlib.sha256(points.tobytes()).hexdigest(), "exact_input_parity": True,
                 "cpu": {}, "cuda_pipeline": {}}
        elapsed = {backend: [] for backend in ("numpy", "numba")}
        # Alternate which implementation runs first to reduce order bias.
        for index in range(warmup + iterations):
            for backend in (("numpy", "numba") if index % 2 == 0 else ("numba", "numpy")):
                start = time.perf_counter()
                prep(points, backend)
                if index >= warmup:
                    elapsed[backend].append((time.perf_counter() - start) * 1000)
        frame["cpu"] = {key: summary(values) for key, values in elapsed.items()}
        frame["cpu_speedup"] = frame["cpu"]["numpy"]["mean_ms"] / frame["cpu"]["numba"]["mean_ms"]
        if model is not None:
            with torch.inference_mode():
                torch.testing.assert_close(forward(baseline), forward(optimized), rtol=0, atol=0)
                frame["exact_output_parity"] = True
                elapsed = {backend: [] for backend in ("numpy", "numba")}
                for index in range(warmup + iterations):
                    for backend in (("numpy", "numba") if index % 2 == 0 else ("numba", "numpy")):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        forward(prep(points, backend))
                        torch.cuda.synchronize()
                        if index >= warmup:
                            elapsed[backend].append((time.perf_counter() - start) * 1000)
                frame["cuda_pipeline"] = {key: summary(values) for key, values in elapsed.items()}
                frame["pipeline_speedup"] = (frame["cuda_pipeline"]["numpy"]["mean_ms"] /
                                             frame["cuda_pipeline"]["numba"]["mean_ms"])
        report["frames"].append(frame)
    report["aggregate"] = {}
    for scope in ("cpu", "cuda_pipeline"):
        if not report["frames"][0][scope]:
            continue
        values = {backend: float(np.mean([row[scope][backend]["mean_ms"] for row in report["frames"]]))
                  for backend in ("numpy", "numba")}
        report["aggregate"][scope] = {**values, "speedup": values["numpy"] / values["numba"],
                                      "latency_reduction_percent": 100 * (1 - values["numba"] / values["numpy"])}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments/encoders/pillar_rich_gate.json")
    parser.add_argument("--pointcloud-dir", type=Path, required=True)
    parser.add_argument("--frames", nargs="+", default=["000000", "000001", "000010", "000100", "001000"])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    clouds = {}
    for frame in args.frames:
        raw = np.fromfile(args.pointcloud_dir / f"{frame}.bin", dtype=np.float32)
        if raw.size % 4:
            parser.error(f"{frame}: incomplete float32 point records")
        clouds[frame] = raw.reshape(-1, 4)
    report = benchmark_preparation(json.loads(args.config.read_text()), clouds,
                                   device=args.device, precision=args.precision,
                                   warmup=args.warmup, iterations=args.iterations)
    report["config_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    for frame in report["frames"]:
        print(f"{frame['name']}: CPU {frame['cpu']['numpy']['mean_ms']:.3f} → "
              f"{frame['cpu']['numba']['mean_ms']:.3f} ms ({frame['cpu_speedup']:.2f}x)")
    print(json.dumps(report["aggregate"], indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
