"""Matched CPU comparison records scope, interleaving and numeric parity."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_benchmark():
    path = ROOT / "tools/benchmarks/benchmark_hist14_vs_binary.py"
    assert path.is_file(), "matched Hist14/Binary benchmark is missing"
    spec = importlib.util.spec_from_file_location("hist14_speed_comparison", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_comparison_interleaves_identical_frames_and_records_actual_payload(tmp_path):
    module = load_benchmark()
    geometry = dict(x_min=0, x_max=4, x_res=.25, y_min=-2, y_max=2,
                    y_res=.5, z_min=-2.5, z_max=1, z_res=.5)
    paths = []
    for index in range(2):
        path = tmp_path / f"{index:06}.bin"
        np.array([[.1 + index, -1.8, -.5, .4]], np.float32).tofile(path)
        paths.append(path)
    report = module.compare(paths, geometry, warmup=1, iterations=3, seed=17)
    assert report["scope"]["file_io"] is False
    assert report["scope"]["h2d"] is False
    assert report["scope"]["model"] is False
    assert report["protocol"]["order"] == "randomized_interleaving_per_frame"
    assert len(report["frames"]) == 2
    for frame in report["frames"]:
        assert len(frame["sha256"]) == 64
        assert frame["point_count"] == 1
        assert len(frame["iteration_order"]) == 3
        methods = frame["results"]
        assert set(methods) == {"binary_slices_raster", "binary_slices_chw",
                                "hist14_numba_raster", "hist14_numba_chw"}
        for order in frame["iteration_order"]:
            assert sorted(order) == sorted(methods)
        assert len({tuple(order) for order in frame["iteration_order"]}) > 1
        for name, values in methods.items():
            assert len(values["steady_ms"]) == 3
            assert values["first_call_ms"] >= 0
            assert values["mean_ms"] >= 0
            if name.endswith("_chw"):
                assert values["contiguous"]
                assert values["shape"][1:] == [8, 16]
        assert methods["hist14_numba_chw"]["payload_bytes"] == 14 * 8 * 16 * 4
        assert methods["binary_slices_chw"]["payload_bytes"] == 7 * 8 * 16 * 4
        assert frame["hist14_layout_bit_mismatches"] == 0
    assert report["environment"]["numba"]
    assert all(len(value) == 64 for value in report["source_sha256"].values())
    # The report can be published without custom NumPy JSON encoders.
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("warmup,iterations", [(-1, 2), (1, 0)])
def test_comparison_rejects_invalid_repeat_counts(warmup, iterations):
    with pytest.raises(ValueError, match="warmup|iterations"):
        load_benchmark().compare([], {}, warmup=warmup, iterations=iterations)


def test_comparison_rejects_invalid_kitti_payload(tmp_path):
    path = tmp_path / "bad.bin"
    np.zeros(5, np.float32).tofile(path)
    with pytest.raises(ValueError, match="four"):
        load_benchmark().compare([path], {})


def test_comparison_checks_trusted_baseline_bitwise(tmp_path):
    module = load_benchmark()
    geometry = dict(x_min=0, x_max=4, x_res=.25, y_min=-2, y_max=2,
                    y_res=.5, z_min=-2.5, z_max=1, z_res=.5)
    cloud = tmp_path / "000000.bin"
    np.array([[.1, -1.8, -.5, .4]], np.float32).tofile(cloud)
    baseline = tmp_path / "baseline.py"
    baseline.write_text((ROOT / "detector/core/datasets/utils_1/bev_backend.py").read_text())
    report = module.compare([cloud], geometry, warmup=0, iterations=1,
                            baseline_backend=baseline)
    assert report["frames"][0]["hist14_before_bit_mismatches"] == 0
    assert str(baseline) in report["source_sha256"]
    assert "hist14_before_chw" in report["mean_ms"]
    # A real numeric regression in the supplied baseline must stop publication.
    baseline = tmp_path / "changed_baseline.py"
    baseline.write_text("from utils_1.bev_backend import encode_hist14_chw as encode\n"
                        "def encode_hist14_chw(points, schema):\n"
                        "    return encode(points, schema) + 1\n")
    with pytest.raises(AssertionError, match="differs from baseline"):
        module.compare([cloud], geometry, warmup=0, iterations=1,
                       baseline_backend=baseline)
