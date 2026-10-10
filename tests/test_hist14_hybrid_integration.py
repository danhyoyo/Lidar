"""Real hybrid sampling/global transforms must precede hist14 rasterization."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "detector"),
               str(ROOT / "detector/core/datasets"), str(ROOT / "tools/kitti_training_pipeline")]

from core.datasets.dataset import Dataset
from utils_1.preprocess import encode_bev
from test_hybrid_augmentation import database, recipe, CLASSES


@pytest.mark.parametrize("encoding_backend", ["numpy", "numba"])
@pytest.mark.parametrize("target_backend", ["python", "numba"])
@pytest.mark.parametrize("grouped", [False, True])
def test_actual_hybrid_paste_then_global_scale_then_hist14(tmp_path, encoding_backend, target_backend, grouped):
    root, source, object_points = database(tmp_path)
    np.empty((0, 4), np.float32).tofile(root / "pointcloud/000000.bin")
    (root / "label/000000.txt").write_text("")
    manifest = tmp_path / "frames.txt"
    manifest.write_text("000000;kitti\n000001;kitti\n")
    geometry = dict(x_min=0, x_max=32, x_res=.5, y_min=-8, y_max=8,
                    y_res=.5, z_min=-2.5, z_max=1, z_res=.1)
    encoding = {"name": "hist14", "backend": encoding_backend}
    data = {"num_classes": 3, "out_size_factor": 4, "gaussian_overlap": .1,
            "min_radius": 1, "bev_encoding": encoding,
            "kitti": {"location": str(root), "objects": CLASSES, "geometry": geometry}}
    if grouped:
        data["head_groups"] = [{"name": "car", "classes": ["Car"]},
                               {"name": "ped_cyc", "classes": ["Pedestrian", "Cyclist"]}]
    augmentation = {"mode": "openpcdet", "AUG_CONFIG_LIST": [
        recipe(PROBABILITY=1),
        {"NAME": "random_world_scaling", "WORLD_SCALE_RANGE": [1.25, 1.25]},
    ]}
    dataset = Dataset(str(manifest), data, augmentation, "gaussian", "train", target_backend)
    sample = dataset[0]
    transformed = object_points.copy()
    transformed[:, :3] *= 1.25
    expected = encode_bev(transformed, geometry, {"name": "hist14", "backend": "numpy"})
    np.testing.assert_allclose(sample["voxel"].numpy(), expected.transpose(2, 0, 1), atol=1e-6, rtol=1e-5)
    ix = np.floor(transformed[:, 0].astype(np.float64) / .5).astype(int)
    iy = np.floor((transformed[:, 1].astype(np.float64) + 8) / .5).astype(int)
    occupied = np.unique(np.column_stack([iy, ix]), axis=0)
    np.testing.assert_array_equal(np.argwhere(sample["voxel"][11].numpy()), occupied)
    center = source[4:6] * 1.25
    peak_x, peak_y = int(center[0] / 2), int((center[1] + 8) / 2)
    target = sample["groups"]["car"] if grouped else sample
    assert target["cls"][0, peak_y, peak_x] == 1
    np.testing.assert_allclose(target["offset"][:, peak_y, peak_x].numpy(),
                               center - [peak_x * 2, peak_y * 2 - 8], atol=1e-6)
    if grouped:
        assert all(not tensor.any() for tensor in sample["groups"]["ped_cyc"].values())
    # A different transform on the next access must re-encode newly augmented points.
    dataset.augment.queue[-1] = ("random_world_scaling", 1., np.array([1., 1.]))
    second = dataset[0]
    original_encoding = encode_bev(object_points, geometry, {"name": "hist14"})
    np.testing.assert_allclose(second["voxel"].numpy(), original_encoding.transpose(2, 0, 1), atol=1e-6, rtol=1e-5)
    assert not np.array_equal(second["voxel"].numpy(), sample["voxel"].numpy())
    validation = Dataset(str(manifest), data, augmentation, "gaussian", "validation", target_backend)
    assert not validation[0]["voxel"].any()
    np.testing.assert_array_equal(dataset.read_points(root / "pointcloud/000001.bin"), object_points)


def test_encoding_benchmark_cli_records_initialization_warmup_and_steady_state(tmp_path):
    path = ROOT / "tools/benchmarks/benchmark_bev_encodings.py"
    assert path.is_file(), "encoding benchmark CLI has not been implemented"
    output = tmp_path / "encoding.json"
    subprocess.run([sys.executable, str(path), "--synthetic", "--points", "30",
                    "--warmup", "1", "--iterations", "2", "--encodings",
                    "rich8", "hist14_numpy", "hist14_numba", "--output", str(output)],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
    report = json.loads(output.read_text())
    assert report["input"]["point_count"] == 30
    assert report["input"]["dtype"] == "float32"
    assert report["environment"]["cpu_count"] > 0
    results = report["results"]
    assert set(results) == {"rich8", "hist14_numpy", "hist14_numba"}
    for value in results.values():
        assert value["first_call_ms"] >= 0
        assert len(value["warmup_ms"]) == 1
        assert len(value["steady_state_ms"]) == 2
        assert 0 <= value["p50_ms"] <= value["p95_ms"]
        assert value["mean_ms"] >= 0
    assert results["hist14_numpy"]["shape_yxc"] == [800, 704, 14]
    assert results["hist14_numpy"]["semantic_hash"] == results["hist14_numba"]["semantic_hash"]
    assert results["hist14_numpy"]["backend"] == "numpy"
    assert results["hist14_numba"]["backend"] == "numba"


def test_encoding_benchmark_chw_cli_measures_contiguous_model_input(tmp_path):
    output = tmp_path / "chw.json"
    subprocess.run([sys.executable, str(ROOT / "tools/benchmarks/benchmark_bev_encodings.py"),
                    "--synthetic", "--points", "30", "--warmup", "1", "--iterations", "2",
                    "--encodings", "rich8", "hist14_numpy", "hist14_numba",
                    "--layout", "chw", "--output", str(output)],
                   cwd=tmp_path, check=True, capture_output=True, text=True)
    report = json.loads(output.read_text())
    assert "CHW" in report["scope"]
    for name, value in report["results"].items():
        channels = 8 if name == "rich8" else 14
        assert value["output_layout"] == "chw"
        assert value["result_shape"] == [channels, 800, 704]
        assert value["shape_yxc"] == [800, 704, channels]
        assert value["output_contiguous"] is True
