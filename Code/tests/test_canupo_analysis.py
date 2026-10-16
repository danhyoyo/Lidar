"""Protect the geometry and binary-row alignment used by the feasibility study."""
import importlib.util
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/analysis/analyze_canupo_kitti.py"
spec = importlib.util.spec_from_file_location("canupo_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_split_preserves_frame_ids_and_removes_dataset_suffix(tmp_path):
    path = tmp_path / "split.txt"
    path.write_text("000001;kitti\n000003;kitti\n\n")
    assert analysis.read_split(path) == {"000001", "000003"}


def test_camera_box_membership_rotates_length_and_preserves_bottom():
    fields = "Car 0 0 0 0 0 1 1 2 2 6 10 3 20 0".split()
    points = np.array([[12.5, 2., 20.], [10., 2., 22.5], [10., 3.1, 20.]])
    np.testing.assert_array_equal(analysis.box_membership(points, fields), [True, False, False])
    fields[-1] = str(np.pi / 2)
    np.testing.assert_array_equal(analysis.box_membership(points, fields), [False, True, False])


def test_msc_roundtrip_and_openmp_output_reordering(tmp_path):
    scales = np.array([4., 2., 1., .5], dtype=np.float32)
    rows = np.zeros(3, dtype=analysis.msc_dtype(4, 4))
    points = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    rows["point"][:, :3] = points
    rows["point"][:, 3] = np.arange(3)
    rows["ab"][:, :, 0] = np.arange(3)[:, None] * .1
    rows["count"] = 10
    path = tmp_path / "core.msc"
    analysis.write_msc(path, scales, rows[[2, 0, 1]])
    loaded_scales, loaded = analysis.read_msc(path)
    np.testing.assert_array_equal(loaded_scales, scales)
    aligned = analysis.align_msc(loaded, points)
    np.testing.assert_array_equal(aligned, rows)
    loaded["point"][0, 3] = 0
    with pytest.raises(ValueError, match="permutation"):
        analysis.align_msc(loaded, points)


def test_original_binary_plane_export_matches_parser():
    # Smoke output is optional; regular tests must not depend on external data/binaries.
    path = SCRIPT.parents[3] / "artifacts/canupo_feasibility_20261016/smoke/core.msc"
    export = path.with_name("core_descriptors.xyz")
    if not path.exists() or not export.exists():
        pytest.skip("Original-binary smoke artifacts unavailable")
    scales, rows = analysis.read_msc(path)
    text = np.loadtxt(export)
    np.testing.assert_allclose(rows["point"][:, :3], text[:, :3], atol=1e-7)
    barycentric = text[:, 3:3 + 3 * len(scales)].reshape(-1, len(scales), 3)
    np.testing.assert_allclose(rows["ab"], barycentric[:, :, :2], atol=1e-7)
    np.testing.assert_allclose(barycentric[:, :, 2], 0, atol=1e-5)


def test_weak_labels_exclude_camera_fov_dontcare_other_classes_and_halos(tmp_path):
    calib = tmp_path / "calib.txt"
    calib.write_text("\nP2: 100 0 50 0 0 100 50 0 0 0 1 0\n"
                     "R0_rect: 1 0 0 0 1 0 0 0 1\n"
                     "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n")
    label = tmp_path / "label.txt"
    label.write_text("Car 0 0 0 0 0 100 100 2 2 6 0 2 10 0\n"
                     "Van 0 0 0 0 0 100 100 2 2 2 4 2 10 0\n"
                     "DontCare -1 -1 -10 48 52 52 57 -1 -1 -1 -1000 -1000 -1000 -10\n")
    points = np.array([[0, 1, 10], [0, 1, 20], [4, 1, 10], [-10, 0, 10],
                       [3.1, 1, 10], [-4, 1, 20]], dtype=float)
    labels, _, objects = analysis.weak_labels(points, calib, label, {"width": 100, "height": 100})
    np.testing.assert_array_equal(labels, [1, -1, -1, -1, -1, 0])
    assert len(objects) == 1


def test_original_canupo_distinguishes_line_plane_and_volume(tmp_path):
    binary = Path(os.environ.get("CANUPO_BIN", "/tmp/canupo_linux_static_64bits/canupo"))
    if not binary.exists():
        pytest.skip("Original CANUPO binary unavailable")
    rng = np.random.default_rng(123)
    shapes = {
        "line": np.c_[np.linspace(-1, 1, 1001), np.zeros((1001, 2))],
        "plane": np.c_[rng.uniform(-1, 1, (12000, 2)), np.zeros(12000)],
        "volume": rng.uniform(-1, 1, (40000, 3)),
    }
    observed = {}
    for name, points in shapes.items():
        source, core, msc = (tmp_path / f"{name}.{ext}" for ext in ["xyz", "core", "msc"])
        np.savetxt(source, points, fmt="%.8f")
        core.write_text("0 0 0 0\n")
        subprocess.run([str(binary), "1", ":", str(source), str(core), str(msc)],
                       env=dict(os.environ, OMP_NUM_THREADS="1"), capture_output=True, check=True, timeout=30)
        _, rows = analysis.read_msc(msc)
        a, b = rows["ab"][0, 0]
        observed[name] = [a, b, 1 - a - b]
    assert observed["line"][0] > .99
    assert observed["plane"][1] > .85
    assert observed["volume"][2] > .85
