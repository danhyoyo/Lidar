"""Explicit backend selection and real compiled/reference equivalence."""

import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "detector"), str(ROOT / "detector/core/datasets")]

from core.bev_encoding import resolve_bev_encoding
from utils_1.preprocess import encode_bev

GEOMETRY = {
    "x_min": 0, "x_max": 4, "x_res": .25,
    "y_min": -2, "y_max": 2, "y_res": .5,
    "z_min": -2.5, "z_max": 1, "z_res": .5,
}


def encoded(points, backend, geometry=GEOMETRY, **options):
    return encode_bev(points, geometry, {"name": "hist14", "backend": backend, **options})


@pytest.mark.parametrize("kind", ["empty", "analytic", "random", "shuffled", "constant", "nonfinite"])
def test_compiled_reference_all_channel_parity(kind):
    for dtype in [np.float32, np.float64]:
        rng = np.random.default_rng(96)
        if kind == "empty":
            points = np.empty((0, 4), dtype)
        elif kind == "analytic":
            points = np.array([[.1, -1.8, -2.0625, .2], [.2, -1.7, -1.1875, .7],
                               [.1, -1.8, -.3125, 1.8], [.2, -1.7, .5625, -.5]], dtype)
        elif kind == "constant":
            points = np.tile(np.array([.1, -1.8, -.7, .4], dtype), (10001, 1))
        else:
            points = rng.uniform([-.2, -2.2, -2.7, -.5], [4.2, 2.2, 1.2, 1.5], (5000, 4)).astype(dtype)
            if kind == "shuffled":
                points = points[rng.permutation(len(points))]
            if kind == "nonfinite":
                points[:3, :4] = np.nan
                points[3:6, 2] = np.inf
                points[6:9, 3] = -np.inf
        reference = encoded(points, "numpy")
        actual = encoded(points, "numba")
        np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=1e-5)
        assert actual.dtype == np.float32 and actual.flags.c_contiguous
        assert np.isfinite(actual).all()


def test_compiled_boundary_parity_including_nextafter_and_current_height_edges():
    points = []
    center = [.1, -1.8, -.5, .4]
    for column, axis in enumerate("xyz"):
        for boundary in (GEOMETRY[f"{axis}_min"] + .001, GEOMETRY[f"{axis}_max"] - .001):
            for value in (np.nextafter(boundary, -np.inf), boundary, np.nextafter(boundary, np.inf)):
                point = center.copy()
                point[column] = value
                points.append(point)
    for boundary in (.25, .5, 1., 1.5):
        for value in (np.nextafter(boundary, -np.inf), boundary, np.nextafter(boundary, np.inf)):
            points.append([value, -1.8, -.5, .4])
    for boundary in (-1.625, -.75, .125):
        for value in (np.nextafter(boundary, -np.inf), boundary, np.nextafter(boundary, np.inf)):
            points.append([.1, -1.8, value, .4])
    # Check points independently: counts aggregated together could hide misindexing.
    for point in points:
        cloud = np.array([point], np.float64)
        np.testing.assert_array_equal(encoded(cloud, "numba"), encoded(cloud, "numpy"))


def test_compiled_handles_noncontiguous_extra_features_without_mutating_input():
    cloud = np.zeros((8, 6), dtype=np.float64)
    cloud[:, :4] = [.1, -1.8, -.5, .4]
    cloud[:, 4:] = np.nan
    points = cloud[::2]
    assert not points.flags.c_contiguous
    original = cloud.copy()
    np.testing.assert_allclose(encoded(points, "numba"), encoded(points, "numpy"), atol=1e-6, rtol=1e-5)
    np.testing.assert_array_equal(cloud, original)


def test_compiled_saturation_and_custom_normalization():
    points = np.tile([.1, -1.8, -.5, 1e308], (100, 1))
    for norm, scale in [(8, 2), (64, .5)]:
        expected = encoded(points, "numpy", density_norm=norm, intensity_scale=scale)
        actual = encoded(points, "numba", density_norm=norm, intensity_scale=scale)
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_numba_request_does_not_call_numpy_reference(monkeypatch):
    module = importlib.import_module("utils_1.bev_backend")
    points = np.array([[.1, -1.8, -.5, .4]])
    expected = encoded(points, "numpy")
    def forbidden_reference(*args, **kwargs):
        raise AssertionError("Numba must run its compiled kernel without a NumPy fallback")
    monkeypatch.setattr(module, "encode_hist14_numpy", forbidden_reference)
    np.testing.assert_allclose(encoded(points, "numba"), expected, atol=1e-6, rtol=1e-5)


def test_missing_numba_is_actionable_and_numpy_remains_available_in_fresh_process():
    script = f"""
import builtins, sys, numpy as np
sys.path[:0] = [{str(ROOT / 'detector')!r}, {str(ROOT / 'detector/core/datasets')!r}]
original_import = builtins.__import__
def without_numba(name, *args, **kwargs):
    if name == 'numba' or name.startswith('numba.'):
        raise ImportError('Numba intentionally unavailable in dependency fixture')
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_numba
from utils_1.preprocess import encode_bev
cloud = np.array([[.1, -1.8, -.5, .4]])
result = encode_bev(cloud, {GEOMETRY!r}, {{'name': 'hist14', 'backend': 'numpy'}})
assert result.shape == (8, 16, 14) and result[..., 11].sum() == 1
assert 'numba' not in sys.modules
for points in [cloud, np.empty((0, 4))]:
    try:
        encode_bev(points, {GEOMETRY!r}, {{'name': 'hist14', 'backend': 'numba'}})
    except ImportError as exc:
        assert 'Numba' in str(exc) and 'backend=numpy' in str(exc), str(exc)
    else:
        raise AssertionError('Explicit missing Numba must fail, even for empty clouds')
assert encode_bev(cloud, {GEOMETRY!r}, {{'name': 'hist14', 'backend': 'numpy'}})[..., 11].sum() == 1
"""
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True)


def test_backend_choice_keeps_semantic_identity():
    numpy = resolve_bev_encoding({"name": "hist14", "backend": "numpy"}, GEOMETRY)
    numba = resolve_bev_encoding({"name": "hist14", "backend": "numba"}, GEOMETRY)
    assert numpy.semantic_hash == numba.semantic_hash


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_public_backends_reject_invalid_points_before_kernel(backend):
    with pytest.raises(ValueError, match="points"):
        encoded(np.zeros((2, 3)), backend)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_geometry_tolerance_cannot_alias_an_out_of_grid_point_into_next_row(backend):
    # The legacy relative divisibility tolerance accepts this near-integer grid.
    # A valid ROI point beyond the rounded grid must not alias the next row.
    geometry = dict(GEOMETRY, x_max=1000.0095, x_res=1)
    cloud = np.array([[1000.005, -1.8, -.5, .4]])
    with pytest.raises(ValueError, match="outside.*grid"):
        encoded(cloud, backend, geometry)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("kind", ["empty", "random", "strided", "integer", "constant"])
def test_direct_chw_matches_public_hwc_and_owns_each_result(backend, kind):
    module = importlib.import_module("utils_1.bev_backend")
    rng = np.random.default_rng(42)
    points = rng.uniform([0, -2, -2.5, 0], [4, 2, 1, 1], (2000, 4)).astype(np.float32)
    if kind == "empty":
        points = points[:0]
    elif kind == "strided":
        points = points[::2, ::-1]
    elif kind == "integer":
        points = np.array([[1, -1, 0, 1], [2, 1, -1, 0]], dtype=np.int64)
    elif kind == "constant":
        points = np.tile([.1, -1.8, -.7, .4], (10001, 1))
    original = points.copy()
    schema = resolve_bev_encoding({"name": "hist14", "backend": backend}, GEOMETRY)
    actual = module.encode_hist14_chw(points, schema)
    expected = encoded(points, "numpy").transpose(2, 0, 1)
    assert actual.shape == (14, 8, 16)
    assert actual.dtype == np.float32 and actual.flags.c_contiguous
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
    saved = actual.copy()
    later = module.encode_hist14_chw(np.array([[3.5, 1.5, .5, .8]]), schema)
    assert not np.shares_memory(actual, later)
    np.testing.assert_array_equal(actual, saved)
    np.testing.assert_array_equal(points, original)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_direct_chw_calls_are_safe_to_run_concurrently(backend):
    from concurrent.futures import ThreadPoolExecutor
    module = importlib.import_module("utils_1.bev_backend")
    schema = resolve_bev_encoding({"name": "hist14", "backend": backend}, GEOMETRY)
    clouds = [np.tile([.1 + index * .3, -1.8, -.5, .4], (index + 1, 1))
              for index in range(8)]
    expected = [encoded(points, "numpy").transpose(2, 0, 1) for points in clouds]
    # Initialize compilation before exercising concurrent rasterization.
    module.encode_hist14_chw(clouds[0], schema)
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(lambda points: module.encode_hist14_chw(points, schema), clouds))
    for value, reference in zip(actual, expected):
        np.testing.assert_allclose(value, reference, atol=1e-6, rtol=1e-5)
    assert all(not np.shares_memory(a, b) for index, a in enumerate(actual) for b in actual[index + 1:])


def test_compiled_large_unsaturated_counts_and_read_only_strided_inputs():
    module = importlib.import_module("utils_1.bev_backend")
    schema = resolve_bev_encoding({"name": "hist14", "backend": "numba", "density_norm": 4096}, GEOMETRY)
    cloud = np.tile([.1, -1.8, -.5, .4, np.nan, np.inf], (2048, 1))
    points = cloud[::2]
    points.flags.writeable = False
    reference = encoded(points, "numpy", density_norm=4096)
    actual = module.encode_hist14_chw(points, schema)
    np.testing.assert_allclose(actual, reference.transpose(2, 0, 1), atol=1e-6, rtol=1e-5)
    assert 0 < actual[2, 0, 0] < 1
    assert 0 < actual[10, 0, 0] < 1
