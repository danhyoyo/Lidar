"""CPU preparation optimizations must preserve the existing packed representation."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "detector"), str(ROOT / "detector/core/datasets")]
from core.datasets.utils_1 import pillar_backend as backend

GEOMETRY = {"x_min": 0, "x_max": 16, "x_res": .1,
            "y_min": -8, "y_max": 8, "y_res": .1,
            "z_min": -2.5, "z_max": 1, "z_res": .1}


def compare(points, geometry=GEOMETRY, **options):
    encoding = {"name": "pillar_rich_gate", **options}
    before = points.copy()
    old = backend.prepare_pillars(points, geometry, encoding, cpu_backend="numpy")
    new = backend.prepare_pillars(points, geometry, encoding, cpu_backend="numba")
    assert new.keys() == old.keys()
    for key in old:
        if isinstance(old[key], np.ndarray):
            assert new[key].dtype == old[key].dtype
            assert new[key].flags.c_contiguous
            np.testing.assert_array_equal(new[key], old[key], err_msg=key)
            np.testing.assert_array_equal(new[key].view(np.uint8), old[key].view(np.uint8), err_msg=key)
        else:
            assert new[key] == old[key]
    np.testing.assert_array_equal(points, before)
    return new


@pytest.mark.parametrize("name", ["pillar32", "pillar_rich", "pillar_rich_gate"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scale,density", [(1, 32), (.5, 8), (2, 64)])
def test_random_cloud_matches_reference_exactly(name, dtype, scale, density):
    rng = np.random.default_rng(32)
    points = rng.uniform([-1, -9, -3, -.2], [17, 9, 2, 1.2], (3000, 4)).astype(dtype)
    compare(points, name=name, intensity_scale=scale, density_norm=density)


@pytest.mark.parametrize("size", [0, 1, 2, 4096])
def test_empty_singleton_and_many_duplicate_points_match_reference(size):
    points = np.tile(np.array([[.11, .12, -.5, .2], [.15, .12, .5, .8]], np.float32), (size, 1))
    if size == 1:
        points = points[:1]
    compare(points)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_xy_height_roi_boundaries_and_nonfinite_values_match_reference(dtype):
    rows = []
    for column, values in ((0, [.1, .5, 1., 15.9]), (1, [-7.9, -.5, 0., .5, 7.9]),
                           (2, [-2.5 + 3.5/3, -2.5 + 7/3])):
        for value in values:
            edge = dtype(value)
            for coordinate in (np.nextafter(edge, dtype(-np.inf)), edge, np.nextafter(edge, dtype(np.inf))):
                row = [.25, -.25, -.5, .3]
                row[column] = coordinate
                rows.append(row)
    for column, axis in enumerate("xyz"):
        for side in ("min", "max"):
            edge = dtype(GEOMETRY[f"{axis}_{side}"] + (.001 if side == "min" else -.001))
            for coordinate in (np.nextafter(edge, dtype(-np.inf)), edge, np.nextafter(edge, dtype(np.inf))):
                row = [.25, -.25, -.5, .3]
                row[column] = coordinate
                rows.append(row)
    for column in range(4):
        for invalid in (np.nan, np.inf, -np.inf):
            row = [.25, -.25, -.5, .3]
            row[column] = invalid
            rows.append(row)
    compare(np.asarray(rows, dtype=dtype))


def test_extra_columns_noncontiguous_input_and_permutation_match_reference():
    rng = np.random.default_rng(19)
    points = rng.uniform([0, -8, -2.5, 0], [16, 8, 1, 1], (500, 4)).astype(np.float32)
    extra = np.column_stack((points, np.full((500, 2), np.nan, dtype=np.float32)))
    compare(extra[::2])
    compare(np.asfortranarray(points))
    compare(points[rng.permutation(len(points))])


@pytest.mark.parametrize("dtype", [np.float16, np.int32, np.float32, np.float64])
def test_unusual_input_dtypes_and_signed_zero_match_reference(dtype):
    points = np.array([[1, 1, 0, -0.0], [2, 2, 0, 0.0]], dtype=dtype)
    compare(points)


def test_large_sparse_grid_uses_bounded_workspace_without_changing_pillar_order():
    geometry = dict(GEOMETRY, x_max=10000, y_min=0, y_max=10000, x_res=1, y_res=1)
    compare(np.array([[9000.1, 8999.2, -.5, .2], [.1, .2, -.5, .3]], np.float32), geometry)


def test_missing_numba_falls_back_only_for_auto(monkeypatch):
    monkeypatch.setattr(backend, "_get_numba_kernel", lambda: None)
    points = np.array([[.25, -.25, -.5, .3]], np.float32)
    old = backend.prepare_pillars(points, GEOMETRY, {"name": "pillar_rich_gate"}, cpu_backend="numpy")
    new = backend.prepare_pillars(points, GEOMETRY, {"name": "pillar_rich_gate"})
    for key in old:
        np.testing.assert_array_equal(old[key], new[key])
    with pytest.raises(RuntimeError, match="Numba"):
        backend.prepare_pillars(points, GEOMETRY, cpu_backend="numba")


def test_invalid_preparation_backend_is_rejected():
    with pytest.raises(ValueError, match="cpu_backend"):
        backend.prepare_pillars(np.empty((0, 4), np.float32), GEOMETRY, cpu_backend="cuda")
