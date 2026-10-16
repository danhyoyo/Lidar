import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/analysis/bev_geometry_maps.py"
spec = importlib.util.spec_from_file_location("bev_geometry_maps", SCRIPT)
maps = importlib.util.module_from_spec(spec)
spec.loader.exec_module(maps)


def geometry():
    return {f"{a}_{k}": v for a in "xyz" for k, v in
            [("min", -2.), ("max", 2.), ("res", .1)]}


def test_keeps_original_hist14_and_empty_cells():
    points = np.random.default_rng(3).uniform(-1.5, 1.5, (150, 4)).astype(np.float32)
    result = maps.encode_maps(points, geometry())
    expected = maps.encode_bev(points, geometry(), {"name": "hist14", "backend": "numpy"})
    np.testing.assert_array_equal(result["hist22"][..., :14], expected)
    np.testing.assert_array_equal(result["hist18"], result["hist22"][..., :18])
    assert not result["hist22"][~result["occupied"]].any()
    assert result["hist22"].shape == (40, 40, 22)


def test_pooled_covariance_matches_brute_force_windows():
    rng = np.random.default_rng(9)
    points = rng.uniform(-1.7, 1.7, (1000, 4))
    geo = geometry(); result = maps.encode_maps(points, geo)
    ix = np.floor((points[:, 0] + 2) / .1).astype(int)
    iy = np.floor((points[:, 1] + 2) / .1).astype(int)
    occupied_y, occupied_x = np.where(result["occupied"])
    for y, x in list(zip(occupied_y, occupied_x))[::37]:
        for j, radius in enumerate([5, 10]):
            local = points[(abs(ix - x) <= radius) & (abs(iy - y) <= radius), :3]
            assert result["counts"][y, x, j] == len(local)
            cov = np.cov(local, rowvar=False, bias=True) if len(local) > 1 else np.zeros((3, 3))
            bary, valid = maps.dimensionality(cov[None], np.array([len(local)]))
            np.testing.assert_allclose(result["barycentric"][y, x, j], bary[0], atol=2e-6)


def test_dimensionality_axes_and_degenerate_support():
    covariance = np.array([np.diag([1., 0, 0]), np.diag([1., 1, 0]),
                           np.eye(3), np.zeros((3, 3)), np.eye(3)])
    bary, valid = maps.dimensionality(covariance, np.array([10, 10, 10, 10, 2]))
    np.testing.assert_allclose(bary[:3], np.eye(3), atol=1e-6)
    np.testing.assert_array_equal(valid, [True, True, True, False, False])
    assert not bary[3:].any()


def test_empty_cloud_and_nonfinite_rows_are_handled():
    empty = maps.encode_maps(np.empty((0, 4)), geometry())
    assert not empty["hist22"].any()
    result = maps.encode_maps(np.array([[np.nan, 0, 0, 1], [0, 0, 0, np.inf], [0, 0, 0, 1]]), geometry())
    assert result["roi_points"] == 1
    assert np.isfinite(result["hist22"]).all()
    assert not result["hist22"][..., 18:].any()


def test_geometry_unchanged_by_joint_translation_of_points_and_roi():
    points = np.random.default_rng(7).uniform(-1.5, 1.5, (500, 4))
    original = maps.encode_maps(points, geometry())
    shift = np.array([1000.13, -400.37, 2.71])
    moved = points.copy(); moved[:, :3] += shift
    geo = geometry()
    for j, axis in enumerate("xyz"):
        geo[f"{axis}_min"] += shift[j]; geo[f"{axis}_max"] += shift[j]
    translated = maps.encode_maps(moved, geo)
    np.testing.assert_array_equal(original["counts"], translated["counts"])
    np.testing.assert_allclose(original["hist22"][..., 18:], translated["hist22"][..., 18:], atol=2e-5)


def test_core_lookup_preserves_each_encoders_real_bin_boundaries():
    path = SCRIPT.parents[1] / "visualization/compare_bev_geometry_maps.py"
    loader = importlib.util.spec_from_file_location("compare_geometry_maps", path)
    module = importlib.util.module_from_spec(loader); loader.loader.exec_module(module)
    # At 0.7m, float32 division and promoted division can fall on opposite sides.
    points = np.array([[.7, .3, .2, .8], [1.3, -.7, .4, .4], [.9, -.3, .1, .5]], dtype=np.float32)
    bundle = maps.encode_maps(points, geometry())
    features = module.sample_features(bundle, points[:, :3], geometry())
    np.testing.assert_array_equal(features["hist14"][:, 11], np.ones(len(points)))
    assert (features["rich8"][:, 7] > 0).all()
    np.testing.assert_array_equal(features["hist18"][:, :14], features["hist14"])


def test_moment_roi_mask_uses_same_promoted_boundary_as_hist14():
    geo = geometry(); geo.update(x_min=0., x_max=4.)
    points = np.array([[.001, .2, .2, .5], [.3, -1.999, .4, .8],
                       [3.999, .4, .4, .5]], dtype=np.float32)
    bundle = maps.encode_maps(points, geo)
    np.testing.assert_array_equal(bundle["occupied"], bundle["hist14"][..., 11] > 0)
    assert not bundle["hist22"][~bundle["occupied"]].any()
