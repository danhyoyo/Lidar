"""Measured vertical extents for BEV predictions, without GT or class priors."""
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2] / "Code"
sys.path.insert(0, str(ROOT))


def api():
    assert (ROOT / "tools/visualization/lift_kitti_bev_to_3d.py").is_file(), "3D lifting exporter missing"
    return importlib.import_module("tools.visualization.lift_kitti_bev_to_3d")


def test_robust_ground_plane_preserves_slope_with_elevated_outliers():
    module = api()
    x, y = np.meshgrid(np.linspace(-4, 4, 17), np.linspace(-4, 4, 17))
    z = .03*x - .02*y - 1.6
    ground = np.c_[x.ravel(), y.ravel(), z.ravel()]
    obstacles = ground[::9].copy()
    obstacles[:, 2] += 1.7
    plane, indices = module.fit_ground_plane(np.r_[ground, obstacles])
    np.testing.assert_allclose(plane, [.03, -.02, -1.6], atol=1e-5)
    assert len(indices) >= 30


def test_lifting_keeps_bev_fields_and_ignores_isolated_high_return():
    module = api()
    from tools.visualization.predict_kitti_overview import bev_detections
    row = np.array([0, .83, 10, 2, 4, 2, .4], dtype=np.float32)
    obj = bev_detections(row[None], {"Car": 0})[0]
    heights = np.linspace(.25, 1.5, 50)
    points = np.c_[np.full(51, 10), np.full(51, 2), np.r_[-1.6+heights, 2.5], np.full(51, .4)]
    lifted, report = module.lift_object(points, obj, np.array([0, 0, -1.6]), min_points=5)
    assert lifted is not None
    assert report["foreground_support_points"] == 50
    assert report["discarded_vertical_outliers"] == 1
    np.testing.assert_array_equal(lifted.row[[0, 1, 2, 3, 5, 6, 8]], row)
    assert lifted.row[4] == pytest.approx(-1.6)
    assert 1.4 < lifted.row[7] < 1.6
    np.testing.assert_allclose(lifted.corners[:4, :2], obj.footprint)


def test_no_height_is_assigned_to_ground_only_or_sparse_detections():
    module = api()
    from tools.visualization.predict_kitti_overview import bev_detections
    obj = bev_detections(np.array([[1, .7, 10, 2, 1, 1, 0]]), {"Pedestrian": 1})[0]
    for z in (np.full(20, -1.6), np.array([-.5, -.4])):
        points = np.c_[np.full(len(z), 10), np.full(len(z), 2), z]
        lifted, report = module.lift_object(points, obj, np.array([0, 0, -1.6]), min_points=5)
        assert lifted is None
        assert report["status"] == "insufficient_foreground_support"


def test_ground_plane_rejects_degenerate_and_missing_support():
    module = api()
    for points in (np.empty((0, 3)), np.c_[np.arange(20), np.zeros(20), np.full(20, -1.6)]):
        with pytest.raises(ValueError, match="ground"):
            module.fit_ground_plane(points)
