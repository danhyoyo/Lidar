"""Actual BEV inference must not manufacture vertical predictions."""
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2] / "Code"
sys.path.insert(0, str(ROOT))


def api():
    path = ROOT / "tools/visualization/predict_kitti_overview.py"
    assert path.is_file(), "The checkpoint overview predictor is missing"
    return importlib.import_module("tools.visualization.predict_kitti_overview")


def test_bev_footprint_preserves_predicted_xy_dimensions_and_rotation():
    module = api()
    row = np.array([2, .87, 10, 3, 4, 2, np.pi / 2])
    obj = module.bev_detections(row[None], {"Cyclist": 2})[0]
    np.testing.assert_allclose(obj.footprint.mean(axis=0), [10, 3])
    np.testing.assert_allclose(np.ptp(obj.footprint, axis=0), [2, 4])
    assert obj.score == .87
    np.testing.assert_array_equal(obj.row, row)


def test_bev_point_crop_does_not_invent_height_or_filter_by_ground_truth():
    module = api()
    obj = module.bev_detections(np.array([[0, .8, 10, 3, 4, 2, 0]]), {"Car": 0})[0]
    points = np.array([[10, 3, -100], [10, 3, 100], [12, 4, 0], [12.1, 3, 0], [10, 4.1, 0]])
    assert module.points_in_bev(points, obj).tolist() == [True, True, True, False, False]


@pytest.mark.parametrize("row", [[0,.8,10,3,0,2,0], [0,float('nan'),10,3,4,2,0],
                                  [9,.8,10,3,4,2,0], [0,.8,10,3,-2,4,2,1,0]])
def test_invalid_or_3d_rows_are_not_silently_reinterpreted_as_bev(row):
    with pytest.raises(ValueError):
        api().bev_detections(np.array([row]), {"Car": 0})


def test_default_config_comes_from_checkpoint_and_does_not_mutate_it(tmp_path):
    import json
    module = api()
    checkpoint = {"config": {"model": {"header_architecture": "independent"}}}
    unrelated = tmp_path / "config.resolved.json"
    unrelated.write_text(json.dumps({"model": {"header_architecture": "shared"}}))
    config = module.inference_config(checkpoint)
    assert config["model"]["header_architecture"] == "independent"
    config["model"]["header_architecture"] = "changed"
    assert checkpoint["config"]["model"]["header_architecture"] == "independent"
    assert module.inference_config(checkpoint, unrelated)["model"]["header_architecture"] == "shared"
