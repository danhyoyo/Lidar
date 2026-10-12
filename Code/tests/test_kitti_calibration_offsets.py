"""Read real rectified KITTI calibration and retain its projection translation."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.kitti_training_pipeline.kitti_box_conversion import (
    camera_box_corners,
    camera_box_to_lidar,
    predictions_to_annotation,
    read_calibration,
)


# P2, R0_rect and Tr_velo_to_cam from KITTI object training frame 000000.
CALIBRATION = """P2: 707.0493 0 604.0814 45.75831 0 707.0493 180.5066 -0.3454157 0 0 1 0.004981016
R0_rect: 0.9999128 0.01009263 -0.008511932 -0.01012729 0.9999406 -0.004037671 0.008470675 0.004123522 0.9999556
Tr_velo_to_cam: 0.006927964 -0.9999722 -0.002757829 -0.02457729 -0.001162982 0.002749836 -0.9999955 -0.06127237 0.9999753 0.006931141 -0.001143899 -0.3321029
"""


def test_calibration_file_with_real_depth_translation_projects_reference_annotation(tmp_path):
    path = tmp_path / "000000.txt"
    path.write_text(CALIBRATION)
    calibration = read_calibration(path)
    assert calibration.p2[2, 3] == pytest.approx(0.004981016)

    camera = np.array([1., 1.8, 20., 4.2, 1.8, 1.6, 0.8])
    lidar = camera_box_to_lidar(camera, calibration)
    annotation = predictions_to_annotation(
        np.array([[0., .9, *lidar]]), calibration, {"Car": 0}, image_size=(1242, 375)
    )
    assert annotation["name"].tolist() == ["Car"]
    np.testing.assert_allclose(annotation["location"][0], camera[:3], atol=1e-9)

    corners = camera_box_corners(camera)
    projected = np.column_stack((corners, np.ones(8))) @ calibration.p2.T
    pixels = projected[:, :2] / projected[:, 2:3]
    expected = np.r_[pixels.min(axis=0), pixels.max(axis=0)]
    np.testing.assert_allclose(annotation["bbox"][0], expected, atol=1e-9)
    # Dropping the depth translation changes the bbox and must not be a workaround.
    wrong_pixels = projected[:, :2] / corners[:, 2:3]
    wrong_bbox = np.r_[wrong_pixels.min(axis=0), wrong_pixels.max(axis=0)]
    assert np.max(np.abs(expected - wrong_bbox)) > .01


@pytest.mark.parametrize("depth_offset", [0., 0.002616315, 0.004981016])
def test_file_parser_preserves_supported_depth_offsets(tmp_path, depth_offset):
    path = tmp_path / "calib.txt"
    path.write_text(CALIBRATION.replace("0.004981016", str(depth_offset)))
    assert read_calibration(path).p2[2, 3] == pytest.approx(depth_offset)
