import json
from pathlib import Path
import numpy as np
import pytest
import torch

from core.datasets.utils_1 import preprocess
from core.models.model import CustomModel
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "kitti_training_pipeline"))
from common import build_model, generate_run_name, input_shape


@pytest.fixture
def sample_geometry():
    return {
        "x_min": 0.0,
        "x_max": 4.0,
        "x_res": 1.0,
        "y_min": -2.0,
        "y_max": 2.0,
        "y_res": 1.0,
        "z_min": -2.0,
        "z_max": 1.0,
        "z_res": 1.0,
    }


def test_encode_bev_rich10(sample_geometry):
    # Points in cell (0, 0) in grid: x in [0, 1), y in [-2, -1)
    points = np.array(
        [
            # Cell (0, 0): x=0.5, y=-1.5, multiple heights and intensities
            [0.5, -1.5, -1.0, 0.2],
            [0.5, -1.5, 0.0, 0.8],
            [0.5, -1.5, 0.5, 0.5],
            # Cell (1, 1): x=1.5, y=-0.5, single point
            [1.5, -0.5, -0.5, 0.4],
        ],
        dtype=np.float32,
    )

    bev = preprocess.encode_bev(
        points,
        sample_geometry,
        {"name": "rich10", "density_norm": 32.0, "intensity_scale": 1.0},
    )

    # Grid size: x=4, y=4
    assert bev.shape == (4, 4, 10)
    assert bev.dtype == np.float32

    # All values must be bounded in [0, 1]
    assert np.all(bev >= 0.0)
    assert np.all(bev <= 1.0)

    # Cell (0, 1) has no points -> all zeros
    np.testing.assert_array_equal(bev[0, 1], np.zeros(10, dtype=np.float32))

    # Cell (1, 1) has exactly 1 point:
    # delta_z (ch 8) and sigma_z (ch 9) must be 0.0
    assert bev[1, 1, 8] == 0.0
    assert bev[1, 1, 9] == 0.0

    # Cell (0, 0) has 3 points with different z:
    # z_norm values:
    # z_min_norm = (-1.0 - (-2.0)) / 3.0 = 1.0 / 3.0
    # z_max_norm = (0.5 - (-2.0)) / 3.0 = 2.5 / 3.0
    expected_delta_z = (2.5 - 1.0) / 3.0  # 1.5 / 3.0 = 0.5
    np.testing.assert_allclose(bev[0, 0, 8], expected_delta_z, atol=1e-5)

    # sigma_z must be strictly positive
    assert bev[0, 0, 9] > 0.0


def test_encode_bev_rich12(sample_geometry):
    points = np.array(
        [
            [0.5, -1.5, -1.0, 0.2],
            [0.5, -1.5, 0.5, 0.8],
            [1.5, -0.5, 0.0, 0.4],
        ],
        dtype=np.float32,
    )

    bev = preprocess.encode_bev(
        points,
        sample_geometry,
        {"name": "rich12", "density_norm": 32.0, "intensity_scale": 1.0},
    )

    assert bev.shape == (4, 4, 12)
    assert bev.dtype == np.float32
    assert np.all(bev >= 0.0)
    assert np.all(bev <= 1.0)

    # Delta_i (ch 10) for cell (0, 0): i_max = 0.8, i_mean = 0.5 -> delta_i = 0.3
    np.testing.assert_allclose(bev[0, 0, 10], 0.3, atol=1e-5)

    # Range-compensated density (ch 11) must be strictly positive for occupied cell
    assert bev[0, 0, 11] > 0.0
    # Empty cell must have 0
    assert bev[0, 1, 11] == 0.0


def test_run_name_generation_rich10_and_rich12():
    with open(ROOT / "configs/kitti/rich10/kitti_mobilepixornext_rich10_sgfpn.json") as f:
        cfg10 = json.load(f)
    name10 = generate_run_name(cfg10, seed=42)
    assert "rich10" in name10
    assert name10 == "mobilepixornext-standard_aug-baseline_loss-rich10-baseline_iou-sgfpn-s42"

    with open(ROOT / "configs/kitti/rich12/kitti_mobilepixornext_rich12_sgfpn.json") as f:
        cfg12 = json.load(f)
    name12 = generate_run_name(cfg12, seed=42)
    assert "rich12" in name12
    assert name12 == "mobilepixornext-standard_aug-baseline_loss-rich12-baseline_iou-sgfpn-s42"


def test_model_build_and_forward_rich10_and_rich12():
    with open(ROOT / "configs/kitti/rich10/kitti_mobilepixornext_rich10_sgfpn.json") as f:
        cfg10 = json.load(f)
    shape10 = input_shape(cfg10)
    assert shape10[1] == 10

    model10 = build_model(cfg10)
    assert model10.backbone.stem[0].in_channels == 10
    dummy_x10 = torch.zeros((2, 10, shape10[2], shape10[3]))
    out10 = model10(dummy_x10)
    assert "cls" in out10

    with open(ROOT / "configs/kitti/rich12/kitti_mobilepixornext_rich12_sgfpn.json") as f:
        cfg12 = json.load(f)
    shape12 = input_shape(cfg12)
    assert shape12[1] == 12

    model12 = build_model(cfg12)
    assert model12.backbone.stem[0].in_channels == 12
    dummy_x12 = torch.zeros((2, 12, shape12[2], shape12[3]))
    out12 = model12(dummy_x12)
    assert "cls" in out12
