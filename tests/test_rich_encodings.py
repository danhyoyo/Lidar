import json
from pathlib import Path
import numpy as np
import pytest
import torch
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "detector"),
    str(ROOT / "detector" / "core" / "datasets"),
    str(ROOT / "tools" / "kitti_training_pipeline"),
]

from utils_1 import preprocess
from common import build_model, generate_run_name, input_shape, create_experiment_config


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
    expected_delta_z = (2.5 - 1.0) / 3.0  # 1.5 / 3.0 = 0.5
    np.testing.assert_allclose(bev[0, 0, 8], expected_delta_z, atol=1e-5)

    # sigma_z must be strictly positive
    assert bev[0, 0, 9] > 0.0


def test_encode_bev_rich11(sample_geometry):
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
        {"name": "rich11", "density_norm": 32.0, "intensity_scale": 1.0},
    )

    assert bev.shape == (4, 4, 11)
    assert bev.dtype == np.float32
    assert np.all(bev >= 0.0)
    assert np.all(bev <= 1.0)

    # Delta_i (ch 10) for cell (0, 0): i_max = 0.8, i_mean = 0.5 -> delta_i = 0.3
    np.testing.assert_allclose(bev[0, 0, 10], 0.3, atol=1e-5)


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


def test_run_name_generation_rich_encodings():
    with open(ROOT / "configs/config.json") as f:
        base = json.load(f)

    for name in ["rich8", "rich10", "rich11", "rich12"]:
        cfg = create_experiment_config(base, {"data": {"bev_encoding": {"name": name}}})
        run_name = generate_run_name(cfg, seed=42)
        assert name in run_name


def test_model_build_and_forward_rich_encodings():
    with open(ROOT / "configs/config.json") as f:
        base = json.load(f)

    for name, expected_ch in [
        ("rich8", 8),
        ("rich10", 10),
        ("rich11", 11),
        ("rich12", 12),
    ]:
        cfg = create_experiment_config(base, {"data": {"bev_encoding": {"name": name}}})
        shape = input_shape(cfg)
        assert shape[1] == expected_ch

        model = build_model(cfg)
        assert model.backbone.stem[0].in_channels == expected_ch
        dummy_x = torch.zeros((2, expected_ch, shape[2], shape[3]))
        out = model(dummy_x)
        assert "cls" in out
