"""Frozen target behavior, IQA and adaptive loss state."""

import copy
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "detector"), str(ROOT / "detector/core/datasets")]

from core.datasets.dataset import Dataset
from core.losses.strategies.oga import OgaLossStrategy

GEOMETRY = {
    "x_min": 0.0, "x_max": 4.8, "x_res": 0.1,
    "y_min": -1.6, "y_max": 1.6, "y_res": 0.1,
    "z_min": -2.5, "z_max": 1.0, "z_res": 0.1,
}
FIXTURE = ROOT / "tests/fixtures/legacy_detector_contract.json"


def legacy_targets(cls_encoding="gaussian"):
    dataset = Dataset.__new__(Dataset)
    dataset.output_shape = [12, 8]  # Internal XY; public tensors are YX.
    dataset.out_size_factor = 4
    dataset.num_classes = 3
    dataset.cls_encoding = cls_encoding
    dataset.target_backend = "python"
    dataset.config = {"gaussian_overlap": 0.1, "min_radius": 2,
                      "regression_assignment": "nearest_center"}
    boxes = torch.tensor([
        [0, 1.5, 1.6, 3.5, 1.05, -0.35, -1.7, 0.2],
        [1, 1.7, 0.6, 0.8, 2.25, 0.45, -1.7, -0.4],
        [2, 1.6, 0.7, 1.2, 3.45, 0.05, -1.7, 0.0],
    ], dtype=torch.float32)
    return {key: value.unsqueeze(0) for key, value in dataset.get_label(boxes, GEOMETRY).items()}


def assert_snapshot_close(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_snapshot_close(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for current, frozen in zip(actual, expected):
            assert_snapshot_close(current, frozen)
    elif isinstance(expected, float):
        assert actual == pytest.approx(expected, rel=2e-5, abs=1e-7)
    else:
        assert actual == expected


def test_binary_target_values_and_public_axis_order():
    frozen = json.loads(FIXTURE.read_text())["binary_targets"]
    assert_snapshot_close({k: v.tolist() for k, v in legacy_targets("binary").items()}, frozen)


def test_adaptive_iqa_state_restore_and_eval_do_not_update_ema():
    targets = legacy_targets()
    predictions = {key: value.float().clone().requires_grad_()
                   for key, value in targets.items() if key != "reg_mask"}
    predictions["iou"] = torch.zeros(1, 1, 8, 12, requires_grad=True)
    criterion = OgaLossStrategy("gaussian", {"use_iou": True})
    criterion(predictions, targets)
    state = copy.deepcopy(criterion.state_dict())
    restored = OgaLossStrategy("gaussian", {"use_iou": True}).eval()
    restored.load_state_dict(state, strict=True)
    criterion.eval()
    first, second = criterion(predictions, targets), restored(predictions, targets)
    torch.testing.assert_close(first["loss"], second["loss"], rtol=0, atol=0)
    for key in state:
        torch.testing.assert_close(restored.state_dict()[key], state[key], rtol=0, atol=0)
    second["loss"].backward()
    assert torch.isfinite(predictions["iou"].grad).all()
    assert predictions["iou"].grad.abs().sum() > 0


def test_frozen_gaussian_targets():
    frozen = json.loads(FIXTURE.read_text())["gaussian_targets"]
    assert_snapshot_close({k: v.tolist() for k, v in legacy_targets().items()}, frozen)
