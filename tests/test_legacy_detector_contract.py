"""Frozen pre-schema behavior: targets, predictions, IQA and adaptive loss state."""

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "detector"), str(ROOT / "detector/core/datasets")]

from core.datasets.dataset import Dataset
from core.losses.strategies.oga import OgaLossStrategy
from core.models.model import CustomModel

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


def tensor_summary(value):
    flat = value.detach().double().flatten()
    indices = torch.linspace(0, flat.numel() - 1, min(16, flat.numel())).long()
    return {"shape": list(value.shape), "sum": flat.sum().item(),
            "squared_sum": flat.square().sum().item(), "samples": flat[indices].tolist()}


def capture_legacy_contract(use_iou):
    """Used once to capture the fixture before production changes, never in assertions."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261007)
        cfg = {"backbone": "mobilepixornext", "backbone_out_dim": 16,
               "scale_gated_fpn": True, "c4_attention": "litemla",
               "c4_attention_scales": [5], "cls_encoding": "gaussian",
               "header_use_iou": use_iou, "bev_encoding": {"name": "rich8"}}
        model = CustomModel(cfg, num_classes=3, input_channels=35).eval()
        inputs = torch.linspace(-0.5, 0.5, 8 * 32 * 48).reshape(1, 8, 32, 48)
        predictions = model({"voxel": inputs})
        targets = legacy_targets()
        criterion = OgaLossStrategy("gaussian", {"use_iou": use_iou})
        first = criterion(predictions, targets)
        initial_ema = criterion.weighting.running_loss_means.detach().clone()
        perturbed = {key: value + 0.125 for key, value in predictions.items()}
        second = criterion(perturbed, targets)
        second["loss"].backward()
        for gradient in (model.backbone.stem[0].weight.grad, criterion.weighting.log_scales.grad):
            assert gradient is not None and torch.isfinite(gradient).all()
            assert gradient.abs().sum() > 0
        state = [(key, list(value.shape)) for key, value in model.state_dict().items()]
        result = {
            "state_keys_shapes_sha256": hashlib.sha256(json.dumps(state).encode()).hexdigest(),
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "predictions": {key: tensor_summary(value) for key, value in predictions.items()},
            "targets": {key: value.tolist() for key, value in targets.items()},
            "first_loss": {key: float(value.detach()) if torch.is_tensor(value) else float(value)
                           for key, value in first.items()},
            "second_loss": {key: float(value.detach()) if torch.is_tensor(value) else float(value)
                            for key, value in second.items()},
            "initial_ema": initial_ema.tolist(),
            "second_ema": criterion.weighting.running_loss_means.tolist(),
            "criterion_state_keys": list(criterion.state_dict()),
            "gradients": {"stem": tensor_summary(model.backbone.stem[0].weight.grad),
                          "log_scales": tensor_summary(criterion.weighting.log_scales.grad)},
        }
        for key, value in predictions.items():
            assert torch.isfinite(value).all(), key
        assert criterion.weighting.initialized.item()
        return result


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


@pytest.mark.parametrize("use_iou", [False, True])
def test_frozen_legacy_detector_behavior(use_iou):
    frozen = json.loads(FIXTURE.read_text())[str(use_iou)]
    assert_snapshot_close(capture_legacy_contract(use_iou), frozen)


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
