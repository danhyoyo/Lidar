"""RC additive gating, configuration routing and deployment compatibility."""

import copy
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "detector"), str(ROOT / "tools/kitti_training_pipeline")]

from core.models.backbones.rc_sgfpn import RangeConditionedScaleGate
from common import (build_model, checkpoint_identity, generate_run_name, read_json,
                    validate_evaluation_checkpoint)


@pytest.mark.parametrize("mode", ["mul", "add"])
def test_gate_modes_start_as_sum_and_learn_range(mode):
    gate = RangeConditionedScaleGate(3, 8, 12, gate_mode=mode)
    lateral = torch.randn(2, 3, 8, 12)
    upper = torch.randn_like(lateral)
    torch.testing.assert_close(gate(lateral, upper), lateral + upper, rtol=0, atol=0)
    with torch.no_grad():
        gate.content_conv.weight.fill_(0.1)
    gate(lateral, upper).square().mean().backward()
    assert gate.range_proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("mode", ["mul", "add"])
def test_gate_uses_selected_range_formula(mode):
    gate = RangeConditionedScaleGate(2, 8, 12, gate_mode=mode)
    with torch.no_grad():
        gate.content_conv.bias.fill_(-0.5)
        gate.range_proj.bias.fill_(1.0)
    lateral = torch.ones(1, 2, 8, 12)
    upper = torch.zeros_like(lateral)
    content = torch.full_like(lateral, -0.5)
    prior = torch.ones_like(lateral)
    logits = content + prior if mode == "add" else content * (1 + 0.5 * prior.tanh())
    torch.testing.assert_close(gate(lateral, upper), 2 * logits.sigmoid())


@pytest.mark.parametrize("mode", ["mul", "add"])
def test_trained_gate_deploy_roundtrip_preserves_outputs(mode):
    gate = RangeConditionedScaleGate(3, 8, 12, gate_mode=mode)
    with torch.no_grad():
        gate.content_conv.weight.normal_(0, 0.1)
        gate.content_conv.bias.fill_(0.2)
        gate.range_proj.weight.normal_(0, 0.1)
        gate.range_proj.bias.fill_(0.3)
    lateral = torch.randn(2, 3, 8, 12)
    upper = torch.randn_like(lateral)
    expected = gate(lateral, upper)
    gate.switch_to_deploy()
    gate.switch_to_deploy()
    buffer = "static_spatial_bias" if mode == "add" else "static_spatial_scale"
    assert hasattr(gate, buffer)
    assert not hasattr(gate, "range_proj")
    torch.testing.assert_close(gate(lateral, upper), expected)
    restored = RangeConditionedScaleGate(3, 8, 12, gate_mode=mode, deploy=True)
    restored.load_state_dict(gate.state_dict(), strict=True)
    torch.testing.assert_close(restored(lateral, upper), expected)


def test_omitted_mode_preserves_multiplicative_initialization_and_output():
    torch.manual_seed(42)
    implicit = RangeConditionedScaleGate(3, 8, 12)
    torch.manual_seed(42)
    explicit = RangeConditionedScaleGate(3, 8, 12, gate_mode="mul")
    assert implicit.state_dict().keys() == explicit.state_dict().keys()
    for key, value in implicit.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[key])
    lateral = torch.randn(1, 3, 8, 12)
    upper = torch.randn_like(lateral)
    torch.testing.assert_close(implicit(lateral, upper), explicit(lateral, upper), rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["invalid", None, 1])
def test_invalid_gate_mode_is_rejected(mode):
    with pytest.raises(ValueError, match="gate_mode"):
        RangeConditionedScaleGate(3, 8, 12, gate_mode=mode)


def rc_config(mode=None):
    config = read_json(ROOT / "configs/config.json")
    config["model"]["neck_type"] = "rc_sgfpn"
    if mode is not None:
        config["model"]["rc_gate_mode"] = mode
    return config


def test_model_config_routes_add_mode_and_distinguishes_checkpoint_identity():
    additive = rc_config("add")
    model = build_model(additive)
    assert model.backbone.rc_neck.gate_td3.gate_mode == "add"
    assert model.backbone.rc_neck.gate_td4.gate_mode == "add"
    legacy, multiplicative = rc_config(), rc_config("mul")
    assert checkpoint_identity(legacy) == checkpoint_identity(multiplicative)
    assert checkpoint_identity(additive) != checkpoint_identity(multiplicative)
    assert generate_run_name(legacy) == generate_run_name(multiplicative)
    assert generate_run_name(additive) != generate_run_name(multiplicative)


def test_additive_mode_cannot_be_silently_ignored_by_standard_neck():
    config = rc_config("add")
    config["model"]["neck_type"] = "scale_gated_fpn"
    with pytest.raises(ValueError, match="rc_gate_mode"):
        build_model(config)


@pytest.mark.parametrize("neck", [None, "sgfpn"])
def test_existing_standard_neck_aliases_still_build(neck):
    config = rc_config("mul")
    config["model"]["neck_type"] = neck
    assert build_model(config).backbone.rc_neck is None


def test_checkpoint_validation_rejects_mode_changes_and_accepts_legacy_mul():
    legacy = rc_config()
    checkpoint = {"checkpoint_identity": checkpoint_identity(legacy), "config": legacy}
    validate_evaluation_checkpoint(checkpoint, rc_config("mul"))
    with pytest.raises(ValueError, match="identity"):
        validate_evaluation_checkpoint(checkpoint, rc_config("add"))


def test_default_mode_keeps_non_rc_backbone_checkpoint_identity():
    config = read_json(ROOT / "configs/config.json")
    config["model"]["backbone"] = "mobilepixor"
    expected = checkpoint_identity(config)
    config["model"]["rc_gate_mode"] = "mul"
    assert checkpoint_identity(config) == expected


def test_notebook_keeps_trained_legacy_snapshot_for_explicit_mul(tmp_path):
    from common import write_json
    from notebook_config import save_notebook_config
    legacy = rc_config()
    path = save_notebook_config(tmp_path, legacy)
    original = path.read_bytes()
    write_json(tmp_path / "run.json", {"seed": 42})
    assert save_notebook_config(tmp_path, rc_config("mul")) == path
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="config"):
        save_notebook_config(tmp_path, rc_config("add"))
    assert path.read_bytes() == original


def test_additive_detector_deploy_roundtrip_with_trained_range_weights():
    config = rc_config("add")
    model = build_model(config).eval()
    with torch.no_grad():
        for gate in (model.backbone.rc_neck.gate_td3, model.backbone.rc_neck.gate_td4):
            gate.content_conv.weight.normal_(0, 0.02)
            gate.range_proj.weight.normal_(0, 0.1)
    voxel = torch.randn(1, 8, 800, 704)
    with torch.inference_mode():
        expected = model({"voxel": voxel})
        model.switch_to_deploy()
        config = copy.deepcopy(config)
        config["model"]["deploy"] = True
        restored = build_model(config).eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        actual = restored({"voxel": voxel})
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("mode", ["mul", "add"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gate_modes_preserve_half_precision_during_deploy(mode, dtype):
    gate = RangeConditionedScaleGate(3, 8, 12, gate_mode=mode).to(dtype)
    with torch.no_grad():
        gate.content_conv.weight.normal_(0, 0.02)
        gate.range_proj.weight.normal_(0, 0.1)
    lateral = torch.randn(1, 3, 8, 12, dtype=dtype)
    upper = torch.randn_like(lateral)
    expected = gate(lateral, upper)
    gate.switch_to_deploy()
    actual = gate(lateral, upper)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
