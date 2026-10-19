"""SimAM routing, registry options and notebook validation share one contract."""
import copy
import json
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "detector")]
from core.models.backbones.mobilepixornext import MobilePixorNeXtBackbone
from core.models.backbones.registry import build_backbone
from tools.kitti_training_pipeline.notebook_config import validate_notebook_config
from tools.kitti_training_pipeline.common import generate_run_name

BASE = dict(backbone="mobilepixornext", backbone_out_dim=32, stage_depths=[3, 4, 2],
            c4_attention="none", c4_attention_scales=[], c4_attention_qk_norm="none")


@pytest.mark.parametrize("local", ["none", "simam"])
def test_hooks_route_refined_features_to_both_downstream_and_laterals(local):
    torch.set_num_threads(2)
    model = build_backbone("mobilepixornext", {**BASE, "local_attention": local}, 14).train()
    if local == "none":
        assert isinstance(model.c3_light_attention, nn.Identity)
    seen, handles = {}, []
    for name in ("c3_light_attention", "attention_c4_output"):
        handles.append(getattr(model, name).register_forward_hook(
            lambda module, args, output, name=name: seen.update({name: output})))
    for name in ("down3", "lat_c3", "down4", "lat_c4"):
        handles.append(getattr(model, name).register_forward_pre_hook(
            lambda module, args, name=name: seen.update({name: args[0]})))
    try:
        output = model(torch.randn(2, 14, 32, 48, requires_grad=True))
        assert seen["down3"] is seen["lat_c3"] is seen["c3_light_attention"]
        assert seen["down4"] is seen["lat_c4"] is seen["attention_c4_output"]
        assert output.shape == (2, 32, 8, 12)
        output.square().mean().backward()
        for parameter in model.c3_light_attention.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
    finally:
        for handle in handles:
            handle.remove()


def test_registry_threads_every_feature_option_without_mutating_config():
    config = {**BASE, "local_attention": "simam", "local_attention_stage": "c5",
              "local_attention_simam_lambda": .002, "local_attention_layer_scale_init": .003,
              "neck_fusion_channels": 32, "detail_path": True, "block_simam_stages": ["c4"]}
    before = copy.deepcopy(config)
    model = build_backbone("mobilepixornext", config, 14)
    assert config == before
    assert model.c3_light_attention.core.lambda_value == .002
    assert model.local_attention_stage == "c5" and model.block_simam_stages == ("c4",)
    torch.testing.assert_close(model.c3_light_attention.beta, torch.tensor(.003))
    assert model.neck_fusion_channels == 32 and model.detail_path


@pytest.mark.parametrize("patch", [
    {"c4_context_version": 1}, {"local_attention": "eca"},
    {"detail_path": True, "neck_type": "rc_sgfpn"},
    {"local_attention": "simam", "c4_attention_scales": [5]}, {"c4_context_typo": 1},
])
def test_registry_rejects_conflicting_and_removed_fields(patch):
    with pytest.raises(ValueError):
        build_backbone("mobilepixornext", {**BASE, **patch}, 14)


def test_registry_reads_legacy_disabled_context_without_creating_it():
    model = build_backbone("mobilepixornext", {**BASE, "c4_context": "none"}, 14)
    assert not hasattr(model, "c4_context")


def test_direct_constructor_and_registry_have_same_resolved_weights():
    options = {"local_attention": "simam", "local_attention_stage": "c4"}
    torch.manual_seed(721)
    direct = MobilePixorNeXtBackbone(input_channels=14,
        **{key: value for key, value in BASE.items() if key != "backbone"}, **options)
    torch.manual_seed(721)
    built = build_backbone("mobilepixornext", {**BASE, **options}, 14)
    assert list(direct.state_dict()) == list(built.state_dict())
    for key, value in direct.state_dict().items():
        assert torch.equal(value, built.state_dict()[key]), key


def test_notebook_resolves_simam_defaults_and_rejects_conflicts():
    config = json.loads((ROOT / "configs/config.json").read_text())
    config["model"].update(**BASE, local_attention="simam")
    validate_notebook_config(config)
    assert config["model"]["local_attention_simam_lambda"] == .0001
    assert config["model"]["local_attention_stage"] == "c3"
    conflict = copy.deepcopy(config)
    conflict["model"]["c4_attention_scales"] = [5]
    with pytest.raises(ValueError):
        validate_notebook_config(conflict)


def test_candidate_grid_check_occurs_before_hooks():
    model = build_backbone("mobilepixornext", {**BASE, "local_attention": "simam"}, 14)
    with pytest.raises(ValueError, match="divisible by 16"):
        model(torch.zeros(1, 14, 31, 48))


def test_supported_architectures_have_distinct_reproducible_run_names():
    config = json.loads((ROOT / "configs/config.json").read_text())
    variants = [{}, {"local_attention": "simam"},
                {"local_attention": "simam", "local_attention_stage": "c5"},
                {"detail_path": True}, {"neck_fusion_channels": 32},
                {"block_simam_stages": ["c4", "c5"]}]
    names = set()
    for patch in variants:
        candidate = copy.deepcopy(config)
        candidate["model"].update(BASE, **patch)
        name = generate_run_name(candidate)
        assert name == generate_run_name(copy.deepcopy(candidate))
        names.add(name)
    assert len(names) == len(variants)


@pytest.mark.parametrize("option", ["c4_context", "c4_context_version", "c4_context_bottleneck",
    "c4_context_dilations", "c4_context_layer_scale_init", "focal_stage",
    "attention_order", "local_attention_eca_kernel_size"])
def test_constructor_has_no_retired_module_controls(option):
    import inspect
    assert option not in inspect.signature(MobilePixorNeXtBackbone).parameters


def test_retired_implementations_are_absent():
    from core.models.backbones import light_attention
    assert not hasattr(light_attention, "ECACore")
    assert not (ROOT / "detector/core/models/backbones/focal_context.py").exists()
    assert not hasattr(MobilePixorNeXtBackbone(), "c4_context")
