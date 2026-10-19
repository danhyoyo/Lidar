"""Numerical routing, gradients and parameter-free checkpoint semantics."""

import copy
from itertools import combinations
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "detector"), str(ROOT / "tools/kitti_training_pipeline")]
from core.models.backbones.light_attention import SimAMCore
from core.models.backbones.mobilepixornext_blocks import MobilePixorNeXtBlock
from core.models.backbones.mobilepixornext import MobilePixorNeXtBackbone
from core.models.backbones.registry import build_backbone

STAGES = ("c3", "c4", "c5")
SUBSETS = [list(stages) for size in range(4) for stages in combinations(STAGES, size)]
BASE = dict(backbone="mobilepixornext", backbone_out_dim=32, stage_depths=[3, 4, 2])


@pytest.mark.parametrize("stages", SUBSETS)
def test_only_selected_stages_apply_simam_once_inside_every_block(stages):
    torch.set_num_threads(2)
    model = build_backbone("mobilepixornext", {**BASE, "block_simam_stages": stages}, 14).eval()
    calls, handles = [], []
    for stage, name in zip(STAGES, ("stage2", "stage3", "stage4")):
        for index, block in enumerate(getattr(model, name)):
            assert isinstance(block.simam, SimAMCore if stage in stages else nn.Identity)
            if stage in stages:
                handles.append(block.simam.register_forward_hook(
                    lambda module, args, output, key=(stage, index): calls.append(key)))
    try:
        with torch.inference_mode():
            result = model(torch.randn(1, 14, 32, 48))
        assert result.shape == (1, 32, 8, 12)
        assert calls == [(stage, index) for stage, depth in zip(STAGES, (3, 4, 2))
                         if stage in stages for index in range(depth)]
    finally:
        for handle in handles:
            handle.remove()


def test_block_attention_operates_on_projection_bn_before_layer_scale_and_shortcut():
    block = MobilePixorNeXtBlock(8, layer_scale_init=.7, simam_lambda=.002).eval()
    x = torch.randn(2, 8, 4, 6, requires_grad=True)
    branch = block.norm1(block.dwconv(x))
    branch = block.norm2(block.pw_project(block.act(block.pw_expand(branch))))
    expected = x + block.layer_scale * SimAMCore(.002)(branch)
    result = block(x)
    torch.testing.assert_close(result, expected)
    result.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert block.pw_project.weight.grad.abs().sum() > 0
    with torch.no_grad():
        block.layer_scale.zero_()
    assert torch.equal(block(x), x)


def test_disabled_attention_preserves_seeded_parameters_state_and_output():
    torch.manual_seed(51)
    original = MobilePixorNeXtBackbone(input_channels=14, **{k: v for k, v in BASE.items() if k != "backbone"}).eval()
    torch.manual_seed(51)
    disabled = build_backbone("mobilepixornext", {**BASE, "block_simam_stages": []}, 14).eval()
    torch.manual_seed(51)
    enabled = build_backbone("mobilepixornext", {**BASE, "block_simam_stages": ["c4", "c5"]}, 14).eval()
    assert list(original.state_dict()) == list(disabled.state_dict()) == list(enabled.state_dict())
    for key, value in original.state_dict().items():
        assert torch.equal(value, disabled.state_dict()[key])
        assert torch.equal(value, enabled.state_dict()[key])
    assert sum(p.numel() for p in original.parameters()) == sum(p.numel() for p in enabled.parameters())
    x = torch.randn(1, 14, 32, 48)
    with torch.inference_mode():
        assert torch.equal(original(x), disabled(x))


@pytest.mark.parametrize("stages", [["c4", "c5"], list(STAGES)])
def test_full_kitti_output_shape_and_torchscript(stages):
    with torch.device("meta"):
        full = build_backbone("mobilepixornext", {**BASE, "block_simam_stages": stages}, 14).eval()
        assert full(torch.zeros(1, 14, 800, 704)).shape == (1, 32, 200, 176)
    small = build_backbone("mobilepixornext", {**BASE, "block_simam_stages": stages}, 14).eval()
    scripted = torch.jit.script(small)
    x = torch.randn(1, 14, 32, 48)
    with torch.inference_mode():
        torch.testing.assert_close(scripted(x), small(x))


def test_real_grouped_loss_backward_checkpoint_roundtrip_and_mismatch_rejection(tmp_path):
    from attention_stage_test_utils import configured
    from common import checkpoint_identity, generate_run_name, validate_evaluation_checkpoint
    from test_grouped_checkpoint import components, payload, restore
    from test_grouped_loss_state import snapshot, assert_state_equal

    config, target = configured((None, None, None))
    config["model"].update(block_simam_stages=["c4", "c5"], header_architecture="shared")
    source = components(config)
    result = source[1](source[0](target["voxel"]), target)["loss"]
    assert torch.isfinite(result)
    result.backward()
    for stage in (source[0].backbone.stage3, source[0].backbone.stage4):
        for block in stage:
            assert block.pw_project.weight.grad is not None
            assert torch.isfinite(block.pw_project.weight.grad).all()
    source[2].step()
    source[3].step()
    saved = payload(config, source)
    path = tmp_path / "state.pt"
    torch.save(saved, path)
    saved = torch.load(path, weights_only=True)
    destination = components(config)
    assert restore(saved, config, destination)["mode"] == "resume"
    assert_state_equal(destination[0], snapshot(source[0]))
    assert_state_equal(destination[1], snapshot(source[1]))
    torch.testing.assert_close(destination[2].state_dict(), source[2].state_dict(), rtol=0, atol=0)
    assert destination[3].state_dict() == source[3].state_dict()
    assert "blocksimam_c4c5" in generate_run_name(config)
    for patch in ({"block_simam_stages": ["c5"]}, {"block_simam_lambda": .002}):
        changed = copy.deepcopy(config)
        changed["model"].update(patch)
        assert checkpoint_identity(changed) != checkpoint_identity(config)
        assert generate_run_name(changed) != generate_run_name(config)
        with pytest.raises(ValueError, match="identity"):
            validate_evaluation_checkpoint(saved, changed)
        target_parts = components(changed)
        before = snapshot(target_parts[0])
        with pytest.raises(ValueError, match="identity"):
            restore(saved, changed, target_parts)
        assert_state_equal(target_parts[0], before)


def test_block_simam_coexists_with_stage_attention_and_repconv_deploy():
    model = build_backbone("mobilepixornext", {
        **BASE, "block_simam_stages": ["c4", "c5"], "use_reparam": True,
        "local_attention": "simam", "local_attention_stage": "c5",
    }, 14).eval()
    assert sum(isinstance(module, SimAMCore) for module in model.modules()) == 7
    x = torch.randn(1, 14, 32, 48, requires_grad=True)
    before = model(x)
    before.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert model.c3_light_attention.beta.grad is not None
    for stage in (model.stage2, model.stage3, model.stage4):
        for block in stage:
            block.switch_to_deploy()
    with torch.inference_mode():
        torch.testing.assert_close(model(x.detach()), before.detach(), rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_block_simam_amp_forward_backward_is_finite(dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA BF16 unsupported")
    model = build_backbone("mobilepixornext", {**BASE, "block_simam_stages": ["c4", "c5"]}, 14).cuda().train()
    x = torch.randn(2, 14, 32, 48, device="cuda", requires_grad=True)
    with torch.autocast("cuda", dtype=dtype):
        output = model(x)
        loss = output.float().square().mean()
    loss.backward()
    assert torch.isfinite(output).all() and torch.isfinite(x.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
