import json
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "detector"))

from core.models.backbones.mobilepixornext_blocks import LiteMLARefinement
from core.models.backbones.registry import build_backbone


def test_multiscale_litemla_shape_and_gradient():
    module = LiteMLARefinement(
        channels=96,
        head_dim=16,
        scales=(3, 5),
        qk_norm="rmsnorm",
    )
    x = torch.randn(2, 96, 20, 16, requires_grad=True)

    output = module(x)
    output.mean().backward()

    assert output.shape == x.shape
    assert len(module.aggregations) == 2
    assert x.grad is not None
    assert torch.isfinite(output).all()
    assert torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


@pytest.mark.parametrize(
    ("mode", "expected_type"),
    [
        ("none", torch.nn.Identity),
        ("rmsnorm", torch.nn.RMSNorm),
        ("layernorm", torch.nn.LayerNorm),
    ],
)
def test_qk_norm_modes(mode, expected_type):
    module = LiteMLARefinement(
        channels=32,
        head_dim=8,
        scales=(3, 5),
        qk_norm=mode,
    )
    assert isinstance(module.query_norm, expected_type)
    assert isinstance(module.key_norm, expected_type)
    assert module.qk_norm_name == mode


def test_rmsnorm_uses_head_dimension_not_token_dimension():
    module = LiteMLARefinement(
        channels=32,
        head_dim=8,
        scales=(3,),
        qk_norm="rmsnorm",
        eps=1e-6,
    )
    query = torch.randn(2, 4, 8, 37)
    key = torch.randn(2, 4, 8, 37)

    normalized_query, normalized_key = module._normalize_qk(query, key)
    query_rms = normalized_query.square().mean(dim=2).sqrt()
    key_rms = normalized_key.square().mean(dim=2).sqrt()

    assert normalized_query.shape == query.shape
    assert normalized_key.shape == key.shape
    assert torch.allclose(query_rms, torch.ones_like(query_rms), atol=2e-4, rtol=2e-4)
    assert torch.allclose(key_rms, torch.ones_like(key_rms), atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize(
    "scales",
    [(), (2,), (3, 3), (3, 4), (3, True), "3,5"],
)
def test_invalid_scales_are_rejected(scales):
    with pytest.raises(ValueError, match="distinct odd integers"):
        LiteMLARefinement(channels=32, head_dim=8, scales=scales)


def test_invalid_qk_norm_is_rejected():
    with pytest.raises(ValueError, match="qk_norm"):
        LiteMLARefinement(
            channels=32,
            head_dim=8,
            scales=(3, 5),
            qk_norm="batchnorm",
        )


def test_legacy_defaults_keep_state_dict_compatible():
    reference = LiteMLARefinement(channels=32, head_dim=8)
    restored = LiteMLARefinement(channels=32, head_dim=8)

    restored.load_state_dict(reference.state_dict(), strict=True)

    assert reference.scales == (5,)
    assert reference.qk_norm_name == "none"
    assert len(reference.aggregations) == 1
    assert not any("query_norm" in key or "key_norm" in key for key in reference.state_dict())


def test_m2_config_wires_multiscale_rmsnorm():
    config_path = (
        ROOT
        / "configs/kitti/mobilepixornext_oga/kitti_mobilepixornext_ms_litemla_oga.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    backbone = build_backbone("mobilepixornext", config["model"], input_channels=8)

    assert backbone.c4_attention.scales == (3, 5)
    assert backbone.c4_attention.qk_norm_name == "rmsnorm"
    assert len(backbone.c4_attention.aggregations) == 2
    assert sum(parameter.numel() for parameter in backbone.parameters()) < 2_000_000


def test_scale_only_ablation_config_disables_qk_norm():
    config_path = (
        ROOT
        / "configs/kitti/mobilepixornext_oga/"
        "kitti_mobilepixornext_ms_litemla_no_norm_oga.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    backbone = build_backbone("mobilepixornext", config["model"], input_channels=8)

    assert backbone.c4_attention.scales == (3, 5)
    assert backbone.c4_attention.qk_norm_name == "none"
    assert isinstance(backbone.c4_attention.query_norm, torch.nn.Identity)


def test_multiscale_litemla_torchscript_matches_eager():
    torch.manual_seed(42)
    module = LiteMLARefinement(
        channels=32,
        head_dim=8,
        scales=(3, 5),
        qk_norm="rmsnorm",
    ).eval()
    x = torch.randn(1, 32, 12, 10)

    scripted = torch.jit.script(module)
    with torch.no_grad():
        expected = module(x)
        actual = scripted(x)

    assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="CUDA BF16 support is required",
)
def test_multiscale_litemla_bf16_extreme_values_are_finite():
    device = torch.device("cuda")
    module = LiteMLARefinement(
        channels=32,
        head_dim=8,
        scales=(3, 5),
        qk_norm="rmsnorm",
    ).to(device)
    values = torch.randn(2, 32, 20, 16, device=device)
    values[:, :, ::2, ::2] *= 1e3
    values[:, :, 1::2, 1::2] *= 1e-4
    x = values.requires_grad_(True)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        output = module(x)
        loss = output.square().mean()
    loss.backward()

    # The direct test input is FP32, so the residual addition promotes the
    # BF16 attention branch back to the input dtype. In the full autocast
    # backbone the incoming feature map may itself be BF16.
    assert output.dtype == x.dtype
    assert torch.isfinite(output).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
        if parameter.requires_grad
    )
