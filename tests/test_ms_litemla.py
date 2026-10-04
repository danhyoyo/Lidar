import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "detector"),
    str(ROOT / "detector" / "core" / "datasets"),
    str(ROOT / "tools" / "kitti_training_pipeline"),
]

from core.models.backbones.mobilepixornext_blocks import LiteMLARefinement


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

    assert output.dtype == x.dtype
    assert torch.isfinite(output).all()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


from core.models.backbones.mobilepixornext import MobilePixorNeXtBackbone
from core.models.backbones.registry import build_backbone


def test_backbone_wires_multiscale_and_reparam_together():
    backbone = MobilePixorNeXtBackbone(
        input_channels=8,
        c4_attention="litemla",
        c4_attention_scales=(3, 5),
        c4_attention_qk_norm="rmsnorm",
        use_reparam=True,
    )
    assert backbone.c4_attention.scales == (3, 5)
    assert backbone.c4_attention.qk_norm_name == "rmsnorm"
    assert backbone.use_reparam is True
    assert hasattr(backbone.stage2[0], "dw_block")

    # Verify forward pass
    x = torch.randn(2, 8, 800, 704)
    out = backbone(x)
    assert out.shape == (2, 16, 200, 176)

    # Verify switch_to_deploy recursion
    backbone.eval()
    backbone.switch_to_deploy()
    out_deploy = backbone(x)
    assert out_deploy.shape == (2, 16, 200, 176)


def test_registry_builds_mobilepixornext_with_m2_options():
    cfg = {
        "c4_attention_scales": [3, 5],
        "c4_attention_qk_norm": "rmsnorm",
        "use_reparam": True,
    }
    model = build_backbone("mobilepixornext", cfg, input_channels=8)
    assert model.c4_attention.scales == (3, 5)
    assert model.c4_attention.qk_norm_name == "rmsnorm"
    assert model.use_reparam is True


def test_m2_config_wires_multiscale_rmsnorm():
    from config_helpers import oga_config
    config = oga_config(multiscale=True, qk_norm='rmsnorm')
    backbone = build_backbone("mobilepixornext", config["model"], input_channels=8)

    assert backbone.c4_attention.scales == (3, 5)
    assert backbone.c4_attention.qk_norm_name == "rmsnorm"
    assert len(backbone.c4_attention.aggregations) == 2
    assert sum(parameter.numel() for parameter in backbone.parameters()) < 2_000_000


def test_scale_only_ablation_config_disables_qk_norm():
    from config_helpers import oga_config
    config = oga_config(multiscale=True)
    backbone = build_backbone("mobilepixornext", config["model"], input_channels=8)

    assert backbone.c4_attention.scales == (3, 5)
    assert backbone.c4_attention.qk_norm_name == "none"
    assert isinstance(backbone.c4_attention.query_norm, torch.nn.Identity)


def test_cumulative_m1_m2_m4_config_instantiates_full_model():
    from core.models.model import CustomModel
    from core.losses.strategies.oga import OgaLossStrategy

    from config_helpers import oga_config
    config = oga_config(iqa=True, reparam=True, multiscale=True, qk_norm='rmsnorm')
    model = CustomModel(config["model"], input_channels=8)

    # 1. Verify M1 (use_reparam)
    assert model.backbone.use_reparam is True

    # 2. Verify M2 (scales=(3, 5), rmsnorm)
    assert model.backbone.c4_attention.scales == (3, 5)
    assert model.backbone.c4_attention.qk_norm_name == "rmsnorm"

    # 3. Verify M4 (Header with 5 branches including IoU)
    assert model.header.use_iou is True
    assert hasattr(model.header, "iou")

    # 4. Verify Parameter budget
    train_params = sum(p.numel() for p in model.parameters())
    assert train_params < 2_000_000

    # 5. Forward pass
    x = torch.randn(2, 8, 800, 704)
    preds = model(x)
    assert "cls" in preds and "offset" in preds and "size" in preds and "yaw" in preds and "iou" in preds
    assert preds["iou"].shape == (2, 1, 200, 176)

    # 6. Verify switch_to_deploy
    model.eval()
    model.switch_to_deploy()
    deploy_params = sum(p.numel() for p in model.parameters())
    assert deploy_params < train_params

    # 7. Verify Loss Strategy compatibility
    loss_strategy = OgaLossStrategy(
        cls_encoding=config["model"].get("cls_encoding", "gaussian"),
        config=config["loss"],
    )
    assert loss_strategy.use_iou is True
    assert len(loss_strategy.TASKS) == 6
    assert "iou" in loss_strategy.TASKS
    assert loss_strategy.weighting.num_tasks == 6
