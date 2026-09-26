import sys
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "detector"),
]

from core.models.backbones.bevnext_blocks import (
    BEVNeXtBlock,
    DownsampleBlock,
    LiteMLARefinement,
)


def test_bevnext_block_shape_and_grad():
    blk = BEVNeXtBlock(channels=64, expansion=2.5)
    x = torch.randn(2, 64, 50, 44, requires_grad=True)
    out = blk(x)
    assert out.shape == x.shape, f"Expected shape {x.shape}, got {out.shape}"
    out.sum().backward()
    assert x.grad is not None, "Gradient should propagate back to input x"
    assert not torch.isnan(x.grad).any(), "Gradient should not contain NaNs"


def test_bevnext_block_layer_scale():
    # When layer_scale is 0, output should equal input (pure identity)
    blk = BEVNeXtBlock(channels=32, layer_scale_init=0.0)
    x = torch.randn(1, 32, 20, 20)
    out = blk(x)
    assert torch.allclose(out, x), "Zero-initialized LayerScale must produce pure Identity"


def test_downsample_block():
    down = DownsampleBlock(in_channels=32, out_channels=64, stride=2)
    x = torch.randn(2, 32, 100, 88)
    out = down(x)
    assert out.shape == (2, 64, 50, 44), f"Expected (2, 64, 50, 44), got {out.shape}"


def test_litemla_block():
    attn = LiteMLARefinement(channels=96, head_dim=16, scales=(5,), layer_scale_init=0.01)
    x = torch.randn(2, 96, 50, 44, requires_grad=True)
    out = attn(x)
    assert out.shape == (2, 96, 50, 44), f"Expected (2, 96, 50, 44), got {out.shape}"
    out.sum().backward()
    assert x.grad is not None, "Gradient should propagate through LiteMLA"
    assert not torch.isnan(x.grad).any(), "Gradient should not contain NaNs"


def test_bevnext_backbone_forward_and_shapes():
    from core.models.backbones.bevnext import BEVNeXtBackbone

    backbone = BEVNeXtBackbone(input_channels=8, backbone_out_dim=16, c4_attention="litemla")
    x = torch.randn(2, 8, 800, 704, requires_grad=True)
    out = backbone(x)
    assert out.shape == (2, 16, 200, 176), f"Expected (2, 16, 200, 176), got {out.shape}"
    out.sum().backward()
    assert x.grad is not None, "Gradients should flow back to input"
    assert not torch.isnan(x.grad).any(), "Gradients must not be NaN"


def test_bevnext_backbone_parameter_budget():
    from core.models.backbones.bevnext import BEVNeXtBackbone

    backbone = BEVNeXtBackbone(input_channels=8, backbone_out_dim=16, c4_attention="litemla")
    total_params = sum(p.numel() for p in backbone.parameters())
    print(f"\nBEVNeXt Backbone Parameters: {total_params:,}")
    assert total_params < 2_000_000, f"Backbone must be < 2M params, got {total_params:,}"
    assert total_params > 500_000, f"Backbone should have enough capacity, got {total_params:,}"


def test_bevnext_backbone_attention_options():
    from core.models.backbones.bevnext import BEVNeXtBackbone

    bb_none = BEVNeXtBackbone(input_channels=8, c4_attention="none")
    bb_litemla = BEVNeXtBackbone(input_channels=8, c4_attention="litemla")
    params_none = sum(p.numel() for p in bb_none.parameters())
    params_litemla = sum(p.numel() for p in bb_litemla.parameters())
    assert params_litemla > params_none, "LiteMLA should add attention parameters"


def test_custom_model_bevnext():
    from core.models.model import CustomModel

    cfg = {
        "backbone": "bevnext",
        "cls_encoding": "gaussian",
        "backbone_out_dim": 16,
        "c4_attention": "litemla",
        "scale_gated_fpn": True,
    }
    model = CustomModel(cfg, num_classes=4, input_channels=8)
    total_model_params = sum(p.numel() for p in model.parameters())
    print(f"\nCustomModel (BEVNeXt + Header) Total Parameters: {total_model_params:,}")
    assert total_model_params < 2_000_000, f"Full model must be < 2M params, got {total_model_params:,}"

    x = torch.randn(2, 8, 800, 704)
    pred = model(x)
    assert isinstance(pred, dict)
    assert "cls" in pred and "offset" in pred and "size" in pred and "yaw" in pred
    assert pred["cls"].shape == (2, 4, 200, 176)
    assert pred["offset"].shape == (2, 2, 200, 176)
    assert pred["size"].shape == (2, 2, 200, 176)
    assert pred["yaw"].shape == (2, 2, 200, 176)


def test_detection_header_activations_and_bn():
    from core.models.heads.cnn import Head, Header

    # Modern head: BN + SiLU
    head_modern = Head(16, 4, use_bn=True, act="silu")
    assert isinstance(head_modern.bn1, torch.nn.BatchNorm2d)
    assert isinstance(head_modern.act1, torch.nn.SiLU)

    x = torch.randn(2, 16, 20, 20, requires_grad=True)
    out = head_modern(x)
    assert out.shape == (2, 4, 20, 20)
    out.sum().backward()
    assert x.grad is not None and not torch.isnan(x.grad).any()

    # Legacy head: no BN, identity act
    head_legacy = Head(16, 4, use_bn=False, act="none")
    assert isinstance(head_legacy.bn1, torch.nn.Identity)
    assert isinstance(head_legacy.act1, torch.nn.Identity)

    # Full header
    header = Header(num_classes=3, in_channels=16, use_bn=True, act="silu")
    feats = torch.randn(2, 16, 50, 44)
    pred = header(feats)
    assert set(pred.keys()) == {"cls", "offset", "size", "yaw"}
    assert pred["cls"].shape == (2, 3, 50, 44)
    assert pred["offset"].shape == (2, 2, 50, 44)
    assert pred["size"].shape == (2, 2, 50, 44)
    assert pred["yaw"].shape == (2, 2, 50, 44)



