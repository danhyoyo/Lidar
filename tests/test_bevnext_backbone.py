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

