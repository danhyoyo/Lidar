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
