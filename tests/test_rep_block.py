import sys
from pathlib import Path
from typing import Tuple
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [
    str(ROOT / "detector"),
]

from core.models.backbones.rep_blocks import (
    RepConv7x7,
    trans_conv_bn_to_kernel_bias,
    trans_identity_bn_to_kernel_bias,
)


def test_trans_conv_bn_fusion_math():
    channels = 8
    conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
    bn = nn.BatchNorm2d(channels)
    conv.eval()
    bn.eval()

    x = torch.randn(2, channels, 16, 16)
    expected = bn(conv(x))

    kernel, bias = trans_conv_bn_to_kernel_bias(conv, bn)
    fused_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True)
    fused_conv.weight.data.copy_(kernel)
    fused_conv.bias.data.copy_(bias)
    actual = fused_conv(x)

    assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-4)


def test_trans_identity_bn_fusion_math():
    channels = 8
    bn = nn.BatchNorm2d(channels)
    bn.eval()

    x = torch.randn(2, channels, 16, 16)
    expected = bn(x)

    kernel, bias = trans_identity_bn_to_kernel_bias(bn, channels=channels, kernel_size=7)
    fused_conv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels, bias=True)
    fused_conv.weight.data.copy_(kernel)
    fused_conv.bias.data.copy_(bias)
    actual = fused_conv(x)

    assert torch.allclose(expected, actual, atol=1e-5, rtol=1e-4)


def test_repconv7x7_numerical_equivalence():
    torch.manual_seed(42)
    channels = 48
    rep = RepConv7x7(channels=channels, deploy=False)
    rep.eval()

    x = torch.randn(2, channels, 50, 44)
    with torch.no_grad():
        out_multi = rep(x)

    assert not rep.deploy
    assert hasattr(rep, "rbr_conv7")
    assert hasattr(rep, "rbr_conv3")
    assert hasattr(rep, "rbr_identity")

    rep.switch_to_deploy()

    assert rep.deploy
    assert hasattr(rep, "rbr_reparam")
    assert not hasattr(rep, "rbr_conv7")
    assert not hasattr(rep, "rbr_conv3")
    assert not hasattr(rep, "rbr_identity")

    with torch.no_grad():
        out_fused = rep(x)

    diff = torch.max(torch.abs(out_multi - out_fused)).item()
    assert diff < 1e-5, f"Max difference {diff} exceeds tolerance 1e-5"


def test_repconv7x7_gradient_flow():
    channels = 16
    rep = RepConv7x7(channels=channels, deploy=False)
    rep.train()

    x = torch.randn(2, channels, 20, 20, requires_grad=True)
    out = rep(x)
    loss = out.sum()
    loss.backward()

    assert rep.rbr_conv7[0].weight.grad is not None
    assert rep.rbr_conv7[1].weight.grad is not None
    assert rep.rbr_conv3[0].weight.grad is not None
    assert rep.rbr_conv3[1].weight.grad is not None
    assert rep.rbr_identity.weight.grad is not None
    assert x.grad is not None


def test_mobilepixornext_block_backward_compat():
    from core.models.backbones.mobilepixornext_blocks import MobilePixorNeXtBlock

    torch.manual_seed(42)
    block_default = MobilePixorNeXtBlock(48)
    assert not getattr(block_default, "use_reparam", False)
    assert hasattr(block_default, "dwconv")
    assert hasattr(block_default, "norm1")

    x = torch.randn(2, 48, 20, 20)
    out = block_default(x)
    assert out.shape == (2, 48, 20, 20)


def test_mobilepixornext_block_reparam_equivalence():
    from core.models.backbones.mobilepixornext_blocks import MobilePixorNeXtBlock

    torch.manual_seed(42)
    block = MobilePixorNeXtBlock(48, use_reparam=True)
    block.eval()

    x = torch.randn(2, 48, 20, 20)
    with torch.no_grad():
        out_multi = block(x)

    assert block.use_reparam
    assert hasattr(block, "dw_block")
    assert not block.dw_block.deploy

    block.switch_to_deploy()
    assert block.dw_block.deploy

    with torch.no_grad():
        out_fused = block(x)

    diff = torch.max(torch.abs(out_multi - out_fused)).item()
    assert diff < 1e-5, f"Block diff {diff} exceeds tolerance 1e-5"

