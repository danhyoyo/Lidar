"""Core modern building blocks for the MobilePixorNeXt LiDAR backbone.

Integrates:
- 7x7 Depthwise Inverted Bottleneck blocks with SiLU and LayerScale (ConvNeXt/UIB style)
- Strided Downsampling blocks
"""

import math
import torch
import torch.nn as nn
from torch import Tensor


from core.models.backbones.rep_blocks import RepConv7x7


def _layer_scale(channels: int, value: float) -> nn.Parameter:
    """Create a per-channel scale parameter initialized to `value`."""
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError("layer_scale_init must be finite and nonnegative")
    return nn.Parameter(torch.full((1, channels, 1, 1), float(value)))


class MobilePixorNeXtBlock(nn.Module):
    """Modern BEV building block with 7x7 Depthwise Conv and Inverted Expansion.

    Structure:
        x -> DW-Conv 7x7 (or RepConv7x7) -> BN -> 1x1 PW (C -> e*C) -> SiLU
          -> 1x1 PW (e*C -> C) -> BN -> LayerScale -> (+) -> Output
    """

    def __init__(
        self,
        channels: int,
        expansion: float = 2.5,
        layer_scale_init: float = 1e-5,
        use_reparam: bool = False,
        deploy: bool = False,
    ):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if expansion <= 0:
            raise ValueError("expansion must be positive")

        hidden_dim = int(round(channels * expansion))
        self.channels = channels
        self.use_reparam = bool(use_reparam)
        self.deploy = bool(deploy)

        # 1. Spatial aggregation with a 7x7 kernel or RepConv7x7 multi-branch
        if self.use_reparam:
            self.dw_block = RepConv7x7(channels, deploy=self.deploy)
            self.dwconv = nn.Identity()
            self.norm1 = nn.Identity()
        else:
            self.dw_block = nn.Identity()
            self.dwconv = nn.Conv2d(
                channels,
                channels,
                kernel_size=7,
                padding=3,
                groups=channels,
                bias=False,
            )
            self.norm1 = nn.BatchNorm2d(channels)

        # 2. Pointwise inverted bottleneck expansion
        self.pw_expand = nn.Conv2d(channels, hidden_dim, kernel_size=1, bias=False)
        self.act = nn.SiLU(inplace=False)

        # 3. Pointwise projection back to input channels
        self.pw_project = nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)

        # 4. LayerScale for stable training dynamics
        self.layer_scale = _layer_scale(channels, layer_scale_init)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        if self.use_reparam:
            feat = self.dw_block(x)
        else:
            feat = self.norm1(self.dwconv(x))
        feat = self.pw_expand(feat)
        feat = self.act(feat)
        feat = self.pw_project(feat)
        feat = self.norm2(feat)
        return residual + self.layer_scale.to(feat.dtype) * feat

    def switch_to_deploy(self):
        """Delegate reparameterization to internal RepConv7x7."""
        if self.use_reparam and hasattr(self.dw_block, "switch_to_deploy"):
            self.dw_block.switch_to_deploy()


class DownsampleBlock(nn.Module):
    """Strided transition block for spatial downsampling and channel expansion."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))
