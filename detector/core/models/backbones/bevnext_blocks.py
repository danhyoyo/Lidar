"""Core modern building blocks for the BEVNeXt LiDAR backbone.

Integrates:
- 7x7 Depthwise Inverted Bottleneck blocks with SiLU and LayerScale (ConvNeXt/UIB style)
- Strided Downsampling blocks
- Efficient Multi-Scale Linear Attention (LiteMLA) with FP32 accumulation
"""

import math
import torch
import torch.nn as nn
from torch import Tensor


def _layer_scale(channels: int, value: float) -> nn.Parameter:
    """Create a per-channel scale parameter initialized to `value`."""
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError("layer_scale_init must be finite and nonnegative")
    return nn.Parameter(torch.full((1, channels, 1, 1), float(value)))


class BEVNeXtBlock(nn.Module):
    """Modern BEV building block with 7x7 Depthwise Conv and Inverted Expansion.

    Structure:
        x -> DW-Conv 7x7 (groups=C) -> BN -> 1x1 PW (C -> e*C) -> SiLU
          -> 1x1 PW (e*C -> C) -> BN -> LayerScale -> (+) -> Output
    """

    def __init__(
        self,
        channels: int,
        expansion: float = 2.5,
        layer_scale_init: float = 1e-5,
    ):
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if expansion <= 0:
            raise ValueError("expansion must be positive")

        hidden_dim = int(round(channels * expansion))
        self.channels = channels

        # 1. Spatial aggregation with large 7x7 kernel (0.7m receptive field in BEV)
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
        self.act = nn.SiLU(inplace=True)

        # 3. Pointwise projection back to input channels
        self.pw_project = nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)

        # 4. LayerScale for stable training dynamics
        self.layer_scale = _layer_scale(channels, layer_scale_init)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        feat = self.dwconv(x)
        feat = self.norm1(feat)
        feat = self.pw_expand(feat)
        feat = self.act(feat)
        feat = self.pw_project(feat)
        feat = self.norm2(feat)
        return residual + self.layer_scale.to(feat.dtype) * feat


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
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))


class LiteMLARefinement(nn.Module):
    """Residual Multi-Scale Linear Attention adapter for high-resolution BEV stages.

    Operates without NxN spatial attention matrices by evaluating:
        V @ (K^T @ Q) / (sum(K)^T @ Q + eps)
    Accumulates in FP32 under mixed precision and initializes near Identity.
    """

    def __init__(
        self,
        channels: int = 96,
        head_dim: int = 16,
        scales: tuple = (5,),
        eps: float = 1e-6,
        layer_scale_init: float = 0.01,
    ):
        super().__init__()
        if type(head_dim) is not int or head_dim < 1 or channels % head_dim:
            raise ValueError(f"head_dim ({head_dim}) must divide channels ({channels})")
        if not isinstance(scales, (list, tuple)) or any(
            type(scale) is not int or scale < 3 or scale % 2 == 0 for scale in scales
        ) or len(set(scales)) != len(scales):
            raise ValueError("scales must contain distinct odd integers >= 3")
        if isinstance(eps, bool) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")

        self.channels = channels
        self.head_dim = head_dim
        self.eps = float(eps)
        heads = channels // head_dim

        self.qkv = nn.Conv2d(channels, 3 * channels, 1, bias=False)
        self.aggregations = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    3 * channels,
                    3 * channels,
                    scale,
                    padding=scale // 2,
                    groups=3 * channels,
                    bias=False,
                ),
                nn.Conv2d(
                    3 * channels,
                    3 * channels,
                    1,
                    groups=3 * heads,
                    bias=False,
                ),
            )
            for scale in scales
        ])
        self.output_projection = nn.Sequential(
            nn.Conv2d(channels * (1 + len(scales)), channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.layer_scale = _layer_scale(channels, layer_scale_init)

    def linear_attention(self, qkv: Tensor) -> Tensor:
        batch, _, height, width = qkv.shape
        original_dtype = qkv.dtype
        with torch.autocast(device_type=qkv.device.type, enabled=False):
            if original_dtype in (torch.float16, torch.bfloat16):
                qkv = qkv.float()
            packed = qkv.reshape(batch, -1, 3 * self.head_dim, height * width)
            query, key, value = packed.split(self.head_dim, dim=2)
            query, key = query.relu(), key.relu()
            numerator = (value @ key.transpose(-1, -2)) @ query
            denominator = key.sum(dim=-1, keepdim=True).transpose(-1, -2) @ query
            attended = numerator / (denominator + self.eps)
            attended = attended.reshape(batch, -1, height, width)
        return attended.to(original_dtype)

    def forward(self, x: Tensor) -> Tensor:
        qkv = self.qkv(x)
        packed = torch.cat([qkv] + [op(qkv) for op in self.aggregations], dim=1)
        refined = self.output_projection(self.linear_attention(packed))
        return x + self.layer_scale.to(refined.dtype) * refined
