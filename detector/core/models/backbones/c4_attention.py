"""Optional C4 refinements for dense LiDAR BEV features.

LSK follows the spatial-selection mechanism of Li et al., ICCV 2023.
LiteMLA follows the multi-scale ReLU linear attention of Cai et al., ICCV 2023.
These are residual adapters, not complete LSKNet/EfficientViT backbones.
See docs/backbone_c4_ablation.md and third_party/attention/NOTICE.md for
equations, adaptation details, citations, and upstream license notices.
"""

import math

import torch
from torch import nn


def _layer_scale(channels, value):
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        raise ValueError("layer_scale_init must be finite and nonnegative")
    return nn.Parameter(torch.full((1, channels, 1, 1), float(value)))


class LSKRefinement(nn.Module):
    """Residual large selective kernel attention with a small LayerScale.

    Sequential depthwise 5x5 and dilated 7x7 (dilation=3) branches have
    effective receptive fields 5x5 and 23x23. Two spatial sigmoid gates select
    their projected features. The selected feature modulates the input to
    the spatial gate, followed by an output projection and residual addition.
    We omit LSKNet's separate MLP and stochastic depth for this ablation.
    """

    def __init__(self, channels=64, layer_scale_init=0.01):
        super().__init__()
        if channels < 2 or channels % 2:
            raise ValueError("LSK channels must be positive and even")
        self.norm = nn.BatchNorm2d(channels)
        self.input_projection = nn.Conv2d(channels, channels, 1)
        self.activation = nn.GELU()
        self.local = nn.Conv2d(channels, channels, 5, padding=2, groups=channels)
        self.context = nn.Conv2d(
            channels, channels, 7, padding=9, dilation=3, groups=channels
        )
        self.local_projection = nn.Conv2d(channels, channels // 2, 1)
        self.context_projection = nn.Conv2d(channels, channels // 2, 1)
        self.selector = nn.Conv2d(2, 2, 7, padding=3)
        self.gate_projection = nn.Conv2d(channels // 2, channels, 1)
        self.output_projection = nn.Conv2d(channels, channels, 1)
        self.layer_scale = _layer_scale(channels, layer_scale_init)

    def forward(self, x):
        features = self.activation(self.input_projection(self.norm(x)))
        local = self.local(features)
        context = self.context(local)
        local = self.local_projection(local)
        context = self.context_projection(context)
        joined = torch.cat((local, context), dim=1)
        descriptors = torch.cat(
            (joined.mean(dim=1, keepdim=True), joined.amax(dim=1, keepdim=True)),
            dim=1,
        )
        gates = self.selector(descriptors).sigmoid()
        selected = gates[:, :1] * local + gates[:, 1:] * context
        refined = self.output_projection(features * self.gate_projection(selected))
        return x + self.layer_scale.to(refined.dtype) * refined


class LiteMLARefinement(nn.Module):
    """Residual multi-scale linear attention; no spatial NxN score matrix.

    Native and depthwise/grouped-convolution QKV scales become extra heads.
    ReLU(Q), ReLU(K) use normalized kernel attention, not softmax attention.
    Accumulation and normalization use FP32 under FP16/BF16 autocast. The
    existing MobilePIXOR blocks supply local processing; no extra EfficientViT
    MBConv/FFN is added. A per-channel LayerScale initializes near identity.
    """

    def __init__(
        self, channels=64, head_dim=16, scales=(5,), eps=1e-6,
        layer_scale_init=0.01,
    ):
        super().__init__()
        if type(head_dim) is not int or head_dim < 1 or channels % head_dim:
            raise ValueError("head_dim must be a positive integer dividing channels")
        if not isinstance(scales, (list, tuple)) or any(
            type(scale) is not int or scale < 3 or scale % 2 == 0
            for scale in scales
        ) or len(set(scales)) != len(scales):
            raise ValueError("scales must contain distinct odd integers >= 3")
        if isinstance(eps, bool) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.head_dim = head_dim
        self.eps = float(eps)
        heads = channels // head_dim
        self.qkv = nn.Conv2d(channels, 3 * channels, 1, bias=False)
        self.aggregations = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    3 * channels, 3 * channels, scale, padding=scale // 2,
                    groups=3 * channels, bias=False,
                ),
                nn.Conv2d(
                    3 * channels, 3 * channels, 1, groups=3 * heads, bias=False,
                ),
            )
            for scale in scales
        ])
        self.output_projection = nn.Sequential(
            nn.Conv2d(channels * (1 + len(scales)), channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.layer_scale = _layer_scale(channels, layer_scale_init)

    def linear_attention(self, qkv):
        """Evaluate V K^T Q / (sum(K)^T Q + eps) in channel-first layout."""
        batch, _, height, width = qkv.shape
        original_dtype = qkv.dtype
        with torch.autocast(device_type=qkv.device.type, enabled=False):
            if original_dtype in (torch.float16, torch.bfloat16):
                qkv = qkv.float()
            packed = qkv.reshape(batch, -1, 3 * self.head_dim, height * width)
            query, key, value = packed.split(self.head_dim, dim=2)
            query, key = query.relu(), key.relu()
            # Only dxd and dxN intermediates; never NxN (N=H*W).
            numerator = (value @ key.transpose(-1, -2)) @ query
            denominator = key.sum(dim=-1, keepdim=True).transpose(-1, -2) @ query
            attended = numerator / (denominator + self.eps)
            attended = attended.reshape(batch, -1, height, width)
        return attended.to(original_dtype)

    def forward(self, x):
        qkv = self.qkv(x)
        packed = torch.cat([qkv] + [op(qkv) for op in self.aggregations], dim=1)
        refined = self.output_projection(self.linear_attention(packed))
        return x + self.layer_scale.to(refined.dtype) * refined


def build_c4_attention(name, *, lsk=None, litemla=None):
    if not isinstance(name, str):
        raise ValueError("c4_attention must be 'none', 'lsk', or 'litemla'")
    name = name.lower()
    if name == "none":
        return nn.Identity()
    if name == "lsk":
        return LSKRefinement(channels=64, **(lsk or {}))
    if name == "litemla":
        return LiteMLARefinement(channels=64, **(litemla or {}))
    raise ValueError(f"Unsupported C4 attention {name!r}; expected 'none', 'lsk', or 'litemla'")
