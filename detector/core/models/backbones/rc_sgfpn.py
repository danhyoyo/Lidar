"""Range-Conditioned Scale-Gated Feature Pyramid Network (RC-SGFPN).

Injects continuous Fourier Range Embeddings into multi-scale depthwise scale gating
to dynamically balance geometric precision (near range) and semantic context (far range).
"""

from typing import Tuple, List, Optional, Dict, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FourierRangeEmbedding(nn.Module):
    """Continuous Fourier Range Embedding across orthographic BEV metric space."""

    def __init__(
        self,
        height: int,
        width: int,
        x_bounds: Tuple[float, float] = (0.0, 70.4),
        y_bounds: Tuple[float, float] = (-40.0, 40.0),
        num_bands: int = 4,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.num_bands = num_bands
        self.out_dim = 2 * num_bands

        x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
        y_min, y_max = float(y_bounds[0]), float(y_bounds[1])

        # Generate physical metric coordinates in meters
        y_coords = torch.linspace(y_min, y_max, height, dtype=torch.float32)
        x_coords = torch.linspace(x_min, x_max, width, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Absolute Euclidean distance r = sqrt(x^2 + y^2)
        r = torch.sqrt(xx**2 + yy**2)
        r_max = math.sqrt(max(abs(x_min), abs(x_max))**2 + max(abs(y_min), abs(y_max))**2)
        r_norm = torch.clamp(r / r_max, 0.0, 1.0).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        # Multi-band sinusoidal basis: [sin(2^b * pi * r), cos(2^b * pi * r)]
        bands = []
        for b in range(num_bands):
            freq = (2.0**b) * math.pi
            bands.append(torch.sin(freq * r_norm))
            bands.append(torch.cos(freq * r_norm))

        embedding = torch.cat(bands, dim=1)  # (1, 2*num_bands, H, W)
        self.register_buffer("embedding", embedding, persistent=False)

    def forward(self) -> Tensor:
        return self.embedding



class RangeConditionedScaleGate(nn.Module):
    """Dynamic depthwise scale gate conditioned on spatial Fourier Range Embeddings."""

    def __init__(
        self,
        channels: int,
        height: int,
        width: int,
        x_bounds: Tuple[float, float] = (0.0, 70.4),
        y_bounds: Tuple[float, float] = (-40.0, 40.0),
        num_bands: int = 4,
    ):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        self.deploy = False

        # 1. Continuous Fourier Range Engine
        self.fre = FourierRangeEmbedding(height, width, x_bounds, y_bounds, num_bands)
        self.range_proj = nn.Conv2d(self.fre.out_dim, channels, kernel_size=1, bias=True)

        # 2. Local content depthwise aggregator
        self.content_conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=True,
        )

        # 3. Mathematical Zero-Initialization Guarantee: Gate = 2 * sigmoid(0) = 1.0
        nn.init.zeros_(self.content_conv.weight)
        nn.init.zeros_(self.content_conv.bias)
        nn.init.zeros_(self.range_proj.weight)
        nn.init.zeros_(self.range_proj.bias)

    def forward(self, l_feat: Tensor, u_feat: Tensor) -> Tensor:
        if self.deploy:
            # Zero-latency path: precomputed spatial bias is folded into buffer
            gate_logits = self.content_conv(l_feat + u_feat) + self.static_spatial_bias
        else:
            feat_logits = self.content_conv(l_feat + u_feat)
            range_logits = self.range_proj(self.fre())
            gate_logits = feat_logits + range_logits

        gate = 2.0 * torch.sigmoid(gate_logits)
        return u_feat + gate * l_feat

    @torch.no_grad()
    def switch_to_deploy(self):
        """Fuses static range projection into a constant 2D spatial bias buffer."""
        if self.deploy:
            return
        # Precompute static spatial bias: (1, C, H, W)
        range_bias = self.range_proj(self.fre())
        self.register_buffer("static_spatial_bias", range_bias)

        # Delete dynamic projection submodules to release memory and remove ONNX nodes
        del self.range_proj
        del self.fre
        self.deploy = True
