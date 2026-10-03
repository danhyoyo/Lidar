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
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds
        self.num_bands = num_bands
        self.out_dim = 2 * num_bands

        embedding = self._generate_embedding(height, width)
        self.register_buffer("embedding", embedding, persistent=False)

    def _generate_embedding(self, h: int, w: int) -> Tensor:
        x_min, x_max = float(self.x_bounds[0]), float(self.x_bounds[1])
        y_min, y_max = float(self.y_bounds[0]), float(self.y_bounds[1])

        # Generate physical metric coordinates in meters
        y_coords = torch.linspace(y_min, y_max, h, dtype=torch.float32)
        x_coords = torch.linspace(x_min, x_max, w, dtype=torch.float32)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        # Absolute Euclidean distance r = sqrt(x^2 + y^2)
        r = torch.sqrt(xx**2 + yy**2)
        r_max = math.sqrt(max(abs(x_min), abs(x_max))**2 + max(abs(y_min), abs(y_max))**2)
        r_norm = torch.clamp(r / r_max, 0.0, 1.0).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

        # Multi-band sinusoidal basis: [sin(2^b * pi * r), cos(2^b * pi * r)]
        bands = []
        for b in range(self.num_bands):
            freq = (2.0**b) * math.pi
            bands.append(torch.sin(freq * r_norm))
            bands.append(torch.cos(freq * r_norm))

        return torch.cat(bands, dim=1)  # (1, 2*num_bands, H, W)

    def forward(self, h: Optional[int] = None, w: Optional[int] = None) -> Tensor:
        if h is None or w is None or (h == self.height and w == self.width):
            return self.embedding
        device = self.embedding.device
        dtype = self.embedding.dtype
        return self._generate_embedding(h, w).to(device=device, dtype=dtype)


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
        _, _, h, w = l_feat.shape
        if self.deploy:
            # Zero-latency path: precomputed spatial bias is folded into buffer
            bias = self.static_spatial_bias
            if bias.shape[-2:] != (h, w):
                bias = F.interpolate(bias, size=(h, w), mode="bilinear", align_corners=False)
            gate_logits = self.content_conv(l_feat + u_feat) + bias
        else:
            feat_logits = self.content_conv(l_feat + u_feat)
            range_logits = self.range_proj(self.fre(h, w))
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



class RangeConditionedSGFPN(nn.Module):
    """Range-Conditioned Scale-Gated Feature Pyramid Network.

    Supports both Unidirectional (top-down) and Bidirectional (top-down + bottom-up)
    routing pathways, with continuous Fourier Range Embedding modulation.
    """

    def __init__(
        self,
        in_channels: Tuple[int, int, int] = (48, 96, 128),  # C3, C4, C5
        out_channels: int = 16,
        lateral_channels: Tuple[int, int, int] = (24, 48, 48),
        bidirectional: bool = False,
        num_range_bands: int = 4,
        geometry: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lateral_channels = lateral_channels
        self.bidirectional = bool(bidirectional)

        if geometry is None:
            geometry = {"x_min": 0.0, "x_max": 70.4, "y_min": -40.0, "y_max": 40.0}
        x_bounds = (float(geometry.get("x_min", 0.0)), float(geometry.get("x_max", 70.4)))
        y_bounds = (float(geometry.get("y_min", -40.0)), float(geometry.get("y_max", 40.0)))

        c3_in, c4_in, c5_in = in_channels
        l3_ch, l4_ch, l5_ch = lateral_channels

        # 1. Lateral Projections (1x1 convs)
        self.lat_c5 = nn.Conv2d(c5_in, l5_ch, kernel_size=1, bias=False)
        self.lat_c4 = nn.Conv2d(c4_in, l4_ch, kernel_size=1, bias=False)
        self.lat_c3 = nn.Conv2d(c3_in, l3_ch, kernel_size=1, bias=False)

        # 2. Top-Down Pathway
        # Level 4: 50x44 -> 100x88
        self.refine_u4 = nn.Sequential(
            nn.Conv2d(l5_ch, l4_ch, kernel_size=3, padding=1, groups=min(l4_ch, l5_ch), bias=False),
            nn.BatchNorm2d(l4_ch),
            nn.SiLU(inplace=True),
        )
        self.gate_td4 = RangeConditionedScaleGate(
            l4_ch, height=100, width=88, x_bounds=x_bounds, y_bounds=y_bounds, num_bands=num_range_bands
        )

        # Level 3: 100x88 -> 200x176
        self.proj_u3 = nn.Sequential(
            nn.Conv2d(l4_ch, l3_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(l3_ch),
            nn.SiLU(inplace=True),
        )
        self.gate_td3 = RangeConditionedScaleGate(
            l3_ch, height=200, width=176, x_bounds=x_bounds, y_bounds=y_bounds, num_bands=num_range_bands
        )

        # 3. Bottom-Up Pathway (Active when bidirectional=True)
        if self.bidirectional:
            # P3 -> P4: stride 2 downsampling (200x176 -> 100x88)
            self.down_p3 = nn.Sequential(
                nn.Conv2d(l3_ch, l4_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(l4_ch),
                nn.SiLU(inplace=True),
            )
            self.gate_bu4 = RangeConditionedScaleGate(
                l4_ch, height=100, width=88, x_bounds=x_bounds, y_bounds=y_bounds, num_bands=num_range_bands
            )

            # P4 -> P5: stride 2 downsampling (100x88 -> 50x44)
            self.down_p4 = nn.Sequential(
                nn.Conv2d(l4_ch, l5_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(l5_ch),
                nn.SiLU(inplace=True),
            )
            self.gate_bu5 = RangeConditionedScaleGate(
                l5_ch, height=50, width=44, x_bounds=x_bounds, y_bounds=y_bounds, num_bands=num_range_bands
            )

            # Aggregation from reinforced P4 back into P3
            self.bu_refine_p3 = nn.Sequential(
                nn.Conv2d(l4_ch, l3_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(l3_ch),
                nn.SiLU(inplace=True),
            )

        # 4. Final Header Output Projection (Stride 4, 16 channels)
        self.out_conv = nn.Sequential(
            nn.Conv2d(l3_ch, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, c3: Tensor, c4: Tensor, c5: Tensor) -> Tensor:
        # Lateral feature projections
        l5 = self.lat_c5(c5)  # (B, 48, 50, 44)
        l4 = self.lat_c4(c4)  # (B, 48, 100, 88)
        l3 = self.lat_c3(c3)  # (B, 24, 200, 176)

        # Top-Down Pass
        u4 = F.interpolate(l5, scale_factor=2.0, mode="bilinear", align_corners=False)
        u4 = self.refine_u4(u4)
        p4 = self.gate_td4(l4, u4)  # (B, 48, 100, 88)

        u3 = F.interpolate(p4, scale_factor=2.0, mode="bilinear", align_corners=False)
        u3 = self.proj_u3(u3)
        p3 = self.gate_td3(l3, u3)  # (B, 24, 200, 176)

        # Bottom-Up Pass (optional)
        if self.bidirectional:
            d4 = self.down_p3(p3)
            p4_bu = self.gate_bu4(d4, p4) + 0.5 * l4

            d5 = self.down_p4(p4_bu)
            _ = self.gate_bu5(d5, l5)  # Reinforces P5 state

            p4_up = F.interpolate(p4_bu, scale_factor=2.0, mode="bilinear", align_corners=False)
            p3 = p3 + self.bu_refine_p3(p4_up)

        return self.out_conv(p3)

    def switch_to_deploy(self):
        """Recursively delegates switch_to_deploy to internal gates."""
        for m in self.modules():
            if hasattr(m, "switch_to_deploy") and m is not self:
                m.switch_to_deploy()
