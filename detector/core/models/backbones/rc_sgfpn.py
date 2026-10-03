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
