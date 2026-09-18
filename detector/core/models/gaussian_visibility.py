"""Fixed Gaussian propagation over occupied RichBEV pillars.

The input remains an 8-channel RichBEV tensor, optionally followed by one
or three observed-free masks. Gaussian propagation is applied only to empty
cells; measured pillar values are preserved exactly.
"""

import torch
from torch import nn
from torch.nn import functional as F


class GaussianPillarPropagation(nn.Module):
    def __init__(self, mode="plain", sigma_cells=1.5, radius_cells=4, strength=1.0,
                 kernel_type="gaussian"):
        super().__init__()
        if kernel_type not in {"gaussian", "uniform"}:
            raise ValueError(f"unsupported propagation kernel: {kernel_type!r}")
        if mode not in {"plain", "global", "height"}:
            raise ValueError(f"unsupported Gaussian propagation mode: {mode!r}")
        if sigma_cells <= 0 or radius_cells < 1 or strength < 0:
            raise ValueError("sigma/radius/strength must be positive")
        self.mode = mode
        self.kernel_type = kernel_type
        self.strength = float(strength)
        axis = torch.arange(-radius_cells, radius_cells + 1, dtype=torch.float32)
        kernel = (torch.exp(-0.5 * (axis / float(sigma_cells)) ** 2)
                  if kernel_type == "gaussian" else torch.ones_like(axis))
        kernel /= kernel.sum()
        self.register_buffer("kernel_x", kernel.view(1, 1, 1, -1).repeat(8, 1, 1, 1),
                             persistent=False)
        self.register_buffer("kernel_y", kernel.view(1, 1, -1, 1).repeat(8, 1, 1, 1),
                             persistent=False)

    def forward(self, x):
        expected = {"plain": 8, "global": 9, "height": 11}[self.mode]
        if x.ndim != 4 or x.shape[1] != expected:
            raise ValueError(f"{self.mode} expects Bx{expected}xHxW, got {tuple(x.shape)}")
        base = x[:, :8]
        occupied = (base[:, :3].amax(dim=1, keepdim=True) > 0).to(base.dtype)
        spread = F.conv2d(base, self.kernel_x, padding=(0, self.kernel_x.shape[-1] // 2),
                          groups=8)
        spread = F.conv2d(spread, self.kernel_y,
                          padding=(self.kernel_y.shape[-2] // 2, 0), groups=8)
        if self.mode == "plain":
            allowed = 1.0
        elif self.mode == "global":
            allowed = 1.0 - x[:, 8:9].clamp(0, 1)
        else:
            free = x[:, 8:11].clamp(0, 1)
            # Geometry-aligned height bands for the three occupancy channels.
            # Summary channels are blocked only if all three bands are free.
            aggregate = 1.0 - free.prod(dim=1, keepdim=True)
            allowed = torch.cat((1.0 - free, aggregate.expand(-1, 5, -1, -1)),
                                dim=1)
        return base + self.strength * (1.0 - occupied) * allowed * spread
