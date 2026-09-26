"""BEVNeXt: A modern, lightweight 3D LiDAR BEV backbone under 2.0M parameters.

Architectural highlights:
- 7x7 Depthwise Conv Inverted Bottleneck blocks (0.7m metric receptive field in BEV).
- Smooth SiLU activations replacing legacy ReLU.
- Strategic single-stage C4 LiteMLA attention (linear complexity, FP32 accumulation).
- Pure-convolution C5 stage to prevent cascading attention interference.
- Bilinear Scale-Gated FPN Neck replacing ConvTranspose2d to eliminate checkerboard artifacts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from core.models.backbones.bevnext_blocks import (
    BEVNeXtBlock,
    DownsampleBlock,
    LiteMLARefinement,
)


class BEVNeXtBackbone(nn.Module):
    """Modern BEV backbone for 3D LiDAR object detection.

    Args:
        input_channels: Number of BEV input slices (e.g. 8 for RichBEV, 35 for binary slices).
        backbone_out_dim: Channels of the output feature map (default 16 for Header).
        c4_attention: Attention adapter at Stage 3 ('none' or 'litemla').
        scale_gated_fpn: Whether to use learnable depthwise scale gating in FPN fusion.
        expansion: Channel expansion ratio inside BEVNeXt blocks.
    """

    def __init__(
        self,
        input_channels: int = 8,
        backbone_out_dim: int = 16,
        c4_attention: str = "litemla",
        scale_gated_fpn: bool = True,
        expansion: float = 2.5,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.backbone_out_dim = backbone_out_dim
        self.scale_gated_fpn = scale_gated_fpn

        # -------------------------------------------------------------
        # 1. Stem (Input 800x704 -> 400x352, stride 2, 32 channels)
        # -------------------------------------------------------------
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )

        # -------------------------------------------------------------
        # 2. Stage 2 (400x352 -> 200x176, stride 4, 48 channels)
        # -------------------------------------------------------------
        self.down2 = DownsampleBlock(32, 48, stride=2)
        self.stage2 = nn.Sequential(
            BEVNeXtBlock(48, expansion=expansion),
            BEVNeXtBlock(48, expansion=expansion),
        )

        # -------------------------------------------------------------
        # 3. Stage 3 (200x176 -> 100x88, stride 8, 96 channels)
        # Core semantic-geometric stage with single-stage attention hook
        # -------------------------------------------------------------
        self.down3 = DownsampleBlock(48, 96, stride=2)
        self.stage3 = nn.Sequential(
            BEVNeXtBlock(96, expansion=expansion),
            BEVNeXtBlock(96, expansion=expansion),
            BEVNeXtBlock(96, expansion=expansion),
            BEVNeXtBlock(96, expansion=expansion),
        )

        attn_choice = str(c4_attention).lower()
        if attn_choice == "none":
            self.c4_attention = nn.Identity()
        elif attn_choice == "litemla":
            self.c4_attention = LiteMLARefinement(
                channels=96,
                head_dim=16,
                scales=(5,),
                layer_scale_init=0.01,
            )
        else:
            raise ValueError(f"Unsupported c4_attention {c4_attention!r}; expected 'none' or 'litemla'")
        self.c4_attention_name = attn_choice

        # -------------------------------------------------------------
        # 4. Stage 4 (100x88 -> 50x44, stride 16, 128 channels)
        # Pure convolution stage (no attention) to avoid cascading interference
        # -------------------------------------------------------------
        self.down4 = DownsampleBlock(96, 128, stride=2)
        self.stage4 = nn.Sequential(
            BEVNeXtBlock(128, expansion=expansion),
            BEVNeXtBlock(128, expansion=expansion),
        )

        # -------------------------------------------------------------
        # 5. Bilinear Scale-Gated FPN Neck
        # -------------------------------------------------------------
        # Lateral projections
        self.lat_c5 = nn.Conv2d(128, 48, kernel_size=1, bias=False)
        self.lat_c4 = nn.Conv2d(96, 48, kernel_size=1, bias=False)
        self.lat_c3 = nn.Conv2d(48, 24, kernel_size=1, bias=False)

        # Refinement after bilinear interpolation (avoids deconv checkerboards)
        self.refine_u4 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1, groups=48, bias=False),
            nn.BatchNorm2d(48),
            nn.SiLU(inplace=True),
        )
        self.proj_u3 = nn.Sequential(
            nn.Conv2d(48, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.SiLU(inplace=True),
        )

        # Zero-initialized scale gates
        if self.scale_gated_fpn:
            self.gate_c4 = nn.Conv2d(48, 48, kernel_size=3, padding=1, groups=48, bias=True)
            self.gate_c3 = nn.Conv2d(24, 24, kernel_size=3, padding=1, groups=24, bias=True)
            nn.init.zeros_(self.gate_c4.weight)
            nn.init.zeros_(self.gate_c4.bias)
            nn.init.zeros_(self.gate_c3.weight)
            nn.init.zeros_(self.gate_c3.bias)

        # Final projection to header input dimension (16 channels at stride 4)
        self.out_conv = nn.Sequential(
            nn.Conv2d(24, backbone_out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(backbone_out_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Bottom-up feature hierarchy
        c1 = self.stem(x)              # (B, 32, 400, 352)
        c2 = self.stage2(self.down2(c1))  # (B, 48, 200, 176)
        c3 = self.stage3(self.down3(c2))  # (B, 96, 100, 88)
        c4 = self.c4_attention(c3)     # (B, 96, 100, 88) refined with LiteMLA
        c5 = self.stage4(self.down4(c4))  # (B, 128, 50, 44)

        # Top-down FPN path
        l5 = self.lat_c5(c5)           # (B, 48, 50, 44)
        l4 = self.lat_c4(c4)           # (B, 48, 100, 88)
        u4 = F.interpolate(l5, scale_factor=2, mode="bilinear", align_corners=False)
        u4 = self.refine_u4(u4)        # (B, 48, 100, 88)

        if self.scale_gated_fpn:
            p4 = u4 + 2 * torch.sigmoid(self.gate_c4(l4 + u4)) * l4
        else:
            p4 = u4 + l4

        l3 = self.lat_c3(c2)           # (B, 24, 200, 176)
        u3 = F.interpolate(p4, scale_factor=2, mode="bilinear", align_corners=False)
        u3 = self.proj_u3(u3)          # (B, 24, 200, 176)

        if self.scale_gated_fpn:
            p3 = u3 + 2 * torch.sigmoid(self.gate_c3(l3 + u3)) * l3
        else:
            p3 = u3 + l3

        # Final feature map feeding detection header (stride 4, 16ch, 200x176)
        return self.out_conv(p3)
