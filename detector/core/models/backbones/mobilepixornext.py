"""MobilePixorNeXt: A modern, lightweight 3D LiDAR BEV backbone under 2.0M parameters.

Architectural highlights:
- 7x7 Depthwise Conv Inverted Bottleneck blocks; metric footprint depends on stage stride.
- Smooth SiLU activations replacing legacy ReLU.
- Strategic single-stage C4 LiteMLA attention (linear complexity, FP32 accumulation).
- Pure-convolution C5 stage to prevent cascading attention interference.
- Bilinear Scale-Gated FPN Neck replacing ConvTranspose2d to eliminate checkerboard artifacts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from core.backbone_config import resolve_backbone_features

from core.models.backbones.mobilepixornext_blocks import (
    MobilePixorNeXtBlock,
    DownsampleBlock,
    LiteMLARefinement,
)
from core.models.backbones.rc_sgfpn import RangeConditionedSGFPN


class MobilePixorNeXtBackbone(nn.Module):
    """Modern BEV backbone for 3D LiDAR object detection.

    Args:
        input_channels: Number of BEV input slices (e.g. 8 for RichBEV, 35 for binary slices).
        backbone_out_dim: Channels of the output feature map (default 16 for Header).
        c4_attention: Attention adapter at Stage 3 ('none' or 'litemla').
        c4_attention_scales: Regional kernel scales used by LiteMLA.
        c4_attention_qk_norm: QK normalization mode ('none', 'rmsnorm', or 'layernorm').
        scale_gated_fpn: Whether to use learnable depthwise scale gating in FPN fusion.
        expansion: Channel expansion ratio inside MobilePixorNeXt blocks.
        use_reparam: Whether to use RepConv7x7 structural reparameterization.
        deploy: Whether model is instantiated in deployed state.
        neck_type: Type of neck ('scale_gated_fpn', 'rc_sgfpn', or 'rc_bisgfpn').
        num_range_bands: Number of frequency bands for Fourier Range Embedding in RC-SGFPN.
        geometry: LiDAR metric bounds dictionary.
        stage_depths: Positive block counts at strides 4/8/16; legacy default (2,4,2).
        neck_fusion_channels: Standard SG-FPN final fusion width; 24 legacy or 32 explicit.
    """

    def __init__(
        self,
        input_channels: int = 8,
        backbone_out_dim: int = 16,
        c4_attention: str = "litemla",
        c4_attention_scales: tuple = None,
        c4_attention_qk_norm: str = "none",
        scale_gated_fpn: bool = True,
        expansion: float = 2.5,
        use_reparam: bool = False,
        deploy: bool = False,
        neck_type: str = "scale_gated_fpn",
        num_range_bands: int = 4,
        geometry: dict = None,
        stage_depths: tuple = (2, 4, 2),
        neck_fusion_channels: int = 24,
        detail_path: bool = False,
        c4_context: str = "none",
        local_attention: str = "none",
        c4_context_version=None,
        c4_context_bottleneck=None,
        c4_context_dilations=None,
        c4_context_layer_scale_init=None,
        local_attention_eca_kernel_size=None,
        local_attention_simam_lambda=None,
        local_attention_layer_scale_init=None,
    ):
        super().__init__()
        if (not isinstance(stage_depths, (list, tuple)) or len(stage_depths) != 3 or
                any(type(depth) is not int or depth <= 0 for depth in stage_depths)):
            raise ValueError("stage_depths must contain exactly three positive integers")
        self.stage_depths = tuple(stage_depths)
        self.input_channels = input_channels
        self.backbone_out_dim = backbone_out_dim
        self.scale_gated_fpn = scale_gated_fpn
        self.use_reparam = bool(use_reparam)
        self.deploy = bool(deploy)
        self.neck_type = str(neck_type).lower() if neck_type else "scale_gated_fpn"
        if self.neck_type == "sgfpn":
            self.neck_type = "scale_gated_fpn"
        options = dict(c4_attention=c4_attention, c4_attention_qk_norm=c4_attention_qk_norm,
                       c4_context=c4_context, local_attention=local_attention,
                       neck_fusion_channels=neck_fusion_channels, detail_path=detail_path,
                       stage_depths=stage_depths)
        for name, value in (("c4_attention_scales", c4_attention_scales),
                            ("c4_context_version", c4_context_version),
                            ("c4_context_bottleneck", c4_context_bottleneck),
                            ("c4_context_dilations", c4_context_dilations),
                            ("c4_context_layer_scale_init", c4_context_layer_scale_init),
                            ("local_attention_eca_kernel_size", local_attention_eca_kernel_size),
                            ("local_attention_simam_lambda", local_attention_simam_lambda),
                            ("local_attention_layer_scale_init", local_attention_layer_scale_init)):
            if value is not None:
                options[name] = value
        feature_options = resolve_backbone_features(options, geometry=geometry)
        self.neck_fusion_channels = feature_options.neck_fusion_channels
        self.detail_path = feature_options.detail_path
        if self.neck_fusion_channels != 24 and self.neck_type != "scale_gated_fpn":
            raise ValueError("neck_fusion_channels=32 requires standard SG-FPN (scale_gated_fpn)")
        if self.detail_path and self.neck_type != "scale_gated_fpn":
            raise ValueError("detail_path requires standard SG-FPN")
        self.requires_aligned_grid = (feature_options.c4_context != "none" or
                                      feature_options.local_attention != "none" or self.detail_path or
                                      self.neck_fusion_channels != 24 or self.stage_depths != (2, 4, 2))

        if self.neck_type in ("rc_sgfpn", "rc_bisgfpn"):
            self.rc_neck = RangeConditionedSGFPN(
                in_channels=(48, 96, 128),
                out_channels=backbone_out_dim,
                lateral_channels=(24, 48, 48),
                bidirectional=(self.neck_type == "rc_bisgfpn"),
                num_range_bands=num_range_bands,
                geometry=geometry,
                deploy=self.deploy,
            )
        else:
            self.rc_neck = None

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
            *(MobilePixorNeXtBlock(48, expansion=expansion, use_reparam=self.use_reparam, deploy=self.deploy)
              for _ in range(self.stage_depths[0])),
        )

        # -------------------------------------------------------------
        # 3. Stage 3 (200x176 -> 100x88, stride 8, 96 channels)
        # Core semantic-geometric stage with single-stage attention hook
        # -------------------------------------------------------------
        self.down3 = DownsampleBlock(48, 96, stride=2)
        self.stage3 = nn.Sequential(
            *(MobilePixorNeXtBlock(96, expansion=expansion, use_reparam=self.use_reparam, deploy=self.deploy)
              for _ in range(self.stage_depths[1])),
        )

        attn_choice = feature_options.c4_attention
        if attn_choice == "none":
            self.c4_attention = nn.Identity()
        elif attn_choice == "litemla":
            self.c4_attention = LiteMLARefinement(
                channels=96,
                head_dim=16,
                scales=feature_options.c4_attention_scales,
                layer_scale_init=0.01,
                qk_norm=feature_options.c4_attention_qk_norm,
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
            *(MobilePixorNeXtBlock(128, expansion=expansion, use_reparam=self.use_reparam, deploy=self.deploy)
              for _ in range(self.stage_depths[2])),
        )

        # -------------------------------------------------------------
        # 5. Bilinear Scale-Gated FPN Neck (instantiated only when rc_neck is None)
        # -------------------------------------------------------------
        if self.rc_neck is None:
            # Lateral projections
            self.lat_c5 = nn.Conv2d(128, 48, kernel_size=1, bias=False)
            self.lat_c4 = nn.Conv2d(96, 48, kernel_size=1, bias=False)
            self.lat_c3 = nn.Conv2d(48, self.neck_fusion_channels, kernel_size=1, bias=False)

            # Refinement after bilinear interpolation (avoids deconv checkerboards)
            self.refine_u4 = nn.Sequential(
                nn.Conv2d(48, 48, kernel_size=3, padding=1, groups=48, bias=False),
                nn.BatchNorm2d(48),
                nn.SiLU(inplace=True),
            )
            self.proj_u3 = nn.Sequential(
                nn.Conv2d(48, self.neck_fusion_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.neck_fusion_channels),
                nn.SiLU(inplace=True),
            )

            # Zero-initialized scale gates
            if self.scale_gated_fpn:
                self.gate_c4 = nn.Conv2d(48, 48, kernel_size=3, padding=1, groups=48, bias=True)
                self.gate_c3 = nn.Conv2d(self.neck_fusion_channels, self.neck_fusion_channels,
                                         kernel_size=3, padding=1, groups=self.neck_fusion_channels, bias=True)
                nn.init.zeros_(self.gate_c4.weight)
                nn.init.zeros_(self.gate_c4.bias)
                nn.init.zeros_(self.gate_c3.weight)
                nn.init.zeros_(self.gate_c3.bias)
            else:
                self.gate_c4 = nn.Identity()
                self.gate_c3 = nn.Identity()

            # Final projection to header input dimension (16 channels at stride 4)
            self.out_conv = nn.Sequential(
                nn.Conv2d(self.neck_fusion_channels, backbone_out_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(backbone_out_dim),
                nn.SiLU(inplace=True),
            )

        # Append optional initialization after all historical modules so disabled
        # and gamma-zero fixtures retain the exact legacy initialization stream.
        self.detail_branch = nn.Identity()
        self.register_parameter("detail_gamma", None)
        if self.detail_path:
            self.detail_branch = nn.Sequential(
                nn.Conv2d(32, 32, 3, stride=2, padding=1, groups=32, bias=False),
                nn.BatchNorm2d(32), nn.SiLU(inplace=True),
                nn.Conv2d(32, self.neck_fusion_channels, 1, bias=False),
                nn.BatchNorm2d(self.neck_fusion_channels), nn.SiLU(inplace=True))
            self.detail_gamma = nn.Parameter(torch.full((1, self.neck_fusion_channels, 1, 1), 0.001))

        self.c3_light_attention = nn.Identity()
        if feature_options.local_attention != "none":
            from core.models.backbones.light_attention import build_light_attention
            self.c3_light_attention = build_light_attention(
                feature_options.local_attention,
                eca_kernel_size=feature_options.local_attention_eca_kernel_size,
                simam_lambda=feature_options.local_attention_simam_lambda,
                layer_scale_init=feature_options.local_attention_layer_scale_init)
        self.c4_context = nn.Identity()
        if feature_options.c4_context == "focal":
            from core.models.backbones.focal_context import FocalContext
            self.c4_context = FocalContext(
                channels=96, bottleneck=feature_options.c4_context_bottleneck,
                dilations=feature_options.c4_context_dilations,
                layer_scale_init=feature_options.c4_context_layer_scale_init,
                version=feature_options.c4_context_version)

    def forward(self, x: Tensor) -> Tensor:
        if self.requires_aligned_grid and (x.shape[-2] % 16 != 0 or x.shape[-1] % 16 != 0):
            raise ValueError("candidate input grid height and width must be divisible by 16")
        # Bottom-up feature hierarchy
        c1 = self.stem(x)              # (B, 32, 400, 352)
        c2 = self.c3_light_attention(self.stage2(self.down2(c1)))
        c3 = self.stage3(self.down3(c2))  # (B, 96, 100, 88)
        c4 = self.c4_context(self.c4_attention(c3))
        c5 = self.stage4(self.down4(c4))  # (B, 128, 50, 44)

        if self.rc_neck is not None:
            return self.rc_neck(c2, c4, c5)

        # Top-down FPN path
        l5 = self.lat_c5(c5)           # (B, 48, 50, 44)
        l4 = self.lat_c4(c4)           # (B, 48, 100, 88)
        u4 = F.interpolate(l5, scale_factor=2.0, mode="bilinear", align_corners=False)
        u4 = self.refine_u4(u4)        # (B, 48, 100, 88)

        if self.scale_gated_fpn:
            p4 = u4 + 2 * torch.sigmoid(self.gate_c4(l4 + u4)) * l4
        else:
            p4 = u4 + l4

        l3 = self.lat_c3(c2)           # (B, 24, 200, 176)
        u3 = F.interpolate(p4, scale_factor=2.0, mode="bilinear", align_corners=False)
        u3 = self.proj_u3(u3)          # (B, 24, 200, 176)

        if self.scale_gated_fpn:
            p3 = u3 + 2 * torch.sigmoid(self.gate_c3(l3 + u3)) * l3
        else:
            p3 = u3 + l3

        detail_gamma = self.detail_gamma
        if detail_gamma is not None:
            detail_feature = self.detail_branch(c1)
            if detail_feature.shape != p3.shape:
                raise ValueError("detail branch and final fusion shapes must agree; resizing is unsupported")
            p3 = (p3.float() + detail_gamma.float() * detail_feature.float()).to(dtype=p3.dtype)

        # Final feature map feeding detection header (stride 4, 16ch, 200x176)
        return self.out_conv(p3)

    def switch_to_deploy(self):
        """Recursively trigger switch_to_deploy across all submodules."""
        for m in self.modules():
            if hasattr(m, "switch_to_deploy") and m is not self:
                m.switch_to_deploy()
