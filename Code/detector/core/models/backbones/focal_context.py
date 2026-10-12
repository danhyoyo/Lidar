"""Standalone feature-based focal context v1; backbone integration is separate."""

import torch
from torch import nn
from torch.nn import functional as F

from core.backbone_config import positive_integer, resolve_focal_settings


class FocalContext(nn.Module):
    """Stride-8 context operator with bias-free projections and ordinary BN.

    Defaults match C=96, D=64, L=3. Explicit variants share the pure config
    validator. BN retains PyTorch/repository eps=1e-5, momentum=0.1, affine and
    running statistics; there is no special single-element training fallback.
    """

    def __init__(self, channels=96, bottleneck=64, dilations=(1, 2, 3),
                 layer_scale_init=0.001, version=1):
        super().__init__()
        self.channels = positive_integer(channels, "channels")
        (self.version, self.bottleneck, self.dilations,
         self.layer_scale_init) = resolve_focal_settings(
            version=version, bottleneck=bottleneck, dilations=dilations,
            layer_scale_init=layer_scale_init)
        self.num_levels = len(self.dilations)
        self.pre_norm = nn.BatchNorm2d(self.channels)
        self.in_projection = nn.Conv2d(
            self.channels, 2 * self.bottleneck + self.num_levels + 1, 1, bias=False)
        self.level_convs = nn.ModuleList(
            nn.Conv2d(self.bottleneck, self.bottleneck, 3, padding=d,
                      dilation=d, groups=self.bottleneck, bias=False)
            for d in self.dilations)
        self.context_projection = nn.Conv2d(self.bottleneck, self.bottleneck, 1, bias=False)
        self.out_projection = nn.Conv2d(self.bottleneck, self.channels, 1, bias=False)
        self.out_norm = nn.BatchNorm2d(self.channels)
        self.gamma = nn.Parameter(torch.full((1, self.channels, 1, 1), self.layer_scale_init))

    @staticmethod
    def _validate_feature(x, channels, name):
        if (not isinstance(x, torch.Tensor) or x.ndim != 4 or
                x.shape[1] != channels or any(d <= 0 for d in x.shape) or
                not x.is_floating_point()):
            raise ValueError(f"{name} must be a floating BCHW tensor with C={channels} and positive dimensions")

    def project(self, x):
        """Pre-normalize input and split its single projection into Q/S/G."""
        self._validate_feature(x, self.channels, "input")
        return self.in_projection(self.pre_norm(x)).split(
            (self.bottleneck, self.bottleneck, self.num_levels + 1), dim=1)

    def context_levels(self, seed):
        """Return sequential local levels and FP32 global mean of the last level.

        The pooled tensor has no normalization layer and remains FP32 until
        fusion casts it to the level dtype. No raw-point occupancy mask is used.
        """
        self._validate_feature(seed, self.bottleneck, "context seed")
        levels = []
        feature = seed
        for conv in self.level_convs:
            feature = F.silu(conv(feature))
            levels.append(feature)
        pooled = feature.float().mean(dim=(2, 3), keepdim=True)
        return tuple(levels), pooled

    def gate_weights(self, logits):
        """Stable channel softmax in FP32, cast back to the gate feature dtype."""
        self._validate_feature(logits, self.num_levels + 1, "gate logits")
        return torch.softmax(logits.float(), dim=1).to(dtype=logits.dtype)

    def context_from_seed(self, seed, logits):
        """Fuse local levels and broadcast global context with position gates."""
        self._validate_feature(seed, self.bottleneck, "context seed")
        self._validate_feature(logits, self.num_levels + 1, "gate logits")
        if (seed.shape[0] != logits.shape[0] or seed.shape[2:] != logits.shape[2:] or
                seed.device != logits.device):
            raise ValueError("context seed and gate logits must share batch, spatial shape and device")
        levels, pooled = self.context_levels(seed)
        weights = self.gate_weights(logits).to(dtype=levels[0].dtype)
        fused = torch.zeros_like(levels[0])
        for i, level in enumerate(levels):
            fused = fused + weights[:, i:i+1] * level
        fused = fused + weights[:, -1:] * pooled.to(dtype=fused.dtype)
        return fused, weights

    def prepare_context(self, x):
        """Return query, fused context and gate weights for later modulation."""
        query, seed, logits = self.project(x)
        fused, weights = self.context_from_seed(seed, logits)
        return query, fused, weights

    def forward(self, x):
        query, fused, _ = self.prepare_context(x)
        context = self.context_projection(fused)
        product = query.float() * context.float()
        # Autocast handles the output projection boundary for FP32 model weights.
        # Explicit low-precision model weights require a matching convolution
        # input outside autocast; form the modulation in FP32 before that cast.
        projected = self.out_projection(product.to(dtype=self.out_projection.weight.dtype))
        residual = self.out_norm(projected)
        accumulation_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        output = (x.to(dtype=accumulation_dtype) + self.gamma.to(dtype=accumulation_dtype) *
                  residual.to(dtype=accumulation_dtype))
        return output.to(dtype=x.dtype)
