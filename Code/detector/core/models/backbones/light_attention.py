"""Lightweight attention cores and residual adapters; backbone hooks are separate."""

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from core.backbone_config import finite_scalar, positive_integer, resolve_backbone_features


def _validate_input(x):
    if (not isinstance(x, torch.Tensor) or x.ndim != 4 or
            any(d <= 0 for d in x.shape) or not x.is_floating_point()):
        raise ValueError("input must be a floating BCHW tensor with positive dimensions")


class ECACore(nn.Module):
    """Channel attention with FP32 pooling/correlation and zero-init gates of 0.5.

    The core has exactly k weights and supports any positive input channel count.
    The core excludes the learned residual beta owned by ResidualLightAttention.
    All FP32 conversions retain device and gradients to the registered weights.
    """

    def __init__(self, kernel_size=3):
        super().__init__()
        self.kernel_size = positive_integer(kernel_size, "kernel_size")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.channel_conv = nn.Conv1d(1, 1, self.kernel_size,
                                      padding=(self.kernel_size - 1) // 2, bias=False)
        nn.init.zeros_(self.channel_conv.weight)

    def channel_weights(self, x):
        """Return FP32 [B,C,1,1] gates, independent of other batch items."""
        _validate_input(x)
        # Meta has no autocast backend; shape/parameter audits need no precision
        # context. Disable runtime autocast to keep this very small gate FP32.
        precision_context = (nullcontext() if x.device.type == "meta" else
                             torch.autocast(device_type=x.device.type, enabled=False))
        with precision_context:
            descriptor = x.float().mean(dim=(2, 3)).unsqueeze(1)
            correlated = F.conv1d(descriptor, self.channel_conv.weight.float(),
                                  padding=(self.kernel_size - 1) // 2)
            return correlated.sigmoid().transpose(1, 2).unsqueeze(-1)

    def forward(self, x):
        weights = self.channel_weights(x)
        return x * weights.to(dtype=x.dtype)


class SimAMCore(nn.Module):
    """Parameter-free spatial energy attention with FP32 channel-local statistics.

    The guarded variance denominator defines a single-spatial-element map, whose
    gate is sigmoid(0.5). This core contains no occupancy mask or learned beta.
    """

    def __init__(self, lambda_value=0.0001):
        super().__init__()
        self.lambda_value = finite_scalar(lambda_value, "simam_lambda", positive=True)

    def spatial_weights(self, x):
        """Return FP32 gates using independent per-batch/channel spatial moments."""
        _validate_input(x)
        values = x.float()
        squared_deviation = (values - values.mean(dim=(2, 3), keepdim=True)).square()
        denominator = max(x.shape[2] * x.shape[3] - 1, 1)
        variance = (squared_deviation.sum(dim=(2, 3), keepdim=True) / denominator).clamp_min(0)
        energy = squared_deviation / (4 * (variance + self.lambda_value)) + 0.5
        return energy.sigmoid()

    def forward(self, x):
        return x * self.spatial_weights(x).to(dtype=x.dtype)


class ResidualLightAttention(nn.Module):
    """Scalar residual X + beta * (T(X) - X), preserving input dtype and aliases."""

    def __init__(self, core, layer_scale_init=0.001):
        super().__init__()
        self.core = core
        scale = finite_scalar(layer_scale_init, "local_attention_layer_scale_init")
        self.beta = nn.Parameter(torch.tensor(scale))

    def forward(self, x):
        attended = self.core(x)
        accumulation_dtype = torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        values = x.to(dtype=accumulation_dtype)
        output = values + self.beta.to(dtype=accumulation_dtype) * (
            attended.to(dtype=accumulation_dtype) - values)
        return output.to(dtype=x.dtype)


def build_light_attention(kind="none", *, eca_kernel_size=None, simam_lambda=None,
                          layer_scale_init=None):
    """Build one selected adapter using the shared pure validation/defaults.

    None means no module-specific override. Explicit inactive overrides are
    rejected. Disabled attention is an Identity without beta or persistent state.
    """
    config = {"local_attention": kind}
    for key, value in (("local_attention_eca_kernel_size", eca_kernel_size),
                       ("local_attention_simam_lambda", simam_lambda),
                       ("local_attention_layer_scale_init", layer_scale_init)):
        if value is not None:
            config[key] = value
    resolved = resolve_backbone_features(config)
    if resolved.local_attention == "none":
        return nn.Identity()
    if resolved.local_attention == "eca":
        core = ECACore(resolved.local_attention_eca_kernel_size)
    else:
        core = SimAMCore(resolved.local_attention_simam_lambda)
    return ResidualLightAttention(core, resolved.local_attention_layer_scale_init)
