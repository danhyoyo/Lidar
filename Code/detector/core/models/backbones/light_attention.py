"""Lightweight attention cores and residual adapters; backbone hooks are separate."""

import torch
from torch import nn

from core.backbone_config import finite_scalar, resolve_backbone_features


def _validate_input(x):
    if (not isinstance(x, torch.Tensor) or x.ndim != 4 or
            any(d <= 0 for d in x.shape) or not x.is_floating_point()):
        raise ValueError("input must be a floating BCHW tensor with positive dimensions")


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


def build_light_attention(kind="none", *, simam_lambda=None,
                          layer_scale_init=None):
    """Build one selected adapter using the shared pure validation/defaults.

    None means no module-specific override. Explicit inactive overrides are
    rejected. Disabled attention is an Identity without beta or persistent state.
    """
    config = {"local_attention": kind}
    for key, value in (("local_attention_simam_lambda", simam_lambda),
                       ("local_attention_layer_scale_init", layer_scale_init)):
        if value is not None:
            config[key] = value
    resolved = resolve_backbone_features(config)
    if resolved.local_attention == "none":
        return nn.Identity()
    core = SimAMCore(resolved.local_attention_simam_lambda)
    return ResidualLightAttention(core, resolved.local_attention_layer_scale_init)
