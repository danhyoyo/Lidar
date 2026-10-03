import math
import torch
import pytest
from core.models.backbones.rc_sgfpn import FourierRangeEmbedding


def test_fourier_range_embedding_shapes_and_values():
    h, w = 100, 88
    x_bounds = (0.0, 70.4)
    y_bounds = (-40.0, 40.0)
    num_bands = 4

    fre = FourierRangeEmbedding(h, w, x_bounds=x_bounds, y_bounds=y_bounds, num_bands=num_bands)

    assert hasattr(fre, "embedding")
    assert fre.embedding.shape == (1, 2 * num_bands, h, w)
    assert fre.out_dim == 2 * num_bands

    # Values must be bounded in [-1.0, 1.0]
    assert torch.all(fre.embedding >= -1.0 - 1e-5)
    assert torch.all(fre.embedding <= 1.0 + 1e-5)
    assert torch.isfinite(fre.embedding).all()

    # Exact origin check with odd dimensions: at (x=0, y=0), r=0 => sin(0)=0, cos(0)=1
    fre_odd = FourierRangeEmbedding(101, 89, x_bounds=x_bounds, y_bounds=y_bounds, num_bands=num_bands)
    origin_y = 50
    origin_x = 0
    origin_feats = fre_odd.embedding[0, :, origin_y, origin_x]
    for b in range(num_bands):
        sin_val = origin_feats[2 * b].item()
        cos_val = origin_feats[2 * b + 1].item()
        assert abs(sin_val) < 1e-5
        assert abs(cos_val - 1.0) < 1e-5


from core.models.backbones.rc_sgfpn import RangeConditionedScaleGate


def test_range_conditioned_scale_gate_zero_init():
    channels = 48
    h, w = 100, 88
    gate_module = RangeConditionedScaleGate(channels, h, w)

    l_feat = torch.randn(2, channels, h, w)
    u_feat = torch.randn(2, channels, h, w)

    # At epoch 0 (init), gate must be identically 1.000000
    out = gate_module(l_feat, u_feat)
    expected = u_feat + l_feat

    assert torch.allclose(out, expected, atol=1e-6)


def test_range_conditioned_scale_gate_switch_to_deploy():
    channels = 48
    h, w = 100, 88
    gate_module = RangeConditionedScaleGate(channels, h, w)

    # Perturb weights slightly to simulate training
    with torch.no_grad():
        gate_module.content_conv.weight.add_(torch.randn_like(gate_module.content_conv.weight) * 0.1)
        gate_module.range_proj.weight.add_(torch.randn_like(gate_module.range_proj.weight) * 0.1)

    l_feat = torch.randn(2, channels, h, w)
    u_feat = torch.randn(2, channels, h, w)

    out_train = gate_module(l_feat, u_feat)
    gate_module.switch_to_deploy()

    assert gate_module.deploy is True
    assert not hasattr(gate_module, "range_proj")
    assert not hasattr(gate_module, "fre")

    out_deploy = gate_module(l_feat, u_feat)
    assert torch.allclose(out_train, out_deploy, atol=1e-5)
