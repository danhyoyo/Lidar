"""Optional neck widths preserve legacy defaults and explicit parameter budgets."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "detector"))

from core.models.backbones.mobilepixornext import MobilePixorNeXtBackbone

REFERENCE = dict(input_channels=14, backbone_out_dim=32, c4_attention="none",
                 c4_attention_scales=(), stage_depths=(3, 4, 2))


def test_fusion24_explicit_and_default_keep_exact_legacy_weights_keys_and_predictions():
    torch.manual_seed(510)
    default = MobilePixorNeXtBackbone().eval()
    torch.manual_seed(510)
    explicit = MobilePixorNeXtBackbone(neck_fusion_channels=24).eval()
    assert set(default.state_dict()) == set(explicit.state_dict())
    for key, value in default.state_dict().items():
        assert torch.equal(value, explicit.state_dict()[key]), key
    assert sum(p.numel() for p in default.parameters()) == 616080
    x = torch.randn(1, 8, 32, 48)
    with torch.no_grad():
        torch.testing.assert_close(default(x), explicit(x), rtol=0, atol=0)


@pytest.mark.parametrize("gates,out_dim,delta", [(True, 32, 6240), (False, 32, 6160), (True, 16, 5088)])
def test_fusion32_changes_only_final_fusion_shapes_and_exact_parameter_delta(gates, out_dim, delta):
    kwargs = {**REFERENCE, "scale_gated_fpn": gates, "backbone_out_dim": out_dim}
    narrow = MobilePixorNeXtBackbone(**kwargs, neck_fusion_channels=24)
    wide = MobilePixorNeXtBackbone(**kwargs, neck_fusion_channels=32)
    assert set(narrow.state_dict()) == set(wide.state_dict())
    assert sum(p.numel() for p in wide.parameters()) - sum(p.numel() for p in narrow.parameters()) == delta
    assert wide.lat_c5.out_channels == wide.lat_c4.out_channels == 48
    assert wide.refine_u4[0].out_channels == 48
    assert wide.lat_c3.weight.shape == (32, 48, 1, 1)
    assert wide.proj_u3[0].weight.shape == (32, 48, 3, 3)
    assert wide.proj_u3[1].num_features == 32
    assert wide.out_conv[0].weight.shape == (out_dim, 32, 3, 3)
    if gates:
        assert wide.gate_c3.groups == wide.gate_c3.in_channels == wide.gate_c3.out_channels == 32
        assert torch.equal(wide.gate_c3.weight, torch.zeros_like(wide.gate_c3.weight))
        assert torch.equal(wide.gate_c3.bias, torch.zeros_like(wide.gate_c3.bias))


@pytest.mark.parametrize("shape", [(2, 14, 32, 48), (1, 14, 48, 80)])
def test_fusion32_asymmetric_shapes_and_finite_backward(shape):
    torch.manual_seed(512)
    m = MobilePixorNeXtBackbone(**REFERENCE, neck_fusion_channels=32).train()
    x = torch.randn(shape, requires_grad=True)
    y = m(x)
    assert y.shape == (shape[0], 32, shape[2] // 4, shape[3] // 4)
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    for p in (m.lat_c3.weight, m.proj_u3[0].weight, m.gate_c3.weight, m.out_conv[0].weight):
        assert p.grad.abs().sum() > 0


def test_fusion32_full_resolution_meta_no_context_count_and_shape():
    with torch.device("meta"):
        narrow = MobilePixorNeXtBackbone(**REFERENCE, neck_fusion_channels=24).eval()
        wide = MobilePixorNeXtBackbone(**REFERENCE, neck_fusion_channels=32).eval()
        x = torch.empty(1, 14, 800, 704)
    assert sum(p.numel() for p in narrow.parameters()) == 635408
    assert sum(p.numel() for p in wide.parameters()) == 641648
    assert wide(x).shape == (1, 32, 200, 176)


def test_fusion_invalid_width_fails():
    for width in [0, 16, 25, True, 24.0, "32", None]:
        with pytest.raises(ValueError, match="neck_fusion_channels"):
            MobilePixorNeXtBackbone(neck_fusion_channels=width)


def test_fusion32_unsupported_neck_is_rejected():
    for neck in ["rc_sgfpn", "rc_bisgfpn", "unknown"]:
        with pytest.raises(ValueError, match="neck_fusion_channels.*SG-FPN"):
            MobilePixorNeXtBackbone(neck_type=neck, neck_fusion_channels=32)


def test_fusion32_misaligned_input_grid_is_rejected_before_downsampling():
    m = MobilePixorNeXtBackbone(**REFERENCE, neck_fusion_channels=32).eval()
    for shape in [(1, 14, 31, 48), (1, 14, 32, 47)]:
        with pytest.raises(ValueError, match="divisible by 16"):
            m(torch.zeros(shape))


def test_fusion24_keeps_existing_rc_neck_topology():
    m = MobilePixorNeXtBackbone(c4_attention="none", neck_type="rc_sgfpn", neck_fusion_channels=24)
    assert m.rc_neck is not None and not hasattr(m, "lat_c3")


@pytest.mark.parametrize("width,delta", [(24, 1192), (32, 1472)])
def test_detail_parameter_delta_and_zero_gamma_exact_identity(width, delta):
    torch.manual_seed(710)
    plain = MobilePixorNeXtBackbone(**REFERENCE, neck_fusion_channels=width).eval()
    torch.manual_seed(710)
    detail = MobilePixorNeXtBackbone(**REFERENCE, neck_fusion_channels=width, detail_path=True).eval()
    for key, value in plain.state_dict().items():
        assert torch.equal(value, detail.state_dict()[key]), key
    assert sum(p.numel() for p in detail.parameters()) - sum(p.numel() for p in plain.parameters()) == delta
    assert detail.detail_gamma.shape == (1, width, 1, 1)
    torch.testing.assert_close(detail.detail_gamma, torch.full_like(detail.detail_gamma, 0.001), rtol=0, atol=0)
    assert detail.detail_branch[0].groups == 32 and detail.detail_branch[0].stride == (2, 2)
    assert detail.detail_branch[3].out_channels == width
    with torch.no_grad():
        detail.detail_gamma.zero_()
    x = torch.randn(1, 14, 48, 80)
    torch.testing.assert_close(detail(x), plain(x), rtol=0, atol=0)


def test_detail_disabled_keeps_exact_default_state_and_predictions():
    torch.manual_seed(711)
    plain = MobilePixorNeXtBackbone().eval()
    torch.manual_seed(711)
    disabled = MobilePixorNeXtBackbone(detail_path=False).eval()
    assert set(plain.state_dict()) == set(disabled.state_dict())
    x = torch.randn(1, 8, 32, 48)
    torch.testing.assert_close(plain(x), disabled(x), rtol=0, atol=0)


@pytest.mark.parametrize("width", [24, 32])
def test_detail_asymmetric_shape_backward_and_fusion_placement(width):
    m = MobilePixorNeXtBackbone(**REFERENCE, neck_fusion_channels=width, detail_path=True).train()
    x = torch.randn(2, 14, 48, 80, requires_grad=True)
    seen = {}
    handles = [m.detail_branch.register_forward_hook(lambda _m, _i, y: seen.update(detail=y)),
               m.out_conv.register_forward_pre_hook(lambda _m, args: seen.update(fusion=args[0]))]
    try:
        y = m(x)
        assert y.shape == (2, 32, 12, 20)
        assert seen['detail'].shape == seen['fusion'].shape == (2, width, 12, 20)
        y.square().mean().backward()
        for p in [m.detail_gamma, *m.detail_branch.parameters()]:
            assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    finally:
        for h in handles:
            h.remove()


def test_detail_invalid_selection_and_alignment_are_rejected():
    for value in [1, 'true', None]:
        with pytest.raises(ValueError, match='detail_path'):
            MobilePixorNeXtBackbone(detail_path=value)
    for neck in ['rc_sgfpn', 'rc_bisgfpn']:
        with pytest.raises(ValueError, match='detail_path.*SG-FPN'):
            MobilePixorNeXtBackbone(neck_type=neck, detail_path=True)
    m = MobilePixorNeXtBackbone(**REFERENCE, detail_path=True).eval()
    with pytest.raises(ValueError, match='divisible by 16'):
        m(torch.zeros(1, 14, 31, 48))
