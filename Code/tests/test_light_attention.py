"""Analytic fixtures for lightweight attention cores; residuals are separate."""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "detector"))


def eca(**kwargs):
    from core.models.backbones.light_attention import ECACore
    return ECACore(**kwargs)


def test_eca_module_exists():
    assert (ROOT / "detector/core/models/backbones/light_attention.py").exists()


@pytest.mark.parametrize("kernel", [1, 3, 5])
def test_eca_has_exactly_kernel_weights_and_initial_gate_half(kernel):
    m = eca(kernel_size=kernel)
    assert sum(p.numel() for p in m.parameters()) == kernel
    assert isinstance(m.channel_conv, nn.Conv1d) and m.channel_conv.bias is None
    assert m.channel_conv.kernel_size == (kernel,)
    assert m.channel_conv.padding == ((kernel - 1) // 2,)
    assert torch.equal(m.channel_conv.weight, torch.zeros_like(m.channel_conv.weight))
    x = torch.randn(2, 48, 5, 7)
    before = x.clone()
    assert torch.equal(m.channel_weights(x), torch.full((2, 48, 1, 1), 0.5))
    assert torch.equal(m(x), x * 0.5) and torch.equal(x, before)


def test_eca_known_kernel_correlates_across_channels_and_broadcasts_spatially():
    m = eca()
    with torch.no_grad():
        m.channel_conv.weight.copy_(torch.tensor([[[0.1, 0.2, 0.3]]]))
    descriptors = torch.tensor([1., 2., 4., 8.])
    spatial = torch.tensor([[-1., 0., 1.], [-1., 0., 1.]])
    x = descriptors.reshape(1, 4, 1, 1) + spatial.reshape(1, 1, 2, 3)
    gates = torch.sigmoid(torch.tensor([0.8, 1.7, 3.4, 2.0])).reshape(1, 4, 1, 1)
    torch.testing.assert_close(m.channel_weights(x), gates)
    torch.testing.assert_close(m(x), x * gates)


def test_eca_invalid_kernel_fails():
    for kernel in [0, -1, 2, True, 3.0, "3", None]:
        with pytest.raises(ValueError, match="kernel_size"):
            eca(kernel_size=kernel)


def test_eca_invalid_input_fails():
    for shape, dtype in [
        ((2, 48, 0, 5), torch.float32), ((48, 5, 7), torch.float32),
        ((1, 0, 5, 7), torch.float32), ((1, 48, 5, 7), torch.int64),
    ]:
        with pytest.raises(ValueError, match="input"):
            eca()(torch.zeros(shape, dtype=dtype))


def test_eca_zero_and_single_channel_single_element_are_finite():
    m = eca(kernel_size=5)
    assert torch.equal(m(torch.zeros(1, 1, 1, 1)), torch.zeros(1, 1, 1, 1))
    assert torch.equal(m(torch.tensor([[[[6.]]]])), torch.tensor([[[[3.]]]]))


def test_eca_zero_initialized_kernel_receives_nonzero_gradient():
    m = eca()
    x = torch.arange(1, 17, dtype=torch.float32).reshape(1, 4, 2, 2).requires_grad_()
    m(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 0.5), rtol=0, atol=0)
    grad = m.channel_conv.weight.grad
    assert grad is not None and torch.isfinite(grad).all() and (grad > 0).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_eca_pooling_is_fp32_with_explicit_low_precision_weights_and_returns_input_dtype(dtype):
    m = eca(kernel_size=1).to(dtype=dtype)
    with torch.no_grad():
        m.channel_conv.weight.fill_(0.001)
    x = torch.tensor([2048., 1., 1., 1.], dtype=dtype).reshape(1, 1, 2, 2)
    expected_gate = torch.sigmoid(x.float().mean() * m.channel_conv.weight.float().reshape(()))
    torch.testing.assert_close(m.channel_weights(x), expected_gate.reshape(1, 1, 1, 1), rtol=0, atol=0)
    y = m(x)
    assert y.dtype == dtype and y.device == x.device
    torch.testing.assert_close(y, x * expected_gate.to(dtype), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_eca_cpu_autocast_preserves_fp32_gates_and_kernel_gradient(dtype):
    m = eca()
    with torch.no_grad():
        m.channel_conv.weight.fill_(0.01)
    x = torch.randn(2, 48, 5, 7, requires_grad=True)
    with torch.autocast("cpu", dtype=dtype):
        gates = m.channel_weights(x)
        y = m(x)
    assert gates.dtype == torch.float32 and y.dtype == x.dtype
    assert torch.isfinite(y).all()
    y.square().sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(m.channel_conv.weight.grad).all()


def test_eca_eval_batch_isolation_and_state_reload():
    torch.manual_seed(310)
    m = eca(kernel_size=5).eval()
    with torch.no_grad():
        m.channel_conv.weight.normal_(0, 0.01)
    x = torch.randn(2, 48, 5, 7)
    expected = m(x)
    other = x.clone()
    other[1].add_(100)
    torch.testing.assert_close(m(other)[0], expected[0], rtol=0, atol=0)
    restored = eca(kernel_size=5).eval()
    restored.load_state_dict(m.state_dict(), strict=True)
    torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)


def test_eca_meta_full_stride4_shape_for_later_parameter_audit():
    with torch.device("meta"):
        m = eca()
        x = torch.empty(1, 48, 200, 176)
    assert m(x).shape == x.shape
    assert sum(p.numel() for p in m.parameters()) == 3


def simam(**kwargs):
    from core.models.backbones.light_attention import SimAMCore
    return SimAMCore(**kwargs)


@pytest.mark.parametrize("value,shape", [(0., (2, 3, 5, 7)), (3., (1, 4, 3, 5)), (7., (1, 1, 1, 1))])
def test_simam_constant_and_single_element_gate_is_sigmoid_half(value, shape):
    m = simam()
    assert sum(p.numel() for p in m.parameters()) == 0
    assert not m.state_dict()
    x = torch.full(shape, value)
    expected = torch.sigmoid(torch.tensor(0.5))
    torch.testing.assert_close(m.spatial_weights(x), torch.full_like(x, expected), rtol=0, atol=0)
    torch.testing.assert_close(m(x), x * expected, rtol=0, atol=0)


def test_simam_analytic_impulse_uses_unbiased_variance_without_channel_or_batch_mixing():
    m = simam(lambda_value=0.25)
    # Four spatial values [4,0,0,0]: mean=1, squared deviations [9,1,1,1], variance=4.
    x = torch.zeros(2, 3, 2, 2)
    x[0, 1, 0, 0] = 4
    expected = torch.full_like(x, torch.sigmoid(torch.tensor(0.5)))
    expected[0, 1] = torch.sigmoid(torch.tensor([[9., 1.], [1., 1.]]) / 17 + 0.5)
    torch.testing.assert_close(m.spatial_weights(x), expected, rtol=0, atol=0)
    torch.testing.assert_close(m(x), x * expected, rtol=0, atol=0)
    other = x.clone()
    other[1].normal_(10, 3)
    torch.testing.assert_close(m(other)[0], m(x)[0], rtol=0, atol=0)


def test_simam_invalid_lambda_fails():
    for value in [0, -1, True, "0.001", None, float("nan"), float("inf"), 10**400]:
        with pytest.raises(ValueError, match="lambda"):
            simam(lambda_value=value)


def test_simam_invalid_input_fails():
    for shape, dtype in [
        ((1, 3, 0, 5), torch.float32), ((3, 5, 7), torch.float32),
        ((1, 0, 5, 7), torch.float32), ((1, 3, 5, 7), torch.int64),
    ]:
        with pytest.raises(ValueError, match="input"):
            simam()(torch.zeros(shape, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_simam_statistics_are_fp32_and_output_preserves_dtype_device_and_input(dtype):
    m = simam(lambda_value=0.01)
    x = torch.tensor([2048., 1., 1., 1.], dtype=dtype).reshape(1, 1, 2, 2)
    before = x.clone()
    values = x.float()
    differences = (values - values.mean((2, 3), keepdim=True)).square()
    variance = differences.sum((2, 3), keepdim=True) / 3
    expected_gate = torch.sigmoid(differences / (4 * (variance + 0.01)) + 0.5)
    gates = m.spatial_weights(x)
    torch.testing.assert_close(gates, expected_gate, rtol=0, atol=0)
    assert gates.dtype == torch.float32 and gates.device == x.device
    y = m(x)
    assert y.dtype == dtype and y.device == x.device and torch.isfinite(y).all()
    torch.testing.assert_close(y, x * expected_gate.to(dtype), rtol=0, atol=0)
    assert torch.equal(x, before)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_simam_cpu_autocast_and_sparse_backward_are_finite(dtype):
    x = torch.zeros(2, 3, 5, 7)
    x[0, 1, 2, 3] = 4
    x[1, 0, 3, 2] = -3
    x.requires_grad_()
    m = simam()
    with torch.autocast("cpu", dtype=dtype):
        gates = m.spatial_weights(x)
        y = m(x)
    assert gates.dtype == torch.float32 and y.dtype == x.dtype
    y.square().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0


def test_simam_meta_stride4_shape_and_parameter_free_core():
    with torch.device("meta"):
        m = simam()
        x = torch.empty(1, 48, 200, 176)
    assert m(x).shape == x.shape
    assert sum(p.numel() for p in m.parameters()) == 0


def adapter(kind, **kwargs):
    from core.models.backbones.light_attention import build_light_attention
    return build_light_attention(kind, **kwargs)


def test_adapter_none_is_parameter_and_state_free_identity():
    m = adapter("none")
    assert isinstance(m, nn.Identity) and not m.state_dict()
    assert sum(p.numel() for p in m.parameters()) == 0
    x = torch.randn(2, 48, 5, 7)
    assert m(x) is x


@pytest.mark.parametrize("kind,settings,count", [
    ("eca", {}, 4), ("eca", {"eca_kernel_size": 5}, 6), ("simam", {}, 1),
])
def test_adapter_total_parameters_include_single_registered_scalar_beta(kind, settings, count):
    m = adapter(kind, **settings)
    assert sum(p.numel() for p in m.parameters()) == count
    assert m.beta.shape == () and isinstance(m.beta, nn.Parameter)
    torch.testing.assert_close(m.beta, torch.tensor(0.001), rtol=0, atol=0)
    assert len({id(p) for p in m.parameters()}) == len(list(m.parameters()))


@pytest.mark.parametrize("kind", ["eca", "simam"])
def test_adapter_beta_zero_exact_identity_and_core_gradient_behavior(kind):
    torch.manual_seed(410)
    m = adapter(kind, layer_scale_init=0)
    x = torch.randn(2, 48, 5, 7, requires_grad=True)
    before = x.detach().clone()
    y = m(x)
    assert torch.equal(y, x) and torch.equal(x.detach(), before)
    y.square().sum().backward()
    torch.testing.assert_close(x.grad, 2 * x.detach(), rtol=0, atol=0)
    assert torch.isfinite(m.beta.grad) and m.beta.grad.abs() > 0
    for p in m.core.parameters():
        assert p.grad is not None and torch.equal(p.grad, torch.zeros_like(p.grad))


def test_adapter_eca_initial_scale_and_nonzero_beta_kernel_gradients():
    m = adapter("eca")
    x = torch.arange(1, 17, dtype=torch.float32).reshape(1, 4, 2, 2).requires_grad_()
    y = m(x)
    torch.testing.assert_close(y, x * 0.9995, atol=1e-6, rtol=1e-6)
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 0.9995))
    assert torch.isfinite(m.beta.grad) and m.beta.grad.abs() > 0
    grad = m.core.channel_conv.weight.grad
    assert torch.isfinite(grad).all() and (grad > 0).all()
    assert set(m.state_dict()) == {"beta", "core.channel_conv.weight"}


def test_adapter_simam_beta_and_input_gradients_are_finite():
    m = adapter("simam")
    x = torch.arange(1, 17, dtype=torch.float32).reshape(1, 4, 2, 2).requires_grad_()
    m(x).square().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert torch.isfinite(m.beta.grad) and m.beta.grad.abs() > 0


def test_adapter_learned_beta_is_not_clamped_after_initialization():
    m = adapter("eca")
    with torch.no_grad():
        m.beta.fill_(3)
    x = torch.randn(2, 48, 5, 7)
    torch.testing.assert_close(m(x), -0.5 * x)


def test_adapter_invalid_or_ignored_options_fail():
    for kind, settings in [
        ("other", {}), (True, {}), ("none", {"layer_scale_init": 0}),
        ("none", {"eca_kernel_size": 3}), ("eca", {"simam_lambda": 0.001}),
        ("simam", {"eca_kernel_size": 3}), ("eca", {"eca_kernel_size": 2}),
        ("simam", {"simam_lambda": 0}), ("eca", {"layer_scale_init": True}),
        ("simam", {"layer_scale_init": -1}), ("eca", {"layer_scale_init": float("nan")}),
    ]:
        with pytest.raises(ValueError):
            adapter(kind, **settings)


@pytest.mark.parametrize("kind", ["eca", "simam"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_adapter_explicit_dtype_restore_and_identity(kind, dtype):
    m = adapter(kind).to(dtype=dtype).eval()
    x = torch.randn(2, 48, 5, 7, dtype=dtype)
    before = x.clone()
    y = m(x)
    assert y.dtype == dtype and y.device == x.device and torch.isfinite(y).all()
    assert torch.equal(x, before)
    restored = adapter(kind).to(dtype=dtype).eval()
    restored.load_state_dict(m.state_dict(), strict=True)
    torch.testing.assert_close(restored(x), y, rtol=0, atol=0)
    with torch.no_grad():
        m.beta.zero_()
    assert torch.equal(m(x), x)


@pytest.mark.parametrize("kind", ["eca", "simam"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_adapter_cpu_autocast_preserves_input_dtype_and_gradients(kind, dtype):
    m = adapter(kind)
    x = torch.randn(2, 48, 5, 7, requires_grad=True)
    with torch.autocast("cpu", dtype=dtype):
        y = m(x)
        loss = y.square().sum()
    assert y.dtype == x.dtype and torch.isfinite(y).all()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    for p in m.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


@pytest.mark.parametrize("kind", ["none", "eca", "simam"])
def test_adapter_meta_stride4_shape(kind):
    with torch.device("meta"):
        m = adapter(kind)
        x = torch.empty(1, 48, 200, 176)
    assert m(x).shape == x.shape
