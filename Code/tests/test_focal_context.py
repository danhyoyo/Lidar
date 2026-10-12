"""Operator fixtures for the staged feature-based focal context implementation."""

import io
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "detector"))


def module(**kwargs):
    from core.models.backbones.focal_context import FocalContext
    return FocalContext(**kwargs)


def test_projection_module_exists():
    assert (ROOT / "detector/core/models/backbones/focal_context.py").exists()


@pytest.mark.parametrize("channels,bottleneck,dilations,shape", [
    (96, 64, (1, 2, 3), (2, 96, 7, 11)),
    (8, 4, (1, 3), (1, 8, 3, 5)),
])
def test_projection_split_matches_prenorm_and_single_bias_free_projection(channels, bottleneck, dilations, shape):
    torch.manual_seed(51)
    m = module(channels=channels, bottleneck=bottleneck, dilations=dilations).eval()
    x = torch.randn(shape)
    before = x.clone()
    query, seed, logits = m.project(x)
    expected = m.in_projection(m.pre_norm(x))
    assert expected.shape == (shape[0], 2 * bottleneck + len(dilations) + 1, *shape[2:])
    torch.testing.assert_close(torch.cat((query, seed, logits), dim=1), expected, rtol=0, atol=0)
    assert query.shape == seed.shape == (shape[0], bottleneck, *shape[2:])
    assert logits.shape == (shape[0], len(dilations) + 1, *shape[2:])
    assert all(t.dtype == x.dtype and t.device == x.device for t in (query, seed, logits))
    assert torch.equal(x, before)
    assert isinstance(m.pre_norm, nn.BatchNorm2d)
    assert m.pre_norm.eps == 1e-5 and m.pre_norm.momentum == 0.1
    assert m.pre_norm.affine and m.pre_norm.track_running_stats
    assert torch.equal(m.pre_norm.weight, torch.ones(channels))
    assert torch.equal(m.pre_norm.bias, torch.zeros(channels))
    assert all(c.bias is None for c in m.modules() if isinstance(c, nn.Conv2d))


def test_projection_invalid_settings_fail_at_construction():
    invalid_settings = [
        {"channels": 0}, {"channels": True}, {"channels": 96.0},
        {"bottleneck": 0}, {"bottleneck": True}, {"bottleneck": 64.0},
        {"version": 0}, {"version": True}, {"version": 1.0},
        {"dilations": []}, {"dilations": [1, 1]}, {"dilations": [2, 1]},
        {"dilations": [True, 2]}, {"dilations": [1.0, 2]},
        {"layer_scale_init": -1}, {"layer_scale_init": True},
        {"layer_scale_init": float("nan")}, {"layer_scale_init": float("inf")},
    ]
    for settings in invalid_settings:
        with pytest.raises(ValueError):
            module(**settings)


def test_projection_invalid_input_is_rejected():
    invalid_inputs = [
        ((2, 95, 7, 11), torch.float32), ((96, 7, 11), torch.float32),
        ((1, 96, 0, 3), torch.float32), ((1, 96, 3, 3), torch.int64),
    ]
    for shape, dtype in invalid_inputs:
        with pytest.raises(ValueError, match="input"):
            module().eval().project(torch.zeros(shape, dtype=dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_projection_keeps_explicit_cpu_dtype_and_device(dtype):
    m = module(channels=8, bottleneck=4).to(dtype=dtype).eval()
    x = torch.randn(1, 8, 3, 5, dtype=dtype)
    for t in m.project(x):
        assert t.dtype == dtype and t.device == x.device and torch.isfinite(t).all()


def test_projection_eval_single_element_is_supported_and_train_bn_constraint_is_explicit():
    m = module(channels=8, bottleneck=4).eval()
    assert m.project(torch.zeros(1, 8, 1, 1))[0].shape == (1, 4, 1, 1)
    with pytest.raises(ValueError, match="more than 1 value per channel"):
        m.train().project(torch.zeros(1, 8, 1, 1))


def test_levels_are_sequential_with_known_center_weights():
    m = module(channels=2, bottleneck=2).eval()
    with torch.no_grad():
        for index, conv in enumerate(m.level_convs, start=1):
            conv.weight.zero_()
            conv.weight[:, 0, 1, 1] = index
            assert conv.groups == conv.in_channels == conv.out_channels == 2
            assert conv.dilation == conv.padding == (index, index)
            assert conv.kernel_size == (3, 3) and conv.bias is None
    seed = torch.full((1, 2, 5, 7), 0.5)
    levels, pooled = m.context_levels(seed)
    expected = seed
    for index, level in enumerate(levels, start=1):
        expected = torch.nn.functional.silu(expected * index)
        torch.testing.assert_close(level, expected)
    assert not torch.allclose(levels[1], torch.nn.functional.silu(seed * 2))
    torch.testing.assert_close(pooled, expected.float().mean((2, 3), keepdim=True))
    assert pooled.shape == (1, 2, 1, 1)


def test_levels_impulse_has_cumulative_receptive_field_without_cross_channel_mix():
    m = module(channels=2, bottleneck=2).eval()
    with torch.no_grad():
        for conv in m.level_convs:
            conv.weight.fill_(1)
    seed = torch.zeros(1, 2, 21, 25)
    seed[0, 0, 10, 12] = 1
    before = seed.clone()
    levels, _ = m.context_levels(seed)
    for radius, level in zip((1, 3, 6), levels):
        expected_support = torch.zeros_like(seed, dtype=torch.bool)
        expected_support[0, 0, 10-radius:11+radius, 12-radius:13+radius] = True
        assert torch.equal(level > 0, expected_support)
    assert torch.equal(seed, before)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_pooling_and_gates_use_fp32_reductions_and_keep_device(dtype):
    m = module(channels=2, bottleneck=2).to(dtype=dtype).eval()
    seed = torch.linspace(-4, 7, 70).reshape(1, 2, 5, 7).to(dtype)
    levels, pooled = m.context_levels(seed)
    assert pooled.dtype == torch.float32 and pooled.device == seed.device
    torch.testing.assert_close(pooled, levels[-1].float().mean((2, 3), keepdim=True), rtol=0, atol=0)
    logits = torch.tensor([10000, 9992, -10000, 0], dtype=dtype).view(1, 4, 1, 1).expand(1, 4, 5, 7)
    weights = m.gate_weights(logits)
    expected = torch.softmax(logits.float(), dim=1).to(dtype)
    torch.testing.assert_close(weights, expected, rtol=0, atol=0)
    assert weights.dtype == dtype and weights.device == seed.device
    assert torch.isfinite(weights).all() and (weights >= 0).all()
    torch.testing.assert_close(weights.float().sum(1), torch.ones(1, 5, 7), atol=0.004, rtol=0)


def test_gates_uniform_and_selected_global_context_broadcast():
    m = module(channels=2, bottleneck=2).eval()
    seed = torch.arange(70, dtype=torch.float32).reshape(1, 2, 5, 7) / 70
    logits = torch.zeros(1, 4, 5, 7)
    fused, weights = m.context_from_seed(seed, logits)
    levels, pooled = m.context_levels(seed)
    assert torch.equal(weights, torch.full_like(weights, 0.25))
    expected = (levels[0] + levels[1] + levels[2] + pooled) / 4
    torch.testing.assert_close(fused, expected)
    global_logits = torch.full_like(logits, -10000)
    global_logits[:, -1] = 10000
    global_fused, global_weights = m.context_from_seed(seed, global_logits)
    assert torch.equal(global_weights[:, -1], torch.ones_like(global_weights[:, -1]))
    torch.testing.assert_close(global_fused, pooled.expand_as(seed), rtol=0, atol=0)
    assert fused.dtype == seed.dtype and fused.device == seed.device


def test_levels_do_not_hard_mask_zero_seed_cells():
    m = module(channels=1, bottleneck=1).eval()
    with torch.no_grad():
        for conv in m.level_convs:
            conv.weight.fill_(1)
    seed = torch.zeros(1, 1, 15, 17)
    seed[:, :, 7, 8] = 1
    fused, _ = m.context_from_seed(seed, torch.zeros(1, 4, 15, 17))
    assert fused[0, 0, 7, 9] > 0 and seed[0, 0, 7, 9] == 0
    # Global context even reaches locations outside the largest local support.
    assert fused[0, 0, 0, 0] > 0


def test_pooling_eval_batch_item_isolation_through_projection_levels_and_gates():
    torch.manual_seed(101)
    m = module(channels=8, bottleneck=4).eval()
    x = torch.randn(2, 8, 5, 7)
    first = m.prepare_context(x)
    changed = x.clone()
    changed[1] = changed[1] * 20 + 100
    second = m.prepare_context(changed)
    for a, b in zip(first, second):
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
        assert not torch.equal(a[1], b[1])
    assert tuple(b for b in m.modules() if isinstance(b, nn.BatchNorm2d)) == (m.pre_norm, m.out_norm)


def test_levels_and_gates_preserve_gradient_paths():
    torch.manual_seed(115)
    m = module(channels=8, bottleneck=4).eval()
    seed = torch.randn(2, 4, 5, 7, requires_grad=True)
    logits = torch.randn(2, 4, 5, 7, requires_grad=True)
    fused, _ = m.context_from_seed(seed, logits)
    fused.square().sum().backward()
    for grad in [seed.grad, logits.grad, *(c.weight.grad for c in m.level_convs)]:
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_levels_variant_number_controls_gate_width_and_pooled_broadcast():
    m = module(channels=8, bottleneck=4, dilations=(1, 3)).eval()
    query, fused, weights = m.prepare_context(torch.randn(1, 8, 3, 5))
    assert query.shape == fused.shape == (1, 4, 3, 5)
    assert weights.shape == (1, 3, 3, 5)
    torch.testing.assert_close(weights.sum(1), torch.ones(1, 3, 5))


@pytest.mark.parametrize("seed_shape,gate_shape", [
    ((1, 3, 5, 7), (1, 4, 5, 7)), ((1, 4, 5, 7), (1, 3, 5, 7)),
    ((1, 4, 5, 7), (2, 4, 5, 7)), ((1, 4, 5, 7), (1, 4, 7, 5)),
])
def test_levels_and_gates_reject_misaligned_component_shapes(seed_shape, gate_shape):
    m = module(channels=8, bottleneck=4).eval()
    with pytest.raises(ValueError):
        m.context_from_seed(torch.zeros(seed_shape), torch.zeros(gate_shape))


def test_parameters_match_complete_focal_operator_without_hidden_biases():
    m = module()
    assert sum(p.numel() for p in m.parameters()) == 25120
    assert m.context_projection.weight.shape == (64, 64, 1, 1)
    assert m.out_projection.weight.shape == (96, 64, 1, 1)
    assert m.gamma.shape == (1, 96, 1, 1)
    torch.testing.assert_close(m.gamma, torch.full_like(m.gamma, 0.001), rtol=0, atol=0)
    assert all(c.bias is None for c in m.modules() if isinstance(c, nn.Conv2d))
    assert tuple(b for b in m.modules() if isinstance(b, nn.BatchNorm2d)) == (m.pre_norm, m.out_norm)
    assert torch.equal(m.out_norm.weight, torch.ones(96))
    assert torch.equal(m.out_norm.bias, torch.zeros(96))


def test_identity_zero_gamma_is_exact_with_zero_branch_and_nonzero_scale_gradients():
    torch.manual_seed(202)
    m = module(channels=8, bottleneck=4, layer_scale_init=0).eval()
    x = torch.randn(2, 8, 5, 7, requires_grad=True)
    before = x.detach().clone()
    y = m(x)
    assert torch.equal(y, x) and torch.equal(x.detach(), before)
    y.square().sum().backward()
    torch.testing.assert_close(x.grad, 2 * x.detach(), rtol=0, atol=0)
    assert torch.isfinite(m.gamma.grad).all() and m.gamma.grad.abs().sum() > 0
    for name, p in m.named_parameters():
        if name != "gamma":
            assert p.grad is not None and torch.equal(p.grad, torch.zeros_like(p.grad))


@pytest.mark.parametrize("training", [False, True])
def test_gradient_nonzero_default_scale_reaches_every_branch_parameter(training):
    torch.manual_seed(204)
    m = module(channels=8, bottleneck=4).train(training)
    x = torch.randn(2, 8, 5, 7, requires_grad=True)
    before = x.detach().clone()
    y = m(x)
    assert y.shape == x.shape and y.dtype == x.dtype and y.device == x.device
    assert torch.equal(x.detach(), before) and not torch.equal(y, x)
    (y * torch.randn_like(y)).sum().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, name
    assert torch.isfinite(x.grad).all()


def test_identity_residual_matches_defined_modulation_and_channel_scale():
    torch.manual_seed(207)
    m = module(channels=8, bottleneck=4).eval()
    with torch.no_grad():
        m.gamma.copy_(torch.linspace(0.001, 0.008, 8).reshape(1, 8, 1, 1))
    x = torch.randn(2, 8, 3, 5)
    q, fused, _ = m.prepare_context(x)
    context = m.context_projection(fused)
    product = q.float() * context.float()
    expected = x + m.gamma * m.out_norm(m.out_projection(product))
    torch.testing.assert_close(m(x), expected, rtol=0, atol=0)


@pytest.mark.parametrize("training", [False, True])
def test_numerical_full_forward_maps_are_finite_without_inplace_input_mutation(training):
    torch.manual_seed(211)
    for kind in ["zero", "constant", "sparse", "random"]:
        m = module(channels=8, bottleneck=4).train(training)
        x = torch.zeros(2, 8, 5, 7)
        if kind == "constant":
            x.fill_(3)
        elif kind == "sparse":
            x[0, 1, 2, 3] = 7
            x[1, 3, 3, 4] = -5
        elif kind == "random":
            x.normal_()
        x.requires_grad_()
        before = x.detach().clone()
        y = m(x)
        assert y.shape == x.shape and y.dtype == x.dtype and torch.isfinite(y).all()
        assert torch.equal(x.detach(), before)
        y.square().mean().backward()
        assert torch.isfinite(x.grad).all()
        for p in m.parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all()


def test_state_reload_restores_bn_and_learned_channel_scale_exactly():
    torch.manual_seed(212)
    m = module(channels=8, bottleneck=4).train()
    for _ in range(3):
        m(torch.randn(2, 8, 5, 7) + 2)
    assert m.pre_norm.num_batches_tracked == m.out_norm.num_batches_tracked == 3
    with torch.no_grad():
        m.gamma.copy_(torch.linspace(-0.002, 0.007, 8).reshape_as(m.gamma))
    x = torch.randn(2, 8, 3, 5)
    m.eval()
    expected = m(x)
    state = {k: v.clone() for k, v in m.state_dict().items()}
    stream = io.BytesIO()
    torch.save(state, stream)
    stream.seek(0)
    restored = module(channels=8, bottleneck=4).eval()
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)
    torch.testing.assert_close(m(x), expected, rtol=0, atol=0)
    for key, value in m.state_dict().items():
        assert torch.equal(value, state[key]), key  # eval never updates BN
        assert torch.equal(restored.state_dict()[key], value), key


def test_bn_single_spatial_element_obeys_ordinary_training_batch_constraints():
    m = module(channels=8, bottleneck=4).eval()
    assert torch.isfinite(m(torch.ones(1, 8, 1, 1))).all()
    with pytest.raises(ValueError, match="more than 1 value per channel"):
        m.train()(torch.ones(1, 8, 1, 1))
    assert torch.isfinite(m(torch.randn(2, 8, 1, 1))).all()
    assert torch.isfinite(m(torch.randn(1, 8, 1, 2))).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_numerical_explicit_dtype_forward_and_zero_scale_identity(dtype):
    torch.manual_seed(214)
    m = module(channels=8, bottleneck=4).to(dtype=dtype).eval()
    x = torch.randn(2, 8, 5, 7, dtype=dtype)
    before = x.clone()
    y = m(x)
    assert y.dtype == dtype and y.device == x.device and torch.isfinite(y).all()
    assert torch.equal(x, before)
    with torch.no_grad():
        m.gamma.zero_()
    assert torch.equal(m(x), x)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cpu_autocast_modulation_product_is_fp32_before_projection_and_backward_is_finite(dtype):
    torch.manual_seed(215)
    m = module(channels=8, bottleneck=4).train()
    seen = {}
    handles = [
        m.in_projection.register_forward_hook(lambda _m, _i, out: seen.update(query=out[:, :4])),
        m.context_projection.register_forward_hook(lambda _m, _i, out: seen.update(context=out)),
        m.out_projection.register_forward_pre_hook(lambda _m, args: seen.update(product=args[0])),
    ]
    try:
        x = torch.randn(2, 8, 5, 7, requires_grad=True)
        with torch.autocast("cpu", dtype=dtype):
            y = m(x)
            loss = y.square().mean()
        assert seen["query"].dtype == seen["context"].dtype == dtype
        assert seen["product"].dtype == torch.float32
        torch.testing.assert_close(seen["product"], seen["query"].float() * seen["context"].float(), rtol=0, atol=0)
        assert y.dtype == x.dtype and torch.isfinite(y).all()
        loss.backward()
        for name, p in m.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert torch.isfinite(x.grad).all()
    finally:
        for handle in handles:
            handle.remove()


def test_eval_full_forward_is_isolated_between_batch_items():
    torch.manual_seed(216)
    m = module(channels=8, bottleneck=4).eval()
    x = torch.randn(2, 8, 5, 7)
    before = m(x)
    changed = x.clone()
    changed[1].mul_(4).add_(20)
    after = m(changed)
    torch.testing.assert_close(before[0], after[0], rtol=0, atol=0)
    assert not torch.equal(before[1], after[1])
