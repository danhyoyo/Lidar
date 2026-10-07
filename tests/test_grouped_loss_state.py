"""Independent adaptive parameters, EMA updates and criterion state parity."""

import copy
import io

import pytest
import torch

from test_grouped_losses import build, clone_predictions, groups, tensors
from test_grouped_header import header
from core.losses.loss_fn import LossFunction


STRATEGIES = ("oga", "uwag", "gw_qal", "q_oga")


def snapshot(module):
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


def assert_state_equal(module, expected):
    assert module.state_dict().keys() == expected.keys()
    for key, value in module.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_adaptive_parameters_and_buffers_are_registered_once_and_not_shared(strategy):
    criterion = build("gaussian", {"name": strategy}, groups())
    car, vru = criterion.criteria.values()
    expected = 8 if strategy == "uwag" else 10
    assert sum(p.numel() for p in criterion.parameters()) == expected
    assert len(list(criterion.parameters())) == 2
    assert {p.data_ptr() for p in car.parameters()}.isdisjoint(p.data_ptr() for p in vru.parameters())
    assert {b.data_ptr() for b in car.buffers()}.isdisjoint(b.data_ptr() for b in vru.buffers())
    assert all(key.startswith(("criteria.car.strategy.", "criteria.ped_cyc.strategy."))
               for key in criterion.state_dict())
    with torch.no_grad():
        next(car.parameters()).fill_(.7)
    assert torch.equal(next(vru.parameters()), torch.zeros_like(next(vru.parameters())))
    # Module traversal moves every registered group parameter and buffer.
    criterion.to("meta")
    assert all(value.device.type == "meta" for value in criterion.state_dict().values())


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_grouped_adaptive_objective_and_all_gradients_match_independent_legacy_criteria(strategy):
    config = {"name": strategy, "ema_momentum": .7,
              "group_weights": {"car": 1., "ped_cyc": 3.}}
    criterion = build("gaussian", config, groups())
    references = {name: LossFunction("gaussian", {k: v for k, v in config.items() if k != "group_weights"})
                  for name in criterion.criteria}
    for name, reference in references.items():
        reference.load_state_dict(criterion.criteria[name].state_dict(), strict=True)
    # Two updates exercise both initial calibration and later EMA behavior.
    for step in range(2):
        prediction, target = tensors(empty_vru=bool(step))
        with torch.no_grad():
            prediction["groups"]["car"]["offset"].add_(.13 * (step + 1))
            prediction["groups"]["ped_cyc"]["cls"].sub_(.4)
        cloned = clone_predictions(prediction)
        expected = sum(weight * references[name](cloned["groups"][name], target["groups"][name])["loss"]
                       for name, weight in criterion.group_weights)
        actual = criterion(prediction, target)
        torch.testing.assert_close(actual["loss"], expected, rtol=0, atol=0)
        actual["loss"].backward()
        expected.backward()
        for name in references:
            for key in prediction["groups"][name]:
                torch.testing.assert_close(prediction["groups"][name][key].grad,
                                           cloned["groups"][name][key].grad, rtol=0, atol=0)
            for parameter, reference in zip(criterion.criteria[name].parameters(), references[name].parameters()):
                assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
                torch.testing.assert_close(parameter.grad, reference.grad, rtol=0, atol=0)
            assert_state_equal(criterion.criteria[name], snapshot(references[name]))
        criterion.zero_grad(set_to_none=True)
        for reference in references.values():
            reference.zero_grad(set_to_none=True)


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_group_only_objective_updates_only_its_head_and_criterion(strategy):
    torch.manual_seed(71)
    heads = header(iqa=False).eval()
    criterion = build("gaussian", {"name": strategy}, groups()).train()
    feature = torch.randn(1, 32, 4, 6, requires_grad=True)
    prediction = heads(feature)
    _, target = tensors()
    other_head = snapshot(heads.heads["ped_cyc"])
    other_criterion = snapshot(criterion.criteria["ped_cyc"])
    optimizer = torch.optim.SGD([*heads.parameters(), *criterion.parameters()], lr=.01)
    loss = criterion.criteria["car"](prediction["groups"]["car"], target["groups"]["car"])["loss"]
    loss.backward()
    assert feature.grad is not None and feature.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in heads.heads["car"].parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in criterion.criteria["car"].parameters())
    assert all(p.grad is None for p in heads.heads["ped_cyc"].parameters())
    assert all(p.grad is None for p in criterion.criteria["ped_cyc"].parameters())
    optimizer.step()
    assert_state_equal(heads.heads["ped_cyc"], other_head)
    assert_state_equal(criterion.criteria["ped_cyc"], other_criterion)
    if strategy != "uwag":
        assert criterion.criteria["car"].weighting.initialized
        assert not criterion.criteria["ped_cyc"].weighting.initialized


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_eval_preserves_group_emas_and_criterion_roundtrip_preserves_next_training_update(strategy):
    criterion = build("gaussian", {"name": strategy, "ema_momentum": .6}, groups()).train()
    prediction, target = tensors(empty_vru=True)
    criterion(prediction, target)
    if strategy != "uwag":
        assert all(child.weighting.initialized for child in criterion.criteria.values())
        assert not torch.equal(criterion.criteria["car"].weighting.running_loss_means,
                               criterion.criteria["ped_cyc"].weighting.running_loss_means)
    criterion.eval()
    state = snapshot(criterion)
    prediction, target = tensors()
    expected = criterion(prediction, target)
    criterion(prediction, target)
    assert_state_equal(criterion, state)
    stream = io.BytesIO()
    torch.save(state, stream)
    stream.seek(0)
    restored = build("gaussian", copy.deepcopy(criterion.config), groups()).eval()
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    actual = restored(clone_predictions(prediction), target)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    criterion.train()
    restored.train()
    first, second = criterion(prediction, target), restored(clone_predictions(prediction), target)
    torch.testing.assert_close(first["loss"], second["loss"], rtol=0, atol=0)
    assert_state_equal(restored, snapshot(criterion))


@pytest.mark.parametrize("strategy", ("oga", "gw_qal", "q_oga"))
def test_invalid_later_group_shape_does_not_update_earlier_ema(strategy):
    criterion = build("gaussian", {"name": strategy}, groups())
    prediction, target = tensors()
    target["groups"]["ped_cyc"]["offset"] = torch.zeros(1, 2, 3, 6)
    before = snapshot(criterion)
    with pytest.raises(ValueError, match="shape mismatch"):
        criterion(prediction, target)
    assert_state_equal(criterion, before)
