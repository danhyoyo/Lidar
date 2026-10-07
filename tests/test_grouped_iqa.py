"""Group-local quality supervision and isolation from box-target gradients."""

import pytest
import torch
import torch.nn.functional as F

from test_grouped_losses import build, groups, tensors
from test_grouped_header import header
from core.losses.iou_targets import (compute_iou_targets, compute_mgiou_targets,
                                    compute_yaw_footprint_targets, compute_rotated_iou_targets)
from core.losses.loss_fn import LossFunction


METHODS = ("mgiou", "yaw_footprint", "rotated_iou")


def known_boxes(classification="gaussian", empty_vru=False):
    prediction, target = tensors(classification, iqa=True, empty_vru=empty_vru)
    # Different assigned geometry makes accidental reuse of Car targets
    # detectable even though both groups supervise the same output cell.
    target["groups"]["ped_cyc"]["offset"][:, 0].fill_(1.25)
    target["groups"]["ped_cyc"]["offset"][:, 1].fill_(-.65)
    target["groups"]["ped_cyc"]["size"][:, 0].fill_(-.6)
    target["groups"]["ped_cyc"]["size"][:, 1].fill_(.1)
    for name in prediction["groups"]:
        for key in ("offset", "size", "yaw"):
            prediction["groups"][name][key] = target["groups"][name][key].clone().requires_grad_()
        prediction["groups"][name]["iou"] = torch.full((1, 1, 4, 6), 1., requires_grad=True)
    # Both groups own cell (2,3), but Car is perfect and VRU is disjoint.
    with torch.no_grad():
        prediction["groups"]["ped_cyc"]["offset"][:, 0].add_(10.)
    return prediction, target


@pytest.mark.parametrize("strategy", ("baseline", "oga"))
@pytest.mark.parametrize("classification", ("gaussian", "binary"))
@pytest.mark.parametrize("method", METHODS)
def test_each_group_iqa_uses_its_own_assigned_box_at_the_same_cell(strategy, classification, method):
    weight = 1.7
    criterion = build(classification, {"name": strategy, "use_iou": True,
                                      "iou_target_type": method, "iou_loss_weight": weight}, groups()).eval()
    prediction, target = known_boxes(classification)
    result = criterion(prediction, target)
    qualities = {}
    for name in prediction["groups"]:
        pred, tgt = prediction["groups"][name], target["groups"][name]
        qualities[name] = compute_iou_targets(pred, tgt, method=method)
        mask = tgt["reg_mask"].bool()
        expected = F.binary_cross_entropy_with_logits(pred["iou"][:, 0][mask], qualities[name][mask])
        # Historical OGA reports raw BCE; baseline reports weighted BCE.
        multiplier = weight if strategy == "baseline" else 1.
        torch.testing.assert_close(result[f"group/{name}/iou"], multiplier * expected.detach())
        if strategy == "oga":
            torch.testing.assert_close(result[f"group/{name}/mean_iou_target"], qualities[name][mask].mean())
    assert qualities["car"][0, 2, 3] > .999
    # A mean projection GIoU can remain positive when boxes are disjoint
    # on x but perfectly aligned on y; it is a proxy, not polygon IoU.
    if method == "mgiou":
        assert 0 < qualities["ped_cyc"][0, 2, 3] < .1
    else:
        assert qualities["ped_cyc"][0, 2, 3] == 0
    assert result["group/car/iou"] < result["group/ped_cyc/iou"]
    # Reverse which group is perfect without changing assigned targets,
    # masks or class labels. Using the other group's box now gives a
    # different answer and cannot satisfy the expected BCE.
    with torch.no_grad():
        prediction["groups"]["car"]["offset"][:, 0].add_(10.)
        prediction["groups"]["ped_cyc"]["offset"].copy_(target["groups"]["ped_cyc"]["offset"])
    swapped = criterion(prediction, target)
    assert swapped["group/car/iou"] > swapped["group/ped_cyc/iou"]
    for name in prediction["groups"]:
        pred, tgt = prediction["groups"][name], target["groups"][name]
        mask = tgt["reg_mask"].bool()
        quality = compute_iou_targets(pred, tgt, method=method)
        expected = F.binary_cross_entropy_with_logits(pred["iou"][:, 0][mask], quality[mask])
        torch.testing.assert_close(swapped[f"group/{name}/iou"],
                                   (weight if strategy == "baseline" else 1.) * expected.detach())
    foreign = compute_iou_targets(prediction["groups"]["ped_cyc"], target["groups"]["car"], method=method)
    assert foreign[0, 2, 3] < .99


@pytest.mark.parametrize("strategy", ("baseline", "oga"))
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("empty_vru", (False, True))
def test_enabling_iqa_adds_only_quality_branch_gradients_with_detached_targets(strategy, method, empty_vru):
    coefficient = 1.8
    config = {"name": strategy, "use_iou": True, "iou_target_type": method,
              "iou_loss_weight": coefficient}
    criterion = build("gaussian", config, groups()).eval()
    prediction, target = known_boxes(empty_vru=empty_vru)
    # eval with default unit calibration gives identical box-task weights
    # for OGA with five tasks or six tasks. IQA is the only added objective.
    actual = criterion(prediction, target)
    actual["loss"].backward()
    for name, group_weight in criterion.group_weights:
        pred, tgt = prediction["groups"][name], target["groups"][name]
        reference_pred = {key: value.detach().clone().requires_grad_()
                          for key, value in pred.items() if key != "iou"}
        reference = LossFunction("gaussian", {"name": strategy}).eval()
        (group_weight * reference(reference_pred, tgt)["loss"]).backward()
        for key in reference_pred:
            torch.testing.assert_close(pred[key].grad, reference_pred[key].grad, rtol=1e-6, atol=1e-6)
        quality = compute_iou_targets(pred, tgt, method=method)
        assert not quality.requires_grad and quality.grad_fn is None
        mask = tgt["reg_mask"].bool()
        expected_gradient = torch.zeros_like(pred["iou"])
        if mask.any():
            expected_gradient[:, 0][mask] = (group_weight * coefficient *
                (pred["iou"].detach()[:, 0][mask].sigmoid() - quality[mask]) / mask.sum())
        torch.testing.assert_close(pred["iou"].grad, expected_gradient, rtol=1e-6, atol=1e-7)
        assert pred["iou"].grad is not None and torch.isfinite(pred["iou"].grad).all()
        if empty_vru and name == "ped_cyc":
            assert actual[f"group/{name}/iou"] == 0


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("direct", (False, True))
def test_quality_targets_have_no_gradient_to_prediction_or_assigned_box(method, direct):
    prediction, target = known_boxes()
    for name in prediction["groups"]:
        for key in ("offset", "size", "yaw"):
            target["groups"][name][key].requires_grad_()
        pred, tgt = prediction["groups"][name], target["groups"][name]
        if direct:
            generator = {"mgiou": compute_mgiou_targets, "yaw_footprint": compute_yaw_footprint_targets,
                         "rotated_iou": compute_rotated_iou_targets}[method]
            quality = generator(*(pred[key] for key in ("offset", "size", "yaw")),
                                *(tgt[key] for key in ("offset", "size", "yaw")), tgt["reg_mask"])
        else:
            quality = compute_iou_targets(pred, tgt, method=method)
        assert not quality.requires_grad
        mask = tgt["reg_mask"].bool()
        F.binary_cross_entropy_with_logits(pred["iou"][:, 0][mask], quality[mask]).backward()
        assert pred["iou"].grad is not None and pred["iou"].grad.abs().sum() > 0
        for key in ("cls", "offset", "size", "yaw"):
            assert pred[key].grad is None
        assert all(tgt[key].grad is None for key in ("offset", "size", "yaw"))


@pytest.mark.parametrize("strategy", ("baseline", "oga"))
@pytest.mark.parametrize("method", METHODS)
def test_iqa_objective_reaches_only_its_quality_head_and_shared_features(strategy, method):
    torch.manual_seed(82)
    heads = header(iqa=True).eval()
    criterion = build("gaussian", {"name": strategy, "use_iou": True,
                                  "iou_target_type": method}, groups()).eval()
    feature = torch.randn(1, 32, 4, 6, requires_grad=True)
    prediction = heads(feature)
    _, target = tensors(iqa=True)
    pred = prediction["groups"]["car"]
    # Detach only other predicted branches to isolate actual production IQA
    # supervision in the selected existing strategy, without recreating BCE.
    selected = {key: value if key == "iou" else value.detach() for key, value in pred.items()}
    criterion.criteria["car"](selected, target["groups"]["car"])["loss"].backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all() and feature.grad.abs().sum() > 0
    for name, parameter in heads.heads["car"].named_parameters():
        if name.startswith("iou."):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        else:
            assert parameter.grad is None
    assert all(parameter.grad is None for parameter in heads.heads["ped_cyc"].parameters())


@pytest.mark.parametrize("strategy", ("uwag", "gw_qal", "q_oga"))
def test_direct_grouped_factory_rejects_strategies_without_separate_iqa(strategy):
    with pytest.raises(ValueError, match="IQA"):
        build("gaussian", {"name": strategy, "use_iou": True}, groups())
