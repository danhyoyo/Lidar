"""Fixed normalized objectives, gradient parity and guarded group routing."""

import copy
import importlib
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "detector"), str(ROOT / "detector/core/datasets")]
from core.losses.loss_fn import LossFunction
from core.task_groups import resolve_task_groups

OBJECTS = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
DEFINITIONS = [{"name": "car", "classes": ["Car"]},
               {"name": "ped_cyc", "classes": ["Pedestrian", "Cyclist"]}]


def groups():
    return resolve_task_groups(DEFINITIONS, OBJECTS)


def build(classification="gaussian", config=None, task_groups=None, **kwargs):
    factory = importlib.import_module("core.losses.loss_fn").build_loss_function
    return factory(classification, config, task_groups=task_groups, **kwargs)


def tensors(classification="gaussian", iqa=False, empty_vru=False):
    predictions, targets = {}, {}
    for group in groups():
        channels = group.num_classes + int(classification == "binary")
        predictions[group.name] = {
            "cls": torch.full((1, channels, 4, 6), .2, requires_grad=True),
            "offset": torch.full((1, 2, 4, 6), .1, requires_grad=True),
            "size": torch.full((1, 2, 4, 6), .2, requires_grad=True),
            "yaw": torch.full((1, 2, 4, 6), .3, requires_grad=True),
        }
        if iqa:
            predictions[group.name]["iou"] = torch.zeros(1, 1, 4, 6, requires_grad=True)
        cls_shape = (1, 4, 6) if classification == "binary" else (1, group.num_classes, 4, 6)
        cls = torch.zeros(cls_shape, dtype=torch.int64 if classification == "binary" else torch.float32)
        mask = torch.zeros(1, 4, 6)
        if not (empty_vru and group.name == "ped_cyc"):
            if classification == "binary":
                cls[0, 2, 3] = group.num_classes
            else:
                cls[0, group.num_classes - 1, 2, 3] = 1.
            mask[0, 2, 3] = 1.
        targets[group.name] = {"cls": cls, "offset": torch.full((1, 2, 4, 6), .25),
                               "size": torch.full((1, 2, 4, 6), -.2),
                               "yaw": torch.cat([torch.ones(1, 1, 4, 6), torch.zeros(1, 1, 4, 6)], 1),
                               "reg_mask": mask}
    return {"groups": predictions}, {"groups": targets}


def clone_predictions(prediction):
    return {"groups": {name: {key: tensor.detach().clone().requires_grad_()
                               for key, tensor in pred.items()}
                       for name, pred in prediction["groups"].items()}}


@pytest.mark.parametrize("classification", ["gaussian", "binary"])
@pytest.mark.parametrize("iqa", [False, True])
@pytest.mark.parametrize("weighted", [False, True])
def test_grouped_total_and_leaf_gradients_equal_explicit_weighted_sum(classification, iqa, weighted):
    config = {"name": "baseline", "use_iou": iqa}
    if weighted:
        config["group_weights"] = {"ped_cyc": 3., "car": 1.}
    saved = copy.deepcopy(config)
    criterion = build(classification, config, groups())
    prediction, target = tensors(classification, iqa)
    cloned = clone_predictions(prediction)
    weights = {"car": .25, "ped_cyc": .75} if weighted else {"car": .5, "ped_cyc": .5}
    strategy_config = {k: v for k, v in config.items() if k != "group_weights"}
    expected = sum(weights[g.name] * LossFunction(classification, copy.deepcopy(strategy_config))(
        cloned["groups"][g.name], target["groups"][g.name])["loss"] for g in groups())
    result = criterion(prediction, target)
    torch.testing.assert_close(result["loss"], expected, rtol=0, atol=0)
    assert result["loss"].requires_grad
    assert config == saved
    expected.backward()
    result["loss"].backward()
    for name in prediction["groups"]:
        assert result[f"group/{name}/group_weight"] == weights[name]
        for key in prediction["groups"][name]:
            torch.testing.assert_close(prediction["groups"][name][key].grad,
                                       cloned["groups"][name][key].grad, rtol=0, atol=0)
    assert all(torch.is_tensor(t) and t.ndim == 0 for t in result.values())
    assert all(not t.requires_grad for key, t in result.items() if key != "loss")
    for key in ("cls", "offset", "size", "yaw", "geo") + (("iou",) if iqa else ()):
        expected_metric = sum(weights[name] * result[f"group/{name}/{key}"] for name in weights)
        torch.testing.assert_close(result[key], expected_metric, rtol=0, atol=0)


@pytest.mark.parametrize("classification", ["gaussian", "binary"])
def test_empty_group_keeps_fixed_weight_negative_classification_and_graph_zeros(classification):
    criterion = build(classification, {"name": "baseline", "use_iou": True}, groups())
    prediction, target = tensors(classification, True, empty_vru=True)
    result = criterion(prediction, target)
    assert result["group/ped_cyc/group_weight"] == result["group/car/group_weight"] == .5
    assert result["group/ped_cyc/cls"] > 0
    for key in ("offset", "size", "yaw", "geo", "iou"):
        assert result[f"group/ped_cyc/{key}"] == 0
    torch.testing.assert_close(result["loss"].detach(),
                               .5 * result["group/car/loss"] + .5 * result["group/ped_cyc/loss"])
    result["loss"].backward()
    for key, tensor in prediction["groups"]["ped_cyc"].items():
        assert tensor.grad is not None and torch.isfinite(tensor.grad).all()
        if key == "cls":
            assert tensor.grad.abs().sum() > 0
        else:
            assert tensor.grad.abs().sum() == 0


@pytest.mark.parametrize("strategy", ["baseline", "oga"])
def test_singleton_group_matches_legacy_loss_and_gradients_after_state_alignment(strategy):
    singleton = resolve_task_groups([{"name": "all", "classes": list(OBJECTS)}], OBJECTS)
    config = {"name": strategy, "use_iou": False}
    criterion = build("gaussian", config, singleton)
    legacy = LossFunction("gaussian", copy.deepcopy(config))
    legacy.load_state_dict(criterion.criteria["all"].state_dict(), strict=True)
    prediction, target = tensors()
    pred = prediction["groups"]["ped_cyc"]
    pred["cls"] = torch.full((1, 3, 4, 6), .2, requires_grad=True)
    tgt = target["groups"]["ped_cyc"]
    tgt["cls"] = torch.zeros(1, 3, 4, 6)
    tgt["cls"][0, 1, 2, 3] = 1.
    other = {key: value.detach().clone().requires_grad_() for key, value in pred.items()}
    first = criterion({"groups": {"all": pred}}, {"groups": {"all": tgt}})
    second = legacy(other, tgt)
    torch.testing.assert_close(first["loss"], second["loss"], rtol=0, atol=0)
    first["loss"].backward()
    second["loss"].backward()
    for key in pred:
        torch.testing.assert_close(pred[key].grad, other[key].grad, rtol=0, atol=0)


@pytest.mark.parametrize("weights", [{}, {"car": 1}, {"car": 1, "ped_cyc": 1, "other": 1},
    {"car": 0, "ped_cyc": 1}, {"car": float("inf"), "ped_cyc": 1},
    {"car": float("nan"), "ped_cyc": 1}, {"car": True, "ped_cyc": 1}, None])
def test_direct_grouped_factory_rejects_invalid_weights(weights):
    with pytest.raises(ValueError, match="group_weights"):
        build("gaussian", {"group_weights": weights}, groups())


def test_factory_keeps_legacy_facade_and_strict_state_keys():
    config = {"name": "oga"}
    direct = LossFunction("gaussian", config)
    factory = build("gaussian", config)
    assert type(factory) is LossFunction
    assert direct.state_dict().keys() == factory.state_dict().keys()
    factory.load_state_dict(direct.state_dict(), strict=True)
    with pytest.raises(ValueError, match="legacy_single"):
        build("gaussian", {"group_weights": {}})
    with pytest.raises(ValueError, match="legacy_single"):
        build("gaussian", config, groups(), head_mode="legacy_single")
    with pytest.raises(ValueError, match="task_groups"):
        build("gaussian", config, head_mode="grouped")


@pytest.mark.parametrize("classification,config", [
    ("gaussian", {"name": "uwag", "use_iou": True}),
    ("gaussian", {"name": "q_oga", "quality_target": "rotated_iou", "use_iou": True}),
    ("binary", {"name": "q_oga"}), ("binary", {"name": "gw_qal"}),
])
def test_direct_grouped_construction_enforces_loss_capabilities(classification, config):
    with pytest.raises(ValueError):
        build(classification, config, groups())


@pytest.mark.parametrize("invalid", ["missing_group", "extra_group", "flat", "wrong_width",
                                     "extra_iqa", "missing_iqa", "wrong_iqa_shape"])
def test_grouped_forward_rejects_malformed_routes_before_computing_loss(invalid):
    iqa = invalid in ("missing_iqa", "wrong_iqa_shape")
    criterion = build("gaussian", {"name": "baseline", "use_iou": iqa}, groups())
    prediction, target = tensors(iqa=iqa)
    if invalid == "missing_group":
        del target["groups"]["car"]
    elif invalid == "extra_group":
        prediction["groups"]["unknown"] = prediction["groups"]["car"]
    elif invalid == "flat":
        prediction = prediction["groups"]["car"]
    elif invalid == "wrong_width":
        prediction["groups"]["car"]["cls"] = torch.zeros(1, 3, 4, 6)
    elif invalid == "extra_iqa":
        prediction["groups"]["ped_cyc"]["iou"] = torch.zeros(1, 1, 4, 6)
    elif invalid == "missing_iqa":
        del prediction["groups"]["ped_cyc"]["iou"]
    else:
        prediction["groups"]["ped_cyc"]["iou"] = torch.zeros(1, 2, 4, 6)
    with pytest.raises((ValueError, KeyError)):
        criterion(prediction, target)
