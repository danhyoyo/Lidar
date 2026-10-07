"""Measured binary training capabilities and inherited overlap limitations."""

import copy

import pytest
import torch

from test_grouped_losses import build, clone_predictions, groups, tensors
from test_grouped_header import header
from test_grouped_targets import GEOMETRY, make_dataset
from core.losses.iou_targets import compute_iou_targets
from core.losses.loss_fn import LossFunction


SUPPORTED = (("baseline", False), ("baseline", True), ("oga", False), ("oga", True), ("uwag", False))


@pytest.mark.parametrize("strategy,iqa", SUPPORTED)
@pytest.mark.parametrize("empty_vru", (False, True))
@pytest.mark.parametrize("backend", ("python", "numba"))
def test_binary_real_targets_support_two_optimizer_steps_with_local_background(tmp_path, strategy, iqa, empty_vru, backend):
    dataset = make_dataset(tmp_path, classification="binary", backend=backend)
    source = torch.from_numpy(dataset.get_boxes(0))
    if empty_vru:
        source = source[:1]
    raw = dataset.get_label(source, GEOMETRY)
    target = {"groups": {name: {key: value.unsqueeze(0) for key, value in item.items()}
                          for name, item in raw["groups"].items()}}
    for group in groups():
        cls = target["groups"][group.name]["cls"]
        assert cls.dtype == torch.int64 and cls.min() == 0
        assert cls.max() <= group.num_classes
    if not empty_vru:
        assert set(target["groups"]["ped_cyc"]["cls"].unique().tolist()) == {0, 1, 2}
    torch.manual_seed(93)
    heads = header("binary", iqa=iqa).train()
    criterion = build("binary", {"name": strategy, "use_iou": iqa,
                                 "iou_target_type": "rotated_iou"}, dataset.task_groups).train()
    parameters = [*heads.parameters(), *criterion.parameters()]
    assert len(parameters) == len({id(parameter) for parameter in parameters})
    optimizer = torch.optim.SGD(parameters, lr=.003)
    initial = heads.heads["car"].cls.head.weight.detach().clone()
    feature = torch.randn(1, 32, 16, 12)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        prediction = heads(feature)
        result = criterion(prediction, target)
        assert torch.isfinite(result["loss"])
        assert result["group/car/group_weight"] == result["group/ped_cyc/group_weight"] == .5
        result["loss"].backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in parameters)
        if empty_vru:
            assert result["group/ped_cyc/cls"] > 0
            for key in ("offset", "size", "yaw", "geo") + (("iou",) if iqa else ()):
                assert result[f"group/ped_cyc/{key}"] == 0
                if key != "geo":
                    assert all(parameter.grad.abs().sum() == 0 for parameter in getattr(heads.heads["ped_cyc"], key).parameters())
        optimizer.step()
        assert all(torch.isfinite(parameter).all() for parameter in parameters)
    assert not torch.equal(initial, heads.heads["car"].cls.head.weight)


@pytest.mark.parametrize("strategy", ("baseline", "oga", "uwag"))
@pytest.mark.parametrize("empty_vru", (False, True))
def test_supported_binary_total_and_gradients_match_group_local_legacy_criteria(strategy, empty_vru):
    criterion = build("binary", {"name": strategy}, groups())
    prediction, target = tensors("binary", empty_vru=empty_vru)
    cloned = clone_predictions(prediction)
    references = {name: LossFunction("binary", {"name": strategy}) for name in criterion.criteria}
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


@pytest.mark.parametrize("strategy", ("gw_qal", "q_oga"))
@pytest.mark.parametrize("name", ("car", "ped_cyc"))
@pytest.mark.parametrize("empty", (False, True))
def test_legacy_binary_quality_coupled_classification_is_not_end_to_end_supported(strategy, name, empty):
    prediction, target = tensors("binary", empty_vru=empty)
    pred, tgt = prediction["groups"][name], target["groups"][name]
    if empty:
        tgt["cls"].zero_()
        tgt["reg_mask"].zero_()
    legacy = LossFunction("binary", {"name": strategy})
    if strategy == "gw_qal" and empty:
        # Its empty branch falls back to ordinary softmax focal loss; this
        # says nothing about the populated CQFL branch.
        result = legacy(pred, tgt)
        assert torch.isfinite(result["loss"])
        result["loss"].backward()
        assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in pred.values())
    else:
        # CQFL requires Gaussian [B,C,H,W] targets, whereas binary produces
        # integer [B,H,W] labels. Retain legacy behavior; grouped fails early.
        with pytest.raises(IndexError, match="shape of the mask"):
            legacy(pred, tgt)
    with pytest.raises(ValueError, match="binary grouped support"):
        build("binary", {"name": strategy}, groups())


@pytest.mark.parametrize("strategy", ("baseline", "oga"))
@pytest.mark.parametrize("method", ("mgiou", "yaw_footprint", "rotated_iou"))
def test_binary_overlap_quality_tracks_canonical_box_not_last_painted_class(tmp_path, strategy, method):
    dataset = make_dataset(tmp_path, classification="binary")
    source = torch.from_numpy(dataset.get_boxes(0))[1:]
    source[1, 4:6] = source[0, 4:6]
    targets = [dataset.get_label(ordered, GEOMETRY) for ordered in (source, source.flip(0))]
    first, second = [target["groups"]["ped_cyc"] for target in targets]
    assert first["cls"][8, 2] == 2 and second["cls"][8, 2] == 1
    torch.testing.assert_close(first["size"][:, 8, 2], source[0, 2:4].log())
    for key in ("offset", "size", "yaw", "reg_mask"):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    prediction = {"groups": {}}
    for group in groups():
        tgt = targets[0]["groups"][group.name]
        pred = {key: tgt[key].unsqueeze(0).clone().requires_grad_() for key in ("offset", "size", "yaw")}
        pred["cls"] = torch.full((1, group.num_classes + 1, 16, 12), -1., requires_grad=True)
        if group.name == "ped_cyc":
            with torch.no_grad():
                pred["cls"][:, 2].fill_(3.)
        pred["iou"] = torch.full((1, 1, 16, 12), 1., requires_grad=True)
        prediction["groups"][group.name] = pred
    batched = [{"groups": {name: {key: value.unsqueeze(0) for key, value in tgt.items()}
                           for name, tgt in target["groups"].items()}} for target in targets]
    criterion = build("binary", {"name": strategy, "use_iou": True, "iou_target_type": method}, groups()).eval()
    outputs = [criterion(prediction, target) for target in batched]
    assert outputs[0]["group/ped_cyc/cls"] != outputs[1]["group/ped_cyc/cls"]
    torch.testing.assert_close(outputs[0]["group/ped_cyc/iou"], outputs[1]["group/ped_cyc/iou"], rtol=0, atol=0)
    quality = compute_iou_targets(prediction["groups"]["ped_cyc"], batched[0]["groups"]["ped_cyc"], method=method)
    assert quality[0, 8, 2] > .999
    cyclist = copy.copy(prediction["groups"]["ped_cyc"])
    cyclist["size"] = cyclist["size"].detach().clone()
    cyclist["yaw"] = cyclist["yaw"].detach().clone()
    cyclist["size"][0, :, 8, 2] = source[1, 2:4].log()
    cyclist["yaw"][0, :, 8, 2] = torch.stack((torch.cos(2 * source[1, 7]), torch.sin(2 * source[1, 7])))
    mismatch = compute_iou_targets(cyclist, batched[0]["groups"]["ped_cyc"], method=method)
    assert mismatch[0, 8, 2] < quality[0, 8, 2] - .05
