"""Frozen flat decode and group-local Gaussian box/quality associations."""

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "detector"))
import postprocess
from core.task_groups import resolve_task_groups


GEOMETRY = {"x_min": 2., "x_max": 12., "x_res": .5,
            "y_min": -3., "y_max": 9., "y_res": 1.}
DEFINITIONS = [{"name": "car", "classes": ["Car"]},
               {"name": "ped_cyc", "classes": ["Cyclist", "Pedestrian"]}]
OBJECTS = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}


def groups():
    return resolve_task_groups(DEFINITIONS, OBJECTS)


def flat_predictions(iqa=False):
    pred = {"cls": torch.linspace(-2.3, 2.7, 45).reshape(1, 3, 3, 5),
            "offset": torch.linspace(-.3, .7, 30).reshape(1, 2, 3, 5),
            "size": torch.linspace(-.2, 1.5, 30).reshape(1, 2, 3, 5),
            "yaw": torch.cat([torch.ones(1, 1, 3, 5), torch.linspace(-1., 1., 15).reshape(1, 1, 3, 5)], 1)}
    pred["cls"][0, 0, 0, 0] = 3.
    pred["cls"][0, 1, 0, :2] = 2.
    if iqa:
        pred["iou"] = torch.linspace(-2., 2., 15).reshape(1, 1, 3, 5)
    return pred


@pytest.mark.parametrize("peak_mode", ["per_class", "legacy"])
@pytest.mark.parametrize("iqa", [False, True])
@pytest.mark.parametrize("nms", [None, 0., .2, .6])
@pytest.mark.parametrize("threshold", [.2, .6])
def test_frozen_flat_predictions_preserve_rows_scores_order_and_nms(peak_mode, iqa, nms, threshold):
    frozen = json.loads((ROOT / "tests/fixtures/flat_decode_contract.json").read_text())
    key = json.dumps([peak_mode, iqa, nms, threshold])
    expected = np.asarray(frozen["outputs"][key], dtype=np.float32).reshape(-1, 7)
    actual = postprocess.filter_pred(flat_predictions(iqa), {"geometry": GEOMETRY, "peak_mode": peak_mode},
                                     4, threshold, nms)
    assert actual.dtype == np.float32 and actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("nms", [None, 0., .5])
def test_candidates_are_unsuppressed_until_shared_finalization(nms):
    prediction = flat_predictions(True)
    config = {"geometry": GEOMETRY}
    candidates = postprocess.decode_candidates(prediction, config, 4, .2)
    assert torch.is_tensor(candidates) and candidates.shape[1] == 7
    assert not candidates.requires_grad
    actual = postprocess.finalize_detections(candidates, nms)
    np.testing.assert_array_equal(actual, postprocess.filter_pred(prediction, config, 4, .2, nms))


def grouped_predictions(iqa=True, empty=False):
    result = {}
    for group in groups():
        result[group.name] = {
            "cls": torch.full((1, group.num_classes, 3, 5), -20.),
            "offset": torch.zeros(1, 2, 3, 5), "size": torch.zeros(1, 2, 3, 5),
            "yaw": torch.cat([torch.ones(1, 1, 3, 5), torch.zeros(1, 1, 3, 5)], 1)}
        if iqa:
            result[group.name]["iou"] = torch.zeros(1, 1, 3, 5)
    if not empty:
        result["car"]["cls"][0, 0, 1, 3] = torch.logit(torch.tensor(.9))
        result["ped_cyc"]["cls"][0, 0, 1, 3] = torch.logit(torch.tensor(.8))
        result["ped_cyc"]["cls"][0, 1, 1, 3] = torch.logit(torch.tensor(.7))
        result["car"]["offset"][0, :, 1, 3] = torch.tensor([.3, -.4])
        result["car"]["size"][0, :, 1, 3] = torch.tensor([math.log(1.8), math.log(4.2)])
        result["car"]["yaw"][0, :, 1, 3] = torch.tensor([math.cos(.6), math.sin(.6)])
        result["ped_cyc"]["offset"][0, :, 1, 3] = torch.tensor([-.2, .6])
        result["ped_cyc"]["size"][0, :, 1, 3] = torch.tensor([math.log(.6), math.log(.9)])
        result["ped_cyc"]["yaw"][0, :, 1, 3] = torch.tensor([math.cos(-.8), math.sin(-.8)])
    return {"groups": result}


def decode(prediction, **kwargs):
    return postprocess.filter_pred(prediction, {"geometry": GEOMETRY, "objects": OBJECTS, **kwargs},
                                   4, .1, task_groups=groups())


@pytest.mark.parametrize("iqa", [False, True])
@pytest.mark.parametrize("peak_mode", ["per_class", "legacy"])
def test_grouped_gaussian_asymmetric_metric_decode_uses_group_box_and_reversed_global_mapping(iqa, peak_mode):
    prediction = grouped_predictions(iqa)
    saved = copy.deepcopy(prediction)
    actual = decode(prediction, peak_mode=peak_mode, nms_alpha=0.)
    expected = np.array([[0., .9, 8.3, .6, 4.2, 1.8, .3],
                         [2., .8, 7.8, 1.6, .9, .6, -.4],
                         [1., .7, 7.8, 1.6, .9, .6, -.4]], np.float32)
    if peak_mode == "legacy":
        expected = expected[:2]
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
    for name in saved["groups"]:
        for key in saved["groups"][name]:
            torch.testing.assert_close(prediction["groups"][name][key], saved["groups"][name][key], rtol=0, atol=0)


@pytest.mark.parametrize("alpha", [0., .5, 1.])
def test_car_only_iqa_change_affects_only_car_score(alpha):
    prediction = grouped_predictions()
    before = decode(prediction, nms_alpha=alpha)
    prediction["groups"]["car"]["iou"].fill_(torch.logit(torch.tensor(.9)))
    after = decode(prediction, nms_alpha=alpha)
    for class_id in (1, 2):
        np.testing.assert_array_equal(before[before[:, 0] == class_id], after[after[:, 0] == class_id])
    car = after[after[:, 0] == 0][0]
    assert car[1] == pytest.approx(.9 ** (1 - alpha) * .9 ** alpha)


def test_quality_ranking_and_raw_class_threshold_are_both_required():
    prediction = grouped_predictions()
    prediction["groups"]["car"]["cls"].fill_(torch.logit(torch.tensor(.09)))
    prediction["groups"]["car"]["iou"].fill_(20.)
    assert 0 not in decode(prediction, nms_alpha=1.)[:, 0]
    prediction = grouped_predictions()
    prediction["groups"]["car"]["iou"].fill_(-20.)
    assert 0 not in decode(prediction, nms_alpha=.5)[:, 0]


@pytest.mark.parametrize("iqa", [False, True])
def test_all_empty_grouped_scene_returns_float32_zero_by_seven(iqa):
    actual = decode(grouped_predictions(iqa, empty=True))
    assert actual.shape == (0, 7) and actual.dtype == np.float32


@pytest.mark.parametrize("malformed", ["missing", "extra", "mixed", "width", "batch", "spatial", "iqa_missing", "iou_width", "nan", "box_width"])
def test_malformed_grouped_predictions_are_rejected(malformed):
    prediction = grouped_predictions()
    group = prediction["groups"]["ped_cyc"]
    if malformed == "missing":
        del prediction["groups"]["ped_cyc"]
    elif malformed == "extra":
        prediction["groups"]["unknown"] = copy.deepcopy(group)
    elif malformed == "mixed":
        prediction["cls"] = torch.zeros(1, 3, 3, 5)
    elif malformed == "width":
        group["cls"] = torch.zeros(1, 3, 3, 5)
    elif malformed == "batch":
        group["cls"] = torch.zeros(2, 2, 3, 5)
    elif malformed == "spatial":
        group["offset"] = torch.zeros(1, 2, 5, 3)
    elif malformed == "iqa_missing":
        del group["iou"]
    elif malformed == "iou_width":
        group["iou"] = torch.zeros(1, 2, 3, 5)
    elif malformed == "nan":
        group["yaw"][0, 0, 0, 0] = float("nan")
    else:
        group["offset"] = group["size"] = torch.zeros(1, 3, 3, 5)
    with pytest.raises((ValueError, KeyError, FloatingPointError)):
        decode(prediction)


def test_grouped_predictions_require_resolved_metadata_and_reject_unknown_classification():
    with pytest.raises(ValueError, match="task_groups"):
        postprocess.filter_pred(grouped_predictions(), {"geometry": GEOMETRY}, 4, .1)
    with pytest.raises(ValueError, match="classification|cls_encoding"):
        postprocess.filter_pred(grouped_predictions(), {"geometry": GEOMETRY}, 4, .1,
                                task_groups=groups(), cls_encoding="typo")


def test_same_cell_cross_group_objects_survive_classwise_nms_with_their_own_boxes():
    result = postprocess.filter_pred(grouped_predictions(), {"geometry": GEOMETRY, "nms_alpha": 0.},
                                    4, .1, .2, task_groups=groups(), use_iou=True)
    assert result[:, 0].tolist() == [0., 2., 1.]
    np.testing.assert_allclose(result[0, 2:6], [8.3, .6, 4.2, 1.8], atol=1e-6)
    np.testing.assert_allclose(result[1, 2:6], [7.8, 1.6, .9, .6], atol=1e-6)


def test_grouped_absent_quality_uses_raw_classification_despite_configured_alpha():
    result = decode(grouped_predictions(iqa=False), nms_alpha=.9)
    np.testing.assert_allclose(result[:, 1], [.9, .8, .7], atol=1e-6)


@pytest.mark.parametrize("bad", ["offset_none", "missing_yaw", "extra_head", "group_none", "wrong_global_ids", "quality_contract", "flat_metadata"])
def test_additional_malformed_metadata_and_heads_fail_clearly(bad):
    prediction = grouped_predictions()
    config = {"geometry": GEOMETRY, "objects": dict(OBJECTS)}
    kwargs = {"task_groups": groups()}
    if bad == "offset_none":
        prediction["groups"]["car"]["offset"] = None
    elif bad == "missing_yaw":
        del prediction["groups"]["car"]["yaw"]
    elif bad == "extra_head":
        prediction["groups"]["car"]["vertical"] = torch.zeros(1, 2, 3, 5)
    elif bad == "group_none":
        prediction["groups"]["car"] = None
    elif bad == "wrong_global_ids":
        config["objects"].update(Pedestrian=2, Cyclist=1)
    elif bad == "quality_contract":
        kwargs["use_iou"] = False
    else:
        prediction = flat_predictions()
    with pytest.raises((ValueError, KeyError)):
        postprocess.filter_pred(prediction, config, 4, .1, **kwargs)


@pytest.mark.parametrize("mode", ["per_class", "legacy"])
@pytest.mark.parametrize("nms", [None, .2])
def test_legacy_flat_binary_sigmoid_policy_remains_unchanged(mode, nms):
    prediction = flat_predictions(iqa=True)
    config = {"geometry": GEOMETRY, "peak_mode": mode}
    expected = postprocess.filter_pred(prediction, config, 4, .2, nms)
    actual = postprocess.filter_pred(prediction, config, 4, .2, nms, cls_encoding="binary")
    np.testing.assert_array_equal(actual, expected)


def binary_predictions(iqa=False, empty_vru=False):
    prediction = grouped_predictions(iqa)
    for group in groups():
        heads = prediction["groups"][group.name]
        background = [.6] + [.4 / group.num_classes] * group.num_classes
        heads["cls"] = torch.tensor(background).log().reshape(1, -1, 1, 1).expand(1, -1, 3, 5).clone()
    prediction["groups"]["car"]["cls"][0, :, 1, 3] = torch.tensor([.1, .9]).log()
    if not empty_vru:
        prediction["groups"]["ped_cyc"]["cls"][0, :, 1, 3] = torch.tensor([.08, .87, .05]).log()
        prediction["groups"]["ped_cyc"]["cls"][0, :, 0, 0] = torch.tensor([.17, .03, .8]).log()
    return prediction


@pytest.mark.parametrize("iqa", [False, True])
@pytest.mark.parametrize("peak_mode", ["per_class", "legacy"])
@pytest.mark.parametrize("empty_vru", [False, True])
def test_binary_softmax_discards_background_and_maps_foreground_with_own_quality(iqa, peak_mode, empty_vru):
    prediction = binary_predictions(iqa, empty_vru)
    result = postprocess.filter_pred(prediction, {"geometry": GEOMETRY, "peak_mode": peak_mode, "nms_alpha": .5},
                                    4, .1, task_groups=groups(), cls_encoding="binary", use_iou=iqa)
    assert result[:, 0].tolist() == ([0.] if empty_vru else [0., 2., 1.])
    expected = np.array([.9] if empty_vru else [.9, .87, .8])
    if iqa:
        expected = np.sqrt(expected * .5)
    np.testing.assert_allclose(result[:, 1], expected, atol=1e-6)
    np.testing.assert_allclose(result[0, 2:], [8.3, .6, 4.2, 1.8, .3], atol=1e-6)
    # Local softmax must be invariant to an additive shift of all logits.
    for heads in prediction["groups"].values():
        heads["cls"].add_(13.)
    shifted = postprocess.filter_pred(prediction, {"geometry": GEOMETRY, "peak_mode": peak_mode, "nms_alpha": .5},
                                     4, .1, task_groups=groups(), cls_encoding="binary", use_iou=iqa)
    np.testing.assert_allclose(shifted, result, atol=2e-6)


@pytest.mark.parametrize("alpha", [0., .5, 1.])
def test_binary_background_winner_is_masked_before_quality_peak_pooling(alpha):
    prediction = binary_predictions(iqa=True, empty_vru=True)
    car = prediction["groups"]["car"]
    car["cls"][0, :, 1, 2] = torch.tensor([.51, .49]).log()
    car["iou"][0, 0, 1, 2] = 20.
    result = postprocess.filter_pred(prediction, {"geometry": GEOMETRY, "nms_alpha": alpha},
                                    4, .1, task_groups=groups(), cls_encoding="binary")
    assert result.shape == (1, 7) and result[0, 0] == 0
    assert result[0, 2] == pytest.approx(8.3)


@pytest.mark.parametrize("limit", [1, 2, 3, 10, None])
@pytest.mark.parametrize("nms", [None, .2])
def test_grouped_cap_is_applied_once_after_global_sort_and_independent_of_group_order(limit, nms):
    prediction = grouped_predictions(iqa=False)
    expected = postprocess.filter_pred(prediction, {"geometry": GEOMETRY}, 4, .1, nms, task_groups=groups())
    capped = postprocess.filter_pred(prediction, {"geometry": GEOMETRY}, 4, .1, nms,
                                    task_groups=groups(), max_detections=limit)
    np.testing.assert_array_equal(capped, expected[:limit])
    reverse = postprocess.filter_pred(prediction, {"geometry": GEOMETRY}, 4, .1, nms,
                                     task_groups=tuple(reversed(groups())), max_detections=limit)
    np.testing.assert_array_equal(reverse, capped)


def test_same_class_duplicate_suppression_happens_before_the_global_cap():
    prediction = grouped_predictions(iqa=False)
    ped = prediction["groups"]["ped_cyc"]
    ped["cls"].fill_(-20.)
    for x, score in [(0, .85), (2, .95)]:
        ped["cls"][0, 0, 0, x] = torch.logit(torch.tensor(score))
        ped["offset"][0, :, 0, x] = torch.tensor([6. - 2. * x, 4.])
    ped["cls"][0, 1, 2, 4] = torch.logit(torch.tensor(.8))
    result = postprocess.filter_pred(prediction, {"geometry": GEOMETRY}, 4, .1, .2,
                                    task_groups=groups(), max_detections=3)
    assert result[:, 0].tolist() == [2., 0., 1.]
    np.testing.assert_allclose(result[:, 1], [.95, .9, .8], atol=1e-6)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, "2"])
def test_invalid_global_cap_fails_even_for_an_empty_scene(limit):
    with pytest.raises(ValueError, match="max_detections"):
        postprocess.filter_pred(grouped_predictions(empty=True), {"geometry": GEOMETRY},
                                4, .1, task_groups=groups(), max_detections=limit)


def test_grouped_integral_grid_uses_rounding_tolerance_instead_of_truncating_float_edges():
    prediction = grouped_predictions(iqa=False, empty=True)
    for name, heads in prediction["groups"].items():
        for key, value in heads.items():
            heads[key] = torch.full((1, value.shape[1], 3, 12), -20. if key == "cls" else 0.)
        heads["yaw"][:, 0].fill_(1.)
    prediction["groups"]["car"]["cls"][0, 0, 1, 11] = 3.
    geometry = dict(x_min=0., x_max=9.6, x_res=.2, y_min=0., y_max=1.2, y_res=.1)
    result = postprocess.filter_pred(prediction, {"geometry": geometry}, 4, .5, task_groups=groups())
    np.testing.assert_allclose(result[0, 2:4], [8.8, .4], atol=1e-6)
    geometry["x_max"] = 9.65
    with pytest.raises(ValueError, match="integral|grid"):
        postprocess.filter_pred(prediction, {"geometry": geometry}, 4, .5, task_groups=groups())


@pytest.mark.parametrize("iqa", [False, True])
def test_binary_all_background_ties_do_not_emit_detection_even_with_high_quality(iqa):
    prediction = binary_predictions(iqa=iqa)
    for heads in prediction["groups"].values():
        heads["cls"].zero_()
        if iqa:
            heads["iou"].fill_(20.)
    result = postprocess.filter_pred(prediction, {"geometry": GEOMETRY, "nms_alpha": 1.},
                                    4, .1, task_groups=groups(), cls_encoding="binary")
    assert result.shape == (0, 7) and result.dtype == np.float32
