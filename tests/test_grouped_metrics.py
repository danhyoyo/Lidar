"""Quality means follow supervised counts, independent of objective weights."""

import sys
from pathlib import Path

import pytest
import torch

from test_grouped_qoga import boxes, criterion as qoga
from test_grouped_losses import build, groups, tensors
from test_grouped_header import header

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/kitti_training_pipeline"))
from train import QualityMetrics, validate


def extra_vru_cells(target):
    for y, x in ((0, 0), (1, 1)):
        target["groups"]["ped_cyc"]["reg_mask"][0, y, x] = 1
        target["groups"]["ped_cyc"]["cls"][0, 0, y, x] = 1


@pytest.mark.parametrize("empty", [False, True])
def test_exact_quality_means_use_peak_counts_not_group_objective_weights(empty):
    config = {"name": "q_oga", "quality_target": "rotated_iou", "quality_warmup_epochs": 4,
              "group_weights": {"car": 1., "ped_cyc": 3.}}
    loss = build("gaussian", config, groups()).eval()
    loss.set_epoch(2)
    prediction, target = boxes(empty)
    if not empty:
        extra_vru_cells(target)
    result = loss(prediction, target)
    count = 1 if empty else 4
    assert result["quality_peak_count"] == count
    torch.testing.assert_close(result["quality_iou_mean"], torch.tensor(1. / count))
    torch.testing.assert_close(result["quality_target_mean"], torch.tensor(.5 + .5 / count))
    assert result["quality_iou_mix"] == .5
    assert all(not value.requires_grad for key, value in result.items() if key != "loss")


@pytest.mark.parametrize("strategy", ["baseline", "oga"])
@pytest.mark.parametrize("empty", [False, True])
def test_iqa_means_use_regression_counts_including_nonpeak_cells(strategy, empty):
    prediction, target = boxes(empty)
    for pred in prediction["groups"].values():
        pred["iou"] = torch.zeros(1, 1, 4, 6, requires_grad=True)
    if not empty:
        target["groups"]["ped_cyc"]["reg_mask"][0, 0, 0] = 1
        target["groups"]["ped_cyc"]["reg_mask"][0, 1, 1] = 1
    loss = build("gaussian", {"name": strategy, "use_iou": True, "iou_target_type": "rotated_iou",
                              "group_weights": {"car": 1., "ped_cyc": 9.}}, groups()).eval()
    result = loss(prediction, target)
    count = 1 if empty else 4
    assert result["iou_target_count"] == count
    assert result["group/ped_cyc/iou_target_count"] == (0 if empty else 3)
    torch.testing.assert_close(result["mean_iou_target"], torch.tensor(1. / count))


def test_all_empty_quality_means_are_finite_zeros_and_mix_is_preserved():
    prediction, target = boxes(True)
    target["groups"]["car"]["cls"].zero_()
    target["groups"]["car"]["reg_mask"].zero_()
    loss = qoga().eval()
    loss.set_epoch(2)
    result = loss(prediction, target)
    assert result["quality_peak_count"] == 0
    for key in QualityMetrics.MEANS:
        assert result[key] == 0 and torch.isfinite(result[key])
    assert result["quality_iou_mix"] == .5


def test_different_group_curriculum_mixes_are_rejected():
    loss = qoga().eval()
    loss.set_epoch(2)
    loss.criteria["ped_cyc"].strategy.quality_warmup_epochs = 8
    prediction, target = boxes()
    with pytest.raises(ValueError, match="mix"):
        loss(prediction, target)


def test_epoch_quality_accumulator_preserves_aggregate_and_group_counts_and_means():
    stats = QualityMetrics()
    loss = qoga().eval()
    loss.set_epoch(2)
    for empty in (False, True):
        prediction, target = boxes(empty)
        if not empty:
            extra_vru_cells(target)
        stats.update(loss(prediction, target))
    result = stats.summarize()
    assert result["quality_peak_count"] == 5
    torch.testing.assert_close(torch.tensor(result["quality_iou_mean"]), torch.tensor(.4))
    assert result["group/car/quality_peak_count"] == 2
    assert result["group/ped_cyc/quality_peak_count"] == 3
    assert result["group/car/quality_iou_mean"] > .999
    assert result["group/ped_cyc/quality_iou_mean"] == 0
    assert result["group/ped_cyc/quality_iou_mix"] == .5


@pytest.mark.parametrize("iqa", [False, True])
def test_validate_excludes_namespaced_quality_from_sample_averaging(iqa):
    torch.manual_seed(105)
    model = header(iqa=iqa).eval()
    config = {"name": "oga", "use_iou": True, "iou_target_type": "yaw_footprint"} if iqa else {
        "name": "q_oga", "quality_target": "rotated_iou"}
    loss = build("gaussian", config, groups()).eval()
    batches = []
    outputs = []
    for batch_size in (1, 2):
        _, target = tensors(iqa=iqa, empty_vru=True)
        target = {"groups": {name: {key: tensor.repeat(batch_size, *([1] * (tensor.ndim - 1)))
                                   for key, tensor in item.items()} for name, item in target["groups"].items()}}
        target["voxel"] = torch.randn(batch_size, 32, 4, 6)
        if batch_size == 2:
            for y, x in ((0, 0), (1, 1)):
                target["groups"]["car"]["reg_mask"][:, y, x] = 1
                target["groups"]["car"]["cls"][:, 0, y, x] = 1
            target["groups"]["car"]["offset"].add_(20.)
        batches.append(target)
        with torch.no_grad():
            outputs.append(loss(model(target["voxel"]), target))
    result = validate(model, loss, batches, torch.device("cpu"), "fp32")
    count_key = "iou_target_count" if iqa else "quality_peak_count"
    mean_key = "mean_iou_target" if iqa else "quality_iou_mean"
    for prefix in ("", "group/car/", "group/ped_cyc/"):
        counts = [out[prefix + count_key] for out in outputs]
        count = sum(counts)
        expected = sum(out[prefix + mean_key] * c for out, c in zip(outputs, counts)) / count.clamp_min(1)
        assert result[prefix + count_key] == count
        assert result[prefix + mean_key] == pytest.approx(float(expected), abs=1e-7)
    assert result["samples"] == 3
    assert result["loss"] == pytest.approx(float((outputs[0]["loss"] + 2 * outputs[1]["loss"]) / 3))


@pytest.mark.parametrize("name", ["quality_peak_count", "group/car/quality_target_mean",
                                 "group/ped_cyc/quality_iou_mix", "mean_iou_target", "group/car/iou_target_count"])
def test_quality_metric_classifier_recognizes_nested_names(name):
    assert QualityMetrics.is_quality_metric(name)
    assert not QualityMetrics.is_quality_metric("group/car/iou")
    assert not QualityMetrics.is_quality_metric("group/car/loss")


def test_legacy_oga_iqa_mean_is_retained_with_supervised_cell_count():
    from core.losses.loss_fn import LossFunction
    loss = LossFunction("gaussian", {"name": "oga", "use_iou": True,
                                      "iou_target_type": "rotated_iou"}).eval()
    stats = QualityMetrics()
    expected_sum = torch.tensor(0.)
    expected_count = 0
    for extra in (False, True):
        prediction, target = tensors(iqa=True)
        pred, tgt = prediction["groups"]["car"], target["groups"]["car"]
        if extra:
            tgt["reg_mask"][0, 0, 0] = 1
            tgt["offset"].add_(10.)
        output = loss(pred, tgt)
        count = tgt["reg_mask"].bool().sum()
        expected_count += int(count)
        expected_sum += output["mean_iou_target"] * count
        stats.update(output, tgt)
    result = stats.summarize()
    assert result["iou_target_count"] == expected_count
    assert result["mean_iou_target"] == pytest.approx(float(expected_sum / expected_count))
