"""Exact-quality curriculum broadcast, group ownership and persistent state."""

import copy
import sys
from pathlib import Path

import pytest
import torch

from test_grouped_iqa import known_boxes
from test_grouped_losses import build, groups, tensors
from test_grouped_loss_state import snapshot, assert_state_equal
from core.losses.iou_targets import compute_iou_targets

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/kitti_training_pipeline"))
from train import set_loss_epoch


def criterion(warmup=4):
    return build("gaussian", {"name": "q_oga", "quality_target": "rotated_iou",
                              "quality_warmup_epochs": warmup}, groups())


def boxes(empty_vru=False):
    prediction, target = known_boxes(empty_vru=empty_vru)
    for pred in prediction["groups"].values():
        del pred["iou"]
    return prediction, target


@pytest.mark.parametrize("epoch,mix", [(0, 0.), (2, .5), (4, 1.), (7, 1.)])
@pytest.mark.parametrize("empty_vru", [False, True])
def test_epoch_broadcast_uses_each_group_peak_and_own_exact_quality(epoch, mix, empty_vru):
    loss = criterion().eval()
    set_loss_epoch(loss, epoch)
    prediction, target = boxes(empty_vru)
    result = loss(prediction, target)
    for group in groups():
        name = group.name
        assert loss.criteria[name].quality_epoch == epoch
        count = target["groups"][name]["cls"].ge(1).any(1).sum()
        raw = compute_iou_targets(prediction["groups"][name], target["groups"][name], method="rotated_iou")
        mask = target["groups"][name]["cls"].ge(1).any(1)
        expected = raw[mask].mean() if count else raw.new_zeros(())
        torch.testing.assert_close(result[f"group/{name}/quality_iou_mean"], expected)
        assert result[f"group/{name}/quality_iou_mix"] == mix
        assert result[f"group/{name}/quality_peak_count"] == count
        torch.testing.assert_close(result[f"group/{name}/quality_target_mean"],
                                   (1 - mix) + mix * expected if count else raw.new_zeros(()))


@pytest.mark.parametrize("epoch", [-1, True, 1.5, "2", None])
def test_invalid_broadcast_epoch_does_not_mutate_any_group(epoch):
    loss = criterion()
    before = snapshot(loss)
    with pytest.raises(ValueError, match="non-negative integer"):
        loss.set_epoch(epoch)
    assert_state_equal(loss, before)


@pytest.mark.parametrize("invalid", ["missing_epoch", "different_epoch", "later_peak_without_owner"])
def test_invalid_curriculum_or_later_peak_fails_before_any_ema_update(invalid):
    loss = criterion()
    prediction, target = boxes()
    if invalid != "missing_epoch":
        loss.set_epoch(2)
    if invalid == "different_epoch":
        loss.criteria["ped_cyc"].set_epoch(3)
    elif invalid == "later_peak_without_owner":
        target["groups"]["ped_cyc"]["reg_mask"][0, 2, 3] = 0
    before = snapshot(loss)
    with pytest.raises(ValueError, match="set_epoch|same epoch|assigned regression"):
        loss(prediction, target)
    assert_state_equal(loss, before)


def test_curriculum_state_roundtrip_preserves_next_loss_and_requires_each_epoch_buffer():
    loss = criterion()
    loss.set_epoch(2)
    prediction, target = boxes()
    loss(prediction, target)
    restored = criterion()
    restored.load_state_dict(copy.deepcopy(loss.state_dict()), strict=True)
    first, second = loss(prediction, target), restored(prediction, target)
    torch.testing.assert_close(first["loss"], second["loss"], rtol=0, atol=0)
    assert_state_equal(restored, snapshot(loss))
    state = copy.deepcopy(loss.state_dict())
    del state["criteria.ped_cyc.strategy.quality_epoch"]
    with pytest.raises(RuntimeError, match="quality_epoch"):
        criterion().load_state_dict(state, strict=True)


def test_target_only_exact_has_no_epoch_buffers_and_noop_broadcast_preserves_state():
    loss = criterion(0).eval()
    assert not any("quality_epoch" in key for key in loss.state_dict())
    before = snapshot(loss)
    loss.set_epoch(8)
    assert_state_equal(loss, before)
    prediction, target = boxes()
    assert loss(prediction, target)["group/car/quality_iou_mix"] == 1


def test_two_vru_class_peaks_share_one_canonical_box_quality_and_cell_count():
    loss = criterion().eval()
    loss.set_epoch(4)
    prediction, target = boxes()
    target["groups"]["ped_cyc"]["cls"][0, :, 2, 3] = 1
    result = loss(prediction, target)
    assert result["group/ped_cyc/quality_peak_count"] == 1
    assert result["group/ped_cyc/quality_iou_mean"] == 0
    assert result["group/car/quality_iou_mean"] > .999
    result["loss"].backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all()
               for pred in prediction["groups"].values() for value in pred.values())
