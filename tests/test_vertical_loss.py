"""Explicit fixed vertical supervision after each preserved BEV objective."""

import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "detector"), str(ROOT / "detector/core/datasets"),
               str(ROOT / "tools/kitti_training_pipeline")]

from core.losses.loss_fn import LossFunction, build_loss_function
from common import checkpoint_identity, detection_spec
from test_grouped_losses import tensors, groups
from test_vertical_head import config_3d


@pytest.mark.parametrize("strategy", ["baseline", "oga"])
@pytest.mark.parametrize("classification", ["gaussian", "binary"])
def test_fixed_vertical_term_is_added_after_bev_before_group_aggregation(strategy, classification):
    pred, target = tensors(classification, iqa=True)
    for name in pred["groups"]:
        pred["groups"][name]["vertical"] = torch.full_like(pred["groups"][name]["offset"], .5, requires_grad=True)
        target["groups"][name]["vertical"] = torch.zeros_like(target["groups"][name]["offset"])
    config = {"name": strategy, "use_iou": True, "vertical_loss_weight": 2., "group_weights": {"car": 1., "ped_cyc": 3.}}
    vertical = build_loss_function(classification, config, task_groups=groups(), box_mode="3d")
    bev = build_loss_function(classification, config, task_groups=groups())
    bev.load_state_dict(vertical.state_dict(), strict=True)
    clean_pred = {"groups": {n: {k: v for k, v in h.items() if k != "vertical"} for n,h in pred["groups"].items()}}
    clean_target = {"groups": {n: {k: v for k, v in h.items() if k != "vertical"} for n,h in target["groups"].items()}}
    base, actual = bev(clean_pred, clean_target), vertical(pred, target)
    expected = torch.tensor(.125)  # Average over owned cells and two channels.
    torch.testing.assert_close(actual["vertical"], expected)
    torch.testing.assert_close(actual["loss"], base["loss"] + 2 * expected)
    for key in ("cls", "offset", "size", "yaw", "iou"):
        torch.testing.assert_close(actual[key], base[key], rtol=0, atol=0)
    actual["loss"].backward()
    for heads in pred["groups"].values():
        assert heads["vertical"].grad is not None and torch.isfinite(heads["vertical"].grad).all()


@pytest.mark.parametrize("empty", [False, True])
def test_exact_or_empty_vertical_loss_is_connected_zero_with_no_bev_gradients(empty):
    pred, target = tensors(empty_vru=empty)
    p,t = pred["groups"]["ped_cyc"], target["groups"]["ped_cyc"]
    p["vertical"] = torch.zeros_like(p["offset"], requires_grad=True)
    t["vertical"] = torch.zeros_like(t["offset"])
    objective = LossFunction("gaussian", {"name": "baseline", "vertical_loss_weight": 1.}, box_mode="3d")
    result = objective(p,t)
    assert result["vertical"].dtype == torch.float32 and result["vertical"].item() == 0
    result["vertical"].backward()
    assert p["vertical"].grad is not None and not p["vertical"].grad.any()
    assert all(p[k].grad is None for k in ("cls", "offset", "size", "yaw"))


def test_unsupervised_vertical_nan_is_masked_before_smooth_l1_and_amp_reduces_fp32():
    pred,target = tensors()
    p,t = pred["groups"]["car"],target["groups"]["car"]
    p["vertical"] = torch.full_like(p["offset"], float("nan"), dtype=torch.bfloat16)
    p["vertical"][:,:,2,3] = 2.
    p["vertical"].requires_grad_()
    t["vertical"] = torch.full_like(t["offset"], float("nan"))
    t["vertical"][:,:,2,3] = 0.
    result = LossFunction("gaussian", {"name":"baseline","vertical_loss_weight":1.}, box_mode="3d")(p,t)
    assert result["vertical"].dtype == torch.float32
    torch.testing.assert_close(result["vertical"], torch.tensor(1.5))
    result["vertical"].backward()
    assert torch.isfinite(p["vertical"].grad).all()


@pytest.mark.parametrize("invalid", [None,0.,-1.,float("nan"),True])
def test_3d_requires_explicit_positive_finite_vertical_weight(invalid):
    config = config_3d()
    if invalid is None: config["loss"].pop("vertical_loss_weight",None)
    else: config["loss"]["vertical_loss_weight"] = invalid
    with pytest.raises(ValueError,match="vertical_loss_weight"):
        detection_spec(config)


def test_all_group_vertical_inputs_validate_before_adaptive_state_updates():
    pred,target=tensors()
    for n in pred["groups"]:
        pred["groups"][n]["vertical"]=torch.ones_like(pred["groups"][n]["offset"],requires_grad=True)
        target["groups"][n]["vertical"]=torch.zeros_like(target["groups"][n]["offset"])
    objective=build_loss_function("gaussian",{"name":"oga","vertical_loss_weight":1.},task_groups=groups(),box_mode="3d")
    before={k:v.clone() for k,v in objective.state_dict().items()}
    del target["groups"]["ped_cyc"]["vertical"]
    with pytest.raises((ValueError,KeyError),match="vertical"):
        objective(pred,target)
    assert all(torch.equal(v,objective.state_dict()[k]) for k,v in before.items())


def test_real_3d_optimizer_checkpoint_resume_includes_vertical_objective_identity(tmp_path):
    import train
    from common import build_model
    from test_grouped_checkpoint import components,payload,restore
    config=config_3d();config["loss"]["vertical_loss_weight"]=1.
    pred,target=tensors(iqa=True)
    target["voxel"]=torch.randn(1,14,16,32)
    for h in target["groups"].values():
        for k,v in tuple(h.items()):h[k]=F.pad(v,(0,2))
        h["vertical"]=torch.zeros_like(h["offset"])
    parts=components(config)
    result=parts[1](parts[0](target["voxel"]),target);result["loss"].backward()
    parts[2].step();parts[3].step()
    saved=payload(config,parts);path=tmp_path/"vertical.pt";torch.save(saved,path)
    saved=torch.load(path,weights_only=True);destination=components(config)
    assert restore(saved,config,destination)["mode"]=="resume"
    for old,new in zip(parts[:2],destination[:2]):
        assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    changed=copy.deepcopy(config);changed["loss"]["vertical_loss_weight"]=2.
    assert checkpoint_identity(changed)!=checkpoint_identity(config)
    with pytest.raises(ValueError,match="identity"):
        restore(saved,changed,components(changed))
