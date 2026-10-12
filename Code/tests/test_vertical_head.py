"""Explicit vertical prediction, unchanged BEV topology and consumer guards."""

import copy
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "detector"), str(ROOT / "tools/kitti_training_pipeline")]

from common import build_model, checkpoint_identity, detection_spec, generate_run_name
from detector.core.models.heads.cnn import Header
from detector.core.models.heads.grouped import GroupedHeader
from detector.core.models.model import CustomModel
from detector.core.task_groups import resolve_task_groups


def config_3d(*, grouped=True, strategy="oga", iqa=True, classification="gaussian"):
    config = json.loads((ROOT / "configs/experiments/under1m/hist14_local3_focal_grouped_oga_iqa.json").read_text())
    config["data"]["box_mode"] = "3d"
    config["model"].update(header_use_iou=iqa, cls_encoding=classification)
    config["loss"] = {"name": strategy, "use_iou": iqa, "vertical_loss_weight": 1.}
    if not grouped:
        config["model"]["head_mode"] = "legacy_single"
        del config["data"]["head_groups"]
    return config


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("strategy", ["baseline", "oga"])
@pytest.mark.parametrize("classification", ["gaussian", "binary"])
def test_pipeline_constructs_vertical_heads_without_mutating_canonical_data_mode(grouped, strategy, classification):
    config = config_3d(grouped=grouped, strategy=strategy, classification=classification)
    before = copy.deepcopy(config)
    model = build_model(config).eval()
    assert detection_spec(config).box_mode == model.box_mode == "3d"
    assert config == before and "box_mode" not in config["model"]
    with torch.no_grad():
        output = model(torch.randn(1, 14, 32, 48))
    heads = output["groups"].values() if grouped else [output]
    for head in heads:
        assert set(head) == {"cls", "offset", "size", "yaw", "iou", "vertical"}
        assert head["vertical"].shape == (1, 2, 8, 12)
        assert torch.isfinite(head["vertical"]).all()


def test_vertical_branch_gradient_isolated_from_other_group_and_bev_tasks():
    config = config_3d()
    groups = resolve_task_groups(config["data"]["head_groups"], config["data"]["kitti"]["objects"])
    head = GroupedHeader(groups, 32, use_bn=True, act="silu", use_iou=True, box_mode="3d")
    features = torch.randn(2, 32, 4, 6, requires_grad=True)
    output = head(features)["groups"]
    output["car"]["vertical"].square().mean().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.heads["car"].vertical.parameters())
    assert all(p.grad is None for key in ("cls", "offset", "size", "yaw", "iou")
               for p in getattr(head.heads["car"], key).parameters())
    assert all(p.grad is None for p in head.heads["ped_cyc"].parameters())


def test_bev_defaults_preserve_seeded_weights_and_3d_adds_exact_head_budget():
    torch.manual_seed(23)
    default = Header(3, 32, use_bn=True, act="silu", use_iou=True).eval()
    torch.manual_seed(23)
    explicit = Header(3, 32, use_bn=True, act="silu", use_iou=True, box_mode="bev").eval()
    torch.manual_seed(23)
    vertical = Header(3, 32, use_bn=True, act="silu", use_iou=True, box_mode="3d").eval()
    assert default.state_dict().keys() == explicit.state_dict().keys()
    for key, value in default.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(value, vertical.state_dict()[key], rtol=0, atol=0)
    features = torch.randn(1, 32, 4, 6)
    for key, value in default(features).items():
        torch.testing.assert_close(value, explicit(features)[key], rtol=0, atol=0)
        torch.testing.assert_close(value, vertical(features)[key], rtol=0, atol=0)
    assert sum(p.numel() for p in vertical.vertical.parameters()) == 18626
    model = build_model(config_3d()).eval()
    assert sum(p.numel() for p in model.parameters()) == 883941
    assert sum(p.numel() for p in model.backbone.parameters()) == 660528


def test_vertical_model_strict_state_roundtrip_and_semantic_identity():
    config = config_3d()
    original, restored = build_model(config).eval(), build_model(config).eval()
    restored.load_state_dict(original.state_dict(), strict=True)
    features = torch.randn(1, 14, 32, 48)
    with torch.no_grad():
        expected, actual = original(features)["groups"], restored(features)["groups"]
    for group in expected:
        for key in expected[group]:
            torch.testing.assert_close(actual[group][key], expected[group][key], rtol=0, atol=0)
    bev = copy.deepcopy(config)
    bev["data"]["box_mode"] = "bev"
    assert checkpoint_identity(config) != checkpoint_identity(bev)
    assert generate_run_name(config) != generate_run_name(bev)
    with pytest.raises(RuntimeError):
        build_model(bev).load_state_dict(original.state_dict(), strict=True)


@pytest.mark.parametrize("strategy", ["uwag", "q_oga"])
def test_3d_rejects_unimplemented_strategy_extensions(strategy):
    with pytest.raises(ValueError, match="3D.*baseline.*oga"):
        detection_spec(config_3d(strategy=strategy, iqa=False))


@pytest.mark.parametrize("mode", [None, "volumetric", 3])
def test_direct_head_and_model_reject_invalid_box_mode(mode):
    with pytest.raises(ValueError, match="box_mode"):
        Header(3, 32, box_mode=mode)
    with pytest.raises(ValueError, match="box_mode"):
        CustomModel({"backbone": "mobilepixornext"}, num_classes=3, box_mode=mode)


def test_3d_training_factory_requires_explicit_vertical_weight_before_optimizer_creation():
    import train
    config = config_3d()
    del config["loss"]["vertical_loss_weight"]
    with pytest.raises(ValueError, match="vertical_loss_weight"):
        train.build_training_criterion(config, torch.device("cpu"))


def test_bev_loss_and_decoder_reject_vertical_instead_of_silently_ignoring_it():
    from core.losses.loss_fn import LossFunction
    from detector.postprocess import decode_candidates
    from test_grouped_losses import tensors
    pred, target = tensors()
    pred, target = pred["groups"]["car"], target["groups"]["car"]
    pred["vertical"] = torch.zeros_like(pred["offset"])
    target["vertical"] = torch.zeros_like(target["offset"])
    with pytest.raises(ValueError, match="vertical loss"):
        LossFunction("gaussian", {"name": "baseline"})(pred, target)
    with pytest.raises(ValueError, match="3D.*decod"):
        decode_candidates(pred, {}, 4, .05)


def test_bev_evaluator_rejects_3d_before_loading_checkpoint(tmp_path):
    from evaluate_kitti_bev import PyTorchRunner
    with pytest.raises(ValueError, match="3D.*evaluat"):
        PyTorchRunner(tmp_path / "missing.pt", config_3d(), "cpu")


def test_legacy_export_rejects_vertical_even_without_grouped_heads_or_iqa():
    from export_onnx import RawHeadWrapper
    model = build_model(config_3d(grouped=False, iqa=False)).eval()
    with pytest.raises(ValueError, match="3D.*export"):
        RawHeadWrapper(model)
