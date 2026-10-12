"""Independent regression/IQA branches and unchanged legacy topology."""

import copy
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "detector"))

from detector.core.models.model import CustomModel
from detector.core.task_groups import resolve_task_groups


OBJECTS = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
DEFINITIONS = [{"name": "car", "classes": ["Car"]},
               {"name": "ped_cyc", "classes": ["Pedestrian", "Cyclist"]}]


def groups():
    return resolve_task_groups(DEFINITIONS, OBJECTS)


def header(classification="gaussian", iqa=True):
    cls = importlib.import_module("detector.core.models.heads.grouped").GroupedHeader
    return cls(groups(), in_channels=32, cls_encoding=classification,
               use_bn=True, act="silu", use_iou=iqa)


@pytest.mark.parametrize("classification,widths", [("gaussian", (1, 2)), ("binary", (2, 3))])
@pytest.mark.parametrize("iqa", [False, True])
def test_prediction_contract_widths_and_independent_regression_storage(classification, widths, iqa):
    model = header(classification, iqa).eval()
    assert isinstance(model.heads, torch.nn.ModuleDict)
    x = torch.randn(2, 32, 8, 12)
    outputs = model(x)
    assert tuple(outputs) == ("groups",)
    assert tuple(outputs["groups"]) == ("car", "ped_cyc")
    for name, width in zip(("car", "ped_cyc"), widths):
        pred = outputs["groups"][name]
        assert set(pred) == {"cls", "offset", "size", "yaw"} | ({"iou"} if iqa else set())
        assert pred["cls"].shape == (2, width, 8, 12)
        for key in ("offset", "size", "yaw"):
            assert pred[key].shape == (2, 2, 8, 12)
            assert pred[key].data_ptr() != outputs["groups"]["ped_cyc" if name == "car" else "car"][key].data_ptr()
        if iqa:
            assert pred["iou"].shape == (2, 1, 8, 12)
    car_ptrs = {p.data_ptr() for p in model.heads["car"].parameters()}
    assert car_ptrs.isdisjoint(p.data_ptr() for p in model.heads["ped_cyc"].parameters())


@pytest.mark.parametrize("iqa,expected", [(False, 148975), (True, 186161)])
def test_exact_grouped_head_parameter_budget(iqa, expected):
    assert sum(p.numel() for p in header(iqa=iqa).parameters()) == expected


def test_one_group_backward_does_not_touch_other_head_or_iqa():
    model = header()
    x = torch.randn(2, 32, 8, 12, requires_grad=True)
    outputs = model(x)
    sum(t.square().mean() for t in outputs["groups"]["car"].values()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.heads["car"].parameters())
    assert all(p.grad is None for p in model.heads["ped_cyc"].parameters())


def test_strict_state_roundtrip_including_group_bn_and_iqa():
    original = header().eval()
    restored = header().eval()
    restored.load_state_dict(original.state_dict(), strict=True)
    x = torch.randn(1, 32, 8, 12)
    expected, actual = original(x)["groups"], restored(x)["groups"]
    for name in expected:
        for key in expected[name]:
            torch.testing.assert_close(actual[name][key], expected[name][key], rtol=0, atol=0)
    assert any("heads.ped_cyc.iou.bn1.running_mean" in k for k in original.state_dict())


def pipeline_config():
    config = json.loads((ROOT / "configs/config.json").read_text())
    config["data"]["head_groups"] = copy.deepcopy(DEFINITIONS)
    config["data"]["bev_encoding"] = {"name": "hist14"}
    config["model"].update(head_mode="grouped", backbone_out_dim=32, stage_depths=[3, 4, 2],
                           c4_attention="none", c4_attention_scales=[], c4_context="focal",
                           header_use_iou=True)
    config["loss"] = {"name": "oga", "use_iou": True}
    return config


def test_pipeline_copies_config_and_constructs_shared_backbone_with_grouped_heads():
    from tools.kitti_training_pipeline.common import build_model, configure_detector_imports
    configure_detector_imports(ROOT / "detector")
    config = pipeline_config()
    before = copy.deepcopy(config)
    model = build_model(config).eval()
    assert config == before
    assert model.head_mode == "grouped" and not hasattr(model, "header")
    assert tuple(g.global_ids for g in model.task_groups) == ((0,), (1, 2))
    pred = model({"voxel": torch.zeros(1, 14, 32, 48)})
    assert pred["groups"]["car"]["cls"].shape == (1, 1, 8, 12)
    assert pred["groups"]["ped_cyc"]["cls"].shape == (1, 2, 8, 12)
    assert sum(p.numel() for p in model.backbone.parameters()) == 660528
    assert sum(p.numel() for p in model.grouped_header.parameters()) == 186161


def test_full_resolution_grouped_shape_and_total_budget_on_meta():
    from tools.kitti_training_pipeline.common import build_model, configure_detector_imports
    configure_detector_imports(ROOT / "detector")
    with torch.device("meta"):
        model = build_model(pipeline_config()).eval()
        output = model(torch.zeros(1, 14, 800, 704))
    assert output["groups"]["car"]["cls"].shape == (1, 1, 200, 176)
    assert output["groups"]["ped_cyc"]["iou"].shape == (1, 1, 200, 176)
    assert sum(p.numel() for p in model.parameters()) == 846689


def test_direct_grouped_model_requires_resolved_groups_and_validates_class_count():
    cfg = {"backbone": "mobilepixornext", "head_mode": "grouped", "c4_attention": "none"}
    with pytest.raises(ValueError, match="head_groups"):
        CustomModel(cfg, num_classes=3, input_channels=14)
    with pytest.raises(ValueError, match="num_classes"):
        CustomModel(cfg, num_classes=2, input_channels=14, task_groups=groups())
    model = CustomModel(cfg, num_classes=3, input_channels=14, task_groups=groups()).eval()
    assert set(model(torch.zeros(1, 14, 32, 48))) == {"groups"}


@pytest.mark.parametrize("metadata", [[], [DEFINITIONS[0]],
    [SimpleNamespace(name="all", classes=("Car", "Pedestrian"), global_ids=(0,))],
    [SimpleNamespace(name="all", classes=("Car", "Pedestrian"), global_ids=(0, 0))]])
def test_grouped_header_rejects_malformed_resolved_metadata(metadata):
    cls = importlib.import_module("detector.core.models.heads.grouped").GroupedHeader
    with pytest.raises(ValueError):
        cls(metadata, in_channels=32)


@pytest.mark.parametrize("cfg", [{"head_mode": "bogus"}, {"cls_encoding": "bogus"},
                                 {"header_use_iou": "false"}])
def test_direct_model_rejects_invalid_prediction_settings(cfg):
    with pytest.raises(ValueError):
        CustomModel({"backbone": "mobilepixornext", **cfg}, num_classes=3)


def test_legacy_flat_header_registration_and_seeded_weights_are_unchanged():
    cfg = {"backbone": "mobilepixornext", "c4_attention": "none", "header_use_iou": True}
    torch.manual_seed(23)
    implicit = CustomModel(cfg, num_classes=3, input_channels=14).eval()
    torch.manual_seed(23)
    explicit = CustomModel({**cfg, "head_mode": "legacy_single"}, num_classes=3, input_channels=14).eval()
    assert hasattr(implicit, "header") and not hasattr(implicit, "grouped_header")
    assert implicit.state_dict().keys() == explicit.state_dict().keys()
    assert any(k.startswith("header.iou.") for k in implicit.state_dict())
    for key in implicit.state_dict():
        torch.testing.assert_close(implicit.state_dict()[key], explicit.state_dict()[key], rtol=0, atol=0)
    x = torch.randn(1, 14, 32, 48)
    for key, value in implicit(x).items():
        torch.testing.assert_close(value, explicit(x)[key], rtol=0, atol=0)
