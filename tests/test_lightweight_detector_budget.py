"""Audit instantiated body/neck/head budgets and full-resolution shape contracts."""

import copy
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "detector"), str(ROOT / "detector/core/datasets")]

from core.models.backbones.mobilepixornext_blocks import LiteMLARefinement
from core.models.model import CustomModel


def profiler():
    path = ROOT / "tools/benchmarks/profile_detector.py"
    assert path.is_file(), "detector profiler has not been implemented"
    return importlib.import_module("tools.benchmarks.profile_detector")


def test_actual_reference_backbone_full_shape_and_strict_under1m_budget():
    module = profiler()
    config = module.reference_config()
    assert config["model"]["stage_depths"] == [3, 4, 2]
    assert config["model"]["backbone_out_dim"] == 32
    assert config["data"]["bev_encoding"]["name"] == "hist14"
    with torch.device("meta"):
        model = CustomModel(dict(config["model"], bev_encoding=config["data"]["bev_encoding"]), num_classes=3)
    assert not any(isinstance(m, LiteMLARefinement) for m in model.modules())
    report = module.profile_detector(config)
    counts = report["parameter_counts"]
    assert counts == {"backbone_body": 604864, "neck": 30544,
                      "backbone_including_neck": 635408, "heads": 93130,
                      "total_detector": 728538, "criterion_train_only": 6}
    assert counts["backbone_including_neck"] < 1_000_000
    assert counts["backbone_body"] + counts["neck"] == counts["backbone_including_neck"]
    assert counts["backbone_including_neck"] + counts["heads"] == counts["total_detector"]
    assert report["input_shape"] == [1, 14, 800, 704]
    assert report["backbone_output_shape"] == [1, 32, 200, 176]
    assert report["output_shapes"]["cls"] == [1, 3, 200, 176]
    assert report["output_shapes"]["iou"] == [1, 1, 200, 176]
    assert report["conv2d_mac_estimate"]["total"] > 0
    assert "interpolation" in report["conv2d_mac_estimate"]["excluded_operations"]
    assert "latency_ms" not in report


def test_parameter_counts_change_with_actual_iqa_and_stage_selection():
    module = profiler()
    config = module.reference_config()
    config["model"].update(header_use_iou=False, stage_depths=[2, 4, 2])
    config["loss"] = {"name": "baseline", "use_iou": False}
    original = copy.deepcopy(config)
    report = module.profile_detector(config, shape=(1, 14, 32, 48), device="cpu")
    assert config == original
    counts = report["parameter_counts"]
    assert counts["backbone_including_neck"] == 621296
    assert counts["heads"] == 74537
    assert counts["criterion_train_only"] == 0
    assert report["backbone_output_shape"] == [1, 32, 8, 12]
    assert "iou" not in report["output_shapes"]


def test_budget_gate_rejects_equal_to_limit_before_forward():
    module = profiler()
    with pytest.raises(ValueError, match="635408.*strictly less"):
        module.profile_detector(module.reference_config(), max_backbone_parameters=635408)


def test_mac_estimate_includes_grouped_depthwise_convolutions_once():
    module = profiler()
    report = module.profile_detector(module.reference_config(), shape=(1, 14, 32, 48))
    # First stage2 DW7: output [1,48,8,12], one input channel per group.
    assert report["conv2d_mac_estimate"]["by_module"]["backbone.stage2.0.dwconv"] == 48 * 8 * 12 * 49


def test_profile_cli_is_standalone_and_saves_resolved_configuration(tmp_path):
    path = ROOT / "tools/benchmarks/profile_detector.py"
    assert path.is_file(), "detector profiler has not been implemented"
    output = tmp_path / "profile.json"
    subprocess.run([sys.executable, str(path), "--output", str(output)], cwd=tmp_path,
                   check=True, capture_output=True, text=True)
    report = json.loads(output.read_text())
    assert report["resolved_config"]["model"]["stage_depths"] == [3, 4, 2]
    assert report["parameter_counts"]["backbone_including_neck"] == 635408
    assert len(report["config_sha256"]) == 64


@pytest.mark.parametrize('variant,backbone,neck,local', [
    ('reference',635408,30544,0), ('focal',660528,30544,0),
    ('focal_eca',660532,30544,4), ('focal_simam',660529,30544,1),
    ('focal_fusion32',666768,36784,0), ('focal_detail',661720,31736,0),
    ('focal_fusion32_detail_eca',668244,38256,4),
])
def test_integrated_variants_full_shape_budget_and_feature_accounting(variant, backbone, neck, local):
    module=profiler()
    config=module.variant_config(variant)
    before=copy.deepcopy(config)
    report=module.profile_detector(config)
    assert config==before
    counts=report['parameter_counts']
    assert counts['backbone_including_neck']==backbone<1_000_000
    assert counts['neck']==neck
    assert counts['heads']==93130 and counts['total_detector']==backbone+93130
    assert counts['criterion_train_only']==6
    assert report['backbone_output_shape']==[1,32,200,176]
    features=report['feature_parameter_counts']
    assert features['context']['total']==(0 if variant=='reference' else 25120)
    assert features['local']['total']==local
    assert features['local']['residual_scale']==(1 if local else 0)
    assert features['local']['core']==(3 if 'eca' in variant else 0)
    assert features['detail']['total']==(1472 if 'fusion32_detail' in variant else 1192 if 'detail' in variant else 0)
    assert report['feature_semantic_hash'] and report['head_mode']=='legacy_single'
    if variant!='reference':
        assert report['resolved_config']['model']['c4_context_bottleneck']==64
        assert features['context']['residual_scale']==96
    assert 'Conv1d channel correlation' in report['conv2d_mac_estimate']['excluded_operations']


def test_integrated_focal_budget_rejects_equality_and_cpu_output_is_finite():
    module=profiler()
    cfg=module.variant_config('focal_eca')
    with pytest.raises(ValueError,match='660532.*strictly less'):
        module.profile_detector(cfg,max_backbone_parameters=660532)
    report=module.profile_detector(cfg,device='cpu',shape=(1,14,32,48))
    assert report['backbone_output_shape']==[1,32,8,12]
    assert report['feature_parameter_counts']['local']['total']==4


def test_profile_cli_selects_focal_variant(tmp_path):
    output=tmp_path/'focal.json'
    subprocess.run([sys.executable,str(ROOT/'tools/benchmarks/profile_detector.py'),
                    '--variant','focal','--output',str(output)],cwd=tmp_path,
                   check=True,capture_output=True,text=True)
    report=json.loads(output.read_text())
    assert report['parameter_counts']['backbone_including_neck']==660528
