"""Deferred deployment must fail before discarding grouped or quality outputs."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "detector"), str(ROOT / "tools/kitti_training_pipeline")]

from common import build_model
from export_onnx import RawHeadWrapper
from evaluate_kitti_bev import TensorRTRunner


def configuration(grouped=False, iqa=False, binary=False):
    config = json.loads((ROOT / "configs/config.json").read_text())
    config["model"].update(c4_attention="none", c4_attention_scales=[],
                           head_mode="grouped" if grouped else "legacy_single",
                           header_use_iou=iqa, cls_encoding="binary" if binary else "gaussian")
    config["loss"] = {"name": "oga", "use_iou": iqa}
    if grouped:
        config["data"]["head_groups"] = [
            {"name": "car", "classes": ["Car"]},
            {"name": "ped_cyc", "classes": ["Pedestrian", "Cyclist"]}]
    return config


@pytest.mark.parametrize("binary", [False, True])
def test_legacy_wrapper_preserves_four_outputs_exactly(binary):
    model = build_model(configuration(binary=binary)).eval()
    voxel = torch.randn(1, 8, 32, 48)
    with torch.no_grad():
        expected = model(voxel)
        actual = RawHeadWrapper(model)(voxel)
    assert RawHeadWrapper.OUTPUT_NAMES == ("cls", "offset", "size", "yaw")
    assert len(actual) == 4
    for name, tensor in zip(RawHeadWrapper.OUTPUT_NAMES, actual):
        torch.testing.assert_close(tensor, expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("grouped,iqa,reason", [(True, False, "grouped"),
                                              (True, True, "grouped"), (False, True, "IQA")])
def test_wrapper_rejects_unsupported_actual_model(grouped, iqa, reason):
    with pytest.raises(ValueError, match=reason):
        RawHeadWrapper(build_model(configuration(grouped, iqa)))


@pytest.mark.parametrize("grouped,iqa,reason", [(True, False, "grouped"),
                                              (True, True, "grouped"), (False, True, "IQA")])
def test_export_cli_rejects_before_checkpoint_loading_or_output_write(tmp_path, grouped, iqa, reason):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(configuration(grouped, iqa)))
    checkpoint = tmp_path / "unreadable.pt"
    checkpoint.write_bytes(b"not a checkpoint")
    output = tmp_path / "export/model.onnx"
    result = subprocess.run([sys.executable, str(ROOT / "tools/kitti_training_pipeline/export_onnx.py"),
        "--config", str(path), "--checkpoint", str(checkpoint), "--detector-root", str(ROOT / "detector"),
        "--output", str(output), "--device", "cpu"], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode != 0
    assert reason in result.stderr and "deferred" in result.stderr
    assert not output.parent.exists()


@pytest.mark.parametrize("grouped,iqa,reason", [(True, False, "grouped"),
                                              (True, True, "grouped"), (False, True, "IQA")])
def test_trt_rejects_before_cuda_package_or_plan_access(tmp_path, grouped, iqa, reason):
    config = configuration(grouped, iqa)
    original = copy.deepcopy(config)
    with pytest.raises(ValueError, match=reason):
        TensorRTRunner(tmp_path / "missing.plan", config, "cpu")
    assert config == original


def test_legacy_trt_reaches_existing_cuda_requirement(tmp_path):
    with pytest.raises(ValueError, match="TensorRT requires CUDA"):
        TensorRTRunner(tmp_path / "missing.plan", configuration(), "cpu")
