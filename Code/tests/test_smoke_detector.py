"""CPU orchestration/error checks for the short detector smoke CLI."""

import copy
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "detector"), str(ROOT / "tools/kitti_training_pipeline")]
CONFIG = ROOT / "configs/experiments/under1m/hist14_local3_focal_grouped_oga_iqa.json"


def smoke():
    assert (ROOT / "tools/benchmarks/smoke_detector.py").is_file(), "Detector smoke CLI is not implemented"
    return importlib.import_module("tools.benchmarks.smoke_detector")


def test_cpu_fp32_smoke_uses_real_dataset_optimizer_checkpoint_and_decoder(tmp_path):
    config = json.loads(CONFIG.read_text())
    original = copy.deepcopy(config)
    result = smoke().run_smoke(config, device="cpu", precisions=["fp32"], steps=2, num_workers=0)
    assert config == original
    assert result["status"] == "passed" and result["device"] == "cpu"
    assert result["accuracy_measured"] is False
    assert result["synthetic"] is True and result["compile_model"] is False
    run = result["precisions"]["fp32"]
    assert run["status"] == "passed" and run["optimizer_updates"] == 2
    assert run["finite_loss"] and run["finite_gradients"] and run["head_weight_changed"]
    assert run["checkpoint_restore_verified"] and run["criterion_state_unchanged_in_validation"]
    assert run["loss_dtype"] == "torch.float32" and run["focal_modulation_input_dtype"] == "torch.float32"
    assert run["input_shape"] == [2, 14, 64, 48]
    assert run["validation_groups"] == ["car", "ped_cyc"]
    assert run["negative_only_group_seen"] and run["empty_scene_seen"]
    assert run["detection_shape"][1] == 7
    assert result["source_config_identity"]["model"]["features"]["config"]["c4_context"] == "focal"
    assert result["worker_count"] == 0


@pytest.mark.parametrize("kwargs", [{"steps": 0}, {"steps": True}, {"num_workers": -1},
    {"num_workers": True}, {"precisions": ["fp16"]}, {"precisions": ["fp32", "typo"]},
    {"precisions": ["fp32", "fp32"]}, {"device": "meta"}])
def test_invalid_smoke_request_fails_before_data_or_device_work(kwargs):
    options = {"device": "cpu", "precisions": ["fp32"], "steps": 1, "num_workers": 0, **kwargs}
    with pytest.raises(ValueError):
        smoke().run_smoke(json.loads(CONFIG.read_text()), **options)


def test_cpu_optional_precisions_are_reported_unsupported_without_fake_gpu_pass():
    result = smoke().run_smoke(json.loads(CONFIG.read_text()), device="cpu",
        precisions=["fp32", "fp16", "bf16"], steps=1, num_workers=0)
    assert result["status"] == "partial"
    for precision in ("fp16", "bf16"):
        assert result["precisions"][precision]["status"] == "unsupported"
        assert "CUDA" in result["precisions"][precision]["reason"]


def test_cpu_cli_writes_machine_readable_report_outside_repository(tmp_path):
    assert (ROOT / "tools/benchmarks/smoke_detector.py").is_file()
    output = tmp_path / "smoke.json"
    result = subprocess.run([sys.executable, str(ROOT / "tools/benchmarks/smoke_detector.py"),
        "--config", str(CONFIG), "--device", "cpu", "--precisions", "fp32", "--steps", "1",
        "--num-workers", "0", "--output", str(output)], capture_output=True, text=True, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text())
    assert report["status"] == "passed" and report["precisions"]["fp32"]["optimizer_updates"] == 1
