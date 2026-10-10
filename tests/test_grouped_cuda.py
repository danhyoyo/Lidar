"""Real CUDA/AMP gates: hardware skips are explicitly not readiness evidence."""

import importlib
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "detector"), str(ROOT / "tools/kitti_training_pipeline")]
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Dedicated CUDA device required")


def module():
    assert (ROOT / "tools/benchmarks/smoke_detector.py").is_file(), "CUDA smoke runner is missing"
    return importlib.import_module("tools.benchmarks.smoke_detector")


def config(local="none", strategy="oga", iqa=True, classification="gaussian"):
    filename = ("hist14_local3_focal_grouped_oga_iqa.json" if local == "none" else
                f"hist14_local3_focal_{local}_grouped_oga_iqa.json")
    value = json.loads((ROOT / "configs/experiments/under1m" / filename).read_text())
    value["model"].update(header_use_iou=iqa, cls_encoding=classification)
    value["loss"] = {"name": "q_oga" if strategy == "exact_q_oga" else strategy, "use_iou": iqa,
                     "group_weights": {"car": 1., "ped_cyc": 1.}}
    if strategy == "exact_q_oga":
        value["loss"].update(quality_target="rotated_iou", quality_warmup_epochs=2)
    return value


@pytest.mark.parametrize("local", ["none", "eca", "simam"])
@pytest.mark.parametrize("classification,strategy,iqa", [
    ("gaussian", "baseline", True), ("gaussian", "oga", True), ("gaussian", "uwag", False),
    ("gaussian", "q_oga", False), ("gaussian", "exact_q_oga", False),
    ("binary", "baseline", True), ("binary", "oga", True), ("binary", "uwag", False)])
def test_cuda_fp32_all_adapters_supported_objectives_and_empty_targets(local, classification, strategy, iqa):
    report = module().run_smoke(config(local, strategy, iqa, classification), device="cuda",
        precisions=["fp32"], steps=2, num_workers=0)
    run = report["precisions"]["fp32"]
    assert report["status"] == "passed" and report["cuda_device_name"]
    assert run["optimizer_updates"] == 2 and run["finite_gradients"]
    assert run["checkpoint_restore_verified"]
    assert run["negative_only_group_seen"] and run["empty_scene_seen"]
    assert run["model_and_criterion_on_device"] and run["optimizer_membership_exact"]
    assert run["peak_cuda_memory_bytes"] > 0 and run["loss_dtype"] == "torch.float32"


@pytest.mark.parametrize("local", ["none", "eca", "simam"])
@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_cuda_amp_main_objective_preserves_fp32_boundaries_and_updates(local, precision):
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("Device does not support BF16; this is not BF16 readiness")
    report = module().run_smoke(config(local), device="cuda", precisions=["fp32", precision],
                              steps=2, num_workers=0)
    assert report["status"] == "passed"
    run = report["precisions"][precision]
    assert run["optimizer_updates"] == 2 and run["head_weight_changed"]
    assert run["loss_dtype"] == run["focal_modulation_input_dtype"] == "torch.float32"
    assert run["head_output_dtype"] == ("torch.float16" if precision == "fp16" else "torch.bfloat16")
    if local != "none":
        assert run["local_gate_dtype"] == "torch.float32"


@pytest.mark.parametrize("local", ["none", "eca", "simam"])
def test_cuda_actual_configured_worker_count_and_full_resolution_inference(local):
    value = config(local)
    report = module().run_smoke(value, device="cuda", precisions=["fp32"], steps=1,
                              num_workers=value["train"]["num_workers"], full_resolution=True)
    assert report["status"] == "passed" and report["worker_count"] == 6
    run = report["precisions"]["fp32"]
    assert run["full_resolution_input_shape"] == [1, 14, 800, 704]
    assert run["full_resolution_output_shapes"]["groups"]["ped_cyc"]["cls"] == [1, 2, 200, 176]
    assert run["full_resolution_finite"] and run["checkpoint_restore_verified"]
