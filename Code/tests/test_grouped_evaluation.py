"""Actual grouped checkpoint inference, saved predictions and parameter reports."""

import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from test_grouped_checkpoint import components, payload
from test_grouped_training import configured_model
from test_grouped_targets import make_dataset
from test_grouped_decode import groups, grouped_predictions, GEOMETRY
import common
import postprocess
import evaluate_kitti_bev as evaluation
from tools.benchmarks.profile_detector import profile_detector

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("classification,iqa", [("gaussian", True), ("binary", True), ("binary", False)])
def test_profiler_counts_grouped_heads_criterion_and_nested_output_shapes(classification, iqa):
    config = configured_model(classification, "oga", iqa)
    before = copy.deepcopy(config)
    report = profile_detector(config, shape=(1, 14, 32, 48))
    assert config == before and report["head_mode"] == "grouped"
    counts = report["parameter_counts"]
    assert counts["backbone_including_neck"] == 660528 < 1_000_000
    assert counts["backbone_body"] + counts["neck"] == 660528
    assert counts["heads"] == sum(report["head_parameter_counts"].values())
    assert counts["total_detector"] == counts["heads"] + 660528
    assert counts["criterion_train_only"] == (12 if iqa else 10)
    if classification == "gaussian":
        assert counts["heads"] == 186161 and counts["total_detector"] == 846689
    output = report["output_shapes"]["groups"]
    assert output["car"]["cls"] == [1, 1 + int(classification == "binary"), 8, 12]
    assert output["ped_cyc"]["cls"] == [1, 2 + int(classification == "binary"), 8, 12]
    assert report["task_groups"][1]["global_ids"] == [1, 2]
    assert any(name.startswith("grouped_header.heads.car.") for name in report["conv2d_mac_estimate"]["by_module"])


def experiment(tmp_path, classification="gaussian", iqa=True, reordered=False):
    dataset = make_dataset(tmp_path, classification=classification)
    config = configured_model(classification, "oga", iqa)
    config["data"] = dataset.config
    if reordered:
        config["data"]["kitti"]["objects"] = {"Car": 2, "Pedestrian": 0, "Cyclist": 1}
    config["data"]["head_groups"][1]["classes"] = ["Cyclist", "Pedestrian"]
    config["augmentation"] = {"p": 0., "rotation": {"use": False}, "scaling": {"use": False}, "translation": {"use": False}}
    parts = components(config)
    with torch.no_grad():
        for name, head in parts[0].grouped_header.heads.items():
            for parameter in head.parameters():
                parameter.zero_()
            probabilities = ([.9] if name == "car" else [.8, .7]) if classification == "gaussian" else (
                [.1, .9] if name == "car" else [.1, .6, .3])
            bias = torch.logit(torch.tensor(probabilities)) if classification == "gaussian" else torch.tensor(probabilities).log()
            head.cls.head.bias.copy_(bias)
            head.size.head.bias.fill_(math.log(20.))
            head.yaw.head.bias[0] = 1.
    source = tmp_path / "checkpoint.pt"
    torch.save(payload(config, parts, epoch=0), source)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    raw = tmp_path / "raw"
    for folder in ("label_2", "calib"):
        (raw / "training" / folder).mkdir(parents=True)
    (raw / "training/label_2/000000.txt").write_text("\n".join(
        f"{cls} 0 0 0 0 0 50 100 1.7 .6 .8 2.05 .05 -1 0" for cls in ("Car", "Pedestrian", "Cyclist")) + "\n")
    (raw / "training/calib/000000.txt").write_text(
        "R0_rect: 1 0 0 0 1 0 0 0 1\nTr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n")
    return config, source, config_path, Path(dataset.data_file), raw


@pytest.mark.parametrize("classification,iqa,reordered", [("gaussian", True, False), ("binary", True, False),
                                                       ("gaussian", False, True)])
def test_actual_grouped_checkpoint_evaluation_saves_global_classes_quality_protocol_and_counts(tmp_path, classification, iqa, reordered):
    config, source, path, split, raw = experiment(tmp_path, classification, iqa, reordered)
    output = tmp_path / "evaluation.json"
    result = evaluation.run_evaluation(name="grouped", backend="pytorch", model_path=source,
        config_path=path, detector_root=ROOT / "detector", kitti_root=raw, split_path=split,
        output_path=output, device="cpu", score_threshold=.1, nms_threshold=.1, max_detections=3,
        warmup_frames=0, progress_every=0)
    predictions = np.load(output.with_suffix(".predictions.npz"))["000000"]
    assert predictions.shape == (3, 7) and predictions.dtype == np.float32
    ids = config["data"]["kitti"]["objects"]
    assert predictions[:, 0].tolist() == [ids["Car"], ids["Cyclist"], ids["Pedestrian"]]
    assert result["protocol"]["head_mode"] == "grouped"
    assert result["protocol"]["box_mode"] == "bev"
    assert result["protocol"]["cls_encoding"] == classification
    assert result["protocol"]["class_ids"] == ids
    assert result["protocol"]["nms_alpha"] == (.5 if iqa else 0.)
    assert result["protocol"]["quality_target"] == ("mgiou" if iqa else None)
    assert result["protocol"]["quality_warmup_epochs"] == 0
    assert result["protocol"]["group_weights"] == [["car", .5], ["ped_cyc", .5]]
    assert result["protocol"]["peak_scope"] == "within each group"
    assert result["model"]["checkpoint_epoch"] == 0
    assert result["protocol"]["max_detections_scope"] == "global after classwise NMS"
    assert result["protocol"]["task_groups"][1]["classes"] == ["Cyclist", "Pedestrian"]
    assert len(result["data"]["resolved_config_sha256"]) == 64
    assert result["data"]["split_sha256"] == common.sha256(split)
    assert result["data"]["encoding_semantic_hash"] == common.bev_encoding_spec(config).semantic_hash
    assert "dirty" in result["data"]["source"]
    counts = result["model"]["parameter_counts"]
    assert counts["backbone_including_neck"] == 660528
    assert counts["heads"] == sum(result["model"]["head_parameter_counts"].values())
    assert counts["criterion_train_only"] == (12 if iqa else 10)
    assert counts["total_detector"] == result["model"]["parameters"]
    for cls in evaluation.CLASSES:
        metrics = result["accuracy"]["per_class"][cls]["difficulties"]["Moderate"]
        assert metrics["ground_truth"] == 1 and metrics["ranked_predictions"] == 1
    saved = json.loads(output.read_text())
    assert saved["counts"]["detections"] == 3


@pytest.mark.parametrize("changed", ["class_order", "dilations", "iqa_method", "missing_identity", "raw_weights"])
def test_grouped_evaluation_rejects_semantic_mismatch_even_when_shapes_match(tmp_path, changed):
    config, source, _, _, _ = experiment(tmp_path)
    checkpoint = torch.load(source, weights_only=True)
    if changed == "class_order":
        config["data"]["head_groups"][1]["classes"].reverse()
    elif changed == "dilations":
        config["model"]["c4_context_dilations"] = [1, 2, 4]
    elif changed == "iqa_method":
        config["loss"]["iou_target_type"] = "rotated_iou"
    elif changed == "missing_identity":
        del checkpoint["checkpoint_identity"]
    else:
        checkpoint = checkpoint["model_state_dict"]
    torch.save(checkpoint, source)
    with pytest.raises(ValueError, match="identity"):
        evaluation.PyTorchRunner(source, config, "cpu")


def test_evaluation_ignores_training_only_identity_fields(tmp_path):
    config, source, _, _, _ = experiment(tmp_path)
    config["train"]["epochs"] = 99
    runner = evaluation.PyTorchRunner(source, config, "cpu")
    assert runner.metadata()["parameter_counts"]["criterion_train_only"] == 12


def test_standalone_profiler_accepts_grouped_config_and_preserves_nested_shapes(tmp_path):
    config = configured_model()
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    output = tmp_path / "profile.json"
    subprocess.run([sys.executable, str(ROOT / "tools/benchmarks/profile_detector.py"), "--config", str(path),
                    "--shape", "1", "14", "32", "48", "--output", str(output)], cwd=tmp_path,
                   check=True, capture_output=True, text=True)
    report = json.loads(output.read_text())
    assert report["head_mode"] == "grouped" and report["parameter_counts"]["total_detector"] == 846689
    assert "ped_cyc" in report["output_shapes"]["groups"]


def test_saved_grouped_rows_feed_ap_evaluator_with_supplied_global_ids():
    ids = {"Car": 2, "Pedestrian": 0, "Cyclist": 1}
    predictions = {"frame": np.array([[2, .9, 3, 1, 4, 2, 0], [0, .8, 3, 1, 1, .6, 0],
                                       [1, .7, 3, 1, 1.2, .7, 0]], np.float32)}
    labels = {"frame": [evaluation.GroundTruth(cls, 0., 0, 100., *row[2:])
                        for cls, row in zip(("Car", "Pedestrian", "Cyclist"), predictions["frame"])]}
    accuracy = evaluation.evaluate_accuracy(predictions, labels, class_ids=ids)
    assert accuracy["map_moderate_percent"] == 100.


def test_comparison_flatten_keeps_architecture_quality_and_parameter_metadata():
    from compare_models import flatten
    result = {"model": {"parameters": 846689, "parameter_counts": {
        "backbone_body": 629984, "neck": 30544, "backbone_including_neck": 660528,
        "heads": 186161, "total_detector": 846689, "criterion_train_only": 12}},
        "protocol": {"head_mode": "grouped", "cls_encoding": "gaussian", "quality_target": "mgiou", "nms_alpha": .5},
        "data": {"resolved_config_sha256": "abc", "split_sha256": "def"}}
    row = flatten(result)
    assert row["head_mode"] == "grouped" and row["backbone_including_neck_parameters"] == 660528
    assert row["heads_parameters"] == 186161 and row["criterion_train_only_parameters"] == 12
    assert row["quality_target"] == "mgiou" and row["nms_alpha"] == .5
    assert row["resolved_config_sha256"] == "abc" and row["split_sha256"] == "def"


def test_grouped_exact_qoga_eval_reports_classification_quality_and_curriculum_identity(tmp_path):
    config, source, path, split, raw = experiment(tmp_path, iqa=False)
    config["loss"] = {"name": "q_oga", "use_iou": False, "quality_target": "rotated_iou", "quality_warmup_epochs": 4}
    torch.save(payload(config, components(config), epoch=0), source)
    path.write_text(json.dumps(config))
    result = evaluation.run_evaluation(name="exact", backend="pytorch", model_path=source,
        config_path=path, detector_root=ROOT / "detector", kitti_root=raw, split_path=split,
        device="cpu", score_threshold=1., warmup_frames=0, progress_every=0)
    assert result["protocol"]["quality_target"] == "rotated_iou"
    assert result["protocol"]["quality_warmup_epochs"] == 4
    assert result["protocol"]["nms_alpha"] == 0.
    assert result["counts"]["detections"] == 0


def test_real_comparison_cli_preserves_grouped_reports_in_json_and_csv(tmp_path):
    from compare_models import main
    _, source, path, split, raw = experiment(tmp_path)
    output = tmp_path / "comparison"
    main(["--model", f"grouped=pytorch:{source}", "--config", str(path), "--detector-root", str(ROOT / "detector"),
          "--kitti-root", str(raw), "--split", str(split), "--output-dir", str(output), "--device", "cpu",
          "--score-threshold", ".1", "--nms-threshold", ".1", "--max-detections", "3",
          "--warmup-frames", "0", "--progress-every", "0", "--fail-on-model-error"])
    result = json.loads((output / "comparison.json").read_text())
    assert result["status"] == "ok" and result["models_succeeded"] == 1
    assert result["models"][0]["model"]["parameter_counts"]["total_detector"] == 846689
    assert "backbone_including_neck_parameters" in (output / "comparison.csv").read_text()


def test_grouped_profile_full_resolution_has_iqa_shapes_and_correct_total_budget():
    report = profile_detector(configured_model())
    assert report["input_shape"] == [1, 14, 800, 704]
    assert report["output_shapes"]["groups"]["ped_cyc"]["iou"] == [1, 1, 200, 176]
    assert report["parameter_counts"]["total_detector"] == 846689
