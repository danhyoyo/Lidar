"""Regression targets and decoding must preserve nearby small objects."""

import math
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "detector"), str(ROOT / "detector/core/datasets")]

from core.datasets.dataset import Dataset
from postprocess import filter_pred

GEOMETRY = {
    "x_min": 0.0, "x_max": 6.4, "x_res": 0.1,
    "y_min": -3.2, "y_max": 3.2, "y_res": 0.1,
    "z_min": -2.5, "z_max": 1.0, "z_res": 0.1,
}


def dataset(backend="python", assignment="nearest_center"):
    if backend == "numba":
        pytest.importorskip("numba")
    value = Dataset.__new__(Dataset)
    value.output_shape = [16, 16]
    value.out_size_factor = 4
    value.num_classes = 3
    value.cls_encoding = "gaussian"
    value.target_backend = backend
    value.config = {
        "gaussian_overlap": 0.1, "min_radius": 4,
        "regression_assignment": assignment,
    }
    return value


def boxes(*centers):
    return torch.tensor(
        [[1, 1.7, 0.6, 0.8, x, y, -1.7, 0.0] for x, y in centers],
        dtype=torch.float32,
    )


def decoded_target(target, box):
    px = int((box[4] - GEOMETRY["x_min"]) / 0.1 / 4)
    py = int((box[5] - GEOMETRY["y_min"]) / 0.1 / 4)
    assert target["cls"][1, py, px] == 1
    assert target["reg_mask"][py, px] == 1
    origin = torch.tensor([px * 0.4, py * 0.4 - 3.2])
    return origin + target["offset"][:, py, px]


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_nearby_objects_keep_their_own_center_targets(backend):
    values = boxes((2.05, 0.05), (2.65, 0.05))
    targets = dataset(backend).get_label(values, GEOMETRY)
    for box in values:
        torch.testing.assert_close(decoded_target(targets, box), box[4:6])


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_center_is_reserved_even_if_another_center_is_closer_to_cell_origin(backend):
    values = boxes((2.39, 0.39), (1.99, 0.39))
    targets = dataset(backend).get_label(values, GEOMETRY)
    for box in values:
        torch.testing.assert_close(decoded_target(targets, box), box[4:6])


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_overlap_targets_do_not_depend_on_label_order(backend):
    values = boxes((2.05, 0.05), (2.65, 0.05))
    first = dataset(backend).get_label(values, GEOMETRY)
    reversed_order = dataset(backend).get_label(values.flip(0), GEOMETRY)
    for name in first:
        assert torch.equal(first[name], reversed_order[name]), name
    # Cell x=7 is in both regression disks and is nearer the second center.
    expected = values[1, 4:6] - torch.tensor([2.8, 0.0])
    torch.testing.assert_close(first["offset"][:, 8, 7], expected)


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_same_cell_collision_chooses_closest_center_deterministically(backend):
    values = boxes((2.25, 0.25), (2.05, 0.05))
    first = dataset(backend).get_label(values, GEOMETRY)
    reverse = dataset(backend).get_label(values.flip(0), GEOMETRY)
    torch.testing.assert_close(decoded_target(first, values[1]), values[1, 4:6])
    for name in first:
        assert torch.equal(first[name], reverse[name]), name


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_small_regression_disk_still_supervises_its_quantized_peak(backend):
    values = boxes((2.39, 0.39))
    value = dataset(backend)
    value.config["min_radius"] = 1
    targets = value.get_label(values, GEOMETRY)
    torch.testing.assert_close(decoded_target(targets, values[0]), values[0, 4:6])


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_exact_same_center_tie_has_label_order_independent_geometry(backend):
    values = boxes((2.05, 0.05), (2.05, 0.05))
    values[1, 2:4] = torch.tensor([0.7, 1.0])
    first = dataset(backend).get_label(values, GEOMETRY)
    reverse = dataset(backend).get_label(values.flip(0), GEOMETRY)
    for name in first:
        assert torch.equal(first[name], reverse[name]), name
    torch.testing.assert_close(first["size"][:, 8, 5], values[0, 2:4].log())


def test_python_and_numba_targets_match_with_overlaps_boundaries_and_empty_boxes():
    pytest.importorskip("numba")
    rng = np.random.default_rng(42)
    random_boxes = boxes(*rng.uniform([0.01, -3.19], [6.39, 3.19], size=(30, 2)))
    for values in [random_boxes, boxes((0.01, -3.19), (6.39, 3.19)), boxes()]:
        values = values.reshape(-1, 8)
        for assignment in ["nearest_center", "legacy"]:
            first = dataset("python", assignment).get_label(values, GEOMETRY)
            compiled = dataset("numba", assignment).get_label(values, GEOMETRY)
            for name in first:
                assert first[name].numpy().tobytes() == compiled[name].numpy().tobytes(), name


def test_legacy_target_assignment_remains_available():
    values = boxes((2.05, 0.05), (2.65, 0.05))
    targets = dataset(assignment="legacy").get_label(values, GEOMETRY)
    torch.testing.assert_close(decoded_target(targets, values[0]), values[1, 4:6])


def predictions(with_iou=False):
    result = {
        "cls": torch.full((1, 3, 16, 16), -20.0),
        "offset": torch.zeros(1, 2, 16, 16),
        "size": torch.full((1, 2, 16, 16), math.log(0.2)),
        "yaw": torch.cat([torch.ones(1, 1, 16, 16), torch.zeros(1, 1, 16, 16)], 1),
    }
    if with_iou:
        result["iou"] = torch.full((1, 1, 16, 16), 3.0)
    return result


@pytest.mark.parametrize("nms_threshold", [None, 0.1])
@pytest.mark.parametrize("with_iou", [False, True])
def test_adjacent_different_classes_survive_peak_extraction(nms_threshold, with_iou):
    pred = predictions(with_iou)
    pred["cls"][0, 0, 8, 5] = 4.0
    pred["cls"][0, 1, 8, 6] = 3.0
    result = filter_pred(pred, {"geometry": GEOMETRY}, 4, 0.5, nms_threshold)
    assert result.shape == (2, 7)
    assert set(result[:, 0]) == {0, 1}
    assert sorted(result[:, 2].tolist()) == pytest.approx([2.0, 2.4])


@pytest.mark.parametrize("nms_threshold", [None, 0.1])
def test_different_classes_at_same_cell_are_independent_candidates(nms_threshold):
    pred = predictions()
    pred["cls"][0, 0, 8, 5] = 4.0
    pred["cls"][0, 1, 8, 5] = 3.0
    result = filter_pred(pred, {"geometry": GEOMETRY}, 4, 0.5, nms_threshold)
    assert result.shape == (2, 7)
    assert set(result[:, 0]) == {0, 1}


def test_weaker_adjacent_peak_in_same_class_is_still_suppressed():
    pred = predictions()
    pred["cls"][0, 1, 8, 5] = 4.0
    pred["cls"][0, 1, 8, 6] = 3.0
    result = filter_pred(pred, {"geometry": GEOMETRY}, 4, 0.5, 0.1)
    assert result.shape == (1, 7)


def test_legacy_peak_extraction_remains_available():
    pred = predictions()
    pred["cls"][0, 0, 8, 5] = 4.0
    pred["cls"][0, 1, 8, 6] = 3.0
    result = filter_pred(pred, {"geometry": GEOMETRY, "peak_mode": "legacy"}, 4, 0.5, 0.1)
    assert result.shape == (1, 7)


def test_unknown_peak_mode_is_rejected():
    with pytest.raises(ValueError, match="peak_mode"):
        filter_pred(predictions(), {"geometry": GEOMETRY, "peak_mode": "typo"}, 4, 0.5)


def test_unknown_assignment_is_rejected():
    with pytest.raises(ValueError, match="regression_assignment"):
        dataset(assignment="typo").get_label(boxes((2.05, 0.05)), GEOMETRY)


def test_evaluator_accepts_explicit_peak_mode():
    sys.path.insert(0, str(ROOT / "tools/kitti_training_pipeline"))
    from evaluate_kitti_bev import parser
    args = parser().parse_args([
        "--name", "comparison", "--backend", "pytorch", "--model", "model.pt",
        "--config", "config.json", "--detector-root", "detector",
        "--kitti-root", "kitti", "--split", "val.txt", "--output", "eval.json",
        "--peak-mode", "legacy",
    ])
    assert args.peak_mode == "legacy"


def test_evaluator_rejects_unknown_peak_mode_before_loading_files():
    sys.path.insert(0, str(ROOT / "tools/kitti_training_pipeline"))
    from evaluate_kitti_bev import run_evaluation
    with pytest.raises(ValueError, match="peak_mode"):
        run_evaluation(
            name="bad", backend="pytorch", model_path=Path("missing"),
            config_path=Path("missing"), detector_root=Path("missing"),
            kitti_root=Path("missing"), split_path=Path("missing"), peak_mode="typo",
        )


@pytest.mark.parametrize("peak_mode", [None, "legacy"])
@pytest.mark.parametrize("use_iou", [False, True])
def test_evaluation_records_actual_peak_and_quality_modes(tmp_path, peak_mode, use_iou):
    sys.path.insert(0, str(ROOT / "tools/kitti_training_pipeline"))
    from common import build_model
    from evaluate_kitti_bev import run_evaluation

    processed = tmp_path / "processed"
    (processed / "pointcloud").mkdir(parents=True)
    np.array([[2.05, 0.05, -1.0, 0.8]], dtype=np.float32).tofile(
        processed / "pointcloud/000000.bin"
    )
    raw = tmp_path / "raw"
    for folder in ["label_2", "calib"]:
        (raw / "training" / folder).mkdir(parents=True)
    (raw / "training/label_2/000000.txt").write_text(
        "Pedestrian 0 0 0 0 0 50 100 1.7 0.6 0.8 2.05 0.05 -1 0\n"
    )
    (raw / "training/calib/000000.txt").write_text(
        "R0_rect: 1 0 0 0 1 0 0 0 1\n"
        "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    )
    split = tmp_path / "val.txt"
    split.write_text("000000;kitti\n")
    config = {
        "data": {
            "num_classes": 3, "out_size_factor": 4,
            "bev_encoding": {"name": "rich8", "out_channels": 8},
            "kitti": {"location": str(processed), "geometry": GEOMETRY,
                      "objects": {"Car": 0, "Pedestrian": 1, "Cyclist": 2}},
        },
        "model": {"backbone": "mobilepixornext", "backbone_out_dim": 16,
                  "c4_attention": "none", "cls_encoding": "gaussian",
                  "header_use_iou": use_iou},
        "loss": {"name": "baseline", "use_iou": use_iou},
        "augmentation": {"p": 0.0, "rotation": {"use": False},
                         "scaling": {"use": False}, "translation": {"use": False}},
    }
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    model_path = tmp_path / "model.pt"
    torch.save(build_model(config).state_dict(), model_path)
    result = run_evaluation(
        name="synthetic", backend="pytorch", model_path=model_path,
        config_path=config_path, detector_root=ROOT / "detector",
        kitti_root=raw, split_path=split, output_path=tmp_path / "evaluation.json",
        device="cpu", score_threshold=1.0, warmup_frames=0, progress_every=0,
        peak_mode=peak_mode,
    )
    assert result["protocol"]["peak_mode"] == (peak_mode or "per_class")
    assert result["protocol"]["nms_alpha"] == (0.5 if use_iou else 0.0)
    assert result["counts"]["detections"] == 0
    assert result["accuracy"]["per_class"]["Pedestrian"]["difficulties"]["Moderate"]["ground_truth"] == 1
