#!/usr/bin/env python3
"""Small dependency-free checks for MobileBEV-Lite."""

import argparse
import importlib
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "detector" / "core" / "datasets"


def load_preprocess():
    sys.path.insert(0, str(DATASETS))
    return importlib.import_module("utils_1.preprocess")


def legacy_reference(points, geometry):
    eps = 0.001
    keep = (
        (points[:, 0] > geometry["x_min"] + eps)
        & (points[:, 0] < geometry["x_max"] - eps)
        & (points[:, 1] > geometry["y_min"] + eps)
        & (points[:, 1] < geometry["y_max"] - eps)
        & (points[:, 2] > geometry["z_min"] + eps)
        & (points[:, 2] < geometry["z_max"] - eps)
    )
    pts = points[keep]
    shape = tuple(
        int((geometry[f"{axis}_max"] - geometry[f"{axis}_min"]) / geometry[f"{axis}_res"])
        for axis in "xyz"
    )
    voxels = np.zeros(shape, dtype=np.float32)
    indexes = np.column_stack(
        [
            ((pts[:, i] - geometry[f"{axis}_min"]) // geometry[f"{axis}_res"]).astype(np.int32)
            for i, axis in enumerate("xyz")
        ]
    )
    if indexes.size:
        voxels[indexes[:, 0], indexes[:, 1], indexes[:, 2]] = 1
    return np.swapaxes(voxels, 0, 1)


def check_legacy():
    preprocess = load_preprocess()
    geometry = {
        "x_min": 0.0, "x_max": 2.0, "x_res": 1.0,
        "y_min": -1.0, "y_max": 1.0, "y_res": 1.0,
        "z_min": -1.0, "z_max": 1.0, "z_res": 1.0,
    }
    points = np.array(
        [
            [0.25, -0.75, -0.75, 0.2],
            [1.25, 0.25, 0.25, 0.8],
            [2.00, 0.00, 0.00, 0.5],
        ],
        dtype=np.float32,
    )
    actual = preprocess.voxelize(points, geometry)
    expected = legacy_reference(points, geometry)
    assert actual.dtype == np.float32
    assert actual.tobytes() == expected.tobytes()


def check_encoder():
    preprocess = load_preprocess()
    geometry = {
        "x_min": 0.0, "x_max": 2.0, "x_res": 1.0,
        "y_min": 0.0, "y_max": 2.0, "y_res": 1.0,
        "z_min": 0.0, "z_max": 3.0, "z_res": 1.0,
    }
    points = np.array(
        [
            [0.25, 0.25, 0.20, 0.10],
            [0.25, 0.25, 1.00, 0.50],
            [0.25, 0.25, 2.00, 1.20],
        ]
        + [[1.25, 1.25, 0.50, 0.25]] * 32,
        dtype=np.float32,
    )
    bev = preprocess.encode_bev(
        points,
        geometry,
        {"name": "rich8", "density_norm": 32.0, "intensity_scale": 1.0},
    )
    assert bev.shape == (2, 2, 8)
    assert bev.dtype == np.float32
    np.testing.assert_array_equal(bev[0, 0, :3], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(
        bev[0, 0, 3:],
        [2.0 / 3.0, 3.2 / 9.0, 1.0, 1.6 / 3.0, np.log(4.0) / np.log(33.0)],
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_array_equal(bev[0, 1], np.zeros(8, dtype=np.float32))
    assert bev[1, 1, 7] == 1.0
    scaled = preprocess.encode_bev(
        points,
        geometry,
        {"name": "rich8", "density_norm": 32.0, "intensity_scale": 0.5},
    )
    np.testing.assert_allclose(scaled[0, 0, 5:7], [0.6, 0.3], rtol=0, atol=1e-6)
    legacy = preprocess.encode_bev(points, geometry, {"name": "binary_slices"})
    assert legacy.tobytes() == preprocess.voxelize(points, geometry).tobytes()

    for invalid in (
        {"name": "rich8", "density_norm": 1.0, "intensity_scale": 1.0},
        {"name": "rich8", "density_norm": 32.0, "intensity_scale": 0.0},
        {"name": "unknown"},
    ):
        try:
            preprocess.encode_bev(points, geometry, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid encoding accepted: {invalid}")


def check_sampling():
    preprocess = load_preprocess()
    points = np.arange(80, dtype=np.float32).reshape(20, 4)
    dense = preprocess.sample_points(points, 1.0, "000123")
    half_a = preprocess.sample_points(points, 0.5, "000123")
    half_b = preprocess.sample_points(points, 0.5, "000123")
    quarter = preprocess.sample_points(points, 0.25, "000123")
    assert dense is points
    np.testing.assert_array_equal(half_a, half_b)
    assert len(half_a) == 10 and len(quarter) == 5
    assert {tuple(row) for row in quarter} <= {tuple(row) for row in half_a}
    source_positions = [np.flatnonzero((points == row).all(axis=1))[0] for row in half_a]
    assert source_positions == sorted(source_positions)
    assert preprocess.sample_points(points[:0], 0.25, "empty").shape == (0, 4)
    for invalid in (0.0, -0.1, 1.01, float("nan")):
        try:
            preprocess.sample_points(points, invalid, "000123")
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid sampling rate accepted: {invalid}")


def check_shapes():
    sys.path.insert(0, str(ROOT / "tools" / "kitti_training_pipeline"))
    common = importlib.import_module("common")
    config = {
        "data": {
            "kitti": {
                "geometry": {
                    "x_min": 0.0, "x_max": 4.0, "x_res": 1.0,
                    "y_min": -2.0, "y_max": 2.0, "y_res": 1.0,
                    "z_min": -1.0, "z_max": 2.0, "z_res": 1.0,
                }
            }
        }
    }
    assert common.input_shape(config) == (1, 3, 4, 4)
    config["data"]["bev_encoding"] = {"name": "rich8"}
    assert common.input_shape(config) == (1, 8, 4, 4)


def load_torch_modules():
    import torch

    sys.path.insert(0, str(DATASETS))
    sys.path.insert(0, str(ROOT / "detector"))
    from core.models.backbones.mobilepixor_coordinate_attention import MobilePixorBackBone
    from core.models.heads.cnn import Header
    return torch, MobilePixorBackBone, Header


def check_gates():
    torch, Backbone, _ = load_torch_modules()
    torch.manual_seed(7)
    summed = Backbone(input_channels=8, scale_gated_fpn=False).eval()
    gated = Backbone(input_channels=8, scale_gated_fpn=True).eval()
    missing, unexpected = gated.load_state_dict(summed.state_dict(), strict=False)
    assert not unexpected
    assert set(missing) == {
        "gate_c4.weight", "gate_c4.bias", "gate_c3.weight", "gate_c3.bias"
    }
    extra = sum(p.numel() for p in gated.parameters()) - sum(p.numel() for p in summed.parameters())
    assert extra == 480
    sample = torch.randn(1, 8, 32, 32)
    with torch.no_grad():
        expected = summed(sample)
        actual = gated(sample)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)


def check_head():
    torch, _, Header = load_torch_modules()
    sample = torch.randn(1, 16, 8, 8)
    legacy = Header(3, 16)
    center3d = Header(3, 16, box_encoding="center3d")
    legacy_output = legacy(sample)
    output_3d = center3d(sample)
    assert {key: value.shape[1] for key, value in legacy_output.items()} == {
        "cls": 3, "offset": 2, "size": 2, "yaw": 2
    }
    assert {key: value.shape[1] for key, value in output_3d.items()} == {
        "cls": 3, "offset": 3, "size": 3, "yaw": 2
    }


def check_targets():
    torch, _, _ = load_torch_modules()
    from core.datasets.dataset import Dataset

    dataset = Dataset.__new__(Dataset)
    dataset.output_shape = [4, 4]
    dataset.cls_encoding = "gaussian"
    dataset.num_classes = 3
    dataset.out_size_factor = 1
    dataset.box_encoding = "center3d"
    dataset.config = {"gaussian_overlap": 0.1, "min_radius": 4}
    geometry = {
        "x_min": 0.0, "x_max": 4.0, "x_res": 1.0,
        "y_min": 0.0, "y_max": 4.0, "y_res": 1.0,
        "z_min": -2.0, "z_max": 2.0, "z_res": 1.0,
    }
    boxes = torch.tensor([[1.0, 2.0, 1.0, 1.5, 1.25, 1.25, -1.0, 0.0]])
    target = dataset.get_label(boxes, geometry)
    mask = target["reg_mask"].bool()
    assert tuple(target["offset"].shape) == (3, 4, 4)
    assert tuple(target["size"].shape) == (3, 4, 4)
    torch.testing.assert_close(target["offset"][2][mask], torch.zeros(mask.sum()))
    torch.testing.assert_close(
        target["size"][2][mask], torch.full((mask.sum(),), np.log(2.0)), atol=1e-6, rtol=0
    )
    from postprocess import filter_pred
    prediction = {
        "cls": torch.full((1, 3, 4, 4), -10.0),
        "offset": target["offset"].unsqueeze(0),
        "size": target["size"].unsqueeze(0),
        "yaw": target["yaw"].unsqueeze(0),
    }
    y_index, x_index = mask.nonzero()[0]
    prediction["cls"][0, 1, y_index, x_index] = 10
    decoded = filter_pred(
        prediction, {"geometry": geometry}, out_size_factor=1, thres=0.5
    )
    np.testing.assert_allclose(
        decoded[0, [0, 2, 3, 4, 5, 6, 7, 8]],
        [1, 1.25, 1.25, 0.0, 1.0, 1.5, 2.0, 0.0],
        rtol=0,
        atol=1e-6,
    )


def check_loss():
    torch, _, _ = load_torch_modules()
    from core.losses.loss_fn import LossFunction
    from core.losses.focal_loss import modified_focal_loss

    criterion = LossFunction("gaussian", {"name": "uwag", "geometric_weight": 0.2})
    target = {
        "cls": torch.zeros(1, 3, 2, 2),
        "offset": torch.zeros(1, 3, 2, 2),
        "size": torch.zeros(1, 3, 2, 2),
        "yaw": torch.zeros(1, 2, 2, 2),
        "reg_mask": torch.ones(1, 2, 2),
    }
    target["yaw"][:, 0] = 1
    pred = {key: value.clone() for key, value in target.items() if key != "reg_mask"}
    baseline_geo = criterion(pred, target)["geo"]
    pred["offset"][:, 2] = 100
    pred["size"][:, 2] = -100
    result = criterion(pred, target)
    assert np.isfinite(result["loss"].item())
    assert result["geo"] == baseline_geo

    pred["offset"][0, 2, 0, 0] = float("nan")
    try:
        criterion(pred, target)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("non-finite regression output was accepted")

    binary_target = {
        **{key: value.clone() for key, value in target.items() if key != "cls"},
        "cls": torch.zeros(1, 2, 2, dtype=torch.long),
    }
    binary_pred = {
        **{key: value.clone() for key, value in pred.items() if key != "cls"},
        "cls": torch.zeros(1, 4, 2, 2),
    }
    binary_pred["offset"].nan_to_num_(0)
    assert torch.isfinite(LossFunction("binary")(binary_pred, binary_target)["loss"])

    saturated = torch.full((1, 3, 2, 2), 100.0, dtype=torch.bfloat16, requires_grad=True)
    saturated_loss = modified_focal_loss(saturated, torch.zeros_like(saturated))
    assert torch.isfinite(saturated_loss)
    saturated_loss.backward()
    assert torch.isfinite(saturated.grad).all()


def check_decode():
    torch, _, _ = load_torch_modules()
    from postprocess import filter_pred

    pred = {
        "cls": torch.full((1, 3, 2, 2), -10.0),
        "offset": torch.zeros(1, 3, 2, 2),
        "size": torch.zeros(1, 3, 2, 2),
        "yaw": torch.zeros(1, 2, 2, 2),
    }
    pred["cls"][0, 0, 0, 0] = 10
    pred["offset"][0, :, 0, 0] = torch.tensor([0.25, 0.50, -0.75])
    pred["size"][0, :, 0, 0] = torch.log(torch.tensor([1.5, 3.0, 2.0]))
    pred["yaw"][0, 0] = 1
    geometry = {
        "x_min": 0.0, "x_max": 2.0, "x_res": 1.0,
        "y_min": 0.0, "y_max": 2.0, "y_res": 1.0,
    }
    boxes = filter_pred(pred, {"geometry": geometry}, out_size_factor=1, thres=0.5)
    assert boxes.shape == (1, 9)
    np.testing.assert_allclose(
        boxes[0, [0, 2, 3, 4, 5, 6, 7, 8]],
        [0, 0.25, 0.50, -0.75, 1.5, 3.0, 2.0, 0.0],
        rtol=0,
        atol=1e-6,
    )
    nms_boxes = filter_pred(
        pred, {"geometry": geometry}, out_size_factor=1, thres=0.5, nms_thres=0.1
    )
    np.testing.assert_allclose(nms_boxes, boxes, rtol=0, atol=1e-6)
    invalid = {name: value.clone() for name, value in pred.items()}
    invalid["size"][0, 2, 0, 0] = float("inf")
    try:
        filter_pred(invalid, {"geometry": geometry}, out_size_factor=1, thres=0.5)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("non-finite detector output was accepted")


def check_metrics():
    sys.path.insert(0, str(ROOT / "tools" / "kitti_training_pipeline"))
    evaluator = importlib.import_module("evaluate_kitti_bev")
    first = evaluator.GroundTruth(
        name="Car", truncation=0.0, occlusion=0, bbox_height=50.0,
        x=10.0, y=0.0, z_center=0.0, length=2.0, width=2.0, height=2.0, yaw=0.0,
    )
    identical = evaluator.GroundTruth(**first.__dict__)
    disjoint = evaluator.GroundTruth(**{**first.__dict__, "z_center": 3.0})
    assert evaluator.iou3d(first, identical) == 1.0
    assert evaluator.iou3d(first, disjoint) == 0.0

    predictions = {
        "000001": np.array([[0, 0.99, 10, 0, 0, 2, 2, 2, 0]], dtype=np.float32)
    }
    labels = {"000001": [first]}
    accuracy = evaluator.evaluate_accuracy(predictions, labels, space="3d")
    assert accuracy["map_moderate_percent"] == 100.0
    legacy_predictions = {
        "000001": np.array([[0, 0.99, 10, 0, 2, 2, 0]], dtype=np.float32)
    }
    assert evaluator.evaluate_accuracy(
        legacy_predictions, labels, space="bev"
    )["map_moderate_percent"] == 100.0
    comparison = importlib.import_module("compare_models")
    flat = comparison.flatten({
        "accuracy": {"map_moderate_percent": 12.0, "mean_ap_9_percent": 13.0}
    })
    assert flat["map_3d_moderate_percent"] is None
    assert flat["map_bev_moderate_percent"] == 12.0

    gt = evaluator.GroundTruth(
        "Car", 0.0, 0, 50.0, 10.0, 0.0, 0.0, 4.0, 1.8, 1.6, 0.0
    )
    hit = np.array(
        [[0, 0.9, 10.0, 0.0, 0.0, 1.8, 4.0, 1.6, 0.0]], dtype=np.float32
    )
    false_ped = np.array(
        [[1, 0.8, 20.0, 0.0, 0.0, 0.8, 0.8, 1.7, 0.0]], dtype=np.float32
    )
    assert evaluator.frame_quality(hit, [gt]) == 1.0
    np.testing.assert_allclose(
        evaluator.frame_quality(np.concatenate([hit, false_ped]), [gt]), 2.0 / 3.0
    )

    args = evaluator.parser().parse_args([
        "--name", "sampled", "--backend", "pytorch", "--model", "model.pt",
        "--config", "config.json", "--detector-root", "detector",
        "--kitti-root", "kitti", "--split", "split.txt", "--output", "out.json",
        "--sampling-rate", "0.5", "--sampling-seed", "7",
    ])
    assert args.sampling_rate == 0.5 and args.sampling_seed == 7

    geometry = {
        "x_min": 0.0, "x_max": 2.0, "x_res": 1.0,
        "y_min": 0.0, "y_max": 2.0, "y_res": 1.0,
        "z_min": 0.0, "z_max": 3.0, "z_res": 1.0,
    }
    with tempfile.TemporaryDirectory() as temporary:
        evaluator.configure_detector_imports(ROOT / "detector")
        root = Path(temporary)
        points = np.arange(80, dtype=np.float32).reshape(20, 4)
        points.tofile(root / "000123.bin")
        voxel, timings, counts = evaluator.load_bev_frame(
            "000123", root, geometry,
            {"name": "rich8", "density_norm": 32, "intensity_scale": 1}, 0.5, 42,
        )
        assert tuple(voxel.shape) == (8, 2, 2)
        assert counts == {"raw_points": 20, "selected_points": 10}
        assert set(timings) == {"point_load", "sampling", "bev_encode", "preprocess"}


def check_selector():
    sys.path.insert(0, str(ROOT / "tools" / "kitti_training_pipeline"))
    selector = importlib.import_module("select_checkpoint")
    records = [
        {"epoch": 10, "map_3d_moderate_percent": 50.0, "mean_3d_ap_9_percent": 60.0},
        {"epoch": 15, "map_3d_moderate_percent": 51.0, "mean_3d_ap_9_percent": 59.0},
        {"epoch": 20, "map_3d_moderate_percent": 51.0, "mean_3d_ap_9_percent": 61.0},
        {"epoch": 25, "map_3d_moderate_percent": 51.0, "mean_3d_ap_9_percent": 61.0},
    ]
    assert selector.select_best(records)["epoch"] == 20


def check_training_guard():
    torch, _, _ = load_torch_modules()
    sys.path.insert(0, str(ROOT / "tools" / "kitti_training_pipeline"))
    training = importlib.import_module("train")
    args = training.build_parser().parse_args([
        "--config", "config.json", "--detector-root", "detector",
        "--output-root", "artifacts", "--seed", "43",
    ])
    assert args.seed == 43


CHECKS = {
    "legacy": check_legacy,
    "encoder": check_encoder,
    "sampling": check_sampling,
    "shapes": check_shapes,
    "gates": check_gates,
    "head": check_head,
    "targets": check_targets,
    "loss": check_loss,
    "decode": check_decode,
    "metrics": check_metrics,
    "selector": check_selector,
    "training_guard": check_training_guard,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-only", action="store_true")
    parser.add_argument("--encoder", action="store_true")
    parser.add_argument("--sampling", action="store_true")
    parser.add_argument("--shapes", action="store_true")
    parser.add_argument("--gates", action="store_true")
    parser.add_argument("--head", action="store_true")
    parser.add_argument("--targets", action="store_true")
    parser.add_argument("--loss", action="store_true")
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--selector", action="store_true")
    parser.add_argument("--training-guard", action="store_true")
    args = parser.parse_args()
    selected = (
        ["legacy"] if args.legacy_only else
        ["encoder"] if args.encoder else
        ["sampling"] if args.sampling else
        ["shapes"] if args.shapes else
        ["gates"] if args.gates else
        ["head"] if args.head else
        ["targets"] if args.targets else
        ["loss"] if args.loss else
        ["decode"] if args.decode else
        ["metrics"] if args.metrics else
        ["selector"] if args.selector else
        ["training_guard"] if args.training_guard else
        list(CHECKS)
    )
    for name in selected:
        CHECKS[name]()
        print(f"PASS {name}")


if __name__ == "__main__":
    main()
