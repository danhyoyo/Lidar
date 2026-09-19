#!/usr/bin/env python3
"""Focused checks for LiDAR ray visibility and BEV feature propagation."""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "detector"))
sys.path.insert(0, str(ROOT / "detector" / "core" / "datasets"))
sys.path.insert(0, str(ROOT / "tools" / "kitti_training_pipeline"))

from core.datasets.dataset import Dataset
from core.models.gaussian_visibility import GaussianPillarPropagation
from utils_1.preprocess import encode_bev
from utils_1.visibility import free_space_maps
from common import build_model, input_shape, read_json
from core.losses.loss_fn import LossFunction
from core.losses.focal_loss import modified_focal_loss
from evaluate_kitti_bev import (
    GroundTruth, center_is_observed_free, evaluate_one, prediction_box,
)


def small_geometry():
    return {
        "x_min": 0.0, "x_max": 2.0, "x_res": 0.1,
        "y_min": -0.2, "y_max": 0.2, "y_res": 0.1,
        "z_min": -1.0, "z_max": 1.0, "z_res": 1.0,
    }


def test_ray_visibility():
    geometry = small_geometry()
    point = np.asarray([[1.05, 0.0, 0.0, 0.5]], dtype=np.float32)
    visibility = {
        "mode": "global", "height_ranges": [[-1.0, 1.0]],
        "ray_length_m": 0.5, "step_m": 0.05, "range_margin_m": 0.1,
    }
    bev = encode_bev(point, geometry, {"name": "rich8", "visibility": visibility})
    assert bev.shape == (4, 20, 9)
    assert bev[2, 8, 8] == 1.0  # ray passes before the return
    assert bev[2, 10, 8] == 0.0  # measured cell remains occupied
    assert np.all(bev[0, :, 8] == 0.0)  # no ray passes this row
    assert np.all(bev[2, 11:, 8] == 0.0)  # behind the return is unknown

    bands = [[-1.0, -0.2], [-0.2, 0.2], [0.2, 1.0]]
    height_bev = encode_bev(point, geometry, {
        "name": "rich8", "visibility": dict(visibility, mode="height",
                                            height_ranges=bands),
    })
    assert height_bev.shape == (4, 20, 11)
    assert np.all(height_bev[..., 8] == 0.0)
    assert height_bev[2, 8, 9] == 1.0
    assert np.all(height_bev[..., 10] == 0.0)

    elevated = point.copy()
    elevated[0, 2] = 0.8
    slope_bands = [[-1.0, 0.25], [0.25, 0.6], [0.6, 1.0]]
    sloped = free_space_maps(elevated, geometry, height_ranges=slope_bands,
                             ray_length_m=0.7, step_m=0.025,
                             range_margin_m=0.1)
    assert sloped[..., 1].sum() > 0
    assert sloped[..., 2].sum() > 0  # height changes along one physical ray


def test_sensor_origin():
    geometry = small_geometry()
    point = np.asarray([[1.05, 0.0, 0.0, 0.5]], dtype=np.float32)
    args = dict(height_ranges=[[-1.0, 1.0]], ray_length_m=0.5,
                step_m=0.05, range_margin_m=0.1)
    base = free_space_maps(point, geometry, **args)
    shifted = point.copy()
    shifted[0, 0] += 0.2
    translated = free_space_maps(shifted, geometry, sensor_origin=(0.2, 0, 0),
                                 **args)
    np.testing.assert_array_equal(translated[:, 2:, 0], base[:, :-2, 0])

    # Dataset must pass the transformed origin but exclude its marker point.
    dataset = Dataset.__new__(Dataset)
    dataset.data_list = ["000000"]
    dataset.data_type_list = ["kitti"]
    dataset.config = {"kitti": {"location": "unused", "geometry": geometry}}
    dataset.task = "train"
    dataset.read_points = lambda path: point.copy()
    dataset.get_boxes = lambda idx: np.asarray(
        [[0, 1, 1, 1, 1, 0, 0, 0]], dtype=np.float32)
    delta = np.asarray([0.2, 0.1, 0.3, 0.0], dtype=np.float32)
    dataset.augment = lambda points, boxes: (points + delta, boxes)
    dataset.filter_boxes = lambda boxes, data_type: boxes
    captured = {}
    def record_voxelize(points, geom, origin):
        captured["points"] = points.copy()
        captured["origin"] = origin.copy()
        return np.zeros((4, 20, 8), dtype=np.float32)
    dataset.voxelize = record_voxelize
    dataset.get_label = lambda boxes, geom: {}
    dataset[0]
    assert captured["points"].shape[0] == 1
    np.testing.assert_allclose(captured["points"][0, :3], point[0, :3] + delta[:3])
    np.testing.assert_allclose(captured["origin"], delta[:3])


def test_gaussian_gating_and_control():
    base = torch.zeros(1, 8, 9, 9)
    base[0, 1, 4, 4] = 1.0
    base[0, 2, 4, 4] = 1.0
    base[0, 5, 4, 4] = 0.5
    plain = GaussianPillarPropagation("plain")(base)
    assert plain[0, 1, 4, 5] > 0.0
    torch.testing.assert_close(plain[0, :, 4, 4], base[0, :, 4, 4])
    global_input = torch.cat((base, torch.zeros(1, 1, 9, 9)), dim=1)
    global_input[0, 8, 4, 5] = 1.0
    gated = GaussianPillarPropagation("global")(global_input)
    assert torch.all(gated[0, :, 4, 5] == 0.0)
    assert gated[0, 1, 4, 3] > 0.0
    height_input = torch.cat((base, torch.zeros(1, 3, 9, 9)), dim=1)
    height_input[0, 9, 4, 5] = 1.0
    height_gated = GaussianPillarPropagation("height")(height_input)
    assert height_gated[0, 1, 4, 5] == 0.0
    assert height_gated[0, 2, 4, 5] > 0.0
    assert height_gated[0, 5, 4, 5] > 0.0

    uniform = GaussianPillarPropagation("plain", radius_cells=2,
                                        kernel_type="uniform")
    assert uniform(base)[0, 1, 4, 5] > 0
    axis = torch.arange(-2, 3, dtype=torch.float32)
    uniform_variance = float((uniform.kernel_x[0, 0, 0] * axis.square()).sum())
    gaussian = GaussianPillarPropagation("plain")
    g_axis = torch.arange(-4, 5, dtype=torch.float32)
    gaussian_variance = float((gaussian.kernel_x[0, 0, 0] * g_axis.square()).sum())
    assert abs(uniform_variance - gaussian_variance) < 0.3


def test_variant_shapes_and_height_alignment():
    directory = ROOT / "configs" / "kitti" / "gaussian_visibility"
    expected = {"a0": 8, "a1": 8, "a2": 9, "a3": 11, "a4": 8}
    for path in sorted(directory.glob("a*.json")):
        cfg = read_json(path)
        channels = expected[path.name[:2]]
        assert input_shape(cfg)[1] == channels
        assert cfg["model"]["backbone"] == "mobilepixor"
        assert "scale_gated_fpn" not in cfg["model"]
        assert cfg["loss"] == {"name": "baseline"}
        assert cfg["train"]["epochs"] == 50
        assert max(cfg["train"]["lr_decay_at"]) < 50
        assert not list(LossFunction("gaussian", cfg["loss"]).parameters())
        model = build_model(cfg).eval()
        torch.testing.assert_close(model.header.cls.head.bias.detach(),
                                   torch.full_like(model.header.cls.head.bias, -2.19))
        assert not any("CoordAtt" in type(layer).__name__
                       for layer in model.modules())
        with torch.no_grad():
            pred = model(torch.zeros(1, channels, 32, 32))
        assert pred["cls"].shape == (1, 3, 8, 8)
        assert pred["offset"].shape == (1, 3, 8, 8)
        if path.name.startswith("a3"):
            geom = cfg["data"]["kitti"]["geometry"]
            bounds = np.linspace(geom["z_min"], geom["z_max"], 4)
            actual = cfg["data"]["bev_encoding"]["visibility"]["height_ranges"]
            np.testing.assert_allclose(actual, np.column_stack((bounds[:-1], bounds[1:])))
        if path.name.startswith("a4"):
            assert cfg["model"]["gaussian_propagation"]["kernel_type"] == "uniform"


def test_bf16_heatmap_stability():
    logits = torch.tensor([[[[80.0, -80.0]]]], dtype=torch.bfloat16,
                          requires_grad=True)
    target = torch.tensor([[[[1.0, 0.0]]]])
    loss = modified_focal_loss(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_observed_free_false_positives():
    geometry = {
        "x_min": 0.0, "x_max": 8.0, "x_res": 1.0,
        "y_min": -2.0, "y_max": 2.0, "y_res": 1.0,
        "z_min": -1.0, "z_max": 1.0, "z_res": 1.0,
    }
    free = np.zeros((4, 8, 3), dtype=np.float32)
    free[2, 5, 1] = 1.0
    boxes = np.asarray([
        [0, 0.9, 1, 0, 0, 2, 4, 2, 0],
        [0, 0.8, 5, 0, 0, 2, 4, 2, 0],
    ], dtype=np.float32)
    flags = np.asarray([
        center_is_observed_free(prediction_box(row, "3d"), free, geometry)
        for row in boxes
    ], dtype=np.bool_)
    np.testing.assert_array_equal(flags, [False, True])
    frame_id = "000001"
    labels = {frame_id: [GroundTruth(
        "Car", 0.0, 0, 50.0, 1.0, 0.0, 0.0, 4.0, 2.0, 2.0, 0.0
    )]}
    result = evaluate_one({frame_id: boxes}, labels, "Car", "Moderate",
                          space="3d", free_space={frame_id: flags})
    assert result["true_positives"] == 1
    assert result["false_positives"] == 1
    assert result["observed_free_false_positives"] == 1
    assert result["observed_free_false_positives_per_frame"] == 1.0


if __name__ == "__main__":
    test_ray_visibility()
    test_sensor_origin()
    test_gaussian_gating_and_control()
    test_variant_shapes_and_height_alignment()
    test_observed_free_false_positives()
    test_bf16_heatmap_stability()
    print("gaussian_visibility checks passed")
