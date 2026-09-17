#!/usr/bin/env python3
"""Small dependency-free checks for MobileBEV-Lite."""

import argparse
import importlib
import json
import sys
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


def check_shapes():
    sys.path.insert(0, str(ROOT / "tools" / "kitti_training_pipeline"))
    common = importlib.import_module("common")
    trt_build = importlib.import_module("build_tensorrt")
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


def check_probgeo_config():
    sys.path.insert(0, str(ROOT / "tools" / "kitti_training_pipeline"))
    common = importlib.import_module("common")
    valid = {
        "model": {"box_encoding": "center3d", "predict_log_variance": True,
                  "initial_log_variance": -2.0},
        "loss": {"name": "probgeo_uq", "log_var_min": -7.0,
                 "log_var_max": 4.0, "gwd_weight": 0.2},
    }
    common.validate_probgeo_config(valid)
    invalid = {**valid, "model": {**valid["model"], "predict_log_variance": False}}
    try:
        common.validate_probgeo_config(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("UQ loss without variance output was accepted")
    common.validate_probgeo_config({"model": {}, "loss": {}})
    for replacement in ({"epsilon": 0.0}, {"gwd_weight": float("nan")}, {"log_var_min": float("nan")}):
        broken = {"model": {**valid["model"]}, "loss": {**valid["loss"], **replacement}}
        try:
            common.validate_probgeo_config(broken)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid config accepted: {replacement}")


def check_export_contract():
    torch, _, Header = load_torch_modules()
    sys.path.insert(0, str(ROOT / "tools" / "kitti_training_pipeline"))
    exporter = importlib.import_module("export_onnx")
    class Model(torch.nn.Module):
        def __init__(self, header):
            super().__init__()
            self.header = header
        def forward(self, x):
            return self.header(x)
    legacy = exporter.RawHeadWrapper(Model(Header(3, 16, box_encoding="center3d")))
    uq = exporter.RawHeadWrapper(Model(Header(3, 16, box_encoding="center3d", predict_log_variance=True)))
    assert legacy.output_names == ("cls", "offset", "size", "yaw")
    assert uq.output_names == ("cls", "offset", "size", "yaw", "log_var")
    assert len(uq(torch.zeros(1, 16, 2, 2))) == 5


def check_uncertainty_evaluator():
    sys.path.insert(0, str(ROOT / "tools" / "kitti_training_pipeline"))
    uq = importlib.import_module("evaluate_uncertainty")
    errors = np.ones((40, 6), dtype=np.float64)
    log_vars = np.zeros_like(errors)
    metrics = uq.evaluate_arrays(errors, log_vars)
    np.testing.assert_allclose(metrics["nll_per_dim"], 0.5)
    np.testing.assert_allclose(metrics["coverage_1sigma_per_dim"], 1.0)
    alpha = uq.fit_variance_scale(errors, log_vars)
    np.testing.assert_allclose(alpha, 1.0)
    calibrated = uq.evaluate_arrays(errors, log_vars + np.log(alpha))
    assert calibrated["nll"] == metrics["nll"]
    assert uq.binary_auroc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0


def check_probgeo_review():
    torch, _, _ = load_torch_modules()
    sys.path.insert(0, str(ROOT / "tools" / "kitti_training_pipeline"))
    uq = importlib.import_module("evaluate_uncertainty")
    evaluator = importlib.import_module("evaluate_kitti_bev")
    common = importlib.import_module("common")
    trt_build = importlib.import_module("build_tensorrt")
    from core.losses.loss_fn import heteroscedastic_nll
    errors = np.ones((4, 6))
    ranges = np.arange(4.0)
    baselines = uq.fit_baselines(errors, ranges, np.arange(4.0))
    assert baselines["range_point_count_coefficients"].shape == (3, 6)
    box = evaluator.GroundTruth("Car", 0, 0, 50, 1, 2, 0, 4, 2, 2, 0)
    points = np.array([[1, 2, 0, 1], [2.9, 2, 0, 1], [4, 2, 0, 1], [1, 2, 2, 1]], dtype=np.float32)
    assert uq.count_points_in_box(points, box) == 2
    try:
        heteroscedastic_nll(torch.zeros(1, 6, 1, 1), torch.full((1, 6, 1, 1), float("inf")), torch.ones(1, 1, 1), -7, 4)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("infinite log variance was accepted")
    extreme = uq.evaluate_arrays(np.ones((2, 6)), np.full((2, 6), 1e6))
    assert np.isfinite(extreme["nll"])
    try:
        evaluator.prediction_box(np.zeros(10), "3d")
    except ValueError:
        pass
    else:
        raise AssertionError("invalid prediction schema was accepted")
    bad = {"model": {"box_encoding": "center3d", "predict_log_variance": True, "initial_log_variance": -2}, "loss": {"name": "uwag", "epsilon": 1e-4}}
    try:
        common.validate_probgeo_config(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("untrained UQ head was accepted")
    for name in ("gwd", "heteroscedastic", "probgeo_uq"):
        broken = {"model": {"box_encoding": "bev", "predict_log_variance": name != "gwd", "initial_log_variance": -2}, "loss": {"name": name, "epsilon": 1e-4}}
        try:
            common.validate_probgeo_config(broken)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{name} accepted BEV regression")
    extreme_rows = {"f": np.array([[0, 0.9, 1, 2, 0, 2, 4, 2, 0, 1e6, 1e6, 1e6, 1e6, 1e6, 1e6]])}
    labels = {"f": [box]}
    with np.errstate(over="raise"):
        assert np.isfinite(uq.matched_residuals(extreme_rows, labels, log_var_min=-7, log_var_max=4)[3]).all()
    from postprocess import filter_pred
    empty = {"cls": torch.full((1, 3, 2, 2), -10.0), "offset": torch.zeros(1, 3, 2, 2), "size": torch.zeros(1, 3, 2, 2), "yaw": torch.zeros(1, 2, 2, 2), "log_var": torch.zeros(1, 6, 2, 2)}
    assert filter_pred(empty, {"geometry": {"x_min": 0, "x_max": 2, "x_res": 1, "y_min": 0, "y_max": 2, "y_res": 1}}, 1, 0.5).shape == (0, 15)
    assert filter_pred(empty, {"geometry": {"x_min": 0, "x_max": 2, "x_res": 1, "y_min": 0, "y_max": 2, "y_res": 1}}, 1, 0.5, nms_thres=0.1).shape == (0, 15)
    assert trt_build.validate_output_metadata({"outputs": {"cls": [1, 3, 2, 2], "offset": [1, 3, 2, 2], "size": [1, 3, 2, 2], "yaw": [1, 2, 2, 2]}}) == {"cls": (1, 3, 2, 2), "offset": (1, 3, 2, 2), "size": (1, 3, 2, 2), "yaw": (1, 2, 2, 2)}
    base = json.loads((ROOT / "configs/kitti/mobilebev/a1_legacy35_center3d.json").read_text())
    for path in (ROOT / "configs/kitti/probgeo_uq").glob("*.json"):
        candidate = json.loads(path.read_text())
        for field in ("name", "gwd_weight", "log_var_min", "log_var_max"):
            candidate["loss"].pop(field, None)
        candidate["model"].pop("predict_log_variance", None)
        candidate["model"].pop("initial_log_variance", None)
        candidate.pop("note", None)
        expected = json.loads(json.dumps(base))
        expected["loss"].pop("name")
        expected.pop("note", None)
        expected["val"]["data"] = "splits/kitti/uq_calibration.txt"
        assert candidate == expected

    notebook = json.loads(
        (ROOT / "3D_Lidar_Object_Detection_Notebook_optimized.ipynb").read_text()
    )
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    assert 'BRANCH = "Proposal2-Loss-Function"' in code
    assert 'PRECISION = "bf16"' in code
    assert "EPOCHS = 100" in code
    assert "RESUME = False" in code
    assert "EPOCHS <= RESUME_EPOCH" in code
    assert "passed_tests != EXPECTED_TESTS" in code
    assert "python3 -m tqdm --bytes" in code
    assert "mbuffer" not in code
    for variant, filename in {
        "B0": "b0_deterministic.json",
        "B1": "b1_gwd.json",
        "B2": "b2_heteroscedastic.json",
        "B3": "b3_probgeo_uq.json",
    }.items():
        assert f'"{variant}": "configs/kitti/probgeo_uq/{filename}"' in code
    assert "--split splits/kitti/uq_calibration.txt" in code
    assert "--split splits/kitti/uq_test.txt" in code
    assert "--calibration-split splits/kitti/uq_calibration.txt" in code
    assert "evaluate_uncertainty.py" in code


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
    uq = Header(3, 16, box_encoding="center3d", predict_log_variance=True,
                initial_log_variance=-2.0)
    uq_output = uq(sample)
    assert {key: value.shape[1] for key, value in uq_output.items()} == {
        "cls": 3, "offset": 3, "size": 3, "yaw": 2, "log_var": 6
    }
    assert torch.equal(uq_output["offset"], uq.offset(sample)[:, :3])
    assert torch.equal(uq_output["size"], uq.size(sample)[:, :3])
    np.testing.assert_allclose(uq.offset.head.bias.detach().numpy()[3:], -2.0)
    np.testing.assert_allclose(uq.size.head.bias.detach().numpy()[3:], -2.0)
    assert sum(p.numel() for p in uq.parameters()) - sum(p.numel() for p in center3d.parameters()) == 102


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

    saturated = torch.full(
        (1, 3, 2, 2), 100.0, dtype=torch.bfloat16, requires_grad=True
    )
    saturated_loss = modified_focal_loss(saturated, torch.zeros_like(saturated))
    assert torch.isfinite(saturated_loss)
    saturated_loss.backward()
    assert torch.isfinite(saturated.grad).all()


def check_gwd():
    torch, _, _ = load_torch_modules()
    from core.losses.loss_fn import gwd_footprint_loss

    def maps(yaw=0.0, width=2.0, length=4.0):
        pred = {
            "offset": torch.zeros(1, 3, 1, 1, requires_grad=True),
            "size": torch.tensor([[[[np.log(width)]], [[np.log(length)]], [[0.0]]]], requires_grad=True),
            "yaw": torch.tensor([[[[np.cos(2 * yaw)]], [[np.sin(2 * yaw)]]]], requires_grad=True),
        }
        return pred

    target = {**maps(), "reg_mask": torch.ones(1, 1, 1)}
    same = maps()
    assert gwd_footprint_loss(same, target, 1e-4).item() == 0.0
    pi_equivalent = maps(np.pi)
    assert gwd_footprint_loss(pi_equivalent, target, 1e-4).item() < 1e-6
    swapped = maps(np.pi / 2, width=4.0, length=2.0)
    assert gwd_footprint_loss(swapped, target, 1e-4).item() < 1e-5
    square = maps(np.pi / 4, width=2.0, length=2.0)
    square_target = {**maps(0.0, width=2.0, length=2.0), "reg_mask": torch.ones(1, 1, 1)}
    assert gwd_footprint_loss(square, square_target, 1e-4).item() < 1e-6
    shifted = maps()
    with torch.no_grad():
        shifted["offset"][0, 0, 0, 0] = 10.0
    loss = gwd_footprint_loss(shifted, target, 1e-4)
    assert 0.0 < loss.item() < 1.0
    loss.backward()
    assert torch.isfinite(shifted["offset"].grad).all()
    assert torch.isfinite(shifted["size"].grad).all()
    boundary = maps(np.pi / 2 - 1e-6)
    gwd_footprint_loss(boundary, target, 1e-4).backward()
    assert torch.isfinite(boundary["yaw"].grad).all()


def check_uncertainty_loss():
    torch, _, _ = load_torch_modules()
    from core.losses.loss_fn import heteroscedastic_nll, LossFunction

    residual = torch.full((1, 6, 1, 1), 2.0)
    log_var = torch.zeros_like(residual)
    mask = torch.ones(1, 1, 1)
    assert heteroscedastic_nll(residual, log_var, mask, -7, 4).item() == 2.0
    np.testing.assert_allclose(heteroscedastic_nll(residual, torch.full_like(log_var, 99), mask, -7, 4).item(), 0.5 * (4 * np.exp(-4) + 4), rtol=1e-6)
    np.testing.assert_allclose(heteroscedastic_nll(residual, torch.full_like(log_var, -99), mask, -7, 4).item(), 0.5 * (4 * np.exp(7) - 7), rtol=1e-6)
    large = torch.full_like(residual, 100.0)
    attenuated = heteroscedastic_nll(large, torch.full_like(log_var, 4.0), mask, -7, 4)
    assert 2.0 < attenuated.item() < heteroscedastic_nll(large, log_var, mask, -7, 4).item()
    target = {
        "cls": torch.zeros(1, 3, 1, 1), "offset": torch.zeros(1, 3, 1, 1),
        "size": torch.zeros(1, 3, 1, 1), "yaw": torch.tensor([[[[1.]], [[0.]]]]),
        "reg_mask": mask,
    }
    pred = {key: value.clone() for key, value in target.items() if key != "reg_mask"}
    pred["log_var"] = torch.zeros(1, 6, 1, 1)
    result = LossFunction("gaussian", {"name": "heteroscedastic"})(pred, target)
    assert result["offset"] == 0.0 and result["size"] == 0.0
    assert result["log_var_saturation_min"] == 0.0 and result["log_var_saturation_max"] == 0.0
    pred["log_var"][0, 0, 0, 0] = float("nan")
    try:
        LossFunction("gaussian", {"name": "heteroscedastic"})(pred, target)
    except FloatingPointError:
        pass
    else:
        raise AssertionError("non-finite log variance was accepted")
    try:
        LossFunction("gaussian", {"name": "probgeo_uq", "log_var_min": 1, "log_var_max": 1})
    except ValueError:
        pass
    else:
        raise AssertionError("invalid log-variance bounds were accepted")


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
    uq_pred = {name: value.clone() for name, value in pred.items()}
    uq_pred["log_var"] = torch.zeros(1, 6, 2, 2)
    uq_pred["log_var"][0, :, 0, 0] = torch.arange(6)
    uq_boxes = filter_pred(uq_pred, {"geometry": geometry}, out_size_factor=1, thres=0.5)
    assert uq_boxes.shape == (1, 15)
    np.testing.assert_array_equal(uq_boxes[:, :9], boxes)
    np.testing.assert_array_equal(uq_boxes[0, 9:], np.arange(6))
    np.testing.assert_allclose(
        filter_pred(uq_pred, {"geometry": geometry}, out_size_factor=1, thres=0.5, nms_thres=0.1),
        uq_boxes, rtol=0, atol=1e-6,
    )
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
    uq_predictions = {
        "000001": np.array([[0, 0.99, 10, 0, 0, 2, 2, 2, 0, -2, -2, -2, -2, -2, -2]], dtype=np.float32)
    }
    assert evaluator.evaluate_accuracy(uq_predictions, labels, space="3d") == accuracy
    comparison = importlib.import_module("compare_models")
    flat = comparison.flatten({
        "accuracy": {"map_moderate_percent": 12.0, "mean_ap_9_percent": 13.0}
    })
    assert flat["map_3d_moderate_percent"] is None
    assert flat["map_bev_moderate_percent"] == 12.0
    uq_flat = comparison.flatten({"uncertainty": {"raw": {"nll": 1.5, "coverage_1sigma_gap_per_dim": [0.1] * 6, "aurc": 0.2}, "calibrated": {"nll": 1.0}}})
    assert uq_flat["uq_raw_nll"] == 1.5 and uq_flat["uq_calibrated_nll"] == 1.0


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
    layer = torch.nn.Linear(2, 1)
    layer(torch.ones(1, 2)).sum().backward()
    assert training.gradients_are_finite(layer.parameters())
    layer.weight.grad[0, 0] = float("nan")
    assert not training.gradients_are_finite(layer.parameters())
    args = training.build_parser().parse_args([
        "--config", "config.json", "--detector-root", "detector",
        "--output-root", "artifacts", "--seed", "43",
    ])
    assert args.seed == 43


CHECKS = {
    "legacy": check_legacy,
    "encoder": check_encoder,
    "shapes": check_shapes,
    "probgeo_config": check_probgeo_config,
    "export_contract": check_export_contract,
    "uncertainty_evaluator": check_uncertainty_evaluator,
    "probgeo_review": check_probgeo_review,
    "gates": check_gates,
    "head": check_head,
    "targets": check_targets,
    "loss": check_loss,
    "gwd": check_gwd,
    "uncertainty_loss": check_uncertainty_loss,
    "decode": check_decode,
    "metrics": check_metrics,
    "selector": check_selector,
    "training_guard": check_training_guard,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-only", action="store_true")
    parser.add_argument("--encoder", action="store_true")
    parser.add_argument("--shapes", action="store_true")
    parser.add_argument("--probgeo-config", action="store_true")
    parser.add_argument("--export-contract", action="store_true")
    parser.add_argument("--uncertainty-evaluator", action="store_true")
    parser.add_argument("--probgeo-review", action="store_true")
    parser.add_argument("--gates", action="store_true")
    parser.add_argument("--head", action="store_true")
    parser.add_argument("--targets", action="store_true")
    parser.add_argument("--loss", action="store_true")
    parser.add_argument("--gwd", action="store_true")
    parser.add_argument("--uncertainty-loss", action="store_true")
    parser.add_argument("--decode", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--selector", action="store_true")
    parser.add_argument("--training-guard", action="store_true")
    args = parser.parse_args()
    selected = (
        ["legacy"] if args.legacy_only else
        ["encoder"] if args.encoder else
        ["shapes"] if args.shapes else
        ["probgeo_config"] if args.probgeo_config else
        ["export_contract"] if args.export_contract else
        ["uncertainty_evaluator"] if args.uncertainty_evaluator else
        ["probgeo_review"] if args.probgeo_review else
        ["gates"] if args.gates else
        ["head"] if args.head else
        ["targets"] if args.targets else
        ["loss"] if args.loss else
        ["gwd"] if args.gwd else
        ["uncertainty_loss"] if args.uncertainty_loss else
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
