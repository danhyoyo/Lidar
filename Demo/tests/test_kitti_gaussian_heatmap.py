"""Heatmap geometry and channels must match the actual training/inference pipeline."""
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2] / "Code"
sys.path.insert(0, str(ROOT))


def api():
    return importlib.import_module("tools.visualization.export_kitti_gaussian_heatmap")


def config():
    return {"model": {"cls_encoding": "gaussian"}, "data": {
        "num_classes": 3, "out_size_factor": 4, "gaussian_overlap": .1, "min_radius": 2,
        "bev_encoding": {"name": "hist14", "backend": "numpy"},
        "kitti": {"objects": {"Car": 0, "Pedestrian": 1, "Cyclist": 2},
                  "geometry": {"x_min": 0., "x_max": 16., "x_res": .1,
                               "y_min": -12., "y_max": 12., "y_res": .1,
                               "z_min": -2.5, "z_max": 1., "z_res": .1}}}}


def test_targets_match_dataset_including_xy_transpose_and_boundary_clipping():
    module = api()
    cfg = config()
    boxes = np.array([[0, 1.5, 2, 4, 4.1, 2.1, -1.7, 0],
                      [2, 1.6, .6, 1.8, .1, -11.9, -1.7, .7],
                      [1, 1.7, .6, .6, 10.1, -4.3, -1.7, 0]], dtype=np.float32)
    heatmaps, accepted = module.build_gaussian_targets(boxes, cfg)
    from core.datasets.dataset import Dataset
    dataset = Dataset.__new__(Dataset)
    dataset.config = cfg["data"]
    dataset.cls_encoding = "gaussian"
    dataset.out_size_factor = 4
    dataset.output_shape = [40, 60]
    dataset.num_classes = 3
    reference = dataset.get_label(torch.from_numpy(boxes), cfg["data"]["kitti"]["geometry"])
    np.testing.assert_array_equal(heatmaps, reference["cls"].numpy())
    np.testing.assert_array_equal(accepted, boxes)
    assert heatmaps.shape == (3, 60, 40)
    assert heatmaps[0, 35, 10] == 1
    assert heatmaps[2, 0, 0] == 1


def test_overlapping_gaussians_use_max_and_strict_training_roi():
    boxes = np.array([[0, 1.5, 2, 4, 4.1, 2.1, -1.7, 0],
                      [0, 1.5, 2, 4, 4.1, 2.1, -1.7, 0],
                      [1, 1.5, .6, .6, 0, 1, -1.7, 0],
                      [2, 1.5, .6, .6, 16, 1, -1.7, 0]], dtype=np.float32)
    maps, accepted = api().build_gaussian_targets(boxes, config())
    single, _ = api().build_gaussian_targets(boxes[:1], config())
    np.testing.assert_array_equal(maps, single)
    assert len(accepted) == 2
    assert maps.max() == 1


def test_grouped_model_logits_are_gathered_by_global_id_before_sigmoid():
    api().configure_imports()
    from core.task_groups import resolve_task_groups
    from tools.visualization.predict_kitti_overview import classification_heatmaps
    groups = resolve_task_groups([
        {"name": "ped_cyc", "classes": ["Cyclist", "Pedestrian"]},
        {"name": "car", "classes": ["Car"]}], config()["data"]["kitti"]["objects"])
    prediction = {"groups": {
        "car": {"cls": torch.full((1, 1, 3, 5), -2.)},
        "ped_cyc": {"cls": torch.stack([torch.full((3, 5), 2.), torch.zeros(3, 5)])[None]}}}
    logits, probabilities = classification_heatmaps(prediction, groups=groups, num_classes=3)
    assert logits.shape == (3, 3, 5)
    np.testing.assert_array_equal(logits[:, 0, 0], [-2, 0, 2])
    np.testing.assert_allclose(probabilities[:, 0, 0], torch.sigmoid(torch.tensor([-2., 0., 2.])))


def test_crop_keeps_y_rows_x_columns_and_snaps_to_real_cell_edges():
    cfg = config()
    maps = np.arange(3 * 60 * 40).reshape(3, 60, 40)
    cropped, extent, indices = api().crop_heatmaps(maps, cfg, [1.1, 4.1, -3.1, 2.1])
    np.testing.assert_array_equal(cropped, maps[:, 22:36, 2:11])
    np.testing.assert_allclose(extent, [.8, 4.4, -3.2, 2.4])
    assert indices == {"x": [2, 11], "y": [22, 36]}


def test_crop_outside_grid_is_rejected():
    with pytest.raises(ValueError, match="intersect"):
        api().crop_heatmaps(np.zeros((3, 60, 40)), config(), [-10, -5, -2, 2])


def test_non_gaussian_checkpoint_is_not_misrepresented():
    cfg = config()
    cfg["model"]["cls_encoding"] = "binary"
    with pytest.raises(ValueError, match="gaussian"):
        api().build_gaussian_targets(np.empty((0, 8)), cfg)


def test_nonfinite_classification_head_is_rejected():
    from tools.visualization.predict_kitti_overview import classification_heatmaps
    with pytest.raises(ValueError, match="finite"):
        classification_heatmaps({"cls": torch.full((1, 3, 2, 2), float("nan"))},
                                groups=(), num_classes=3)
