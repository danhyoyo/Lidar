"""The actual augmented scene is encoded as contiguous Hist22 model input."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools/kitti_training_pipeline"), str(ROOT / "detector"),
               str(ROOT / "detector/core/datasets")]
from core.datasets.dataset import Dataset
from core.bev_encoding import resolve_bev_encoding
from utils_1.hist22_backend import encode_hist22_chw
from test_hybrid_augmentation import database, recipe, CLASSES


@pytest.mark.parametrize("backend", ["numpy", "numba"])
@pytest.mark.parametrize("grouped,box_mode", [(False, "bev"), (True, "bev"), (True, "3d")])
def test_actual_sampling_removal_scaling_reencoded_and_val_test_unsampled(tmp_path, backend, grouped, box_mode):
    root, source, object_points = database(tmp_path)
    # Interior point must be removed by the sampler; outside point must survive.
    original = np.array([[15, 2, -.5, .1], [25, 0, -.6, .4]], np.float32)
    original.tofile(root / "pointcloud/000000.bin")
    (root / "label/000000.txt").write_text("")
    manifest = tmp_path / "frames.txt"
    manifest.write_text("000000;kitti\n000001;kitti\n")
    geometry = dict(x_min=0., x_max=32., x_res=.1, y_min=-8., y_max=8.,
                    y_res=.1, z_min=-2.5, z_max=1., z_res=.1)
    encoding = {"name": "hist22", "backend": backend}
    data = {"num_classes": 3, "out_size_factor": 4, "gaussian_overlap": .1,
            "min_radius": 1, "bev_encoding": encoding, "box_mode": box_mode,
            "kitti": {"location": str(root), "objects": CLASSES, "geometry": geometry}}
    if grouped:
        data["head_groups"] = [{"name": "car", "classes": ["Car"]},
                               {"name": "ped_cyc", "classes": ["Pedestrian", "Cyclist"]}]
    augmentation = {"mode": "openpcdet", "AUG_CONFIG_LIST": [recipe(PROBABILITY=1),
        {"NAME": "random_world_scaling", "WORLD_SCALE_RANGE": [1.25, 1.25]}]}
    dataset = Dataset(str(manifest), data, augmentation, "gaussian", "train")
    schema = resolve_bev_encoding(encoding, geometry)
    sample = dataset[0]
    transformed = np.concatenate([original[1:], object_points]).copy()
    transformed[:, :3] *= 1.25
    expected = encode_hist22_chw(transformed, schema)
    assert sample["voxel"].is_contiguous()
    assert sample["voxel"].dtype == torch.float32
    # Database placement reconstructs source coordinates from local box points;
    # its float32 roundoff is below 3e-8 in dimensionality on this fixture.
    np.testing.assert_allclose(sample["voxel"], expected, atol=1e-7, rtol=1e-6)
    target = sample["groups"]["car"] if grouped else sample
    assert target["reg_mask"].shape == (40, 80)
    assert target["cls"].any()
    if box_mode == "3d":
        assert "vertical" in target
    dataset.augment.queue[-1] = ("random_world_scaling", 1., np.array([1., 1.]))
    second = dataset[0]
    unscaled = np.concatenate([original[1:], object_points])
    np.testing.assert_allclose(second["voxel"], encode_hist22_chw(unscaled, schema), atol=1e-7, rtol=1e-6)
    assert not torch.equal(sample["voxel"], second["voxel"])
    # Validation/test never construct the train-only database, even if missing.
    invalid_aug = copy.deepcopy(augmentation)
    invalid_aug["AUG_CONFIG_LIST"][0]["DB_INFO_PATH"] = ["does_not_exist.json"]
    for task in ("validation", "val", "test"):
        validation = Dataset(str(manifest), data, invalid_aug, "gaussian", task)
        np.testing.assert_array_equal(validation[0]["voxel"], encode_hist22_chw(original, schema))


def test_target_shape_uses_schema_grid_at_float_division_ties(tmp_path):
    from test_grouped_targets import make_dataset
    old = make_dataset(tmp_path, grouped=False)
    data = copy.deepcopy(old.config)
    data["bev_encoding"] = {"name": "hist22"}
    data["kitti"]["geometry"].update(x_max=4.8, x_res=.1, y_min=-1.6, y_max=1.6)
    dataset = Dataset(old.data_file, data, {"mode": "openpcdet", "AUG_CONFIG_LIST": []},
                      "gaussian", "validation")
    sample = dataset[0]
    assert sample["voxel"].shape == (22, 32, 48)
    assert sample["reg_mask"].shape == (8, 12)
