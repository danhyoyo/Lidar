"""Analytic hist14 fixtures, indexing boundaries and input invariants."""

import importlib
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "detector"), str(ROOT / "detector/core/datasets")]

from core.bev_encoding import resolve_bev_encoding

GEOMETRY = {
    "x_min": 0, "x_max": 4, "x_res": 1,
    "y_min": -2, "y_max": 2, "y_res": 0.5,
    "z_min": -2.5, "z_max": 1, "z_res": 0.5,
}


def reference(points, geometry=GEOMETRY, **options):
    path = ROOT / "detector/core/datasets/utils_1/bev_backend.py"
    assert path.is_file(), "hist14 reference backend has not been implemented"
    module = importlib.import_module("core.datasets.utils_1.bev_backend")
    schema = resolve_bev_encoding({"name": "hist14", **options}, geometry)
    return module.encode_hist14_numpy(points, schema)


def log_density(count, norm=32):
    return min(1.0, math.log1p(count) / math.log1p(norm))


def test_empty_cloud_has_exact_zero_contiguous_float32_yxc_output():
    bev = reference(np.empty((0, 4)))
    assert bev.shape == (8, 4, 14)
    assert bev.dtype == np.float32
    assert bev.flags.c_contiguous
    assert not bev.any()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_one_point_analytic_channels_and_signed_centroid_offsets(dtype):
    point = np.array([[0.25, -1.625, -1.625, 1.4]], dtype=dtype)
    expected = [0, log_density(1), 0, 0, .25, .25, 0, 0, 1, 1,
                log_density(1), 1, -.25, .25]
    bev = reference(point)
    np.testing.assert_allclose(bev[0, 0], expected, atol=1e-7)
    assert np.count_nonzero(bev[..., 11]) == 1
    assert not bev[1:].any()


def test_multi_bin_counts_and_all_statistics_have_analytic_values():
    # Normalized heights .125/.375/.625/.875, one point per height bin.
    heights = -2.5 + 3.5 * np.array([.125, .375, .625, .875])
    points = np.column_stack([[.1, .2, .3, .4], [-1.9, -1.8, -1.7, -1.6],
                              heights, [.1, .3, .5, .7]])
    expected = [log_density(1)] * 4 + [.875, .5, .75, math.sqrt(5/64),
                .7, .4, log_density(4), 1, -.25, 0]
    bev = reference(points)
    np.testing.assert_allclose(bev[0, 0], expected, atol=1e-7)
    assert np.count_nonzero(bev[..., 11]) == 1


@pytest.mark.parametrize("edge,bin_index", [(-1.625, 1), (-.75, 2), (.125, 3)])
def test_current_geometry_internal_height_ties_go_to_upper_bin(edge, bin_index):
    bev = reference(np.array([[.5, -1.75, edge, .2]]))
    assert np.flatnonzero(bev[0, 0, :4]).tolist() == [bin_index]


@pytest.mark.parametrize("edge", [1., 2., 3.])
def test_height_edges_and_immediate_neighbors_use_float64_arithmetic(edge):
    geometry = dict(GEOMETRY, z_min=0., z_max=4., z_res=1.)
    for height, expected_bin in [(np.nextafter(edge, -np.inf), int(edge)-1),
                                 (edge, int(edge)), (np.nextafter(edge, np.inf), int(edge))]:
        bev = reference(np.array([[.5, -1.75, height, .2]]), geometry)
        assert np.flatnonzero(bev[0, 0, :4]).tolist() == [expected_bin]


@pytest.mark.parametrize("column", [0, 1])
def test_xy_cell_edges_and_immediate_neighbors(column):
    geometry = dict(GEOMETRY, y_min=0, y_max=4, y_res=.5)
    edge = 1.0
    for coordinate, index in [(np.nextafter(edge, -np.inf), (0 if column == 0 else 1)),
                               (edge, (1 if column == 0 else 2)),
                               (np.nextafter(edge, np.inf), (1 if column == 0 else 2))]:
        point = np.array([[.25, .25, -.5, .2]])
        point[0, column] = coordinate
        bev = reference(point, geometry)
        expected = [0, index] if column == 0 else [index, 0]
        assert np.argwhere(bev[..., 11]).tolist() == [expected]
        assert np.abs(bev[..., 12:]).max() <= .5


def test_open_epsilon_roi_with_nextafter_neighbors():
    for axis in ["x", "y", "z"]:
        for side in ["min", "max"]:
            column = "xyz".index(axis)
            boundary = GEOMETRY[f"{axis}_{side}"] + (.001 if side == "min" else -.001)
            inside = np.nextafter(boundary, np.inf if side == "min" else -np.inf)
            outside = np.nextafter(boundary, -np.inf if side == "min" else np.inf)
            for coordinate, kept in [(boundary, False), (outside, False), (inside, True)]:
                point = np.array([[1.5, -.75, -.5, .2]])
                point[0, column] = coordinate
                bev = reference(point)
                assert np.count_nonzero(bev[..., 11]) == int(kept)


def test_nonfinite_first_four_columns_are_filtered():
    for column in range(4):
        for invalid in [float("nan"), float("inf"), -float("inf")]:
            points = np.array([[.5, -1.75, -.5, .2], [1.5, -.75, -.5, .2]])
            points[1, column] = invalid
            np.testing.assert_array_equal(reference(points), reference(points[:1]))


def test_extra_columns_ignored_and_inputs_not_mutated():
    points = np.array([[.5, -1.75, -.5, .2, np.nan], [1.5, -.75, .5, .7, np.inf]])
    original = points.copy()
    np.testing.assert_array_equal(reference(points), reference(points[:, :4]))
    np.testing.assert_array_equal(points, original)


def test_density_and_intensity_saturate_without_overflow_warnings():
    points = np.tile([.25, -1.625, -.5, 1e308], (100, 1))
    with np.errstate(all="raise"):
        bev = reference(points, density_norm=8, intensity_scale=2)
    assert bev[0, 0, 2] == bev[0, 0, 10] == 1
    assert bev[0, 0, 8] == bev[0, 0, 9] == 1
    points[:, 3] = -1e308
    assert not reference(points, intensity_scale=2)[..., 8:10].any()


def test_intensity_scale_applies_before_clipping():
    bev = reference(np.array([[.5, -1.75, -.5, .2], [.5, -1.75, -.5, .8]]), intensity_scale=2)
    np.testing.assert_allclose(bev[0, 0, 8:10], [1, .7])


def test_variance_guard_constant_and_nearly_constant_heights():
    points = np.tile([.5, -1.75, -.7, .2], (1001, 1))
    points[-1, 2] = np.nextafter(-.7, np.inf)
    bev = reference(points)
    assert np.isfinite(bev).all()
    assert 0 <= bev[0, 0, 7] < 1e-6


def test_random_cloud_permutations_preserve_all_channels():
    rng = np.random.default_rng(31)
    points = rng.uniform([-.1, -2.1, -2.6, -.5], [4.1, 2.1, 1.1, 1.5], (3000, 4))
    expected = reference(points)
    actual = reference(points[rng.permutation(len(points))])
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert np.isfinite(actual).all()
    assert np.all((actual[..., :12] >= 0) & (actual[..., :12] <= 1))
    assert np.abs(actual[..., 12:]).max() <= .5


def test_invalid_point_arrays_fail_clearly():
    invalid_points = [
        np.array([]), np.zeros((2, 3)), np.zeros((1, 4, 1)),
        np.array([["x", "y", "z", "i"]]), np.ones((1, 4), dtype=np.complex128),
    ]
    for points in invalid_points:
        with pytest.raises(ValueError, match="points"):
            reference(points)


def test_public_hist14_dispatch_matches_reference_and_validates_schema():
    from utils_1.preprocess import encode_bev
    points = np.array([[.25, -1.625, -1.625, .4]])
    actual = encode_bev(points, GEOMETRY, {"name": "hist14"})
    np.testing.assert_array_equal(actual, reference(points))
    with pytest.raises(ValueError, match="exactly integer 14"):
        encode_bev(points, GEOMETRY, {"name": "hist14", "out_channels": 8})


def small_hist14_dataset(tmp_path, task="train", empty=False):
    from core.datasets.dataset import Dataset
    geometry = {"x_min": 0, "x_max": 4.8, "x_res": .1,
                "y_min": -1.6, "y_max": 1.6, "y_res": .1,
                "z_min": -2.5, "z_max": 1, "z_res": .1}
    data = {"num_classes": 3, "out_size_factor": 4, "min_radius": 1,
            "gaussian_overlap": .1, "bev_encoding": {"name": "hist14"},
            "kitti": {"location": str(tmp_path), "geometry": geometry,
                      "objects": {"Car": 0, "Pedestrian": 1, "Cyclist": 2}}}
    (tmp_path / "pointcloud").mkdir(exist_ok=True)
    (tmp_path / "label").mkdir(exist_ok=True)
    points = np.empty((0, 4), np.float32) if empty else np.array([[1.25, -.35, -.75, .4]], np.float32)
    points.tofile(tmp_path / "pointcloud/000001.bin")
    (tmp_path / "label/000001.txt").write_text("" if empty else "Car 1.5 .6 .8 1.25 -.35 -1.5 0\n")
    manifest = tmp_path / "frames.txt"
    manifest.write_text("000001;kitti\n")
    augmentation = {"p": 0, "rotation": {"use": False},
                    "scaling": {"use": False}, "translation": {"use": False}}
    return Dataset(str(manifest), data, augmentation, "gaussian", task), data, points


def test_dataset_hist14_shapes_agree_with_model_on_float_roundoff_grid(tmp_path):
    from tools.kitti_training_pipeline.common import build_model, input_shape
    import torch
    dataset, data, _ = small_hist14_dataset(tmp_path)
    sample = dataset[0]
    config = {"data": data, "model": {"backbone": "mobilepixornext", "c4_attention": "none"}}
    assert sample["voxel"].shape == input_shape(config)[1:] == (14, 32, 48)
    assert sample["cls"].shape == (3, 8, 12)
    model = build_model(config).eval()
    with torch.no_grad():
        outputs = model(sample["voxel"].unsqueeze(0))
    for name in ("cls", "offset", "size", "yaw"):
        assert outputs[name].shape[1:] == sample[name].shape


def test_dataset_encodes_transformed_points_and_reencodes_each_sample(tmp_path):
    from utils_1.transform import Compose, Random_Scaling
    dataset, _, raw = small_hist14_dataset(tmp_path)
    source_boxes = dataset.get_boxes(0)
    dataset.augment = Compose([Random_Scaling((1.5, 1.5), p=1)], p=1)
    first = dataset[0]
    occupancy = first["voxel"][11].numpy()
    assert np.argwhere(occupancy).tolist() == [[10, 18]]
    assert first["voxel"][1, 10, 18] > 0  # Scaling changed the height bin too.
    assert first["voxel"][2, 10, 18] == 0
    np.testing.assert_allclose(first["voxel"][4, 10, 18].numpy(), 1.375/3.5, atol=1e-7)
    np.testing.assert_allclose(first["voxel"][12:, 10, 18].numpy(), [.25, .25], atol=1e-6)
    assert first["cls"][0, 2, 4] == 1
    np.testing.assert_allclose(first["offset"][:, 2, 4].numpy(), [.275, .275], atol=1e-6)
    dataset.augment = Compose([Random_Scaling((2, 2), p=1)], p=1)
    second = dataset[0]
    # Re-read the raw point (x=1.25), rather than caching/reusing the first BEV.
    assert np.argwhere(second["voxel"][11].numpy()).tolist() == [[9, 25]]
    np.testing.assert_array_equal(dataset.get_boxes(0), source_boxes)
    np.testing.assert_array_equal(dataset.read_points(tmp_path / "pointcloud/000001.bin"), raw)


@pytest.mark.parametrize("task", ["validation", "val", "test"])
def test_hist14_validation_and_test_keep_unaugmented_points(tmp_path, task):
    dataset, _, points = small_hist14_dataset(tmp_path, task)
    sample = dataset[0]
    assert sample["voxel"].shape == (14, 32, 48)
    assert np.argwhere(sample["voxel"][11].numpy()).tolist() == [[12, 12]]
    if task in ("val", "test"):
        np.testing.assert_array_equal(sample["points"], points)


def test_hist14_dataset_accepts_empty_scene(tmp_path):
    dataset, _, _ = small_hist14_dataset(tmp_path, empty=True)
    sample = dataset[0]
    assert sample["voxel"].shape == (14, 32, 48)
    assert not sample["voxel"].any()
    assert not sample["reg_mask"].any()


def test_legacy_validation_corners_preserve_frozen_values_without_array_wrap_warning():
    from core.datasets.dataset import Dataset
    import torch
    frozen = json.loads((ROOT / "tests/fixtures/legacy_validation_corners.json").read_text())
    dataset = Dataset.__new__(Dataset)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        for box, expected in zip(frozen["boxes"], frozen["corners"]):
            actual = dataset.get3D_corners(torch.tensor(box, dtype=torch.float32))
            np.testing.assert_array_equal(actual, expected)
