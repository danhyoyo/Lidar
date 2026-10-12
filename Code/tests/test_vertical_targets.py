"""Bottom-z/log-height targets must follow the exact BEV cell owner."""

import copy
import importlib

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from test_grouped_targets import GEOMETRY, make_dataset, target_dataset


def api():
    module = importlib.import_module("utils_1.target_backend")
    assert hasattr(module, "fill_regression_targets_3d"), "Explicit 3D target API is missing"
    return module


def boxes():
    return np.array([[0, 1.5, 1.6, 3.5, 2.15, .05, -1.7, .2],
                     [1, 2.1, .6, .8, 2.55, .25, -.9, -.4],
                     [2, .8, .9, 1.2, 4.15, .45, -2.2, .6]], np.float32)


@pytest.mark.parametrize("backend", ["python", "numba"])
@pytest.mark.parametrize("assignment", ["nearest_center", "legacy"])
def test_vertical_api_preserves_bev_tuple_and_uses_owner_at_every_supervised_cell(backend, assignment):
    module = api()
    source = boxes()
    radii = np.array([3., 3., 2.], np.float32)
    before = source.copy()
    old = getattr(module, f"fill_regression_targets_{backend}")(source, radii, (12, 16), GEOMETRY, 4, assignment)
    actual = module.fill_regression_targets_3d(source, radii, (12, 16), GEOMETRY, 4,
                                              backend=backend, assignment=assignment)
    assert len(old) == 4 and len(actual) == 5
    for original, extended in zip(old, actual[:4]):
        np.testing.assert_array_equal(original, extended)
    offset, size, yaw, mask, vertical = actual
    assert vertical.shape == (12, 16, 2) and vertical.dtype == np.float32
    assert np.isfinite(vertical).all() and not vertical[mask == 0].any()
    for px, py in np.argwhere(mask):
        xy = offset[px, py] + np.array([px * 4 * .2, py * 4 * .1 - 3.2], np.float32)
        owner = np.argmin(np.linalg.norm(source[:, 4:6] - xy, axis=1))
        np.testing.assert_allclose(xy, source[owner, 4:6], atol=1e-6)
        np.testing.assert_allclose(size[px, py], np.log(source[owner, 2:4]), atol=1e-6)
        np.testing.assert_allclose(yaw[px, py], [np.cos(2 * source[owner, 7]), np.sin(2 * source[owner, 7])], atol=1e-6)
        np.testing.assert_allclose(vertical[px, py], [source[owner, 6], np.log(source[owner, 1])], atol=1e-6)
    np.testing.assert_array_equal(source, before)


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_reserved_small_radius_peak_and_exact_same_cell_collision_use_bev_winner(backend):
    module = api()
    source = boxes()[:2]
    source[1, 4:6] = source[0, 4:6]
    # Canonical global Car ID owns the exact same-cell tie despite input order.
    result = module.fill_regression_targets_3d(source[::-1], [.2, .2], (12, 16), GEOMETRY, 4, backend=backend)
    assert result[3][2, 8] == 1
    np.testing.assert_allclose(result[4][2, 8], [-1.7, np.log(1.5)], atol=1e-6)
    reordered = module.fill_regression_targets_3d(source, [.2, .2], (12, 16), GEOMETRY, 4, backend=backend)
    for first, second in zip(result, reordered):
        np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_empty_3d_maps_are_zero_and_backend_parity_is_actual(backend):
    module = api()
    empty = module.fill_regression_targets_3d(np.empty((0, 8)), [], (12, 16), GEOMETRY, 4, backend=backend)
    assert len(empty) == 5 and all(not array.any() for array in empty)
    expected = module.fill_regression_targets_3d(boxes(), [2, 3, 1], (12, 16), GEOMETRY, 4)
    actual = module.fill_regression_targets_3d(boxes(), [2, 3, 1], (12, 16), GEOMETRY, 4, backend=backend)
    for first, second in zip(expected, actual):
        np.testing.assert_allclose(first, second, rtol=1e-6, atol=1e-6)
    if backend == "numba":
        assert module._compiled_fill.nopython_signatures


@pytest.mark.parametrize("classification", ["gaussian", "binary"])
@pytest.mark.parametrize("backend", ["python", "numba"])
def test_grouped_dataset_keeps_cross_group_vertical_targets_and_collates(tmp_path, classification, backend):
    dataset = make_dataset(tmp_path, classification=classification, backend=backend)
    dataset.config["box_mode"] = "3d"
    source = torch.from_numpy(dataset.get_boxes(0))
    source[1, 4:6] = source[0, 4:6]
    source[1, 1], source[1, 6] = 2.3, -.9
    before = source.clone()
    actual = dataset.get_label(source, GEOMETRY)["groups"]
    for group, owner in [("car", source[0]), ("ped_cyc", source[1])]:
        vertical = actual[group]["vertical"]
        assert vertical.shape == (2, 16, 12) and vertical.dtype == torch.float32
        torch.testing.assert_close(vertical[:, 8, 2], torch.stack([owner[6], owner[1].log()]))
        assert not vertical[:, ~actual[group]["reg_mask"].bool()].any()
    torch.testing.assert_close(source, before, rtol=0, atol=0)
    loaded = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
    assert loaded["groups"]["car"]["vertical"].shape == (1, 2, 16, 12)
    dataset.config["box_mode"] = "bev"
    old = dataset.get_label(source, GEOMETRY)["groups"]
    for group in old:
        assert "vertical" not in old[group]
        for key in old[group]:
            torch.testing.assert_close(old[group][key], actual[group][key], rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_reversed_local_class_mapping_does_not_reassign_vertical_owner(backend):
    dataset = target_dataset(backend=backend)
    dataset.config["box_mode"] = "3d"
    source = torch.from_numpy(boxes())[1:]
    source[1, 4:6] = source[0, 4:6]
    target = dataset.get_single_group_label(source, GEOMETRY, num_classes=2, class_mapping={2: 0, 1: 1})
    torch.testing.assert_close(target["vertical"][:, 8, 3], torch.tensor([-.9, np.log(2.1)], dtype=torch.float32))
    empty = dataset.get_single_group_label(source[:0], GEOMETRY, num_classes=2, class_mapping={2: 0, 1: 1})
    assert all(not value.any() for value in empty.values())


@pytest.mark.parametrize("index,value", [(1, 0.), (1, -1.), (1, np.nan), (6, np.inf)])
def test_invalid_height_or_bottom_z_is_rejected_before_target_generation(index, value):
    module = api()
    source = boxes()
    source[0, index] = value
    with pytest.raises(ValueError, match="finite.*positive"):
        module.fill_regression_targets_3d(source, [2, 2, 2], (12, 16), GEOMETRY, 4)


@pytest.mark.parametrize("radii", [[1, 2], [1, np.nan, 2]])
def test_invalid_radius_count_or_values_are_rejected(radii):
    with pytest.raises(ValueError, match="radii"):
        api().fill_regression_targets_3d(boxes(), radii, (12, 16), GEOMETRY, 4)


def test_unknown_target_backend_and_box_mode_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="backend"):
        api().fill_regression_targets_3d(boxes(), [2, 2, 2], (12, 16), GEOMETRY, 4, backend="auto")
    dataset = make_dataset(tmp_path)
    dataset.config["box_mode"] = "volumetric"
    with pytest.raises(ValueError, match="box_mode"):
        dataset.get_label(torch.from_numpy(boxes()), GEOMETRY)


def test_3d_numba_request_does_not_silently_fall_back(monkeypatch):
    module = api()
    monkeypatch.setattr(module, "_compiled_fill", None)
    with pytest.raises(RuntimeError, match="Numba"):
        module.fill_regression_targets_3d(boxes(), [2, 2, 2], (12, 16), GEOMETRY, 4, backend="numba")
