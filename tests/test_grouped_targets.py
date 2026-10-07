"""Group-local assignment must preserve geometry, ownership and source labels."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "detector"), str(ROOT / "detector/core/datasets")]

from core.datasets.dataset import Dataset
from core.task_groups import resolve_task_groups


GEOMETRY = {"x_min": 0., "x_max": 9.6, "x_res": .2,
            "y_min": -3.2, "y_max": 3.2, "y_res": .1,
            "z_min": -2.5, "z_max": 1., "z_res": .1}
OBJECTS = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}
GROUPS = [{"name": "car", "classes": ["Car"]},
          {"name": "ped_cyc", "classes": ["Pedestrian", "Cyclist"]}]


def target_dataset(classification="gaussian", backend="python", assignment="nearest_center"):
    value = Dataset.__new__(Dataset)
    value.output_shape = [12, 16]  # Internal X,Y; external tensors Y,X.
    value.out_size_factor = 4
    value.num_classes = 3 + int(classification == "binary")
    value.cls_encoding = classification
    value.target_backend = backend
    value.config = {"num_classes": 3, "gaussian_overlap": .1, "min_radius": 2,
                    "regression_assignment": assignment,
                    "kitti": {"objects": OBJECTS, "geometry": GEOMETRY}}
    return value


def boxes():
    return torch.tensor([[0, 1.5, 1.6, 3.5, 2.15, .05, -1.7, .2],
                         [1, 1.7, .6, .8, 4.15, .45, -1.7, -.4],
                         [2, 1.6, .9, 1.2, 4.15, .45, -1.7, 0.]], dtype=torch.float32)


@pytest.mark.parametrize("classification", ["gaussian", "binary"])
@pytest.mark.parametrize("assignment", ["nearest_center", "legacy"])
@pytest.mark.parametrize("backend", ["python", "numba"])
def test_single_group_builder_preserves_flat_targets(classification, assignment, backend):
    value = target_dataset(classification, backend, assignment)
    source = boxes()
    expected = value.get_label(source, GEOMETRY)
    actual = value.get_single_group_label(source, GEOMETRY, num_classes=value.num_classes)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_explicit_mapping_filters_before_assignment_without_mutating_dataset_or_boxes(backend):
    value = target_dataset(backend=backend)
    source = boxes()
    saved_boxes, saved_config = source.clone(), copy.deepcopy(value.config)
    # Reverse local channel order; exact regression ties still use global IDs.
    expected = value.get_label(source[source[:, 0] != 0], GEOMETRY)
    actual = value.get_single_group_label(source, GEOMETRY, num_classes=2,
                                          class_mapping={2: 0, 1: 1})
    torch.testing.assert_close(actual["cls"], expected["cls"][[2, 1]], rtol=0, atol=0)
    for key in ("offset", "size", "yaw", "reg_mask"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    torch.testing.assert_close(source, saved_boxes, rtol=0, atol=0)
    assert value.config == saved_config and value.num_classes == 3
    assert value.cls_encoding == "gaussian"
    assert actual["cls"].shape == (2, 16, 12)
    # Global Pedestrian ID wins this exact center tie, despite reversed local IDs.
    torch.testing.assert_close(actual["size"][:, 9, 5], source[1, 2:4].log())


def test_explicit_encoding_uses_local_binary_foreground_labels_without_changing_dataset():
    value = target_dataset()
    source = boxes()
    actual = value.get_single_group_label(source, GEOMETRY, num_classes=2,
                                          class_mapping={2: 0, 1: 1}, cls_encoding="binary")
    assert actual["cls"].shape == (16, 12)
    assert actual["cls"].dtype == torch.int64
    assert actual["cls"][9, 5] == 1  # Cyclist is last in the preserved source order.
    assert value.cls_encoding == "gaussian" and value.num_classes == 3
    assert actual["cls"][0, 0] == 0


def test_empty_filtered_group_produces_negative_only_maps():
    value = target_dataset()
    target = value.get_single_group_label(boxes()[:1], GEOMETRY, num_classes=2,
                                          class_mapping={1: 0, 2: 1})
    assert target["cls"].shape == (2, 16, 12)
    assert all(not tensor.any() for tensor in target.values())


def make_dataset(tmp_path, *, classification="gaussian", backend="python",
                 grouped=True, task="validation", task_groups=None):
    root = tmp_path / "processed"
    (root / "pointcloud").mkdir(parents=True, exist_ok=True)
    (root / "label").mkdir(exist_ok=True)
    points = np.array([[2.15, .05, -.5, .8], [6.65, -1.05, -.7, .3]], np.float32)
    points.tofile(root / "pointcloud/000000.bin")
    (root / "label/000000.txt").write_text(
        "Car 1.5 1.6 3.5 2.15 0.05 -1.7 0.2\n"
        "Pedestrian 1.7 0.6 0.8 2.35 0.25 -1.7 -0.4\n"
        "Cyclist 1.6 0.7 1.2 6.65 -1.05 -1.7 0.0\n")
    manifest = tmp_path / "frames.txt"
    manifest.write_text("000000;kitti\n")
    config = {"num_classes": 3, "out_size_factor": 4, "gaussian_overlap": .1,
              "min_radius": 2, "regression_assignment": "nearest_center",
              "bev_encoding": {"name": "hist14"},
              "kitti": {"location": str(root), "objects": copy.deepcopy(OBJECTS),
                        "geometry": copy.deepcopy(GEOMETRY)}}
    if grouped:
        config["head_groups"] = copy.deepcopy(GROUPS)
    aug = {"p": 0., "rotation": {"use": False}, "scaling": {"use": False},
           "translation": {"use": False}}
    options = {} if task_groups is None else {"task_groups": task_groups}
    return Dataset(str(manifest), config, aug, classification, task, backend, **options)


def assert_box_at_cell(target, box):
    px = int((float(box[4]) - GEOMETRY["x_min"]) / GEOMETRY["x_res"] / 4)
    py = int((float(box[5]) - GEOMETRY["y_min"]) / GEOMETRY["y_res"] / 4)
    assert target["reg_mask"][py, px] == 1
    origin = torch.tensor([px * .8, py * .4 - 3.2])
    torch.testing.assert_close(target["offset"][:, py, px] + origin, box[4:6])
    torch.testing.assert_close(target["size"][:, py, px], box[2:4].log())
    torch.testing.assert_close(target["yaw"][:, py, px],
                               torch.stack([torch.cos(2 * box[7]), torch.sin(2 * box[7])]))


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_gaussian_cross_group_same_cell_keeps_both_regression_targets(tmp_path, backend):
    value = make_dataset(tmp_path, backend=backend)
    source = torch.from_numpy(value.get_boxes(0))
    saved = source.clone()
    target = value.get_label(source, GEOMETRY)
    assert tuple(target) == ("groups",)
    car, vru = target["groups"]["car"], target["groups"]["ped_cyc"]
    assert car["cls"].shape == (1, 16, 12)
    assert vru["cls"].shape == (2, 16, 12)
    assert car["cls"][0, 8, 2] == vru["cls"][0, 8, 2] == 1
    assert_box_at_cell(car, source[0])
    assert_box_at_cell(vru, source[1])
    assert_box_at_cell(vru, source[2])
    for key in ("offset", "size", "yaw", "reg_mask"):
        assert car[key].data_ptr() != vru[key].data_ptr()
    torch.testing.assert_close(source, saved, rtol=0, atol=0)
    modified = source.clone()
    modified[1, 2] = .9
    updated = value.get_label(modified, GEOMETRY)["groups"]
    for key in car:
        torch.testing.assert_close(updated["car"][key], car[key], rtol=0, atol=0)
    assert not torch.equal(updated["ped_cyc"]["size"], vru["size"])


@pytest.mark.parametrize("task", ["train", "validation", "val", "test"])
def test_dataset_sample_keeps_voxel_and_validation_metadata_at_top_level(tmp_path, task):
    value = make_dataset(tmp_path, task=task)
    saved_config, cached = copy.deepcopy(value.config), value.get_boxes(0) if task != "test" else None
    sample = value[0]
    assert sample["voxel"].shape == (14, 64, 48)
    if task == "test":
        assert set(sample) == {"voxel", "points", "dtype"}
    else:
        assert set(sample["groups"]) == {"car", "ped_cyc"}
        assert set(sample["groups"]["car"]) == {"cls", "offset", "size", "yaw", "reg_mask"}
        assert not {"cls", "offset", "size", "yaw", "reg_mask"} & set(sample)
        np.testing.assert_array_equal(value.get_boxes(0), cached)
        if task == "val":
            assert set(sample) == {"voxel", "groups", "points", "dtype", "cls_list", "boxes"}
            assert [int(i) for i in sample["cls_list"]] == [0, 1, 2]
            assert len(sample["boxes"]) == 3
        else:
            assert set(sample) == {"voxel", "groups"}
    assert value.config == saved_config


@pytest.mark.parametrize("empty_all", [False, True])
def test_gaussian_empty_groups_have_negative_only_classification_and_zero_regression(tmp_path, empty_all):
    value = make_dataset(tmp_path)
    source = torch.from_numpy(value.get_boxes(0))[:0 if empty_all else 1]
    target = value.get_label(source, GEOMETRY)["groups"]
    assert target["ped_cyc"]["cls"].shape == (2, 16, 12)
    assert all(not tensor.any() for tensor in target["ped_cyc"].values())
    if empty_all:
        assert all(not tensor.any() for tensor in target["car"].values())
    else:
        assert_box_at_cell(target["car"], source[0])


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_within_vru_same_cell_retains_existing_single_regression_owner(tmp_path, backend):
    value = make_dataset(tmp_path, backend=backend)
    source = torch.from_numpy(value.get_boxes(0))[1:]
    source[1, 4:6] = source[0, 4:6]  # Exact tie; global ID 1 owns geometry.
    target = value.get_label(source, GEOMETRY)["groups"]["ped_cyc"]
    assert target["cls"][0, 8, 2] == target["cls"][1, 8, 2] == 1
    assert_box_at_cell(target, source[0])
    reversed_target = value.get_label(source.flip(0), GEOMETRY)["groups"]["ped_cyc"]
    for key in target:
        torch.testing.assert_close(reversed_target[key], target[key], rtol=0, atol=0)


def test_reordered_groups_and_classes_use_the_shared_resolver(tmp_path):
    value = make_dataset(tmp_path)
    config = copy.deepcopy(value.config)
    config["head_groups"] = [{"name": "vru", "classes": ["Cyclist", "Pedestrian"]}, GROUPS[0]]
    aug = {"p": 0., "rotation": {"use": False}, "scaling": {"use": False}, "translation": {"use": False}}
    reordered = Dataset(value.data_file, config, aug, "gaussian", task="validation")
    assert reordered.task_groups == resolve_task_groups(config["head_groups"], OBJECTS)
    target = reordered[0]["groups"]
    assert tuple(target) == ("vru", "car")
    assert target["vru"]["cls"][1, 8, 2] == 1
    assert target["vru"]["cls"][0, 5, 8] == 1


def test_explicit_metadata_is_checked_against_authoritative_class_map(tmp_path):
    supplied = resolve_task_groups(GROUPS, OBJECTS)
    value = make_dataset(tmp_path, grouped=False, task_groups=supplied)
    assert value.task_groups == supplied
    assert "groups" in value[0]
    wrong = resolve_task_groups(GROUPS, {"Car": 1, "Pedestrian": 0, "Cyclist": 2})
    with pytest.raises(ValueError, match="mapping"):
        make_dataset(tmp_path, task_groups=wrong)


@pytest.mark.parametrize("class_id", [-1., 3., .5, float("nan"), float("inf")])
def test_grouped_target_rejects_invalid_global_box_class_ids(tmp_path, class_id):
    value = make_dataset(tmp_path)
    source = torch.from_numpy(value.get_boxes(0))
    source[0, 0] = class_id
    with pytest.raises(ValueError, match="class ID"):
        value.get_label(source, GEOMETRY)


@pytest.mark.parametrize("backend", ["python", "numba"])
def test_binary_group_labels_remap_foreground_and_keep_cross_group_regression_independent(tmp_path, backend):
    value = make_dataset(tmp_path, classification="binary", backend=backend)
    source = torch.from_numpy(value.get_boxes(0))
    saved = source.clone()
    targets = value.get_label(source, GEOMETRY)["groups"]
    for target in targets.values():
        assert target["cls"].dtype == torch.int64
        assert target["cls"].shape == (16, 12)
        assert target["cls"][0, 0] == 0
    assert targets["car"]["cls"][8, 2] == 1
    assert targets["ped_cyc"]["cls"][8, 2] == 1
    assert targets["ped_cyc"]["cls"][5, 8] == 2
    assert set(targets["car"]["cls"].unique().tolist()) == {0, 1}
    assert set(targets["ped_cyc"]["cls"].unique().tolist()) == {0, 1, 2}
    assert_box_at_cell(targets["car"], source[0])
    assert_box_at_cell(targets["ped_cyc"], source[1])
    assert_box_at_cell(targets["ped_cyc"], source[2])
    torch.testing.assert_close(source, saved, rtol=0, atol=0)
    assert value.num_classes == 4 and value.cls_encoding == "binary"


@pytest.mark.parametrize("empty_all", [False, True])
def test_binary_background_only_groups_have_zero_maps(tmp_path, empty_all):
    value = make_dataset(tmp_path, classification="binary")
    source = torch.from_numpy(value.get_boxes(0))[:0 if empty_all else 1]
    target = value.get_label(source, GEOMETRY)["groups"]
    assert all(not t.any() for t in target["ped_cyc"].values())
    if empty_all:
        assert all(not t.any() for t in target["car"].values())


def test_binary_reordered_foreground_maps_through_global_ids(tmp_path):
    value = make_dataset(tmp_path, classification="binary")
    config = copy.deepcopy(value.config)
    config["head_groups"] = [{"name": "vru", "classes": ["Cyclist", "Pedestrian"]}, GROUPS[0]]
    aug = {"p": 0., "rotation": {"use": False}, "scaling": {"use": False}, "translation": {"use": False}}
    value = Dataset(value.data_file, config, aug, "binary", task="validation")
    vru = value[0]["groups"]["vru"]
    assert vru["cls"][8, 2] == 2  # Global Pedestrian=1 -> local foreground=2.
    assert vru["cls"][5, 8] == 1  # Global Cyclist=2 -> local foreground=1.
    assert value.task_groups[0].binary_to_global(int(vru["cls"][8, 2])) == 1
    assert value.task_groups[0].binary_to_global(int(vru["cls"][5, 8])) == 2


@pytest.mark.parametrize("classification", ["gaussian", "binary"])
def test_group_partition_matches_separately_assigned_filtered_boxes(tmp_path, classification):
    value = make_dataset(tmp_path, classification=classification)
    source = torch.from_numpy(value.get_boxes(0))
    targets = value.get_label(source, GEOMETRY)["groups"]
    flat = target_dataset(classification)
    for group in value.task_groups:
        selected = source[torch.tensor([int(i) in group.global_ids for i in source[:, 0]])]
        reference = flat.get_label(selected, GEOMETRY)
        for key in ("offset", "size", "yaw", "reg_mask"):
            torch.testing.assert_close(targets[group.name][key], reference[key], rtol=0, atol=0)
        if classification == "gaussian":
            torch.testing.assert_close(targets[group.name]["cls"], reference["cls"][list(group.global_ids)], rtol=0, atol=0)
        else:
            expected = torch.zeros_like(reference["cls"])
            for local_id, global_id in enumerate(group.global_ids):
                expected[reference["cls"] == global_id + 1] = local_id + 1
            torch.testing.assert_close(targets[group.name]["cls"], expected, rtol=0, atol=0)


@pytest.mark.parametrize("mapping,count", [({1: 1, 2: 2}, 2), ({1: 0, 2: 0}, 2),
    ({1: 0}, 2), ({True: 0}, 1), ({1: True}, 1), ({-1: 0}, 1), ({1: 0}, True)])
def test_explicit_mapping_rejects_invalid_local_foreground_space(mapping, count):
    value = target_dataset("binary")
    with pytest.raises(ValueError, match="class"):
        value.get_single_group_label(boxes(), GEOMETRY, num_classes=count,
                                     class_mapping=mapping)


def test_nondefault_classes_share_the_same_partition_and_local_label_contract(tmp_path):
    value = make_dataset(tmp_path)
    config = copy.deepcopy(value.config)
    config["num_classes"] = 2
    config["kitti"]["objects"] = {"Bus": 0, "Person": 1}
    config["head_groups"] = [{"name": "people", "classes": ["Person"]},
                             {"name": "vehicles", "classes": ["Bus"]}]
    (Path(config["kitti"]["location"]) / "label/000000.txt").write_text(
        "Bus 2.0 2.0 4.0 2.15 0.05 -1.7 0.0\n"
        "Person 1.7 0.6 0.8 2.35 0.25 -1.7 0.0\n")
    aug = {"p": 0., "rotation": {"use": False}, "scaling": {"use": False}, "translation": {"use": False}}
    for classification in ("gaussian", "binary"):
        actual = Dataset(value.data_file, config, aug, classification, task="validation")
        target = actual[0]["groups"]
        assert tuple(target) == ("people", "vehicles")
        for group_target in target.values():
            index = (0, 8, 2) if classification == "gaussian" else (8, 2)
            assert group_target["cls"][index] == 1


@pytest.mark.parametrize("classification", ["Gaussian", "BINARY"])
def test_grouped_classification_names_match_central_case_normalization(tmp_path, classification):
    value = make_dataset(tmp_path, classification=classification)
    assert value.cls_encoding == classification.lower()
    target = value[0]["groups"]["car"]["cls"]
    assert target.shape == ((16, 12) if classification == "BINARY" else (1, 16, 12))
    assert value.num_classes == (4 if classification == "BINARY" else 3)


@pytest.mark.parametrize("classification", ["gaussian", "binary"])
def test_grouped_constructor_rejects_invalid_class_count_and_unsupported_encoding(tmp_path, classification):
    value = make_dataset(tmp_path)
    aug = {"p": 0., "rotation": {"use": False}, "scaling": {"use": False}, "translation": {"use": False}}
    for count in (None, True, 4):
        config = copy.deepcopy(value.config)
        config["num_classes"] = count
        with pytest.raises(ValueError, match="num_classes"):
            Dataset(value.data_file, config, aug, classification)
    with pytest.raises(ValueError, match="classification"):
        Dataset(value.data_file, value.config, aug, "inverse_distance")


def test_binary_within_group_overlap_retains_legacy_painting_and_regression_rules(tmp_path):
    value = make_dataset(tmp_path, classification="binary")
    source = torch.from_numpy(value.get_boxes(0))[1:]
    source[1, 4:6] = source[0, 4:6]
    flat = target_dataset("binary")
    for ordered in (source, source.flip(0)):
        actual = value.get_label(ordered, GEOMETRY)["groups"]["ped_cyc"]
        legacy = flat.get_label(ordered, GEOMETRY)
        expected = torch.zeros_like(legacy["cls"])
        for global_id, local_id in ((1, 1), (2, 2)):
            expected[legacy["cls"] == global_id + 1] = local_id
        torch.testing.assert_close(actual["cls"], expected, rtol=0, atol=0)
        for key in ("offset", "size", "yaw", "reg_mask"):
            torch.testing.assert_close(actual[key], legacy[key], rtol=0, atol=0)
    # Binary classification paints last-box-wins; nearest-center regression is
    # canonical. Preserve this legacy limitation for task33's capability audit.
    first = value.get_label(source, GEOMETRY)["groups"]["ped_cyc"]
    reverse = value.get_label(source.flip(0), GEOMETRY)["groups"]["ped_cyc"]
    assert first["cls"][8, 2] == 2 and reverse["cls"][8, 2] == 1
    torch.testing.assert_close(first["size"], reverse["size"], rtol=0, atol=0)
