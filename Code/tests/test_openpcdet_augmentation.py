"""Physical consistency of the opt-in ordered LiDAR augmentation queue."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "detector"), str(ROOT / "detector/core/datasets"),
               str(ROOT / "tools/kitti_training_pipeline")]
CLASSES = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}


def augmentor(configs, tmp_path, seed=42, **kwargs):
    from core.datasets.augmentor.data_augmentor import DataAugmentor
    return DataAugmentor(tmp_path, configs, CLASSES, rng=np.random.default_rng(seed), **kwargs)


def scene():
    # Bottom Z is -1; the point is inside a box whose heading is zero.
    points = np.array([[3, 2, -.5, .7], [8, 5, 0, .2]], dtype=np.float32)
    boxes = np.array([[1, 2, 2, 4, 2, 2, -1, 0]], dtype=np.float32)
    return points, boxes


def test_ordered_queue_rotates_then_scales_with_point_box_agreement(tmp_path):
    points, boxes = scene()
    before_points, before_boxes = points.copy(), boxes.copy()
    aug = augmentor({"AUG_CONFIG_LIST": [
        {"NAME": "random_world_rotation", "WORLD_ROT_ANGLE": [math.pi/2, math.pi/2]},
        {"NAME": "random_world_scaling", "WORLD_SCALE_RANGE": [2, 2]},
    ]}, tmp_path)
    out_points, out_boxes = aug(points, boxes)
    np.testing.assert_allclose(out_points[:, :3], [[-4, 6, -1], [-10, 16, 0]], atol=1e-6)
    np.testing.assert_allclose(out_boxes[0, 1:7], [4, 4, 8, -4, 4, -2], atol=1e-6)
    assert out_boxes[0, 7] == pytest.approx(math.pi/2)
    np.testing.assert_array_equal(out_points[:, 3], points[:, 3])
    np.testing.assert_array_equal(out_boxes[:, 0], boxes[:, 0])
    np.testing.assert_array_equal(points, before_points)
    np.testing.assert_array_equal(boxes, before_boxes)


@pytest.mark.parametrize("axis,expected_xy,expected_yaw", [
    ("x", [2, -2], -.3), ("y", [-2, 2], math.pi-.3),
])
def test_flip_uses_openpcdet_axis_semantics(tmp_path, axis, expected_xy, expected_yaw):
    points, boxes = scene()
    boxes[0, 7] = .3
    aug = augmentor([{"NAME": "random_world_flip", "ALONG_AXIS_LIST": [axis], "PROBABILITY": 1}], tmp_path)
    transformed_points, transformed_boxes = aug(points, boxes)
    np.testing.assert_allclose(transformed_boxes[0, 4:6], expected_xy)
    assert transformed_boxes[0, 7] == pytest.approx(expected_yaw)
    dimension = 1 if axis == "x" else 0
    np.testing.assert_array_equal(transformed_points[:, dimension], -points[:, dimension])
    np.testing.assert_array_equal(transformed_points[:, 2:], points[:, 2:])


def test_disabled_operations_and_zero_probability_do_not_change_scene(tmp_path):
    points, boxes = scene()
    aug = augmentor({"DISABLE_AUG_LIST": ["random_world_scaling"], "AUG_CONFIG_LIST": [
        {"NAME": "random_world_scaling", "WORLD_SCALE_RANGE": [2, 2]},
        {"NAME": "random_world_flip", "ALONG_AXIS_LIST": ["x"], "PROBABILITY": 0},
    ]}, tmp_path)
    actual_points, actual_boxes = aug(points, boxes)
    np.testing.assert_array_equal(actual_points, points)
    np.testing.assert_array_equal(actual_boxes, boxes)


def test_translation_uses_independent_axis_standard_deviations(tmp_path):
    seed = 11
    config = [{"NAME": "random_world_translation", "NOISE_TRANSLATE_STD": [0, 0, .2]}]
    points, boxes = scene()
    expected = np.random.default_rng(seed).normal(0, [0, 0, .2], size=3)
    out_points, out_boxes = augmentor(config, tmp_path, seed=seed)(points, boxes)
    np.testing.assert_allclose(out_points[:, :3], points[:, :3] + expected, atol=1e-7)
    np.testing.assert_allclose(out_boxes[:, 4:7], boxes[:, 4:7] + expected, atol=1e-7)


def test_same_seed_produces_same_sequence_and_empty_boxes_are_supported(tmp_path):
    configs = [
        {"NAME": "random_world_flip", "ALONG_AXIS_LIST": ["x"]},
        {"NAME": "random_world_rotation", "WORLD_ROT_ANGLE": [-.4, .4]},
        {"NAME": "random_world_scaling", "WORLD_SCALE_RANGE": [.95, 1.05]},
        {"NAME": "random_world_translation", "NOISE_TRANSLATE_STD": [.1, .1, .1]},
    ]
    first, second = augmentor(configs, tmp_path), augmentor(configs, tmp_path)
    points, boxes = scene()
    for _ in range(3):
        for a, b in zip(first(points, boxes), second(points, boxes)):
            np.testing.assert_array_equal(a, b)
    out_points, out_boxes = first(points, np.empty((0, 8), dtype=np.float32))
    assert out_points.shape == points.shape and out_boxes.shape == (0, 8)
    out_points, out_boxes = first(np.empty((0, 4), dtype=np.float32), boxes)
    assert out_points.shape == (0, 4) and out_boxes.shape == boxes.shape


@pytest.mark.parametrize("config", [
    {"NAME": "typo"},
    {"NAME": "random_world_flip", "ALONG_AXIS_LIST": ["z"]},
    {"NAME": "random_world_scaling", "WORLD_SCALE_RANGE": [-1, 2]},
    {"NAME": "random_world_rotation", "WORLD_ROT_ANGLE": [1, -1]},
    {"NAME": "random_world_translation", "NOISE_TRANSLATE_STD": [0, -1, 0]},
    {"NAME": "random_world_flip", "ALONG_AXIS_LIST": ["x"], "PROBABILITY": 2},
])
def test_invalid_augmentation_config_fails_before_training(tmp_path, config):
    with pytest.raises(ValueError):
        augmentor([config], tmp_path)


@pytest.mark.parametrize("config", [{}, {"AUG_CONFIG_LIST": ["random_world_flip"]},
                                     {"AUG_CONFIG_LIST": [{"NAME": "random_world_rotation"}]}])
def test_malformed_recipe_reports_a_configuration_error(tmp_path, config):
    with pytest.raises(ValueError, match="AUG_CONFIG_LIST|entry|WORLD_ROT_ANGLE"):
        augmentor(config, tmp_path)


def processed_scene(tmp_path):
    root = tmp_path / "processed"
    (root / "pointcloud").mkdir(parents=True)
    (root / "label").mkdir()
    points, boxes = scene()
    points.tofile(root / "pointcloud/000000.bin")
    (root / "label/000000.txt").write_text("Pedestrian 2 2 4 2 2 -1 0\n")
    manifest = tmp_path / "train.txt"
    manifest.write_text("000000;kitti\n")
    data = {
        "num_classes": 3, "out_size_factor": 4, "min_radius": 4, "gaussian_overlap": .1,
        "bev_encoding": {"name": "rich8"},
        "kitti": {"location": str(root), "objects": CLASSES,
                  "geometry": {"x_min": 0, "x_max": 6.4, "y_min": -3.2, "y_max": 3.2,
                               "z_min": -2.5, "z_max": 2, "x_res": .1, "y_res": .1, "z_res": .1}},
    }
    return root, manifest, data


def test_dataset_uses_ordered_augmentation_only_for_train(tmp_path):
    from core.datasets.dataset import Dataset
    root, manifest, data = processed_scene(tmp_path)
    config = {"mode": "openpcdet", "AUG_CONFIG_LIST": [
        {"NAME": "random_world_flip", "ALONG_AXIS_LIST": ["x"], "PROBABILITY": 1},
    ]}
    training = Dataset(str(manifest), data, config, "gaussian", "train")
    validation = Dataset(str(manifest), data, config, "gaussian", "validation")
    # The augmentation changes the supervised peak from y=+2 to y=-2.
    for value, expected_y in [(training, -2), (validation, 2)]:
        target = value[0]
        py, px = np.unravel_index(target["cls"][1].argmax().item(), (16, 16))
        center = np.array([px * .4, py * .4 - 3.2]) + target["offset"][:, py, px].numpy()
        np.testing.assert_allclose(center, [2, expected_y], atol=1e-6)
    np.testing.assert_array_equal(training.get_boxes(0), validation.get_boxes(0))


def test_legacy_one_of_remains_byte_identical_for_a_fixed_seed(tmp_path):
    from core.datasets.dataset import Dataset
    from utils_1.transform import OneOf, Random_Rotation, Random_Scaling, Random_Translation
    root, manifest, data = processed_scene(tmp_path)
    config = {"mode": "one_of", "p": .5, "rotation": {"use": True, "p": 1, "limit_angle": 20},
              "scaling": {"use": True, "p": 1, "range": [.95, 1.05]},
              "translation": {"use": True, "p": 1, "scale": .4}}
    dataset = Dataset(str(manifest), data, config, "gaussian")
    reference = OneOf([Random_Rotation(20, 1), Random_Scaling((.95, 1.05), 1), Random_Translation(.4, 1)], .5)
    points, boxes = scene()
    np.random.seed(42)
    actual = dataset.augment(points.copy(), boxes[:, 1:].copy())
    np.random.seed(42)
    expected = reference(points.copy(), boxes[:, 1:].copy())
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a, b)


def database_builder():
    from build_gt_database import build_database
    return build_database


def gt_recipe(path="gt_database/dbinfos_train.json", quota=1, **extra):
    return {"NAME": "gt_sampling", "DB_INFO_PATH": [path], "NUM_POINT_FEATURES": 4,
            "SAMPLE_GROUPS": [f"Pedestrian:{quota}"], "LIMIT_WHOLE_SCENE": True,
            "PREPARE": {"filter_by_min_points": ["Pedestrian:1"]}, **extra}


def build_small_database(tmp_path):
    root, manifest, data = processed_scene(tmp_path)
    build = database_builder()
    path = build(root, manifest, root / "gt_database", class_names=CLASSES)
    return root, manifest, data, path


def test_gt_database_roundtrip_extracts_only_inside_points_and_preserves_bottom_z(tmp_path):
    root, manifest, data, path = build_small_database(tmp_path)
    metadata = json.loads(path.read_text())
    assert metadata["source_frame_ids"] == ["000000"]
    entry = metadata["db_infos"]["Pedestrian"][0]
    assert entry["num_points_in_gt"] == 1
    np.testing.assert_allclose(entry["box3d_lidar"], [2, 2, 0, 4, 2, 2, 0])
    local = np.fromfile(root / entry["path"], dtype=np.float32).reshape(-1, 4)
    np.testing.assert_allclose(local, [[1, 0, -.5, .7]])
    original_points, original_boxes = scene()
    # The old scene point inside the inserted box is removed before object points
    # are inserted. A point above the box is retained: removal is truly 3D.
    background = np.array([[2, 2, 0, .1], [2, 2, 3, .3], [8, 5, 0, .2]], np.float32)
    aug = augmentor([gt_recipe()], root, allowed_frame_ids=["000000"])
    out_points, out_boxes = aug(background, np.empty((0, 8), np.float32))
    np.testing.assert_allclose(out_boxes, original_boxes)
    np.testing.assert_allclose(out_points[-1], original_points[0])
    assert out_points.shape == (3, 4)
    assert .1 not in out_points[:, 3]


def test_gt_sampler_rejects_collisions_and_obeys_whole_scene_quota(tmp_path):
    root, manifest, data, path = build_small_database(tmp_path)
    points, boxes = scene()
    existing = boxes.copy()
    existing[:, 0] = 0
    actual = augmentor([gt_recipe()], root)(points, existing)
    np.testing.assert_array_equal(actual[0], points)
    np.testing.assert_array_equal(actual[1], existing)
    existing = boxes.copy()
    existing[:, 4] = 20
    actual = augmentor([gt_recipe()], root)(points, existing)
    assert len(actual[1]) == 1  # Existing Pedestrian already meets scene quota.
    actual = augmentor([gt_recipe(LIMIT_WHOLE_SCENE=False)], root)(points, existing)
    assert len(actual[1]) == 2


def test_gt_sampler_filters_point_counts_and_disabled_database_is_not_loaded(tmp_path):
    root, manifest, data, path = build_small_database(tmp_path)
    points, boxes = scene()
    aug = augmentor([gt_recipe(PREPARE={"filter_by_min_points": ["Pedestrian:5"]})], root)
    out_points, out_boxes = aug(points, np.empty((0, 8), np.float32))
    assert len(out_boxes) == 0
    np.testing.assert_array_equal(out_points, points)
    aug = augmentor({"DISABLE_AUG_LIST": ["gt_sampling"], "AUG_CONFIG_LIST": [gt_recipe("missing.json")]}, root)
    np.testing.assert_array_equal(aug(points, boxes)[1], boxes)


def test_database_provenance_prevents_sampling_validation_frames(tmp_path):
    root, manifest, data, path = build_small_database(tmp_path)
    with pytest.raises(ValueError, match="train.*split|training.*split"):
        augmentor([gt_recipe()], root, allowed_frame_ids=["000001"])


def test_builder_rejects_overlapping_train_val_before_writing_objects(tmp_path):
    root, manifest, data = processed_scene(tmp_path)
    build = database_builder()
    with pytest.raises(ValueError, match="overlap"):
        build(root, manifest, root / "gt_database", class_names=CLASSES, val_manifest=manifest)
    assert not (root / "gt_database").exists()


def test_database_builder_uses_only_train_ids(tmp_path):
    root, manifest, data = processed_scene(tmp_path)
    (root / "label/000001.txt").write_text("Cyclist 1.7 .6 1.7 4 0 -1 0\n")
    np.array([[4, 0, 0, .5]], np.float32).tofile(root / "pointcloud/000001.bin")
    val = tmp_path / "val.txt"
    val.write_text("000001;kitti\n")
    path = database_builder()(root, manifest, root / "gt_database", class_names=CLASSES, val_manifest=val)
    metadata = json.loads(path.read_text())
    assert metadata["source_frame_ids"] == ["000000"]
    assert metadata["db_infos"]["Cyclist"] == []
    assert all(entry["image_idx"] == "000000" for values in metadata["db_infos"].values() for entry in values)


def test_gt_sampler_runs_before_global_flip_including_on_empty_training_scene(tmp_path):
    from core.datasets.dataset import Dataset
    root, manifest, data, path = build_small_database(tmp_path)
    (root / "label/000000.txt").write_text("")
    (root / "pointcloud/000000.bin").write_bytes(b"")
    config = {"mode": "openpcdet", "AUG_CONFIG_LIST": [gt_recipe(),
        {"NAME": "random_world_flip", "ALONG_AXIS_LIST": ["x"], "PROBABILITY": 1}]}
    training = Dataset(str(manifest), data, config, "gaussian", "train")
    sample = training[0]
    assert sample["cls"][1].max() == 1
    assert sample["reg_mask"].sum() > 0
    assert sample["voxel"].abs().sum() > 0
    # Validation must not require the database or sample extra objects.
    config["AUG_CONFIG_LIST"][0]["DB_INFO_PATH"] = ["missing.json"]
    validation = Dataset(str(manifest), data, config, "gaussian", "validation")
    assert validation[0]["cls"].sum() == 0


def test_rotated_box_extraction_uses_local_axes(tmp_path):
    root, manifest, data = processed_scene(tmp_path)
    (root / "label/000000.txt").write_text(f"Pedestrian 2 1 4 2 2 -1 {math.pi/2}\n")
    np.array([[2, 3.5, 0, .7], [3.5, 2, 0, .2]], np.float32).tofile(root / "pointcloud/000000.bin")
    path = database_builder()(root, manifest, root / "gt_database", class_names=CLASSES)
    entry = json.loads(path.read_text())["db_infos"]["Pedestrian"][0]
    assert entry["num_points_in_gt"] == 1
    np.testing.assert_allclose(np.fromfile(root / entry["path"], np.float32).reshape(-1, 4), [[0, 1.5, 0, .7]])


def test_sampler_rejects_rotated_intersections_even_with_different_classes(tmp_path):
    root, manifest, data, path = build_small_database(tmp_path)
    points, boxes = scene()
    boxes[0, 0] = 0
    boxes[0, 4:6] = [4, 2]
    boxes[0, 7] = math.pi/4
    out_points, out_boxes = augmentor([gt_recipe()], root)(points, boxes)
    assert len(out_boxes) == 1
    np.testing.assert_array_equal(out_points, points)


def test_sampler_and_dataset_are_serializable_and_work_in_two_workers(tmp_path):
    import pickle
    import torch
    from core.datasets.dataset import Dataset
    root, manifest, data, path = build_small_database(tmp_path)
    (root / "label/000000.txt").write_text("")
    manifest.write_text("000000;kitti\n000000;kitti\n")
    config = {"mode": "openpcdet", "AUG_CONFIG_LIST": [gt_recipe()]}
    dataset = Dataset(str(manifest), data, config, "gaussian", "train")
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored[0]["cls"][1].max() == 1
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2,
                                        generator=torch.Generator().manual_seed(42))
    batches = list(loader)
    assert len(batches) == 2
    assert all(batch["cls"][:, 1].max() == 1 for batch in batches)


@pytest.mark.parametrize("profile,with_gt", [("openpcdet_global", False), ("openpcdet_gt", True)])
def test_profiles_support_real_model_forward_backward(tmp_path, profile, with_gt):
    import torch
    from common import build_model, create_experiment_config
    from core.datasets.dataset import Dataset
    from core.losses.loss_fn import LossFunction
    profile_path = ROOT / "configs/augmentation" / f"{profile}.json"
    assert profile_path.is_file(), "Augmentation profile missing"
    base = json.loads((ROOT / "configs/config.json").read_text())
    config = create_experiment_config(base, json.loads(profile_path.read_text()))
    root, manifest, data = processed_scene(tmp_path)
    if with_gt:
        # The GT profile keeps objects with at least five points.
        np.tile(scene()[0][:1], (6, 1)).tofile(root / "pointcloud/000000.bin")
        database_builder()(root, manifest, root / "gt_database", class_names=CLASSES)
        (root / "label/000000.txt").write_text("")
    config["data"] = data
    dataset = Dataset(str(manifest), data, config["augmentation"], "gaussian", "train")
    torch.manual_seed(42)
    np.random.seed(42)
    batch = torch.utils.data.default_collate([dataset[0], dataset[0]])
    model = build_model(config)
    criterion = LossFunction("gaussian", config["loss"])
    prediction = model(batch["voxel"])
    assert prediction["cls"].shape == batch["cls"].shape
    assert batch["reg_mask"].sum() > 0
    loss = criterion(prediction, batch)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


@pytest.mark.parametrize("with_gt,tag", [(False, "openpcdet_aug"), (True, "openpcdet_gt_aug")])
def test_run_names_identify_ordered_and_gt_augmentation(with_gt, tag):
    from common import generate_run_name
    operations = [{"NAME": "random_world_flip", "ALONG_AXIS_LIST": ["x"]}]
    if with_gt:
        operations.insert(0, gt_recipe())
    config = {"model": {"backbone": "mobilepixornext"},
              "augmentation": {"mode": "openpcdet", "AUG_CONFIG_LIST": operations}}
    assert tag in generate_run_name(config)


def test_notebook_can_select_profiles_and_build_train_database():
    notebook = json.loads((ROOT / "3D_Lidar_Object_Detection_Notebook_standard.ipynb").read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert '"openpcdet_global"' in source and '"openpcdet_gt"' in source
    assert "build_gt_database.py" in source
    assert "--train-split" in source and "--val-split" in source


def test_builder_cli_writes_database(tmp_path):
    import subprocess
    root, manifest, data = processed_scene(tmp_path)
    val = tmp_path / "val.txt"
    val.write_text("000001;kitti\n")
    result = subprocess.run([sys.executable, str(ROOT / "tools/kitti_training_pipeline/build_gt_database.py"),
                             "--processed-root", str(root), "--train-split", str(manifest),
                             "--val-split", str(val)], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert (root / "gt_database/dbinfos_train.json").is_file()


def test_gt_profile_completes_trainer_smoke_and_writes_checkpoint(tmp_path):
    import torch
    from common import create_experiment_config
    from train import main
    root, manifest, data = processed_scene(tmp_path)
    (root / "label/000000.txt").write_text("Pedestrian 1.7 .6 .8 3 0 -1.2 0\n")
    np.tile(np.array([[3, 0, -1, .7]], np.float32), (6, 1)).tofile(root / "pointcloud/000000.bin")
    for identifier in ["000001", "000002"]:
        (root / "label" / f"{identifier}.txt").write_text("")
        np.empty((0, 4), np.float32).tofile(root / "pointcloud" / f"{identifier}.bin")
    manifest.write_text("000000;kitti\n000001;kitti\n")
    val = tmp_path / "val.txt"
    val.write_text("000002;kitti\n")
    database_builder()(root, manifest, root / "gt_database", class_names=CLASSES, val_manifest=val)
    (root / "label/000000.txt").write_text("")
    base = json.loads((ROOT / "configs/config.json").read_text())
    profile = json.loads((ROOT / "configs/augmentation/openpcdet_gt.json").read_text())
    config = create_experiment_config(base, profile)
    config["data"] = data
    config["train"].update(data=str(manifest), epochs=1, warmup_epochs=0, precision="fp32",
                           physical_batch_size=2, accumulation_steps=1, num_workers=0)
    config["val"].update(data=str(val), physical_batch_size=1)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    output_root = tmp_path / "runs"
    main(["--config", str(config_path), "--detector-root", str(ROOT / "detector"),
          "--output-root", str(output_root), "--run-name", "augmentor_smoke",
          "--epochs", "1", "--num-workers", "0", "--device", "cpu", "--precision", "fp32",
          "--target-backend", "python", "--max-train-batches", "1", "--max-val-batches", "1"])
    checkpoint = output_root / "augmentor_smoke/checkpoints/last.pt"
    assert checkpoint.is_file()
    saved = torch.load(checkpoint, map_location="cpu")
    assert saved["epoch"] == 1
    assert saved["config"]["augmentation"]["mode"] == "openpcdet"
