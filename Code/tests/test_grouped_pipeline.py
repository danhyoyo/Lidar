"""Backend parity and real collation/worker contracts for grouped targets."""

import copy
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Subset

from test_grouped_targets import (
    Dataset, GEOMETRY, GROUPS, OBJECTS, make_dataset, resolve_task_groups, target_dataset,
)


@pytest.mark.parametrize("classification", ["gaussian", "binary"])
@pytest.mark.parametrize("assignment", ["nearest_center", "legacy"])
@pytest.mark.parametrize("reordered", [False, True])
def test_grouped_backends_match_bitwise_for_empty_boundary_overlap_and_random_boxes(classification, assignment, reordered):
    definitions = ([{"name": "vru", "classes": ["Cyclist", "Pedestrian"]}, GROUPS[0]]
                   if reordered else GROUPS)
    groups = resolve_task_groups(definitions, OBJECTS)
    rng = np.random.default_rng(42)
    values = np.zeros((25, 8), np.float32)
    values[:, 0] = rng.integers(0, 3, len(values))
    values[:, 1:4] = rng.uniform([1.4, .5, .7], [2., 1.6, 3.5], (len(values), 3))
    values[:, 4:6] = rng.uniform([.001, -3.199], [9.599, 3.199], (len(values), 2))
    values[:, 6], values[:, 7] = -1.7, rng.uniform(-1.5, 1.5, len(values))
    values[:4, 4:6] = [[.001, -3.199], [9.599, 3.199], [2.15, .05], [2.15, .05]]
    for source in (torch.empty(0, 8), torch.from_numpy(values), torch.from_numpy(values[::-1].copy())):
        results = []
        for backend in ("python", "numba"):
            dataset = target_dataset(classification, backend, assignment)
            dataset.task_groups = groups
            results.append(dataset.get_label(source, GEOMETRY)["groups"])
        for group in groups:
            for key in results[0][group.name]:
                first, second = results[0][group.name][key], results[1][group.name][key]
                assert first.dtype == second.dtype and first.shape == second.shape
                assert first.numpy().tobytes() == second.numpy().tobytes(), (group.name, key)


def car_only_dataset(dataset):
    root = Path(dataset.config["kitti"]["location"])
    (root / "label/000000.txt").write_text("Car 1.5 1.6 3.5 2.15 0.05 -1.7 0.2\n")
    augmentation = {"p": 0., "rotation": {"use": False}, "scaling": {"use": False},
                    "translation": {"use": False}}
    return Dataset(dataset.data_file, copy.deepcopy(dataset.config), augmentation,
                   dataset.cls_encoding, "validation", dataset.target_backend)


def assert_grouped_batch(batch, classification, batch_size):
    assert set(batch) == {"voxel", "groups"}
    assert batch["voxel"].shape == (batch_size, 14, 64, 48)
    assert tuple(batch["groups"]) == ("car", "ped_cyc")
    for name, channels in (("car", 1), ("ped_cyc", 2)):
        target = batch["groups"][name]
        cls_shape = ((batch_size, 16, 12) if classification == "binary"
                     else (batch_size, channels, 16, 12))
        assert target["cls"].shape == cls_shape
        assert target["cls"].dtype == (torch.int64 if classification == "binary" else torch.float32)
        for key in ("offset", "size", "yaw"):
            assert target[key].shape == (batch_size, 2, 16, 12)
        assert target["reg_mask"].shape == (batch_size, 16, 12)
        assert all(torch.isfinite(t).all() for t in target.values())
    assert batch["groups"]["car"]["reg_mask"].data_ptr() != batch["groups"]["ped_cyc"]["reg_mask"].data_ptr()


@pytest.mark.parametrize("classification", ["gaussian", "binary"])
@pytest.mark.parametrize("empty_vru", [False, True])
def test_default_collation_retains_nested_maps_and_negative_only_groups(tmp_path, classification, empty_vru):
    dataset = make_dataset(tmp_path, classification=classification)
    if empty_vru:
        dataset = car_only_dataset(dataset)
    batch = next(iter(DataLoader(Subset(dataset, [0, 0]), batch_size=2, num_workers=0)))
    assert_grouped_batch(batch, classification, 2)
    if empty_vru:
        assert all(not tensor.any() for tensor in batch["groups"]["ped_cyc"].values())


@pytest.mark.parametrize("classification", ["gaussian", "binary"])
def test_legacy_default_collation_remains_flat(tmp_path, classification):
    dataset = make_dataset(tmp_path, classification=classification, grouped=False)
    batch = next(iter(DataLoader(Subset(dataset, [0, 0]), batch_size=2, num_workers=0)))
    assert set(batch) == {"voxel", "cls", "offset", "size", "yaw", "reg_mask"}
    assert batch["cls"].shape == ((2, 16, 12) if classification == "binary" else (2, 3, 16, 12))


def test_validation_and_test_metadata_stay_outside_group_tensor_maps(tmp_path):
    validation = make_dataset(tmp_path, task="val")[0]
    assert tuple(int(i) for i in validation["cls_list"]) == (0, 1, 2)
    assert isinstance(validation["points"], np.ndarray)
    assert len(validation["boxes"]) == 3
    assert all(set(t) == {"cls", "offset", "size", "yaw", "reg_mask"}
               for t in validation["groups"].values())
    testing = make_dataset(tmp_path, task="test")[0]
    assert set(testing) == {"voxel", "points", "dtype"}


@pytest.mark.parametrize("classification", ["gaussian", "binary"])
@pytest.mark.parametrize("backend", ["python", "numba"])
def test_grouped_worker_loader_after_parent_preview_and_pickle_roundtrip(tmp_path, classification, backend):
    dataset = make_dataset(tmp_path, classification=classification, backend=backend)
    expected = dataset[0]  # Warm Numba in the parent before worker creation.
    dataset = pickle.loads(pickle.dumps(dataset))
    assert dataset.task_groups == resolve_task_groups(GROUPS, OBJECTS)
    loader = DataLoader(Subset(dataset, [0, 0]), batch_size=1, num_workers=2,
                        multiprocessing_context="spawn", timeout=30)
    batches = list(loader)
    assert len(batches) == 2
    for batch in batches:
        assert_grouped_batch(batch, classification, 1)
        torch.testing.assert_close(batch["voxel"][0], expected["voxel"], rtol=0, atol=0)
        for name, target in batch["groups"].items():
            for key, tensor in target.items():
                torch.testing.assert_close(tensor[0], expected["groups"][name][key], rtol=0, atol=0)


@pytest.mark.parametrize("local", ["none", "eca", "simam"])
@pytest.mark.parametrize("backend", ["python", "numba"])
@pytest.mark.parametrize("scene", ["empty", "car_only", "same_cell"])
def test_assembled_hybrid_hist14_focal_training_resume_and_decode(tmp_path, local, backend, scene):
    from test_hybrid_augmentation import database, recipe
    from test_grouped_checkpoint import components, payload, restore
    from test_grouped_loss_state import snapshot, assert_state_equal
    from common import read_json
    from postprocess import filter_pred
    import train

    train.seed_everything(42)
    root, source, points = database(tmp_path)
    np.empty((0, 4), np.float32).tofile(root / "pointcloud/000000.bin")
    (root / "label/000000.txt").write_text(
        "Car " + " ".join(map(str, source[1:])) + "\nPedestrian .8 .4 .6 15 2 -1.6 -.2\n"
        if scene == "same_cell" else "")
    if scene == "same_cell":
        points.tofile(root / "pointcloud/000000.bin")
    manifest = tmp_path / "host.txt"
    manifest.write_text("000000;kitti\n000001;kitti\n")
    recipe_path = Path(__file__).resolve().parents[1] / "configs/experiments/under1m"
    filename = ("hist14_local3_focal_grouped_oga_iqa.json" if local == "none" else
                f"hist14_local3_focal_{local}_grouped_oga_iqa.json")
    config = read_json(recipe_path / filename)
    geometry = dict(x_min=0., x_max=32., x_res=.5, y_min=-8., y_max=8., y_res=.5,
                    z_min=-2.5, z_max=1., z_res=.1)
    config["data"]["kitti"].update(location=str(root), geometry=geometry)
    config["data"].update(min_radius=1)
    config["data"]["bev_encoding"]["backend"] = "numpy" if backend == "python" else "numba"
    # Existing overlapping labels characterize group ownership; hybrid pasting
    # remains collision-safe and is exercised separately by the car-only scene.
    config["augmentation"] = {"mode": "openpcdet", "AUG_CONFIG_LIST": [
        recipe(PROBABILITY=1. if scene == "car_only" else 0.),
        {"NAME": "random_world_scaling", "WORLD_SCALE_RANGE": [1.25, 1.25]}]}
    config["train"].update(data=str(manifest), precision="fp32", num_workers=0, epochs=4,
                           physical_batch_size=1, accumulation_steps=1, warmup_epochs=0)
    dataset = Dataset(str(manifest), config["data"], config["augmentation"], "gaussian", "train", backend)
    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))
    assert batch["voxel"].shape == (1, 14, 32, 64)
    car, vru = batch["groups"]["car"], batch["groups"]["ped_cyc"]
    if scene == "empty":
        assert not batch["voxel"].any() and not car["reg_mask"].any() and not vru["reg_mask"].any()
    else:
        assert batch["voxel"][:, 11].sum() > 0 and car["reg_mask"].any()
    if scene == "same_cell":
        shared = car["reg_mask"].bool() & vru["reg_mask"].bool()
        assert shared.any()
        assert not torch.equal(car["size"][:, :, shared[0]], vru["size"][:, :, shared[0]])
    else:
        assert not vru["reg_mask"].any()

    parts = components(config)
    model, criterion, optimizer, scheduler, _, _ = parts
    before = model.grouped_header.heads["car"].cls.head.weight.detach().clone()
    train.set_loss_epoch(criterion, 0)
    optimizer.zero_grad(set_to_none=True)
    losses = criterion(model(batch["voxel"]), batch)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for module in (model, criterion)
               for p in module.parameters() if p.grad is not None)
    optimizer.step()
    scheduler.step()
    assert not torch.equal(before, model.grouped_header.heads["car"].cls.head.weight)
    checkpoint = tmp_path / "assembled.pt"
    torch.save(payload(config, parts, epoch=1), checkpoint)
    restored = components(config)
    result = restore(torch.load(checkpoint, weights_only=True), config, restored)
    assert result["epoch"] == 1 and result["mode"] == "resume"
    assert_state_equal(restored[0], snapshot(model))
    assert_state_equal(restored[1], snapshot(criterion))
    model.eval()
    restored[0].eval()
    with torch.no_grad():
        prediction = restored[0](batch["voxel"])
        expected = model(batch["voxel"])
    for name in prediction["groups"]:
        for key, value in prediction["groups"][name].items():
            torch.testing.assert_close(value, expected["groups"][name][key], rtol=0, atol=0)
    boxes = filter_pred(prediction, config["data"]["kitti"], 4, .05, .1,
        task_groups=dataset.task_groups, use_iou=True, max_detections=7)
    assert boxes.dtype == np.float32 and boxes.ndim == 2 and boxes.shape[1] == 7
    assert len(boxes) <= 7 and np.isfinite(boxes).all()
    assert set(boxes[:, 0]).issubset({0., 1., 2.})
