"""Geometry, provenance, transaction and worker contracts for hybrid GT paste."""

import importlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "detector"), str(ROOT / "tools/kitti_training_pipeline")]
CLASSES = {"Car": 0, "Pedestrian": 1, "Cyclist": 2}


def hybrid_module(name):
    path = ROOT / "detector/core/datasets/augmentor" / f"{name}.py"
    assert path.is_file(), f"Missing hybrid implementation: {name}"
    return importlib.import_module(f"core.datasets.augmentor.{name}")


def box(x=10, y=0, yaw=0, cls=0, h=2, w=2, length=4, z=-1.6):
    return np.array([cls, h, w, length, x, y, z, yaw], np.float32)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_collision_matrix_matches_shapely_randomized(backend):
    geometry = hybrid_module("hybrid_geometry")
    from core.datasets.augmentor.geometry import bev_polygon
    rng = np.random.default_rng(21)
    boxes = np.array([box(*rng.uniform(-10, 10, 2), yaw=rng.uniform(-np.pi, np.pi),
                          w=rng.uniform(.3, 3), length=rng.uniform(.5, 6)) for _ in range(80)])
    expected = np.array([[bev_polygon(a).intersection(bev_polygon(b)).area > 1e-9
                          for b in boxes[35:]] for a in boxes[:35]])
    np.testing.assert_array_equal(geometry.collision_matrix(boxes[:35], boxes[35:], backend=backend), expected)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_collision_touching_margin_and_empty_sets(backend):
    geometry = hybrid_module("hybrid_geometry")
    a = box(x=0, length=2)[None]
    b = box(x=2, length=2)[None]
    assert not geometry.collision_matrix(a, b, backend=backend)[0, 0]
    assert geometry.collision_matrix(a, b, margin=.1, backend=backend)[0, 0]
    assert geometry.collision_matrix(a, np.empty((0, 8)), backend=backend).shape == (1, 0)


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_union_point_membership_matches_existing_geometry(backend):
    geometry = hybrid_module("hybrid_geometry")
    from core.datasets.augmentor.geometry import points_in_box
    rng = np.random.default_rng(2)
    points = rng.uniform([-5, -5, -3, 0], [20, 8, 3, 1], (5000, 4)).astype(np.float32)
    boxes = np.stack([box(x=3, yaw=.4), box(x=10, y=2, yaw=-.7)])
    extra = np.array([.2, .1, .15])
    expected = np.logical_or.reduce([points_in_box(points, b, extra) for b in boxes])
    np.testing.assert_array_equal(geometry.points_in_boxes_union(points, boxes, extra, backend=backend), expected)
    assert not geometry.points_in_boxes_union(points, np.empty((0, 8)), backend=backend).any()


@pytest.mark.parametrize("backend", ["numpy", "numba"])
def test_ray_intersections_parallel_surface_and_shadow(backend):
    geometry = hybrid_module("hybrid_geometry")
    # Box extends X=[4,6], Y=[-1,1], Z=[-1,1]. Ray parameter t=1 at point.
    candidate = box(x=5, length=2, z=-1)
    points = np.array([[3, 0, 0, 0], [4, 0, 0, 0], [5, 0, 0, 0],
                       [6, 0, 0, 0], [8, 0, 0, 0], [0, 0, 0, 0],
                       [8, 4, 0, 0], [8, 0, 3, 0]], np.float32)
    hit, enter, leave = geometry.ray_box_intervals(points, candidate, backend=backend)
    np.testing.assert_array_equal(hit, [True, True, True, True, True, False, False, False])
    np.testing.assert_allclose(enter[:5], 4 / points[:5, 0])
    np.testing.assert_allclose(leave[:5], 6 / points[:5, 0])
    assert np.isinf(enter[5:]).all()


def database(tmp_path):
    from build_gt_database import build_database
    root = tmp_path / "processed"
    (root / "pointcloud").mkdir(parents=True)
    (root / "label").mkdir()
    source = box(x=15, y=2, yaw=.35)
    rng = np.random.default_rng(5)
    local = rng.uniform([-1.8, -.8, .1], [1.8, .8, 1.9], (40, 3))
    c, s = np.cos(source[7]), np.sin(source[7])
    local[:, :2] = local[:, :2] @ np.array([[c, -s], [s, c]]).T
    points = np.column_stack([local + source[4:7], np.full(40, .6)]).astype(np.float32)
    points.tofile(root / "pointcloud/000001.bin")
    (root / "label/000001.txt").write_text("Car " + " ".join(map(str, source[1:])) + "\n")
    manifest = tmp_path / "train.txt"
    manifest.write_text("000001;kitti\n")
    build_database(root, manifest, root / "gt_database")
    return root, source, points


def recipe(**overrides):
    defaults = dict(
        NAME="hybrid_gt_sampling", DB_INFO_PATH=["gt_database/dbinfos_train.json"],
        NUM_POINT_FEATURES=4, SAMPLE_GROUPS=["Car:1"], LIMIT_WHOLE_SCENE=False,
        PREPARE={"filter_by_min_points": ["Car:5"]}, REMOVE_EXTRA_WIDTH=[0, 0, 0],
        GEOMETRY_BACKEND="numba", CACHE_SIZE_MB=1,
        PLACEMENT_MODE="source", MAX_PLACEMENT_ATTEMPTS=1, CANDIDATE_MULTIPLIER=1,
        ENABLE_GROUND_VALIDATION=False, ENABLE_STATIC_COLLISION=False,
        ENABLE_LINE_OF_SIGHT=False, ENABLE_SHADOW_MASKING=False,
        ENABLE_DENSITY_SUBSAMPLE=False, ENABLE_RADIOMETRIC_CALIBRATION=False,
        ENABLE_VISIBILITY_PROTECTION=True,
    )
    return defaults | overrides


def sampler(tmp_path, config=None):
    module = hybrid_module("hybrid_sampler")
    root, source, object_points = database(tmp_path)
    value = module.HybridDataBaseSampler(root, config or recipe(), CLASSES,
        geometry={"x_min": 0, "x_max": 70.4, "y_min": -40, "y_max": 40},
        allowed_frame_ids=["000001"])
    return value, root, source, object_points


def test_source_mode_preserves_box_points_and_scene_inputs(tmp_path):
    value, root, source, object_points = sampler(tmp_path)
    background = np.array([[15, 2, -.5, .1], [40, 0, -1, .2]], np.float32)
    original = background.copy()
    points, boxes, meta = value(background, np.empty((0, 8), np.float32), np.random.default_rng(8), return_metadata=True)
    np.testing.assert_array_equal(background, original)
    np.testing.assert_allclose(boxes[0], source, atol=2e-6)
    np.testing.assert_allclose(points[1:], object_points, atol=2e-6)
    assert meta["point_is_sampled"].sum() == 40
    assert meta["box_is_sampled"].tolist() == [True]
    assert meta["removed_interior_points"] == 1
    assert meta["accepted_by_class"]["Car"] == 1


def test_collision_rejection_is_transactional(tmp_path):
    value, root, source, object_points = sampler(tmp_path)
    points, boxes, meta = value(object_points, source[None], np.random.default_rng(3), return_metadata=True)
    np.testing.assert_array_equal(points, object_points)
    np.testing.assert_array_equal(boxes, source[None])
    assert meta["num_inserted"] == 0
    assert meta["rejected_by_reason"]["box_collision"] == 1


def test_source_mode_matches_existing_sampler_without_optional_physics(tmp_path):
    from core.datasets.augmentor.database_sampler import DataBaseSampler
    value, root, source, object_points = sampler(tmp_path, recipe(
        COLLISION_MARGIN=0, ENABLE_VISIBILITY_PROTECTION=False))
    old = DataBaseSampler(root, recipe(), CLASSES, geometry=value.geometry,
                          allowed_frame_ids=["000001"])
    rng = np.random.default_rng(9)
    points = rng.uniform([0, -10, -2, 0], [40, 10, 2, 1], (2000, 4)).astype(np.float32)
    boxes = box(x=35)[None]
    old_points, old_boxes = old(points, boxes, np.random.default_rng(7))
    new_points, new_boxes = value(points, boxes, np.random.default_rng(7))
    np.testing.assert_allclose(new_points, old_points, atol=2e-6)
    np.testing.assert_allclose(new_boxes, old_boxes, atol=2e-6)


def test_visibility_protection_covers_shared_boundary_points(tmp_path):
    value, _, source, _ = sampler(tmp_path, recipe(COLLISION_MARGIN=0))
    # Create an axis-aligned candidate and an existing box that just touches it.
    value.db_infos["Car"][0]["box3d_lidar"][6] = 0
    original = box(x=19, y=2, yaw=0)
    points = np.array([[17, 2, -1, .5]] * 6, np.float32)
    out, boxes, meta = value(points, original[None], np.random.default_rng(3), return_metadata=True)
    assert meta["num_inserted"] == 0
    assert meta["rejected_by_reason"]["visibility"] == 1
    np.testing.assert_array_equal(out, points)
    np.testing.assert_array_equal(boxes, original[None])


def test_cache_hits_limits_and_worker_serialization(tmp_path):
    module = hybrid_module("hybrid_sampler")
    path = tmp_path / "object.bin"
    np.arange(16, dtype=np.float32).tofile(path)
    cache = module.ObjectPointCache(max_bytes=64)
    first = cache.load(path, 4, 4)
    second = cache.load(path, 4, 4)
    assert first is second and not first.flags.writeable
    assert cache.stats()["hits"] == 1
    other = tmp_path / "other.bin"
    np.arange(16, dtype=np.float32).tofile(other)
    cache.load(other, 4, 4)
    assert cache.stats()["bytes"] <= 64 and cache.stats()["entries"] == 1
    restored = pickle.loads(pickle.dumps(cache))
    assert restored.stats()["entries"] == 0


def test_database_train_provenance_is_still_enforced(tmp_path):
    module = hybrid_module("hybrid_sampler")
    root, _, _ = database(tmp_path)
    with pytest.raises(ValueError, match="training split"):
        module.HybridDataBaseSampler(root, recipe(), CLASSES, allowed_frame_ids=["000002"])


def test_source_relative_rotates_observed_side_and_snaps_bottom_to_ground(tmp_path):
    value, root, source, object_points = sampler(tmp_path, recipe(
        PLACEMENT_MODE="source_relative", RANGE_SCALE=[1, 1], AZIMUTH_JITTER_DEG=0,
        ENABLE_GROUND_VALIDATION=True,
    ))
    road = np.array([[15 + dx, 2 + dy, -1.9, .1] for dx in (-1, 0, 1) for dy in (-1, 0, 1)], np.float32)
    out, boxes = value(road, np.empty((0, 8), np.float32), np.random.default_rng(4))
    assert len(boxes) == 1 and boxes[0, 6] == pytest.approx(-1.9)
    np.testing.assert_allclose(out[-40:, :2], object_points[:, :2], atol=2e-6)
    np.testing.assert_allclose(out[-40:, 2], object_points[:, 2] - .3, atol=2e-6)


def test_missing_ground_rejects_candidate_without_clearing_scene(tmp_path):
    value, _, _, _ = sampler(tmp_path, recipe(ENABLE_GROUND_VALIDATION=True))
    background = np.array([[40, 0, -1.6, .2]], np.float32)
    out, boxes, meta = value(background, np.empty((0, 8), np.float32), np.random.default_rng(4), return_metadata=True)
    np.testing.assert_array_equal(out, background)
    assert len(boxes) == 0 and meta["rejected_by_reason"]["ground"] == 1
    assert meta["cache"]["disk_reads"] == 0


def test_static_collision_checks_unlabelled_obstacles(tmp_path):
    value, _, source, _ = sampler(tmp_path, recipe(ENABLE_STATIC_COLLISION=True))
    obstacle = np.tile([source[4], source[5], -.5, .2], (5, 1)).astype(np.float32)
    out, boxes, meta = value(obstacle, np.empty((0, 8), np.float32), np.random.default_rng(4), return_metadata=True)
    assert len(boxes) == 0 and meta["rejected_by_reason"]["static_collision"] == 1
    np.testing.assert_array_equal(out, obstacle)


def test_shadow_clearing_protects_existing_object_visibility(tmp_path):
    value, _, source, _ = sampler(tmp_path, recipe(ENABLE_SHADOW_MASKING=True))
    # Existing object is farther away on the same sensor ray and does not overlap in BEV.
    old = source.copy()
    old[4:6] *= 2
    old[6] = -1.6
    points = np.tile([old[4], old[5], -.2, .4], (10, 1)).astype(np.float32)
    out, boxes, meta = value(points, old[None], np.random.default_rng(4), return_metadata=True)
    np.testing.assert_array_equal(out, points)
    np.testing.assert_array_equal(boxes, old[None])
    assert meta["rejected_by_reason"]["visibility"] == 1
    assert meta["removed_shadow_points"] == 0


def test_shadow_metadata_marks_only_surviving_inserted_points(tmp_path):
    value, _, source, _ = sampler(tmp_path, recipe(ENABLE_SHADOW_MASKING=True))
    points = np.array([[30, 4, -.2, .1], [40, -20, -1.6, .2]], np.float32)
    out, boxes, meta = value(points, np.empty((0, 8), np.float32), np.random.default_rng(4), return_metadata=True)
    assert len(boxes) == 1 and meta["removed_shadow_points"] == 1
    assert len(out) == 41 and meta["point_is_sampled"].sum() == 40
    np.testing.assert_array_equal(out[0], points[1])


def test_shadow_clearing_protects_an_already_inserted_object(tmp_path):
    import copy
    value, _, _, _ = sampler(tmp_path, recipe(ENABLE_SHADOW_MASKING=True, SAMPLE_GROUPS=["Car:2"]))
    near = value.db_infos["Car"][0]
    far = copy.deepcopy(near)
    far["box3d_lidar"][0] *= 2
    far["box3d_lidar"][1] *= 2
    far["gt_idx"] = 1
    value.db_infos["Car"] = [far, near]
    value.orders["Car"] = np.array([0, 1])
    out, boxes, meta = value(np.empty((0, 4), np.float32), np.empty((0, 8), np.float32),
                             np.random.default_rng(4), return_metadata=True)
    assert len(boxes) == 1 and len(out) == 40
    assert boxes[0, 4] == pytest.approx(far["box3d_lidar"][0])
    assert meta["rejected_by_reason"]["visibility"] == 1
    assert meta["point_is_sampled"].all()
    assert meta["removed_shadow_points"] == 0


def test_line_of_sight_rejects_foreground_blockers(tmp_path):
    value, _, source, _ = sampler(tmp_path, recipe(ENABLE_LINE_OF_SIGHT=True))
    points = np.tile([8, 16/15, -.2, .2], (6, 1)).astype(np.float32)
    out, boxes, meta = value(points, np.empty((0, 8), np.float32), np.random.default_rng(4), return_metadata=True)
    assert len(boxes) == 0 and meta["rejected_by_reason"]["line_of_sight"] == 1
    np.testing.assert_array_equal(out, points)


def test_density_intensity_and_pose_change_are_deterministic(tmp_path):
    value, _, source, original = sampler(tmp_path, recipe(
        PLACEMENT_MODE="source_relative", RANGE_SCALE=[1.8, 1.8], AZIMUTH_JITTER_DEG=5,
        ENABLE_DENSITY_SUBSAMPLE=True, ENABLE_RADIOMETRIC_CALIBRATION=True,
    ))
    background = np.empty((0, 4), np.float32)
    a, boxes = value(background, np.empty((0, 8), np.float32), np.random.default_rng(12))
    value.orders = {name: None for name in value.orders}
    value.pointers = {name: 0 for name in value.pointers}
    b, boxes_b = value(background, np.empty((0, 8), np.float32), np.random.default_rng(12))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(boxes, boxes_b)
    assert 5 <= len(a) < len(original)
    assert np.all(a[:, 3] < original[0, 3])
    assert np.hypot(*boxes[0, 4:6]) == pytest.approx(np.hypot(*source[4:6]) * 1.8)


@pytest.mark.parametrize("overrides,key", [
    ({"MAX_PLACEMENT_ATTEMPTS": 0}, "MAX_PLACEMENT_ATTEMPTS"),
    ({"CACHE_SIZE_MB": -1}, "CACHE_SIZE_MB"),
    ({"RANGE_SCALE": [1.2, .8]}, "RANGE_SCALE"),
    ({"PLACEMENT_MODE": "typo"}, "PLACEMENT_MODE"),
    ({"MIN_VISIBLE_RATIO": 1.1}, "MIN_VISIBLE_RATIO"),
    ({"ENABLE_SHADOW_MASKING": "yes"}, "ENABLE_SHADOW_MASKING"),
])
def test_bad_configuration_is_rejected(tmp_path, overrides, key):
    with pytest.raises(ValueError, match=key):
        sampler(tmp_path, recipe(**overrides))


def test_cache_detects_changed_file_without_exceeding_budget(tmp_path):
    import os
    module = hybrid_module("hybrid_sampler")
    path = tmp_path / "object.bin"
    np.ones(16, np.float32).tofile(path)
    cache = module.ObjectPointCache(max_bytes=64)
    first = cache.load(path, 4, 4)
    np.full(16, 2, np.float32).tofile(path)
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
    second = cache.load(path, 4, 4)
    assert np.all(first == 1) and np.all(second == 2)
    assert cache.stats()["bytes"] == 64 and cache.stats()["entries"] == 1


def test_whole_scene_quota_exits_without_loading_object_points(tmp_path):
    value, _, source, points = sampler(tmp_path, recipe(LIMIT_WHOLE_SCENE=True))
    out, boxes, meta = value(points, source[None], np.random.default_rng(0), return_metadata=True)
    np.testing.assert_array_equal(out, points)
    assert meta["attempted_by_class"]["Car"] == 0
    assert meta["cache"]["disk_reads"] == 0


def test_hybrid_profile_is_registered_in_notebook_and_ordered_augmentor(tmp_path):
    profile_path = ROOT / "configs/augmentation/hybrid_gt.json"
    assert profile_path.is_file(), "Missing hybrid augmentation profile"
    from notebook_config import resolve_notebook_config
    from common import generate_run_name
    cfg = resolve_notebook_config(ROOT, preset="config", augmentation="hybrid_gt")
    assert cfg["augmentation"]["AUG_CONFIG_LIST"][0]["NAME"] == "hybrid_gt_sampling"
    assert "hybrid_gt_aug" in generate_run_name(cfg, seed=42)
    from core.datasets.augmentor.data_augmentor import DataAugmentor
    root, source, points = database(tmp_path)
    value = DataAugmentor(root, {"AUG_CONFIG_LIST": [recipe()]}, CLASSES,
                          rng=np.random.default_rng(42), allowed_frame_ids=["000001"])
    _, boxes = value(np.empty((0, 4), np.float32), np.empty((0, 8), np.float32))
    np.testing.assert_allclose(boxes[0], source, atol=2e-6)


def training_scene(tmp_path):
    root, source, points = database(tmp_path)
    road = np.array([[source[4]+x, source[5]+y, -1.6, .1]
                     for x in np.linspace(-3, 3, 15) for y in np.linspace(-3, 3, 15)], np.float32)
    for identifier in ("000000", "000002"):
        road.tofile(root / "pointcloud" / f"{identifier}.bin")
        (root / "label" / f"{identifier}.txt").write_text("")
    manifest = tmp_path / "train.txt"
    manifest.write_text("000000;kitti\n000001;kitti\n")
    val = tmp_path / "val.txt"
    val.write_text("000002;kitti\n")
    base = json.loads((ROOT / "configs/config.json").read_text())
    base.update(json.loads((ROOT / "configs/augmentation/hybrid_gt.json").read_text()))
    base["data"]["kitti"]["location"] = str(root)
    base["data"]["kitti"]["geometry"].update(x_max=25.6, y_min=-12.8, y_max=12.8, x_res=.4, y_res=.4)
    base["train"].update(data=str(manifest), epochs=1, warmup_epochs=0, precision="fp32",
                         physical_batch_size=2, accumulation_steps=1, num_workers=0)
    base["val"].update(data=str(val), physical_batch_size=1)
    # Ensure the smoke exercises sampling every batch rather than a probability skip.
    base["augmentation"]["AUG_CONFIG_LIST"][0]["PROBABILITY"] = 1
    return root, manifest, val, base


def test_hybrid_pipeline_model_backward_and_trainer_smoke(tmp_path):
    assert (ROOT / "configs/augmentation/hybrid_gt.json").is_file()
    import torch
    from common import build_model, configure_detector_imports
    from train import main
    from core.datasets.dataset import Dataset
    configure_detector_imports(ROOT / "detector")
    root, manifest, val, cfg = training_scene(tmp_path)
    dataset = Dataset(str(manifest), cfg["data"], cfg["augmentation"], "gaussian", target_backend="numba")
    np.random.seed(42)
    sample = dataset[0]
    assert sample["reg_mask"].sum() > 0
    model = build_model(cfg)
    pred = model(sample["voxel"][None].expand(2, -1, -1, -1))
    sum(t.square().mean() for t in pred.values()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))
    main(["--config", str(config_path), "--detector-root", str(ROOT / "detector"),
          "--output-root", str(tmp_path / "runs"), "--run-name", "hybrid_smoke", "--device", "cpu",
          "--num-workers", "0", "--target-backend", "numba", "--max-train-batches", "1", "--max-val-batches", "1"])
    checkpoint = tmp_path / "runs/hybrid_smoke/checkpoints/last.pt"
    assert checkpoint.is_file()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["epoch"] == 1
    assert state["config"]["augmentation"]["AUG_CONFIG_LIST"][0]["NAME"] == "hybrid_gt_sampling"


def test_zero_probability_hybrid_does_not_open_database(tmp_path):
    from core.datasets.augmentor.data_augmentor import DataAugmentor
    value = DataAugmentor(tmp_path, {"AUG_CONFIG_LIST": [recipe(PROBABILITY=0)]}, CLASSES)
    points, boxes = value(np.empty((0, 4), np.float32), np.empty((0, 8), np.float32))
    assert points.shape == (0, 4) and boxes.shape == (0, 8)


def test_hybrid_preview_preserves_inserted_point_provenance_and_renders(tmp_path):
    assert (ROOT / "tools/visualization/visualize_gt_sampling.py").is_file(), "Missing hybrid preview"
    import importlib.util
    spec = importlib.util.spec_from_file_location("hybrid_preview", ROOT / "tools/visualization/visualize_gt_sampling.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root, _, _, cfg = training_scene(tmp_path)
    preview = module.preview_sampling(cfg, ROOT, frame_index=0, seed=42)
    assert len(preview["stages"]) == 3
    sampled = preview["stages"][1]
    assert sampled["point_is_sampled"].sum() >= 5
    assert sampled["box_is_sampled"].sum() == 1
    assert preview["stages"][2]["point_is_sampled"].shape == sampled["point_is_sampled"].shape
    figure = module.render_preview(preview, cfg)
    figure.savefig(tmp_path / "preview.png")
    assert (tmp_path / "preview.png").stat().st_size > 1000


def test_worker_pickle_resets_cache_and_sampling_cursor(tmp_path):
    value, _, _, _ = sampler(tmp_path)
    value(np.empty((0, 4), np.float32), np.empty((0, 8), np.float32), np.random.default_rng(3))
    assert value.cache.stats()["entries"] == 1
    restored = pickle.loads(pickle.dumps(value))
    assert restored.cache.stats()["entries"] == 0
    assert all(order is None for order in restored.orders.values())
    assert all(pointer == 0 for pointer in restored.pointers.values())


def test_two_worker_loader_handles_hybrid_after_parent_preview(tmp_path):
    import torch
    from common import configure_detector_imports
    configure_detector_imports(ROOT / "detector")
    from core.datasets.dataset import Dataset
    root, manifest, _, cfg = training_scene(tmp_path)
    dataset = Dataset(str(manifest), cfg["data"], cfg["augmentation"], "gaussian", target_backend="numba")
    np.random.seed(42)
    dataset[0]  # Warm the parent cache before worker creation.
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2,
                                        multiprocessing_context="spawn", timeout=30)
    batches = list(loader)
    assert len(batches) == 2
    assert all(torch.isfinite(batch["voxel"]).all() for batch in batches)


def test_cache_and_sampler_reset_after_process_ownership_change(tmp_path):
    value, _, _, _ = sampler(tmp_path)
    value(np.empty((0, 4), np.float32), np.empty((0, 8), np.float32), np.random.default_rng(3))
    assert value.cache.stats()["entries"] == 1
    # An impossible old PID exercises the same guard used after fork.
    value._owner_pid = -1
    value.cache._owner_pid = -1
    value(np.empty((0, 4), np.float32), np.empty((0, 8), np.float32), np.random.default_rng(3))
    assert value.cache.stats()["disk_reads"] == 1
    assert value.cache.stats()["hits"] == 0
