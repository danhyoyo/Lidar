"""Complete BEV recipes must rebuild the manifest contract and actual budget."""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "detector"), str(ROOT / "tools/kitti_training_pipeline")]

from common import (bev_encoding_spec, checkpoint_identity, create_experiment_config,
                    detection_spec, generate_run_name, read_json)
import notebook_config as notebook
from tools.benchmarks.profile_detector import profile_detector

MANIFEST = read_json(ROOT / "docs/plans/lightweight_lidar_backbone_2026/experiment_manifest.json")
BEV_VARIANTS = [variant for variant in MANIFEST["variants"]
                if variant["overrides"].get("data", {}).get("box_mode", "bev") == "bev"
                and variant["overrides"].get("model", {}).get("head_mode") != "legacy_single"]


@pytest.mark.parametrize("variant", BEV_VARIANTS, ids=lambda variant: variant["id"])
def test_manifest_bev_recipe_is_complete_rebuildable_and_matches_actual_counts(variant):
    path = ROOT / variant["planned_config"]
    assert path.is_file(), f"Complete recipe missing: {path}"
    saved = read_json(path)
    resolved = notebook.resolve_under1m_recipe(ROOT, variant["id"])
    assert saved == resolved
    assert checkpoint_identity(saved) == checkpoint_identity(json.loads(json.dumps(saved)))
    schema = bev_encoding_spec(saved)
    assert schema.channels == 14 and schema.backend == "numpy"
    assert schema.input_shape == (1, 14, 800, 704)
    detection = detection_spec(saved)
    assert detection.box_mode == "bev" and detection.use_iou
    assert detection.cls_encoding == "gaussian"
    assert detection.head_mode == "grouped"
    assert detection.group_weights == (("car", .5), ("ped_cyc", .5))
    assert detection.groups[1].global_ids == (1, 2)
    assert saved["augmentation"] == read_json(ROOT / MANIFEST["augmentation_config"])["augmentation"]
    base = read_json(ROOT / MANIFEST["base_config"])
    for section in ("train", "val"):
        assert saved[section] == base[section]
    assert saved["seed"] == base["seed"]
    assert saved["model"]["stage_depths"] == [3, 4, 2]
    assert saved["model"]["backbone_out_dim"] == 32
    assert saved["model"]["c4_attention"] == "none" and saved["model"]["c4_attention_scales"] == []
    report = profile_detector(saved)
    counts = report["parameter_counts"]
    assert counts["backbone_including_neck"] == variant["backbone_including_neck_projected"] < 1_000_000
    assert counts["total_detector"] == variant["detector_iqa_on_projected"]
    assert counts["heads"] == 186161
    assert counts["criterion_train_only"] == (12 if saved["loss"]["name"] == "oga" else 0)
    assert report["output_shapes"]["groups"]["ped_cyc"]["cls"] == [1, 2, 200, 176]


def test_single_reference_has_fresh_flat_contract_and_no_group_only_fields():
    path = ROOT / "configs/experiments/under1m/hist14_local3_single_oga_iqa.json"
    assert path.is_file()
    saved = read_json(path)
    assert saved == notebook.resolve_under1m_recipe(ROOT, "reference_single_oga_iqa")
    assert detection_spec(saved).head_mode == "legacy_single"
    assert "head_groups" not in saved["data"] and "group_weights" not in saved["loss"]
    assert saved["model"]["c4_context"] == saved["model"]["local_attention"] == "none"
    report = profile_detector(saved, shape=(1, 14, 32, 48))
    assert report["parameter_counts"]["backbone_including_neck"] == 635408
    assert "groups" not in report["output_shapes"]


def test_recipe_generation_is_deterministic_and_does_not_mutate_inputs(tmp_path):
    for relative in [MANIFEST["base_config"], MANIFEST["augmentation_config"],
                     "docs/plans/lightweight_lidar_backbone_2026/experiment_manifest.json"]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    first = notebook.write_under1m_presets(tmp_path)
    snapshots = {path: path.read_bytes() for path in first}
    second = notebook.write_under1m_presets(tmp_path)
    assert first == second and len(first) == 9
    assert snapshots == {path: path.read_bytes() for path in second}
    assert not list((tmp_path / "configs/experiments/under1m").glob("*3d*"))
    base = read_json(ROOT / MANIFEST["base_config"])
    original = copy.deepcopy(base)
    notebook.resolve_under1m_recipe(ROOT, "main_focal_oga_iqa", base_config=base)
    assert base == original


@pytest.mark.parametrize("recipe", ["main_focal_3d", "optional_focal_eca_3d", "typo"])
def test_unimplemented_or_unknown_recipes_are_rejected(recipe):
    with pytest.raises(ValueError, match="recipe"):
        notebook.resolve_under1m_recipe(ROOT, recipe)


def test_runtime_presets_have_distinct_complete_semantic_and_readable_run_identities():
    configs = [read_json(ROOT / variant["planned_config"]) for variant in BEV_VARIANTS]
    names = [generate_run_name(config, seed=config["seed"]) for config in configs]
    assert len(set(names)) == len(names)
    for config, name in zip(configs, names):
        model = config["model"]
        assert f"ctx_{model['c4_context']}" in name
        assert f"local_{model['local_attention']}" in name
        assert f"f{model['neck_fusion_channels']}" in name
        assert f"detail{int(model['detail_path'])}" in name
        assert "grouped_" in name and "hybrid_gt_aug" in name
    main = next(c for c in configs if c["experiment"]["name"] == "main_focal_oga_iqa")
    assert main["model"]["local_attention"] == "none"
    for change in [{"c4_context_bottleneck": 32}, {"c4_context_dilations": [1, 3]},
                   {"local_attention": "eca"}, {"detail_path": True}, {"neck_fusion_channels": 32}]:
        assert generate_run_name(create_experiment_config(main, {"model": change})) != generate_run_name(main)


def test_candidate_run_name_exposes_context_local_fusion_and_detail():
    config = create_experiment_config(read_json(ROOT / MANIFEST["base_config"]), MANIFEST["common_overrides"])
    variant = next(v for v in BEV_VARIANTS if v["id"] == "main_focal_oga_iqa")
    config = create_experiment_config(config, variant["overrides"])
    name = generate_run_name(config)
    assert "ctx_focal_local_none_f24_detail0_" in name


@pytest.mark.parametrize("overrides", [
    {"data": {"bev_encoding": {"density_norm": 16}}},
    {"loss": {"iou_target_type": "yaw_footprint"}},
    {"loss": {"iou_loss_weight": .25}},
    {"model": {"header_act": "relu"}},
])
def test_new_candidate_run_identity_covers_encoding_and_objective_semantics(overrides):
    config = read_json(ROOT / "configs/experiments/under1m/hist14_local3_focal_grouped_oga_iqa.json")
    changed = create_experiment_config(config, overrides)
    assert checkpoint_identity(changed) != checkpoint_identity(config)
    assert generate_run_name(changed) != generate_run_name(config)


def test_new_candidate_run_identity_normalizes_defaults_and_excludes_backend_or_schedule():
    config = read_json(ROOT / "configs/experiments/under1m/hist14_local3_focal_grouped_oga_iqa.json")
    changed = create_experiment_config(config, {"train": {"epochs": 99},
        "data": {"bev_encoding": {"backend": "numba"}}, "loss": {"iou_loss_weight": 1.0}})
    assert generate_run_name(changed) == generate_run_name(config)
