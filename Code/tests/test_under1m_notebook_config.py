"""New preset selection preserves notebook precedence and existing run state."""

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools/kitti_training_pipeline")]
from common import (checkpoint_identity, create_experiment_config, generate_run_name, read_json)
import notebook_config as notebook

PRESETS = {
    "UNDER1M_SINGLE_OGA_IQA": "reference_single_oga_iqa",
    "UNDER1M_GROUPED_OGA_IQA": "reference_oga_iqa",
    "UNDER1M_GROUPED_BASELINE_IQA": "reference_baseline_iqa",
    "UNDER1M_SIMAM_OGA_IQA": "optional_simam",
    "UNDER1M_FUSION32_OGA_IQA": "optional_fusion32",
    "UNDER1M_DETAIL_OGA_IQA": "optional_detail",
}


@pytest.mark.parametrize("preset", [
    "ENCODER_PILLAR32",
    "ENCODER_PILLAR_RICH", "ENCODER_PILLAR_RICH_GATE",
    "ENCODER_PILLAR_RICH_ECA", "ENCODER_PILLAR_RICH_MAX_MEAN_ECA",
])
def test_retired_pillar_presets_are_rejected(preset):
    with pytest.raises(ValueError, match="Unknown PRESET"):
        notebook.resolve_notebook_config(ROOT, preset=preset, augmentation="config")


@pytest.mark.parametrize("name", ["pillar32", "pillar_rich", "pillar_rich_gate", "pillar_rich_eca"])
def test_custom_notebook_rejects_retired_learned_encodings(name):
    with pytest.raises(ValueError, match="unsupported BEV encoding"):
        notebook.resolve_notebook_config(ROOT, preset="custom", augmentation="config",
            custom_overrides={"data": {"bev_encoding": {"name": name}}})


@pytest.mark.parametrize("preset,recipe", list(PRESETS.items()))
def test_named_preset_rebuilds_exact_runtime_file_and_checkpoint_identity(preset, recipe):
    config = notebook.resolve_notebook_config(ROOT, preset=preset, augmentation="config")
    expected = notebook.resolve_under1m_recipe(ROOT, recipe)
    assert config == expected
    assert checkpoint_identity(config) == checkpoint_identity(expected)
    assert generate_run_name(config) == generate_run_name(expected)


@pytest.mark.parametrize("augmentation", ["none", "standard", "compose", "hybrid_gt"])
def test_new_preset_augmentation_remains_independent_and_runtime_wins(augmentation):
    runtime = {"seed": 17, "train": {"physical_batch_size": 2, "precision": "fp32"}}
    original = copy.deepcopy(runtime)
    config = notebook.resolve_notebook_config(ROOT, preset="UNDER1M_GROUPED_OGA_IQA",
        augmentation=augmentation, runtime_overrides=runtime, hybrid_options={"PROBABILITY": .25})
    assert runtime == original
    assert config["seed"] == 17 and config["train"]["physical_batch_size"] == 2
    assert "c4_context" not in config["model"]
    if augmentation == "hybrid_gt":
        assert config["augmentation"]["AUG_CONFIG_LIST"][0]["PROBABILITY"] == .25
    else:
        assert "AUG_CONFIG_LIST" not in config["augmentation"]
        assert config["augmentation"]["p"] == (0 if augmentation == "none" else .5)


def test_file_override_uses_same_preset_contract_and_runtime_precedence(tmp_path):
    path = tmp_path / "override.json"
    path.write_text(json.dumps({"seed": 12, "model": {"local_attention": "simam"},
                                "train": {"precision": "fp16"}}))
    config = notebook.resolve_notebook_config(ROOT, preset="UNDER1M_GROUPED_OGA_IQA",
        augmentation="config", file_override=path, runtime_overrides={"seed": 19, "train": {"precision": "fp32"}})
    assert config["model"]["local_attention"] == "simam"
    assert config["seed"] == 19 and config["train"]["precision"] == "fp32"
    assert config["model"]["local_attention_simam_lambda"] == .0001


@pytest.mark.parametrize("model_change", [{"c4_context": "focal", "c4_attention": "litemla"},
    {"local_attention": ["eca", "simam"]}, {"local_attention_simam_lambda": 0},
    {"neck_fusion_channels": 16}, {"detail_path": "true"}])
def test_preset_cannot_bypass_attention_validation(model_change):
    with pytest.raises(ValueError):
        notebook.resolve_notebook_config(ROOT, preset="UNDER1M_GROUPED_OGA_IQA",
            augmentation="config", runtime_overrides={"model": model_change})


def test_existing_trained_run_is_preserved_when_recipe_changes(tmp_path):
    first = notebook.resolve_notebook_config(ROOT, preset="UNDER1M_GROUPED_OGA_IQA", augmentation="config")
    path = notebook.save_notebook_config(tmp_path, first)
    before = path.read_bytes()
    (tmp_path / "run.json").write_text("{}")
    changed = notebook.resolve_notebook_config(ROOT, preset="UNDER1M_SIMAM_OGA_IQA", augmentation="config")
    with pytest.raises(ValueError, match="config"):
        notebook.save_notebook_config(tmp_path, changed)
    assert path.read_bytes() == before
    notebook.save_notebook_config(tmp_path, first)
    assert read_json(path) == first


@pytest.mark.parametrize("preset", ["TRUC_A_M1", "TRUC_B_M1", "RICH8_SGFPN", "LEGACY35_BASELINE"])
def test_historical_presets_keep_their_flat_contract(preset):
    config = notebook.resolve_notebook_config(ROOT, preset=preset)
    assert config["model"].get("head_mode", "legacy_single") == "legacy_single"
    assert "head_groups" not in config["data"]
    assert config["data"]["bev_encoding"]["name"] != "hist14"


@pytest.mark.parametrize("encoding,width", [("rich8", 8), ("rich8", 16), ("binary_slices", 35)])
def test_hist14_preset_replaces_previous_encoding_width_from_edited_base(tmp_path, encoding, width):
    base = read_json(ROOT / "configs/config.json")
    base["data"]["bev_encoding"] = {"name": encoding, "out_channels": width}
    base["seed"] = 11
    base_path = tmp_path / "base.json"
    base_path.write_text(json.dumps(base))
    config = notebook.resolve_notebook_config(ROOT, base_path=base_path,
        preset="UNDER1M_GROUPED_OGA_IQA", augmentation="config")
    assert config["seed"] == 11
    assert config["data"]["bev_encoding"]["name"] == "hist14"
    assert config["data"]["bev_encoding"].get("out_channels", 14) == 14
