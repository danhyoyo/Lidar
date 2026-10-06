"""Execute the real notebook configuration cell without Colab or shell magics."""

import ast
import copy
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "3D_Lidar_Object_Detection_Notebook_standard.ipynb"


def configuration_cell(settings=None):
    notebook = json.loads(NOTEBOOK.read_text())
    source = "".join(notebook["cells"][3]["source"])
    tree = ast.parse("\n".join(line for line in source.splitlines() if not line.startswith("%")))
    # Changing top-level option assignments models editing the notebook controls.
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in (settings or {}):
                node.value = ast.parse(repr(settings[target.id]), mode="eval").body
    return compile(ast.fix_missing_locations(tree), str(NOTEBOOK), "exec")


def namespace(tmp_path):
    return dict(
        REPO_DIR=ROOT, ARTIFACT_ROOT=tmp_path / "artifacts",
        PROCESSED_DATASET_DIR=tmp_path / "processed", Path=Path, json=json,
        EPOCHS=100, WARMUP_EPOCHS=8, LEARNING_RATE=.0007,
        NUM_WORKERS=0, PHYSICAL_BATCH_SIZE=8, ACCUMULATION_STEPS=2,
        PRECISION="bf16", TARGET_BACKEND="python", COMPILE_MODEL=False,
        SEED=17, RUN_NAME=None, CUSTOM_RUN_NAME=None,
    )


def execute(tmp_path, settings=None, state=None, base_changes=None):
    state = namespace(tmp_path) if state is None else state
    settings = dict(settings or {})
    if base_changes:
        base = json.loads((ROOT / "configs/config.json").read_text())
        for section, values in base_changes.items():
            base[section].update(copy.deepcopy(values))
        base_path = tmp_path / "base.json"
        base_path.write_text(json.dumps(base))
        settings["CONFIG_BASE"] = str(base_path)
    exec(configuration_cell(settings), state)
    return state


@pytest.mark.parametrize("preset", [
    "custom", "TRUC_A_SOTA", "TRUC_B_SOTA", "TRUC_A_M1", "TRUC_A_M2", "TRUC_B_M1",
    "RICH8_SGFPN", "RICH10_SGFPN", "RICH11_SGFPN", "RICH12_SGFPN",
    "RC_SGFPN", "RC_BISGFPN", "MOBILEPIXOR_BASELINE", "LEGACY35_BASELINE",
])
@pytest.mark.parametrize("augmentation", ["standard", "none", "openpcdet_global", "openpcdet_gt"])
def test_augmentation_selection_applies_to_every_model_preset(tmp_path, preset, augmentation):
    state = execute(tmp_path, {"PRESET": preset, "AUGMENTATION": augmentation})
    aug = state["config_dict"]["augmentation"]
    if augmentation in ("standard", "none"):
        assert aug["mode"] == "one_of"
        assert aug["p"] == (.5 if augmentation == "standard" else 0)
        assert "AUG_CONFIG_LIST" not in aug
    else:
        assert aug["mode"] == "openpcdet"
        names = [entry["NAME"] for entry in aug["AUG_CONFIG_LIST"]]
        assert ("gt_sampling" in names) == (augmentation == "openpcdet_gt")
    assert json.loads(state["CONFIG"].read_text()) == state["config_dict"]


def test_reexecuting_cell_regenerates_the_run_name(tmp_path):
    state = execute(tmp_path, {"AUGMENTATION": "standard"})
    first_name = state["RUN_NAME"]
    first_config = state["CONFIG"].read_text()
    execute(tmp_path, {"AUGMENTATION": "none"}, state=state)
    assert state["RUN_NAME"] != first_name
    assert "noaug" in state["RUN_NAME"]
    assert (state["ARTIFACT_ROOT"] / first_name / "config.json").read_text() == first_config


def test_standard_selection_replaces_an_openpcdet_base_recipe(tmp_path):
    recipe = json.loads((ROOT / "configs/augmentation/openpcdet_gt.json").read_text())["augmentation"]
    state = execute(tmp_path, {"AUGMENTATION": "standard"}, base_changes={"augmentation": recipe})
    aug = state["config_dict"]["augmentation"]
    assert aug["mode"] == "one_of"
    assert aug["p"] == .5
    assert "AUG_CONFIG_LIST" not in aug


def test_preset_resets_bev_channels_neck_and_quality_from_edited_base(tmp_path):
    state = execute(tmp_path, {"PRESET": "RICH8_SGFPN"}, base_changes={
        "data": {"bev_encoding": {"name": "rich12", "out_channels": 12}},
        "model": {"neck_type": "rc_bisgfpn", "header_use_iou": True, "use_reparam": True},
        "loss": {"name": "oga", "use_iou": True},
    })
    cfg = state["config_dict"]
    assert state["bev_channels"] == 8
    assert cfg["model"].get("neck_type") is None
    assert not cfg["model"]["header_use_iou"]
    assert not cfg["model"]["use_reparam"]
    assert cfg["loss"]["name"] == "baseline"
    assert not cfg["loss"]["use_iou"]


def test_custom_none_neck_clears_a_range_conditioned_base(tmp_path):
    state = execute(tmp_path, {"NECK_TYPE": None}, base_changes={"model": {"neck_type": "rc_sgfpn"}})
    assert state["config_dict"]["model"].get("neck_type") is None


def test_runtime_schedule_and_seed_are_saved_in_effective_config(tmp_path):
    cfg = execute(tmp_path)["config_dict"]
    assert cfg["seed"] == 17
    assert cfg["train"]["epochs"] == 100
    assert cfg["train"]["warmup_epochs"] == 8


@pytest.mark.parametrize("augmentation", ["config", "compose"])
def test_additional_augmentation_modes(tmp_path, augmentation):
    recipe = json.loads((ROOT / "configs/augmentation/openpcdet_gt.json").read_text())["augmentation"]
    state = execute(tmp_path, {"AUGMENTATION": augmentation}, base_changes={"augmentation": recipe})
    aug = state["config_dict"]["augmentation"]
    if augmentation == "config":
        assert aug["mode"] == "openpcdet"
        assert aug["AUG_CONFIG_LIST"] == recipe["AUG_CONFIG_LIST"]
    else:
        assert aug["mode"] == "compose"
        assert "AUG_CONFIG_LIST" not in aug


def test_use_base_config_without_model_overrides(tmp_path):
    state = execute(tmp_path, {"PRESET": "config", "AUGMENTATION": "config"}, base_changes={
        "data": {"bev_encoding": {"name": "rich12", "out_channels": 12}},
        "model": {"neck_type": "rc_bisgfpn", "backbone_out_dim": 32},
    })
    assert state["bev_channels"] == 12
    assert state["config_dict"]["model"]["neck_type"] == "rc_bisgfpn"
    assert state["config_dict"]["model"]["backbone_out_dim"] == 32


def test_custom_head_width_is_applied(tmp_path):
    state = execute(tmp_path, {"BACKBONE_OUT_DIM": 32})
    assert state["config_dict"]["model"]["backbone_out_dim"] == 32


def test_bad_warmup_fails_before_writing_config(tmp_path):
    state = namespace(tmp_path)
    state["WARMUP_EPOCHS"] = 100
    with pytest.raises(ValueError, match="warmup"):
        execute(tmp_path, state=state)
    assert not list((tmp_path / "artifacts").glob("*/config.json"))


def test_existing_training_config_is_preserved_on_incompatible_rerun(tmp_path):
    state = execute(tmp_path)
    previous = state["CONFIG"].read_text()
    (state["RUN_DIR"] / "run.json").write_text("{}")
    state["LEARNING_RATE"] = .001
    with pytest.raises(ValueError, match="EXPERIMENT_TAG|CUSTOM_RUN_NAME"):
        execute(tmp_path, state=state)
    assert state["CONFIG"].read_text() == previous


def test_file_override_has_priority_but_runtime_remains_synchronized(tmp_path):
    overrides = json.loads((ROOT / "configs/augmentation/openpcdet_global.json").read_text())
    overrides.update({
        "model": {"header_use_iou": True, "backbone_out_dim": 32},
        "data": {"bev_encoding": {"name": "rich10"}},
        "train": {"epochs": 7, "learning_rate": .1}, "seed": 99,
    })
    path = tmp_path / "override.json"
    path.write_text(json.dumps(overrides))
    state = execute(tmp_path, {"CONFIG_OVERRIDE": str(path), "AUGMENTATION": "none"})
    cfg = state["config_dict"]
    assert cfg["augmentation"] == overrides["augmentation"]
    assert cfg["model"]["header_use_iou"] and cfg["loss"]["use_iou"]
    assert cfg["model"]["backbone_out_dim"] == 32
    assert state["bev_channels"] == 10
    assert cfg["train"]["epochs"] == 100 and cfg["seed"] == 17
    assert cfg["train"]["learning_rate"] == .0007
    assert "openpcdet_aug" in state["RUN_NAME"]


def test_matching_config_can_resume_without_changing_saved_configuration(tmp_path):
    state = execute(tmp_path)
    previous = state["CONFIG"].read_text()
    (state["RUN_DIR"] / "run.json").write_text("{}")
    execute(tmp_path, state=state)
    assert state["CONFIG"].read_text() == previous


def test_checkpoint_without_metadata_still_protects_existing_config(tmp_path):
    state = execute(tmp_path)
    previous = state["CONFIG"].read_text()
    (state["RUN_DIR"] / "checkpoints").mkdir()
    (state["RUN_DIR"] / "checkpoints/last.pt").write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="EXPERIMENT_TAG|CUSTOM_RUN_NAME"):
        execute(tmp_path, {"C4_ATTENTION": "none"}, state=state)
    assert state["CONFIG"].read_text() == previous


def test_manual_run_name_and_validation_batch_size(tmp_path):
    state = namespace(tmp_path)
    state["VAL_BATCH_SIZE"] = 8
    execute(tmp_path, {"CUSTOM_RUN_NAME": "my_experiment"}, state=state)
    assert state["RUN_NAME"] == "my_experiment"
    assert state["config_dict"]["val"]["physical_batch_size"] == 8


@pytest.mark.parametrize("settings,message", [
    ({"AUGMENTATION": "typo"}, "AUGMENTATION"),
    ({"PRESET": "typo"}, "PRESET"),
    ({"BACKBONE": "typo"}, "BACKBONE"),
    ({"NECK_TYPE": "typo"}, "NECK_TYPE"),
    ({"BACKBONE": "mobilepixor", "BACKBONE_OUT_DIM": 32}, "BACKBONE_OUT_DIM"),
    ({"BACKBONE": "mobilepixor", "NECK_TYPE": "rc_sgfpn"}, "require mobilepixornext"),
    ({"C4_ATTENTION_SCALES": [4]}, "C4_ATTENTION_SCALES"),
    ({"HEADER_USE_IOU": True, "LOSS_NAME": "q_oga"}, "IQA"),
    ({"CUSTOM_RUN_NAME": "../another_run"}, "CUSTOM_RUN_NAME"),
])
def test_invalid_selections_fail_before_writing_config(tmp_path, settings, message):
    with pytest.raises(ValueError, match=message):
        execute(tmp_path, settings)
    assert not list((tmp_path / "artifacts").glob("*/config.json"))


def test_head_32_matches_real_model_output_and_backward(tmp_path):
    import torch
    state = execute(tmp_path, {"BACKBONE_OUT_DIM": 32, "BEV_ENCODING": "rich12"})
    from common import build_model, configure_detector_imports
    configure_detector_imports(ROOT / "detector")
    model = build_model(state["config_dict"])
    pred = model(torch.randn(2, 12, 64, 64))
    assert pred["cls"].shape == (2, 3, 16, 16)
    assert pred["offset"].shape == (2, 2, 16, 16)
    sum(value.square().mean() for value in pred.values()).backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
