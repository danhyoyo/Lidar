"""The actual Standard/Robust cells expose and persist selective block SimAM."""

import json
from pathlib import Path

import pytest

from tools.kitti_training_pipeline.notebook_config import resolve_notebook_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("filename", ["3D_Lidar_Object_Detection_Notebook_standard.ipynb",
                                      "3D_Lidar_Detection_Pipeline_Robust.ipynb"])
def test_notebook_block_toggle_stage_selection_and_disabling(filename, tmp_path, monkeypatch):
    import test_under1m_notebook_controls as controls
    notebook = json.loads((ROOT / filename).read_text())
    cell = next("".join(cell["source"]) for cell in notebook["cells"]
                if "config_dict = resolve_notebook_config(" in "".join(cell["source"]))
    monkeypatch.setattr(controls, "config_cell", lambda: cell)
    namespace = {"REPO_DIR": ROOT, "ARTIFACT_ROOT": tmp_path,
                 "PROCESSED_DATASET_DIR": tmp_path / "processed", "RAW_KITTI_ROOT": ROOT / "Kitti_Raw",
                 "SEED": 42, "EPOCHS": 8, "WARMUP_EPOCHS": 1, "LEARNING_RATE": .0007,
                 "NUM_WORKERS": 0, "PHYSICAL_BATCH_SIZE": 2, "ACCUMULATION_STEPS": 1,
                 "PRECISION": "fp32", "TARGET_BACKEND": "python", "COMPILE_MODEL": False,
                 "VAL_BATCH_SIZE": 3}
    edits = {"PRESET": "custom", "C4_CONTEXT": "none", "LOCAL_ATTENTION": "none",
             "AUGMENTATION": "none"}
    active = controls.execute_cell(tmp_path, edits, namespace)
    assert active["BLOCK_SIMAM"] is True
    model = active["config_dict"]["model"]
    assert model["block_simam_stages"] == ["c4", "c5"]
    assert model["block_simam_lambda"] == .0001
    from common import build_model
    from core.models.backbones.light_attention import SimAMCore
    from torch import nn
    backbone = build_model(active["config_dict"]).backbone
    assert isinstance(backbone.c3_light_attention, nn.Identity)
    assert sum(isinstance(module, SimAMCore) for module in backbone.modules()) == 6
    summary = active["attention_placement_summary"](active["config_dict"])
    assert "Block SimAM: C4" in summary and "4 blocks" in summary
    assert "Block SimAM: C5" in summary and "2 blocks" in summary
    assert json.loads(active["CONFIG"].read_text())["model"] == model
    first_name = active["RUN_NAME"]
    selected = controls.execute_cell(tmp_path, {**edits, "BLOCK_SIMAM_STAGES": ["c5"]}, namespace)
    assert selected["config_dict"]["model"]["block_simam_stages"] == ["c5"]
    assert selected["RUN_NAME"] != first_name
    empty = controls.execute_cell(tmp_path, {**edits, "BLOCK_SIMAM_STAGES": []}, namespace)
    assert not any(key.startswith("block_simam") for key in empty["config_dict"]["model"])
    disabled = controls.execute_cell(tmp_path, {**edits, "BLOCK_SIMAM": False}, namespace)
    assert not any(key.startswith("block_simam") for key in disabled["config_dict"]["model"])


def test_custom_merge_clears_only_inherited_inactive_block_settings(tmp_path):
    config = json.loads((ROOT / "configs/config.json").read_text())
    config["model"].update(block_simam_stages=["c4", "c5"], block_simam_lambda=.002)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/config.json").write_text(json.dumps(config))
    result = resolve_notebook_config(tmp_path, preset="custom", augmentation="config",
                                    custom_overrides={"model": {"block_simam_stages": []}})
    assert not any(key.startswith("block_simam") for key in result["model"])
    with pytest.raises(ValueError, match="block_simam"):
        resolve_notebook_config(tmp_path, preset="custom", augmentation="config",
            custom_overrides={"model": {"block_simam_stages": [], "block_simam_lambda": .002}})
    assert json.loads((tmp_path / "configs/config.json").read_text()) == config


def test_disabled_notebook_controls_do_not_validate_unused_stages_and_lambda():
    from tools.kitti_training_pipeline.notebook_config import attention_placement_overrides
    result = attention_placement_overrides({}, block_simam=False,
                                           block_simam_stages=None, block_simam_lambda=-1)
    assert result["block_simam_stages"] == []
    with pytest.raises(ValueError, match="BLOCK_SIMAM"):
        attention_placement_overrides({}, block_simam="yes")


def test_saved_trained_config_canonicalizes_disabled_options_and_preserves_file(tmp_path):
    from tools.kitti_training_pipeline.notebook_config import save_notebook_config
    config = json.loads((ROOT / "configs/config.json").read_text())
    path = save_notebook_config(tmp_path, config)
    original = path.read_bytes()
    (tmp_path / "run.json").write_text("{}")
    explicit = json.loads(json.dumps(config))
    explicit["model"]["block_simam_stages"] = []
    assert save_notebook_config(tmp_path, explicit) == path
    assert path.read_bytes() == original
    explicit["model"]["block_simam_stages"] = ["c4", "c5"]
    with pytest.raises(ValueError, match="config"):
        save_notebook_config(tmp_path, explicit)
    assert path.read_bytes() == original
