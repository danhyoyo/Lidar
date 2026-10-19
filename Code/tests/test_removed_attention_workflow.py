"""Removed backbone modules cannot be selected through supported workflows."""

import json
from pathlib import Path

import pytest

from tools.kitti_training_pipeline.notebook_config import resolve_notebook_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("preset", [
    "UNDER1M_FOCAL_OGA_IQA", "UNDER1M_FOCAL_BASELINE_IQA",
    "UNDER1M_FOCAL_ECA_OGA_IQA", "UNDER1M_FOCAL_SIMAM_OGA_IQA",
    "UNDER1M_FOCAL_FUSION32_OGA_IQA", "UNDER1M_FOCAL_DETAIL_OGA_IQA",
])
def test_retired_presets_rejected_before_training(preset):
    with pytest.raises(ValueError, match="Unknown PRESET"):
        resolve_notebook_config(ROOT, preset=preset, augmentation="config")


@pytest.mark.parametrize("filename", [
    "3D_Lidar_Object_Detection_Notebook_standard.ipynb",
    "3D_Lidar_Detection_Pipeline_Robust.ipynb",
])
def test_notebook_has_no_removed_module_controls(filename):
    notebook = json.loads((ROOT / filename).read_text())
    source = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
    for control in ("C4_CONTEXT", "FOCAL_STAGE", "ATTENTION_ORDER", "LOCAL_ATTENTION_ECA_KERNEL_SIZE"):
        assert control not in source


def test_simam_preset_still_resolves_without_context_fields():
    config = resolve_notebook_config(ROOT, preset="UNDER1M_SIMAM_OGA_IQA", augmentation="none")
    assert config["model"]["local_attention"] == "simam"
    assert not any(key.startswith(("c4_context", "focal_")) for key in config["model"])
    assert "attention_order" not in config["model"]


def test_legacy_disabled_context_has_same_checkpoint_identity_and_snapshot(tmp_path):
    from copy import deepcopy
    from tools.kitti_training_pipeline.common import checkpoint_identity
    from tools.kitti_training_pipeline.notebook_config import save_notebook_config

    config = resolve_notebook_config(ROOT, preset="UNDER1M_GROUPED_OGA_IQA", augmentation="none")
    legacy = deepcopy(config)
    legacy["model"]["c4_context"] = "none"
    assert checkpoint_identity(legacy) == checkpoint_identity(config)
    path = save_notebook_config(tmp_path, legacy)
    snapshot = path.read_bytes()
    (tmp_path / "run.json").write_text("{}")
    assert save_notebook_config(tmp_path, config) == path
    assert path.read_bytes() == snapshot
