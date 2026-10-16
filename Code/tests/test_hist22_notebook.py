"""Execute actual configuration cells and protect encoding replacement precedence."""
import copy
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools/kitti_training_pipeline")]
from common import bev_encoding_spec, read_json
from notebook_config import resolve_notebook_config

NOTEBOOKS = ["3D_Lidar_Object_Detection_Notebook_standard.ipynb", "3D_Lidar_Detection_Pipeline_Robust.ipynb"]


def execute_config(name, artifact_root, edits=None, namespace=None):
    notebook = json.loads((ROOT / name).read_text())
    source = next("".join(c["source"]) for c in notebook["cells"]
                  if "config_dict = resolve_notebook_config(" in "".join(c["source"]))
    source = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("%"))
    for key, value in (edits or {}).items():
        source, count = re.subn(rf"^{key} = .*?$", lambda _: f"{key} = {value!r}", source, flags=re.M)
        assert count == 1, key
    if namespace is None:
        namespace = {"Path": Path, "REPO_DIR": ROOT, "ARTIFACT_ROOT": artifact_root,
            "PROCESSED_DATASET_DIR": artifact_root / "processed", "RAW_KITTI_ROOT": artifact_root / "raw",
            "SEED": 42, "EPOCHS": 50, "WARMUP_EPOCHS": 4, "LEARNING_RATE": .0007, "NUM_WORKERS": 4,
            "PHYSICAL_BATCH_SIZE": 16, "ACCUMULATION_STEPS": 1, "PRECISION": "bf16", "TARGET_BACKEND": "numba",
            "COMPILE_MODEL": False, "VAL_BATCH_SIZE": None}
    exec(compile(source, name, "exec"), namespace)
    return namespace


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_actual_cells_switch_encoders_preserving_model_loss_attention(name, tmp_path, capsys):
    ns = execute_config(name, tmp_path)
    baseline = copy.deepcopy(ns["config_dict"])
    original_name = ns["RUN_NAME"]
    ns = execute_config(name, tmp_path, {"BEV_ENCODING": "hist22"}, ns)
    candidate = ns["config_dict"]
    schema = bev_encoding_spec(candidate)
    assert schema.channels == 22 and schema.backend == "numba"
    assert candidate["model"] == baseline["model"] and candidate["loss"] == baseline["loss"]
    for field in ("train", "val", "augmentation", "evaluation"):
        assert candidate[field] == baseline[field]
    output = capsys.readouterr().out
    assert schema.semantic_hash in output
    for value in ("11", "21", "1.1", "2.1", "numba"):
        assert value in output
    assert ns["RUN_NAME"] != original_name
    ns = execute_config(name, tmp_path, {"BEV_ENCODING": "hist14"}, ns)
    assert ns["config_dict"] == baseline and ns["RUN_NAME"] == original_name
    execute_config(name, tmp_path, {"BEV_ENCODING": "pillar_rich_gate"}, ns)
    ns = execute_config(name, tmp_path, {"BEV_ENCODING": "hist22", "BEV_BACKEND": "numpy"}, ns)
    assert bev_encoding_spec(ns["config_dict"]).backend == "numpy"
    ns = execute_config(name, tmp_path, {"BEV_ENCODING": "rich8"}, ns)
    assert bev_encoding_spec(ns["config_dict"]).channels == 8
    assert "backend" not in ns["config_dict"]["data"]["bev_encoding"]


@pytest.mark.parametrize("layer", ["custom", "file", "runtime", "preset", "under1m"])
def test_every_layer_replaces_entire_encoding_on_name_change(layer, tmp_path):
    base = read_json(ROOT / "configs/config.json")
    base["data"]["bev_encoding"] = {"name": "pillar_rich_gate", "out_channels": 32,
        "backend": "torch", "max_points_per_pillar": 8}
    path = tmp_path / "base.json"
    path.write_text(json.dumps(base))
    options = {"base_path": path, "preset": "config", "augmentation": "config"}
    patch = {"data": {"bev_encoding": {"name": "hist22", "backend": "numba"}}}
    if layer == "custom": options.update(preset="custom", custom_overrides=patch)
    elif layer == "runtime": options["runtime_overrides"] = patch
    elif layer == "file":
        override = tmp_path / "override.json"
        override.write_text(json.dumps(patch))
        options["file_override"] = override
    elif layer == "preset": options["preset"] = "ENCODER_RICH8"
    else: options["preset"] = "UNDER1M_FOCAL_OGA_IQA"
    config = resolve_notebook_config(ROOT, **options)
    expected = "hist14" if layer == "under1m" else "rich8" if layer == "preset" else "hist22"
    assert bev_encoding_spec(config).name == expected
    assert "max_points_per_pillar" not in config["data"]["bev_encoding"]


@pytest.mark.parametrize("layer", ["custom", "file", "runtime"])
@pytest.mark.parametrize("options", [{"out_channels": 14}, {"backend": "torch"}, {"bad_option": 1}])
def test_new_explicit_invalid_options_survive_replacement_and_fail(layer, options, tmp_path):
    patch = {"data": {"bev_encoding": {"name": "hist22", **options}}}
    kwargs = {"preset": "config", "augmentation": "config"}
    if layer == "custom": kwargs.update(preset="custom", custom_overrides=patch)
    elif layer == "runtime": kwargs["runtime_overrides"] = patch
    else:
        path = tmp_path / "override.json"; path.write_text(json.dumps(patch)); kwargs["file_override"] = path
    with pytest.raises(ValueError): resolve_notebook_config(ROOT, **kwargs)


@pytest.mark.parametrize("options", [{"out_channels": True}, {"backend": "torch"}, {"bad_option": 1}])
def test_same_name_explicit_invalid_options_still_fail(options, tmp_path):
    base = read_json(ROOT / "configs/config.json")
    base["data"]["bev_encoding"] = {"name": "hist14", "backend": "numba"}
    path = tmp_path / "hist14.json"
    path.write_text(json.dumps(base))
    with pytest.raises(ValueError):
        resolve_notebook_config(ROOT, base_path=path, preset="custom", augmentation="config",
            custom_overrides={"data": {"bev_encoding": {"name": "hist14", **options}}})


def test_standard_inline_override_replaces_encoding_before_resolution(tmp_path):
    ns = execute_config(NOTEBOOKS[0], tmp_path, {"BEV_ENCODING": "pillar_rich_gate",
        "NOTEBOOK_OVERRIDES": {"data": {"bev_encoding": {"name": "hist22", "backend": "numba"}}}})
    assert bev_encoding_spec(ns["config_dict"]).channels == 22


def test_generator_reproduces_committed_robust_notebook(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "tools/generate_robust_notebook.py")],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads((tmp_path / NOTEBOOKS[1]).read_text()) == json.loads((ROOT / NOTEBOOKS[1]).read_text())


def test_standard_fresh_kernel_loads_master_runtime_defaults_and_preserves_supplied_values(tmp_path):
    namespace = {"Path": Path, "REPO_DIR": ROOT, "ARTIFACT_ROOT": tmp_path,
        "PROCESSED_DATASET_DIR": tmp_path / "processed", "RAW_KITTI_ROOT": tmp_path / "raw"}
    ns = execute_config(NOTEBOOKS[0], tmp_path, namespace=namespace)
    master = read_json(ROOT / "configs/config.json")
    for key in ("epochs", "warmup_epochs", "physical_batch_size", "num_workers", "precision", "target_backend"):
        assert ns["config_dict"]["train"][key] == master["train"][key]
    ns["EPOCHS"], ns["WARMUP_EPOCHS"] = 13, 2
    ns = execute_config(NOTEBOOKS[0], tmp_path, namespace=ns)
    assert ns["config_dict"]["train"]["epochs"] == 13 and ns["config_dict"]["train"]["warmup_epochs"] == 2
