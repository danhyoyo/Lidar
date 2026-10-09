"""Execute the notebook configuration cell with edits a user can make."""

import ast
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools/kitti_training_pipeline")]
from common import detection_spec, checkpoint_identity


def config_cell():
    notebook = json.loads((ROOT / "3D_Lidar_Object_Detection_Notebook_standard.ipynb").read_text())
    return next("".join(cell["source"]) for cell in notebook["cells"]
                if "config_dict = resolve_notebook_config(" in "".join(cell["source"]))


def execute_cell(tmp_path, edits=None, namespace=None):
    source = "\n".join(line for line in config_cell().splitlines() if not line.lstrip().startswith("%"))
    for key, value in (edits or {}).items():
        source, count = re.subn(rf"^{key} = .*?$", lambda match: f"{key} = {value!r}", source, flags=re.M)
        assert count == 1, f"Missing or duplicated notebook control: {key}"
    if namespace is None:
        namespace = {"REPO_DIR": ROOT, "ARTIFACT_ROOT": tmp_path, "PROCESSED_DATASET_DIR": tmp_path / "processed",
            "SEED": 42, "EPOCHS": 8, "WARMUP_EPOCHS": 1, "LEARNING_RATE": .0007, "NUM_WORKERS": 0,
            "PHYSICAL_BATCH_SIZE": 2, "ACCUMULATION_STEPS": 1, "PRECISION": "fp32", "TARGET_BACKEND": "python",
            "COMPILE_MODEL": False, "VAL_BATCH_SIZE": 3}
    exec(compile(source, "notebook_configuration_cell", "exec"), namespace)
    return namespace


def test_notebook_defaults_select_main_focal_with_hybrid_and_local_none(tmp_path):
    namespace = execute_cell(tmp_path)
    config = namespace["config_dict"]
    assert namespace["PRESET"] == "custom"
    assert config["model"]["c4_context"] == "focal" and config["model"]["local_attention"] == "none"
    assert config["model"]["backbone_out_dim"] == 32
    assert detection_spec(config).head_mode == "grouped"
    assert config["data"]["bev_encoding"]["name"] == "hist14"
    assert config["augmentation"]["AUG_CONFIG_LIST"][0]["NAME"] == "hybrid_gt_sampling"
    assert config["train"]["precision"] == "fp32" and config["val"]["physical_batch_size"] == 3
    assert json.loads(namespace["CONFIG"].read_text()) == config


@pytest.mark.parametrize("preset", ["custom", "ENCODER_PILLAR_RICH_GATE"])
def test_notebook_selects_rich_gate_with_torch_backend_and_fixed_width(tmp_path, preset):
    namespace = execute_cell(tmp_path, {"PRESET": preset, "BEV_ENCODING": "pillar_rich_gate",
                                        "AUGMENTATION": "standard"})
    config = namespace["config_dict"]
    encoding = config["data"]["bev_encoding"]
    assert encoding["name"] == "pillar_rich_gate"
    assert encoding["out_channels"] == 32 and encoding["backend"] == "torch"
    gate = checkpoint_identity(config)["encoding"]["metadata"]["learned_encoder"]["gate"]
    assert gate["descriptor"] == "concat(learned_max24, rich8)"


def test_notebook_selects_best_ap_and_exposes_independent_ap_interval(tmp_path):
    namespace = execute_cell(tmp_path)
    assert namespace['config_dict']['train'].get('checkpoint_selection', {}).get('primary') == 'ap'
    changed = execute_cell(tmp_path/'sparse', {'AP_EVERY': 5})
    assert changed['config_dict']['train']['checkpoint_selection']['ap_every'] == 5
    assert changed['config_dict']['val']['val_every'] == 1
    assert changed['config_dict']['evaluation']['kitti_root']


def test_notebook_short_smoke_disables_full_split_ap_selection():
    notebook = json.loads((ROOT/'3D_Lidar_Object_Detection_Notebook_standard.ipynb').read_text())
    source = next(''.join(cell['source']) for cell in notebook['cells'] if 'smoke_config =' in ''.join(cell['source']))
    from IPython.core.interactiveshell import InteractiveShell
    tree = ast.parse(InteractiveShell.instance().input_transformer_manager.transform_cell(source))
    assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'smoke_config' for t in node.targets))
    override = ast.literal_eval(assignment.value.args[1])
    assert override['train'].get('checkpoint_selection', {}).get('primary') == 'loss'


@pytest.mark.parametrize("local,options", [("none", {}), ("eca", {"LOCAL_ATTENTION_ECA_KERNEL_SIZE": 5}),
    ("simam", {"LOCAL_ATTENTION_SIMAM_LAMBDA": .002})])
def test_custom_controls_reach_encoding_groups_context_and_neck(tmp_path, local, options):
    namespace = execute_cell(tmp_path, {"PRESET": "custom", "BEV_ENCODING": "hist14", "BEV_BACKEND": "numpy",
        "STAGE_DEPTHS": [3, 4, 2], "BACKBONE_OUT_DIM": 32, "HEAD_MODE": "grouped",
        "HEADER_USE_IOU": True, "LOSS_NAME": "oga", "C4_ATTENTION": "none", "C4_CONTEXT": "focal",
        "C4_CONTEXT_BOTTLENECK": 32, "C4_CONTEXT_DILATIONS": [1, 3], "LOCAL_ATTENTION": local,
        "NECK_FUSION_CHANNELS": 32, "DETAIL_PATH": True, "AUGMENTATION": "none", **options})
    config = namespace["config_dict"]
    model = config["model"]
    assert model["c4_context_bottleneck"] == 32 and model["c4_context_dilations"] == [1, 3]
    assert model["local_attention"] == local and model["detail_path"] is True and model["neck_fusion_channels"] == 32
    assert config["data"]["bev_encoding"]["out_channels"] == 14
    assert detection_spec(config).groups[1].classes == ("Pedestrian", "Cyclist")
    if local == "eca":
        assert model["local_attention_eca_kernel_size"] == 5 and "local_attention_simam_lambda" not in model
    if local == "simam":
        assert model["local_attention_simam_lambda"] == .002 and "local_attention_eca_kernel_size" not in model


def test_reexecuting_cell_switches_presets_without_stale_groups_or_attention(tmp_path):
    namespace = execute_cell(tmp_path, {"PRESET": "UNDER1M_FOCAL_ECA_OGA_IQA"})
    first = namespace["RUN_NAME"]
    first_identity = checkpoint_identity(namespace["config_dict"])
    namespace = execute_cell(tmp_path, {"PRESET": "UNDER1M_SINGLE_OGA_IQA"}, namespace)
    assert namespace["RUN_NAME"] != first
    assert "head_groups" not in namespace["config_dict"]["data"]
    assert "group_weights" not in namespace["config_dict"]["loss"]
    assert "local_attention_eca_kernel_size" not in namespace["config_dict"]["model"]
    assert detection_spec(namespace["config_dict"]).head_mode == "legacy_single"
    namespace = execute_cell(tmp_path, {"PRESET": "UNDER1M_FOCAL_ECA_OGA_IQA"}, namespace)
    assert namespace["RUN_NAME"] == first and checkpoint_identity(namespace["config_dict"]) == first_identity


def test_config_cell_has_explicit_group_weights_and_classification_controls(tmp_path):
    groups = [{"name": "vehicles", "classes": ["Car"]}, {"name": "vru", "classes": ["Cyclist", "Pedestrian"]}]
    namespace = execute_cell(tmp_path, {"PRESET": "custom", "HEAD_MODE": "grouped", "HEAD_GROUPS": groups,
        "GROUP_WEIGHTS": {"vehicles": 1, "vru": 3}, "CLS_ENCODING": "binary", "LOSS_NAME": "oga"})
    contract = detection_spec(namespace["config_dict"])
    assert contract.groups[1].global_ids == (2, 1)
    assert contract.cls_encoding == "binary" and contract.group_weights == (("vehicles", .25), ("vru", .75))


def test_configuration_cell_keeps_single_occurrence_of_controls_and_valid_python():
    source = "\n".join(line for line in config_cell().splitlines() if not line.startswith("%"))
    tree = ast.parse(source)
    controls = [target.id for node in tree.body if isinstance(node, ast.Assign)
                for target in node.targets if isinstance(target, ast.Name)]
    for key in ["PRESET", "BEV_BACKEND", "STAGE_DEPTHS", "HEAD_MODE", "HEAD_GROUPS", "GROUP_WEIGHTS",
                "C4_CONTEXT", "LOCAL_ATTENTION", "NECK_FUSION_CHANNELS", "DETAIL_PATH"]:
        assert controls.count(key) == 1
