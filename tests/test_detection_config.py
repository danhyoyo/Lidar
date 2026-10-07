"""Public construction must enforce the same pure detection capability contract."""

import copy
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def candidate(grouped=True, loss="oga", iqa=True, classification="gaussian"):
    config = json.loads((ROOT / "configs/config.json").read_text())
    config["model"].update(head_mode="grouped" if grouped else "legacy_single",
                           cls_encoding=classification, header_use_iou=iqa)
    config["loss"] = {"name": loss, "use_iou": iqa}
    if grouped:
        config["data"]["head_groups"] = [
            {"name": "car", "classes": ["Car"]},
            {"name": "ped_cyc", "classes": ["Pedestrian", "Cyclist"]}]
    return config


def resolve(config):
    return importlib.import_module("detector.core.detection_config").resolve_detection_config(config)


def test_legacy_defaults_and_immutable_normalized_group_weights():
    legacy = json.loads((ROOT / "configs/config.json").read_text())
    contract = resolve(legacy)
    assert (contract.head_mode, contract.box_mode, contract.groups) == ("legacy_single", "bev", ())
    config = candidate()
    original = copy.deepcopy(config)
    contract = resolve(config)
    assert tuple(g.name for g in contract.groups) == ("car", "ped_cyc")
    assert contract.group_weights == (("car", .5), ("ped_cyc", .5))
    assert config == original
    config["loss"]["group_weights"] = {"ped_cyc": 3, "car": 1}
    assert resolve(config).group_weights == (("car", .25), ("ped_cyc", .75))


@pytest.mark.parametrize("weights", [{}, {"car": 1}, {"car": 1, "ped_cyc": 1, "extra": 1},
    {"car": 0, "ped_cyc": 1}, {"car": -1, "ped_cyc": 1},
    {"car": float("nan"), "ped_cyc": 1}, {"car": float("inf"), "ped_cyc": 1},
    {"car": True, "ped_cyc": 1}, {"car": "1", "ped_cyc": 1}, None])
def test_invalid_weights(weights):
    config = candidate()
    config["loss"]["group_weights"] = weights
    with pytest.raises(ValueError, match="group_weights"):
        resolve(config)


@pytest.mark.parametrize("strategy", ["baseline", "oga", "uwag", "gw_qal", "q_oga"])
@pytest.mark.parametrize("classification", ["gaussian", "binary"])
def test_grouped_strategy_support_matrix(strategy, classification):
    config = candidate(loss=strategy, iqa=False, classification=classification)
    if classification == "binary" and strategy in ("q_oga", "gw_qal"):
        with pytest.raises(ValueError, match="binary grouped"):
            resolve(config)
    else:
        assert resolve(config).cls_encoding == classification


@pytest.mark.parametrize("strategy", ["baseline", "oga", "uwag", "gw_qal", "q_oga"])
def test_iqa_only_when_supervised_by_supported_strategy(strategy):
    config = candidate(loss=strategy)
    if strategy in ("baseline", "oga"):
        assert resolve(config).use_iou
    else:
        with pytest.raises(ValueError, match="IQA supervision"):
            resolve(config)


@pytest.mark.parametrize("section,key,value", [
    ("model", "head_mode", "unknown"), ("model", "cls_encoding", "multi_label"),
    ("model", "header_use_iou", "false"), ("loss", "use_iou", 1),
    ("data", "out_size_factor", 8), ("data", "out_size_factor", True),
    ("data", "num_classes", 2), ("data", "num_classes", True),
    ("data", "num_classes", None),
    ("data", "box_mode", "volumetric"), ("model", "box_mode", "3d"),
    ("model", "stage_depths", [3, True, 2]), ("model", "backbone_out_dim", 0),
])
def test_reject_invalid_shared_fields(section, key, value):
    config = candidate()
    config[section][key] = value
    with pytest.raises(ValueError):
        resolve(config)


def test_legacy_rejects_group_only_options_but_retains_unadvertised_binary_losses():
    for field in ["groups", "weights"]:
        config = candidate(grouped=False)
        if field == "groups":
            config["data"]["head_groups"] = []
        else:
            config["loss"]["group_weights"] = {}
        with pytest.raises(ValueError, match="legacy_single"):
            resolve(config)
    for strategy in ("q_oga", "gw_qal"):
        assert resolve(candidate(False, strategy, False, "binary")).head_mode == "legacy_single"


def test_legacy_rejects_an_explicit_null_global_class_map():
    config = candidate(grouped=False)
    config["data"]["kitti"]["objects"] = None
    with pytest.raises(ValueError, match="objects"):
        resolve(config)


def test_exact_qoga_restrictions_and_curriculum_validation():
    config = candidate(loss="q_oga", iqa=False)
    config["loss"].update(quality_target="rotated_iou", quality_warmup_epochs=5)
    assert resolve(config).quality_target == "rotated_iou"
    for field, value in [("quality_target", "bogus"), ("quality_warmup_epochs", True),
                         ("quality_warmup_epochs", -1), ("quality_warmup_epochs", 1.5)]:
        invalid = copy.deepcopy(config)
        invalid["loss"][field] = value
        with pytest.raises(ValueError):
            resolve(invalid)
    invalid = copy.deepcopy(config)
    invalid["loss"]["quality_target"] = "mgiou"
    with pytest.raises(ValueError, match="curriculum"):
        resolve(invalid)
    for grouped in (True, False):
        invalid = candidate(grouped, "q_oga", False, "binary")
        invalid["loss"]["quality_target"] = "rotated_iou"
        with pytest.raises(ValueError, match="Gaussian"):
            resolve(invalid)


def test_shared_backbone_contract_rejects_focal_litemla_conflict():
    config = candidate()
    config["model"]["c4_context"] = "focal"
    with pytest.raises(ValueError, match="mutually exclusive"):
        resolve(config)


@pytest.mark.parametrize("boundary", ["pipeline", "notebook"])
@pytest.mark.parametrize("invalid_case", ["mismatch", "unsupervised", "exact_binary", "groups"])
def test_public_boundaries_reject_invalid_configuration_before_construction(boundary, invalid_case):
    from tools.kitti_training_pipeline.common import build_model, configure_detector_imports
    from tools.kitti_training_pipeline.notebook_config import validate_notebook_config
    configure_detector_imports(ROOT / "detector")
    config = candidate()
    if invalid_case == "mismatch":
        config["loss"]["use_iou"] = False
    elif invalid_case == "unsupervised":
        config["loss"]["name"] = "uwag"
    elif invalid_case == "exact_binary":
        config = candidate(False, "q_oga", False, "binary")
        config["loss"]["quality_target"] = "rotated_iou"
    else:
        config["data"]["head_groups"].pop()
    with pytest.raises(ValueError):
        (build_model if boundary == "pipeline" else validate_notebook_config)(config)


def test_pure_detection_validation_has_no_tensor_backend_imports():
    script = "import sys; import detector.core.detection_config; assert not {'torch', 'numpy', 'numba'} & set(sys.modules)"
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_grouped_run_identity_distinguishes_mappings_weights_and_legacy():
    from tools.kitti_training_pipeline.common import generate_run_name
    config = candidate()
    base = generate_run_name(config)
    assert "grouped_" in base
    variants = [candidate(grouped=False)]
    reordered = copy.deepcopy(config)
    reordered["data"]["head_groups"][1]["classes"].reverse()
    variants.append(reordered)
    weighted = copy.deepcopy(config)
    weighted["loss"]["group_weights"] = {"car": 1, "ped_cyc": 3}
    variants.append(weighted)
    binary = copy.deepcopy(config)
    binary["model"]["cls_encoding"] = "binary"
    variants.append(binary)
    assert all(generate_run_name(v) != base for v in variants)
    equal_weights = copy.deepcopy(config)
    equal_weights["loss"]["group_weights"] = {"car": 2, "ped_cyc": 2}
    assert generate_run_name(equal_weights) == base


def test_existing_exact_quality_recipes_validate_through_both_boundaries():
    from tools.kitti_training_pipeline.notebook_config import validate_notebook_config
    paths = sorted((ROOT / "configs/experiments/qoga_quality").glob("*.json"))
    assert len(paths) == 3
    for path in paths:
        config = json.loads(path.read_text())
        assert resolve(config).quality_target in ("mgiou", "rotated_iou")
        validate_notebook_config(config)
