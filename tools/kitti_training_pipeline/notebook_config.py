"""Fresh, reproducible configuration resolution for the Colab notebook.

Precedence: base -> model selection -> augmentation recipe -> file override
-> notebook runtime. These helpers do not load datasets or GT databases.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

try:
    from .common import backbone_feature_spec, create_experiment_config, detection_spec, input_shape, minimum_bev_channels, read_json, write_json
except ImportError:
    from common import backbone_feature_spec, create_experiment_config, detection_spec, input_shape, minimum_bev_channels, read_json, write_json


MODEL_DEFAULTS = {
    "backbone": "mobilepixornext", "backbone_out_dim": 16,
    "neck_type": None, "scale_gated_fpn": True,
    "use_reparam": False, "header_use_iou": False,
    "header_use_bn": True, "header_act": "silu",
    "c4_attention": "none", "c4_attention_scales": [],
    "c4_attention_qk_norm": "none", "deploy": False,
}


UNDER1M_PRESETS = {
    "UNDER1M_SINGLE_OGA_IQA": "reference_single_oga_iqa",
    "UNDER1M_GROUPED_OGA_IQA": "reference_oga_iqa",
    "UNDER1M_GROUPED_BASELINE_IQA": "reference_baseline_iqa",
    "UNDER1M_FOCAL_OGA_IQA": "main_focal_oga_iqa",
    "UNDER1M_FOCAL_BASELINE_IQA": "main_focal_baseline_iqa",
    "UNDER1M_FOCAL_ECA_OGA_IQA": "optional_focal_eca",
    "UNDER1M_FOCAL_SIMAM_OGA_IQA": "optional_focal_simam",
    "UNDER1M_FOCAL_FUSION32_OGA_IQA": "optional_focal_fusion32",
    "UNDER1M_FOCAL_DETAIL_OGA_IQA": "optional_focal_detail",
}
UNDER1M_MANIFEST = "docs/plans/lightweight_lidar_backbone_2026/experiment_manifest.json"

# Built-in notebook recipes; configs/config.json is the only source JSON required.
UNDER1M_COMMON = {'data': {'bev_encoding': {'name': 'hist14',
                           'version': 1,
                           'density_norm': 32.0,
                           'intensity_scale': 1.0,
                           'backend': 'numba'},
          'out_size_factor': 4,
          'box_mode': 'bev',
          'head_groups': [{'name': 'car', 'classes': ['Car']},
                          {'name': 'ped_cyc', 'classes': ['Pedestrian', 'Cyclist']}]},
 'model': {'backbone': 'mobilepixornext',
           'head_mode': 'grouped',
           'stage_depths': [3, 4, 2],
           'backbone_out_dim': 32,
           'expansion': 2.5,
           'scale_gated_fpn': True,
           'neck_type': 'scale_gated_fpn',
           'c4_attention': 'none',
           'c4_attention_scales': [],
           'c4_attention_qk_norm': 'none',
           'c4_context': 'none',
           'local_attention': 'none',
           'neck_fusion_channels': 24,
           'detail_path': False,
           'header_use_bn': True,
           'header_act': 'silu',
           'header_use_iou': True,
           'use_reparam': False},
 'loss': {'name': 'oga',
          'use_iou': True,
          'iou_target_type': 'mgiou',
          'group_weights': {'car': 1.0, 'ped_cyc': 1.0}}}

UNDER1M_OVERRIDES = {'reference_single_oga_iqa': {'model': {'head_mode': 'legacy_single'}, 'data': {'box_mode': 'bev'}},
 'reference_oga_iqa': {'model': {}, 'data': {'box_mode': 'bev'}},
 'reference_baseline_iqa': {'model': {}, 'data': {'box_mode': 'bev'}, 'loss': {'name': 'baseline'}},
 'main_focal_oga_iqa': {'model': {'c4_context': 'focal',
                                  'c4_context_version': 1,
                                  'c4_context_bottleneck': 64,
                                  'c4_context_dilations': [1, 2, 3],
                                  'c4_context_layer_scale_init': 0.001},
                        'data': {'box_mode': 'bev'}},
 'main_focal_baseline_iqa': {'model': {'c4_context': 'focal',
                                       'c4_context_version': 1,
                                       'c4_context_bottleneck': 64,
                                       'c4_context_dilations': [1, 2, 3],
                                       'c4_context_layer_scale_init': 0.001},
                             'data': {'box_mode': 'bev'},
                             'loss': {'name': 'baseline'}},
 'optional_focal_eca': {'model': {'c4_context': 'focal',
                                  'c4_context_version': 1,
                                  'c4_context_bottleneck': 64,
                                  'c4_context_dilations': [1, 2, 3],
                                  'c4_context_layer_scale_init': 0.001,
                                  'local_attention': 'eca',
                                  'local_attention_eca_kernel_size': 3,
                                  'local_attention_layer_scale_init': 0.001},
                        'data': {'box_mode': 'bev'}},
 'optional_focal_simam': {'model': {'c4_context': 'focal',
                                    'c4_context_version': 1,
                                    'c4_context_bottleneck': 64,
                                    'c4_context_dilations': [1, 2, 3],
                                    'c4_context_layer_scale_init': 0.001,
                                    'local_attention': 'simam',
                                    'local_attention_simam_lambda': 0.0001,
                                    'local_attention_layer_scale_init': 0.001},
                          'data': {'box_mode': 'bev'}},
 'optional_focal_fusion32': {'model': {'c4_context': 'focal',
                                       'c4_context_version': 1,
                                       'c4_context_bottleneck': 64,
                                       'c4_context_dilations': [1, 2, 3],
                                       'c4_context_layer_scale_init': 0.001,
                                       'neck_fusion_channels': 32},
                             'data': {'box_mode': 'bev'}},
 'optional_focal_detail': {'model': {'c4_context': 'focal',
                                     'c4_context_version': 1,
                                     'c4_context_bottleneck': 64,
                                     'c4_context_dilations': [1, 2, 3],
                                     'c4_context_layer_scale_init': 0.001,
                                     'detail_path': True},
                           'data': {'box_mode': 'bev'}},
 'main_focal_3d': {'model': {'c4_context': 'focal',
                             'c4_context_version': 1,
                             'c4_context_bottleneck': 64,
                             'c4_context_dilations': [1, 2, 3],
                             'c4_context_layer_scale_init': 0.001},
                   'data': {'box_mode': '3d'},
                   'loss': {'vertical_loss_weight': 1.0}},
 'optional_focal_eca_3d': {'model': {'c4_context': 'focal',
                                     'c4_context_version': 1,
                                     'c4_context_bottleneck': 64,
                                     'c4_context_dilations': [1, 2, 3],
                                     'c4_context_layer_scale_init': 0.001,
                                     'local_attention': 'eca',
                                     'local_attention_eca_kernel_size': 3,
                                     'local_attention_layer_scale_init': 0.001},
                           'data': {'box_mode': '3d'},
                           'loss': {'vertical_loss_weight': 1.0}}}

AUGMENTATION_PROFILES = {'hybrid_gt': {'mode': 'openpcdet',
               'DISABLE_AUG_LIST': [],
               'AUG_CONFIG_LIST': [{'NAME': 'hybrid_gt_sampling',
                                    'PROBABILITY': 0.5,
                                    'DB_INFO_PATH': ['gt_database/dbinfos_train.json'],
                                    'NUM_POINT_FEATURES': 4,
                                    'USE_ROAD_PLANE': False,
                                    'PREPARE': {'filter_by_min_points': ['Car:5',
                                                                         'Pedestrian:5',
                                                                         'Cyclist:5']},
                                    'SAMPLE_GROUPS': ['Car:8', 'Pedestrian:6', 'Cyclist:6'],
                                    'LIMIT_WHOLE_SCENE': True,
                                    'SAMPLE_RATE': 1.0,
                                    'REMOVE_EXTRA_WIDTH': [0.0, 0.0, 0.0],
                                    'GEOMETRY_BACKEND': 'auto',
                                    'CACHE_SIZE_MB': 64,
                                    'PLACEMENT_MODE': 'source_relative',
                                    'RANGE_SCALE': [0.9, 1.1],
                                    'AZIMUTH_JITTER_DEG': 5,
                                    'MAX_PLACEMENT_ATTEMPTS': 6,
                                    'CANDIDATE_MULTIPLIER': 3,
                                    'COLLISION_MARGIN': 0.1,
                                    'MIN_SAMPLE_POINTS': 5,
                                    'MIN_VISIBLE_POINTS': 5,
                                    'MIN_VISIBLE_RATIO': 0.5,
                                    'ENABLE_GROUND_VALIDATION': True,
                                    'ENABLE_STATIC_COLLISION': True,
                                    'ENABLE_LINE_OF_SIGHT': False,
                                    'ENABLE_SHADOW_MASKING': False,
                                    'ENABLE_DENSITY_SUBSAMPLE': True,
                                    'ENABLE_RADIOMETRIC_CALIBRATION': False,
                                    'ENABLE_VISIBILITY_PROTECTION': True},
                                   {'NAME': 'random_world_flip',
                                    'ALONG_AXIS_LIST': ['x'],
                                    'PROBABILITY': 0.5},
                                   {'NAME': 'random_world_rotation',
                                    'WORLD_ROT_ANGLE': [-0.78539816, 0.78539816]},
                                   {'NAME': 'random_world_scaling',
                                    'WORLD_SCALE_RANGE': [0.95, 1.05]},
                                   {'NAME': 'random_world_translation',
                                    'NOISE_TRANSLATE_STD': [0.2, 0.2, 0.1],
                                    'PROBABILITY': 0.5}]},
 'openpcdet_gt': {'mode': 'openpcdet',
                  'p': 1.0,
                  'DISABLE_AUG_LIST': [],
                  'AUG_CONFIG_LIST': [{'NAME': 'gt_sampling',
                                       'DB_INFO_PATH': ['gt_database/dbinfos_train.json'],
                                       'NUM_POINT_FEATURES': 4,
                                       'USE_ROAD_PLANE': False,
                                       'PREPARE': {'filter_by_min_points': ['Car:5',
                                                                            'Pedestrian:5',
                                                                            'Cyclist:5']},
                                       'SAMPLE_GROUPS': ['Car:15', 'Pedestrian:10', 'Cyclist:10'],
                                       'LIMIT_WHOLE_SCENE': True,
                                       'REMOVE_EXTRA_WIDTH': [0.0, 0.0, 0.0]},
                                      {'NAME': 'random_world_flip',
                                       'ALONG_AXIS_LIST': ['x'],
                                       'PROBABILITY': 0.5},
                                      {'NAME': 'random_world_rotation',
                                       'WORLD_ROT_ANGLE': [-0.78539816, 0.78539816]},
                                      {'NAME': 'random_world_scaling',
                                       'WORLD_SCALE_RANGE': [0.95, 1.05]},
                                      {'NAME': 'random_world_translation',
                                       'NOISE_TRANSLATE_STD': [0.2, 0.2, 0.1],
                                       'PROBABILITY': 0.5}]},
 'openpcdet_global': {'mode': 'openpcdet',
                      'p': 1.0,
                      'DISABLE_AUG_LIST': [],
                      'AUG_CONFIG_LIST': [{'NAME': 'random_world_flip',
                                           'ALONG_AXIS_LIST': ['x'],
                                           'PROBABILITY': 0.5},
                                          {'NAME': 'random_world_rotation',
                                           'WORLD_ROT_ANGLE': [-0.78539816, 0.78539816]},
                                          {'NAME': 'random_world_scaling',
                                           'WORLD_SCALE_RANGE': [0.95, 1.05]},
                                          {'NAME': 'random_world_translation',
                                           'NOISE_TRANSLATE_STD': [0.2, 0.2, 0.1],
                                           'PROBABILITY': 0.5}]}}


def resolve_under1m_recipe(repo_dir, recipe_id, *, base_config=None):
    """Expand an implemented BEV recipe; never enable pending 3D variants."""
    if recipe_id not in UNDER1M_PRESETS.values():
        raise ValueError(f"Unknown or deferred under1m recipe: {recipe_id!r}")
    root = Path(repo_dir)
    base = read_json(root / "configs/config.json") if base_config is None else base_config
    base = copy.deepcopy(base)
    if base["data"].get("bev_encoding", {}).get("name") != "hist14":
        base["data"].setdefault("bev_encoding", {}).pop("out_channels", None)
    config = create_experiment_config(base, UNDER1M_COMMON)
    config = create_experiment_config(config, UNDER1M_OVERRIDES[recipe_id])
    config["augmentation"] = copy.deepcopy(AUGMENTATION_PROFILES["hybrid_gt"])
    if config["model"]["head_mode"] == "legacy_single":
        # The single-head reference intentionally removes the common group contract.
        config["data"].pop("head_groups", None)
        config["loss"].pop("group_weights", None)
    config["experiment"] = {"name": recipe_id}
    config["model"]["deploy"] = False
    validate_notebook_config(config)
    return config


def write_under1m_presets(repo_dir):
    """Materialize only verified BEV options into complete, deterministic JSON."""
    root = Path(repo_dir)
    manifest = read_json(root / UNDER1M_MANIFEST)
    paths = []
    for variant in manifest["variants"]:
        if variant["id"] not in UNDER1M_PRESETS.values():
            continue
        path = root / variant["planned_config"]
        write_json(path, resolve_under1m_recipe(root, variant["id"]))
        paths.append(path)
    return paths


def resolve_under1m_3d_recipe(repo_dir, recipe_id, *, base_config=None):
    """Resolve explicit 3D recipes without changing BEV notebook defaults."""
    companions = {"main_focal_3d": "main_focal_oga_iqa",
                  "optional_focal_eca_3d": "optional_focal_eca"}
    if recipe_id not in companions:
        raise ValueError(f"Unknown 3D under1m recipe: {recipe_id!r}")
    root = Path(repo_dir)
    config = resolve_under1m_recipe(root, companions[recipe_id], base_config=base_config)
    config = create_experiment_config(config, UNDER1M_OVERRIDES[recipe_id])
    config["experiment"] = {"name": recipe_id}
    validate_notebook_config(config)
    return config


def write_under1m_3d_presets(repo_dir):
    """Materialize the two explicit 3D options; never rewrite BEV recipes."""
    root = Path(repo_dir)
    manifest = read_json(root / UNDER1M_MANIFEST)
    paths = []
    for variant in manifest["variants"]:
        if variant["id"] not in {"main_focal_3d", "optional_focal_eca_3d"}:
            continue
        path = root / variant["planned_config"]
        write_json(path, resolve_under1m_3d_recipe(root, variant["id"]))
        paths.append(path)
    return paths


def resolve_notebook_config(
    repo_dir, *, base_path="configs/config.json", preset="custom",
    custom_overrides=None, augmentation="standard", file_override=None,
    runtime_overrides=None, hybrid_options=None,
):
    """Resolve a fresh configuration; never mutate defaults or cached selections."""
    root = Path(repo_dir)
    config = read_json(root / base_path)
    if preset == "custom":
        config = create_experiment_config(config, custom_overrides)
    elif preset in UNDER1M_PRESETS:
        config = resolve_under1m_recipe(root, UNDER1M_PRESETS[preset], base_config=config)
    elif preset != "config":
        if preset not in PRESET_CONFIGS:
            raise ValueError(f"Unknown PRESET: {preset!r}; choose custom, config or "
                             f"{list(PRESET_CONFIGS) + list(UNDER1M_PRESETS)}")
        defaults = {"model": MODEL_DEFAULTS, "loss": {"name": "baseline", "use_iou": False}}
        recipe = create_experiment_config(defaults, PRESET_CONFIGS[preset])
        config = create_experiment_config(config, recipe)
        encoding = config["data"]["bev_encoding"]
        encoding.pop("out_channels", None)
        channels = minimum_bev_channels(encoding["name"])
        if channels is not None:
            encoding["out_channels"] = channels

    if augmentation in ("standard", "compose", "none"):
        # Replace the whole recipe so an OpenPCDet queue cannot survive a mode switch.
        config["augmentation"] = {
            "mode": "compose" if augmentation == "compose" else "one_of",
            "p": 0.0 if augmentation == "none" else 0.5,
            "rotation": {"use": True, "limit_angle": 20, "p": 1},
            "scaling": {"use": True, "range": [0.95, 1.05], "p": 1},
            "translation": {"use": True, "scale": 0.4, "scale_z": 0.4, "p": 1},
        }
    elif augmentation in ("openpcdet_global", "openpcdet_gt", "hybrid_gt"):
        config["augmentation"] = copy.deepcopy(AUGMENTATION_PROFILES[augmentation])
        if augmentation == "hybrid_gt" and hybrid_options:
            if "NAME" in hybrid_options:
                raise ValueError("HYBRID_OPTIONS cannot override operator NAME")
            config["augmentation"]["AUG_CONFIG_LIST"][0] = create_experiment_config(
                config["augmentation"]["AUG_CONFIG_LIST"][0], hybrid_options)
    elif augmentation != "config":
        raise ValueError(f"Unknown AUGMENTATION: {augmentation!r}")

    if file_override:
        overrides = read_json(root / file_override)
        # An override changing the mode starts a new recipe, rather than merging stale keys.
        aug = overrides.get("augmentation")
        if aug is not None and aug.get("mode", config["augmentation"].get("mode")) != config["augmentation"].get("mode"):
            config["augmentation"] = {}
        bev = overrides.get("data", {}).get("bev_encoding", {})
        if "name" in bev and bev["name"] != config["data"]["bev_encoding"].get("name"):
            config["data"]["bev_encoding"].pop("out_channels", None)
        config = create_experiment_config(config, overrides)
        # Changing only the IQA flag also changes its loss supervision.
        if "header_use_iou" in overrides.get("model", {}) and "use_iou" not in overrides.get("loss", {}):
            config["loss"]["use_iou"] = config["model"]["header_use_iou"]
    config = create_experiment_config(config, runtime_overrides)
    encoding = config["data"].get("bev_encoding", {})
    if encoding.get("name") in {"pillar32", "pillar_rich", "pillar_rich_gate"}:
        # Older notebook width lookups produce None for newly added encoders.
        # Resolve that unspecified width here; keep the core schema strict and
        # preserve validation of explicit widths/backends.
        if encoding.get("out_channels") is None:
            encoding["out_channels"] = 32
        encoding.setdefault("backend", "torch")
    validate_notebook_config(config)
    return config


def validate_notebook_config(config):
    """Reject options the advertised notebook backbones cannot actually honor."""
    model, loss, train = config["model"], config["loss"], config["train"]
    try:
        from .checkpoint_selection import selection_settings
    except ImportError:
        from checkpoint_selection import selection_settings
    selection_settings(config)
    detection_spec(config)
    evaluation = config.get("evaluation", {})
    for key, default in (("score_threshold", .05), ("nms_threshold", .10)):
        value = evaluation.get(key, default)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"evaluation.{key} must be finite and between 0 and 1")
    cap = evaluation.get("max_detections", 500)
    if type(cap) is not int or cap < 1:
        raise ValueError("evaluation.max_detections must be a positive integer")
    kitti = config['data']['kitti']
    alpha = config.get('nms_alpha', kitti.get('nms_alpha', .5))
    if type(alpha) not in (int, float) or not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError('nms_alpha must be finite and between 0 and 1')
    if config.get('peak_mode', kitti.get('peak_mode', 'per_class')) not in {'per_class', 'legacy'}:
        raise ValueError('peak_mode must be per_class or legacy')
    # Validate before legacy preset sanitation can erase conflicting fields.
    features = backbone_feature_spec(model, geometry=config["data"]["kitti"]["geometry"])
    if model.get("backbone") == "mobilepixornext":
        model.update(features.to_dict())
    backbone = model.get("backbone")
    if backbone not in ("mobilepixornext", "mobilepixor", "mobilepixor_coordatt"):
        raise ValueError(f"Unsupported notebook BACKBONE: {backbone!r}")
    if model.get("neck_type") not in (None, "scale_gated_fpn", "sgfpn", "rc_sgfpn", "rc_bisgfpn"):
        raise ValueError("Unsupported NECK_TYPE")
    channels = model.get("backbone_out_dim", 16)
    if isinstance(channels, bool) or not isinstance(channels, int) or channels <= 0:
        raise ValueError("BACKBONE_OUT_DIM must be a positive integer")
    if backbone != "mobilepixornext":
        if channels != 16:
            raise ValueError("BACKBONE_OUT_DIM must be 16 for MobilePIXOR backbones")
        if model.get("neck_type") in ("rc_sgfpn", "rc_bisgfpn") or model.get("use_reparam", False):
            raise ValueError("Range-conditioned necks and USE_REPARAM require mobilepixornext")
        model["c4_attention"] = "none"
    input_shape(config)
    epochs, warmup = train["epochs"], train.get("warmup_epochs", 0)
    if not isinstance(epochs, int) or not isinstance(warmup, int) or epochs <= 0 or not 0 <= warmup < epochs:
        raise ValueError("warmup_epochs must satisfy 0 <= warmup_epochs < epochs")
    for section, name in ((train, "physical_batch_size"), (train, "accumulation_steps"), (config["val"], "physical_batch_size")):
        if not isinstance(section[name], int) or isinstance(section[name], bool) or section[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(train["learning_rate"]) or train["learning_rate"] <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if train["precision"] not in ("fp32", "fp16", "bf16"):
        raise ValueError("Unsupported PRECISION")
    if train["target_backend"] not in ("python", "numba"):
        raise ValueError("Unsupported TARGET_BACKEND")
    if not isinstance(train["num_workers"], int) or train["num_workers"] < 0:
        raise ValueError("NUM_WORKERS must be a nonnegative integer")
    aug = config["augmentation"]
    if aug.get("mode", "one_of") not in ("one_of", "compose", "openpcdet"):
        raise ValueError("Unsupported augmentation.mode")
    if aug.get("mode") == "openpcdet":
        if not isinstance(aug.get("AUG_CONFIG_LIST"), list):
            raise ValueError("OpenPCDet augmentation requires AUG_CONFIG_LIST")
    elif not 0 <= aug.get("p", 0) <= 1:
        raise ValueError("augmentation.p must be between 0 and 1")


def save_notebook_config(run_dir, config):
    """Keep the original config intact if a saved training run would be incompatible."""
    run_dir = Path(run_dir)
    if run_dir.name in (".", ".."):
        raise ValueError("Invalid run directory")
    path = run_dir / "config.json"
    trained = (run_dir / "run.json").is_file() or any((run_dir / "checkpoints").glob("*.pt"))
    if path.is_file() and trained and read_json(path) != config:
        saved, requested = read_json(path), copy.deepcopy(config)
        for candidate in (saved, requested):
            model = candidate.get("model", {})
            if model.get("rc_gate_mode", "mul") == "mul":
                model.pop("rc_gate_mode", None)
        if saved == requested:
            # Keep the original snapshot/hash when a new notebook only makes
            # the historical multiplicative default explicit.
            return path
        raise ValueError(
            "Run đã có metadata/checkpoint với config khác. Đổi EXPERIMENT_TAG hoặc "
            "CUSTOM_RUN_NAME để tạo run mới; config cũ được giữ nguyên."
        )
    write_json(path, config)
    return path


ENCODER_COMPARISON_MODEL = {
    "backbone": "mobilepixornext", "backbone_out_dim": 16,
    "head_mode": "legacy_single", "cls_encoding": "gaussian",
    "stage_depths": [3, 4, 2], "expansion": 2.5,
    "neck_type": "scale_gated_fpn", "neck_fusion_channels": 24,
    "scale_gated_fpn": False, "detail_path": False,
    "c4_attention": "none", "c4_attention_scales": [], "c4_attention_qk_norm": "none",
    "c4_context": "none", "local_attention": "none",
    "header_use_bn": True, "header_act": "silu", "header_use_iou": False,
    "use_reparam": False, "deploy": False,
}

PRESET_CONFIGS = {
    "ENCODER_RICH8": {
        "experiment": {"name": "encoder_comparison"},
        "model": copy.deepcopy(ENCODER_COMPARISON_MODEL),
        "loss": {"name": "baseline", "use_iou": False},
        "data": {"bev_encoding": {"name": "rich8", "intensity_scale": 1, "density_norm": 32}},
    },
    "ENCODER_PILLAR32": {
        "experiment": {"name": "encoder_comparison"},
        "model": copy.deepcopy(ENCODER_COMPARISON_MODEL),
        "loss": {"name": "baseline", "use_iou": False},
        "data": {"bev_encoding": {"name": "pillar32", "version": 1, "backend": "torch",
                                  "out_channels": 32, "intensity_scale": 1}},
    },
    "ENCODER_PILLAR_RICH": {
        "experiment": {"name": "encoder_comparison"},
        "model": copy.deepcopy(ENCODER_COMPARISON_MODEL),
        "loss": {"name": "baseline", "use_iou": False},
        "data": {"bev_encoding": {"name": "pillar_rich", "version": 1, "backend": "torch",
                                  "out_channels": 32, "intensity_scale": 1, "density_norm": 32}},
    },
    "ENCODER_PILLAR_RICH_GATE": {
        "experiment": {"name": "encoder_comparison"},
        "model": copy.deepcopy(ENCODER_COMPARISON_MODEL),
        "loss": {"name": "baseline", "use_iou": False},
        "data": {"bev_encoding": {"name": "pillar_rich_gate", "version": 1, "backend": "torch",
                                  "out_channels": 32, "intensity_scale": 1, "density_norm": 32}},
    },
    "TRUC_A_M1": {
        "model": {
            "backbone": "mobilepixornext",
            "scale_gated_fpn": True,
            "use_reparam": True,
            "header_use_iou": False,
            "c4_attention": "none"
        },
        "loss": {
            "name": "oga",
            "use_iou": False
        },
        "data": {
            "bev_encoding": {
                "name": "rich8"
            }
        }
    },
    "TRUC_B_M1": {
        "model": {
            "backbone": "mobilepixornext",
            "scale_gated_fpn": True,
            "use_reparam": True,
            "header_use_iou": False,
            "c4_attention": "none"
        },
        "loss": {
            "name": "q_oga",
            "use_iou": False
        },
        "data": {
            "bev_encoding": {
                "name": "rich8"
            }
        }
    },
    "RICH8_SGFPN": {
        "model": {
            "backbone": "mobilepixornext",
            "scale_gated_fpn": True
        },
        "data": {
            "bev_encoding": {
                "name": "rich8"
            }
        }
    },
    "RC_SGFPN": {
        "model": {
            "backbone": "mobilepixornext",
            "neck_type": "rc_sgfpn",
            "scale_gated_fpn": True
        },
        "loss": {
            "name": "q_oga"
        },
        "data": {
            "bev_encoding": {
                "name": "rich8"
            }
        }
    },
    "RC_BISGFPN": {
        "model": {
            "backbone": "mobilepixornext",
            "neck_type": "rc_bisgfpn",
            "scale_gated_fpn": True
        },
        "loss": {
            "name": "q_oga"
        },
        "data": {
            "bev_encoding": {
                "name": "rich8"
            }
        }
    },
    "MOBILEPIXOR_BASELINE": {
        "model": {
            "backbone": "mobilepixor",
            "scale_gated_fpn": False,
            "c4_attention": "none",
            "header_use_bn": False,
            "header_act": "relu"
        },
        "loss": {
            "name": "baseline"
        },
        "data": {
            "bev_encoding": {
                "name": "rich8"
            }
        }
    },
    "LEGACY35_BASELINE": {
        "model": {
            "backbone": "mobilepixor",
            "scale_gated_fpn": False,
            "c4_attention": "none",
            "header_use_bn": False,
            "header_act": "relu"
        },
        "loss": {
            "name": "baseline"
        },
        "data": {
            "bev_encoding": {
                "name": "binary_slices"
            }
        }
    }
}
