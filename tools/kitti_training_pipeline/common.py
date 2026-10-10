#!/usr/bin/env python3
"""Shared helpers for the KITTI training/export pipeline."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Tuple


def _bev_schema():
    """Load the pure contract lazily, including standalone notebook imports."""
    try:
        from core import bev_encoding
    except ImportError:
        # Notebook cells can expose only this tools directory from another cwd.
        # Append the repository path so a configured detector retains priority.
        root = str(Path(__file__).resolve().parents[2])
        if root not in sys.path:
            sys.path.append(root)
        from detector.core import bev_encoding
    return bev_encoding


def minimum_bev_channels(name: str) -> int | None:
    return _bev_schema().minimum_channels(name)


def backbone_feature_spec(model, *, geometry=None):
    """Load shared feature validation lazily, including standalone notebooks."""
    module = importlib.import_module(_bev_schema().__package__ + ".backbone_config")
    normalized = copy.deepcopy(model)
    if normalized.get("neck_type") in (None, "sgfpn"):
        normalized["neck_type"] = "scale_gated_fpn"
    return module.resolve_backbone_features(normalized, geometry=geometry)


def bev_encoding_spec(config: Dict[str, Any], dataset_name: str = "kitti"):
    """Resolve the data-side schema without loading PyTorch or a dataset."""
    data = config["data"]
    return _bev_schema().resolve_bev_encoding(
        data.get("bev_encoding"), data[dataset_name]["geometry"]
    )


def detection_spec(config):
    """Shared pure detection validation for CLI, evaluator and notebook callers."""
    module = importlib.import_module(_bev_schema().__package__ + ".detection_config")
    return module.resolve_detection_config(config)


def evaluation_asset_hashes(config, frame_ids, kitti_root, *, reference=False):
    """Record actual per-frame inputs used by local or reference evaluation."""
    processed = Path(config["data"]["kitti"]["location"]).resolve()
    raw = Path(kitti_root).resolve()
    if (raw / "training").is_dir():
        raw = raw / "training"
    paths = []
    for identifier in frame_ids:
        paths.extend([processed / "pointcloud" / f"{identifier}.bin",
                      raw / "label_2" / f"{identifier}.txt",
                      raw / "calib" / f"{identifier}.txt"])
        if reference:
            # Reference inference constructs validation targets as well; the
            # historical local test Dataset consumes only the point cloud.
            paths.append(processed / "label" / f"{identifier}.txt")
            paths.append(raw / "image_2" / f"{identifier}.png")
    return {str(path.resolve()): sha256(path) for path in paths}


def verify_retained_checkpoint(selected_path, retained_path, *, selected_state=None):
    """Accept separate torch.save archives only when their full payloads agree."""
    import torch
    if sha256(selected_path) == sha256(retained_path):
        return True
    selected = (torch.load(selected_path, map_location='cpu', weights_only=True)
                if selected_state is None else selected_state)
    retained = torch.load(retained_path, map_location='cpu', weights_only=True)
    pairs = [(selected, retained)]
    while pairs:
        left, right = pairs.pop()
        if isinstance(left, torch.Tensor):
            same = (isinstance(right, torch.Tensor) and left.dtype == right.dtype and
                    left.shape == right.shape and torch.equal(left, right))
        elif isinstance(left, dict):
            same = isinstance(right, dict) and left.keys() == right.keys()
            if same:
                pairs.extend((left[key], right[key]) for key in left)
        elif isinstance(left, (tuple, list)):
            same = type(left) is type(right) and len(left) == len(right)
            if same:
                pairs.extend(zip(left, right))
        else:
            same = type(left) is type(right) and left == right
        if not same:
            raise ValueError('Selected checkpoint state differs from the retained checkpoint')
    return True


def checkpoint_identity(config):
    """Versioned semantic identity; rasterizer backends are recorded separately.

    Normalize shared schema/feature defaults and fixed group weights. Preserve
    additional model/loss settings conservatively rather than silently ignoring
    options that could alter computation without changing tensor shapes.
    """
    detection = detection_spec(config)
    encoding = bev_encoding_spec(config)
    model = copy.deepcopy(config["model"])
    backbone = str(model.get("backbone", "mobilepixor")).lower()
    next_backbone = backbone == "mobilepixornext"
    model.update(backbone=backbone, head_mode=detection.head_mode,
                 cls_encoding=detection.cls_encoding,
                 backbone_out_dim=model.get("backbone_out_dim", 16),
                 stage_depths=list(model.get("stage_depths", (2, 4, 2))),
                 header_use_bn=model.get("header_use_bn", next_backbone),
                 header_act=model.get("header_act", "silu" if next_backbone else "none"),
                 header_use_iou=detection.use_iou)
    model.pop("bev_encoding", None)
    model.pop("geometry", None)
    if detection.backbone_features is not None:
        features = detection.backbone_features.semantic_metadata()
        for key in detection.backbone_features.__dataclass_fields__:
            model.pop(key, None)
        model["features"] = features
    if next_backbone:
        for key, default in {"scale_gated_fpn": True, "expansion": 2.5,
                             "use_reparam": False, "deploy": False,
                             "neck_type": "scale_gated_fpn", "num_range_bands": 4}.items():
            model.setdefault(key, default)
        if model["neck_type"] in (None, "sgfpn"):
            model["neck_type"] = "scale_gated_fpn"
    # Omitted and explicit mul retain historical checkpoint identities,
    # including custom notebooks selecting a backbone without an RC neck.
    # Additive gating has the same training tensor shapes but new semantics.
    gate_mode = str(model.get("rc_gate_mode", "mul")).lower()
    if gate_mode == "mul":
        model.pop("rc_gate_mode", None)
    else:
        model["rc_gate_mode"] = gate_mode
    objective = copy.deepcopy(config.get("loss", {}))
    objective.update(name=str(objective.get("name", "baseline")).lower(),
                     use_iou=detection.use_iou)
    objective.pop("group_weights", None)
    if detection.use_iou:
        objective.setdefault("iou_target_type", "mgiou")
        objective.setdefault("iou_loss_weight", 1.0)
    if objective["name"] == "q_oga":
        objective.update(quality_target=detection.quality_target,
                         quality_warmup_epochs=objective.get("quality_warmup_epochs", 0))
    train = config.get("train", {})
    training = {key: train[key] for key in (
        "learning_rate", "epochs", "lr_decay_at", "lr_decay_gamma",
    ) if key in train}
    for key, default in {"optimizer": "adamw", "scheduler": "cosine", "weight_decay": .0001,
                         "warmup_epochs": 5, "min_lr": 1e-6, "precision": "fp32",
                         "physical_batch_size": train.get("batch_size", 2),
                         "accumulation_steps": 1, "grad_clip_norm": 10.0}.items():
        training[key] = train.get(key, default)
    objects = config["data"]["kitti"]["objects"]
    identity = {"version": 1, "encoding": {"metadata": encoding.semantic_metadata(),
                                               "sha256": encoding.semantic_hash},
                "model": model, "box_mode": detection.box_mode,
                "output_stride": config["data"].get("out_size_factor", 4),
                "class_order": [name for name, _ in sorted(objects.items(), key=lambda item: item[1])],
                "groups": [group.to_dict() for group in detection.groups],
                "group_weights": list(detection.group_weights), "objective": objective,
                "training": training}
    if 'checkpoint_selection' in train:
        try:
            from .checkpoint_selection import selection_protocol
        except ImportError:
            from checkpoint_selection import selection_protocol
        identity['training']['checkpoint_selection'] = selection_protocol(config)
        if train['checkpoint_selection'].get('primary') == 'ap':
            evaluation, kitti = config.get('evaluation', {}), config['data']['kitti']
            identity['training']['selection_decode'] = {
                'score_threshold': evaluation.get('score_threshold', .05),
                'nms_threshold': evaluation.get('nms_threshold', .10),
                'max_detections': evaluation.get('max_detections', 500),
                'nms_alpha': config.get('nms_alpha', kitti.get('nms_alpha', .5)) if detection.use_iou else 0.,
                'peak_mode': config.get('peak_mode', kitti.get('peak_mode', 'per_class'))}
    # JSON canonicalization also ensures tuples/lists serialize identically.
    return json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))


def checkpoint_backends(config):
    return {"bev": bev_encoding_spec(config).backend,
            "targets": config.get("train", {}).get("target_backend", "python")}


def validate_evaluation_checkpoint(checkpoint, config):
    """Check architecture/objective semantics without requiring resume state."""
    identity = checkpoint.get("checkpoint_identity") if isinstance(checkpoint, dict) else None
    if identity is None:
        if detection_spec(config).head_mode == "grouped":
            raise ValueError("Grouped evaluation requires checkpoint identity metadata")
        return
    if not isinstance(identity, dict):
        raise ValueError("Invalid checkpoint identity metadata")
    current = checkpoint_identity(config)
    saved = copy.deepcopy(identity)
    # Epochs/optimizer/precision used to train are provenance, not model-only
    # inference settings. All architecture and objective fields still match.
    current.pop("training", None)
    saved.pop("training", None)
    if current != saved:
        raise ValueError("Evaluation architecture/objective identity does not match checkpoint")
    if "config" in checkpoint and checkpoint_identity(checkpoint["config"]) != identity:
        raise ValueError("Checkpoint identity is inconsistent with saved config")


def validate_deployment_config(config, consumer):
    """Reject contracts the current four-output deployment consumers cannot honor."""
    schema = bev_encoding_spec(config)
    if schema.is_packed:
        raise ValueError(f"{consumer}: {schema.name} packed-point deployment is not supported; use PyTorch evaluation")
    detection = detection_spec(config)
    if detection.box_mode == "3d":
        raise ValueError(f"{consumer}: 3D export/deployment is deferred; vertical output must not be dropped")
    if detection.head_mode == "grouped":
        raise ValueError(f"{consumer}: grouped deployment is deferred; use PyTorch evaluation")
    if detection.use_iou:
        raise ValueError(f"{consumer}: IQA deployment is deferred; quality output must not be dropped")


def model_parameter_report(model, criterion=None):
    """Count actual modules, separating train-only parameters from inference.

    Body/neck separation follows MobilePixorNeXt's registered neck components;
    other backbones retain the combined count without guessing their split.
    """
    backbone = sum(parameter.numel() for parameter in model.backbone.parameters())
    grouped = getattr(model, "head_mode", "legacy_single") == "grouped"
    header = model.grouped_header if grouped else model.header
    heads = ({name: sum(p.numel() for p in head.parameters()) for name, head in header.heads.items()}
             if grouped else {"legacy_single": sum(p.numel() for p in header.parameters())})
    counts = {"backbone_including_neck": backbone, "heads": sum(heads.values()),
              "total_detector": sum(p.numel() for p in model.parameters()),
              "criterion_train_only": (sum(p.numel() for p in criterion.parameters())
                                       if criterion is not None else None)}
    encoder = getattr(model, "point_encoder", None)
    if encoder is not None:
        counts["point_encoder"] = sum(p.numel() for p in encoder.parameters())
    if hasattr(model.backbone, "c4_context"):
        neck_names = {"rc_neck", "lat_c5", "lat_c4", "lat_c3", "refine_u4", "proj_u3",
                      "gate_c4", "gate_c3", "out_conv", "detail_branch"}
        neck = sum(p.numel() for name, module in model.backbone.named_children()
                   if name in neck_names for p in module.parameters())
        detail_scale = model.backbone.detail_gamma
        if detail_scale is not None:
            neck += detail_scale.numel()
        counts.update(backbone_body=backbone - neck, neck=neck)
    return {"parameter_counts": counts, "head_parameter_counts": heads}


def read_json(path: Path | str) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def generate_run_name(
    config: Dict[str, Any],
    seed: int | None = None,
    effective_batch_size: int | None = None,
    extra_tags: list[str] | None = None,
) -> str:
    """Generate standardized run name following:
    {backbone}-{augmentation}-{loss}-{bev_encoder}-{iou}[-{extra}]
    """
    model_cfg = config.get("model", {})
    backbone = str(model_cfg.get("backbone", "unknown")).lower()

    # Augmentation
    aug_cfg = config.get("augmentation", {})
    if aug_cfg.get("mode") == "openpcdet":
        disabled = set(aug_cfg.get("DISABLE_AUG_LIST", []))
        active = [entry for entry in aug_cfg.get("AUG_CONFIG_LIST", [])
                  if entry.get("NAME") not in disabled and entry.get("PROBABILITY", 1) > 0]
        if not active:
            augmentation = "noaug"
        elif any(entry.get("NAME") == "hybrid_gt_sampling" for entry in active):
            augmentation = "hybrid_gt_aug"
        elif any(entry.get("NAME") == "gt_sampling" for entry in active):
            augmentation = "openpcdet_gt_aug"
        else:
            augmentation = "openpcdet_aug"
    elif aug_cfg.get("use_pcu_aug", False):
        augmentation = "pcu"
    elif aug_cfg.get("p", 0.0) > 0.0 and any(
        isinstance(v, dict) and v.get("use", False) for v in aug_cfg.values()
    ):
        augmentation = "standard_aug"
    elif aug_cfg.get("p", 0.0) > 0.0:
        augmentation = "standard_aug"
    else:
        augmentation = "noaug"
    if augmentation == "standard_aug" and aug_cfg.get("mode") == "compose":
        augmentation = "compose_aug"

    # Loss
    loss_cfg = config.get("loss", {})
    loss_raw = str(loss_cfg.get("name", "baseline")).lower()
    loss_name = loss_raw if loss_raw.endswith("_loss") else f"{loss_raw}_loss"

    # BEV encoder
    bev_cfg = config.get("data", {}).get("bev_encoding", {})
    bev_name = bev_cfg.get("name")
    if bev_name in {"rich8", "hist14", "pillar32", "pillar_rich", "pillar_rich_gate"}:
        bev_encoder = str(bev_name)
    else:
        bev_encoder = "legacy35"

    # IOU Quality Aware
    if model_cfg.get("header_use_iou", False) or loss_cfg.get("use_iou", False):
        iou = "iqa"
    else:
        iou = "baseline_iou"

    parts = [backbone, augmentation, loss_name, bev_encoder, iou]

    extras: list[str] = []
    neck_type = model_cfg.get("neck_type")
    if neck_type in ("rc_sgfpn", "rc_bisgfpn"):
        extras.append(neck_type)
        if str(model_cfg.get("rc_gate_mode", "mul")).lower() == "add":
            extras.append("rcadd")
    elif model_cfg.get("scale_gated_fpn", False):
        extras.append("sgfpn")
    if model_cfg.get("use_reparam", False):
        extras.append("reparam")
    if backbone == "mobilepixornext" and (
            model_cfg.get("c4_context", "none") != "none" or
            model_cfg.get("local_attention", "none") != "none" or
            model_cfg.get("detail_path", False) or model_cfg.get("neck_fusion_channels", 24) != 24 or
            tuple(model_cfg.get("stage_depths", (2, 4, 2))) != (2, 4, 2)):
        features = backbone_feature_spec(model_cfg)
        identity = {"features": features.semantic_metadata(),
                    "stage_depths": model_cfg.get("stage_depths", [2, 4, 2]),
                    "backbone_out_dim": model_cfg.get("backbone_out_dim", 16)}
        kitti = config.get("data", {}).get("kitti", {})
        if "geometry" in kitti and "objects" in kitti:
            # New candidate names cover encoding and objective semantics as well
            # as topology. Historical default names and partial configs stay usable.
            identity = checkpoint_identity(config)
            identity.pop("training")
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
        extras.append(f"ctx_{features.c4_context}_local_{features.local_attention}_"
                      f"f{features.neck_fusion_channels}_detail{int(features.detail_path)}_{digest}")
    if str(model_cfg.get("head_mode", "legacy_single")).lower() == "grouped":
        detection = detection_spec(config)
        identity = {"groups": [g.to_dict() for g in detection.groups],
                    "group_weights": detection.group_weights,
                    "cls_encoding": detection.cls_encoding, "box_mode": detection.box_mode,
                    "backbone_out_dim": model_cfg.get("backbone_out_dim", 16),
                    "quality_target": detection.quality_target,
                    "quality_warmup_epochs": loss_cfg.get("quality_warmup_epochs", 0)}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
        extras.append(f"grouped_{digest}")
    experiment_name = config.get("experiment", {}).get("name")
    if experiment_name:
        if not isinstance(experiment_name, str) or any(
            not (character.isalnum() or character == "_") for character in experiment_name
        ):
            raise ValueError("experiment.name must contain only letters, numbers and underscores")
        if experiment_name != "default":
            extras.append(experiment_name)
    if seed is not None:
        extras.append(f"s{seed}")
    if effective_batch_size is not None:
        extras.append(f"eb{effective_batch_size}")
    if extra_tags:
        extras.extend(extra_tags)

    if extras:
        return f"{'-'.join(parts)}-{'-'.join(extras)}"
    return "-".join(parts)


def create_experiment_config(base_config: Dict[str, Any], overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Create a new experiment config dictionary by deep-merging overrides into base_config."""
    import copy

    result = copy.deepcopy(base_config)
    if not overrides:
        return result

    def _deep_update(dst: dict, src: dict) -> dict:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                _deep_update(dst[k], v)
            else:
                dst[k] = copy.deepcopy(v)
        return dst

    return _deep_update(result, overrides)


def write_json(path: Path | str, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def configure_detector_imports(detector_root: Path | str) -> Path:
    """Expose the copied detector package and its legacy utils_1 imports."""
    detector_root = Path(detector_root).expanduser().resolve()
    core_root = detector_root / "core"
    datasets_root = core_root / "datasets"
    required = [
        core_root / "models" / "model.py",
        core_root / "datasets" / "dataset.py",
        core_root / "losses" / "loss_fn.py",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Detector source is incomplete. Missing:\n  " + "\n  ".join(missing)
        )
    for path in (detector_root.parent, detector_root, datasets_root):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    return detector_root


def validate_backbone(config: Dict[str, Any]) -> None:
    from core.models.backbones.registry import get_available_backbones

    backbone = config["model"]["backbone"]
    available = get_available_backbones()
    if str(backbone).lower() not in available:
        raise ValueError(
            f"Unsupported backbone: {backbone!r}. "
            f"Available backbones: {', '.join(available)}"
        )


def build_model(config: Dict[str, Any]):
    detection = detection_spec(config)
    validate_backbone(config)
    from core.models.model import CustomModel

    schema = bev_encoding_spec(config)
    model_cfg = copy.deepcopy(config["model"])
    model_cfg["geometry"] = copy.deepcopy(config["data"]["kitti"]["geometry"])
    model_cfg["bev_encoding"] = copy.deepcopy(
        config["data"].get("bev_encoding") or {"name": "binary_slices"}
    )

    return CustomModel(
        model_cfg,
        config["data"]["num_classes"],
        input_channels=schema.channels,
        task_groups=detection.groups if detection.head_mode == "grouped" else None,
        box_mode=detection.box_mode,
    )


def input_shape(config: Dict[str, Any], dataset_name: str = "kitti") -> Tuple[int, ...]:
    """Return the dense BEV backbone shape (packed encoders have separate inputs)."""
    return bev_encoding_spec(config, dataset_name).input_shape


def model_input_batch_size(value):
    """Count frames in either a dense BEV tensor or packed pillar input."""
    return value["batch_size"] if isinstance(value, dict) else int(value.shape[0])


def dummy_model_input(config, batch_size, device):
    """Build a nonempty compile/profile input that includes learned encoding."""
    import torch
    schema = bev_encoding_spec(config)
    if not schema.is_packed:
        return torch.zeros((batch_size, *schema.input_shape[1:]), device=device)
    features = torch.zeros((batch_size * 2, 10), device=device)
    features[1::2] = .1
    coords = torch.zeros((batch_size, 3), dtype=torch.int64, device=device)
    coords[:, 0] = torch.arange(batch_size, device=device)
    result = {"features": features, "coords": coords, "batch_size": batch_size,
              "pillar_indices": torch.arange(batch_size, device=device).repeat_interleave(2)}
    if schema.is_rich_pillar:
        result["rich_features"] = torch.zeros((batch_size, 8), device=device)
    return result


def normalize_state_dict(checkpoint: Any) -> Dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict):
                checkpoint = nested
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint does not contain a model state dictionary")
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in checkpoint.items()
    }


def warm_start_backbone(model, checkpoint):
    """Copy exact compatible backbone keys only; leave new branches initialized.

    This helper has no access to criterion/optimizer/epoch state. Normalize
    legacy checkpoint wrappers and DDP names, without guessing key mappings or
    expanding the first input convolution.
    """
    import torch

    target = getattr(model, "_orig_mod", model)
    source = normalize_state_dict(checkpoint)
    destination = target.state_dict()
    loaded, skipped = {}, {}
    for name, value in source.items():
        if not name.startswith("backbone."):
            skipped[name] = "not a backbone key"
        elif not torch.is_tensor(value):
            raise ValueError(f"Invalid warm-start backbone tensor: {name}")
        elif name not in destination:
            skipped[name] = "absent in destination"
        elif value.shape != destination[name].shape:
            skipped[name] = "shape mismatch"
        elif value.dtype != destination[name].dtype:
            skipped[name] = "dtype mismatch"
        else:
            loaded[name] = value
    if not loaded:
        raise ValueError("Unsupported warm-start: no compatible backbone tensors")
    # Validate the entire selection before modifying any actual tensor.
    updated = dict(destination)
    updated.update(loaded)
    target.load_state_dict(updated, strict=True)
    return {"loaded": sorted(loaded), "skipped": dict(sorted(skipped.items())),
            "missing": sorted(set(destination) - set(loaded))}


def atomic_torch_save(value: Any, path: Path | str) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_metadata(path: Path | str) -> Dict[str, Any]:
    try:
        root = Path(path).resolve()
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True, capture_output=True, text=True,
        ).stdout
        return {"commit": commit, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
