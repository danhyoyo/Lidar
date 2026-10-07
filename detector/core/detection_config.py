"""Pure detection validation shared by pipeline, notebook and model boundaries.

Grouped predictions can be constructed before grouped training/decode ships;
this capability contract does not advertise those later adapters as implemented.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass

from .backbone_config import finite_scalar, positive_integer, resolve_backbone_features
from .task_groups import resolve_task_groups, validate_objects, validate_resolved_groups


@dataclass(frozen=True)
class DetectionConfig:
    head_mode: str
    cls_encoding: str
    box_mode: str
    use_iou: bool
    groups: tuple
    group_weights: tuple
    quality_target: str | None
    backbone_features: object
    vertical_loss_weight: float | None = None


def _choice(config, key, default, choices):
    value = config.get(key, default)
    if not isinstance(value, str) or value.lower() not in choices:
        raise ValueError(f"{key} must be one of {sorted(choices)}")
    return value.lower()


def _boolean(config, key, default=False):
    value = config.get(key, default)
    if type(value) is not bool:
        raise ValueError(f"{key} must be a boolean")
    return value


def resolve_box_mode(value="bev"):
    """Validate the shared box convention without importing tensor backends."""
    return _choice({"box_mode": value}, "box_mode", "bev", {"bev", "3d"})


def resolve_detection_config(config):
    """Validate full data/model/objective settings without mutating the caller."""
    if not isinstance(config, Mapping):
        raise ValueError("Detection config must be a mapping")
    data, model, loss = (config.get(k, {}) for k in ("data", "model", "loss"))
    if any(not isinstance(s, Mapping) for s in (data, model, loss)):
        raise ValueError("data, model and loss must be mappings")
    mode = _choice(model, "head_mode", "legacy_single", {"legacy_single", "grouped"})
    classification = _choice(model, "cls_encoding", "gaussian", {"gaussian", "binary"})
    box_mode = resolve_box_mode(data.get("box_mode", "bev"))
    if "box_mode" in model:
        raise ValueError("Set box_mode only in data")
    if "head_groups" in model or "group_weights" in model:
        raise ValueError("head_groups belong in data; group_weights belong in loss")
    stride = data.get("out_size_factor", 4)
    if type(stride) is not int or stride != 4:
        raise ValueError("These backbones output stride 4; out_size_factor must be integer 4")
    positive_integer(model.get("backbone_out_dim", 16), "backbone_out_dim")
    depths = model.get("stage_depths", (2, 4, 2))
    if not isinstance(depths, (list, tuple)) or len(depths) != 3:
        raise ValueError("stage_depths must contain three positive integers")
    for depth in depths:
        positive_integer(depth, "stage_depths")
    _boolean(model, "header_use_bn", str(model.get("backbone", "mobilepixor")).lower() == "mobilepixornext")
    if model.get("header_act", "none") not in ("none", None, "silu", "relu"):
        raise ValueError("Unsupported header_act")
    use_iou = _boolean(model, "header_use_iou")
    if use_iou != _boolean(loss, "use_iou"):
        raise ValueError("header_use_iou and loss.use_iou must agree")
    strategy = _choice(loss, "name", "baseline", {"baseline", "oga", "uwag", "gw_qal", "q_oga"})
    if box_mode == "3d" and strategy not in ("baseline", "oga"):
        raise ValueError("3D supports only baseline or oga; other strategy extensions are unimplemented")
    vertical_weight = (finite_scalar(loss.get("vertical_loss_weight"), "vertical_loss_weight", positive=True)
                       if box_mode == "3d" else None)
    if use_iou and strategy not in ("baseline", "oga"):
        raise ValueError("IQA supervision requires baseline or oga loss")

    kitti = data.get("kitti", {})
    if not isinstance(kitti, Mapping):
        raise ValueError("data.kitti must be a mapping")
    objects = kitti.get("objects")
    if "objects" in kitti:
        validate_objects(objects)
    count = data.get("num_classes", len(objects) if objects is not None else None)
    if "num_classes" in data or objects is not None:
        positive_integer(count, "num_classes")
        if objects is not None and count != len(objects):
            raise ValueError("num_classes must agree with data.kitti.objects")

    groups, weights = (), ()
    if mode == "legacy_single":
        if "head_groups" in data or "group_weights" in loss:
            raise ValueError("head_groups/group_weights cannot be silently ignored in legacy_single")
    else:
        groups = resolve_task_groups(data.get("head_groups"), objects)
        configured = loss.get("group_weights", {g.name: 1.0 for g in groups})
        if not isinstance(configured, Mapping) or set(configured) != {g.name for g in groups}:
            raise ValueError("group_weights must match task group names exactly")
        values = [finite_scalar(configured[g.name], "group_weights", positive=True) for g in groups]
        # Avoid overflow when every valid configured weight is very large.
        scaled = [value / max(values) for value in values]
        denominator = sum(scaled)
        weights = tuple((g.name, value / denominator) for g, value in zip(groups, scaled))

    quality = None
    if strategy == "q_oga":
        quality = _choice(loss, "quality_target", "mgiou", {"mgiou", "rotated_iou"})
        warmup = loss.get("quality_warmup_epochs", 0)
        if type(warmup) is not int or warmup < 0:
            raise ValueError("quality_warmup_epochs must be a nonnegative integer")
        if warmup and quality != "rotated_iou":
            raise ValueError("quality curriculum requires quality_target='rotated_iou'")
        if quality == "rotated_iou" and classification != "gaussian":
            raise ValueError("Exact Q-OGA requires Gaussian classification and BEV boxes")
    if mode == "grouped" and classification == "binary" and strategy in ("q_oga", "gw_qal"):
        raise ValueError(f"{strategy}: binary grouped support awaits explicit parity evidence")

    normalized = copy.deepcopy(dict(model))
    normalized.setdefault("backbone", "mobilepixor")
    if normalized.get("neck_type") in (None, "sgfpn"):
        normalized["neck_type"] = "scale_gated_fpn"
    if "bev_encoding" in data:
        normalized["bev_encoding"] = copy.deepcopy(data["bev_encoding"])
    # Preserve registered custom builders; their own options are validated by them.
    builtins = {"mobilepixornext", "mobilepixor", "mobilepixor_coordatt", "pixor", "rpn"}
    if str(normalized["backbone"]).lower() in builtins:
        features = resolve_backbone_features(normalized, geometry=kitti.get("geometry"))
    else:
        features = None
        if any(k.startswith(("c4_context", "local_attention", "neck_fusion", "detail_path")) for k in model):
            raise ValueError("New backbone features require MobilePixorNeXt")
    return DetectionConfig(mode, classification, box_mode, use_iou, groups, weights, quality, features,
                           vertical_weight)


def resolve_model_detection(model, num_classes, task_groups=None, *, box_mode="bev"):
    """Validate model-only construction; objective agreement needs the full config.

    Group definitions must be supplied explicitly as resolved metadata. The
    pipeline validates the chosen loss before calling this model-only boundary.
    """
    mode = _choice(model, "head_mode", "legacy_single", {"legacy_single", "grouped"})
    if mode == "grouped" and task_groups is None:
        raise ValueError("Grouped model requires head_groups supplied as resolved task_groups")
    data = {"num_classes": num_classes, "kitti": {}, "box_mode": box_mode}
    if "geometry" in model:
        data["kitti"]["geometry"] = model["geometry"]
    if "bev_encoding" in model:
        data["bev_encoding"] = model["bev_encoding"]
    if task_groups is not None:
        if mode != "grouped":
            raise ValueError("Resolved task groups require head_mode=grouped")
        task_groups = validate_resolved_groups(task_groups)
        data["head_groups"] = [{"name": g.name, "classes": list(g.classes)} for g in task_groups]
        data["kitti"]["objects"] = {cls: i for g in task_groups for cls, i in zip(g.classes, g.global_ids)}
    return resolve_detection_config({"model": model, "data": data,
                                     "loss": {"name": "baseline", "use_iou": model.get("header_use_iou", False),
                                              **({"vertical_loss_weight": 1.} if box_mode == "3d" else {})}})
