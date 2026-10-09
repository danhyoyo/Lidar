"""Pure resolution of backbone feature options; no model/backend imports.

Inactive new module-specific fields are rejected rather than silently discarded.
Legacy configs retain LiteMLA defaults, saved inactive LiteMLA settings and their
existing geometry behavior. Candidate topologies require inactive LiteMLA fields
to be empty/none explicitly.
Construction and central detector validation consume this contract in task 20/23.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass


def positive_integer(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer (not a boolean)")
    return value


def finite_scalar(value, name, *, positive=False):
    try:
        valid = (type(value) in (int, float) and math.isfinite(value) and
                 (value > 0 if positive else value >= 0))
    except OverflowError:
        valid = False
    if not valid:
        bound = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be a finite {bound} scalar (not a boolean)")
    return float(value)


def resolve_focal_settings(*, version=1, bottleneck=64, dilations=(1, 2, 3),
                           layer_scale_init=0.001):
    """Shared constructor/config validation for the version-1 focal operator."""
    if type(version) is not int or version != 1:
        raise ValueError("c4_context_version must be integer 1")
    positive_integer(bottleneck, "c4_context_bottleneck")
    if (not isinstance(dilations, (list, tuple)) or not dilations or
            any(type(d) is not int or d <= 0 for d in dilations) or
            any(a >= b for a, b in zip(dilations, dilations[1:]))):
        raise ValueError("c4_context_dilations must be distinct ascending positive integers")
    scale = finite_scalar(layer_scale_init, "c4_context_layer_scale_init")
    return version, bottleneck, tuple(dilations), scale


@dataclass(frozen=True)
class BackboneFeatureConfig:
    c4_attention: str = "litemla"
    c4_attention_scales: tuple[int, ...] = (5,)
    c4_attention_qk_norm: str = "none"
    c4_context: str = "none"
    c4_context_version: int | None = None
    c4_context_bottleneck: int | None = None
    c4_context_dilations: tuple[int, ...] | None = None
    c4_context_layer_scale_init: float | None = None
    local_attention: str = "none"
    local_attention_eca_kernel_size: int | None = None
    local_attention_simam_lambda: float | None = None
    local_attention_layer_scale_init: float | None = None
    neck_fusion_channels: int = 24
    detail_path: bool = False

    def to_dict(self):
        """Save active settings and explicit common defaults as JSON values."""
        return {k: list(v) if isinstance(v, tuple) else v
                for k, v in asdict(self).items() if v is not None}

    def semantic_metadata(self):
        return {"resolver_version": 1,
                "placement": {"context_stride": 8, "local_attention_stride": 4},
                "config": self.to_dict()}

    @property
    def semantic_hash(self):
        payload = json.dumps(self.semantic_metadata(), sort_keys=True,
                             separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()


_FIELDS = frozenset(BackboneFeatureConfig.__dataclass_fields__)
_FOCAL_FIELDS = frozenset(k for k in _FIELDS if k.startswith("c4_context_"))
_LOCAL_FIELDS = frozenset(k for k in _FIELDS if k.startswith("local_attention_"))
_NEW_FIELDS = _FOCAL_FIELDS | _LOCAL_FIELDS | {
    "c4_context", "local_attention", "neck_fusion_channels", "detail_path",
}


def _choice(config, name, default, choices):
    value = config.get(name, default)
    if not isinstance(value, str) or value.lower() not in choices:
        raise ValueError(f"{name} must be one of {sorted(choices)}")
    return value.lower()


def resolve_rc_gate_mode(mode="mul"):
    """Keep legacy multiplicative gating unless additive gating is explicit."""
    return _choice({"rc_gate_mode": mode}, "rc_gate_mode", "mul", {"mul", "add"})


def _reject_present(config, fields, reason):
    invalid = set(config) & set(fields)
    if invalid:
        raise ValueError(f"{', '.join(sorted(invalid))}: {reason}")


def _candidate_requested(cfg, context, local):
    encoding = cfg.get("bev_encoding") or {}
    return (context != "none" or local != "none" or
            cfg.get("neck_fusion_channels", 24) != 24 or cfg.get("detail_path", False) or
            (isinstance(encoding, Mapping) and encoding.get("name") == "hist14") or
            cfg.get("stage_depths", (2, 4, 2)) not in ((2, 4, 2), [2, 4, 2]))


def _validate_grid(grid_shape, geometry):
    if grid_shape is None and geometry is not None:
        if not isinstance(geometry, Mapping):
            raise ValueError("grid_shape requires a valid xy geometry mapping")
        cells = []
        for axis in ("y", "x"):
            try:
                lo, hi, res = (geometry[f"{axis}_{part}"] for part in ("min", "max", "res"))
                if any(type(v) not in (int, float) or not math.isfinite(v) for v in (lo, hi, res)):
                    raise ValueError
                if hi <= lo or res <= 0:
                    raise ValueError
                count = (hi - lo) / res
                rounded = round(count)
                if not math.isclose(count, rounded, rel_tol=1e-5, abs_tol=1e-6):
                    raise ValueError
                cells.append(rounded)
            except (KeyError, ValueError, TypeError, OverflowError) as exc:
                raise ValueError("grid_shape requires positive integral xy geometry") from exc
        grid_shape = tuple(cells)
    if grid_shape is not None and (
            not isinstance(grid_shape, (list, tuple)) or len(grid_shape) != 2 or
            any(type(d) is not int or d <= 0 or d % 16 for d in grid_shape)):
        raise ValueError("grid_shape must be (height, width), positive integers divisible by 16")


def resolve_backbone_features(model_config=None, *, geometry=None, grid_shape=None):
    """Resolve a model-section mapping without mutation or construction.

    Pass geometry or a YX grid when known. Geometry-free constructors can resolve
    settings first; their caller must validate candidate grids before execution.
    Other model fields are permitted; misspelled feature-option prefixes are not.
    """
    cfg = {} if model_config is None else model_config
    if not isinstance(cfg, Mapping):
        raise ValueError("model_config must be a mapping")
    unknown = [k for k in cfg if isinstance(k, str) and
               k.startswith(("c4_context", "local_attention", "neck_fusion", "detail_path"))
               and k not in _FIELDS]
    if unknown:
        raise ValueError(f"unsupported feature options: {sorted(unknown)}")
    backbone = _choice(cfg, "backbone", "mobilepixornext",
                       {"mobilepixornext", "mobilepixor", "mobilepixor_coordatt", "pixor", "rpn"})
    if backbone != "mobilepixornext":
        _reject_present(cfg, _NEW_FIELDS, "feature options require MobilePixorNeXt")
    neck = _choice(cfg, "neck_type", "scale_gated_fpn",
                   {"scale_gated_fpn", "rc_sgfpn", "rc_bisgfpn"})
    gate_mode = resolve_rc_gate_mode(cfg.get("rc_gate_mode", "mul"))
    if gate_mode == "add" and (backbone != "mobilepixornext" or
                               neck not in {"rc_sgfpn", "rc_bisgfpn"}):
        raise ValueError("rc_gate_mode=add requires a MobilePixorNeXt RC-SGFPN neck")
    if neck != "scale_gated_fpn":
        incompatible = {key for key, default in (("neck_fusion_channels", 24), ("detail_path", False))
                        if key in cfg and cfg[key] != default}
        _reject_present(cfg, incompatible,
                        "fusion/detail options require standard SG-FPN")
    attention = _choice(cfg, "c4_attention", "litemla", {"none", "litemla"})
    context = _choice(cfg, "c4_context", "none", {"none", "focal"})
    local = _choice(cfg, "local_attention", "none", {"none", "eca", "simam"})
    if context == "focal" and attention == "litemla":
        raise ValueError("c4_context=focal and c4_attention=litemla are mutually exclusive")
    scales = cfg.get("c4_attention_scales", (5,) if attention == "litemla" else ())
    qk = _choice(cfg, "c4_attention_qk_norm", "none", {"none", "rmsnorm", "layernorm"})
    if attention == "litemla":
        if (not isinstance(scales, (list, tuple)) or not scales or
                any(type(s) is not int or s < 3 or s % 2 == 0 for s in scales) or
                len(set(scales)) != len(scales)):
            raise ValueError("c4_attention_scales must contain distinct odd integers >= 3")
    elif not isinstance(scales, (list, tuple)):
        raise ValueError("c4_attention_scales must be a list or tuple")
    elif _candidate_requested(cfg, context, local) and (len(scales) or qk != "none"):
        raise ValueError("c4_attention_scales must be empty and c4_attention_qk_norm none when attention is none")
    values = dict(c4_attention=attention, c4_attention_scales=tuple(scales),
                  c4_attention_qk_norm=qk, c4_context=context, local_attention=local)
    if context == "focal":
        settings = resolve_focal_settings(
            version=cfg.get("c4_context_version", 1),
            bottleneck=cfg.get("c4_context_bottleneck", 64),
            dilations=cfg.get("c4_context_dilations", (1, 2, 3)),
            layer_scale_init=cfg.get("c4_context_layer_scale_init", 0.001))
        values.update(zip(("c4_context_version", "c4_context_bottleneck",
                           "c4_context_dilations", "c4_context_layer_scale_init"), settings))
    else:
        _reject_present(cfg, _FOCAL_FIELDS, "c4_context must be focal")
    allowed_local = set()
    if local != "none":
        allowed_local.add("local_attention_layer_scale_init")
        values["local_attention_layer_scale_init"] = finite_scalar(
            cfg.get("local_attention_layer_scale_init", 0.001), "local_attention_layer_scale_init")
    if local == "eca":
        key = "local_attention_eca_kernel_size"
        allowed_local.add(key)
        kernel = positive_integer(cfg.get(key, 3), key)
        if kernel % 2 == 0:
            raise ValueError(f"{key} must be odd")
        values[key] = kernel
    elif local == "simam":
        key = "local_attention_simam_lambda"
        allowed_local.add(key)
        values[key] = finite_scalar(cfg.get(key, 0.0001), key, positive=True)
    _reject_present(cfg, _LOCAL_FIELDS - allowed_local, f"fields do not apply to local_attention={local}")
    fusion = cfg.get("neck_fusion_channels", 24)
    if type(fusion) is not int or fusion not in (24, 32):
        raise ValueError("neck_fusion_channels must be integer 24 or 32")
    detail = cfg.get("detail_path", False)
    if type(detail) is not bool:
        raise ValueError("detail_path must be a boolean")
    values.update(neck_fusion_channels=fusion, detail_path=detail)
    if _candidate_requested(cfg, context, local):
        _validate_grid(grid_shape, geometry if geometry is not None else cfg.get("geometry"))
    return BackboneFeatureConfig(**values)
