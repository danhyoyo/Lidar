"""Backbone Registry for modular model construction."""

from typing import Any, Callable, Dict, List
import torch.nn as nn
from core.backbone_config import resolve_backbone_features

BackboneBuilder = Callable[[Dict[str, Any], int], nn.Module]

_BACKBONE_REGISTRY: Dict[str, BackboneBuilder] = {}


def register_backbone(name: str, allow_override: bool = False):
    """Decorator to register a backbone builder function or class."""
    def decorator(builder: BackboneBuilder):
        key = name.lower()
        if key in _BACKBONE_REGISTRY and not allow_override:
            raise KeyError(f"Backbone {name!r} is already registered.")
        _BACKBONE_REGISTRY[key] = builder
        return builder
    return decorator


def build_backbone(name: str, cfg: Dict[str, Any], input_channels: int = 35) -> nn.Module:
    """Build a backbone instance by name using registered builders."""
    key = str(name).lower()
    if key not in _BACKBONE_REGISTRY:
        available = ", ".join(sorted(_BACKBONE_REGISTRY.keys()))
        raise ValueError(f"Unsupported backbone: {name!r}. Available backbones: {available}")
    builder = _BACKBONE_REGISTRY[key]
    if key in {"mobilepixornext", "mobilepixor", "mobilepixor_coordatt", "pixor", "rpn"}:
        normalized = dict(cfg, backbone=key)
        if normalized.get("neck_type") in (None, "sgfpn"):
            normalized["neck_type"] = "scale_gated_fpn"
        resolve_backbone_features(normalized, geometry=cfg.get("geometry"))
    return builder(cfg, input_channels)


def get_available_backbones() -> List[str]:
    """Return a sorted list of registered backbone names."""
    return sorted(_BACKBONE_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Default Backbone Registrations
# ---------------------------------------------------------------------------
from core.models.backbones.mobilepixor import MobilePixorBackBone
from core.models.backbones.mobilepixor_coordinate_attention import (
    MobilePixorBackBone as MobilePixorCoordAttBackBone,
)
from core.models.backbones.pixor import PixorBackBone
from core.models.backbones.rpn import RPN
from core.models.backbones.mobilepixornext import MobilePixorNeXtBackbone


@register_backbone("mobilepixor")
def _build_mobilepixor(cfg: Dict[str, Any], input_channels: int = 35) -> nn.Module:
    return MobilePixorBackBone(
        input_channels=input_channels,
        scale_gated_fpn=cfg.get("scale_gated_fpn", False),
    )


@register_backbone("mobilepixor_coordatt")
def _build_mobilepixor_coordatt(cfg: Dict[str, Any], input_channels: int = 35) -> nn.Module:
    return MobilePixorCoordAttBackBone(
        input_channels=input_channels,
        scale_gated_fpn=cfg.get("scale_gated_fpn", False),
    )


@register_backbone("mobilepixornext")
def _build_mobilepixornext(cfg: Dict[str, Any], input_channels: int = 35) -> nn.Module:
    feature_cfg = dict(cfg)
    if feature_cfg.get("neck_type") in (None, "sgfpn"):
        feature_cfg["neck_type"] = "scale_gated_fpn"
    features = resolve_backbone_features(feature_cfg)
    new_options = {k: v for k, v in features.to_dict().items()
                   if k not in {"c4_attention", "c4_attention_scales", "c4_attention_qk_norm"}}
    return MobilePixorNeXtBackbone(
        input_channels=input_channels,
        backbone_out_dim=cfg.get("backbone_out_dim", 16),
        c4_attention=cfg.get("c4_attention", "litemla"),
        c4_attention_scales=features.c4_attention_scales,
        c4_attention_qk_norm=cfg.get("c4_attention_qk_norm", "none"),
        scale_gated_fpn=cfg.get("scale_gated_fpn", True),
        expansion=cfg.get("expansion", 2.5),
        use_reparam=cfg.get("use_reparam", False),
        deploy=cfg.get("deploy", False),
        neck_type=cfg.get("neck_type", "scale_gated_fpn"),
        geometry=cfg.get("geometry") or cfg.get("kitti", {}).get("geometry") or cfg.get("data", {}).get("kitti", {}).get("geometry"),
        num_range_bands=cfg.get("num_range_bands", 4),
        rc_gate_mode=cfg.get("rc_gate_mode", "mul"),
        stage_depths=cfg.get("stage_depths", (2, 4, 2)),
        **new_options,
    )


@register_backbone("pixor")
def _build_pixor(cfg: Dict[str, Any], input_channels: int = 35) -> nn.Module:
    return PixorBackBone()


@register_backbone("rpn")
def _build_rpn(cfg: Dict[str, Any], input_channels: int = 35) -> nn.Module:
    return RPN()
