"""Backbone Registry for modular model construction."""

from typing import Any, Callable, Dict, List
import torch.nn as nn

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
    return MobilePixorBackBone()


@register_backbone("mobilepixor_coordatt")
def _build_mobilepixor_coordatt(cfg: Dict[str, Any], input_channels: int = 35) -> nn.Module:
    return MobilePixorCoordAttBackBone(
        input_channels=input_channels,
        scale_gated_fpn=cfg.get("scale_gated_fpn", False),
    )


@register_backbone("mobilepixornext")
def _build_mobilepixornext(cfg: Dict[str, Any], input_channels: int = 35) -> nn.Module:
    return MobilePixorNeXtBackbone(
        input_channels=input_channels,
        backbone_out_dim=cfg.get("backbone_out_dim", 16),
        c4_attention=cfg.get("c4_attention", "litemla"),
        scale_gated_fpn=cfg.get("scale_gated_fpn", True),
        expansion=cfg.get("expansion", 2.5),
        use_reparam=cfg.get("use_reparam", False),
    )


@register_backbone("pixor")
def _build_pixor(cfg: Dict[str, Any], input_channels: int = 35) -> nn.Module:
    return PixorBackBone()


@register_backbone("rpn")
def _build_rpn(cfg: Dict[str, Any], input_channels: int = 35) -> nn.Module:
    return RPN()
