"""Backbones package."""

try:
    from detector.core.models.backbones.registry import (
        register_backbone,
        build_backbone,
        get_available_backbones,
    )
except ImportError:
    from core.models.backbones.registry import (
        register_backbone,
        build_backbone,
        get_available_backbones,
    )

__all__ = [
    "register_backbone",
    "build_backbone",
    "get_available_backbones",
]
