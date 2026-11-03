"""Backbones package."""

try:
    from detector.core.models.backbones.registry import (
        register_backbone,
        build_backbone,
        get_available_backbones,
    )
    from detector.core.models.backbones.rc_sgfpn import (
        RangeConditionedSGFPN,
        RangeConditionedScaleGate,
    )
except ImportError:
    from core.models.backbones.registry import (
        register_backbone,
        build_backbone,
        get_available_backbones,
    )
    from core.models.backbones.rc_sgfpn import (
        RangeConditionedSGFPN,
        RangeConditionedScaleGate,
    )

__all__ = [
    "register_backbone",
    "build_backbone",
    "get_available_backbones",
    "RangeConditionedSGFPN",
    "RangeConditionedScaleGate",
]

