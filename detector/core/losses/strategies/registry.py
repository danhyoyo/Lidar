"""Registry for modular Loss Strategies."""

from typing import Any, Callable, Dict, List, Type
from core.losses.strategies.base import BaseLossStrategy

StrategyBuilder = Callable[[str, Dict[str, Any]], BaseLossStrategy]
_LOSS_STRATEGY_REGISTRY: Dict[str, StrategyBuilder] = {}


def register_loss_strategy(name: str):
    """Decorator to register a loss strategy class or builder."""
    def decorator(cls_or_builder: Any):
        key = str(name).lower()
        if key in _LOSS_STRATEGY_REGISTRY:
            raise KeyError(f"Loss strategy {name!r} is already registered.")
        _LOSS_STRATEGY_REGISTRY[key] = cls_or_builder
        return cls_or_builder
    return decorator


def build_loss_strategy(
    name: str, cls_encoding: str, config: Dict[str, Any] = None
) -> BaseLossStrategy:
    """Build a loss strategy instance by name."""
    key = str(name).lower()
    if key not in _LOSS_STRATEGY_REGISTRY:
        available = ", ".join(sorted(_LOSS_STRATEGY_REGISTRY.keys()))
        raise ValueError(f"Unsupported loss name: {name!r}. Available strategies: {available}")
    builder = _LOSS_STRATEGY_REGISTRY[key]
    return builder(cls_encoding, config)


def get_available_loss_strategies() -> List[str]:
    """Return a sorted list of registered loss strategy names."""
    return sorted(_LOSS_STRATEGY_REGISTRY.keys())
