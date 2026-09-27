"""Loss strategies package."""

from core.losses.strategies.base import BaseLossStrategy
from core.losses.strategies.registry import (
    register_loss_strategy,
    build_loss_strategy,
    get_available_loss_strategies,
)
from core.losses.strategies.baseline import BaselineLossStrategy
from core.losses.strategies.uwag import UwagLossStrategy
from core.losses.strategies.oga import OgaLossStrategy

# Register default strategies
register_loss_strategy("baseline")(BaselineLossStrategy)
register_loss_strategy("uwag")(UwagLossStrategy)
register_loss_strategy("oga")(OgaLossStrategy)

__all__ = [
    "BaseLossStrategy",
    "BaselineLossStrategy",
    "UwagLossStrategy",
    "OgaLossStrategy",
    "register_loss_strategy",
    "build_loss_strategy",
    "get_available_loss_strategies",
]
