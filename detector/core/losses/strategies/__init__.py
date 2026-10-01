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
from core.losses.strategies.q_oga import QOgaLossStrategy

# Register default strategies
register_loss_strategy("baseline")(BaselineLossStrategy)
register_loss_strategy("uwag")(UwagLossStrategy)
register_loss_strategy("oga")(OgaLossStrategy)
register_loss_strategy("q_oga")(QOgaLossStrategy)

__all__ = [
    "BaseLossStrategy",
    "BaselineLossStrategy",
    "UwagLossStrategy",
    "OgaLossStrategy",
    "QOgaLossStrategy",
    "register_loss_strategy",
    "build_loss_strategy",
    "get_available_loss_strategies",
]
