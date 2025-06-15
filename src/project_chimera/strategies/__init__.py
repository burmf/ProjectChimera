"""
Trading strategies package
Pluggable strategy implementations for different market conditions
"""

from .base import Strategy, StrategyResult, StrategyConfig, TechnicalStrategy
from .vol_breakout import VolatilityBreakoutStrategy, create_vol_breakout_strategy
from .mini_momo import MiniMomentumStrategy, create_mini_momentum_strategy
from .ob_revert import OrderBookMeanReversionStrategy, create_orderbook_reversion_strategy

__all__ = [
    "Strategy", "StrategyResult", "StrategyConfig", "TechnicalStrategy",
    "VolatilityBreakoutStrategy", "create_vol_breakout_strategy",
    "MiniMomentumStrategy", "create_mini_momentum_strategy", 
    "OrderBookMeanReversionStrategy", "create_orderbook_reversion_strategy"
]