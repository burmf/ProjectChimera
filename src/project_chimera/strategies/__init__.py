"""
Trading strategies package
Pluggable strategy implementations for different market conditions
"""

from .base import Strategy, StrategyResult, StrategyConfig, TechnicalStrategy

# STRAT-7: Core trading strategies
from .enhanced_weekend_effect import EnhancedWeekendEffectStrategy as WeekendEffectStrategy, create_enhanced_weekend_effect_strategy as create_weekend_effect_strategy
from .stop_rev import StopReversionStrategy, create_stop_reversion_strategy
from .fund_contra import FundingContraStrategy, create_funding_contra_strategy
from .lob_revert import LimitOrderBookReversionStrategy, create_lob_reversion_strategy
from .vol_breakout import VolatilityBreakoutStrategy, create_volatility_breakout_strategy
from .cme_gap import CMEGapStrategy, create_cme_gap_strategy
from .basis_arb import BasisArbitrageStrategy, create_basis_arbitrage_strategy

# Legacy strategies
from .mini_momo import MiniMomentumStrategy, create_mini_momentum_strategy
from .ob_revert import OrderBookMeanReversionStrategy, create_orderbook_reversion_strategy

__all__ = [
    # Base classes
    "Strategy", "StrategyResult", "StrategyConfig", "TechnicalStrategy",
    
    # STRAT-7: Core strategies
    "WeekendEffectStrategy", "create_weekend_effect_strategy",
    "StopReversionStrategy", "create_stop_reversion_strategy", 
    "FundingContraStrategy", "create_funding_contra_strategy",
    "LimitOrderBookReversionStrategy", "create_lob_reversion_strategy",
    "VolatilityBreakoutStrategy", "create_volatility_breakout_strategy",
    "CMEGapStrategy", "create_cme_gap_strategy",
    "BasisArbitrageStrategy", "create_basis_arbitrage_strategy",
    
    # Legacy strategies
    "MiniMomentumStrategy", "create_mini_momentum_strategy", 
    "OrderBookMeanReversionStrategy", "create_orderbook_reversion_strategy"
]