"""
Abstract base class for trading strategies
Defines the common interface for all trading strategies

Design Reference: CLAUDE.md - Strategy Modules Section 5 (MVP 7 strategies)
Related Classes:
- Concrete strategies: weekend_effect.py, stop_rev.py, fund_contra.py, etc.
- Risk integration: UnifiedRiskEngine for position sizing
- Signal types: MarketFrame -> Signal (buy/sell/hold)
- Performance tracking: PerformanceMixin for metrics
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..analysis import TechnicalAnalyzer, TechnicalSignal
from ..domains.market import MarketFrame, Signal


@dataclass
class StrategyConfig:
    """Base configuration for trading strategies"""

    name: str
    enabled: bool = True
    params: dict[str, Any] = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class StrategyResult:
    """Result from strategy execution"""

    signal: Signal | None
    metadata: dict[str, Any]
    execution_time_ms: float
    success: bool
    error_message: str | None = None


class Strategy(ABC):
    """
    Abstract base class for all trading strategies

    Each strategy must implement:
    1. generate_signal() - core signal generation logic
    2. validate_config() - parameter validation
    3. get_required_data() - specify required market data
    """

    def __init__(self, config: StrategyConfig):
        self.config = config
        self.name = config.name
        self.enabled = config.enabled
        self.params = config.params

        # Validate configuration
        self.validate_config()

    @abstractmethod
    async def generate_signal(self, market_data: MarketFrame) -> Signal | None:
        """
        Generate trading signal based on market data

        Args:
            market_data: Current market state with OHLCV, orderbook, etc.

        Returns:
            Signal object if conditions are met, None otherwise
        """
        pass

    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate strategy configuration parameters
        Raise ValueError if configuration is invalid
        """
        pass

    @abstractmethod
    def get_required_data(self) -> dict[str, Any]:
        """
        Specify what market data this strategy requires

        Returns:
            Dict specifying required data:
            {
                'ohlcv_timeframes': ['1m', '5m'],
                'orderbook_levels': 10,
                'indicators': ['sma_20', 'rsi_14'],
                'lookback_periods': 100
            }
        """
        pass

    def get_description(self) -> str:
        """Get strategy description"""
        return f"{self.name} - {self.__class__.__doc__ or 'No description'}"

    def is_enabled(self) -> bool:
        """Check if strategy is enabled"""
        return self.enabled

    def update_config(self, new_params: dict[str, Any]) -> None:
        """Update strategy parameters"""
        self.params.update(new_params)
        self.validate_config()


class TechnicalStrategy(Strategy):
    """
    Base class for technical analysis strategies
    Uses pandas-ta for advanced technical analysis
    """

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.technical_analyzer = TechnicalAnalyzer()
        self._price_history = []
        self._indicators_cache = {}

        from contextlib import suppress

        # パフォーマンス測定の初期化
        from .performance_mixin import PerformanceMixin

        if not isinstance(self, PerformanceMixin):
            # ミックスインの機能を動的に追加
            with suppress(Exception):
                from ..monitor.performance_logger import get_performance_logger

                self.performance_logger = get_performance_logger()

    def update_price_history(self, market_data: MarketFrame) -> None:
        """Update price history for technical analysis"""
        price_point = {
            "timestamp": market_data.timestamp,
            "open": market_data.ohlcv.open,
            "high": market_data.ohlcv.high,
            "low": market_data.ohlcv.low,
            "close": market_data.ohlcv.close,
            "volume": market_data.ohlcv.volume,
        }

        self._price_history.append(price_point)

        # Keep only last periods for efficiency (configurable via params)
        max_history = self.params.get("max_history_periods", 200)
        if len(self._price_history) > max_history:
            self._price_history = self._price_history[-max_history:]

    def get_price_dataframe(self) -> pd.DataFrame:
        """Convert price history to DataFrame for technical analysis"""
        if not self._price_history:
            return pd.DataFrame()

        df = pd.DataFrame(self._price_history)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.set_index("datetime")
        return df[["open", "high", "low", "close", "volume"]]

    def calculate_indicators(self) -> pd.DataFrame:
        """Calculate all technical indicators using pandas-ta"""
        price_df = self.get_price_dataframe()
        if price_df.empty:
            return pd.DataFrame()

        # Cache key based on the latest timestamp
        cache_key = str(price_df.index[-1]) if len(price_df) > 0 else "empty"

        if cache_key in self._indicators_cache:
            return self._indicators_cache[cache_key]

        # Calculate indicators using pandas-ta
        indicators_df = self.technical_analyzer.calculate_all_indicators(price_df)

        # Cache the result
        self._indicators_cache[cache_key] = indicators_df

        # Limit cache size
        if len(self._indicators_cache) > 10:
            oldest_key = list(self._indicators_cache.keys())[0]
            del self._indicators_cache[oldest_key]

        return indicators_df

    def generate_technical_signals(self) -> list[TechnicalSignal]:
        """Generate technical signals using pandas-ta"""
        indicators_df = self.calculate_indicators()
        if indicators_df.empty:
            return []

        return self.technical_analyzer.generate_signals(indicators_df)

    def get_latest_indicators(self) -> dict[str, float]:
        """Get latest indicator values as dictionary"""
        indicators_df = self.calculate_indicators()
        if indicators_df.empty:
            return {}

        latest_row = indicators_df.iloc[-1]
        return {col: val for col, val in latest_row.items() if pd.notna(val)}

    # Technical Indicator Helper Methods
    def calculate_sma(self, prices: list[float], period: int) -> float | None:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period

    def calculate_rsi(self, prices: list[float], period: int = 14) -> float | None:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return None

        deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        gains = [delta if delta > 0 else 0 for delta in deltas]
        losses = [-delta if delta < 0 else 0 for delta in deltas]

        if len(gains) < period:
            return None

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_bollinger_bands(
        self, prices: list[float], period: int = 20, std_dev: float = 2.0
    ) -> dict[str, float] | None:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return None

        sma = self.calculate_sma(prices, period)
        if sma is None:
            return None

        recent_prices = prices[-period:]
        variance = sum((price - sma) ** 2 for price in recent_prices) / period
        std = variance**0.5

        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        bandwidth = (upper - lower) / sma if sma != 0 else 0

        return {"upper": upper, "middle": sma, "lower": lower, "bandwidth": bandwidth}

    def calculate_atr(
        self,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        period: int = 14,
    ) -> float | None:
        """Calculate Average True Range"""
        if (
            len(highs) < period + 1
            or len(lows) < period + 1
            or len(closes) < period + 1
        ):
            return None

        true_ranges = []
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close_prev = abs(highs[i] - closes[i - 1])
            low_close_prev = abs(lows[i] - closes[i - 1])
            true_range = max(high_low, high_close_prev, low_close_prev)
            true_ranges.append(true_range)

        if len(true_ranges) < period:
            return None

        return sum(true_ranges[-period:]) / period
