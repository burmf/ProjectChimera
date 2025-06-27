"""
Stop Loss Reversion Strategy (STOP_REV)
Exploits oversold rebounds after rapid drops: 5m -3% & vol ×3 → long rebound
"""

import statistics
from typing import Any

from ..domains.market import MarketFrame, Signal, SignalType
from ..settings import get_strategy_config
from .base import StrategyConfig, TechnicalStrategy


class StopReversionStrategy(TechnicalStrategy):
    """
    Stop Loss Reversion Strategy

    Core trigger: 5m -3% & vol ×3 → long rebound
    Exploits liquidation cascades and oversold bounces
    """

    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        # Load strategy-specific settings
        self.strategy_settings = get_strategy_config("stop_reversion")

    def validate_config(self) -> None:
        """Validate strategy configuration"""
        # Load parameters from settings with fallbacks
        self.params.setdefault(
            "min_price_drop_pct",
            getattr(self.strategy_settings, "min_price_drop_pct", 3.0),
        )
        self.params.setdefault(
            "min_volume_multiplier",
            getattr(self.strategy_settings, "min_volume_multiplier", 3.0),
        )
        self.params.setdefault(
            "lookback_periods", getattr(self.strategy_settings, "lookback_periods", 20)
        )
        self.params.setdefault(
            "timeframe", getattr(self.strategy_settings, "timeframe", "5m")
        )
        self.params.setdefault(
            "max_position_minutes",
            getattr(self.strategy_settings, "max_position_minutes", 60),
        )
        self.params.setdefault(
            "stop_loss_pct", getattr(self.strategy_settings, "stop_loss_pct", 2.0)
        )
        self.params.setdefault(
            "take_profit_pct", getattr(self.strategy_settings, "take_profit_pct", 4.0)
        )
        self.params.setdefault(
            "min_price_level", getattr(self.strategy_settings, "min_price_level", 0.01)
        )
        self.params.setdefault(
            "confidence_base", getattr(self.strategy_settings, "confidence_base", 0.5)
        )
        self.params.setdefault(
            "target_size", getattr(self.strategy_settings, "target_size", 0.03)
        )
        self.params.setdefault(
            "rsi_oversold", getattr(self.strategy_settings, "rsi_oversold", 30)
        )

        # Validate ranges
        if self.params["min_price_drop_pct"] <= 0:
            raise ValueError("min_price_drop_pct must be positive")
        if self.params["min_volume_multiplier"] <= 1.0:
            raise ValueError("min_volume_multiplier must be > 1.0")
        if self.params["lookback_periods"] < 5:
            raise ValueError("lookback_periods must be >= 5")

    def get_required_data(self) -> dict[str, Any]:
        """Specify required market data"""
        return {
            "ohlcv_timeframes": ["5m"],
            "orderbook_levels": 0,  # Not needed
            "indicators": [],
            "lookback_periods": max(self.params["lookback_periods"], 50),
        }

    def generate_signal(self, market_data: MarketFrame) -> Signal | None:
        """Generate stop loss reversion signal"""
        if (
            not market_data.ohlcv_5m
            or len(market_data.ohlcv_5m) < self.params["lookback_periods"]
        ):
            return None

        candles = market_data.ohlcv_5m
        current_candle = candles[-1]

        # Skip if price too low (avoid low cap manipulation)
        if float(current_candle.close) < self.params["min_price_level"]:
            return None

        # Check for sharp price drop in current candle
        price_drop_pct = self._calculate_price_drop(current_candle)
        if price_drop_pct < self.params["min_price_drop_pct"]:
            return None

        # Check for volume spike
        volume_multiplier = self._calculate_volume_spike(candles)
        if volume_multiplier < self.params["min_volume_multiplier"]:
            return None

        # Additional confirmation: RSI oversold
        prices = [float(c.close) for c in candles[-20:]]
        rsi = self.calculate_rsi(prices, 14)
        if rsi is None or rsi > self.params["rsi_oversold"]:  # Only trade if oversold
            return None

        # Check if we're at a potential support level
        support_strength = self._check_support_level(candles, float(current_candle.low))

        return Signal(
            symbol=market_data.symbol,
            signal_type=SignalType.BUY,
            confidence=min(
                0.8,
                self.params["confidence_base"]
                + (price_drop_pct / 10)
                + (volume_multiplier / 10)
                + (support_strength / 5),
            ),
            target_size=self.params["target_size"],
            entry_price=market_data.current_price,
            stop_loss=market_data.current_price
            * (1 - self.params["stop_loss_pct"] / 100),
            take_profit=market_data.current_price
            * (1 + self.params["take_profit_pct"] / 100),
            timeframe="5m",
            strategy_id="STOP_REV",
            reasoning=f"Stop reversion: -{price_drop_pct:.1f}% drop, {volume_multiplier:.1f}x volume, RSI {rsi:.1f}",
            metadata={
                "price_drop_pct": price_drop_pct,
                "volume_multiplier": volume_multiplier,
                "rsi": rsi,
                "support_strength": support_strength,
                "pattern_type": "liquidation_reversion",
            },
            timestamp=market_data.timestamp,
        )

    def _calculate_price_drop(self, candle) -> float:
        """Calculate price drop percentage within the candle"""
        open_price = float(candle.open)
        low_price = float(candle.low)

        if open_price == 0:
            return 0.0

        drop_pct = ((open_price - low_price) / open_price) * 100
        return drop_pct

    def _calculate_volume_spike(self, candles: list) -> float:
        """Calculate volume spike multiplier vs recent average"""
        if len(candles) < self.params["lookback_periods"] + 1:
            return 0.0

        current_volume = float(candles[-1].volume)
        recent_volumes = [
            float(c.volume) for c in candles[-self.params["lookback_periods"] - 1 : -1]
        ]

        if not recent_volumes or current_volume == 0:
            return 0.0

        avg_volume = statistics.mean(recent_volumes)
        if avg_volume == 0:
            return 0.0

        return current_volume / avg_volume

    def _check_support_level(self, candles: list, current_low: float) -> float:
        """Check if current low is near historical support level"""
        if len(candles) < 50:
            return 0.0

        # Get recent lows
        recent_lows = [float(c.low) for c in candles[-50:]]

        # Find how many times price touched similar levels
        tolerance = current_low * 0.02  # 2% tolerance
        support_touches = 0

        for low in recent_lows:
            if abs(low - current_low) <= tolerance:
                support_touches += 1

        # More touches = stronger support
        return min(5.0, support_touches)

    def _has_recent_bounce(self, candles: list) -> bool:
        """Check if there was a recent bounce (avoid re-entering)"""
        if len(candles) < 5:
            return False

        # Check last 5 candles for significant bounce
        for i in range(1, 5):
            candle = candles[-i]
            price_change = (
                (float(candle.close) - float(candle.open)) / float(candle.open) * 100
            )

            if price_change > 2.0:  # 2% bounce
                return True

        return False


def create_stop_reversion_strategy(
    config: dict[str, Any] | None = None,
) -> StopReversionStrategy:
    """Factory function to create StopReversionStrategy"""
    if config is None:
        config = {}

    strategy_config = StrategyConfig(name="stop_reversion", enabled=True, params=config)

    return StopReversionStrategy(strategy_config)
