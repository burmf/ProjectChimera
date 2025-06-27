"""
Volatility Breakout Strategy (VOL_BRK)
Exploits volatility expansion: BB squeeze & ±2% breakout → momentum follow
"""

import statistics
from typing import Any

from ..domains.market import MarketFrame, Signal, SignalType
from .base import StrategyConfig, TechnicalStrategy


class VolatilityBreakoutStrategy(TechnicalStrategy):
    """
    Volatility Breakout Strategy

    Core trigger: BB squeeze & ±2% breakout → momentum follow
    Exploits volatility expansion after consolidation periods
    """

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

    def validate_config(self) -> None:
        """Validate strategy configuration"""
        # Set default parameters
        self.params.setdefault("bb_period", 20)  # Bollinger Bands period
        self.params.setdefault("bb_std_dev", 2.0)  # Standard deviations
        self.params.setdefault("squeeze_threshold", 0.05)  # Bandwidth % for squeeze
        self.params.setdefault("breakout_threshold_pct", 2.0)  # Breakout percentage
        self.params.setdefault("volume_confirmation", True)  # Require volume spike
        self.params.setdefault("min_volume_multiplier", 1.5)  # Volume multiplier
        self.params.setdefault("consolidation_periods", 10)  # Min periods in squeeze
        self.params.setdefault("max_position_hours", 4)  # Max hold time
        self.params.setdefault("stop_loss_pct", 1.5)  # Stop loss %
        self.params.setdefault("take_profit_pct", 3.0)  # Take profit %
        self.params.setdefault("atr_period", 14)  # ATR period for volatility
        self.params.setdefault("timeframe", "15m")  # Primary timeframe

        # Validate ranges
        if self.params["bb_period"] < 10:
            raise ValueError("bb_period must be >= 10")
        if self.params["bb_std_dev"] <= 0:
            raise ValueError("bb_std_dev must be positive")
        if self.params["squeeze_threshold"] <= 0:
            raise ValueError("squeeze_threshold must be positive")
        if self.params["breakout_threshold_pct"] <= 0:
            raise ValueError("breakout_threshold_pct must be positive")

    def get_required_data(self) -> dict[str, Any]:
        """Specify required market data"""
        return {
            "ohlcv_timeframes": ["15m"],
            "orderbook_levels": 0,
            "indicators": ["bb", "atr"],
            "lookback_periods": max(self.params["bb_period"] * 2, 50),
        }

    def generate_signal(self, market_data: MarketFrame) -> Signal | None:
        """Generate volatility breakout signal"""
        if (
            not market_data.ohlcv_15m
            or len(market_data.ohlcv_15m) < self.params["bb_period"] + 10
        ):
            return None

        candles = market_data.ohlcv_15m
        current_candle = candles[-1]
        prices = [float(c.close) for c in candles]

        # Calculate Bollinger Bands
        bb = self.calculate_bollinger_bands(
            prices, self.params["bb_period"], self.params["bb_std_dev"]
        )
        if bb is None:
            return None

        # Check for Bollinger Band squeeze
        is_squeezed = self._is_bollinger_squeeze(candles, bb)
        if not is_squeezed:
            return None

        # Check for breakout
        current_price = float(current_candle.close)
        breakout_direction = self._detect_breakout(current_price, bb, candles)
        if breakout_direction is None:
            return None

        # Volume confirmation
        if self.params["volume_confirmation"]:
            volume_spike = self._check_volume_spike(candles)
            if not volume_spike:
                return None

        # Calculate ATR for dynamic stops
        highs = [float(c.high) for c in candles]
        lows = [float(c.low) for c in candles]
        closes = [float(c.close) for c in candles]
        atr = self.calculate_atr(highs, lows, closes, self.params["atr_period"])

        # Determine signal type and parameters
        if breakout_direction == "up":
            signal_type = SignalType.BUY
            reasoning = f"Vol breakout LONG: BB squeeze, +{self._calculate_breakout_magnitude(current_price, bb):.1f}% breakout"
            stop_loss = (
                current_price - (atr * 1.5)
                if atr
                else current_price * (1 - self.params["stop_loss_pct"] / 100)
            )
            take_profit = current_price * (1 + self.params["take_profit_pct"] / 100)
        else:  # breakout_direction == 'down'
            signal_type = SignalType.SELL
            reasoning = f"Vol breakout SHORT: BB squeeze, -{self._calculate_breakout_magnitude(current_price, bb):.1f}% breakout"
            stop_loss = (
                current_price + (atr * 1.5)
                if atr
                else current_price * (1 + self.params["stop_loss_pct"] / 100)
            )
            take_profit = current_price * (1 - self.params["take_profit_pct"] / 100)

        # Calculate confidence based on squeeze duration and breakout strength
        squeeze_duration = self._calculate_squeeze_duration(candles)
        breakout_magnitude = self._calculate_breakout_magnitude(current_price, bb)
        confidence = min(
            0.8, 0.5 + (squeeze_duration / 20) * 0.2 + (breakout_magnitude / 5) * 0.1
        )

        return Signal(
            symbol=market_data.symbol,
            signal_type=signal_type,
            confidence=confidence,
            target_size=0.04,  # 4% position size
            entry_price=market_data.current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timeframe="15m",
            strategy_id="VOL_BRK",
            reasoning=reasoning,
            metadata={
                "bb_bandwidth": bb["bandwidth"],
                "squeeze_duration": squeeze_duration,
                "breakout_magnitude": breakout_magnitude,
                "atr": atr,
                "breakout_direction": breakout_direction,
                "pattern_type": "volatility_breakout",
            },
            timestamp=market_data.timestamp,
        )

    def _is_bollinger_squeeze(self, candles: list, bb: dict[str, float]) -> bool:
        """Check if Bollinger Bands are in a squeeze (low volatility)"""
        # Check current bandwidth
        current_bandwidth = bb["bandwidth"]
        if current_bandwidth > self.params["squeeze_threshold"]:
            return False

        # Check if we've been in squeeze for minimum periods
        if (
            len(candles)
            < self.params["bb_period"] + self.params["consolidation_periods"]
        ):
            return False

        # Check historical bandwidth to confirm squeeze
        squeeze_count = 0
        for i in range(self.params["consolidation_periods"]):
            candle_idx = -(i + 1)
            if abs(candle_idx) > len(candles):
                break

            historical_prices = [float(c.close) for c in candles[:candle_idx]]
            if len(historical_prices) < self.params["bb_period"]:
                continue

            historical_bb = self.calculate_bollinger_bands(
                historical_prices, self.params["bb_period"], self.params["bb_std_dev"]
            )

            if (
                historical_bb
                and historical_bb["bandwidth"] <= self.params["squeeze_threshold"]
            ):
                squeeze_count += 1

        return squeeze_count >= self.params["consolidation_periods"] * 0.7

    def _detect_breakout(
        self, current_price: float, bb: dict[str, float], candles: list
    ) -> str | None:
        """Detect breakout direction and magnitude"""
        upper_band = bb["upper"]
        lower_band = bb["lower"]
        middle_band = bb["middle"]

        # Check for breakout above upper band
        if current_price > upper_band:
            breakout_pct = ((current_price - middle_band) / middle_band) * 100
            if breakout_pct >= self.params["breakout_threshold_pct"]:
                return "up"

        # Check for breakout below lower band
        elif current_price < lower_band:
            breakout_pct = ((middle_band - current_price) / middle_band) * 100
            if breakout_pct >= self.params["breakout_threshold_pct"]:
                return "down"

        return None

    def _check_volume_spike(self, candles: list) -> bool:
        """Check for volume confirmation of breakout"""
        if len(candles) < 20:
            return True  # Assume confirmed if not enough data

        current_volume = float(candles[-1].volume)
        recent_volumes = [float(c.volume) for c in candles[-20:-1]]

        if not recent_volumes or current_volume == 0:
            return True

        avg_volume = statistics.mean(recent_volumes)
        if avg_volume == 0:
            return True

        volume_ratio = current_volume / avg_volume
        return volume_ratio >= self.params["min_volume_multiplier"]

    def _calculate_squeeze_duration(self, candles: list) -> int:
        """Calculate how long we've been in a squeeze"""
        squeeze_periods = 0

        for i in range(min(30, len(candles) - self.params["bb_period"])):
            candle_idx = -(i + 1)
            if abs(candle_idx) > len(candles):
                break

            historical_prices = [float(c.close) for c in candles[:candle_idx]]
            if len(historical_prices) < self.params["bb_period"]:
                break

            bb = self.calculate_bollinger_bands(
                historical_prices, self.params["bb_period"], self.params["bb_std_dev"]
            )

            if bb and bb["bandwidth"] <= self.params["squeeze_threshold"]:
                squeeze_periods += 1
            else:
                break

        return squeeze_periods

    def _calculate_breakout_magnitude(
        self, current_price: float, bb: dict[str, float]
    ) -> float:
        """Calculate breakout magnitude as percentage"""
        middle_band = bb["middle"]

        if middle_band == 0:
            return 0.0

        return abs((current_price - middle_band) / middle_band) * 100


def create_volatility_breakout_strategy(
    config: dict[str, Any] | None = None,
) -> VolatilityBreakoutStrategy:
    """Factory function to create VolatilityBreakoutStrategy"""
    if config is None:
        config = {}

    strategy_config = StrategyConfig(
        name="volatility_breakout", enabled=True, params=config
    )

    return VolatilityBreakoutStrategy(strategy_config)
