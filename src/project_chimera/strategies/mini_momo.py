"""
Mini-Momentum Strategy
N-bar momentum strategy with configurable lookback period (default 7)
"""

from decimal import Decimal
from typing import Any

from ..domains.market import MarketFrame, Signal, SignalStrength, SignalType
from .base import StrategyConfig, TechnicalStrategy


class MiniMomentumStrategy(TechnicalStrategy):
    """
    N-Bar Momentum Strategy

    Logic:
    1. Calculate N-bar momentum (default 7 periods)
    2. Confirm with volume and RSI filters
    3. Entry on momentum threshold breach with confirmation
    4. Dynamic stop-loss based on ATR
    5. Momentum fade detection for exits
    """

    def __init__(self, config: StrategyConfig):
        # Set config attributes first
        self.config = config
        self.name = config.name
        self.enabled = config.enabled
        self.params = config.params

        # Strategy parameters with defaults
        self.momentum_period = self.params.get("momentum_period", 7)
        self.momentum_threshold = self.params.get("momentum_threshold", 0.02)  # 2%
        self.rsi_period = self.params.get("rsi_period", 14)
        self.rsi_oversold = self.params.get("rsi_oversold", 30)
        self.rsi_overbought = self.params.get("rsi_overbought", 70)
        self.volume_lookback = self.params.get("volume_lookback", 20)
        self.volume_threshold = self.params.get(
            "volume_threshold", 1.2
        )  # 20% above average
        self.atr_period = self.params.get("atr_period", 14)
        self.atr_multiplier = self.params.get("atr_multiplier", 2.0)
        self.min_lookback = self.params.get("min_lookback", 50)

        # Now validate after all parameters are set
        self.validate_config()

    def validate_config(self) -> None:
        """Validate strategy configuration"""
        if self.momentum_period < 2 or self.momentum_period > 50:
            raise ValueError("momentum_period must be between 2 and 50")

        if self.momentum_threshold <= 0 or self.momentum_threshold > 0.2:
            raise ValueError("momentum_threshold must be between 0 and 0.2 (20%)")

        if self.rsi_period < 5 or self.rsi_period > 30:
            raise ValueError("rsi_period must be between 5 and 30")

        if not (10 <= self.rsi_oversold <= 40):
            raise ValueError("rsi_oversold must be between 10 and 40")

        if not (60 <= self.rsi_overbought <= 90):
            raise ValueError("rsi_overbought must be between 60 and 90")

    def get_required_data(self) -> dict[str, Any]:
        """Specify required market data"""
        return {
            "ohlcv_timeframes": ["1m", "5m"],
            "orderbook_levels": 3,
            "indicators": ["momentum", "rsi", "atr", "volume_profile"],
            "lookback_periods": max(
                self.min_lookback, self.momentum_period + self.rsi_period
            ),
        }

    def calculate_momentum(self, prices: list[float], period: int) -> float | None:
        """Calculate N-period momentum as percentage change"""
        if len(prices) < period + 1:
            return None

        current_price = prices[-1]
        past_price = prices[-(period + 1)]

        if past_price == 0:
            return None

        return (current_price - past_price) / past_price

    def calculate_momentum_acceleration(
        self, prices: list[float], period: int
    ) -> float | None:
        """Calculate momentum acceleration (rate of change of momentum)"""
        if len(prices) < period + 2:
            return None

        current_momentum = self.calculate_momentum(prices, period)
        previous_momentum = self.calculate_momentum(prices[:-1], period)

        if current_momentum is None or previous_momentum is None:
            return None

        return current_momentum - previous_momentum

    async def generate_signal(self, market_data: MarketFrame) -> Signal | None:
        """Generate mini-momentum signal"""
        if not market_data.ohlcv_1m or len(market_data.ohlcv_1m) < self.min_lookback:
            return None

        # Extract price and volume data
        candles = market_data.ohlcv_1m
        closes = [float(c.close) for c in candles]
        volumes = [float(c.volume) for c in candles]
        highs = [float(c.high) for c in candles]
        lows = [float(c.low) for c in candles]

        current_price = float(market_data.current_price)
        current_volume = volumes[-1]
        avg_volume = sum(volumes[-self.volume_lookback :]) / min(
            self.volume_lookback, len(volumes)
        )

        # Calculate momentum
        momentum = self.calculate_momentum(closes, self.momentum_period)
        if momentum is None:
            return None

        # Calculate momentum acceleration
        momentum_accel = self.calculate_momentum_acceleration(
            closes, self.momentum_period
        )

        # Calculate RSI for confirmation
        rsi = self.calculate_rsi(closes, self.rsi_period)
        if rsi is None:
            return None

        # Calculate ATR for stop-loss
        atr = self.calculate_atr(highs, lows, closes, self.atr_period)

        # Volume confirmation
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
        is_volume_confirmed = volume_ratio >= self.volume_threshold

        # Signal logic
        signal_type = None
        strength = SignalStrength.MEDIUM
        confidence = 0.5

        # Bullish momentum
        if momentum > self.momentum_threshold:
            # Additional confirmations
            rsi_bullish = rsi < self.rsi_overbought  # Not overbought
            momentum_accelerating = momentum_accel is not None and momentum_accel > 0

            if rsi_bullish:
                signal_type = SignalType.BUY
                confidence = 0.6

                # Increase confidence with additional confirmations
                if is_volume_confirmed:
                    confidence += 0.15
                    strength = SignalStrength.STRONG

                if momentum_accelerating:
                    confidence += 0.1

                if momentum > self.momentum_threshold * 2:  # Strong momentum
                    confidence += 0.1
                    strength = SignalStrength.STRONG

        # Bearish momentum
        elif momentum < -self.momentum_threshold:
            # Additional confirmations
            rsi_bearish = rsi > self.rsi_oversold  # Not oversold
            momentum_accelerating = momentum_accel is not None and momentum_accel < 0

            if rsi_bearish:
                signal_type = SignalType.SELL
                confidence = 0.6

                # Increase confidence with additional confirmations
                if is_volume_confirmed:
                    confidence += 0.15
                    strength = SignalStrength.STRONG

                if momentum_accelerating:
                    confidence += 0.1

                if momentum < -self.momentum_threshold * 2:  # Strong momentum
                    confidence += 0.1
                    strength = SignalStrength.STRONG

        if signal_type is None:
            return None

        # Cap confidence at 0.9
        confidence = min(confidence, 0.9)

        # Calculate targets
        atr_value = atr if atr else (current_price * 0.01)  # 1% fallback

        if signal_type == SignalType.BUY:
            target_price = Decimal(str(current_price * (1 + abs(momentum) * 1.5)))
            stop_loss = Decimal(str(current_price - (atr_value * self.atr_multiplier)))
        else:
            target_price = Decimal(str(current_price * (1 - abs(momentum) * 1.5)))
            stop_loss = Decimal(str(current_price + (atr_value * self.atr_multiplier)))

        # Create signal
        signal = Signal(
            symbol=market_data.symbol,
            signal_type=signal_type,
            strength=strength,
            price=Decimal(str(current_price)),
            timestamp=market_data.timestamp,
            strategy_name=self.name,
            confidence=confidence,
            target_price=target_price,
            stop_loss=stop_loss,
            indicators_used={
                "momentum": momentum,
                "momentum_threshold": self.momentum_threshold,
                "momentum_accel": momentum_accel,
                "rsi": rsi,
                "volume_ratio": volume_ratio,
                "atr": atr_value,
                "avg_volume": avg_volume,
            },
            reasoning=f"Mini-momentum: {momentum:.4f} {'>' if momentum > 0 else '<'} "
            f"±{self.momentum_threshold}, RSI={rsi:.1f}, "
            f"volume_ratio={volume_ratio:.2f}, "
            f"momentum_accel={momentum_accel:.6f if momentum_accel else 'N/A'}",
        )

        return signal if signal.is_valid() else None


def create_mini_momentum_strategy(
    name: str = "mini_momentum",
    momentum_period: int = 7,
    momentum_threshold: float = 0.02,
    volume_threshold: float = 1.2,
    atr_multiplier: float = 2.0,
) -> MiniMomentumStrategy:
    """Factory function to create mini-momentum strategy"""

    config = StrategyConfig(
        name=name,
        enabled=True,
        params={
            "momentum_period": momentum_period,
            "momentum_threshold": momentum_threshold,
            "volume_threshold": volume_threshold,
            "atr_multiplier": atr_multiplier,
        },
    )

    return MiniMomentumStrategy(config)
