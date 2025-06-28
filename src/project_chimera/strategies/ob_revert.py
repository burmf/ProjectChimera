"""
Order-Book Mean-Reversion Strategy
Exploits bid/ask imbalance for mean-reversion opportunities
"""

from decimal import Decimal
from typing import Any

from ..domains.market import MarketFrame, Signal, SignalStrength, SignalType
from .base import StrategyConfig, TechnicalStrategy


class OrderBookMeanReversionStrategy(TechnicalStrategy):
    """
    Order Book Mean-Reversion Strategy

    Logic:
    1. Calculate order book imbalance ratio
    2. Detect extreme imbalances as mean-reversion opportunities
    3. Confirm with price action and volume
    4. Enter counter-trend positions expecting reversion
    5. Quick exits when imbalance normalizes
    """

    def __init__(self, config: StrategyConfig):
        # Strategy parameters with defaults - set before calling super()
        self.imbalance_threshold = config.params.get(
            "imbalance_threshold", 0.3
        )  # 30% imbalance
        self.extreme_imbalance = config.params.get(
            "extreme_imbalance", 0.5
        )  # 50% for strong signals
        self.price_deviation_threshold = config.params.get(
            "price_deviation_threshold", 0.005
        )  # 0.5%
        self.volume_confirmation = config.params.get("volume_confirmation", True)
        self.volume_threshold = config.params.get(
            "volume_threshold", 1.5
        )  # 50% above average
        self.spread_max_pct = config.params.get(
            "spread_max_pct", 0.001
        )  # Max 0.1% spread
        self.orderbook_levels = config.params.get("orderbook_levels", 10)
        self.sma_period = config.params.get("sma_period", 20)
        self.atr_period = config.params.get("atr_period", 14)
        self.min_lookback = config.params.get("min_lookback", 30)

        super().__init__(config)

    def validate_config(self) -> None:
        """Validate strategy configuration"""
        if not (0.1 <= self.imbalance_threshold <= 0.8):
            raise ValueError("imbalance_threshold must be between 0.1 and 0.8")

        if not (0.2 <= self.extreme_imbalance <= 0.9):
            raise ValueError("extreme_imbalance must be between 0.2 and 0.9")

        if self.extreme_imbalance <= self.imbalance_threshold:
            raise ValueError(
                "extreme_imbalance must be greater than imbalance_threshold"
            )

        if not (0.0001 <= self.spread_max_pct <= 0.01):
            raise ValueError("spread_max_pct must be between 0.0001 and 0.01")

        if not (5 <= self.orderbook_levels <= 50):
            raise ValueError("orderbook_levels must be between 5 and 50")

    def get_required_data(self) -> dict[str, Any]:
        """Specify required market data"""
        return {
            "ohlcv_timeframes": ["1m"],
            "orderbook_levels": self.orderbook_levels,
            "indicators": ["sma", "atr", "volume_profile"],
            "lookback_periods": max(self.min_lookback, self.sma_period),
        }

    def calculate_weighted_imbalance(self, orderbook) -> float | None:
        """
        Calculate weighted order book imbalance considering volume at different levels
        """
        if not orderbook or not orderbook.bids or not orderbook.asks:
            return None

        # Take top N levels
        n_levels = min(self.orderbook_levels, len(orderbook.bids), len(orderbook.asks))

        if n_levels < 3:  # Need minimum levels for reliable calculation
            return None

        weighted_bid_volume = 0.0
        weighted_ask_volume = 0.0

        # Calculate weighted volumes (higher weight for closer levels)
        for i in range(n_levels):
            weight = 1.0 / (i + 1)  # Decreasing weight for deeper levels

            if i < len(orderbook.bids):
                weighted_bid_volume += float(orderbook.bids[i][1]) * weight

            if i < len(orderbook.asks):
                weighted_ask_volume += float(orderbook.asks[i][1]) * weight

        total_weighted_volume = weighted_bid_volume + weighted_ask_volume

        if total_weighted_volume == 0:
            return None

        # Imbalance: positive = bid heavy, negative = ask heavy
        imbalance = (weighted_bid_volume - weighted_ask_volume) / total_weighted_volume

        return imbalance

    def is_spread_acceptable(self, orderbook) -> bool:
        """Check if bid-ask spread is within acceptable range"""
        if not orderbook or not orderbook.best_bid or not orderbook.best_ask:
            return False

        mid_price = (orderbook.best_bid + orderbook.best_ask) / 2
        spread_pct = float(orderbook.spread) / float(mid_price)

        return spread_pct <= self.spread_max_pct

    async def generate_signal(self, market_data: MarketFrame) -> Signal | None:
        """Generate order book mean-reversion signal"""
        if not market_data.orderbook or not market_data.ohlcv_1m:
            return None

        if len(market_data.ohlcv_1m) < self.min_lookback:
            return None

        # Check spread acceptability
        if not self.is_spread_acceptable(market_data.orderbook):
            return None

        # Extract price and volume data
        candles = market_data.ohlcv_1m
        closes = [float(c.close) for c in candles]
        volumes = [float(c.volume) for c in candles]
        highs = [float(c.high) for c in candles]
        lows = [float(c.low) for c in candles]

        current_price = float(market_data.current_price)
        mid_price = (
            float(market_data.orderbook.best_bid)
            + float(market_data.orderbook.best_ask)
        ) / 2

        # Calculate order book imbalance
        imbalance = self.calculate_weighted_imbalance(market_data.orderbook)
        if imbalance is None:
            return None

        # Calculate reference indicators
        sma = self.calculate_sma(closes, self.sma_period)
        if sma is None:
            return None

        atr = self.calculate_atr(highs, lows, closes, self.atr_period)

        # Volume analysis
        current_volume = volumes[-1]
        avg_volume = sum(volumes[-20:]) / min(20, len(volumes))
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

        # Price deviation from SMA
        price_deviation = (current_price - sma) / sma

        # Signal logic - Mean reversion on extreme imbalances
        signal_type = None
        strength = SignalStrength.MEDIUM
        confidence = 0.5

        # Strong bid imbalance + price above SMA = potential sell (reversion)
        if (
            imbalance > self.imbalance_threshold
            and price_deviation > self.price_deviation_threshold
        ):
            signal_type = SignalType.SELL
            confidence = 0.6

            # Extreme conditions
            if imbalance > self.extreme_imbalance:
                strength = SignalStrength.STRONG
                confidence = 0.75

            # Volume confirmation
            if self.volume_confirmation and volume_ratio >= self.volume_threshold:
                confidence += 0.1

            # Price significantly above SMA increases confidence
            if abs(price_deviation) > self.price_deviation_threshold * 2:
                confidence += 0.05

        # Strong ask imbalance + price below SMA = potential buy (reversion)
        elif (
            imbalance < -self.imbalance_threshold
            and price_deviation < -self.price_deviation_threshold
        ):
            signal_type = SignalType.BUY
            confidence = 0.6

            # Extreme conditions
            if imbalance < -self.extreme_imbalance:
                strength = SignalStrength.STRONG
                confidence = 0.75

            # Volume confirmation
            if self.volume_confirmation and volume_ratio >= self.volume_threshold:
                confidence += 0.1

            # Price significantly below SMA increases confidence
            if abs(price_deviation) > self.price_deviation_threshold * 2:
                confidence += 0.05

        if signal_type is None:
            return None

        # Cap confidence
        confidence = min(confidence, 0.9)

        # Calculate targets - Mean reversion targets
        atr_value = atr if atr else (current_price * 0.005)  # 0.5% fallback

        if signal_type == SignalType.BUY:
            # Target: reversion toward SMA
            target_price = Decimal(str(sma))
            stop_loss = Decimal(str(current_price - (atr_value * 1.5)))
        else:
            # Target: reversion toward SMA
            target_price = Decimal(str(sma))
            stop_loss = Decimal(str(current_price + (atr_value * 1.5)))

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
                "orderbook_imbalance": imbalance,
                "imbalance_threshold": self.imbalance_threshold,
                "price_deviation": price_deviation,
                "sma": sma,
                "volume_ratio": volume_ratio,
                "spread_pct": float(market_data.orderbook.spread) / mid_price,
                "mid_price": mid_price,
                "atr": atr_value,
            },
            reasoning=f"OB Mean-reversion: imbalance={imbalance:.3f} "
            f"({'bid' if imbalance > 0 else 'ask'} heavy), "
            f"price_dev={price_deviation:.4f}, "
            f"volume_ratio={volume_ratio:.2f}, "
            f"expecting reversion to SMA={sma:.2f}",
        )

        return signal if signal.is_valid() else None


def create_orderbook_reversion_strategy(
    name: str = "orderbook_mean_reversion",
    imbalance_threshold: float = 0.3,
    extreme_imbalance: float = 0.5,
    price_deviation_threshold: float = 0.005,
    volume_threshold: float = 1.5,
) -> OrderBookMeanReversionStrategy:
    """Factory function to create order book mean-reversion strategy"""

    config = StrategyConfig(
        name=name,
        enabled=True,
        params={
            "imbalance_threshold": imbalance_threshold,
            "extreme_imbalance": extreme_imbalance,
            "price_deviation_threshold": price_deviation_threshold,
            "volume_threshold": volume_threshold,
        },
    )

    return OrderBookMeanReversionStrategy(config)
