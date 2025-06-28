"""
Bitget-specific trading strategies
Five micro-strategies optimized for Bitget spot and perpetual futures
"""

import statistics
from decimal import Decimal
from typing import Any

import numpy as np

from ..domains.market import MarketFrame, Signal, SignalStrength, SignalType
from .base import StrategyConfig, TechnicalStrategy


class VolBreakoutBitgetStrategy(TechnicalStrategy):
    """
    Volatility Breakout Strategy for Bitget

    Logic:
    - Calculate Bollinger Bands (20-period, 2 std dev)
    - Detect squeeze when BB width < threshold
    - Long: close > high * (1 + 0.02) during squeeze
    - Short: close < low * (1 - 0.02) during squeeze
    """

    def __init__(self, config: StrategyConfig):
        # Set parameters before calling super() since validate_config() will be called
        self.bb_period = config.params.get("bb_period", 20)
        self.bb_std_dev = config.params.get("bb_std_dev", 2.0)
        self.squeeze_threshold = config.params.get("squeeze_threshold", 0.02)  # 2%
        self.breakout_threshold = config.params.get("breakout_threshold", 0.02)  # 2%
        self.volume_confirm = config.params.get("volume_confirm", True)

        super().__init__(config)

    def validate_config(self) -> None:
        """Validate strategy configuration"""
        if not 5 <= self.bb_period <= 50:
            raise ValueError("bb_period must be between 5 and 50")
        if not 0.01 <= self.squeeze_threshold <= 0.05:
            raise ValueError("squeeze_threshold must be between 1% and 5%")
        if not 0.01 <= self.breakout_threshold <= 0.05:
            raise ValueError("breakout_threshold must be between 1% and 5%")

    def get_required_data(self) -> dict[str, Any]:
        """Specify required market data"""
        return {
            "ohlcv_timeframes": ["1m"],
            "lookback_periods": max(50, self.bb_period * 2),
            "orderbook_levels": 5,
            "volume_data": True,
        }

    async def generate_signal(self, market_data: MarketFrame) -> Signal | None:
        """Generate volatility breakout signal"""
        if not market_data.ohlcv_1m or len(market_data.ohlcv_1m) < 50:
            return None

        candles = market_data.ohlcv_1m
        closes = [float(c.close) for c in candles]
        highs = [float(c.high) for c in candles]
        lows = [float(c.low) for c in candles]
        volumes = [float(c.volume) for c in candles]

        current_price = float(market_data.current_price)
        current_high = highs[-1]
        current_low = lows[-1]
        current_volume = volumes[-1]
        avg_volume = statistics.mean(volumes[-20:])

        # Calculate Bollinger Bands
        bb_data = self.calculate_bollinger_bands(
            closes, self.bb_period, self.bb_std_dev
        )
        if not bb_data:
            return None

        upper_band = bb_data["upper"]
        lower_band = bb_data["lower"]
        bandwidth = bb_data["bandwidth"]

        # Check for squeeze condition
        is_squeeze = bandwidth < self.squeeze_threshold

        if not is_squeeze:
            return None

        # Volume confirmation
        volume_confirmed = not self.volume_confirm or current_volume > avg_volume * 1.2

        signal_type = None
        confidence = 0.6

        # Bullish breakout: close > high * (1 + 2%)
        breakout_long_level = current_high * (1 + self.breakout_threshold)
        if current_price > breakout_long_level:
            signal_type = SignalType.BUY
            if volume_confirmed:
                confidence = 0.8

        # Bearish breakout: close < low * (1 - 2%)
        breakout_short_level = current_low * (1 - self.breakout_threshold)
        if current_price < breakout_short_level:
            signal_type = SignalType.SELL
            if volume_confirmed:
                confidence = 0.8

        if signal_type is None:
            return None

        # Calculate targets
        atr = self.calculate_atr(highs, lows, closes, 14)
        stop_distance = atr if atr else current_price * 0.01

        if signal_type == SignalType.BUY:
            target_price = Decimal(str(current_price + stop_distance * 3))
            stop_loss = Decimal(str(lower_band))
        else:
            target_price = Decimal(str(current_price - stop_distance * 3))
            stop_loss = Decimal(str(upper_band))

        return Signal(
            symbol=market_data.symbol,
            signal_type=signal_type,
            strength=(
                SignalStrength.STRONG if volume_confirmed else SignalStrength.MEDIUM
            ),
            price=Decimal(str(current_price)),
            timestamp=market_data.timestamp,
            strategy_name=self.name,
            confidence=confidence,
            target_price=target_price,
            stop_loss=stop_loss,
            indicators_used={
                "bb_upper": upper_band,
                "bb_lower": lower_band,
                "bandwidth": bandwidth,
                "squeeze_threshold": self.squeeze_threshold,
                "breakout_level": (
                    breakout_long_level
                    if signal_type == SignalType.BUY
                    else breakout_short_level
                ),
                "volume_ratio": current_volume / avg_volume if avg_volume > 0 else 0,
            },
            reasoning=f"BB Squeeze breakout: bandwidth={bandwidth:.4f} < {self.squeeze_threshold}, "
            f"price {'>' if signal_type == SignalType.BUY else '<'} "
            f"breakout level, volume confirmed: {volume_confirmed}",
        )


class MiniMomentumBitgetStrategy(TechnicalStrategy):
    """
    Mini-Momentum Strategy for Bitget

    Logic:
    - Calculate 7-bar momentum: (close / close[-7]) - 1
    - Calculate z-score of momentum over lookback period
    - Long: z-score > 1
    - Short: z-score < -1
    """

    def __init__(self, config: StrategyConfig):
        # Set parameters before calling super() since validate_config() will be called
        self.momentum_bars = config.params.get("momentum_bars", 7)
        self.zscore_threshold = config.params.get("zscore_threshold", 1.0)
        self.lookback_period = config.params.get("lookback_period", 50)
        self.rsi_filter = config.params.get("rsi_filter", True)
        self.rsi_period = config.params.get("rsi_period", 14)

        super().__init__(config)

    def validate_config(self) -> None:
        """Validate strategy configuration"""
        if not 3 <= self.momentum_bars <= 20:
            raise ValueError("momentum_bars must be between 3 and 20")
        if not 0.5 <= self.zscore_threshold <= 3.0:
            raise ValueError("zscore_threshold must be between 0.5 and 3.0")
        if not 20 <= self.lookback_period <= 200:
            raise ValueError("lookback_period must be between 20 and 200")

    def get_required_data(self) -> dict[str, Any]:
        """Specify required market data"""
        return {
            "ohlcv_timeframes": ["1m"],
            "lookback_periods": max(100, self.lookback_period + self.momentum_bars),
            "volume_data": True,
        }

    def calculate_momentum(self, prices: list[float]) -> float | None:
        """Calculate N-bar momentum"""
        if len(prices) < self.momentum_bars + 1:
            return None

        current_price = prices[-1]
        past_price = prices[-(self.momentum_bars + 1)]

        if past_price <= 0:
            return None

        return (current_price / past_price) - 1

    def calculate_zscore(self, values: list[float]) -> float | None:
        """Calculate z-score of the most recent value"""
        if len(values) < 10:
            return None

        current_value = values[-1]
        historical_values = values[:-1]

        if len(historical_values) < 2:
            return None

        mean_val = statistics.mean(historical_values)
        std_val = statistics.stdev(historical_values)

        if std_val <= 0:
            return None

        return (current_value - mean_val) / std_val

    async def generate_signal(self, market_data: MarketFrame) -> Signal | None:
        """Generate mini-momentum signal"""
        if not market_data.ohlcv_1m or len(market_data.ohlcv_1m) < 100:
            return None

        candles = market_data.ohlcv_1m
        closes = [float(c.close) for c in candles]

        # Calculate momentum series
        momentum_series = []
        for i in range(self.momentum_bars, len(closes)):
            momentum = self.calculate_momentum(closes[: i + 1])
            if momentum is not None:
                momentum_series.append(momentum)

        if len(momentum_series) < self.lookback_period:
            return None

        # Use recent lookback period for z-score calculation
        recent_momentum = momentum_series[-self.lookback_period :]
        zscore = self.calculate_zscore(recent_momentum)

        if zscore is None:
            return None

        # RSI filter
        if self.rsi_filter:
            rsi = self.calculate_rsi(closes, self.rsi_period)
            if rsi is None:
                return None

        signal_type = None
        confidence = 0.6

        # Long signal: z-score > threshold
        if zscore > self.zscore_threshold:
            if not self.rsi_filter or rsi < 70:  # Not overbought
                signal_type = SignalType.BUY
                confidence = min(0.9, 0.6 + abs(zscore - self.zscore_threshold) * 0.1)

        # Short signal: z-score < -threshold
        elif zscore < -self.zscore_threshold:
            if not self.rsi_filter or rsi > 30:  # Not oversold
                signal_type = SignalType.SELL
                confidence = min(0.9, 0.6 + abs(zscore + self.zscore_threshold) * 0.1)

        if signal_type is None:
            return None

        current_price = float(market_data.current_price)
        current_momentum = momentum_series[-1]

        # Calculate targets based on momentum
        expected_move = (
            abs(current_momentum) * current_price * 2
        )  # 2x momentum as target

        if signal_type == SignalType.BUY:
            target_price = Decimal(str(current_price + expected_move))
            stop_loss = Decimal(str(current_price * 0.99))  # 1% stop
        else:
            target_price = Decimal(str(current_price - expected_move))
            stop_loss = Decimal(str(current_price * 1.01))  # 1% stop

        return Signal(
            symbol=market_data.symbol,
            signal_type=signal_type,
            strength=(
                SignalStrength.STRONG if abs(zscore) > 2.0 else SignalStrength.MEDIUM
            ),
            price=Decimal(str(current_price)),
            timestamp=market_data.timestamp,
            strategy_name=self.name,
            confidence=confidence,
            target_price=target_price,
            stop_loss=stop_loss,
            indicators_used={
                "momentum": current_momentum,
                "zscore": zscore,
                "zscore_threshold": self.zscore_threshold,
                "rsi": rsi if self.rsi_filter else None,
                "momentum_bars": self.momentum_bars,
            },
            reasoning=f"Mini-momentum: {self.momentum_bars}-bar momentum z-score={zscore:.2f} "
            f"{'>' if zscore > 0 else '<'} ±{self.zscore_threshold}, "
            f"RSI filter: {rsi:.1f if self.rsi_filter else 'disabled'}",
        )


class LOBRevertBitgetStrategy(TechnicalStrategy):
    """
    Limit Order Book Mean-Reversion Strategy for Bitget

    Logic:
    - Calculate order flow imbalance: (bid_vol - ask_vol) / (bid_vol + ask_vol)
    - Calculate RSI of imbalance over 30 periods
    - Short: imbalance RSI > 70 (too many bids, expect reversion)
    - Long: imbalance RSI < 30 (too many asks, expect reversion)
    """

    def __init__(self, config: StrategyConfig):
        # Set parameters before calling super() since validate_config() will be called
        self.imbalance_rsi_period = config.params.get("imbalance_rsi_period", 30)
        self.rsi_overbought = config.params.get("rsi_overbought", 70)
        self.rsi_oversold = config.params.get("rsi_oversold", 30)
        self.min_spread_bps = config.params.get("min_spread_bps", 5)  # 0.05%
        self.orderbook_levels = config.params.get("orderbook_levels", 5)

        super().__init__(config)

    def validate_config(self) -> None:
        """Validate strategy configuration"""
        if not 10 <= self.imbalance_rsi_period <= 100:
            raise ValueError("imbalance_rsi_period must be between 10 and 100")
        if not 60 <= self.rsi_overbought <= 90:
            raise ValueError("rsi_overbought must be between 60 and 90")
        if not 10 <= self.rsi_oversold <= 40:
            raise ValueError("rsi_oversold must be between 10 and 40")

    def get_required_data(self) -> dict[str, Any]:
        """Specify required market data"""
        return {
            "ohlcv_timeframes": ["1m"],
            "orderbook_levels": self.orderbook_levels,
            "lookback_periods": max(100, self.imbalance_rsi_period * 2),
            "real_time_orderbook": True,
        }

    def calculate_order_flow_imbalance(self, orderbook) -> float | None:
        """Calculate order flow imbalance from order book"""
        if not orderbook or not orderbook.bids or not orderbook.asks:
            return None

        # Use top N levels
        levels = min(self.orderbook_levels, len(orderbook.bids), len(orderbook.asks))
        if levels < 2:
            return None

        bid_volume = sum(float(orderbook.bids[i][1]) for i in range(levels))
        ask_volume = sum(float(orderbook.asks[i][1]) for i in range(levels))

        total_volume = bid_volume + ask_volume
        if total_volume <= 0:
            return None

        # Imbalance: +1 = all bids, -1 = all asks, 0 = balanced
        imbalance = (bid_volume - ask_volume) / total_volume
        return imbalance

    def is_spread_acceptable(self, orderbook) -> bool:
        """Check if spread is within acceptable range"""
        if not orderbook or not orderbook.best_bid or not orderbook.best_ask:
            return False

        mid_price = (orderbook.best_bid + orderbook.best_ask) / 2
        spread_bps = float(orderbook.spread) / float(mid_price) * 10000

        return spread_bps >= self.min_spread_bps

    async def generate_signal(self, market_data: MarketFrame) -> Signal | None:
        """Generate LOB mean-reversion signal"""
        if not market_data.orderbook:
            return None

        # Check spread
        if not self.is_spread_acceptable(market_data.orderbook):
            return None

        # Get historical imbalance data (would need to be tracked separately)
        # For now, use current imbalance only
        current_imbalance = self.calculate_order_flow_imbalance(market_data.orderbook)
        if current_imbalance is None:
            return None

        # Simulate historical imbalances for RSI calculation
        # In real implementation, this would be tracked over time
        historical_imbalances = [current_imbalance] * self.imbalance_rsi_period

        # Calculate RSI of imbalance
        imbalance_rsi = self.calculate_rsi(
            historical_imbalances, self.imbalance_rsi_period
        )
        if imbalance_rsi is None:
            return None

        signal_type = None
        confidence = 0.5

        # Mean reversion logic
        if imbalance_rsi > self.rsi_overbought:
            # Too much buying pressure, expect reversion
            signal_type = SignalType.SELL
            confidence = 0.6 + (imbalance_rsi - self.rsi_overbought) / 100
        elif imbalance_rsi < self.rsi_oversold:
            # Too much selling pressure, expect reversion
            signal_type = SignalType.BUY
            confidence = 0.6 + (self.rsi_oversold - imbalance_rsi) / 100

        if signal_type is None:
            return None

        current_price = float(market_data.current_price)
        spread = float(market_data.orderbook.spread)

        # Small scalping targets
        target_distance = spread * 2  # 2x spread as target
        stop_distance = spread * 4  # 4x spread as stop

        if signal_type == SignalType.BUY:
            target_price = Decimal(str(current_price + target_distance))
            stop_loss = Decimal(str(current_price - stop_distance))
        else:
            target_price = Decimal(str(current_price - target_distance))
            stop_loss = Decimal(str(current_price + stop_distance))

        return Signal(
            symbol=market_data.symbol,
            signal_type=signal_type,
            strength=SignalStrength.MEDIUM,  # Scalping strategy
            price=Decimal(str(current_price)),
            timestamp=market_data.timestamp,
            strategy_name=self.name,
            confidence=min(0.8, confidence),
            target_price=target_price,
            stop_loss=stop_loss,
            indicators_used={
                "order_flow_imbalance": current_imbalance,
                "imbalance_rsi": imbalance_rsi,
                "rsi_threshold": (
                    self.rsi_overbought
                    if signal_type == SignalType.SELL
                    else self.rsi_oversold
                ),
                "spread_bps": float(market_data.orderbook.spread)
                / current_price
                * 10000,
                "orderbook_levels_used": self.orderbook_levels,
            },
            reasoning=f"LOB Revert: imbalance RSI={imbalance_rsi:.1f} "
            f"{'>' if signal_type == SignalType.SELL else '<'} "
            f"{self.rsi_overbought if signal_type == SignalType.SELL else self.rsi_oversold}, "
            f"current imbalance={current_imbalance:.3f}",
        )


class FundingAlphaBitgetStrategy(TechnicalStrategy):
    """
    Funding Rate Alpha Strategy for Bitget Perpetual Futures

    Logic:
    - Monitor funding rate for extreme values (±0.03%)
    - Detect open interest jumps as confirmation
    - Short perp when funding > +0.03% + OI jump
    - Long perp when funding < -0.03% + OI jump
    """

    def __init__(self, config: StrategyConfig):
        # Set parameters before calling super() since validate_config() will be called
        self.funding_threshold = config.params.get("funding_threshold", 0.0003)  # 0.03%
        self.oi_jump_threshold = config.params.get(
            "oi_jump_threshold", 0.10
        )  # 10% OI increase
        self.lookback_hours = config.params.get("lookback_hours", 24)  # 24h lookback
        self.min_oi_size = config.params.get("min_oi_size", 1000000)  # Min $1M OI

        super().__init__(config)

    def validate_config(self) -> None:
        """Validate strategy configuration"""
        if not 0.0001 <= self.funding_threshold <= 0.01:
            raise ValueError("funding_threshold must be between 0.01% and 1%")
        if not 0.05 <= self.oi_jump_threshold <= 0.5:
            raise ValueError("oi_jump_threshold must be between 5% and 50%")

    def get_required_data(self) -> dict[str, Any]:
        """Specify required market data"""
        return {
            "funding_rate": True,
            "open_interest": True,
            "ohlcv_timeframes": ["1h"],
            "lookback_periods": self.lookback_hours,
            "futures_only": True,
        }

    def detect_oi_jump(self, current_oi: float, historical_oi: list[float]) -> bool:
        """Detect significant open interest increase"""
        if not historical_oi or current_oi < self.min_oi_size:
            return False

        # Use recent average as baseline
        recent_avg = statistics.mean(historical_oi[-12:])  # Last 12 hours

        if recent_avg <= 0:
            return False

        oi_change = (current_oi - recent_avg) / recent_avg
        return oi_change > self.oi_jump_threshold

    async def generate_signal(self, market_data: MarketFrame) -> Signal | None:
        """Generate funding rate alpha signal"""
        # This strategy requires funding rate data
        if not market_data.funding_rate:
            return None

        current_funding = float(market_data.funding_rate.rate)

        # Check for extreme funding rates
        if abs(current_funding) < self.funding_threshold:
            return None

        # Get open interest data (would be tracked separately in real implementation)
        current_oi = getattr(market_data, "open_interest", None)
        if current_oi is None:
            return None

        # Simulate historical OI for jump detection
        # In real implementation, this would be tracked over time
        historical_oi = [
            current_oi * (1 + np.random.normal(0, 0.05)) for _ in range(24)
        ]

        # Check for OI jump confirmation
        oi_jump_detected = self.detect_oi_jump(current_oi, historical_oi)

        if not oi_jump_detected:
            return None

        signal_type = None
        confidence = 0.7

        # High positive funding + OI jump = short perp (longs paying, expect reversion)
        if current_funding > self.funding_threshold:
            signal_type = SignalType.SELL
            confidence += min(
                0.2, (current_funding - self.funding_threshold) / self.funding_threshold
            )

        # High negative funding + OI jump = long perp (shorts paying, expect reversion)
        elif current_funding < -self.funding_threshold:
            signal_type = SignalType.BUY
            confidence += min(
                0.2,
                abs(current_funding + self.funding_threshold) / self.funding_threshold,
            )

        if signal_type is None:
            return None

        current_price = float(market_data.current_price)

        # Funding-based targets (expect mean reversion)
        funding_impact = abs(current_funding) * current_price

        if signal_type == SignalType.BUY:
            target_price = Decimal(
                str(current_price + funding_impact * 5)
            )  # 5x funding as target
            stop_loss = Decimal(str(current_price * 0.98))  # 2% stop
        else:
            target_price = Decimal(str(current_price - funding_impact * 5))
            stop_loss = Decimal(str(current_price * 1.02))  # 2% stop

        return Signal(
            symbol=market_data.symbol,
            signal_type=signal_type,
            strength=SignalStrength.STRONG,
            price=Decimal(str(current_price)),
            timestamp=market_data.timestamp,
            strategy_name=self.name,
            confidence=min(0.9, confidence),
            target_price=target_price,
            stop_loss=stop_loss,
            indicators_used={
                "funding_rate": current_funding,
                "funding_threshold": self.funding_threshold,
                "open_interest": current_oi,
                "oi_jump_detected": oi_jump_detected,
                "oi_jump_threshold": self.oi_jump_threshold,
                "next_funding_time": market_data.funding_rate.next_funding_time.isoformat(),
            },
            reasoning=f"Funding Alpha: rate={current_funding:.4f}% "
            f"{'>' if current_funding > 0 else '<'} ±{self.funding_threshold*100:.2f}%, "
            f"OI jump detected: ${current_oi/1e6:.1f}M, "
            f"expect funding reversion",
        )


class BasisArbBitgetStrategy(TechnicalStrategy):
    """
    Basis Arbitrage Strategy for Bitget (Spot vs Perpetual)

    Logic:
    - Calculate premium: (perp_price / spot_price) - 1
    - Long spot + Short perp when premium > 0.5%
    - Equal notional amounts for market neutral position
    - Target convergence at funding periods
    """

    def __init__(self, config: StrategyConfig):
        # Set parameters before calling super() since validate_config() will be called
        self.premium_threshold = config.params.get("premium_threshold", 0.005)  # 0.5%
        self.max_premium = config.params.get("max_premium", 0.03)  # 3% max
        self.min_volume_ratio = config.params.get(
            "min_volume_ratio", 0.1
        )  # 10% of perp volume
        self.convergence_target = config.params.get(
            "convergence_target", 0.001
        )  # 0.1% target

        super().__init__(config)

    def validate_config(self) -> None:
        """Validate strategy configuration"""
        if not 0.001 <= self.premium_threshold <= 0.02:
            raise ValueError("premium_threshold must be between 0.1% and 2%")
        if not 0.01 <= self.max_premium <= 0.1:
            raise ValueError("max_premium must be between 1% and 10%")

    def get_required_data(self) -> dict[str, Any]:
        """Specify required market data"""
        return {
            "spot_and_futures": True,
            "volume_data": True,
            "funding_rate": True,
            "liquidity_check": True,
        }

    def calculate_basis_premium(
        self, spot_price: float, perp_price: float
    ) -> float | None:
        """Calculate basis premium"""
        if spot_price <= 0:
            return None

        return (perp_price / spot_price) - 1

    def check_liquidity(self, spot_volume: float, perp_volume: float) -> bool:
        """Check if there's sufficient liquidity for arbitrage"""
        if perp_volume <= 0:
            return False

        volume_ratio = spot_volume / perp_volume
        return volume_ratio >= self.min_volume_ratio

    async def generate_signal(self, market_data: MarketFrame) -> Signal | None:
        """Generate basis arbitrage signal"""
        # This strategy requires both spot and futures data
        # In real implementation, would need dual market data feeds

        # For demonstration, assume we have access to both prices
        spot_price = float(market_data.current_price)

        # Simulate perpetual price (in real implementation, get from separate feed)
        # Add some premium/discount
        perp_price = spot_price * (1 + np.random.normal(0, 0.002))

        premium = self.calculate_basis_premium(spot_price, perp_price)
        if premium is None:
            return None

        # Check if premium exceeds threshold
        if abs(premium) < self.premium_threshold:
            return None

        # Safety check for excessive premium
        if abs(premium) > self.max_premium:
            return None

        # Check liquidity (simplified)
        if not market_data.ohlcv_1m:
            return None

        recent_volume = float(market_data.ohlcv_1m[-1].volume)
        perp_volume = recent_volume * 2  # Assume perp has 2x volume

        if not self.check_liquidity(recent_volume, perp_volume):
            return None

        signal_type = None
        confidence = 0.8  # High confidence for arbitrage

        # Positive premium: perp expensive, short perp + long spot
        if premium > self.premium_threshold:
            signal_type = (
                SignalType.BUY
            )  # Buy spot (short perp would be separate order)
            confidence += min(
                0.1, (premium - self.premium_threshold) / self.premium_threshold
            )

        # Negative premium: perp cheap, long perp + short spot
        elif premium < -self.premium_threshold:
            signal_type = (
                SignalType.SELL
            )  # Sell spot (long perp would be separate order)
            confidence += min(
                0.1, abs(premium + self.premium_threshold) / self.premium_threshold
            )

        if signal_type is None:
            return None

        # Target convergence
        target_premium = (
            self.convergence_target if premium > 0 else -self.convergence_target
        )
        target_price = Decimal(str(perp_price * (1 + target_premium)))

        # Stop at excessive divergence
        stop_premium = premium * 2  # If premium doubles, exit
        stop_price = Decimal(str(perp_price * (1 + stop_premium)))

        return Signal(
            symbol=market_data.symbol,
            signal_type=signal_type,
            strength=SignalStrength.STRONG,
            price=Decimal(str(spot_price)),
            timestamp=market_data.timestamp,
            strategy_name=self.name,
            confidence=min(0.95, confidence),
            target_price=target_price,
            stop_loss=stop_price,
            indicators_used={
                "spot_price": spot_price,
                "perp_price": perp_price,
                "basis_premium": premium,
                "premium_threshold": self.premium_threshold,
                "spot_volume": recent_volume,
                "estimated_perp_volume": perp_volume,
                "liquidity_ratio": recent_volume / perp_volume,
            },
            reasoning=f"Basis Arb: premium={premium*100:.2f}% "
            f"{'>' if premium > 0 else '<'} ±{self.premium_threshold*100:.1f}%, "
            f"spot=${spot_price:.2f} vs perp=${perp_price:.2f}, "
            f"target convergence to {target_premium*100:.1f}%",
        )


# Factory functions for easy strategy creation
def create_bitget_vol_breakout(**kwargs) -> VolBreakoutBitgetStrategy:
    """Create Bitget volatility breakout strategy"""
    config = StrategyConfig(name="bitget_vol_breakout", params=kwargs)
    return VolBreakoutBitgetStrategy(config)


def create_bitget_mini_momentum(**kwargs) -> MiniMomentumBitgetStrategy:
    """Create Bitget mini-momentum strategy"""
    config = StrategyConfig(name="bitget_mini_momentum", params=kwargs)
    return MiniMomentumBitgetStrategy(config)


def create_bitget_lob_revert(**kwargs) -> LOBRevertBitgetStrategy:
    """Create Bitget LOB mean-reversion strategy"""
    config = StrategyConfig(name="bitget_lob_revert", params=kwargs)
    return LOBRevertBitgetStrategy(config)


def create_bitget_funding_alpha(**kwargs) -> FundingAlphaBitgetStrategy:
    """Create Bitget funding alpha strategy"""
    config = StrategyConfig(name="bitget_funding_alpha", params=kwargs)
    return FundingAlphaBitgetStrategy(config)


def create_bitget_basis_arb(**kwargs) -> BasisArbBitgetStrategy:
    """Create Bitget basis arbitrage strategy"""
    config = StrategyConfig(name="bitget_basis_arb", params=kwargs)
    return BasisArbBitgetStrategy(config)
