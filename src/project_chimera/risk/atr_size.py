"""
ATR-based position sizing for volatility targeting
Targets specific daily volatility (default 1%) using Average True Range
"""

import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from ..domains.market import OHLCV


@dataclass
class VolatilityTarget:
    """Volatility targeting configuration"""

    target_daily_vol: float  # Target daily portfolio volatility (e.g., 0.01 = 1%)
    lookback_periods: int  # ATR calculation periods
    vol_adjustment_factor: float  # Adjustment for vol regime changes
    min_position_size: float  # Minimum position as % of portfolio
    max_position_size: float  # Maximum position as % of portfolio
    vol_floor: float  # Minimum ATR to prevent division by zero
    vol_ceiling: float  # Maximum ATR to cap extreme volatility


@dataclass
class ATRSizingResult:
    """ATR position sizing result"""

    position_size_pct: float  # Position size as % of portfolio
    atr_value: float  # Current ATR value
    price_volatility: float  # Price volatility (ATR/price)
    vol_adjustment: float  # Volatility regime adjustment
    target_met: bool  # Whether target vol can be achieved
    confidence: float  # Confidence in sizing (0.0 to 1.0)

    def is_valid(self) -> bool:
        """Check if sizing result is valid"""
        return (
            0.0 <= self.position_size_pct <= 1.0
            and self.atr_value > 0
            and self.confidence > 0.3
        )


class ATRPositionSizer:
    """
    ATR-based position sizing for volatility targeting

    Features:
    - Target daily portfolio volatility (e.g., 1%)
    - Dynamic position sizing based on instrument volatility
    - Volatility regime detection and adjustment
    - Confidence scoring based on data quality
    - Risk-adjusted sizing with floor/ceiling constraints
    """

    def __init__(
        self,
        target_daily_vol: float = 0.01,  # 1% daily volatility target
        atr_periods: int = 14,  # ATR calculation periods
        vol_lookback: int = 50,  # Volatility regime lookback
        min_position_pct: float = 0.001,  # 0.1% minimum position
        max_position_pct: float = 0.50,  # 50% maximum position
        vol_floor: float = 0.0001,  # 0.01% volatility floor
        vol_ceiling: float = 0.20,  # 20% volatility ceiling
        confidence_threshold: float = 0.5,  # Minimum confidence for trading
        regime_sensitivity: float = 1.5,  # Vol regime adjustment sensitivity
    ):
        self.target_config = VolatilityTarget(
            target_daily_vol=target_daily_vol,
            lookback_periods=atr_periods,
            vol_adjustment_factor=1.0,
            min_position_size=min_position_pct,
            max_position_size=max_position_pct,
            vol_floor=vol_floor,
            vol_ceiling=vol_ceiling,
        )

        self.vol_lookback = vol_lookback
        self.confidence_threshold = confidence_threshold
        self.regime_sensitivity = regime_sensitivity

        # Historical data for volatility regime detection
        self.atr_history: list[tuple[datetime, float]] = []
        self.vol_regime_multiplier = 1.0

    def calculate_position_size(
        self,
        current_price: float,
        ohlcv_data: list[OHLCV],
        portfolio_value: float,
        timestamp: datetime | None = None,
    ) -> ATRSizingResult:
        """
        Calculate optimal position size based on ATR and volatility target

        Formula:
        Position Size = (Target Vol * Portfolio Value) / (ATR * Price)

        Where:
        - Target Vol = desired daily portfolio volatility
        - ATR = Average True Range (proxy for daily price movement)
        """

        if timestamp is None:
            timestamp = datetime.now()

        # Calculate ATR
        atr_value = self._calculate_atr(ohlcv_data)
        if atr_value <= 0:
            return ATRSizingResult(
                position_size_pct=0.0,
                atr_value=0.0,
                price_volatility=0.0,
                vol_adjustment=1.0,
                target_met=False,
                confidence=0.0,
            )

        # Apply volatility floor and ceiling
        atr_value = max(self.target_config.vol_floor * current_price, atr_value)
        atr_value = min(self.target_config.vol_ceiling * current_price, atr_value)

        # Update volatility regime tracking
        self._update_vol_regime(atr_value, current_price, timestamp)

        # Calculate price volatility (ATR as % of price)
        price_volatility = atr_value / current_price

        # Calculate base position size for target volatility
        # Position Size = (Target Daily Vol) / (Price Daily Vol)
        base_position_size = self.target_config.target_daily_vol / price_volatility

        # Apply volatility regime adjustment
        vol_adjustment = self._get_vol_regime_adjustment()
        adjusted_position_size = base_position_size * vol_adjustment

        # Apply min/max constraints
        position_size_pct = max(
            self.target_config.min_position_size,
            min(self.target_config.max_position_size, adjusted_position_size),
        )

        # Calculate confidence
        confidence = self._calculate_confidence(ohlcv_data, atr_value, price_volatility)

        # Check if target can be realistically met
        target_met = (
            self.target_config.min_position_size
            <= base_position_size
            <= self.target_config.max_position_size
        )

        return ATRSizingResult(
            position_size_pct=position_size_pct,
            atr_value=atr_value,
            price_volatility=price_volatility,
            vol_adjustment=vol_adjustment,
            target_met=target_met,
            confidence=confidence,
        )

    def _calculate_atr(self, ohlcv_data: list[OHLCV]) -> float:
        """Calculate Average True Range"""
        if len(ohlcv_data) < self.target_config.lookback_periods + 1:
            return 0.0

        # Use most recent data for ATR calculation
        recent_data = ohlcv_data[-self.target_config.lookback_periods - 1 :]

        true_ranges = []
        for i in range(1, len(recent_data)):
            current = recent_data[i]
            previous = recent_data[i - 1]

            # True Range = max(H-L, |H-C_prev|, |L-C_prev|)
            hl = float(current.high - current.low)
            hc = abs(float(current.high - previous.close))
            lc = abs(float(current.low - previous.close))

            true_range = max(hl, hc, lc)
            true_ranges.append(true_range)

        if not true_ranges:
            return 0.0

        # Calculate ATR as simple moving average of True Ranges
        atr = statistics.mean(true_ranges)
        return atr

    def _update_vol_regime(
        self, atr_value: float, price: float, timestamp: datetime
    ) -> None:
        """Update volatility regime tracking"""
        price_vol = atr_value / price
        self.atr_history.append((timestamp, price_vol))

        # Maintain lookback window
        cutoff_time = timestamp - timedelta(days=self.vol_lookback)
        self.atr_history = [
            (ts, vol) for ts, vol in self.atr_history if ts >= cutoff_time
        ]

    def _get_vol_regime_adjustment(self) -> float:
        """
        Calculate volatility regime adjustment

        If current volatility is higher than historical average,
        reduce position size and vice versa
        """
        if len(self.atr_history) < 10:
            return 1.0

        current_vol = self.atr_history[-1][1]
        historical_vols = [vol for _, vol in self.atr_history[:-1]]

        if not historical_vols:
            return 1.0

        avg_vol = statistics.mean(historical_vols)
        if avg_vol <= 0:
            return 1.0

        # Calculate volatility ratio
        vol_ratio = current_vol / avg_vol

        # Apply regime adjustment
        # Higher current vol → lower position size
        # Lower current vol → higher position size
        if vol_ratio > 1.0:
            # High vol regime - reduce size
            adjustment = 1.0 / (1.0 + (vol_ratio - 1.0) * self.regime_sensitivity)
        else:
            # Low vol regime - can increase size (but cap the increase)
            adjustment = min(2.0, 1.0 + (1.0 - vol_ratio) * self.regime_sensitivity)

        # Smooth the adjustment
        alpha = 0.1  # Smoothing factor
        self.vol_regime_multiplier = (
            alpha * adjustment + (1 - alpha) * self.vol_regime_multiplier
        )

        return self.vol_regime_multiplier

    def _calculate_confidence(
        self, ohlcv_data: list[OHLCV], atr_value: float, price_volatility: float
    ) -> float:
        """Calculate confidence in the position sizing"""

        # Data quality factor
        required_periods = self.target_config.lookback_periods
        available_periods = len(ohlcv_data)
        data_quality = min(1.0, available_periods / required_periods)

        # ATR stability factor (lower coefficient of variation = higher confidence)
        if len(ohlcv_data) >= required_periods:
            recent_data = ohlcv_data[-required_periods:]
            daily_ranges = []

            for candle in recent_data:
                daily_range = float(candle.high - candle.low)
                daily_ranges.append(daily_range)

            if len(daily_ranges) > 1:
                mean_range = statistics.mean(daily_ranges)
                std_range = statistics.stdev(daily_ranges)

                if mean_range > 0:
                    cv = std_range / mean_range  # Coefficient of variation
                    stability_factor = max(0.0, 1.0 - cv)
                else:
                    stability_factor = 0.0
            else:
                stability_factor = 0.5
        else:
            stability_factor = 0.5

        # Volatility reasonableness factor
        # Very high or very low volatility reduces confidence
        if 0.005 <= price_volatility <= 0.10:  # 0.5% to 10% daily vol
            vol_factor = 1.0
        elif price_volatility > 0.10:
            vol_factor = max(0.0, 1.0 - (price_volatility - 0.10) / 0.10)
        else:
            vol_factor = price_volatility / 0.005

        # Regime consistency factor
        regime_factor = min(1.0, 2.0 - abs(self.vol_regime_multiplier - 1.0))

        # Combine factors
        confidence = (
            data_quality * 0.3
            + stability_factor * 0.3
            + vol_factor * 0.2
            + regime_factor * 0.2
        )

        return min(1.0, confidence)

    def calculate_leverage(
        self,
        position_size_pct: float,
        available_margin: float,
        min_leverage: float = 1.0,
        max_leverage: float = 10.0,
    ) -> float:
        """
        Calculate optimal leverage for given position size

        Args:
            position_size_pct: Desired position size as % of portfolio
            available_margin: Available margin as % of portfolio
            min_leverage: Minimum allowed leverage
            max_leverage: Maximum allowed leverage

        Returns:
            Optimal leverage ratio
        """

        if available_margin <= 0 or position_size_pct <= 0:
            return min_leverage

        # Calculate required leverage
        required_leverage = position_size_pct / available_margin

        # Apply constraints
        leverage = max(min_leverage, min(max_leverage, required_leverage))

        return leverage

    def estimate_daily_portfolio_vol(
        self,
        position_size_pct: float,
        price_volatility: float,
        correlation: float = 1.0,
    ) -> float:
        """
        Estimate resulting daily portfolio volatility

        Args:
            position_size_pct: Position size as % of portfolio
            price_volatility: Daily price volatility of instrument
            correlation: Correlation with existing positions

        Returns:
            Estimated daily portfolio volatility
        """

        # Simple case: single position
        portfolio_vol = position_size_pct * price_volatility * abs(correlation)

        return portfolio_vol

    def optimize_for_sharpe(
        self,
        expected_return: float,
        price_volatility: float,
        risk_free_rate: float = 0.0,
    ) -> float:
        """
        Optimize position size for maximum Sharpe ratio

        Optimal Kelly fraction for maximizing log utility
        f* = (μ - r) / σ²

        Where:
        - μ = expected return
        - r = risk-free rate
        - σ = volatility
        """

        if price_volatility <= 0:
            return 0.0

        excess_return = expected_return - risk_free_rate
        optimal_fraction = excess_return / (price_volatility**2)

        # Apply constraints
        optimal_fraction = max(
            self.target_config.min_position_size,
            min(self.target_config.max_position_size, optimal_fraction),
        )

        return optimal_fraction

    def get_sizing_metrics(self) -> dict[str, Any]:
        """Get comprehensive sizing metrics"""

        current_vol = self.atr_history[-1][1] if self.atr_history else 0.0

        # Calculate volatility statistics
        vol_stats = {}
        if len(self.atr_history) >= 10:
            vols = [vol for _, vol in self.atr_history]
            vol_stats = {
                "current_vol": current_vol,
                "avg_vol": statistics.mean(vols),
                "vol_std": statistics.stdev(vols),
                "vol_min": min(vols),
                "vol_max": max(vols),
                "vol_percentile_50": sorted(vols)[len(vols) // 2],
                "vol_percentile_95": sorted(vols)[int(len(vols) * 0.95)],
            }

        return {
            "target_daily_vol": self.target_config.target_daily_vol,
            "vol_regime_multiplier": self.vol_regime_multiplier,
            "min_position_size": self.target_config.min_position_size,
            "max_position_size": self.target_config.max_position_size,
            "vol_history_length": len(self.atr_history),
            "confidence_threshold": self.confidence_threshold,
            **vol_stats,
        }

    def reset_regime_tracking(self) -> None:
        """Reset volatility regime tracking"""
        self.atr_history.clear()
        self.vol_regime_multiplier = 1.0
