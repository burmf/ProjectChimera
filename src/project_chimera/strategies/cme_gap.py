"""
CME Gap Trading Strategy (CME_GAP)
Exploits weekend futures gaps: Weekend futures gap fill → contrarian position
"""

import statistics
from datetime import datetime, timedelta, timezone
from typing import Any

from ..domains.market import MarketFrame, Signal, SignalType
from .base import Strategy, StrategyConfig


class CMEGapStrategy(Strategy):
    """
    CME Gap Trading Strategy

    Core trigger: Weekend futures gap fill → contrarian position
    Exploits CME futures gaps that tend to fill during the following week
    """

    def __init__(self, config: StrategyConfig):
        super().__init__(config)

    def validate_config(self) -> None:
        """Validate strategy configuration"""
        # Set default parameters
        self.params.setdefault("min_gap_pct", 1.0)  # Minimum gap percentage
        self.params.setdefault("max_gap_pct", 10.0)  # Maximum gap percentage
        self.params.setdefault(
            "gap_fill_timeframe", 168
        )  # Hours to expect gap fill (1 week)
        self.params.setdefault("entry_delay_hours", 4)  # Wait hours after market open
        self.params.setdefault("stop_loss_pct", 2.0)  # Stop loss %
        self.params.setdefault("take_profit_pct", 1.5)  # Take profit % (conservative)
        self.params.setdefault("max_position_hours", 72)  # Max hold time
        self.params.setdefault("volume_confirmation", True)  # Require volume
        self.params.setdefault("min_volume_ratio", 1.2)  # Min volume vs average

        # Validate ranges
        if self.params["min_gap_pct"] <= 0:
            raise ValueError("min_gap_pct must be positive")
        if self.params["max_gap_pct"] <= self.params["min_gap_pct"]:
            raise ValueError("max_gap_pct must be > min_gap_pct")
        if self.params["gap_fill_timeframe"] <= 0:
            raise ValueError("gap_fill_timeframe must be positive")
        if self.params["entry_delay_hours"] < 0:
            raise ValueError("entry_delay_hours must be non-negative")

    def get_required_data(self) -> dict[str, Any]:
        """Specify required market data"""
        return {
            "ohlcv_timeframes": ["1h"],
            "orderbook_levels": 0,
            "indicators": [],
            "cme_futures_data": True,  # Need CME futures prices
            "lookback_periods": 200,  # Need enough data to detect gaps
        }

    async def generate_signal(self, market_data: MarketFrame) -> Signal | None:
        """Generate CME gap fill signal"""
        if (
            not market_data.ohlcv_1h or len(market_data.ohlcv_1h) < 72
        ):  # Need 3 days of data
            return None

        current_time = market_data.timestamp
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        else:
            current_time = current_time.astimezone(timezone.utc)

        # Check if we're in the right time window (Monday-Thursday)
        weekday = current_time.weekday()  # 0=Monday, 6=Sunday
        if weekday > 3:  # Friday, Saturday, Sunday
            return None

        # Detect recent gap
        gap_info = self._detect_cme_gap(market_data.ohlcv_1h)
        if gap_info is None:
            return None

        # Check if gap is within acceptable range
        gap_pct = abs(gap_info["gap_percentage"])
        if gap_pct < self.params["min_gap_pct"] or gap_pct > self.params["max_gap_pct"]:
            return None

        # Check if enough time has passed since market open
        market_open_time = self._get_last_market_open(current_time)
        hours_since_open = (current_time - market_open_time).total_seconds() / 3600
        if hours_since_open < self.params["entry_delay_hours"]:
            return None

        # Check if gap is still unfilled
        current_price = float(market_data.current_price)
        if self._is_gap_filled(gap_info, current_price):
            return None

        # Volume confirmation
        if self.params["volume_confirmation"]:
            volume_ok = self._check_volume_confirmation(market_data.ohlcv_1h)
            if not volume_ok:
                return None

        # Determine signal direction (towards gap fill)
        gap_direction = gap_info["direction"]
        gap_level = gap_info["gap_level"]

        if gap_direction == "up" and current_price > gap_level:
            # Gap up, expect fill down
            signal_type = SignalType.SELL
            reasoning = f"CME gap fill SHORT: {gap_pct:.1f}% gap up at {gap_level:.2f}, expect fill"
            stop_loss = current_price * (1 + self.params["stop_loss_pct"] / 100)
            take_profit = gap_level  # Target the gap level
        elif gap_direction == "down" and current_price < gap_level:
            # Gap down, expect fill up
            signal_type = SignalType.BUY
            reasoning = f"CME gap fill LONG: {gap_pct:.1f}% gap down at {gap_level:.2f}, expect fill"
            stop_loss = current_price * (1 - self.params["stop_loss_pct"] / 100)
            take_profit = gap_level  # Target the gap level
        else:
            return None

        # Calculate confidence based on gap size and time elapsed
        gap_age_hours = (current_time - gap_info["gap_time"]).total_seconds() / 3600
        time_factor = min(1.0, gap_age_hours / 24)  # Higher confidence with more time
        size_factor = min(1.0, gap_pct / 5.0)  # Higher confidence with larger gaps
        confidence = min(0.75, 0.5 + time_factor * 0.15 + size_factor * 0.1)

        return Signal(
            symbol=market_data.symbol,
            signal_type=signal_type,
            confidence=confidence,
            target_size=0.03,  # 3% position size
            entry_price=market_data.current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timeframe="1h",
            strategy_id="CME_GAP",
            reasoning=reasoning,
            metadata={
                "gap_percentage": gap_pct,
                "gap_direction": gap_direction,
                "gap_level": gap_level,
                "gap_age_hours": gap_age_hours,
                "market_open_hours_ago": hours_since_open,
                "pattern_type": "cme_gap_fill",
            },
            timestamp=market_data.timestamp,
        )

    def _detect_cme_gap(self, candles: list) -> dict[str, Any] | None:
        """Detect CME futures gap from weekend"""
        if len(candles) < 72:  # Need at least 3 days
            return None

        # Find the most recent weekend gap (Friday close to Monday open)
        for i in range(len(candles) - 1, 48, -1):  # Look back up to 48 hours
            current_candle = candles[i]
            prev_candle = candles[i - 1]

            current_time = current_candle.timestamp
            if hasattr(current_time, "weekday"):
                weekday = current_time.weekday()
            else:
                # Assume it's a datetime string, convert it
                if isinstance(current_time, str):
                    current_time = datetime.fromisoformat(
                        current_time.replace("Z", "+00:00")
                    )
                weekday = current_time.weekday()

            # Check if this is a Monday opening (gap detection)
            if weekday == 0:  # Monday
                friday_close = float(prev_candle.close)
                monday_open = float(current_candle.open)

                if friday_close == 0:
                    continue

                gap_pct = ((monday_open - friday_close) / friday_close) * 100

                if abs(gap_pct) >= self.params["min_gap_pct"]:
                    return {
                        "gap_percentage": gap_pct,
                        "direction": "up" if gap_pct > 0 else "down",
                        "gap_level": friday_close,  # Level to fill
                        "friday_close": friday_close,
                        "monday_open": monday_open,
                        "gap_time": current_time,
                        "candle_index": i,
                    }

        return None

    def _get_last_market_open(self, current_time: datetime) -> datetime:
        """Get the last market open time (Monday 00:00 UTC for crypto)"""
        weekday = current_time.weekday()

        if weekday == 0:  # Monday
            return current_time.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            # Calculate last Monday
            days_since_monday = weekday
            last_monday = current_time - timedelta(days=days_since_monday)
            return last_monday.replace(hour=0, minute=0, second=0, microsecond=0)

    def _is_gap_filled(self, gap_info: dict[str, Any], current_price: float) -> bool:
        """Check if the gap has been filled"""
        gap_level = gap_info["gap_level"]
        gap_direction = gap_info["direction"]

        if gap_direction == "up":
            # Gap up filled when price goes back down to or below Friday close
            return current_price <= gap_level
        else:
            # Gap down filled when price goes back up to or above Friday close
            return current_price >= gap_level

    def _check_volume_confirmation(self, candles: list) -> bool:
        """Check for adequate volume to support gap fill"""
        if len(candles) < 20:
            return True  # Assume OK if not enough data

        current_volume = float(candles[-1].volume)
        recent_volumes = [float(c.volume) for c in candles[-20:-1]]

        if not recent_volumes or current_volume == 0:
            return True

        avg_volume = statistics.mean(recent_volumes)
        if avg_volume == 0:
            return True

        volume_ratio = current_volume / avg_volume
        return volume_ratio >= self.params["min_volume_ratio"]

    def _calculate_gap_statistics(self, candles: list) -> dict[str, float]:
        """Calculate historical gap fill statistics"""
        gaps_found = 0
        gaps_filled = 0
        avg_fill_time = 0

        # This would analyze historical data to determine gap fill probability
        # For now, return default statistics
        return {
            "fill_probability": 0.7,  # 70% of gaps tend to fill
            "avg_fill_hours": 48,  # Average 48 hours to fill
            "max_fill_days": 7,  # Most fill within a week
        }

    def _get_cme_futures_price(self, symbol: str) -> float | None:
        """Get current CME futures price for comparison"""
        # This would fetch actual CME futures data
        # For now, return None to indicate unavailable
        return None


def create_cme_gap_strategy(config: dict[str, Any] | None = None) -> CMEGapStrategy:
    """Factory function to create CMEGapStrategy"""
    if config is None:
        config = {}

    strategy_config = StrategyConfig(name="cme_gap", enabled=True, params=config)

    return CMEGapStrategy(strategy_config)
