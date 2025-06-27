"""
Dynamic Kelly Criterion with EWMA win-rate and ½-Kelly safety
Enhanced version with real-time adaptation and volatility filtering
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DynamicKellyConfig:
    """Configuration for dynamic Kelly sizing"""

    base_kelly_fraction: float = 0.5  # ½-Kelly by default
    ewma_alpha: float = 0.1  # EWMA decay for win rate
    min_sample_size: int = 20  # Minimum trades for calculation
    lookback_window: int = 100  # Rolling window size
    vol_adjustment: bool = True  # Adjust for volatility regime
    confidence_threshold: float = 0.4  # Minimum confidence to trade
    max_kelly_fraction: float = 0.75  # Maximum Kelly allowed
    outlier_z_threshold: float = 2.5  # Z-score for outlier detection


@dataclass
class DynamicKellyResult:
    """Result from dynamic Kelly calculation"""

    kelly_fraction: float  # Final Kelly fraction to use
    ewma_win_rate: float  # EWMA-weighted win rate
    ewma_avg_win: float  # EWMA-weighted average win
    ewma_avg_loss: float  # EWMA-weighted average loss
    raw_kelly: float  # Raw Kelly before adjustments
    vol_adjustment_factor: float  # Volatility adjustment multiplier
    confidence_score: float  # Confidence in estimate (0-1)
    sample_size: int  # Number of trades used
    last_updated: datetime  # When calculation was performed

    def is_valid(self) -> bool:
        """Check if result is valid for trading"""
        return (
            0.0 <= self.kelly_fraction <= 1.0
            and 0.0 <= self.ewma_win_rate <= 1.0
            and self.confidence_score >= 0.3
            and self.sample_size >= 10
        )


class DynamicKellyCalculator:
    """
    Advanced Kelly Criterion calculator with dynamic adaptation

    Features:
    - EWMA-weighted win rate and return estimation
    - Volatility regime adjustment
    - Real-time adaptation to market conditions
    - Outlier detection and filtering
    - Confidence scoring based on statistical significance
    - Half-Kelly default with adaptive scaling
    """

    def __init__(self, config: DynamicKellyConfig | None = None):
        self.config = config or DynamicKellyConfig()

        # Trade history storage
        self.trade_returns: deque = deque(maxlen=self.config.lookback_window)
        self.trade_timestamps: deque = deque(maxlen=self.config.lookback_window)

        # EWMA state variables
        self.ewma_win_rate: float = 0.5  # Start neutral
        self.ewma_avg_win: float = 0.0
        self.ewma_avg_loss: float = 0.0
        self.ewma_return_variance: float = 0.0

        # Volatility regime tracking
        self.vol_history: deque = deque(maxlen=50)
        self.vol_ewma: float = 0.0

        # Performance tracking
        self.last_calculation: DynamicKellyResult | None = None
        self.calculation_count: int = 0

    def add_trade_result(
        self, return_pct: float, timestamp: datetime | None = None
    ) -> None:
        """Add a new trade result and update EWMA estimates"""
        if timestamp is None:
            timestamp = datetime.now()

        # Store trade data
        self.trade_returns.append(return_pct)
        self.trade_timestamps.append(timestamp)

        # Update volatility tracking
        abs_return = abs(return_pct)
        self.vol_history.append(abs_return)
        if self.vol_ewma == 0.0:
            self.vol_ewma = abs_return
        else:
            self.vol_ewma = (
                self.config.ewma_alpha * abs_return
                + (1 - self.config.ewma_alpha) * self.vol_ewma
            )

        # Update EWMA estimates
        self._update_ewma_estimates(return_pct)

        logger.debug(
            f"Added trade: {return_pct:.4f}%, EWMA win rate: {self.ewma_win_rate:.3f}"
        )

    def _update_ewma_estimates(self, return_pct: float) -> None:
        """Update EWMA estimates with new trade result"""
        alpha = self.config.ewma_alpha

        # Update win rate
        is_win = 1.0 if return_pct > 0 else 0.0
        if len(self.trade_returns) == 1:  # First trade
            self.ewma_win_rate = is_win
        else:
            self.ewma_win_rate = alpha * is_win + (1 - alpha) * self.ewma_win_rate

        # Update average win/loss
        if return_pct > 0:  # Winning trade
            if self.ewma_avg_win == 0.0:
                self.ewma_avg_win = return_pct
            else:
                self.ewma_avg_win = alpha * return_pct + (1 - alpha) * self.ewma_avg_win
        elif return_pct < 0:  # Losing trade
            abs_loss = abs(return_pct)
            if self.ewma_avg_loss == 0.0:
                self.ewma_avg_loss = abs_loss
            else:
                self.ewma_avg_loss = alpha * abs_loss + (1 - alpha) * self.ewma_avg_loss

        # Update return variance for confidence calculation
        if len(self.trade_returns) > 1:
            return_deviation = return_pct - np.mean(list(self.trade_returns))
            variance_update = return_deviation**2
            if self.ewma_return_variance == 0.0:
                self.ewma_return_variance = variance_update
            else:
                self.ewma_return_variance = (
                    alpha * variance_update + (1 - alpha) * self.ewma_return_variance
                )

    def calculate_dynamic_kelly(self) -> DynamicKellyResult:
        """Calculate dynamic Kelly fraction with all adjustments"""

        if len(self.trade_returns) < self.config.min_sample_size:
            return self._get_default_result(
                f"Insufficient data: {len(self.trade_returns)} < {self.config.min_sample_size}"
            )

        # Filter outliers
        clean_returns = self._filter_outliers(list(self.trade_returns))

        if len(clean_returns) < self.config.min_sample_size:
            return self._get_default_result("Too many outliers filtered")

        # Calculate raw Kelly using EWMA estimates
        raw_kelly = self._calculate_raw_kelly()

        if raw_kelly <= 0:
            return self._get_default_result("Negative edge detected")

        # Apply base Kelly fraction (½-Kelly)
        adjusted_kelly = raw_kelly * self.config.base_kelly_fraction

        # Apply volatility adjustment
        vol_adjustment = self._calculate_volatility_adjustment()
        adjusted_kelly *= vol_adjustment

        # Cap at maximum allowed
        adjusted_kelly = min(adjusted_kelly, self.config.max_kelly_fraction)

        # Calculate confidence score
        confidence = self._calculate_confidence_score(clean_returns)

        # Create result
        result = DynamicKellyResult(
            kelly_fraction=adjusted_kelly,
            ewma_win_rate=self.ewma_win_rate,
            ewma_avg_win=self.ewma_avg_win,
            ewma_avg_loss=self.ewma_avg_loss,
            raw_kelly=raw_kelly,
            vol_adjustment_factor=vol_adjustment,
            confidence_score=confidence,
            sample_size=len(clean_returns),
            last_updated=datetime.now(),
        )

        self.last_calculation = result
        self.calculation_count += 1

        logger.info(
            f"Dynamic Kelly: {adjusted_kelly:.3f} (raw: {raw_kelly:.3f}, vol_adj: {vol_adjustment:.3f}, conf: {confidence:.3f})"
        )

        return result

    def _calculate_raw_kelly(self) -> float:
        """Calculate raw Kelly fraction using EWMA estimates"""
        if self.ewma_avg_loss <= 0 or self.ewma_win_rate <= 0:
            return 0.0

        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
        odds_ratio = self.ewma_avg_win / self.ewma_avg_loss
        win_rate = self.ewma_win_rate
        lose_rate = 1.0 - win_rate

        kelly_fraction = (win_rate * odds_ratio - lose_rate) / odds_ratio

        return max(0.0, kelly_fraction)

    def _calculate_volatility_adjustment(self) -> float:
        """Calculate volatility regime adjustment factor"""
        if not self.config.vol_adjustment or len(self.vol_history) < 10:
            return 1.0

        # Calculate current vs historical volatility
        recent_vol = np.mean(list(self.vol_history)[-10:])  # Last 10 trades
        historical_vol = np.mean(list(self.vol_history))

        if historical_vol <= 0:
            return 1.0

        vol_ratio = recent_vol / historical_vol

        # Reduce Kelly in high vol regimes, increase in low vol
        if vol_ratio > 1.5:  # High volatility regime
            adjustment = 0.7  # Reduce position size
        elif vol_ratio > 1.2:
            adjustment = 0.85
        elif vol_ratio < 0.7:  # Low volatility regime
            adjustment = 1.15  # Slightly increase position size
        elif vol_ratio < 0.8:
            adjustment = 1.05
        else:
            adjustment = 1.0  # Normal regime

        return adjustment

    def _calculate_confidence_score(self, returns: list[float]) -> float:
        """Calculate confidence score based on statistical significance"""
        n = len(returns)

        # Sample size component
        size_score = min(1.0, n / 100.0)

        # Statistical significance of edge
        if n >= 10 and self.ewma_return_variance > 0:
            mean_return = np.mean(returns)
            std_error = np.sqrt(self.ewma_return_variance / n)
            t_stat = abs(mean_return / std_error) if std_error > 0 else 0
            significance_score = min(1.0, t_stat / 2.0)  # t > 2 is roughly p < 0.05
        else:
            significance_score = 0.0

        # Stability component (consistent performance)
        if n >= 20:
            recent_returns = returns[-10:]
            older_returns = returns[-20:-10]

            recent_mean = np.mean(recent_returns)
            older_mean = np.mean(older_returns)

            # Check if recent performance is consistent with older
            consistency = 1.0 - min(
                1.0, abs(recent_mean - older_mean) / (abs(older_mean) + 0.01)
            )
            stability_score = max(0.0, consistency)
        else:
            stability_score = 0.5

        # Win rate confidence (closer to extremes = less confident)
        wr_distance_from_center = abs(self.ewma_win_rate - 0.5)
        winrate_score = min(
            1.0, wr_distance_from_center * 4.0
        )  # Peak confidence at 62.5% or 37.5% win rate

        # Combine components
        confidence = (
            size_score * 0.3
            + significance_score * 0.4
            + stability_score * 0.2
            + winrate_score * 0.1
        )

        return min(1.0, confidence)

    def _filter_outliers(self, returns: list[float]) -> list[float]:
        """Filter outliers using z-score method"""
        if len(returns) < 10:
            return returns

        returns_array = np.array(returns)
        z_scores = np.abs(
            (returns_array - np.mean(returns_array)) / np.std(returns_array)
        )

        # Keep returns within z-score threshold
        filtered_indices = z_scores <= self.config.outlier_z_threshold
        filtered_returns = returns_array[filtered_indices].tolist()

        # Ensure we don't filter too aggressively
        if len(filtered_returns) < len(returns) * 0.75:
            return returns  # Too many outliers, keep original data

        outliers_removed = len(returns) - len(filtered_returns)
        if outliers_removed > 0:
            logger.debug(
                f"Filtered {outliers_removed} outliers from {len(returns)} trades"
            )

        return filtered_returns

    def _get_default_result(self, reason: str) -> DynamicKellyResult:
        """Get default result when calculation cannot be performed"""
        logger.warning(f"Using default Kelly result: {reason}")

        return DynamicKellyResult(
            kelly_fraction=0.0,
            ewma_win_rate=self.ewma_win_rate,
            ewma_avg_win=self.ewma_avg_win,
            ewma_avg_loss=self.ewma_avg_loss,
            raw_kelly=0.0,
            vol_adjustment_factor=1.0,
            confidence_score=0.0,
            sample_size=len(self.trade_returns),
            last_updated=datetime.now(),
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive statistics"""
        returns = list(self.trade_returns)

        if not returns:
            return {"error": "No trade data available"}

        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]

        stats = {
            "total_trades": len(returns),
            "win_rate_simple": len(wins) / len(returns) if returns else 0,
            "win_rate_ewma": self.ewma_win_rate,
            "avg_win_simple": np.mean(wins) if wins else 0,
            "avg_win_ewma": self.ewma_avg_win,
            "avg_loss_simple": np.mean([abs(l) for l in losses]) if losses else 0,
            "avg_loss_ewma": self.ewma_avg_loss,
            "total_return": sum(returns),
            "volatility": np.std(returns) if len(returns) > 1 else 0,
            "vol_ewma": self.vol_ewma,
            "sharpe_ratio": (
                np.mean(returns) / np.std(returns)
                if len(returns) > 1 and np.std(returns) > 0
                else 0
            ),
            "calculation_count": self.calculation_count,
        }

        if self.last_calculation:
            stats["last_kelly_fraction"] = self.last_calculation.kelly_fraction
            stats["last_confidence"] = self.last_calculation.confidence_score

        return stats

    def reset(self) -> None:
        """Reset all state for fresh start"""
        self.trade_returns.clear()
        self.trade_timestamps.clear()
        self.vol_history.clear()

        self.ewma_win_rate = 0.5
        self.ewma_avg_win = 0.0
        self.ewma_avg_loss = 0.0
        self.ewma_return_variance = 0.0
        self.vol_ewma = 0.0

        self.last_calculation = None
        self.calculation_count = 0

        logger.info("Dynamic Kelly calculator reset")

    def simulate_performance(self, num_simulations: int = 1000) -> dict[str, float]:
        """Simulate Kelly strategy performance using historical data"""
        if len(self.trade_returns) < 20:
            return {"error": "Insufficient data for simulation"}

        returns = list(self.trade_returns)
        kelly_result = self.calculate_dynamic_kelly()

        if not kelly_result.is_valid():
            return {"error": "Invalid Kelly calculation"}

        kelly_fraction = kelly_result.kelly_fraction

        # Monte Carlo simulation
        final_equities = []
        max_drawdowns = []

        for _ in range(num_simulations):
            equity = 1.0
            peak_equity = 1.0
            max_dd = 0.0

            # Simulate trades by sampling from historical returns
            simulation_trades = np.random.choice(returns, size=min(100, len(returns)))

            for trade_return in simulation_trades:
                # Apply Kelly sizing
                portfolio_return = trade_return * kelly_fraction
                equity *= 1 + portfolio_return

                # Track drawdown
                if equity > peak_equity:
                    peak_equity = equity
                else:
                    drawdown = (peak_equity - equity) / peak_equity
                    max_dd = max(max_dd, drawdown)

            final_equities.append(equity)
            max_drawdowns.append(max_dd)

        return {
            "final_equity_mean": np.mean(final_equities),
            "final_equity_std": np.std(final_equities),
            "final_equity_min": np.min(final_equities),
            "final_equity_max": np.max(final_equities),
            "max_drawdown_mean": np.mean(max_drawdowns),
            "max_drawdown_worst": np.max(max_drawdowns),
            "probability_of_loss": sum(1 for eq in final_equities if eq < 1.0)
            / len(final_equities),
        }
