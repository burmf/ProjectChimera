"""
Kelly Criterion calculator for optimal position sizing
Implements fractional Kelly with win rate estimation and risk adjustment
"""

import math
import statistics
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np


@dataclass
class KellyResult:
    """Kelly calculation result"""
    fraction: float  # Kelly fraction (0.0 to 1.0)
    win_rate: float  # Estimated win rate
    avg_win: float   # Average winning return
    avg_loss: float  # Average losing return (positive value)
    edge: float      # Expected return per trade
    confidence: float  # Confidence in the calculation (0.0 to 1.0)
    sample_size: int  # Number of trades used
    
    def is_valid(self) -> bool:
        """Check if Kelly result is valid for trading"""
        return (
            0.0 <= self.fraction <= 1.0 and
            0.0 <= self.win_rate <= 1.0 and
            self.avg_win > 0 and
            self.avg_loss > 0 and
            self.sample_size >= 10 and
            self.confidence > 0.3
        )


class KellyCalculator:
    """
    Dynamic Kelly Criterion calculator
    
    Features:
    - Rolling window Kelly calculation
    - Win rate and return distribution estimation
    - Fractional Kelly with safety multiplier
    - Confidence scoring based on sample size and stability
    - Outlier filtering for robust estimates
    """
    
    def __init__(
        self,
        lookback_trades: int = 100,
        min_trades: int = 20,
        kelly_fraction: float = 0.5,  # Use 50% of full Kelly by default
        confidence_threshold: float = 0.4,
        outlier_threshold: float = 3.0,  # Standard deviations
        decay_factor: float = 0.95  # Exponential weighting
    ):
        self.lookback_trades = lookback_trades
        self.min_trades = min_trades
        self.kelly_fraction = kelly_fraction
        self.confidence_threshold = confidence_threshold
        self.outlier_threshold = outlier_threshold
        self.decay_factor = decay_factor
        
        # Historical data
        self.trade_returns: List[float] = []
        self.trade_timestamps: List[datetime] = []
    
    def add_trade_result(self, return_pct: float, timestamp: Optional[datetime] = None) -> None:
        """Add a trade result to the history"""
        if timestamp is None:
            timestamp = datetime.now()
        
        self.trade_returns.append(return_pct)
        self.trade_timestamps.append(timestamp)
        
        # Maintain rolling window
        if len(self.trade_returns) > self.lookback_trades:
            self.trade_returns = self.trade_returns[-self.lookback_trades:]
            self.trade_timestamps = self.trade_timestamps[-self.lookback_trades:]
    
    def calculate_kelly(self, returns: Optional[List[float]] = None) -> KellyResult:
        """
        Calculate Kelly fraction from trade returns
        
        Kelly formula: f = (bp - q) / b
        where:
        - f = fraction of capital to bet
        - b = odds (avg_win / avg_loss)
        - p = probability of winning
        - q = probability of losing (1 - p)
        """
        
        if returns is None:
            returns = self.trade_returns
        
        if len(returns) < self.min_trades:
            return KellyResult(
                fraction=0.0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                edge=0.0,
                confidence=0.0,
                sample_size=len(returns)
            )
        
        # Filter outliers for more robust estimates
        filtered_returns = self._filter_outliers(returns)
        
        # Separate wins and losses
        wins = [r for r in filtered_returns if r > 0]
        losses = [abs(r) for r in filtered_returns if r < 0]
        
        if not wins or not losses:
            # Need both wins and losses for Kelly calculation
            return KellyResult(
                fraction=0.0,
                win_rate=len(wins) / len(filtered_returns) if filtered_returns else 0.0,
                avg_win=statistics.mean(wins) if wins else 0.0,
                avg_loss=statistics.mean(losses) if losses else 0.0,
                edge=statistics.mean(filtered_returns) if filtered_returns else 0.0,
                confidence=0.0,
                sample_size=len(filtered_returns)
            )
        
        # Calculate statistics
        win_rate = len(wins) / len(filtered_returns)
        avg_win = statistics.mean(wins)
        avg_loss = statistics.mean(losses)
        edge = statistics.mean(filtered_returns)
        
        # Kelly formula calculation
        if avg_loss == 0:
            kelly_fraction = 0.0
        else:
            odds_ratio = avg_win / avg_loss  # b in Kelly formula
            kelly_fraction = (win_rate * odds_ratio - (1 - win_rate)) / odds_ratio
        
        # Ensure Kelly fraction is non-negative
        kelly_fraction = max(0.0, kelly_fraction)
        
        # Apply fractional Kelly multiplier
        adjusted_fraction = kelly_fraction * self.kelly_fraction
        
        # Cap at 100% (though this should be rare with fractional Kelly)
        adjusted_fraction = min(1.0, adjusted_fraction)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(filtered_returns, kelly_fraction)
        
        return KellyResult(
            fraction=adjusted_fraction,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            edge=edge,
            confidence=confidence,
            sample_size=len(filtered_returns)
        )
    
    def calculate_exponentially_weighted_kelly(self) -> KellyResult:
        """Calculate Kelly with exponentially weighted returns (recent trades more important)"""
        if len(self.trade_returns) < self.min_trades:
            return self.calculate_kelly()
        
        # Apply exponential weights
        weights = []
        for i in range(len(self.trade_returns)):
            weight = self.decay_factor ** (len(self.trade_returns) - 1 - i)
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        # Calculate weighted statistics
        returns = self.trade_returns
        weighted_returns = []
        
        for i, ret in enumerate(returns):
            # Replicate return based on weight (approximate)
            count = max(1, int(weights[i] * len(returns)))
            weighted_returns.extend([ret] * count)
        
        return self.calculate_kelly(weighted_returns)
    
    def _filter_outliers(self, returns: List[float]) -> List[float]:
        """Remove outliers using z-score method"""
        if len(returns) < 10:
            return returns
        
        mean = statistics.mean(returns)
        stdev = statistics.stdev(returns)
        
        if stdev == 0:
            return returns
        
        filtered = []
        for ret in returns:
            z_score = abs(ret - mean) / stdev
            if z_score <= self.outlier_threshold:
                filtered.append(ret)
        
        # Ensure we don't filter too aggressively
        if len(filtered) < len(returns) * 0.8:
            return returns
        
        return filtered
    
    def _calculate_confidence(self, returns: List[float], kelly_fraction: float) -> float:
        """
        Calculate confidence in Kelly estimate
        Based on sample size, stability, and edge significance
        """
        n = len(returns)
        
        # Sample size factor (more data = higher confidence)
        size_factor = min(1.0, n / 100.0)
        
        # Stability factor (consistent returns = higher confidence)
        if n >= 10:
            stdev = statistics.stdev(returns)
            mean_abs = statistics.mean([abs(r) for r in returns])
            if mean_abs > 0:
                cv = stdev / mean_abs  # Coefficient of variation
                stability_factor = max(0.0, 1.0 - cv)
            else:
                stability_factor = 0.0
        else:
            stability_factor = 0.0
        
        # Edge significance (significant edge = higher confidence)
        if n >= 10:
            mean_return = statistics.mean(returns)
            stdev_return = statistics.stdev(returns)
            if stdev_return > 0:
                t_stat = abs(mean_return) / (stdev_return / math.sqrt(n))
                # Simple significance test (t > 2 is roughly p < 0.05)
                edge_factor = min(1.0, t_stat / 2.0)
            else:
                edge_factor = 0.0
        else:
            edge_factor = 0.0
        
        # Kelly reasonableness (reasonable Kelly values = higher confidence)
        if 0.01 <= kelly_fraction <= 0.5:
            kelly_factor = 1.0
        elif kelly_fraction > 0.5:
            kelly_factor = max(0.0, 2.0 - kelly_fraction)  # Penalize high Kelly
        else:
            kelly_factor = kelly_fraction / 0.01  # Scale up small Kelly
        
        # Combine factors
        confidence = (size_factor * 0.3 + 
                     stability_factor * 0.3 + 
                     edge_factor * 0.3 + 
                     kelly_factor * 0.1)
        
        return min(1.0, confidence)
    
    def get_required_sample_size(self, target_confidence: float = 0.8) -> int:
        """Estimate required sample size for target confidence"""
        current_result = self.calculate_kelly()
        
        if current_result.confidence >= target_confidence:
            return len(self.trade_returns)
        
        # Rough estimate: confidence scales with sqrt(sample_size)
        current_n = len(self.trade_returns)
        if current_n == 0 or current_result.confidence == 0:
            return 100  # Default estimate
        
        # Scale factor based on confidence ratio
        ratio = target_confidence / current_result.confidence
        required_n = int(current_n * ratio ** 2)
        
        return min(500, max(self.min_trades, required_n))
    
    def simulate_kelly_performance(
        self, 
        kelly_fraction: float,
        num_trades: int = 1000,
        returns: Optional[List[float]] = None
    ) -> Tuple[float, float, float]:
        """
        Simulate Kelly strategy performance
        
        Returns:
            (final_equity, max_drawdown, sharpe_ratio)
        """
        if returns is None:
            returns = self.trade_returns
        
        if len(returns) < 10:
            return 1.0, 0.0, 0.0
        
        # Bootstrap sampling for simulation
        equity = 1.0
        equity_curve = [equity]
        peak_equity = equity
        max_dd = 0.0
        
        for _ in range(num_trades):
            # Sample return from historical distribution
            trade_return = np.random.choice(returns)
            
            # Apply Kelly sizing
            position_return = trade_return * kelly_fraction
            
            # Update equity (assuming returns are percentage)
            equity *= (1 + position_return)
            equity_curve.append(equity)
            
            # Track drawdown
            if equity > peak_equity:
                peak_equity = equity
            else:
                drawdown = (peak_equity - equity) / peak_equity
                max_dd = max(max_dd, drawdown)
        
        # Calculate Sharpe ratio
        if len(equity_curve) > 1:
            returns_series = [equity_curve[i+1]/equity_curve[i] - 1 
                            for i in range(len(equity_curve)-1)]
            mean_return = statistics.mean(returns_series)
            std_return = statistics.stdev(returns_series) if len(returns_series) > 1 else 0
            sharpe = (mean_return / std_return) if std_return > 0 else 0
        else:
            sharpe = 0.0
        
        return equity, max_dd, sharpe
    
    def clear_history(self) -> None:
        """Clear all historical trade data"""
        self.trade_returns.clear()
        self.trade_timestamps.clear()
    
    def get_statistics(self) -> dict:
        """Get summary statistics of trade history"""
        if not self.trade_returns:
            return {}
        
        returns = self.trade_returns
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        
        return {
            'total_trades': len(returns),
            'win_rate': len(wins) / len(returns),
            'avg_return': statistics.mean(returns),
            'std_return': statistics.stdev(returns) if len(returns) > 1 else 0,
            'avg_win': statistics.mean(wins) if wins else 0,
            'avg_loss': statistics.mean(losses) if losses else 0,
            'best_trade': max(returns),
            'worst_trade': min(returns),
            'total_return': sum(returns),
            'sharpe_estimate': (statistics.mean(returns) / statistics.stdev(returns) 
                              if len(returns) > 1 and statistics.stdev(returns) > 0 else 0)
        }