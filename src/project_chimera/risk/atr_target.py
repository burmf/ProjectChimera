"""
ATR-Target Volatility Control
Implements precise daily volatility targeting using Average True Range
Target: 1% daily portfolio volatility with dynamic adjustments
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from collections import deque
import logging

from ..domains.market import OHLCV

logger = logging.getLogger(__name__)


@dataclass
class ATRTargetConfig:
    """Configuration for ATR target volatility control"""
    target_daily_vol: float = 0.01          # Target daily portfolio volatility (1%)
    atr_periods: int = 14                   # ATR calculation periods
    vol_lookback_days: int = 30             # Days to look back for vol estimation
    min_position_size: float = 0.01         # Minimum position size (1%)
    max_position_size: float = 0.20         # Maximum position size (20%)
    vol_floor: float = 0.0001               # Minimum volatility to prevent division by zero
    vol_ceiling: float = 0.10               # Maximum volatility cap (10%)
    regime_sensitivity: float = 0.7          # Volatility regime change sensitivity
    confidence_threshold: float = 0.5       # Minimum confidence to use calculation
    smoothing_alpha: float = 0.3            # EMA smoothing for vol estimates


@dataclass
class ATRTargetResult:
    """Result from ATR target volatility calculation"""
    position_size_pct: float              # Recommended position size (% of portfolio)
    current_atr: float                     # Current ATR value
    daily_vol_estimate: float             # Estimated daily volatility
    vol_target_ratio: float               # Current vol / target vol ratio
    regime_adjustment: float               # Volatility regime adjustment factor
    confidence_score: float               # Confidence in the calculation (0-1)
    target_met: bool                       # Whether target can be realistically met
    last_updated: datetime                 # When calculation was performed
    price_level: float                     # Price level used for calculation
    
    def is_valid(self) -> bool:
        """Check if result is valid for trading"""
        return (
            0.0 <= self.position_size_pct <= 1.0 and
            self.current_atr > 0 and
            self.confidence_score >= 0.3 and
            self.daily_vol_estimate > 0
        )


class ATRTargetController:
    """
    ATR-based volatility targeting system
    
    Features:
    - Precise daily volatility targeting (default 1%)
    - Real-time ATR calculation with multiple timeframes
    - Volatility regime detection and adjustment
    - Position sizing based on volatility forecasts
    - Confidence scoring for reliability assessment
    - Dynamic adjustment for market conditions
    """
    
    def __init__(self, config: Optional[ATRTargetConfig] = None):
        self.config = config or ATRTargetConfig()
        
        # Price data storage
        self.price_history: deque = deque(maxlen=max(100, self.config.atr_periods * 3))
        self.atr_history: deque = deque(maxlen=50)
        self.vol_history: deque = deque(maxlen=self.config.vol_lookback_days)
        
        # Volatility estimates
        self.current_atr: float = 0.0
        self.daily_vol_ema: float = 0.0
        self.vol_regime_factor: float = 1.0
        
        # Performance tracking
        self.last_calculation: Optional[ATRTargetResult] = None
        self.calculation_count: int = 0
        
    def add_price_data(self, ohlcv: OHLCV) -> None:
        """Add new price data and update ATR calculations"""
        self.price_history.append(ohlcv)
        
        # Calculate ATR if we have enough data
        if len(self.price_history) >= self.config.atr_periods:
            atr = self._calculate_atr()
            self.current_atr = atr
            self.atr_history.append(atr)
            
            # Calculate daily volatility estimate
            daily_vol = self._estimate_daily_volatility(ohlcv.close)
            self.vol_history.append(daily_vol)
            
            # Update EMA of daily volatility
            if self.daily_vol_ema == 0.0:
                self.daily_vol_ema = daily_vol
            else:
                alpha = self.config.smoothing_alpha
                self.daily_vol_ema = alpha * daily_vol + (1 - alpha) * self.daily_vol_ema
            
            # Update volatility regime factor
            self._update_vol_regime()
            
            logger.debug(f"ATR: {atr:.6f}, Daily Vol: {daily_vol:.4f}, Price: {ohlcv.close:.2f}")
    
    def calculate_target_position_size(self, current_price: float) -> ATRTargetResult:
        """Calculate position size to achieve target volatility"""
        
        if len(self.price_history) < self.config.atr_periods:
            return self._get_default_result(current_price, "Insufficient price data")
        
        if self.current_atr <= 0:
            return self._get_default_result(current_price, "Invalid ATR")
        
        # Calculate position size based on volatility targeting
        position_size = self._calculate_position_size(current_price)
        
        # Apply regime adjustment
        adjusted_position_size = position_size * self.vol_regime_factor
        
        # Apply bounds
        adjusted_position_size = max(self.config.min_position_size, 
                                   min(self.config.max_position_size, adjusted_position_size))
        
        # Calculate target achievement ratio
        vol_target_ratio = self.daily_vol_ema / self.config.target_daily_vol if self.daily_vol_ema > 0 else 1.0
        
        # Calculate confidence score
        confidence = self._calculate_confidence_score()
        
        # Check if target can be realistically met
        target_met = (
            self.config.min_position_size <= position_size <= self.config.max_position_size and
            confidence >= self.config.confidence_threshold
        )
        
        result = ATRTargetResult(
            position_size_pct=adjusted_position_size,
            current_atr=self.current_atr,
            daily_vol_estimate=self.daily_vol_ema,
            vol_target_ratio=vol_target_ratio,
            regime_adjustment=self.vol_regime_factor,
            confidence_score=confidence,
            target_met=target_met,
            last_updated=datetime.now(),
            price_level=current_price
        )
        
        self.last_calculation = result
        self.calculation_count += 1
        
        logger.info(f"ATR Target: {adjusted_position_size:.3f} position, "
                   f"ATR: {self.current_atr:.6f}, Vol: {self.daily_vol_ema:.4f}, "
                   f"Confidence: {confidence:.3f}")
        
        return result
    
    def _calculate_atr(self) -> float:
        """Calculate Average True Range"""
        if len(self.price_history) < 2:
            return 0.0
        
        true_ranges = []
        prices = list(self.price_history)
        
        for i in range(1, len(prices)):
            curr = prices[i]
            prev = prices[i-1]
            
            # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
            tr1 = curr.high - curr.low
            tr2 = abs(curr.high - prev.close)
            tr3 = abs(curr.low - prev.close)
            
            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)
        
        # Use last N periods for ATR
        periods = min(self.config.atr_periods, len(true_ranges))
        if periods <= 0:
            return 0.0
        
        recent_tr = true_ranges[-periods:]
        atr = np.mean(recent_tr)
        
        return atr
    
    def _estimate_daily_volatility(self, current_price: float) -> float:
        """Estimate daily volatility from ATR"""
        if self.current_atr <= 0 or current_price <= 0:
            return 0.0
        
        # Convert ATR to daily volatility percentage
        # ATR is typically intraday range, scale to daily
        daily_vol = self.current_atr / current_price
        
        # Apply floor and ceiling
        daily_vol = max(self.config.vol_floor, min(self.config.vol_ceiling, daily_vol))
        
        return daily_vol
    
    def _calculate_position_size(self, current_price: float) -> float:
        """Calculate position size to achieve target volatility"""
        if self.daily_vol_ema <= 0:
            return self.config.min_position_size
        
        # Position size = target_vol / estimated_vol
        # This gives us the fraction of capital to risk
        position_size = self.config.target_daily_vol / self.daily_vol_ema
        
        return position_size
    
    def _update_vol_regime(self) -> None:
        """Update volatility regime adjustment factor"""
        if len(self.vol_history) < 10:
            self.vol_regime_factor = 1.0
            return
        
        vol_list = list(self.vol_history)
        
        # Compare recent volatility to longer-term average
        recent_vol = np.mean(vol_list[-5:])  # Last 5 days
        longer_vol = np.mean(vol_list[-20:] if len(vol_list) >= 20 else vol_list)  # Last 20 days or all
        
        if longer_vol <= 0:
            self.vol_regime_factor = 1.0
            return
        
        vol_ratio = recent_vol / longer_vol
        
        # Adjust position size based on volatility regime
        if vol_ratio > 1.5:  # High volatility regime
            # Reduce position size in high vol periods
            self.vol_regime_factor = 0.7
        elif vol_ratio > 1.2:
            self.vol_regime_factor = 0.85
        elif vol_ratio < 0.7:  # Low volatility regime
            # Slightly increase position size in low vol periods
            self.vol_regime_factor = 1.1
        elif vol_ratio < 0.8:
            self.vol_regime_factor = 1.05
        else:
            self.vol_regime_factor = 1.0
        
        # Apply sensitivity adjustment
        deviation = self.vol_regime_factor - 1.0
        self.vol_regime_factor = 1.0 + (deviation * self.config.regime_sensitivity)
    
    def _calculate_confidence_score(self) -> float:
        """Calculate confidence score for the volatility estimate"""
        if len(self.price_history) < self.config.atr_periods:
            return 0.0
        
        # Data quality score (more data = higher confidence)
        data_score = min(1.0, len(self.price_history) / (self.config.atr_periods * 2))
        
        # ATR stability score (stable ATR = higher confidence)
        if len(self.atr_history) >= 5:
            recent_atr = list(self.atr_history)[-5:]
            atr_cv = np.std(recent_atr) / np.mean(recent_atr) if np.mean(recent_atr) > 0 else float('inf')
            stability_score = max(0.0, 1.0 - atr_cv)  # Lower CV = higher score
        else:
            stability_score = 0.5
        
        # Volatility regime score (normal regimes = higher confidence)
        regime_score = 1.0 - abs(self.vol_regime_factor - 1.0)  # Closer to 1.0 = higher score
        regime_score = max(0.0, regime_score)
        
        # Price level reasonableness (avoid extreme micro/macro prices)
        if self.price_history:
            current_price = self.price_history[-1].close
            if 0.01 <= current_price <= 1000000:  # Reasonable price range
                price_score = 1.0
            else:
                price_score = 0.5
        else:
            price_score = 0.5
        
        # Target achievability score
        if self.daily_vol_ema > 0:
            implied_position = self.config.target_daily_vol / self.daily_vol_ema
            if self.config.min_position_size <= implied_position <= self.config.max_position_size:
                achievability_score = 1.0
            else:
                # Penalize if target requires extreme position sizes
                achievability_score = 0.3
        else:
            achievability_score = 0.0
        
        # Combine all components
        confidence = (
            data_score * 0.25 +
            stability_score * 0.25 +
            regime_score * 0.20 +
            price_score * 0.15 +
            achievability_score * 0.15
        )
        
        return min(1.0, confidence)
    
    def _get_default_result(self, current_price: float, reason: str) -> ATRTargetResult:
        """Get default result when calculation cannot be performed"""
        logger.warning(f"Using default ATR target result: {reason}")
        
        return ATRTargetResult(
            position_size_pct=self.config.min_position_size,
            current_atr=self.current_atr,
            daily_vol_estimate=self.daily_vol_ema,
            vol_target_ratio=1.0,
            regime_adjustment=1.0,
            confidence_score=0.0,
            target_met=False,
            last_updated=datetime.now(),
            price_level=current_price
        )
    
    def get_volatility_forecast(self, days_ahead: int = 1) -> Dict[str, float]:
        """Get volatility forecast for specified days ahead"""
        if len(self.vol_history) < 10:
            return {"error": "Insufficient data for forecasting"}
        
        vol_array = np.array(list(self.vol_history))
        
        # Simple exponential smoothing forecast
        alpha = 0.3
        forecast = self.daily_vol_ema
        
        # Extend forecast for multiple days (assuming some mean reversion)
        forecasts = []
        current_forecast = forecast
        
        for day in range(days_ahead):
            # Add some mean reversion to long-term average
            long_term_avg = np.mean(vol_array)
            mean_reversion_factor = 0.05 * day  # Stronger mean reversion over time
            
            current_forecast = (1 - mean_reversion_factor) * current_forecast + mean_reversion_factor * long_term_avg
            forecasts.append(current_forecast)
        
        return {
            "forecast_days": days_ahead,
            "forecasted_vol": forecasts,
            "current_vol": self.daily_vol_ema,
            "long_term_avg": np.mean(vol_array),
            "vol_regime_factor": self.vol_regime_factor
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive volatility statistics"""
        if not self.price_history:
            return {"error": "No price data available"}
        
        price_list = list(self.price_history)
        vol_list = list(self.vol_history)
        atr_list = list(self.atr_history)
        
        stats = {
            "data_points": len(price_list),
            "current_atr": self.current_atr,
            "current_vol_estimate": self.daily_vol_ema,
            "target_vol": self.config.target_daily_vol,
            "vol_target_ratio": self.daily_vol_ema / self.config.target_daily_vol if self.daily_vol_ema > 0 else 0,
            "vol_regime_factor": self.vol_regime_factor,
            "calculation_count": self.calculation_count
        }
        
        if vol_list:
            stats.update({
                "vol_min": np.min(vol_list),
                "vol_max": np.max(vol_list),
                "vol_mean": np.mean(vol_list),
                "vol_std": np.std(vol_list),
                "vol_25th": np.percentile(vol_list, 25),
                "vol_75th": np.percentile(vol_list, 75)
            })
        
        if atr_list:
            stats.update({
                "atr_min": np.min(atr_list),
                "atr_max": np.max(atr_list),
                "atr_mean": np.mean(atr_list),
                "atr_std": np.std(atr_list)
            })
        
        if self.last_calculation:
            stats.update({
                "last_position_size": self.last_calculation.position_size_pct,
                "last_confidence": self.last_calculation.confidence_score,
                "target_achievable": self.last_calculation.target_met
            })
        
        return stats
    
    def reset(self) -> None:
        """Reset all state for fresh start"""
        self.price_history.clear()
        self.atr_history.clear()
        self.vol_history.clear()
        
        self.current_atr = 0.0
        self.daily_vol_ema = 0.0
        self.vol_regime_factor = 1.0
        
        self.last_calculation = None
        self.calculation_count = 0
        
        logger.info("ATR Target Controller reset")
    
    def simulate_vol_targeting(self, test_prices: List[OHLCV], target_vol: Optional[float] = None) -> Dict[str, Any]:
        """Simulate volatility targeting performance on test data"""
        if target_vol is None:
            target_vol = self.config.target_daily_vol
        
        # Save current state
        original_target = self.config.target_daily_vol
        self.config.target_daily_vol = target_vol
        
        # Reset for clean simulation
        self.reset()
        
        position_sizes = []
        achieved_vols = []
        confidences = []
        
        for ohlcv in test_prices:
            self.add_price_data(ohlcv)
            
            if len(self.price_history) >= self.config.atr_periods:
                result = self.calculate_target_position_size(ohlcv.close)
                position_sizes.append(result.position_size_pct)
                achieved_vols.append(result.daily_vol_estimate)
                confidences.append(result.confidence_score)
        
        # Restore original target
        self.config.target_daily_vol = original_target
        
        if not position_sizes:
            return {"error": "Insufficient test data"}
        
        # Calculate performance metrics
        results = {
            "target_vol": target_vol,
            "avg_position_size": np.mean(position_sizes),
            "avg_achieved_vol": np.mean(achieved_vols),
            "vol_tracking_error": np.std(achieved_vols),
            "avg_confidence": np.mean(confidences),
            "position_size_std": np.std(position_sizes),
            "vol_target_hit_rate": sum(1 for v in achieved_vols if abs(v - target_vol) / target_vol < 0.2) / len(achieved_vols),
            "trades_simulated": len(position_sizes)
        }
        
        return results