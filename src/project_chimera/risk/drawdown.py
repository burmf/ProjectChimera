"""
Drawdown management with automatic de-risking tiers
Implements peak-to-current equity monitoring and position size adjustments
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum
import math


class DrawdownTier(Enum):
    """Drawdown severity tiers"""
    NORMAL = "normal"          # < 5% drawdown
    CAUTION = "caution"        # 5-10% drawdown  
    WARNING = "warning"        # 10-20% drawdown
    CRITICAL = "critical"      # > 20% drawdown


@dataclass
class DrawdownState:
    """Current drawdown state"""
    current_equity: float
    peak_equity: float
    drawdown_pct: float
    tier: DrawdownTier
    position_multiplier: float  # Position size adjustment (0.0 to 1.0)
    cooldown_until: Optional[datetime]  # Trading halt until this time
    consecutive_losses: int
    time_in_drawdown: timedelta
    
    def is_in_cooldown(self) -> bool:
        """Check if we're in a trading cooldown period"""
        return (self.cooldown_until is not None and 
                datetime.now() < self.cooldown_until)


class DrawdownManager:
    """
    Dynamic drawdown management system
    
    Features:
    - Tiered position size adjustments based on drawdown severity
    - Automatic trading halts on severe drawdowns
    - Cooldown periods for recovery
    - Peak tracking with configurable lookback
    - Volatility-adjusted drawdown thresholds
    """
    
    def __init__(
        self,
        # Tier thresholds
        caution_threshold: float = 0.05,    # 5%
        warning_threshold: float = 0.10,    # 10%
        critical_threshold: float = 0.20,   # 20%
        
        # Position multipliers
        caution_multiplier: float = 0.8,    # Reduce size by 20%
        warning_multiplier: float = 0.5,    # Reduce size by 50%
        critical_multiplier: float = 0.0,   # Stop trading
        
        # Cooldown periods (hours)
        caution_cooldown: float = 0,        # No cooldown
        warning_cooldown: float = 4,        # 4 hours
        critical_cooldown: float = 24,      # 24 hours
        
        # Recovery settings
        recovery_threshold: float = 0.5,    # 50% DD recovery to advance tier
        peak_lookback_hours: int = 168,     # 7 days peak lookback
        volatility_adjustment: bool = True  # Adjust thresholds based on volatility
    ):
        # Thresholds
        self.caution_threshold = caution_threshold
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        
        # Position multipliers
        self.caution_multiplier = caution_multiplier
        self.warning_multiplier = warning_multiplier
        self.critical_multiplier = critical_multiplier
        
        # Cooldown periods
        self.caution_cooldown = timedelta(hours=caution_cooldown)
        self.warning_cooldown = timedelta(hours=warning_cooldown)
        self.critical_cooldown = timedelta(hours=critical_cooldown)
        
        # Recovery settings
        self.recovery_threshold = recovery_threshold
        self.peak_lookback = timedelta(hours=peak_lookback_hours)
        self.volatility_adjustment = volatility_adjustment
        
        # State tracking
        self.equity_history: List[tuple[datetime, float]] = []
        self.current_state = DrawdownState(
            current_equity=1.0,
            peak_equity=1.0,
            drawdown_pct=0.0,
            tier=DrawdownTier.NORMAL,
            position_multiplier=1.0,
            cooldown_until=None,
            consecutive_losses=0,
            time_in_drawdown=timedelta()
        )
        
        # Volatility tracking for dynamic thresholds
        self.daily_returns: List[float] = []
        self.volatility_lookback = 30  # days
    
    def update_equity(self, new_equity: float, timestamp: Optional[datetime] = None) -> DrawdownState:
        """
        Update current equity and recalculate drawdown state
        
        Args:
            new_equity: New portfolio equity value
            timestamp: Timestamp of equity update
            
        Returns:
            Updated drawdown state
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Store equity history
        self.equity_history.append((timestamp, new_equity))
        
        # Maintain lookback window
        cutoff_time = timestamp - self.peak_lookback
        self.equity_history = [
            (ts, eq) for ts, eq in self.equity_history 
            if ts >= cutoff_time
        ]
        
        # Update daily returns for volatility calculation
        self._update_daily_returns(new_equity, timestamp)
        
        # Calculate peak equity within lookback window
        peak_equity = max(eq for _, eq in self.equity_history)
        
        # Calculate drawdown
        if peak_equity > 0:
            drawdown_pct = (peak_equity - new_equity) / peak_equity
        else:
            drawdown_pct = 0.0
        
        # Determine tier and multiplier
        tier, multiplier, cooldown_delta = self._calculate_tier_and_multiplier(drawdown_pct)
        
        # Set cooldown if tier changed to more severe
        cooldown_until = None
        if (tier.value != self.current_state.tier.value and 
            cooldown_delta > timedelta(0)):
            cooldown_until = timestamp + cooldown_delta
        elif self.current_state.cooldown_until:
            # Keep existing cooldown if still active
            cooldown_until = self.current_state.cooldown_until
        
        # Calculate time in drawdown
        time_in_drawdown = timedelta()
        if drawdown_pct > 0.01:  # 1% threshold for "in drawdown"
            # Find when drawdown started
            for i in range(len(self.equity_history) - 1, -1, -1):
                ts, eq = self.equity_history[i]
                if eq >= peak_equity * 0.99:  # Within 1% of peak
                    time_in_drawdown = timestamp - ts
                    break
        
        # Update consecutive losses
        consecutive_losses = self._calculate_consecutive_losses()
        
        # Create new state
        self.current_state = DrawdownState(
            current_equity=new_equity,
            peak_equity=peak_equity,
            drawdown_pct=drawdown_pct,
            tier=tier,
            position_multiplier=multiplier,
            cooldown_until=cooldown_until,
            consecutive_losses=consecutive_losses,
            time_in_drawdown=time_in_drawdown
        )
        
        return self.current_state
    
    def _calculate_tier_and_multiplier(self, drawdown_pct: float) -> tuple[DrawdownTier, float, timedelta]:
        """Calculate tier, position multiplier, and cooldown period"""
        
        # Adjust thresholds for volatility if enabled
        if self.volatility_adjustment:
            vol_adjustment = self._get_volatility_adjustment()
            caution_thresh = self.caution_threshold * vol_adjustment
            warning_thresh = self.warning_threshold * vol_adjustment
            critical_thresh = self.critical_threshold * vol_adjustment
        else:
            caution_thresh = self.caution_threshold
            warning_thresh = self.warning_threshold
            critical_thresh = self.critical_threshold
        
        # Determine tier
        if drawdown_pct >= critical_thresh:
            return (DrawdownTier.CRITICAL, self.critical_multiplier, self.critical_cooldown)
        elif drawdown_pct >= warning_thresh:
            return (DrawdownTier.WARNING, self.warning_multiplier, self.warning_cooldown)
        elif drawdown_pct >= caution_thresh:
            return (DrawdownTier.CAUTION, self.caution_multiplier, self.caution_cooldown)
        else:
            return (DrawdownTier.NORMAL, 1.0, timedelta())
    
    def _update_daily_returns(self, new_equity: float, timestamp: datetime) -> None:
        """Update daily returns for volatility calculation"""
        if len(self.equity_history) < 2:
            return
        
        # Find equity from 24 hours ago
        yesterday = timestamp - timedelta(days=1)
        prev_equity = None
        
        for ts, eq in reversed(self.equity_history):
            if ts <= yesterday:
                prev_equity = eq
                break
        
        if prev_equity and prev_equity > 0:
            daily_return = (new_equity - prev_equity) / prev_equity
            self.daily_returns.append(daily_return)
            
            # Maintain lookback window
            if len(self.daily_returns) > self.volatility_lookback:
                self.daily_returns = self.daily_returns[-self.volatility_lookback:]
    
    def _get_volatility_adjustment(self) -> float:
        """
        Calculate volatility adjustment factor for thresholds
        Higher volatility = more lenient thresholds
        """
        if len(self.daily_returns) < 10:
            return 1.0  # No adjustment with insufficient data
        
        # Calculate daily volatility
        import statistics
        vol = statistics.stdev(self.daily_returns)
        
        # Typical daily vol for crypto is ~3-5%
        # Adjust thresholds based on how vol compares to baseline
        baseline_vol = 0.04  # 4% daily vol baseline
        vol_ratio = vol / baseline_vol
        
        # Scale adjustment: higher vol = higher threshold (more lenient)
        # Cap between 0.5x and 2.0x adjustment
        adjustment = max(0.5, min(2.0, vol_ratio))
        
        return adjustment
    
    def _calculate_consecutive_losses(self) -> int:
        """Calculate number of consecutive losses"""
        if len(self.equity_history) < 2:
            return 0
        
        consecutive = 0
        prev_equity = None
        
        # Walk backwards through history
        for ts, equity in reversed(self.equity_history):
            if prev_equity is None:
                prev_equity = equity
                continue
            
            if equity < prev_equity:
                consecutive += 1
            else:
                break
            
            prev_equity = equity
        
        return consecutive
    
    def can_trade(self) -> bool:
        """Check if trading is allowed (not in cooldown)"""
        return not self.current_state.is_in_cooldown()
    
    def get_position_multiplier(self) -> float:
        """Get current position size multiplier"""
        if self.can_trade():
            return self.current_state.position_multiplier
        else:
            return 0.0  # No trading during cooldown
    
    def force_recovery(self) -> None:
        """Manually trigger recovery (reset to normal tier)"""
        self.current_state.tier = DrawdownTier.NORMAL
        self.current_state.position_multiplier = 1.0
        self.current_state.cooldown_until = None
    
    def reset_peak(self, new_peak: Optional[float] = None) -> None:
        """Reset peak equity (useful for new trading periods)"""
        if new_peak is None:
            new_peak = self.current_state.current_equity
        
        # Reset peak in history
        timestamp = datetime.now()
        self.equity_history = [(timestamp, new_peak)]
        
        # Update state
        self.current_state.peak_equity = new_peak
        self.current_state.drawdown_pct = 0.0
        self.current_state.tier = DrawdownTier.NORMAL
        self.current_state.position_multiplier = 1.0
        self.current_state.cooldown_until = None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive drawdown metrics"""
        state = self.current_state
        
        # Calculate volatility if we have data
        volatility = 0.0
        if len(self.daily_returns) > 1:
            import statistics
            volatility = statistics.stdev(self.daily_returns)
        
        # Recovery percentage (how much of drawdown has been recovered)
        recovery_pct = 0.0
        if state.drawdown_pct > 0:
            max_dd_this_period = state.drawdown_pct
            current_dd = state.drawdown_pct
            if max_dd_this_period > 0:
                recovery_pct = 1.0 - (current_dd / max_dd_this_period)
        
        return {
            'current_equity': state.current_equity,
            'peak_equity': state.peak_equity,
            'drawdown_pct': state.drawdown_pct,
            'tier': state.tier.value,
            'position_multiplier': state.position_multiplier,
            'in_cooldown': state.is_in_cooldown(),
            'cooldown_until': state.cooldown_until,
            'consecutive_losses': state.consecutive_losses,
            'time_in_drawdown_hours': state.time_in_drawdown.total_seconds() / 3600,
            'daily_volatility': volatility,
            'recovery_pct': recovery_pct,
            'can_trade': self.can_trade(),
            'equity_history_length': len(self.equity_history)
        }
    
    def simulate_recovery_time(self, target_return_rate: float = 0.02) -> Optional[datetime]:
        """
        Estimate recovery time based on target daily return rate
        
        Args:
            target_return_rate: Expected daily return rate
            
        Returns:
            Estimated recovery datetime, or None if already recovered
        """
        state = self.current_state
        
        if state.drawdown_pct <= 0.01:  # Already recovered
            return None
        
        if target_return_rate <= 0:
            return None  # Can't recover with zero/negative returns
        
        # Calculate required return to reach peak
        required_total_return = (state.peak_equity / state.current_equity) - 1
        
        # Estimate days needed (simplified compound growth)
        days_needed = math.log(1 + required_total_return) / math.log(1 + target_return_rate)
        
        # Add current time
        recovery_time = datetime.now() + timedelta(days=days_needed)
        
        return recovery_time