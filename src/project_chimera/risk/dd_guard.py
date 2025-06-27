"""
Maximum Drawdown Guard System
Implements automatic de-risking with tiered responses:
- 10% DD → size × 0.5 (50% position reduction)
- 20% DD → flat + 24h cooldown (complete trading halt)
"""

import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class DDGuardTier(Enum):
    """Drawdown guard severity tiers"""

    NORMAL = "normal"  # < 5% drawdown - normal operations
    CAUTION = "caution"  # 5-10% drawdown - light position reduction
    WARNING = "warning"  # 10-20% drawdown - significant position reduction
    CRITICAL = "critical"  # > 20% drawdown - full trading halt


@dataclass
class DDGuardConfig:
    """Configuration for drawdown guard system"""

    # Drawdown thresholds
    caution_threshold: float = 0.05  # 5% - start reducing size
    warning_threshold: float = 0.10  # 10% - major size reduction
    critical_threshold: float = 0.20  # 20% - trading halt

    # Position adjustments
    caution_multiplier: float = 0.8  # Reduce to 80% of normal size
    warning_multiplier: float = 0.5  # Reduce to 50% of normal size
    critical_multiplier: float = 0.0  # No trading allowed

    # Cooldown periods
    warning_cooldown_hours: float = 4.0  # 4 hour cooldown at 10%
    critical_cooldown_hours: float = 24.0  # 24 hour cooldown at 20%

    # Recovery settings
    recovery_threshold: float = 0.5  # Recover 50% of DD to move down a tier
    min_recovery_time_hours: float = 2.0  # Minimum time in tier before recovery

    # Performance tracking
    lookback_periods: int = 1000  # Track peak over N periods
    peak_update_threshold: float = 0.01  # Min gain to update peak (1%)
    equity_smoothing: bool = False  # Smooth equity for peak calculation


@dataclass
class DDGuardState:
    """Current state of the drawdown guard system"""

    current_equity: float
    peak_equity: float
    drawdown_pct: float
    tier: DDGuardTier
    position_multiplier: float
    cooldown_until: datetime | None
    tier_entry_time: datetime
    consecutive_losses: int
    time_in_drawdown: timedelta
    recovery_progress: float  # 0.0 = at worst, 1.0 = fully recovered

    def is_in_cooldown(self) -> bool:
        """Check if we're in a trading cooldown period"""
        if self.cooldown_until is None:
            return False
        return datetime.now() < self.cooldown_until

    def can_trade(self) -> bool:
        """Check if trading is allowed"""
        return not self.is_in_cooldown() and self.position_multiplier > 0

    def time_until_cooldown_ends(self) -> timedelta | None:
        """Get time remaining in cooldown"""
        if not self.is_in_cooldown():
            return None
        return self.cooldown_until - datetime.now()


class DDGuardSystem:
    """
    Maximum Drawdown Guard System

    Features:
    - Real-time drawdown monitoring with peak tracking
    - Tiered response system with automatic de-risking
    - Cooldown periods to prevent overtrading during stress
    - Recovery mechanisms with hysteresis to prevent flapping
    - Configurable thresholds and responses
    - Comprehensive state tracking and logging
    """

    def __init__(
        self, config: DDGuardConfig | None = None, initial_equity: float = 1.0
    ):
        self.config = config or DDGuardConfig()

        # Equity tracking
        self.equity_history: deque = deque(maxlen=self.config.lookback_periods)
        self.peak_equity: float = initial_equity
        self.trough_equity: float = initial_equity
        self.current_equity: float = initial_equity

        # State management
        self.current_tier: DDGuardTier = DDGuardTier.NORMAL
        self.tier_entry_time: datetime = datetime.now()
        self.cooldown_until: datetime | None = None
        self.consecutive_losses: int = 0

        # Performance tracking
        self.worst_drawdown: float = 0.0
        self.time_in_drawdown_total: timedelta = timedelta()
        self.tier_history: list[tuple[datetime, DDGuardTier, float]] = []

        # Initialize with first equity point
        self.equity_history.append(initial_equity)

        logger.info(f"DD Guard initialized with equity: {initial_equity:.4f}")

    def update_equity(
        self, new_equity: float, timestamp: datetime | None = None
    ) -> DDGuardState:
        """Update equity and recalculate drawdown state"""
        if timestamp is None:
            timestamp = datetime.now()

        self.current_equity = new_equity
        self.equity_history.append(new_equity)

        # Update peak if significant gain
        if new_equity > self.peak_equity * (1 + self.config.peak_update_threshold):
            old_peak = self.peak_equity
            self.peak_equity = new_equity
            logger.debug(
                f"New equity peak: {self.peak_equity:.4f} (from {old_peak:.4f})"
            )

        # Calculate current drawdown
        current_dd = self._calculate_drawdown()

        # Update worst drawdown
        self.worst_drawdown = max(self.worst_drawdown, current_dd)

        # Determine appropriate tier
        new_tier = self._determine_tier(current_dd)

        # Handle tier transitions
        if new_tier != self.current_tier:
            self._handle_tier_transition(new_tier, current_dd, timestamp)

        # Update trough tracking
        if new_equity < self.trough_equity:
            self.trough_equity = new_equity

        # Track consecutive losses
        if len(self.equity_history) >= 2:
            if new_equity < self.equity_history[-2]:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0

        # Create current state
        state = DDGuardState(
            current_equity=self.current_equity,
            peak_equity=self.peak_equity,
            drawdown_pct=current_dd,
            tier=self.current_tier,
            position_multiplier=self._get_position_multiplier(),
            cooldown_until=self.cooldown_until,
            tier_entry_time=self.tier_entry_time,
            consecutive_losses=self.consecutive_losses,
            time_in_drawdown=timestamp - self.tier_entry_time,
            recovery_progress=self._calculate_recovery_progress(),
        )

        logger.debug(
            f"DD Guard: Equity={new_equity:.4f}, DD={current_dd:.3f}, "
            f"Tier={self.current_tier.value}, Mult={state.position_multiplier:.2f}"
        )

        return state

    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown percentage"""
        if self.peak_equity <= 0:
            return 0.0

        equity_to_use = self.current_equity

        # Optional smoothing for less noisy peak calculation
        if self.config.equity_smoothing and len(self.equity_history) >= 3:
            recent_equities = list(self.equity_history)[-3:]
            equity_to_use = np.mean(recent_equities)

        drawdown = (self.peak_equity - equity_to_use) / self.peak_equity
        return max(0.0, drawdown)

    def _determine_tier(self, drawdown: float) -> DDGuardTier:
        """Determine appropriate tier based on drawdown"""
        if drawdown >= self.config.critical_threshold:
            return DDGuardTier.CRITICAL
        elif drawdown >= self.config.warning_threshold:
            return DDGuardTier.WARNING
        elif drawdown >= self.config.caution_threshold:
            return DDGuardTier.CAUTION
        else:
            return DDGuardTier.NORMAL

    def _handle_tier_transition(
        self, new_tier: DDGuardTier, drawdown: float, timestamp: datetime
    ) -> None:
        """Handle transition between drawdown tiers"""
        old_tier = self.current_tier

        # Check if enough time has passed for tier changes (prevent flapping)
        time_in_current_tier = timestamp - self.tier_entry_time
        min_time_in_tier = timedelta(hours=self.config.min_recovery_time_hours)

        # Allow immediate escalation to higher tiers, but require time for recovery
        tier_severity = {
            DDGuardTier.NORMAL: 0,
            DDGuardTier.CAUTION: 1,
            DDGuardTier.WARNING: 2,
            DDGuardTier.CRITICAL: 3,
        }

        is_escalation = tier_severity[new_tier] > tier_severity[old_tier]
        is_recovery = tier_severity[new_tier] < tier_severity[old_tier]

        if is_recovery and time_in_current_tier < min_time_in_tier:
            logger.debug(
                f"DD Guard: Preventing tier recovery too soon "
                f"(need {min_time_in_tier}, been {time_in_current_tier})"
            )
            return

        # Check recovery requirements
        if is_recovery:
            recovery_needed = self._calculate_recovery_progress()
            if recovery_needed < self.config.recovery_threshold:
                logger.debug(
                    f"DD Guard: Insufficient recovery progress "
                    f"({recovery_needed:.2f} < {self.config.recovery_threshold:.2f})"
                )
                return

        # Execute tier transition
        logger.info(
            f"DD Guard: Tier transition {old_tier.value} → {new_tier.value} "
            f"(DD: {drawdown:.3f}, Equity: {self.current_equity:.4f})"
        )

        self.current_tier = new_tier
        self.tier_entry_time = timestamp

        # Set cooldown if entering warning or critical tiers
        if new_tier == DDGuardTier.WARNING:
            self.cooldown_until = timestamp + timedelta(
                hours=self.config.warning_cooldown_hours
            )
            logger.warning(
                f"DD Guard: 10% DD reached - reducing position size to 50%, "
                f"cooldown until {self.cooldown_until}"
            )
        elif new_tier == DDGuardTier.CRITICAL:
            self.cooldown_until = timestamp + timedelta(
                hours=self.config.critical_cooldown_hours
            )
            logger.critical(
                f"DD Guard: 20% DD reached - TRADING HALT for 24 hours, "
                f"cooldown until {self.cooldown_until}"
            )

        # Record transition
        self.tier_history.append((timestamp, new_tier, drawdown))

    def _get_position_multiplier(self) -> float:
        """Get current position size multiplier based on tier"""
        # Check cooldown first
        if self.is_in_cooldown():
            return 0.0

        # Return multiplier based on tier
        multipliers = {
            DDGuardTier.NORMAL: 1.0,
            DDGuardTier.CAUTION: self.config.caution_multiplier,
            DDGuardTier.WARNING: self.config.warning_multiplier,
            DDGuardTier.CRITICAL: self.config.critical_multiplier,
        }

        return multipliers.get(self.current_tier, 0.0)

    def _calculate_recovery_progress(self) -> float:
        """Calculate recovery progress from trough (0.0 = at trough, 1.0 = at peak)"""
        if self.peak_equity <= self.trough_equity:
            return 1.0  # No drawdown case

        recovery_range = self.peak_equity - self.trough_equity
        current_recovery = self.current_equity - self.trough_equity

        progress = current_recovery / recovery_range
        return max(0.0, min(1.0, progress))

    def is_in_cooldown(self) -> bool:
        """Check if system is in cooldown period"""
        if self.cooldown_until is None:
            return False
        return datetime.now() < self.cooldown_until

    def get_current_state(self) -> DDGuardState:
        """Get current drawdown guard state"""
        return DDGuardState(
            current_equity=self.current_equity,
            peak_equity=self.peak_equity,
            drawdown_pct=self._calculate_drawdown(),
            tier=self.current_tier,
            position_multiplier=self._get_position_multiplier(),
            cooldown_until=self.cooldown_until,
            tier_entry_time=self.tier_entry_time,
            consecutive_losses=self.consecutive_losses,
            time_in_drawdown=datetime.now() - self.tier_entry_time,
            recovery_progress=self._calculate_recovery_progress(),
        )

    def force_tier_change(
        self, new_tier: DDGuardTier, reason: str = "Manual override"
    ) -> None:
        """Force a tier change (for emergency situations)"""
        logger.warning(f"DD Guard: Force tier change to {new_tier.value} - {reason}")

        old_tier = self.current_tier
        self.current_tier = new_tier
        self.tier_entry_time = datetime.now()

        # Set appropriate cooldown
        if new_tier == DDGuardTier.WARNING:
            self.cooldown_until = datetime.now() + timedelta(
                hours=self.config.warning_cooldown_hours
            )
        elif new_tier == DDGuardTier.CRITICAL:
            self.cooldown_until = datetime.now() + timedelta(
                hours=self.config.critical_cooldown_hours
            )
        else:
            self.cooldown_until = None

        self.tier_history.append((datetime.now(), new_tier, self._calculate_drawdown()))

    def reset_peak(self, new_peak: float | None = None) -> None:
        """Reset peak equity (use carefully)"""
        old_peak = self.peak_equity
        self.peak_equity = new_peak or self.current_equity
        self.trough_equity = self.current_equity

        logger.warning(
            f"DD Guard: Peak reset from {old_peak:.4f} to {self.peak_equity:.4f}"
        )

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive drawdown guard statistics"""
        current_dd = self._calculate_drawdown()

        # Calculate time spent in each tier
        tier_times = {tier: timedelta() for tier in DDGuardTier}

        for i, (timestamp, tier, _) in enumerate(self.tier_history):
            if i < len(self.tier_history) - 1:
                next_timestamp = self.tier_history[i + 1][0]
                duration = next_timestamp - timestamp
            else:
                duration = datetime.now() - timestamp

            tier_times[tier] += duration

        # Get equity curve statistics
        if len(self.equity_history) > 1:
            equity_array = np.array(list(self.equity_history))
            equity_returns = np.diff(equity_array) / equity_array[:-1]

            stats = {
                "current_equity": self.current_equity,
                "peak_equity": self.peak_equity,
                "trough_equity": self.trough_equity,
                "current_drawdown": current_dd,
                "worst_drawdown": self.worst_drawdown,
                "current_tier": self.current_tier.value,
                "position_multiplier": self._get_position_multiplier(),
                "is_in_cooldown": self.is_in_cooldown(),
                "consecutive_losses": self.consecutive_losses,
                "recovery_progress": self._calculate_recovery_progress(),
                # Equity curve stats
                "total_return": (
                    (self.current_equity / self.equity_history[0] - 1)
                    if self.equity_history[0] != 0
                    else 0
                ),
                "volatility": np.std(equity_returns) * np.sqrt(252),  # Annualized
                "sharpe_estimate": (
                    np.mean(equity_returns) / np.std(equity_returns) * np.sqrt(252)
                    if np.std(equity_returns) > 0
                    else 0
                ),
                # Tier statistics
                "tier_changes": len(self.tier_history),
                "time_in_normal": tier_times[DDGuardTier.NORMAL].total_seconds() / 3600,
                "time_in_caution": tier_times[DDGuardTier.CAUTION].total_seconds()
                / 3600,
                "time_in_warning": tier_times[DDGuardTier.WARNING].total_seconds()
                / 3600,
                "time_in_critical": tier_times[DDGuardTier.CRITICAL].total_seconds()
                / 3600,
            }

            if self.cooldown_until:
                stats["cooldown_remaining_hours"] = (
                    self.cooldown_until - datetime.now()
                ).total_seconds() / 3600

        else:
            stats = {"error": "Insufficient equity data"}

        return stats

    def simulate_drawdown_response(self, equity_curve: list[float]) -> dict[str, Any]:
        """Simulate drawdown guard behavior on a given equity curve"""
        # Save current state
        original_equity = self.current_equity
        original_peak = self.peak_equity
        original_tier = self.current_tier
        original_cooldown = self.cooldown_until

        # Reset for simulation
        self.reset_system(equity_curve[0] if equity_curve else 1.0)

        simulation_results = []
        tier_changes = 0
        max_position_reduction = 0.0
        total_cooldown_time = 0.0

        for i, equity in enumerate(equity_curve):
            state = self.update_equity(equity, datetime.now() + timedelta(hours=i))

            simulation_results.append(
                {
                    "equity": equity,
                    "drawdown": state.drawdown_pct,
                    "tier": state.tier.value,
                    "position_multiplier": state.position_multiplier,
                    "in_cooldown": state.is_in_cooldown(),
                }
            )

            # Track statistics
            if (
                i > 0
                and simulation_results[i]["tier"] != simulation_results[i - 1]["tier"]
            ):
                tier_changes += 1

            max_position_reduction = max(
                max_position_reduction, 1.0 - state.position_multiplier
            )

            if state.is_in_cooldown():
                total_cooldown_time += 1.0  # Assuming hourly data

        # Restore original state
        self.current_equity = original_equity
        self.peak_equity = original_peak
        self.current_tier = original_tier
        self.cooldown_until = original_cooldown

        return {
            "simulation_points": len(equity_curve),
            "tier_changes": tier_changes,
            "max_drawdown_simulated": max(r["drawdown"] for r in simulation_results),
            "max_position_reduction": max_position_reduction,
            "total_cooldown_hours": total_cooldown_time,
            "final_tier": (
                simulation_results[-1]["tier"] if simulation_results else "unknown"
            ),
            "drawdown_breaches": {
                "caution": sum(
                    1
                    for r in simulation_results
                    if r["drawdown"] >= self.config.caution_threshold
                ),
                "warning": sum(
                    1
                    for r in simulation_results
                    if r["drawdown"] >= self.config.warning_threshold
                ),
                "critical": sum(
                    1
                    for r in simulation_results
                    if r["drawdown"] >= self.config.critical_threshold
                ),
            },
        }

    def reset_system(self, initial_equity: float = 1.0) -> None:
        """Reset entire system state"""
        self.equity_history.clear()
        self.equity_history.append(initial_equity)

        self.peak_equity = initial_equity
        self.trough_equity = initial_equity
        self.current_equity = initial_equity

        self.current_tier = DDGuardTier.NORMAL
        self.tier_entry_time = datetime.now()
        self.cooldown_until = None
        self.consecutive_losses = 0

        self.worst_drawdown = 0.0
        self.time_in_drawdown_total = timedelta()
        self.tier_history.clear()

        logger.info(f"DD Guard system reset with initial equity: {initial_equity:.4f}")
