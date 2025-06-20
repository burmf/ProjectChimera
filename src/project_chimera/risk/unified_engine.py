"""
Unified Risk Engine - Phase D Implementation
Integrates Dynamic Kelly, ATR Target, and DD Guard systems
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import logging

from .dyn_kelly import DynamicKellyCalculator, DynamicKellyConfig, DynamicKellyResult
from .atr_target import ATRTargetController, ATRTargetConfig, ATRTargetResult
from .dd_guard import DDGuardSystem, DDGuardConfig, DDGuardState, DDGuardTier
from ..domains.market import OHLCV, Signal
from ..settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class UnifiedRiskConfig:
    """Configuration for unified risk engine"""
    # Kelly configuration
    kelly_base_fraction: float = 0.5      # ½-Kelly by default
    kelly_ewma_alpha: float = 0.1         # EWMA decay
    kelly_min_trades: int = 20            # Min trades for Kelly
    
    # ATR target configuration  
    atr_target_daily_vol: float = 0.01    # 1% daily volatility target
    atr_periods: int = 14                 # ATR calculation periods
    atr_min_position: float = 0.01        # Min position size (1%)
    atr_max_position: float = 0.20        # Max position size (20%)
    
    # DD guard configuration
    dd_caution_threshold: float = 0.05    # 5% - start reducing
    dd_warning_threshold: float = 0.10    # 10% - major reduction (×0.5)
    dd_critical_threshold: float = 0.20   # 20% - trading halt (×0.0)
    dd_warning_cooldown_hours: float = 4.0   # 4h cooldown at 10%
    dd_critical_cooldown_hours: float = 24.0 # 24h cooldown at 20%
    
    # Engine limits
    max_leverage: float = 10.0            # Maximum leverage allowed
    min_confidence: float = 0.3           # Minimum confidence to trade
    max_portfolio_vol: float = 0.02       # Max portfolio volatility (2%)
    
    @classmethod
    def from_settings(cls) -> 'UnifiedRiskConfig':
        """Create config from application settings"""
        settings = get_settings()
        risk_config = settings.risk
        
        return cls(
            kelly_base_fraction=risk_config.kelly_base_fraction,
            kelly_ewma_alpha=risk_config.kelly_ewma_alpha,
            kelly_min_trades=risk_config.kelly_min_trades,
            atr_target_daily_vol=risk_config.atr_target_daily_vol,
            atr_periods=risk_config.atr_periods,
            atr_min_position=risk_config.atr_min_position,
            atr_max_position=risk_config.atr_max_position,
            dd_caution_threshold=risk_config.dd_caution_threshold,
            dd_warning_threshold=risk_config.dd_warning_threshold,
            dd_critical_threshold=risk_config.dd_critical_threshold,
            dd_warning_cooldown_hours=risk_config.dd_warning_cooldown_hours,
            dd_critical_cooldown_hours=risk_config.dd_critical_cooldown_hours,
            max_leverage=risk_config.max_leverage,
            min_confidence=risk_config.min_confidence,
            max_portfolio_vol=risk_config.max_portfolio_vol
        )


@dataclass
class UnifiedRiskDecision:
    """Final unified risk decision"""
    # Final sizing
    position_size_pct: float              # Final position size (% of portfolio)
    leverage: float                       # Recommended leverage
    can_trade: bool                       # Whether trading is allowed
    confidence: float                     # Overall confidence (0-1)
    
    # Component results
    kelly_result: DynamicKellyResult      # Kelly calculation result
    atr_result: ATRTargetResult           # ATR target result
    dd_state: DDGuardState                # Drawdown guard state
    
    # Risk metrics
    estimated_daily_vol: float            # Estimated daily portfolio volatility
    risk_adjusted_return: float           # Expected risk-adjusted return
    max_loss_estimate: float              # Maximum expected loss
    
    # Decision logic
    primary_constraint: str               # What limited the position
    sizing_method: str                    # Primary sizing method used
    reasoning: str                        # Human-readable explanation
    
    def is_valid(self) -> bool:
        """Check if decision is valid for execution"""
        return (
            self.can_trade and
            0.0 <= self.position_size_pct <= 1.0 and
            self.leverage >= 1.0 and
            self.confidence >= 0.3
        )


class UnifiedRiskEngine:
    """
    Unified Risk Engine - Phase D Implementation
    
    Integrates:
    - Dynamic Kelly Criterion with EWMA win-rate and ½-Kelly safety
    - ATR-Target volatility control (1% daily target)
    - Max DD guard (10%→×0.5, 20%→flat+24h cooldown)
    
    Features:
    - Hot-reload configuration support
    - Comprehensive risk decision logic
    - Real-time adaptation to market conditions
    - Detailed reasoning and constraint identification
    """
    
    def __init__(self, config: Optional[UnifiedRiskConfig] = None, initial_equity: float = 1.0):
        self.config = config or UnifiedRiskConfig.from_settings()
        
        # Initialize risk components
        kelly_config = DynamicKellyConfig(
            base_kelly_fraction=self.config.kelly_base_fraction,
            ewma_alpha=self.config.kelly_ewma_alpha,
            min_sample_size=self.config.kelly_min_trades
        )
        self.kelly_calc = DynamicKellyCalculator(kelly_config)
        
        atr_config = ATRTargetConfig(
            target_daily_vol=self.config.atr_target_daily_vol,
            atr_periods=self.config.atr_periods,
            min_position_size=self.config.atr_min_position,
            max_position_size=self.config.atr_max_position
        )
        self.atr_controller = ATRTargetController(atr_config)
        
        dd_config = DDGuardConfig(
            caution_threshold=self.config.dd_caution_threshold,
            warning_threshold=self.config.dd_warning_threshold,
            critical_threshold=self.config.dd_critical_threshold,
            warning_cooldown_hours=self.config.dd_warning_cooldown_hours,
            critical_cooldown_hours=self.config.dd_critical_cooldown_hours
        )
        self.dd_guard = DDGuardSystem(dd_config, initial_equity)
        
        # State tracking
        self.last_decision: Optional[UnifiedRiskDecision] = None
        self.decision_count: int = 0
        
        logger.info(f"Unified Risk Engine initialized with equity: {initial_equity:.4f}")
    
    async def calculate_position_size(
        self,
        signal: Signal,
        current_price: float,
        portfolio_value: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> UnifiedRiskDecision:
        """
        Calculate optimal position size using all risk factors (ASYNC)
        
        Process:
        1. Await portfolio value from wallet if not provided
        2. Check DD guard status - can we trade?
        3. Get Kelly-optimal sizing if sufficient data
        4. Get ATR-based sizing for volatility targeting  
        5. Apply DD guard multiplier
        6. Calculate final leverage and risk metrics
        7. Generate comprehensive decision with reasoning
        """
        
        # Await wallet equity if not provided
        if portfolio_value is None:
            portfolio_value = await self._get_portfolio_value()
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Update DD guard with current portfolio value
        dd_state = self.dd_guard.update_equity(portfolio_value, timestamp)
        
        # Check if trading is allowed (DD guard + cooldown)
        if not dd_state.can_trade():
            return self._create_no_trade_decision(dd_state, "DD Guard trading halt")
        
        # Get Kelly sizing recommendation
        kelly_result = self.kelly_calc.calculate_dynamic_kelly()
        
        # Get ATR sizing recommendation
        atr_result = self.atr_controller.calculate_target_position_size(current_price)
        
        # Determine base position size
        base_size, sizing_method, confidence = self._calculate_base_size(
            kelly_result, atr_result, signal
        )
        
        # Apply DD guard multiplier
        dd_multiplier = dd_state.position_multiplier
        adjusted_size = base_size * dd_multiplier
        
        # Apply portfolio volatility limits
        vol_limited_size = self._apply_volatility_limits(
            adjusted_size, atr_result, portfolio_value
        )
        
        # Calculate leverage
        leverage = self._calculate_leverage(vol_limited_size, signal.confidence)
        
        # Calculate risk metrics
        estimated_vol = self._estimate_portfolio_volatility(
            vol_limited_size, atr_result.daily_vol_estimate
        )
        
        risk_adjusted_return = self._estimate_risk_adjusted_return(
            vol_limited_size, signal.confidence, estimated_vol
        )
        
        max_loss_estimate = self._estimate_max_loss(
            vol_limited_size, atr_result.daily_vol_estimate
        )
        
        # Identify primary constraint
        primary_constraint = self._identify_constraint(
            base_size, adjusted_size, vol_limited_size, dd_multiplier
        )
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            kelly_result, atr_result, dd_state, signal, confidence
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            signal, kelly_result, atr_result, dd_state,
            vol_limited_size, sizing_method, primary_constraint
        )
        
        decision = UnifiedRiskDecision(
            position_size_pct=vol_limited_size,
            leverage=leverage,
            can_trade=True,
            confidence=overall_confidence,
            kelly_result=kelly_result,
            atr_result=atr_result,
            dd_state=dd_state,
            estimated_daily_vol=estimated_vol,
            risk_adjusted_return=risk_adjusted_return,
            max_loss_estimate=max_loss_estimate,
            primary_constraint=primary_constraint,
            sizing_method=sizing_method,
            reasoning=reasoning
        )
        
        self.last_decision = decision
        self.decision_count += 1
        
        logger.info(f"Risk decision: {vol_limited_size:.3f} position, "
                   f"leverage: {leverage:.2f}x, confidence: {overall_confidence:.3f}")
        
        return decision
    
    def update_trade_result(
        self,
        return_pct: float,
        new_portfolio_value: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Update all risk components with trade result"""
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Update Kelly calculator
        self.kelly_calc.add_trade_result(return_pct, timestamp)
        
        # Update DD guard (automatically called in next position calculation)
        self.dd_guard.update_equity(new_portfolio_value, timestamp)
        
        logger.info(f"Updated trade result: {return_pct:.4f}% return, "
                   f"portfolio: {new_portfolio_value:.2f}")
    
    def add_price_data(self, ohlcv: OHLCV) -> None:
        """Add new price data for ATR calculations"""
        self.atr_controller.add_price_data(ohlcv)
    
    def _calculate_base_size(
        self,
        kelly_result: DynamicKellyResult,
        atr_result: ATRTargetResult,
        signal: Signal
    ) -> tuple[float, str, float]:
        """Calculate base position size using best available method"""
        
        options = []
        
        # Kelly sizing (if sufficient data and valid)
        if kelly_result.is_valid() and kelly_result.sample_size >= self.config.kelly_min_trades:
            options.append(('kelly', kelly_result.kelly_fraction, kelly_result.confidence_score))
        
        # ATR sizing (always available)
        if atr_result.is_valid():
            options.append(('atr', atr_result.position_size_pct, atr_result.confidence_score))
        
        # Conservative fixed sizing (fallback)
        options.append(('conservative', 0.05, 0.7))  # 5% fixed allocation
        
        if not options:
            return 0.01, 'minimal', 0.3  # 1% minimal fallback
        
        # Weight by confidence and signal strength
        total_weight = 0.0
        weighted_size = 0.0
        
        for method, size, confidence in options:
            # Boost weight for high-confidence signals
            weight = confidence * signal.confidence
            weighted_size += size * weight
            total_weight += weight
        
        if total_weight > 0:
            final_size = weighted_size / total_weight
            # Primary method is the highest confidence one
            best_method = max(options, key=lambda x: x[2])[0]
            avg_confidence = total_weight / len(options)
        else:
            final_size = options[0][1]
            best_method = options[0][0]
            avg_confidence = options[0][2]
        
        return final_size, best_method, avg_confidence
    
    def _apply_volatility_limits(
        self,
        position_size: float,
        atr_result: ATRTargetResult,
        portfolio_value: float
    ) -> float:
        """Apply portfolio-level volatility limits"""
        
        # Estimate portfolio volatility with this position size
        estimated_vol = position_size * atr_result.daily_vol_estimate
        
        # If exceeds max portfolio volatility, scale down
        if estimated_vol > self.config.max_portfolio_vol:
            scaling_factor = self.config.max_portfolio_vol / estimated_vol
            return position_size * scaling_factor
        
        return position_size
    
    def _calculate_leverage(self, position_size: float, signal_confidence: float) -> float:
        """Calculate appropriate leverage"""
        
        # Base leverage scales with position size
        base_leverage = 1.0 + (position_size * 10.0)  # 1x + up to 2x for 20% position
        
        # Reduce leverage for low confidence signals
        confidence_adjustment = 0.5 + (signal_confidence * 0.5)
        adjusted_leverage = base_leverage * confidence_adjustment
        
        # Apply limits
        return min(adjusted_leverage, self.config.max_leverage)
    
    def _estimate_portfolio_volatility(self, position_size: float, asset_volatility: float) -> float:
        """Estimate daily portfolio volatility"""
        return position_size * asset_volatility
    
    def _estimate_risk_adjusted_return(
        self,
        position_size: float,
        signal_confidence: float,
        portfolio_volatility: float
    ) -> float:
        """Estimate risk-adjusted expected return"""
        # Assume base expected return of 1% for confident signals
        expected_return = signal_confidence * 0.01
        return (position_size * expected_return) / max(portfolio_volatility, 0.001)
    
    def _estimate_max_loss(self, position_size: float, asset_volatility: float) -> float:
        """Estimate maximum expected loss (2-sigma)"""
        return position_size * asset_volatility * 2.0  # 2-sigma loss
    
    def _identify_constraint(
        self,
        base_size: float,
        dd_adjusted_size: float,
        final_size: float,
        dd_multiplier: float
    ) -> str:
        """Identify what constrained the final position size"""
        
        if dd_multiplier < 1.0:
            dd_tier = "critical" if dd_multiplier == 0.0 else "warning" if dd_multiplier <= 0.5 else "caution"
            return f"drawdown_guard_{dd_tier}"
        
        if final_size < dd_adjusted_size * 0.95:
            return "volatility_limit"
        
        if base_size <= 0.05:
            return "insufficient_confidence"
        
        return "normal_sizing"
    
    def _calculate_overall_confidence(
        self,
        kelly_result: DynamicKellyResult,
        atr_result: ATRTargetResult,
        dd_state: DDGuardState,
        signal: Signal,
        base_confidence: float
    ) -> float:
        """Calculate overall confidence score"""
        
        factors = [
            base_confidence,                              # Sizing method confidence
            signal.confidence,                            # Signal confidence
            atr_result.confidence_score,                  # ATR confidence
            1.0 - dd_state.drawdown_pct,                # DD penalty (higher DD = lower confidence)
        ]
        
        # Add Kelly confidence if available
        if kelly_result.sample_size >= self.config.kelly_min_trades:
            factors.append(kelly_result.confidence_score)
        
        return sum(factors) / len(factors)
    
    def _generate_reasoning(
        self,
        signal: Signal,
        kelly_result: DynamicKellyResult,
        atr_result: ATRTargetResult,
        dd_state: DDGuardState,
        final_size: float,
        sizing_method: str,
        constraint: str
    ) -> str:
        """Generate human-readable reasoning for the decision"""
        
        parts = []
        
        # Signal assessment
        signal_strength = "strong" if signal.confidence > 0.7 else "moderate" if signal.confidence > 0.4 else "weak"
        parts.append(f"{signal_strength} {signal.signal_type.value} signal ({signal.confidence:.2f})")
        
        # Sizing method
        if sizing_method == "kelly" and kelly_result.sample_size >= self.config.kelly_min_trades:
            parts.append(f"Kelly sizing: {kelly_result.kelly_fraction:.3f} "
                        f"(WR: {kelly_result.ewma_win_rate:.2f}, {kelly_result.sample_size} trades)")
        elif sizing_method == "atr":
            parts.append(f"ATR sizing: {atr_result.position_size_pct:.3f} "
                        f"(target: {self.config.atr_target_daily_vol:.1%} vol)")
        else:
            parts.append(f"Conservative sizing: {final_size:.3f}")
        
        # DD guard status
        if dd_state.tier != DDGuardTier.NORMAL:
            parts.append(f"DD guard: {dd_state.tier.value} "
                        f"({dd_state.drawdown_pct:.1%} DD, ×{dd_state.position_multiplier:.1f})")
        
        # Primary constraint
        if constraint != "normal_sizing":
            constraint_desc = {
                "drawdown_guard_critical": "20% DD halt",
                "drawdown_guard_warning": "10% DD reduction",
                "drawdown_guard_caution": "5% DD caution",
                "volatility_limit": "portfolio vol limit",
                "insufficient_confidence": "low confidence"
            }.get(constraint, constraint)
            parts.append(f"Limited by: {constraint_desc}")
        
        return " | ".join(parts)
    
    def _create_no_trade_decision(self, dd_state: DDGuardState, reason: str) -> UnifiedRiskDecision:
        """Create a no-trade decision"""
        
        return UnifiedRiskDecision(
            position_size_pct=0.0,
            leverage=1.0,
            can_trade=False,
            confidence=0.0,
            kelly_result=self.kelly_calc.last_calculation or DynamicKellyResult(
                kelly_fraction=0.0, ewma_win_rate=0.5, ewma_avg_win=0.0,
                ewma_avg_loss=0.0, raw_kelly=0.0, vol_adjustment_factor=1.0,
                confidence_score=0.0, sample_size=0, last_updated=datetime.now()
            ),
            atr_result=self.atr_controller.last_calculation or ATRTargetResult(
                position_size_pct=0.0, current_atr=0.0, daily_vol_estimate=0.0,
                vol_target_ratio=1.0, regime_adjustment=1.0, confidence_score=0.0,
                target_met=False, last_updated=datetime.now(), price_level=0.0
            ),
            dd_state=dd_state,
            estimated_daily_vol=0.0,
            risk_adjusted_return=0.0,
            max_loss_estimate=0.0,
            primary_constraint="trading_halt",
            sizing_method="none",
            reasoning=reason
        )
    
    async def _get_portfolio_value(self) -> float:
        """Get current portfolio value from wallet/exchange"""
        try:
            # In production, this would call the exchange API
            # For now, return the last known equity value from DD guard
            return self.dd_guard.current_equity
        except Exception as e:
            logger.error(f"Failed to get portfolio value: {e}")
            # Fallback to DD guard equity
            return self.dd_guard.current_equity
    
    async def _get_trade_statistics(self) -> Dict[str, float]:
        """Get trade statistics for Kelly calculation"""
        try:
            # TODO: In production, query trade database for actual stats
            # For now, return default conservative estimates
            return {
                'win_rate': 0.55,
                'avg_win': 0.02,
                'avg_loss': 0.01,
                'total_trades': 50
            }
        except Exception as e:
            logger.error(f"Failed to get trade statistics: {e}")
            return {
                'win_rate': 0.50,  # Conservative fallback
                'avg_win': 0.015,
                'avg_loss': 0.01,
                'total_trades': 10
            }
    
    async def evaluate_signal_async(self, signal: Signal, current_price: float) -> UnifiedRiskDecision:
        """Async version of signal evaluation for pipeline integration"""
        return await self.calculate_position_size(signal, current_price)
    
    def can_trade_sync(self) -> bool:
        """Synchronous check if trading is allowed (DD guard)"""
        current_equity = self.dd_guard.current_equity
        dd_state = self.dd_guard.update_equity(current_equity, datetime.now())
        return dd_state.can_trade()
    
    async def health_check(self) -> Dict[str, Any]:
        """Async health check for risk engine status"""
        try:
            portfolio_value = await self._get_portfolio_value()
            trade_stats = await self._get_trade_statistics()
            
            dd_state = self.dd_guard.update_equity(portfolio_value, datetime.now())
            
            return {
                'status': 'healthy',
                'portfolio_value': portfolio_value,
                'can_trade': dd_state.can_trade(),
                'dd_tier': dd_state.tier.value,
                'dd_pct': dd_state.drawdown_pct,
                'kelly_trades': trade_stats['total_trades'],
                'kelly_win_rate': trade_stats['win_rate'],
                'last_decision': self.last_decision.primary_constraint if self.last_decision else 'none'
            }
        except Exception as e:
            logger.error(f"Risk engine health check failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive risk engine statistics"""
        
        kelly_stats = self.kelly_calc.get_statistics()
        atr_stats = self.atr_controller.get_statistics()
        dd_stats = self.dd_guard.get_statistics()
        
        engine_stats = {
            "decision_count": self.decision_count,
            "last_decision_time": self.last_decision.kelly_result.last_updated if self.last_decision else None,
            "config": {
                "kelly_base_fraction": self.config.kelly_base_fraction,
                "atr_target_vol": self.config.atr_target_daily_vol,
                "dd_warning_threshold": self.config.dd_warning_threshold,
                "dd_critical_threshold": self.config.dd_critical_threshold
            }
        }
        
        return {
            "engine": engine_stats,
            "kelly": kelly_stats,
            "atr": atr_stats,
            "drawdown": dd_stats
        }
    
    def hot_reload_config(self, new_config: UnifiedRiskConfig) -> None:
        """Hot-reload configuration (< 1s requirement)"""
        
        logger.info("Hot-reloading risk engine configuration...")
        
        # Update main config
        old_config = self.config
        self.config = new_config
        
        # Update component configs without losing state
        self.kelly_calc.config.base_kelly_fraction = new_config.kelly_base_fraction
        self.kelly_calc.config.ewma_alpha = new_config.kelly_ewma_alpha
        self.kelly_calc.config.min_sample_size = new_config.kelly_min_trades
        
        self.atr_controller.config.target_daily_vol = new_config.atr_target_daily_vol
        self.atr_controller.config.atr_periods = new_config.atr_periods
        self.atr_controller.config.min_position_size = new_config.atr_min_position
        self.atr_controller.config.max_position_size = new_config.atr_max_position
        
        self.dd_guard.config.caution_threshold = new_config.dd_caution_threshold
        self.dd_guard.config.warning_threshold = new_config.dd_warning_threshold
        self.dd_guard.config.critical_threshold = new_config.dd_critical_threshold
        self.dd_guard.config.warning_cooldown_hours = new_config.dd_warning_cooldown_hours
        self.dd_guard.config.critical_cooldown_hours = new_config.dd_critical_cooldown_hours
        
        logger.info(f"Configuration hot-reloaded: "
                   f"Kelly: {old_config.kelly_base_fraction:.2f}→{new_config.kelly_base_fraction:.2f}, "
                   f"ATR: {old_config.atr_target_daily_vol:.3f}→{new_config.atr_target_daily_vol:.3f}, "
                   f"DD: {old_config.dd_warning_threshold:.2f}→{new_config.dd_warning_threshold:.2f}")
    
    def reset_all_state(self, initial_equity: float = 1.0) -> None:
        """Reset all risk component state"""
        
        self.kelly_calc.reset()
        self.atr_controller.reset()
        self.dd_guard.reset_system(initial_equity)
        
        self.last_decision = None
        self.decision_count = 0
        
        logger.info(f"Risk engine state reset with equity: {initial_equity:.4f}")