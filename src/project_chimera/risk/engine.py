"""
Unified Risk Engine that combines Kelly, Drawdown, and ATR sizing
Main compositor for position sizing and leverage decisions
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from ..domains.market import MarketFrame, Signal
from ..settings import Settings
from .atr_size import ATRPositionSizer, ATRSizingResult
from .drawdown import DrawdownManager, DrawdownState, DrawdownTier
from .equity_cache import EquityCache
from .kelly import KellyCalculator, KellyResult

logger = logging.getLogger(__name__)


@dataclass
class RiskDecision:
    """Final risk management decision"""

    position_size_pct: float  # Final position size as % of portfolio
    leverage: float  # Recommended leverage
    kelly_fraction: float  # Kelly-recommended fraction
    atr_fraction: float  # ATR-recommended fraction
    drawdown_multiplier: float  # Drawdown adjustment multiplier
    can_trade: bool  # Whether trading is allowed
    confidence: float  # Overall confidence (0.0 to 1.0)

    # Component results
    kelly_result: KellyResult
    atr_result: ATRSizingResult
    drawdown_state: DrawdownState

    # Risk metrics
    estimated_portfolio_vol: float
    estimated_drawdown_risk: float
    risk_adjusted_return: float

    # Reasoning
    primary_constraint: str  # What limited the position size
    reasoning: str  # Human-readable explanation

    def is_valid(self) -> bool:
        """Check if risk decision is valid for execution"""
        return (
            self.can_trade
            and 0.0 <= self.position_size_pct <= 1.0
            and self.leverage >= 1.0
            and self.confidence >= 0.3
        )


class RiskEngine:
    """
    Unified risk management engine

    Combines:
    - Kelly Criterion for optimal sizing
    - Drawdown management for downside protection
    - ATR-based volatility targeting
    - Dynamic position sizing and leverage

    Features:
    - Multi-factor risk assessment
    - Configuration-driven parameters
    - Real-time equity tracking
    - Comprehensive risk metrics
    """

    def __init__(
        self,
        settings: Settings | None = None,
        equity_cache_file: str | None = "data/equity_cache.json",
    ):
        self.settings = settings

        # Initialize components
        self.kelly_calc = KellyCalculator(
            lookback_trades=settings.risk.kelly_lookback if settings else 100,
            kelly_fraction=settings.risk.kelly_fraction if settings else 0.25,
            min_trades=20,
        )

        self.drawdown_mgr = DrawdownManager(
            warning_threshold=settings.risk.max_drawdown if settings else 0.10,
            critical_threshold=0.20,  # Hard limit
            warning_multiplier=0.5,
            critical_multiplier=0.0,
        )

        self.atr_sizer = ATRPositionSizer(
            target_daily_vol=0.01,  # 1% daily vol target
            max_position_pct=settings.trading.max_position_pct if settings else 0.30,
        )

        self.equity_cache = EquityCache(
            persistence_file=equity_cache_file, max_history_days=365
        )

        # Risk limits
        self.max_leverage = settings.trading.leverage_default * 2 if settings else 10.0
        self.min_confidence = 0.3
        self.max_portfolio_risk = settings.risk.max_portfolio_risk if settings else 0.15

        # State tracking
        self.last_equity_update = datetime.now()
        self.total_positions = 0

    def calculate_position_size(
        self,
        signal: Signal,
        market_data: MarketFrame,
        portfolio_value: float,
        current_positions: dict[str, float] | None = None,
    ) -> RiskDecision:
        """
        Calculate optimal position size considering all risk factors

        Process:
        1. Get Kelly-optimal sizing based on historical performance
        2. Get ATR-based sizing for volatility targeting
        3. Apply drawdown-based position multiplier
        4. Calculate final leverage
        5. Validate against risk limits
        """

        if current_positions is None:
            current_positions = {}

        # Update equity tracking
        self._update_portfolio_tracking(portfolio_value)

        # Get component recommendations
        kelly_result = self.kelly_calc.calculate_kelly()

        atr_result = self.atr_sizer.calculate_position_size(
            current_price=float(signal.price),
            ohlcv_data=market_data.ohlcv_1m or [],
            portfolio_value=portfolio_value,
        )

        drawdown_state = self.drawdown_mgr.current_state

        # Check if trading is allowed
        can_trade = (
            self.drawdown_mgr.can_trade()
            and kelly_result.is_valid()
            and atr_result.is_valid()
        )

        if not can_trade:
            return self._create_no_trade_decision(
                kelly_result,
                atr_result,
                drawdown_state,
                "Trading halted due to risk constraints",
            )

        # Calculate base position size using multiple approaches
        base_sizes = []

        # Kelly approach (if sufficient data)
        if kelly_result.sample_size >= 20 and kelly_result.confidence > 0.4:
            kelly_size = kelly_result.fraction
            base_sizes.append(("kelly", kelly_size, kelly_result.confidence))

        # ATR approach (always available)
        atr_size = atr_result.position_size_pct
        base_sizes.append(("atr", atr_size, atr_result.confidence))

        # Conservative approach (percentage of portfolio)
        conservative_size = 0.05  # 5% base allocation
        base_sizes.append(("conservative", conservative_size, 0.8))

        # Choose primary sizing method based on confidence
        if base_sizes:
            # Weight by confidence and take average
            total_weight = sum(conf for _, _, conf in base_sizes)
            if total_weight > 0:
                weighted_size = (
                    sum(size * conf for _, size, conf in base_sizes) / total_weight
                )
            else:
                weighted_size = min(size for _, size, _ in base_sizes)
        else:
            weighted_size = 0.01  # 1% fallback

        # Apply drawdown multiplier
        drawdown_multiplier = drawdown_state.position_multiplier
        adjusted_size = weighted_size * drawdown_multiplier

        # Apply portfolio-level risk limits
        portfolio_risk_limit = self._calculate_portfolio_risk_limit(current_positions)
        final_size = min(adjusted_size, portfolio_risk_limit)

        # Calculate leverage
        available_margin = 0.95  # Assume 95% of portfolio available for margin
        leverage = self.atr_sizer.calculate_leverage(
            position_size_pct=final_size,
            available_margin=available_margin,
            max_leverage=self.max_leverage,
        )

        # Calculate risk metrics
        estimated_vol = self.atr_sizer.estimate_daily_portfolio_vol(
            position_size_pct=final_size, price_volatility=atr_result.price_volatility
        )

        estimated_dd_risk = self._estimate_drawdown_risk(
            final_size, atr_result.price_volatility
        )

        risk_adjusted_return = (
            signal.confidence * final_size * 0.02  # Assume 2% expected move
        )

        # Determine primary constraint
        primary_constraint = self._identify_primary_constraint(
            weighted_size, adjusted_size, final_size, drawdown_multiplier
        )

        # Calculate overall confidence
        confidence_factors = [
            kelly_result.confidence if kelly_result.sample_size >= 20 else 0.5,
            atr_result.confidence,
            signal.confidence,
            1.0 - drawdown_state.drawdown_pct,  # Less confident in drawdown
        ]
        overall_confidence = sum(confidence_factors) / len(confidence_factors)

        # Generate reasoning
        reasoning = self._generate_reasoning(
            signal,
            kelly_result,
            atr_result,
            drawdown_state,
            final_size,
            primary_constraint,
        )

        return RiskDecision(
            position_size_pct=final_size,
            leverage=leverage,
            kelly_fraction=kelly_result.fraction,
            atr_fraction=atr_result.position_size_pct,
            drawdown_multiplier=drawdown_multiplier,
            can_trade=can_trade,
            confidence=overall_confidence,
            kelly_result=kelly_result,
            atr_result=atr_result,
            drawdown_state=drawdown_state,
            estimated_portfolio_vol=estimated_vol,
            estimated_drawdown_risk=estimated_dd_risk,
            risk_adjusted_return=risk_adjusted_return,
            primary_constraint=primary_constraint,
            reasoning=reasoning,
        )

    def update_trade_result(
        self,
        return_pct: float,
        portfolio_value: float,
        trade_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        """Update all risk components with trade result"""

        if timestamp is None:
            timestamp = datetime.now()

        # Update Kelly calculator
        self.kelly_calc.add_trade_result(return_pct, timestamp)

        # Update equity cache
        pnl = portfolio_value * return_pct
        self.equity_cache.add_equity_point(
            equity=portfolio_value, pnl=pnl, timestamp=timestamp, trade_id=trade_id
        )

        # Update drawdown manager
        self.drawdown_mgr.update_equity(portfolio_value, timestamp)

        logger.info(
            f"Updated trade result: {return_pct:.4f}% return, portfolio: ${portfolio_value:.2f}"
        )

    def _update_portfolio_tracking(self, portfolio_value: float) -> None:
        """Update portfolio-level tracking"""

        now = datetime.now()

        # Update equity tracking (daily frequency)
        if (now - self.last_equity_update).total_seconds() > 3600:  # Every hour
            self.equity_cache.add_equity_point(
                equity=portfolio_value,
                pnl=0.0,  # Mark-to-market update
                timestamp=now,
                notes="portfolio_update",
            )

            self.drawdown_mgr.update_equity(portfolio_value, now)
            self.last_equity_update = now

    def _calculate_portfolio_risk_limit(
        self, current_positions: dict[str, float]
    ) -> float:
        """Calculate maximum allowed position size based on portfolio risk"""

        # Calculate current portfolio risk exposure
        current_exposure = sum(abs(pos) for pos in current_positions.values())

        # Limit total portfolio risk
        remaining_risk_budget = self.max_portfolio_risk - current_exposure

        return max(0.0, remaining_risk_budget)

    def _estimate_drawdown_risk(
        self, position_size: float, price_volatility: float
    ) -> float:
        """Estimate potential drawdown from this position"""

        # Simplified VaR-style calculation
        # Assume 2-sigma move (95% confidence)
        potential_loss = position_size * price_volatility * 2.0

        return potential_loss

    def _identify_primary_constraint(
        self,
        base_size: float,
        drawdown_adjusted: float,
        final_size: float,
        drawdown_multiplier: float,
    ) -> str:
        """Identify what factor most limited the position size"""

        if final_size <= 0:
            return "no_trade_allowed"
        elif drawdown_multiplier < 0.9:
            return "drawdown_limit"
        elif final_size < drawdown_adjusted * 0.9:
            return "portfolio_risk_limit"
        elif base_size > 0.2:
            return "max_position_limit"
        else:
            return "normal_sizing"

    def _generate_reasoning(
        self,
        signal: Signal,
        kelly: KellyResult,
        atr: ATRSizingResult,
        drawdown: DrawdownState,
        final_size: float,
        constraint: str,
    ) -> str:
        """Generate human-readable reasoning for the decision"""

        parts = []

        # Signal assessment
        parts.append(
            f"Signal: {signal.signal_type.value} with {signal.confidence:.2f} confidence"
        )

        # Kelly assessment
        if kelly.sample_size >= 20:
            parts.append(
                f"Kelly: {kelly.fraction:.3f} (win rate: {kelly.win_rate:.2f})"
            )
        else:
            parts.append(f"Kelly: insufficient data ({kelly.sample_size} trades)")

        # ATR assessment
        parts.append(
            f"ATR sizing: {atr.position_size_pct:.3f} (vol: {atr.price_volatility:.3f})"
        )

        # Drawdown impact
        if drawdown.tier != DrawdownTier.NORMAL:
            parts.append(
                f"Drawdown: {drawdown.tier.value} ({drawdown.drawdown_pct:.2f}%), size reduced by {(1-drawdown.position_multiplier)*100:.0f}%"
            )

        # Final decision
        parts.append(
            f"Final: {final_size:.3f} limited by {constraint.replace('_', ' ')}"
        )

        return " | ".join(parts)

    def _create_no_trade_decision(
        self,
        kelly: KellyResult,
        atr: ATRSizingResult,
        drawdown: DrawdownState,
        reason: str,
    ) -> RiskDecision:
        """Create a no-trade decision"""

        return RiskDecision(
            position_size_pct=0.0,
            leverage=1.0,
            kelly_fraction=kelly.fraction,
            atr_fraction=atr.position_size_pct,
            drawdown_multiplier=drawdown.position_multiplier,
            can_trade=False,
            confidence=0.0,
            kelly_result=kelly,
            atr_result=atr,
            drawdown_state=drawdown,
            estimated_portfolio_vol=0.0,
            estimated_drawdown_risk=0.0,
            risk_adjusted_return=0.0,
            primary_constraint="no_trade",
            reasoning=reason,
        )

    def get_risk_metrics(self) -> dict[str, Any]:
        """Get comprehensive risk metrics"""

        kelly_stats = self.kelly_calc.get_statistics()
        drawdown_metrics = self.drawdown_mgr.get_metrics()
        atr_metrics = self.atr_sizer.get_sizing_metrics()
        equity_stats = self.equity_cache.calculate_statistics()

        return {
            "kelly": {
                "total_trades": kelly_stats.get("total_trades", 0),
                "win_rate": kelly_stats.get("win_rate", 0.0),
                "avg_return": kelly_stats.get("avg_return", 0.0),
                "sharpe_estimate": kelly_stats.get("sharpe_estimate", 0.0),
            },
            "drawdown": {
                "current_pct": drawdown_metrics["drawdown_pct"],
                "tier": drawdown_metrics["tier"],
                "can_trade": drawdown_metrics["can_trade"],
                "position_multiplier": drawdown_metrics["position_multiplier"],
            },
            "atr": {
                "target_vol": atr_metrics["target_daily_vol"],
                "current_regime": atr_metrics["vol_regime_multiplier"],
                "vol_history_length": atr_metrics["vol_history_length"],
            },
            "equity": {
                "current": equity_stats.current_equity,
                "peak": equity_stats.peak_equity,
                "total_return": equity_stats.total_return,
                "max_drawdown": equity_stats.max_drawdown,
                "sharpe_ratio": equity_stats.sharpe_ratio,
                "profit_factor": equity_stats.profit_factor,
            },
        }

    def reset_risk_tracking(self, new_portfolio_value: float = 1.0) -> None:
        """Reset all risk tracking (for new trading periods)"""

        self.kelly_calc.clear_history()
        self.drawdown_mgr.reset_peak(new_portfolio_value)
        self.atr_sizer.reset_regime_tracking()
        self.equity_cache.clear_cache()

        # Initialize with starting equity
        self.equity_cache.add_equity_point(
            equity=new_portfolio_value, pnl=0.0, notes="reset"
        )

        logger.info(f"Reset risk tracking with portfolio value: ${new_portfolio_value}")

    def save_state(self) -> None:
        """Save current state to persistence"""
        self.equity_cache.save_to_file()
        logger.info("Saved risk engine state")

    def load_state(self) -> bool:
        """Load state from persistence"""
        success = self.equity_cache.load_from_file()
        if success:
            logger.info("Loaded risk engine state")
        return success
