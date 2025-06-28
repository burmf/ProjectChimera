"""
Risk management configuration
Unified configuration for all risk components
"""

from pydantic import Field

from .base import BaseConfig, ConfigMixin


class RiskConfig(BaseConfig, ConfigMixin):
    """
    Unified Risk Management Configuration

    Design Reference: CLAUDE.md - Risk-Engine Section 6
    Related Classes:
    - UnifiedRiskEngine: Main risk orchestrator
    - DynamicKellyCalculator: Position sizing component
    - ATRTargetController: Volatility-based sizing
    - DDGuardSystem: Drawdown protection
    """

    # Kelly Criterion settings
    kelly_base_fraction: float = Field(
        default=0.5,
        ge=0.1,
        le=1.0,
        description="Base Kelly fraction (½-Kelly by default)",
    )
    kelly_ewma_alpha: float = Field(
        default=0.1, ge=0.01, le=0.5, description="EWMA decay factor for win rate"
    )
    kelly_min_trades: int = Field(
        default=20, ge=5, le=100, description="Minimum trades before using Kelly"
    )

    # ATR Target settings
    atr_target_daily_vol: float = Field(
        default=0.01,
        ge=0.005,
        le=0.05,
        description="Target daily volatility (1% default)",
    )
    atr_periods: int = Field(
        default=14, ge=5, le=50, description="ATR calculation periods"
    )
    atr_min_position: float = Field(
        default=0.01, ge=0.001, le=0.1, description="Minimum position size (1%)"
    )
    atr_max_position: float = Field(
        default=0.20, ge=0.05, le=1.0, description="Maximum position size (20%)"
    )

    # Drawdown Guard settings
    dd_caution_threshold: float = Field(
        default=0.05,
        ge=0.01,
        le=0.15,
        description="Caution threshold (5% - start reducing)",
    )
    dd_warning_threshold: float = Field(
        default=0.10,
        ge=0.05,
        le=0.25,
        description="Warning threshold (10% - major reduction ×0.5)",
    )
    dd_critical_threshold: float = Field(
        default=0.20,
        ge=0.10,
        le=0.50,
        description="Critical threshold (20% - trading halt ×0.0)",
    )
    dd_warning_cooldown_hours: float = Field(
        default=4.0, ge=1.0, le=24.0, description="Cooldown at warning level (hours)"
    )
    dd_critical_cooldown_hours: float = Field(
        default=24.0, ge=4.0, le=168.0, description="Cooldown at critical level (hours)"
    )

    # Portfolio limits
    max_leverage: float = Field(
        default=10.0, ge=1.0, le=100.0, description="Maximum leverage allowed"
    )
    min_confidence: float = Field(
        default=0.3, ge=0.1, le=0.9, description="Minimum confidence to trade"
    )
    max_portfolio_vol: float = Field(
        default=0.02, ge=0.005, le=0.1, description="Maximum portfolio volatility (2%)"
    )

    # Stop loss settings
    global_stop_loss_pct: float = Field(
        default=0.05, ge=0.01, le=0.20, description="Global stop loss percentage"
    )
    trailing_stop_pct: float = Field(
        default=0.02, ge=0.005, le=0.10, description="Trailing stop percentage"
    )

    def validate_thresholds(self) -> None:
        """Validate that drawdown thresholds are properly ordered"""
        if not (
            self.dd_caution_threshold
            < self.dd_warning_threshold
            < self.dd_critical_threshold
        ):
            raise ValueError(
                "Drawdown thresholds must be: caution < warning < critical"
            )

    def validate_kelly_settings(self) -> None:
        """Validate Kelly criterion settings"""
        self.validate_percentage("kelly_base_fraction", self.kelly_base_fraction)
        self.validate_percentage("kelly_ewma_alpha", self.kelly_ewma_alpha)

    def validate_atr_settings(self) -> None:
        """Validate ATR settings"""
        self.validate_percentage("atr_target_daily_vol", self.atr_target_daily_vol)
        self.validate_percentage("atr_min_position", self.atr_min_position)
        self.validate_percentage("atr_max_position", self.atr_max_position)

        if self.atr_min_position >= self.atr_max_position:
            raise ValueError("atr_min_position must be < atr_max_position")

    def model_post_init(self, __context) -> None:
        """Post-initialization validation"""
        self.validate_thresholds()
        self.validate_kelly_settings()
        self.validate_atr_settings()
