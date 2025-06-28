"""
Strategy configuration classes
Unified configuration for all trading strategies
"""

from typing import Any

from pydantic import Field

from .base import BaseConfig, ConfigMixin


class StrategyConfig(BaseConfig, ConfigMixin):
    """
    Base Strategy Configuration

    Design Reference: CLAUDE.md - Strategy Modules Section 5
    Related Classes:
    - Strategy: Base strategy class using this config
    - All concrete strategies inherit this configuration
    - UnifiedRiskEngine: Uses confidence and sizing parameters
    """

    # Core strategy settings
    name: str = Field(description="Strategy name")
    confidence: float = Field(
        default=0.7, ge=0, le=1, description="Base confidence level"
    )

    # Position sizing
    target_size: float = Field(
        default=0.05, gt=0, le=1, description="Target position size % of portfolio"
    )
    max_position_size: float = Field(
        default=0.10, gt=0, le=1, description="Maximum position size % of portfolio"
    )

    # Risk management
    stop_loss_pct: float = Field(
        default=2.0, gt=0, le=50.0, description="Stop loss percentage"
    )
    take_profit_pct: float = Field(
        default=2.0, gt=0, le=50.0, description="Take profit percentage"
    )
    max_hold_time_hours: int = Field(
        default=24, ge=1, le=168, description="Maximum hold time in hours"
    )

    # Signal generation
    confidence_base: float = Field(
        default=0.6, ge=0, le=1, description="Base confidence for calculations"
    )
    min_signal_strength: float = Field(
        default=0.5, ge=0, le=1, description="Minimum signal strength to trigger"
    )

    # Market conditions
    min_volume_ratio: float = Field(
        default=0.5, ge=0.1, le=10.0, description="Minimum volume ratio vs average"
    )
    max_spread_bps: int = Field(
        default=50, ge=1, le=1000, description="Maximum spread in basis points"
    )

    # Strategy-specific parameters (flexible)
    params: dict[str, Any] = Field(
        default_factory=dict, description="Strategy-specific parameters"
    )

    def get_param(self, key: str, default: Any = None) -> Any:
        """Get strategy-specific parameter"""
        return self.params.get(key, default)

    def set_param(self, key: str, value: Any) -> None:
        """Set strategy-specific parameter"""
        self.params[key] = value

    def validate_sizing(self) -> None:
        """Validate position sizing parameters"""
        if self.target_size > self.max_position_size:
            raise ValueError("target_size must be <= max_position_size")

        self.validate_percentage("target_size", self.target_size)
        self.validate_percentage("max_position_size", self.max_position_size)

    def validate_risk_params(self) -> None:
        """Validate risk management parameters"""
        self.validate_positive("stop_loss_pct", self.stop_loss_pct)
        self.validate_positive("take_profit_pct", self.take_profit_pct)

    def model_post_init(self, __context) -> None:
        """Post-initialization validation"""
        self.validate_sizing()
        self.validate_risk_params()


class WeekendEffectConfig(StrategyConfig):
    """Weekend Effect Strategy Configuration"""

    name: str = Field(default="weekend_effect")

    # Weekend-specific parameters
    entry_day: str = Field(default="friday", description="Entry day of week")
    entry_hour: int = Field(default=23, ge=0, le=23, description="Entry hour UTC")
    exit_day: str = Field(default="monday", description="Exit day of week")
    exit_hour: int = Field(default=1, ge=0, le=23, description="Exit hour UTC")

    def model_post_init(self, __context) -> None:
        """Post-initialization setup"""
        super().model_post_init(__context)

        # Set weekend-specific defaults
        if not self.params:
            self.params = {
                "entry_day": self.entry_day,
                "entry_hour": self.entry_hour,
                "exit_day": self.exit_day,
                "exit_hour": self.exit_hour,
            }


class VolBreakoutConfig(StrategyConfig):
    """Volume Breakout Strategy Configuration"""

    name: str = Field(default="vol_breakout")

    # Breakout-specific parameters
    bollinger_periods: int = Field(default=20, ge=5, le=50)
    bollinger_std: float = Field(default=2.0, ge=1.0, le=4.0)
    volume_threshold: float = Field(default=1.5, ge=1.0, le=5.0)
    breakout_threshold_pct: float = Field(default=2.0, ge=0.5, le=10.0)

    def model_post_init(self, __context) -> None:
        """Post-initialization setup"""
        super().model_post_init(__context)

        if not self.params:
            self.params = {
                "bollinger_periods": self.bollinger_periods,
                "bollinger_std": self.bollinger_std,
                "volume_threshold": self.volume_threshold,
                "breakout_threshold_pct": self.breakout_threshold_pct,
            }


# Additional strategy configs can be added here as needed
