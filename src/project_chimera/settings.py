"""
Configuration management with Pydantic Settings
Supports environment variables and YAML configuration files
"""

from typing import List, Optional, Literal
from pathlib import Path
from pydantic import BaseModel, Field, SecretStr, AnyUrl, validator
from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class TradingConfig(BaseModel):
    """Trading strategy configuration"""
    
    # Core trading parameters
    leverage_default: float = Field(default=3.0, ge=1.0, le=100.0, description="Default leverage multiplier")
    max_position_pct: float = Field(default=0.3, ge=0.01, le=1.0, description="Maximum position size as % of portfolio")
    position_size_usd: float = Field(default=1000, gt=0, description="Base position size in USD")
    max_positions: int = Field(default=5, ge=1, le=20, description="Maximum simultaneous positions")
    
    # Risk management
    profit_target: float = Field(default=0.02, gt=0, lt=1, description="Profit target percentage")
    stop_loss: float = Field(default=0.01, gt=0, lt=1, description="Stop loss percentage")
    daily_target: float = Field(default=100, gt=0, description="Daily profit target in USD")
    
    # Trading pairs
    trading_pairs: List[str] = Field(default=["BTCUSDT", "ETHUSDT"], min_items=1)
    
    # Timing
    position_timeout_minutes: int = Field(default=60, ge=1, description="Maximum position duration")
    update_interval_seconds: int = Field(default=5, ge=1, description="System update interval")


class RiskConfig(BaseModel):
    """Risk management configuration"""
    
    max_portfolio_risk: float = Field(default=0.15, gt=0, le=1, description="Maximum portfolio risk")
    max_daily_loss: float = Field(default=0.05, gt=0, le=1, description="Maximum daily loss percentage")
    max_drawdown: float = Field(default=0.10, gt=0, le=1, description="Maximum drawdown allowed")
    kelly_fraction: float = Field(default=0.25, ge=0, le=1, description="Kelly criterion fraction")
    
    var_confidence: float = Field(default=0.95, ge=0.9, le=0.99, description="VaR confidence level")
    var_lookback: int = Field(default=50, ge=10, description="VaR calculation lookback periods")


class APIConfig(BaseModel):
    """API configuration"""
    
    bitget_key: SecretStr = Field(description="Bitget API key")
    bitget_secret: SecretStr = Field(description="Bitget secret key")
    bitget_passphrase: SecretStr = Field(description="Bitget passphrase")
    bitget_sandbox: bool = Field(default=True, description="Use sandbox environment")
    
    # Connection settings
    timeout_seconds: int = Field(default=30, ge=5, description="Request timeout")
    max_retries: int = Field(default=3, ge=1, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, gt=0, description="Delay between retries")
    
    # WebSocket settings
    ws_heartbeat: int = Field(default=30, ge=10, description="WebSocket heartbeat interval")
    ws_reconnect_delay: int = Field(default=5, ge=1, description="WebSocket reconnection delay")


class Settings(BaseSettings):
    """
    Main application settings with environment variable support
    """
    
    # Environment
    env: Literal["dev", "prod"] = Field(default="dev", description="Environment: dev/prod")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Database
    database_url: AnyUrl = Field(
        default="postgresql+asyncpg://chimera:chimera@localhost:5432/chimera",
        description="Database connection URL"
    )
    
    # Sub-configurations
    trading: TradingConfig = Field(default_factory=TradingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    api: APIConfig = Field(default_factory=lambda: APIConfig(
        bitget_key=SecretStr(os.getenv("BITGET_API_KEY", "")),
        bitget_secret=SecretStr(os.getenv("BITGET_SECRET_KEY", "")),
        bitget_passphrase=SecretStr(os.getenv("BITGET_PASSPHRASE", ""))
    ))
    
    # Paths
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    logs_dir: Path = Field(default=Path("logs"), description="Logs directory")
    
    model_config = {
        "env_file": ".env",
        "env_nested_delimiter": "__", 
        "case_sensitive": False,
        "extra": "ignore",
        "yaml_file": "config.{env}.yaml"
    }
    
    @validator("database_url", pre=True)
    def validate_database_url(cls, v):
        if isinstance(v, str) and "{env}" in v:
            env = os.getenv("ENV", "dev")
            return v.format(env=env)
        return v


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    Thread-safe singleton pattern
    """
    return Settings()