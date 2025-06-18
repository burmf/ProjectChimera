"""
Configuration management with Pydantic Settings
Supports environment variables and YAML configuration files
"""

from typing import List, Optional, Literal, Dict
from pathlib import Path
from pydantic import BaseModel, Field, SecretStr, AnyUrl, field_validator
from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class TradingConfig(BaseModel):
    """Trading strategy configuration"""
    
    # Core trading parameters - unified with project requirements
    leverage_default: float = Field(default=25.0, ge=1.0, le=100.0, description="Default leverage multiplier")
    leverage_max: float = Field(default=75.0, ge=1.0, le=100.0, description="Maximum leverage allowed")
    max_position_pct: float = Field(default=0.3, ge=0.01, le=1.0, description="Maximum position size as % of portfolio")
    position_size_usd: float = Field(default=40000, gt=0, description="Base position size in USD")
    max_positions: int = Field(default=8, ge=1, le=20, description="Maximum simultaneous positions")
    
    # Risk management - aligned with Chimera requirements
    profit_target: float = Field(default=0.006, gt=0, lt=1, description="Profit target percentage")
    stop_loss: float = Field(default=0.002, gt=0, lt=1, description="Stop loss percentage")
    daily_target: float = Field(default=800, gt=0, description="Daily profit target in USD")
    
    # AI parameters
    confidence_threshold: float = Field(default=0.7, ge=0, le=1, description="Minimum AI confidence")
    momentum_threshold: float = Field(default=0.0006, gt=0, description="Momentum signal threshold")
    volatility_min: float = Field(default=0.0008, gt=0, description="Minimum volatility requirement")
    
    # Trading pairs - focused on high-volume crypto
    trading_pairs: List[str] = Field(default=["BTCUSDT", "ETHUSDT", "SOLUSDT"], min_items=1)
    
    # Timing - optimized for scalping
    position_timeout_minutes: int = Field(default=3, ge=1, description="Maximum position duration")
    update_interval_seconds: int = Field(default=1, ge=1, description="System update interval")


class RiskConfig(BaseModel):
    """Risk management configuration - unified with Dyn-Kelly/ATR/DD"""
    
    # Portfolio-level risk limits
    max_portfolio_risk: float = Field(default=0.15, gt=0, le=1, description="Maximum portfolio risk")
    max_daily_loss: float = Field(default=0.05, gt=0, le=1, description="Maximum daily loss percentage")
    max_drawdown: float = Field(default=0.10, gt=0, le=1, description="Maximum drawdown allowed")
    max_correlation: float = Field(default=0.7, ge=0, le=1, description="Maximum position correlation")
    
    # Kelly criterion and sizing
    kelly_fraction: float = Field(default=0.25, ge=0, le=1, description="Kelly criterion fraction")
    target_vol_daily: float = Field(default=0.01, gt=0, description="Target daily volatility (1%)")
    
    # ATR and volatility parameters
    atr_lookback: int = Field(default=14, ge=5, description="ATR calculation lookback periods")
    atr_multiplier: float = Field(default=2.0, gt=0, description="ATR stop multiplier")
    
    # Drawdown guard parameters
    dd_reduction_10pct: float = Field(default=0.5, gt=0, le=1, description="Size reduction at 10% DD")
    dd_pause_20pct: bool = Field(default=True, description="Pause trading at 20% DD")
    dd_pause_duration_hours: int = Field(default=24, ge=1, description="Pause duration in hours")
    
    # VaR and risk metrics
    var_confidence: float = Field(default=0.95, ge=0.9, le=0.99, description="VaR confidence level")
    var_lookback: int = Field(default=50, ge=10, description="VaR calculation lookback periods")


class APIConfig(BaseModel):
    """API configuration - unified for all exchange integrations"""
    
    # Bitget API credentials
    bitget_key: SecretStr = Field(description="Bitget API key")
    bitget_secret: SecretStr = Field(description="Bitget secret key")
    bitget_passphrase: SecretStr = Field(description="Bitget passphrase")
    bitget_sandbox: bool = Field(default=True, description="Use sandbox environment")
    
    # Bitget URLs - configurable for sandbox/production
    bitget_rest_url: str = Field(default="https://api.bitget.com", description="Bitget REST API URL")
    bitget_ws_spot_url: str = Field(default="wss://ws.bitget.com/spot/v1/stream", description="Bitget spot WebSocket URL")
    bitget_ws_mix_url: str = Field(default="wss://ws.bitget.com/mix/v1/stream", description="Bitget mix WebSocket URL")
    
    # Connection settings
    timeout_seconds: int = Field(default=30, ge=5, description="Request timeout")
    max_retries: int = Field(default=3, ge=1, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, gt=0, description="Delay between retries")
    rate_limit_delay: float = Field(default=0.05, ge=0, description="Rate limiting delay (50ms)")
    
    # WebSocket settings
    ws_heartbeat: int = Field(default=30, ge=10, description="WebSocket heartbeat interval")
    ws_reconnect_delay: int = Field(default=5, ge=1, description="WebSocket reconnection delay")
    ws_max_reconnect_attempts: int = Field(default=5, ge=1, description="Max reconnection attempts")
    
    # OpenAI API (if needed)
    openai_api_key: Optional[SecretStr] = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4", description="OpenAI model to use")
    
    def get_bitget_urls(self) -> Dict[str, str]:
        """Get Bitget URLs based on sandbox setting"""
        if self.bitget_sandbox:
            return {
                "rest": "https://api.bitgetapi.com",
                "ws_spot": "wss://ws.bitgetapi.com/spot/v1/stream", 
                "ws_mix": "wss://ws.bitgetapi.com/mix/v1/stream"
            }
        return {
            "rest": self.bitget_rest_url,
            "ws_spot": self.bitget_ws_spot_url,
            "ws_mix": self.bitget_ws_mix_url
        }


class LoggingConfig(BaseModel):
    """Logging configuration"""
    
    level: str = Field(default="INFO", description="Log level")
    format: str = Field(
        default="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        description="Log format"
    )
    rotation: str = Field(default="100 MB", description="Log rotation size")
    retention: str = Field(default="30 days", description="Log retention period")
    
    # Security - sanitize sensitive data
    sanitize_keys: List[str] = Field(
        default=["api_key", "secret", "passphrase", "token", "password"],
        description="Keys to sanitize in logs"
    )


class MonitoringConfig(BaseModel):
    """Monitoring and health check configuration"""
    
    # Health check endpoint
    health_port: int = Field(default=8000, ge=1024, le=65535, description="Health check HTTP port")
    health_path: str = Field(default="/health", description="Health check endpoint path")
    
    # Prometheus metrics
    prometheus_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    prometheus_port: int = Field(default=8001, ge=1024, le=65535, description="Prometheus metrics port")
    prometheus_path: str = Field(default="/metrics", description="Prometheus metrics endpoint")
    
    # Performance monitoring
    track_latency: bool = Field(default=True, description="Track WebSocket latency")
    track_execution_time: bool = Field(default=True, description="Track execution timing")
    
    # Alerting thresholds
    ws_latency_threshold_ms: float = Field(default=100.0, gt=0, description="WebSocket latency alert threshold")
    execution_time_threshold_ms: float = Field(default=50.0, gt=0, description="Execution time alert threshold")


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
        bitget_passphrase=SecretStr(os.getenv("BITGET_PASSPHRASE", "")),
        bitget_sandbox=os.getenv("BITGET_SANDBOX", "true").lower() == "true",
        openai_api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")) if os.getenv("OPENAI_API_KEY") else None
    ))
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    
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
    
    @field_validator("database_url", mode='before')
    @classmethod
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