"""
Configuration management with Pydantic Settings
Clean configuration without hardcoding
"""

from typing import List, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class TradingConfig(BaseModel):
    """Trading strategy configuration"""
    
    # Core trading parameters
    base_leverage: int = Field(default=25, ge=1, le=100, description="Base leverage multiplier")
    max_leverage: int = Field(default=75, ge=1, le=100, description="Maximum leverage allowed")
    position_size_usd: float = Field(default=40000, gt=0, description="Base position size in USD")
    max_positions: int = Field(default=8, ge=1, le=20, description="Maximum simultaneous positions")
    
    # Risk management
    profit_target: float = Field(default=0.006, gt=0, lt=1, description="Profit target percentage")
    stop_loss: float = Field(default=0.002, gt=0, lt=1, description="Stop loss percentage")
    daily_target: float = Field(default=800, gt=0, description="Daily profit target in USD")
    
    # AI parameters
    confidence_threshold: float = Field(default=0.7, ge=0, le=1, description="Minimum AI confidence")
    momentum_threshold: float = Field(default=0.0006, gt=0, description="Momentum signal threshold")
    volatility_min: float = Field(default=0.0008, gt=0, description="Minimum volatility requirement")
    
    # Trading pairs
    trading_pairs: List[str] = Field(default=["BTCUSDT", "ETHUSDT", "SOLUSDT"], min_items=1)
    
    # Timing
    position_timeout_minutes: int = Field(default=3, ge=1, description="Maximum position duration")
    update_interval_seconds: int = Field(default=1, ge=1, description="System update interval")


class RiskConfig(BaseModel):
    """Risk management configuration"""
    
    max_portfolio_risk: float = Field(default=0.15, gt=0, le=1, description="Maximum portfolio risk")
    max_daily_loss: float = Field(default=0.05, gt=0, le=1, description="Maximum daily loss percentage")
    max_drawdown: float = Field(default=0.10, gt=0, le=1, description="Maximum drawdown allowed")
    max_correlation: float = Field(default=0.7, ge=0, le=1, description="Maximum position correlation")
    kelly_fraction: float = Field(default=0.25, ge=0, le=1, description="Kelly criterion fraction")
    
    var_confidence: float = Field(default=0.95, ge=0.9, le=0.99, description="VaR confidence level")
    var_lookback: int = Field(default=50, ge=10, description="VaR calculation lookback periods")


class APIConfig(BaseModel):
    """API configuration"""
    
    bitget_api_key: str = Field(description="Bitget API key")
    bitget_secret_key: str = Field(description="Bitget secret key")
    bitget_passphrase: str = Field(description="Bitget passphrase")
    bitget_sandbox: bool = Field(default=True, description="Use sandbox environment")
    
    # Connection settings
    timeout_seconds: int = Field(default=30, ge=5, description="Request timeout")
    max_retries: int = Field(default=3, ge=1, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, gt=0, description="Delay between retries")
    rate_limit_delay: float = Field(default=0.1, ge=0, description="Rate limiting delay")
    
    # WebSocket settings
    ws_heartbeat: int = Field(default=30, ge=10, description="WebSocket heartbeat interval")
    ws_reconnect_delay: int = Field(default=5, ge=1, description="WebSocket reconnection delay")


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


class Settings(BaseSettings):
    """
    Main application settings with environment variable support
    """
    
    # Environment
    environment: str = Field(default="development", description="Environment: development/production")
    debug: bool = Field(default=False, description="Debug mode")
    
    # Sub-configurations
    trading: TradingConfig = Field(default_factory=TradingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    # Paths
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    logs_dir: Path = Field(default=Path("logs"), description="Logs directory")
    config_dir: Path = Field(default=Path("config"), description="Config directory")
    
    model_config = {
        "env_file": ".env",
        "env_nested_delimiter": "__", 
        "case_sensitive": False,
        "extra": "ignore"
    }
    
    def __init__(self, **kwargs):
        # Manually load .env file if not loaded
        from dotenv import load_dotenv
        load_dotenv()
        
        # Load API config from environment variables
        api_config = APIConfig(
            bitget_api_key=os.getenv("BITGET_API_KEY", ""),
            bitget_secret_key=os.getenv("BITGET_SECRET_KEY", ""),
            bitget_passphrase=os.getenv("BITGET_PASSPHRASE", ""),
            bitget_sandbox=os.getenv("BITGET_SANDBOX", "true").lower() == "true"
        )
        kwargs.setdefault("api", api_config)
        super().__init__(**kwargs)


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    Thread-safe singleton pattern
    """
    return Settings()


def load_config_from_yaml(yaml_path: str) -> Settings:
    """
    Load configuration from YAML file
    Useful for production deployments
    """
    import yaml
    
    with open(yaml_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    return Settings(**config_data)


# Development configuration template
DEVELOPMENT_CONFIG = {
    "environment": "development",
    "debug": True,
    "trading": {
        "base_leverage": 10,  # Conservative for development
        "position_size_usd": 1000,  # Small size for testing
        "daily_target": 50,
        "max_positions": 3
    },
    "api": {
        "bitget_sandbox": True,
        "timeout_seconds": 10,
        "max_retries": 2
    },
    "logging": {
        "level": "DEBUG"
    }
}

# Production configuration template
PRODUCTION_CONFIG = {
    "environment": "production",
    "debug": False,
    "trading": {
        "base_leverage": 25,
        "position_size_usd": 40000,
        "daily_target": 800,
        "max_positions": 8
    },
    "api": {
        "bitget_sandbox": False,
        "timeout_seconds": 30,
        "max_retries": 5
    },
    "logging": {
        "level": "INFO"
    }
}


def create_config_template(config_type: str = "development") -> None:
    """Create configuration template file"""
    import yaml
    
    config = DEVELOPMENT_CONFIG if config_type == "development" else PRODUCTION_CONFIG
    
    config_path = Path(f"config/{config_type}.yaml")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Created {config_path}")


if __name__ == "__main__":
    # Create configuration templates
    create_config_template("development")
    create_config_template("production")
    
    # Test configuration loading
    settings = get_settings()
    print(f"✅ Configuration loaded: {settings.environment}")
    print(f"Trading pairs: {settings.trading.trading_pairs}")
    print(f"Base leverage: {settings.trading.base_leverage}x")