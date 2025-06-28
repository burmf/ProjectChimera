"""
Exchange configuration classes
Consolidated Bitget and other exchange configurations
"""

from pydantic import Field, SecretStr

from .base import BaseConfig, ConfigMixin


class BitgetConfig(BaseConfig, ConfigMixin):
    """
    Unified Bitget API configuration

    Design Reference: CLAUDE.md - Configuration consolidation
    Related Classes:
    - Replaces multiple Bitget config classes
    - Used by execution engine and data feeds
    - Provides unified API authentication and endpoints
    """

    # API Authentication
    api_key: SecretStr = Field(default=SecretStr(""), description="Bitget API key")
    secret_key: SecretStr = Field(
        default=SecretStr(""), description="Bitget secret key"
    )
    passphrase: SecretStr = Field(
        default=SecretStr(""), description="Bitget passphrase"
    )

    # Environment settings
    sandbox: bool = Field(default=True, description="Use sandbox environment")

    # API Endpoints
    rest_base_url: str = Field(
        default="https://api.bitget.com", description="REST API base URL"
    )
    spot_ws_url: str = Field(
        default="wss://ws.bitget.com/spot/v1/stream", description="Spot WebSocket URL"
    )
    mix_ws_url: str = Field(
        default="wss://ws.bitget.com/mix/v1/stream", description="Mix WebSocket URL"
    )

    # Connection settings
    timeout_seconds: int = Field(
        default=30, ge=5, le=300, description="Request timeout"
    )
    max_retries: int = Field(
        default=3, ge=1, le=10, description="Maximum retry attempts"
    )
    retry_delay: float = Field(
        default=1.0, ge=0.1, le=10.0, description="Retry delay seconds"
    )
    max_connections: int = Field(
        default=20, ge=1, le=100, description="Max HTTP connections"
    )
    max_keepalive_connections: int = Field(
        default=10, ge=1, le=50, description="Max keepalive connections"
    )

    # WebSocket settings
    reconnect_delay: int = Field(
        default=5, ge=1, le=60, description="WebSocket reconnect delay"
    )
    max_reconnect_attempts: int = Field(
        default=5, ge=1, le=20, description="Max reconnect attempts"
    )
    heartbeat_interval: int = Field(
        default=30, ge=10, le=300, description="Heartbeat interval"
    )

    # Rate limiting
    min_request_interval: float = Field(
        default=0.1, ge=0.01, le=1.0, description="Min interval between requests"
    )
    requests_per_second: int = Field(
        default=10, ge=1, le=100, description="Max requests per second"
    )

    # Circuit breaker settings
    failure_threshold: int = Field(
        default=3, ge=1, le=10, description="Circuit breaker failure threshold"
    )
    pause_duration_seconds: float = Field(
        default=300.0, ge=60.0, le=1800.0, description="Circuit breaker pause duration"
    )

    def __post_init__(self):
        """Post-initialization validation and setup"""
        if self.sandbox:
            # Bitget uses same URLs for sandbox, just different API keys
            pass

    def get_headers(self) -> dict:
        """Get common headers for API requests"""
        return {
            "Content-Type": "application/json",
            "User-Agent": "ProjectChimera/1.0",
        }

    def is_configured(self) -> bool:
        """Check if API credentials are configured"""
        return (
            bool(self.api_key.get_secret_value())
            and bool(self.secret_key.get_secret_value())
            and bool(self.passphrase.get_secret_value())
        )


class ExchangeConfig(BaseConfig, ConfigMixin):
    """
    Multi-exchange configuration

    Design Reference: CLAUDE.md - Extensible exchange support
    Related Classes:
    - Container for all exchange configurations
    - Currently focused on Bitget but extensible
    """

    # Primary exchange
    primary_exchange: str = Field(
        default="bitget", description="Primary exchange for trading"
    )

    # Exchange-specific configs
    bitget: BitgetConfig = Field(
        default_factory=BitgetConfig, description="Bitget configuration"
    )

    # Future exchange support
    # binance: BinanceConfig = Field(default_factory=BinanceConfig)
    # bybit: BybitConfig = Field(default_factory=BybitConfig)

    def get_exchange_config(self, exchange_name: str) -> BaseConfig | None:
        """Get configuration for specific exchange"""
        return getattr(self, exchange_name, None)

    def get_primary_config(self) -> BaseConfig:
        """Get configuration for primary exchange"""
        return self.get_exchange_config(self.primary_exchange)
