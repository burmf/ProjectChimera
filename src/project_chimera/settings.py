"""
Configuration management with Pydantic Settings
Supports environment variables and YAML configuration files
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import AnyUrl, BaseModel, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class StrategyConfig(BaseModel):
    """Individual strategy configuration"""

    enabled: bool = Field(default=True, description="Enable/disable strategy")
    confidence: float = Field(
        default=0.7, ge=0, le=1, description="Base confidence level"
    )
    target_size: float = Field(
        default=0.05, gt=0, le=1, description="Target position size %"
    )
    stop_loss_pct: float = Field(default=2.0, gt=0, description="Stop loss percentage")
    take_profit_pct: float = Field(
        default=2.0, gt=0, description="Take profit percentage"
    )
    confidence_base: float = Field(
        default=0.6, ge=0, le=1, description="Base confidence for calculations"
    )
    # Allow arbitrary fields for strategy-specific parameters
    model_config = {"extra": "allow"}


class StrategiesConfig(BaseModel):
    """All strategy configurations"""

    weekend_effect: StrategyConfig = Field(default_factory=StrategyConfig)
    stop_reversion: StrategyConfig = Field(default_factory=StrategyConfig)
    funding_contrarian: StrategyConfig = Field(default_factory=StrategyConfig)
    volume_breakout: StrategyConfig = Field(default_factory=StrategyConfig)
    cme_gap: StrategyConfig = Field(default_factory=StrategyConfig)
    basis_arbitrage: StrategyConfig = Field(default_factory=StrategyConfig)
    lob_reversion: StrategyConfig = Field(default_factory=StrategyConfig)

    def get_strategy_config(self, strategy_name: str) -> StrategyConfig:
        """Get configuration for a specific strategy"""
        return getattr(self, strategy_name, StrategyConfig())


class ExchangeAdapterConfig(BaseModel):
    """Exchange adapter configuration"""

    spot_ws_url: str = Field(default="wss://ws.bitget.com/spot/v1/stream")
    mix_ws_url: str = Field(default="wss://ws.bitget.com/mix/v1/stream")
    rest_base_url: str = Field(default="https://api.bitget.com")
    demo_rest_url: str = Field(default="https://api.bitgetapi.com")
    timeout_seconds: int = Field(default=30, ge=5)
    max_connections: int = Field(default=20, ge=1)
    max_keepalive_connections: int = Field(default=10, ge=1)
    reconnect_delay: int = Field(default=5, ge=1)
    max_reconnect_attempts: int = Field(default=5, ge=1)
    heartbeat_interval: int = Field(default=30, ge=10)


class ExchangeAdaptersConfig(BaseModel):
    """All exchange adapter configurations"""

    bitget: ExchangeAdapterConfig = Field(default_factory=ExchangeAdapterConfig)


class AIConfig(BaseModel):
    """AI Decision Engine configuration for ProjectChimera 4-layer system"""

    # OpenAI API settings
    openai_api_key: SecretStr = Field(
        default=SecretStr(""), description="OpenAI API key"
    )
    openai_model: str = Field(
        default="o3-mini", description="OpenAI model to use (o3-mini/o3)"
    )
    openai_base_url: str | None = Field(
        default=None, description="Custom OpenAI API base URL"
    )

    # Decision intervals
    decision_1min_interval_seconds: int = Field(
        default=60, ge=30, le=300, description="1-minute decision interval"
    )
    strategy_1hour_interval_seconds: int = Field(
        default=3600, ge=1800, le=7200, description="1-hour strategy interval"
    )

    # Model parameters
    max_tokens: int = Field(
        default=1000, ge=100, le=4000, description="Maximum tokens per request"
    )
    temperature: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Model temperature for consistency"
    )

    # Risk and confidence thresholds
    min_confidence_threshold: float = Field(
        default=0.6, ge=0.3, le=0.9, description="Minimum confidence to trade"
    )
    max_position_size_pct: float = Field(
        default=0.05, gt=0, le=0.20, description="Maximum AI position size %"
    )

    # Cost management
    max_daily_api_cost_usd: float = Field(
        default=50.0, gt=0, description="Maximum daily API cost in USD"
    )
    cost_tracking_window_hours: int = Field(
        default=24, ge=12, le=48, description="Cost tracking window"
    )

    # Performance settings
    enable_1min_decisions: bool = Field(
        default=True, description="Enable 1-minute trading decisions"
    )
    enable_1hour_strategy: bool = Field(
        default=True, description="Enable 1-hour strategy planning"
    )
    parallel_processing: bool = Field(
        default=False, description="Enable parallel symbol processing"
    )


class NewsCollectorConfig(BaseModel):
    """News RSS collector configuration"""

    # Collection settings
    enabled: bool = Field(default=True, description="Enable news collection")
    collection_interval_hours: int = Field(
        default=1, ge=1, le=12, description="News collection interval"
    )

    # News sources priorities
    coindesk_priority: int = Field(
        default=1, ge=1, le=3, description="CoinDesk priority (1=high)"
    )
    cointelegraph_priority: int = Field(
        default=1, ge=1, le=3, description="CoinTelegraph priority"
    )
    reuters_priority: int = Field(default=1, ge=1, le=3, description="Reuters priority")

    # Filtering settings
    min_relevance_score: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Minimum relevance score"
    )
    max_articles_per_source: int = Field(
        default=20, ge=5, le=100, description="Max articles per source"
    )

    # Request settings
    request_timeout_seconds: int = Field(
        default=30, ge=10, le=120, description="HTTP request timeout"
    )
    retry_attempts: int = Field(
        default=3, ge=1, le=5, description="Number of retry attempts"
    )
    delay_between_sources_seconds: float = Field(
        default=1.0, ge=0.5, le=5.0, description="Delay between source requests"
    )


class XPostsCollectorConfig(BaseModel):
    """X/Twitter posts collector configuration"""

    # Collection settings
    enabled: bool = Field(default=True, description="Enable X posts collection")
    collection_interval_hours: int = Field(
        default=1, ge=1, le=6, description="X posts collection interval"
    )

    # Query settings
    max_results_per_query: int = Field(
        default=50, ge=10, le=200, description="Max results per search query"
    )
    query_timeout_seconds: int = Field(
        default=60, ge=30, le=300, description="Query timeout"
    )

    # Filtering settings
    min_engagement_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Minimum engagement score"
    )
    min_relevance_score: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Minimum relevance score"
    )

    # Rate limiting
    delay_between_queries_seconds: float = Field(
        default=2.0, ge=1.0, le=10.0, description="Delay between queries"
    )
    delay_between_accounts_seconds: float = Field(
        default=3.0, ge=1.0, le=10.0, description="Delay between account queries"
    )

    # Advanced settings
    enable_influential_accounts: bool = Field(
        default=True, description="Enable influential accounts monitoring"
    )
    max_influential_accounts_per_cycle: int = Field(
        default=3, ge=1, le=10, description="Max accounts per cycle"
    )


class RedisStreamsConfig(BaseModel):
    """Redis Streams configuration for data pipeline"""

    # Connection settings
    redis_url: str = Field(
        default="redis://localhost:6379", description="Redis connection URL"
    )
    redis_db: int = Field(default=0, ge=0, le=15, description="Redis database number")

    # Stream settings
    max_stream_length: int = Field(
        default=10000, ge=1000, le=100000, description="Maximum stream length"
    )
    consumer_block_ms: int = Field(
        default=1000, ge=100, le=10000, description="Consumer block timeout"
    )
    consumer_count: int = Field(
        default=10, ge=1, le=100, description="Messages per consumer read"
    )

    # Stream names
    market_data_stream: str = Field(
        default="market_data", description="Market data stream name"
    )
    news_stream: str = Field(default="news", description="News stream name")
    x_posts_stream: str = Field(default="x_posts", description="X posts stream name")
    ai_decisions_stream: str = Field(
        default="ai_decisions", description="AI decisions stream name"
    )
    executions_stream: str = Field(
        default="executions", description="Executions stream name"
    )

    # Consumer groups
    ai_processors_group: str = Field(
        default="ai_processors", description="AI processors consumer group"
    )
    execution_group: str = Field(
        default="execution", description="Execution consumer group"
    )
    logging_group: str = Field(default="logging", description="Logging consumer group")


class LayerSystemConfig(BaseModel):
    """Configuration for the 4-layer trading system"""

    # System settings
    enabled: bool = Field(default=True, description="Enable 4-layer system")
    initial_portfolio_value: float = Field(
        default=150000.0, gt=0, description="Initial portfolio value USD"
    )
    trading_symbols: list[str] = Field(
        default=["BTCUSDT", "ETHUSDT"], description="List of trading symbols"
    )

    # Simplified risk settings for orchestrator's decisions
    risk_multiplier: float = Field(
        default=0.5, ge=0.1, le=1.0, description="Conservative risk multiplier for AI decisions"
    )
    max_risk_adjusted_size_pct: float = Field(
        default=0.02, gt=0, le=0.1, description="Max risk-adjusted size percentage cap"
    )
    default_trade_side: str = Field(
        default="buy", description="Default trade side for mock execution"
    )

    # Layer configurations
    ai: AIConfig = Field(default_factory=AIConfig)
    news_collector: NewsCollectorConfig = Field(default_factory=NewsCollectorConfig)
    x_posts_collector: XPostsCollectorConfig = Field(
        default_factory=XPostsCollectorConfig
    )
    redis_streams: RedisStreamsConfig = Field(default_factory=RedisStreamsConfig)

    # Inter-layer settings
    enable_learning_data_storage: bool = Field(
        default=True, description="Enable AI learning data storage"
    )
    health_check_interval_seconds: int = Field(
        default=30, ge=10, le=300, description="Health check interval"
    )

    # Performance settings
    enable_performance_monitoring: bool = Field(
        default=True, description="Enable performance monitoring"
    )
    log_performance_interval_minutes: int = Field(
        default=60, ge=15, le=240, description="Performance logging interval"
    )


class TradingConfig(BaseModel):
    """Trading strategy configuration"""

    # Core trading parameters - unified with project requirements
    leverage_default: float = Field(
        default=25.0, ge=1.0, le=100.0, description="Default leverage multiplier"
    )
    leverage_max: float = Field(
        default=75.0, ge=1.0, le=100.0, description="Maximum leverage allowed"
    )
    max_position_pct: float = Field(
        default=0.3,
        ge=0.01,
        le=1.0,
        description="Maximum position size as % of portfolio",
    )
    position_size_usd: float = Field(
        default=40000, gt=0, description="Base position size in USD"
    )
    max_positions: int = Field(
        default=8, ge=1, le=20, description="Maximum simultaneous positions"
    )

    # Risk management - aligned with Chimera requirements
    profit_target: float = Field(
        default=0.006, gt=0, lt=1, description="Profit target percentage"
    )
    stop_loss: float = Field(
        default=0.002, gt=0, lt=1, description="Stop loss percentage"
    )
    daily_target: float = Field(
        default=800, gt=0, description="Daily profit target in USD"
    )

    # AI parameters
    confidence_threshold: float = Field(
        default=0.7, ge=0, le=1, description="Minimum AI confidence"
    )
    momentum_threshold: float = Field(
        default=0.0006, gt=0, description="Momentum signal threshold"
    )
    volatility_min: float = Field(
        default=0.0008, gt=0, description="Minimum volatility requirement"
    )

    # Trading pairs - focused on high-volume crypto
    trading_pairs: list[str] = Field(
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT"], min_items=1
    )

    # Timing - optimized for scalping
    position_timeout_minutes: int = Field(
        default=3, ge=1, description="Maximum position duration"
    )
    update_interval_seconds: int = Field(
        default=1, ge=1, description="System update interval"
    )


class RiskConfig(BaseModel):
    """Risk management configuration - unified with Dyn-Kelly/ATR/DD"""

    # Portfolio-level risk limits
    max_portfolio_risk: float = Field(
        default=0.15, gt=0, le=1, description="Maximum portfolio risk"
    )
    max_daily_loss: float = Field(
        default=0.05, gt=0, le=1, description="Maximum daily loss percentage"
    )
    max_drawdown: float = Field(
        default=0.10, gt=0, le=1, description="Maximum drawdown allowed"
    )
    max_correlation: float = Field(
        default=0.7, ge=0, le=1, description="Maximum position correlation"
    )

    # Kelly criterion and sizing
    kelly_fraction: float = Field(
        default=0.25, ge=0, le=1, description="Kelly criterion fraction"
    )
    kelly_base_fraction: float = Field(
        default=0.5, ge=0, le=1, description="Kelly base fraction (½-Kelly)"
    )
    kelly_ewma_alpha: float = Field(
        default=0.1, gt=0, le=1, description="EWMA decay for Kelly"
    )
    kelly_min_trades: int = Field(
        default=20, ge=1, description="Min trades for Kelly calculation"
    )
    target_vol_daily: float = Field(
        default=0.01, gt=0, description="Target daily volatility (1%)"
    )

    # ATR and volatility parameters
    atr_target_daily_vol: float = Field(
        default=0.01, gt=0, description="ATR target daily volatility"
    )
    atr_periods: int = Field(default=14, ge=5, description="ATR calculation periods")
    atr_lookback: int = Field(
        default=14, ge=5, description="ATR calculation lookback periods"
    )
    atr_multiplier: float = Field(default=2.0, gt=0, description="ATR stop multiplier")
    atr_min_position: float = Field(
        default=0.01, gt=0, le=1, description="ATR min position size"
    )
    atr_max_position: float = Field(
        default=0.20, gt=0, le=1, description="ATR max position size"
    )

    # Drawdown guard parameters
    dd_caution_threshold: float = Field(
        default=0.05, gt=0, le=1, description="DD caution threshold (5%)"
    )
    dd_warning_threshold: float = Field(
        default=0.10, gt=0, le=1, description="DD warning threshold (10%)"
    )
    dd_critical_threshold: float = Field(
        default=0.20, gt=0, le=1, description="DD critical threshold (20%)"
    )
    dd_reduction_10pct: float = Field(
        default=0.5, gt=0, le=1, description="Size reduction at 10% DD"
    )
    dd_pause_20pct: bool = Field(default=True, description="Pause trading at 20% DD")
    dd_pause_duration_hours: int = Field(
        default=24, ge=1, description="Pause duration in hours"
    )
    dd_warning_cooldown_hours: float = Field(
        default=4.0, gt=0, description="Warning cooldown hours"
    )
    dd_critical_cooldown_hours: float = Field(
        default=24.0, gt=0, description="Critical cooldown hours"
    )

    # Engine limits
    max_leverage: float = Field(
        default=10.0, gt=1, description="Maximum leverage allowed"
    )
    min_confidence: float = Field(
        default=0.3, ge=0, le=1, description="Minimum confidence to trade"
    )
    max_portfolio_vol: float = Field(
        default=0.02, gt=0, description="Max portfolio volatility"
    )

    # VaR and risk metrics
    var_confidence: float = Field(
        default=0.95, ge=0.9, le=0.99, description="VaR confidence level"
    )
    var_lookback: int = Field(
        default=50, ge=10, description="VaR calculation lookback periods"
    )


class APIConfig(BaseModel):
    """API configuration - unified for all exchange integrations"""

    # Bitget API credentials
    bitget_key: SecretStr = Field(default=SecretStr(""), description="Bitget API key")
    bitget_secret: SecretStr = Field(
        default=SecretStr(""), description="Bitget secret key"
    )
    bitget_passphrase: SecretStr = Field(
        default=SecretStr(""), description="Bitget passphrase"
    )
    bitget_sandbox: bool = Field(default=True, description="Use sandbox environment")

    # Bitget URLs - configurable for sandbox/production
    bitget_rest_url: str = Field(
        default="https://api.bitget.com", description="Bitget REST API URL"
    )
    bitget_ws_spot_url: str = Field(
        default="wss://ws.bitget.com/spot/v1/stream",
        description="Bitget spot WebSocket URL",
    )
    bitget_ws_mix_url: str = Field(
        default="wss://ws.bitget.com/mix/v1/stream",
        description="Bitget mix WebSocket URL",
    )

    # Connection settings
    timeout_seconds: int = Field(default=30, ge=5, description="Request timeout")
    max_retries: int = Field(default=3, ge=1, description="Maximum retry attempts")
    retry_delay: float = Field(default=1.0, gt=0, description="Delay between retries")
    rate_limit_delay: float = Field(
        default=0.05, ge=0, description="Rate limiting delay (50ms)"
    )

    # WebSocket settings
    ws_heartbeat: int = Field(
        default=30, ge=10, description="WebSocket heartbeat interval"
    )
    ws_reconnect_delay: int = Field(
        default=5, ge=1, description="WebSocket reconnection delay"
    )
    ws_max_reconnect_attempts: int = Field(
        default=5, ge=1, description="Max reconnection attempts"
    )

    # OpenAI API (if needed)
    openai_api_key: SecretStr = Field(
        default=SecretStr(""), description="OpenAI API key"
    )
    openai_model: str = Field(default="gpt-4", description="OpenAI model to use")

    def get_bitget_urls(self) -> dict[str, str]:
        """Get Bitget URLs based on sandbox setting"""
        if self.bitget_sandbox:
            return {
                "rest": "https://api.bitgetapi.com",
                "ws_spot": "wss://ws.bitgetapi.com/spot/v1/stream",
                "ws_mix": "wss://ws.bitgetapi.com/mix/v1/stream",
            }
        return {
            "rest": self.bitget_rest_url,
            "ws_spot": self.bitget_ws_spot_url,
            "ws_mix": self.bitget_ws_mix_url,
        }


class LoggingConfig(BaseModel):
    """Logging configuration"""

    level: str = Field(default="INFO", description="Log level")
    format: str = Field(
        default="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}",
        description="Log format",
    )
    rotation: str = Field(default="100 MB", description="Log rotation size")
    retention: str = Field(default="30 days", description="Log retention period")

    # Security - sanitize sensitive data
    sanitize_keys: list[str] = Field(
        default=["api_key", "secret", "passphrase", "token", "password"],
        description="Keys to sanitize in logs",
    )


class MonitoringConfig(BaseModel):
    """Monitoring and health check configuration"""

    # Health check endpoint
    health_port: int = Field(
        default=8000, ge=1024, le=65535, description="Health check HTTP port"
    )
    health_path: str = Field(
        default="/health", description="Health check endpoint path"
    )

    # Prometheus metrics
    prometheus_enabled: bool = Field(
        default=True, description="Enable Prometheus metrics"
    )
    prometheus_port: int = Field(
        default=8001, ge=1024, le=65535, description="Prometheus metrics port"
    )
    prometheus_path: str = Field(
        default="/metrics", description="Prometheus metrics endpoint"
    )

    # Performance monitoring
    track_latency: bool = Field(default=True, description="Track WebSocket latency")
    track_execution_time: bool = Field(
        default=True, description="Track execution timing"
    )

    # Alerting thresholds
    ws_latency_threshold_ms: float = Field(
        default=100.0, gt=0, description="WebSocket latency alert threshold"
    )
    execution_time_threshold_ms: float = Field(
        default=50.0, gt=0, description="Execution time alert threshold"
    )


class Settings(BaseSettings):
    """
    Main application settings with environment variable support
    """

    # Environment
    env: Literal["dev", "prod"] = Field(
        default="dev", description="Environment: dev/prod"
    )
    debug: bool = Field(default=False, description="Debug mode")

    # Database
    database_url: AnyUrl = Field(
        default="postgresql+asyncpg://chimera:chimera@localhost:5432/chimera",
        description="Database connection URL",
    )

    # Sub-configurations
    trading: TradingConfig = Field(default_factory=TradingConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    api: APIConfig = Field(
        default_factory=lambda: APIConfig(
            bitget_key=SecretStr(os.getenv("BITGET_API_KEY", "")),
            bitget_secret=SecretStr(os.getenv("BITGET_SECRET_KEY", "")),
            bitget_passphrase=SecretStr(os.getenv("BITGET_PASSPHRASE", "")),
            bitget_sandbox=os.getenv("BITGET_SANDBOX", "true").lower() == "true",
            openai_api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")),
        )
    )
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    strategies: StrategiesConfig = Field(default_factory=StrategiesConfig)
    exchange_adapters: ExchangeAdaptersConfig = Field(
        default_factory=ExchangeAdaptersConfig
    )

    # ProjectChimera 4-Layer System Configuration
    layer_system: LayerSystemConfig = Field(
        default_factory=lambda: LayerSystemConfig(
            ai=AIConfig(
                openai_api_key=SecretStr(os.getenv("OPENAI_API_KEY", "")),
                openai_model=os.getenv("OPENAI_MODEL", "o3-mini"),
                max_daily_api_cost_usd=float(
                    os.getenv("OPENAI_MAX_DAILY_COST", "50.0")
                ),
            )
        )
    )

    # Paths
    data_dir: Path = Field(default=Path("data"), description="Data directory")
    logs_dir: Path = Field(default=Path("logs"), description="Logs directory")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
        yaml_file="config.{env}.yaml",
        yaml_file_encoding="utf-8",
    )

    @field_validator("database_url", mode="before")
    @classmethod
    def validate_database_url(cls, v):
        if isinstance(v, str) and "{env}" in v:
            env = os.getenv("ENV", "dev")
            return v.format(env=env)
        return v


def load_yaml_config(env: str = "dev") -> dict[str, Any]:
    """
    Load YAML configuration file based on environment
    """
    config_file = Path(f"config.{env}.yaml")
    if not config_file.exists():
        # Try alternative locations
        alt_paths = [
            Path(f"../config.{env}.yaml"),
            Path(f"../../config.{env}.yaml"),
            Path(f"/app/config.{env}.yaml"),
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                config_file = alt_path
                break
        else:
            # Return empty dict if no config file found
            return {}

    try:
        with open(config_file, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"Warning: Could not load {config_file}: {e}")
        return {}


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance
    Thread-safe singleton pattern with YAML config support
    """
    env = os.getenv("ENV", "dev")
    yaml_config = load_yaml_config(env)

    # Merge YAML config with environment variables
    return Settings(**yaml_config)


def get_strategy_config(strategy_name: str) -> StrategyConfig:
    """
    Get configuration for a specific strategy
    """
    settings = get_settings()
    return settings.strategies.get_strategy_config(strategy_name)


def get_exchange_config(exchange_name: str = "bitget") -> ExchangeAdapterConfig:
    """
    Get configuration for exchange adapter
    """
    settings = get_settings()
    return getattr(settings.exchange_adapters, exchange_name, ExchangeAdapterConfig())
