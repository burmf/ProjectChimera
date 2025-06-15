"""
Data feed factory for creating exchange-specific feeds
Supports configuration-driven feed selection
"""

from enum import Enum
from typing import Dict, Any, List, Type, Optional
import logging

from .base import AsyncDataFeed
from .protocols import ExchangeAdapter
from .adapters.binance import BinanceAdapter
from .adapters.bybit import BybitAdapter
from ..settings import Settings


logger = logging.getLogger(__name__)


class ExchangeType(Enum):
    """Supported exchange types"""
    BINANCE = "binance"
    BYBIT = "bybit"
    MOCK = "mock"  # For testing


class DataFeedFactory:
    """Factory for creating data feeds"""
    
    # Registry of exchange adapters
    _adapters: Dict[ExchangeType, Type[ExchangeAdapter]] = {
        ExchangeType.BINANCE: BinanceAdapter,
        ExchangeType.BYBIT: BybitAdapter,
    }
    
    @classmethod
    def register_adapter(cls, exchange_type: ExchangeType, adapter_class: Type[ExchangeAdapter]) -> None:
        """Register a new exchange adapter"""
        cls._adapters[exchange_type] = adapter_class
        logger.info(f"Registered adapter for {exchange_type.value}")
    
    @classmethod
    def create_adapter(cls, exchange_type: ExchangeType, config: Dict[str, Any]) -> ExchangeAdapter:
        """Create an exchange adapter"""
        if exchange_type not in cls._adapters:
            raise ValueError(f"Unsupported exchange type: {exchange_type.value}")
        
        adapter_class = cls._adapters[exchange_type]
        return adapter_class(name=exchange_type.value, config=config)
    
    @classmethod
    def create_feed(
        cls,
        exchange_type: ExchangeType,
        symbols: List[str],
        exchange_config: Optional[Dict[str, Any]] = None,
        feed_config: Optional[Dict[str, Any]] = None
    ) -> AsyncDataFeed:
        """Create a complete data feed"""
        
        # Create adapter
        adapter = cls.create_adapter(exchange_type, exchange_config or {})
        
        # Create feed
        feed = AsyncDataFeed(
            adapter=adapter,
            symbols=symbols,
            config=feed_config or {}
        )
        
        logger.info(f"Created {exchange_type.value} data feed for {len(symbols)} symbols")
        return feed


def create_datafeed(
    exchange: str = "binance",
    symbols: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
    settings: Optional[Settings] = None
) -> AsyncDataFeed:
    """
    Convenience function to create data feed from configuration
    
    Args:
        exchange: Exchange name ("binance", "bybit")
        symbols: List of trading symbols
        config: Optional override configuration
        settings: Application settings
    
    Returns:
        Configured AsyncDataFeed instance
    """
    
    # Default symbols
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT"]
    
    # Parse exchange type
    try:
        exchange_type = ExchangeType(exchange.lower())
    except ValueError:
        raise ValueError(f"Unsupported exchange: {exchange}")
    
    # Build configuration
    exchange_config = {}
    feed_config = {}
    
    if config:
        exchange_config = config.get('exchange', {})
        feed_config = config.get('feed', {})
    
    # Apply settings if provided
    if settings:
        exchange_config.update({
            'timeout_seconds': settings.api.timeout_seconds,
            'max_retries': settings.api.max_retries,
            'sandbox': settings.api.bitget_sandbox,  # Use same sandbox setting
        })
        
        feed_config.update({
            'reconnect_attempts': 10,
            'health_check_interval': 30,
            'cache_ttl': 60,
            'enable_orderbook': True,
            'enable_funding': True,
            'orderbook_levels': 20
        })
    
    return DataFeedFactory.create_feed(
        exchange_type=exchange_type,
        symbols=symbols,
        exchange_config=exchange_config,
        feed_config=feed_config
    )


def create_mock_feed(symbols: List[str]) -> AsyncDataFeed:
    """Create a mock feed for testing"""
    from .adapters.mock import MockAdapter
    
    adapter = MockAdapter("mock", {})
    return AsyncDataFeed(
        adapter=adapter,
        symbols=symbols,
        config={'enable_orderbook': True, 'enable_funding': False}
    )