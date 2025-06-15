"""
Data feed layer for real-time market data
Provides unified async interface for multiple exchanges
"""

from .factory import create_datafeed, ExchangeType
from .base import AsyncDataFeed, FeedStatus
from .protocols import MarketDataProtocol

__all__ = [
    "create_datafeed", "ExchangeType", 
    "AsyncDataFeed", "FeedStatus",
    "MarketDataProtocol"
]