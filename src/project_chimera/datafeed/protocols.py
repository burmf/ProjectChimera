"""
Data feed protocols and interfaces
Defines contracts for market data providers
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

from ..domains.market import MarketFrame, Ticker, OrderBook, OHLCV, FundingRate


class ConnectionStatus(Enum):
    """WebSocket connection status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


class MarketDataProtocol(ABC):
    """Protocol for market data providers"""
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to data source"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to data source"""
        pass
    
    @abstractmethod
    async def subscribe_ticker(self, symbol: str) -> None:
        """Subscribe to ticker updates"""
        pass
    
    @abstractmethod
    async def subscribe_orderbook(self, symbol: str, levels: int = 20) -> None:
        """Subscribe to order book updates"""
        pass
    
    @abstractmethod
    async def subscribe_klines(self, symbol: str, interval: str = "1m") -> None:
        """Subscribe to kline/candlestick updates"""
        pass
    
    @abstractmethod
    async def subscribe_funding(self, symbol: str) -> None:
        """Subscribe to funding rate updates"""
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Optional[Ticker]:
        """Get current ticker data"""
        pass
    
    @abstractmethod
    async def get_orderbook(self, symbol: str, levels: int = 20) -> Optional[OrderBook]:
        """Get current order book"""
        pass
    
    @abstractmethod
    async def get_historical_klines(
        self, 
        symbol: str, 
        interval: str = "1m", 
        limit: int = 100
    ) -> List[OHLCV]:
        """Get historical kline data"""
        pass
    
    @abstractmethod
    async def get_funding_rate(self, symbol: str) -> Optional[FundingRate]:
        """Get current funding rate"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connection is active"""
        pass
    
    @abstractmethod
    def get_status(self) -> ConnectionStatus:
        """Get current connection status"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Perform health check"""
        pass


class StreamingProtocol(ABC):
    """Protocol for streaming market data"""
    
    @abstractmethod
    async def stream_ticker(self, symbol: str) -> AsyncIterator[Ticker]:
        """Stream ticker updates"""
        pass
    
    @abstractmethod
    async def stream_orderbook(self, symbol: str) -> AsyncIterator[OrderBook]:
        """Stream order book updates"""
        pass
    
    @abstractmethod
    async def stream_klines(self, symbol: str, interval: str = "1m") -> AsyncIterator[OHLCV]:
        """Stream kline updates"""
        pass
    
    @abstractmethod
    async def stream_funding(self, symbol: str) -> AsyncIterator[FundingRate]:
        """Stream funding rate updates"""
        pass


class ExchangeAdapter(MarketDataProtocol, StreamingProtocol):
    """Base class combining market data and streaming protocols"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.status = ConnectionStatus.DISCONNECTED
        self.last_heartbeat: Optional[datetime] = None
        self.error_count = 0
        self.max_errors = config.get('max_errors', 5)
    
    async def ping(self) -> bool:
        """Send ping to check connection"""
        try:
            result = await self.health_check()
            if result:
                self.last_heartbeat = datetime.now()
                self.error_count = 0
            else:
                self.error_count += 1
            return result
        except Exception:
            self.error_count += 1
            return False
    
    def should_reconnect(self) -> bool:
        """Check if reconnection is needed"""
        return (
            self.status in [ConnectionStatus.DISCONNECTED, ConnectionStatus.FAILED] or
            self.error_count >= self.max_errors
        )
    
    def reset_error_count(self) -> None:
        """Reset error counter after successful operation"""
        self.error_count = 0