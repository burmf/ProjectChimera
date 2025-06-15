"""
Base async data feed implementation
Core functionality for unified market data access
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Set
from enum import Enum
import logging

from tenacity import (
    retry, stop_after_attempt, wait_exponential, 
    retry_if_exception_type, before_sleep_log
)

from .protocols import ExchangeAdapter, ConnectionStatus
from ..domains.market import MarketFrame, Ticker, OrderBook, OHLCV, FundingRate


logger = logging.getLogger(__name__)


class FeedStatus(Enum):
    """Feed operational status"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    DEGRADED = "degraded"  # Some symbols failing
    STOPPED = "stopped"
    ERROR = "error"


class DataCache:
    """Thread-safe cache for market data"""
    
    def __init__(self, ttl_seconds: int = 60):
        self.ttl_seconds = ttl_seconds
        self._data: Dict[str, Dict[str, Any]] = {}
        self._timestamps: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str, data_type: str) -> Optional[Any]:
        """Get cached data if not expired"""
        async with self._lock:
            cache_key = f"{key}:{data_type}"
            
            if cache_key not in self._data:
                return None
            
            timestamp = self._timestamps.get(cache_key)
            if timestamp and datetime.now() - timestamp > timedelta(seconds=self.ttl_seconds):
                # Data expired
                del self._data[cache_key]
                del self._timestamps[cache_key]
                return None
            
            return self._data[cache_key]
    
    async def set(self, key: str, data_type: str, value: Any) -> None:
        """Cache data with timestamp"""
        async with self._lock:
            cache_key = f"{key}:{data_type}"
            self._data[cache_key] = value
            self._timestamps[cache_key] = datetime.now()
    
    async def clear(self) -> None:
        """Clear all cached data"""
        async with self._lock:
            self._data.clear()
            self._timestamps.clear()


class AsyncDataFeed:
    """
    Unified async data feed for real-time market data
    
    Features:
    - Automatic reconnection with exponential backoff
    - Data caching and deduplication
    - Health monitoring and metrics
    - Multiple symbol support
    - Graceful error handling
    """
    
    def __init__(
        self,
        adapter: ExchangeAdapter,
        symbols: List[str],
        config: Optional[Dict[str, Any]] = None
    ):
        self.adapter = adapter
        self.symbols = set(symbols)
        self.config = config or {}
        
        # Configuration
        self.reconnect_attempts = self.config.get('reconnect_attempts', 10)
        self.health_check_interval = self.config.get('health_check_interval', 30)
        self.cache_ttl = self.config.get('cache_ttl', 60)
        self.enable_orderbook = self.config.get('enable_orderbook', True)
        self.enable_funding = self.config.get('enable_funding', True)
        self.orderbook_levels = self.config.get('orderbook_levels', 20)
        
        # State
        self.status = FeedStatus.INITIALIZING
        self.subscribed_symbols: Set[str] = set()
        self.cache = DataCache(ttl_seconds=self.cache_ttl)
        self.metrics = {
            'messages_received': 0,
            'reconnections': 0,
            'errors': 0,
            'last_update': None,
            'latency_samples': []
        }
        
        # Tasks
        self._health_task: Optional[asyncio.Task] = None
        self._stream_tasks: List[asyncio.Task] = []
        self._running = False
    
    async def start(self) -> None:
        """Start the data feed"""
        if self._running:
            logger.warning("Data feed already running")
            return
        
        logger.info(f"Starting data feed for {len(self.symbols)} symbols")
        self._running = True
        self.status = FeedStatus.INITIALIZING
        
        try:
            # Connect adapter
            await self._connect_with_retry()
            
            # Subscribe to symbols
            await self._subscribe_symbols()
            
            # Start health monitoring
            self._health_task = asyncio.create_task(self._health_monitor())
            
            # Start streaming tasks
            await self._start_streaming()
            
            self.status = FeedStatus.RUNNING
            logger.info("Data feed started successfully")
            
        except Exception as e:
            self.status = FeedStatus.ERROR
            logger.error(f"Failed to start data feed: {e}")
            raise
    
    async def stop(self) -> None:
        """Stop the data feed"""
        if not self._running:
            return
        
        logger.info("Stopping data feed")
        self._running = False
        self.status = FeedStatus.STOPPED
        
        # Cancel tasks
        if self._health_task:
            self._health_task.cancel()
        
        for task in self._stream_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._health_task:
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        
        for task in self._stream_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Disconnect adapter
        await self.adapter.disconnect()
        
        # Clear cache
        await self.cache.clear()
        
        logger.info("Data feed stopped")
    
    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type(Exception),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    async def _connect_with_retry(self) -> None:
        """Connect adapter with retry logic"""
        await self.adapter.connect()
        self.metrics['reconnections'] += 1
    
    async def _subscribe_symbols(self) -> None:
        """Subscribe to all configured symbols"""
        for symbol in self.symbols:
            try:
                # Always subscribe to ticker and klines
                await self.adapter.subscribe_ticker(symbol)
                await self.adapter.subscribe_klines(symbol, "1m")
                
                # Optional subscriptions
                if self.enable_orderbook:
                    await self.adapter.subscribe_orderbook(symbol, self.orderbook_levels)
                
                if self.enable_funding:
                    await self.adapter.subscribe_funding(symbol)
                
                self.subscribed_symbols.add(symbol)
                logger.debug(f"Subscribed to {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to subscribe to {symbol}: {e}")
                self.metrics['errors'] += 1
    
    async def _start_streaming(self) -> None:
        """Start streaming tasks for all symbols"""
        for symbol in self.subscribed_symbols:
            # Ticker stream
            task = asyncio.create_task(self._stream_ticker(symbol))
            self._stream_tasks.append(task)
            
            # OrderBook stream (if enabled)
            if self.enable_orderbook:
                task = asyncio.create_task(self._stream_orderbook(symbol))
                self._stream_tasks.append(task)
    
    async def _stream_ticker(self, symbol: str) -> None:
        """Stream ticker data for a symbol"""
        try:
            async for ticker in self.adapter.stream_ticker(symbol):
                await self.cache.set(symbol, "ticker", ticker)
                self.metrics['messages_received'] += 1
                self.metrics['last_update'] = datetime.now()
                
                # Calculate latency
                if ticker.timestamp:
                    latency = (datetime.now() - ticker.timestamp).total_seconds() * 1000
                    self.metrics['latency_samples'].append(latency)
                    
                    # Keep only last 100 samples
                    if len(self.metrics['latency_samples']) > 100:
                        self.metrics['latency_samples'] = self.metrics['latency_samples'][-100:]
        
        except Exception as e:
            logger.error(f"Ticker stream error for {symbol}: {e}")
            self.metrics['errors'] += 1
    
    async def _stream_orderbook(self, symbol: str) -> None:
        """Stream orderbook data for a symbol"""
        try:
            async for orderbook in self.adapter.stream_orderbook(symbol):
                await self.cache.set(symbol, "orderbook", orderbook)
                self.metrics['messages_received'] += 1
        
        except Exception as e:
            logger.error(f"OrderBook stream error for {symbol}: {e}")
            self.metrics['errors'] += 1
    
    async def _health_monitor(self) -> None:
        """Monitor feed health and trigger reconnections if needed"""
        while self._running:
            try:
                await asyncio.sleep(self.health_check_interval)
                
                if not self.adapter.is_connected():
                    logger.warning("Adapter disconnected, attempting reconnection")
                    await self._reconnect()
                
                # Check if we're receiving data
                if self.metrics['last_update']:
                    time_since_update = datetime.now() - self.metrics['last_update']
                    if time_since_update > timedelta(seconds=60):
                        logger.warning("No data received for 60 seconds, checking health")
                        if not await self.adapter.health_check():
                            await self._reconnect()
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                self.metrics['errors'] += 1
    
    async def _reconnect(self) -> None:
        """Reconnect the adapter"""
        try:
            self.status = FeedStatus.DEGRADED
            await self.adapter.disconnect()
            await self._connect_with_retry()
            await self._subscribe_symbols()
            self.status = FeedStatus.RUNNING
            logger.info("Reconnection successful")
        
        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            self.status = FeedStatus.ERROR
            self.metrics['errors'] += 1
    
    async def snapshot(self, symbol: str) -> Optional[MarketFrame]:
        """
        Get current market snapshot for a symbol
        
        Returns MarketFrame with latest cached data
        """
        if symbol not in self.subscribed_symbols:
            logger.warning(f"Symbol {symbol} not subscribed")
            return None
        
        # Get cached data
        ticker = await self.cache.get(symbol, "ticker")
        orderbook = await self.cache.get(symbol, "orderbook") if self.enable_orderbook else None
        
        # Get historical klines
        try:
            ohlcv_1m = await self.adapter.get_historical_klines(symbol, "1m", 100)
        except Exception as e:
            logger.warning(f"Failed to get historical data for {symbol}: {e}")
            ohlcv_1m = []
        
        # Get funding rate if enabled
        funding_rate = None
        if self.enable_funding:
            try:
                funding_rate = await self.adapter.get_funding_rate(symbol)
            except Exception:
                pass  # Funding rate is optional
        
        return MarketFrame(
            symbol=symbol,
            timestamp=datetime.now(),
            ticker=ticker,
            orderbook=orderbook,
            ohlcv_1m=ohlcv_1m[-100:] if ohlcv_1m else None,  # Last 100 candles
            funding_rate=funding_rate
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get feed performance metrics"""
        latency_samples = self.metrics['latency_samples']
        
        return {
            'status': self.status.value,
            'adapter_status': self.adapter.get_status().value,
            'symbols_subscribed': len(self.subscribed_symbols),
            'messages_received': self.metrics['messages_received'],
            'reconnections': self.metrics['reconnections'],
            'errors': self.metrics['errors'],
            'last_update': self.metrics['last_update'],
            'latency_median_ms': sorted(latency_samples)[len(latency_samples)//2] if latency_samples else None,
            'latency_p95_ms': sorted(latency_samples)[int(len(latency_samples)*0.95)] if latency_samples else None,
        }
    
    def add_symbol(self, symbol: str) -> None:
        """Add symbol to subscription list"""
        self.symbols.add(symbol)
    
    def remove_symbol(self, symbol: str) -> None:
        """Remove symbol from subscription list"""
        self.symbols.discard(symbol)
        self.subscribed_symbols.discard(symbol)