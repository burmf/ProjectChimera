"""
Mock exchange adapter for testing
Simulates market data without external dependencies
"""

import asyncio
import random
from datetime import datetime, timedelta
from decimal import Decimal
from typing import AsyncIterator, Optional, Dict, Any, List

from ..protocols import ExchangeAdapter, ConnectionStatus
from ...domains.market import Ticker, OrderBook, OHLCV, FundingRate


class MockAdapter(ExchangeAdapter):
    """
    Mock exchange adapter for testing and development
    
    Features:
    - Simulated market data generation
    - Configurable price movements
    - No external dependencies
    - Instant connection
    """
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        
        # Configuration
        self.base_prices = config.get('base_prices', {
            'BTCUSDT': 45000.0,
            'ETHUSDT': 3000.0
        })
        self.volatility = config.get('volatility', 0.02)  # 2% volatility
        self.trend = config.get('trend', 0.0)  # No trend by default
        
        # State
        self.current_prices = self.base_prices.copy()
        self.running = False
    
    async def connect(self) -> None:
        """Instant connection for mock adapter"""
        self.status = ConnectionStatus.CONNECTED
        self.running = True
        self.reset_error_count()
    
    async def disconnect(self) -> None:
        """Disconnect mock adapter"""
        self.status = ConnectionStatus.DISCONNECTED
        self.running = False
    
    async def subscribe_ticker(self, symbol: str) -> None:
        """Mock ticker subscription"""
        if symbol not in self.current_prices:
            self.current_prices[symbol] = 45000.0  # Default price
    
    async def subscribe_orderbook(self, symbol: str, levels: int = 20) -> None:
        """Mock orderbook subscription"""
        if symbol not in self.current_prices:
            self.current_prices[symbol] = 45000.0
    
    async def subscribe_klines(self, symbol: str, interval: str = "1m") -> None:
        """Mock kline subscription"""
        if symbol not in self.current_prices:
            self.current_prices[symbol] = 45000.0
    
    async def subscribe_funding(self, symbol: str) -> None:
        """Mock funding subscription"""
        pass  # Funding rates are optional
    
    def _update_price(self, symbol: str) -> float:
        """Update price with random walk"""
        if symbol not in self.current_prices:
            return 45000.0
        
        current = self.current_prices[symbol]
        
        # Random walk with trend
        change_pct = random.gauss(self.trend, self.volatility)
        new_price = current * (1 + change_pct)
        
        # Ensure price stays positive
        new_price = max(new_price, current * 0.5)
        
        self.current_prices[symbol] = new_price
        return new_price
    
    def _generate_orderbook(self, symbol: str, levels: int = 20) -> OrderBook:
        """Generate mock orderbook"""
        mid_price = self.current_prices.get(symbol, 45000.0)
        spread_pct = 0.0001  # 0.01% spread
        spread = mid_price * spread_pct
        
        bids = []
        asks = []
        
        for i in range(levels):
            # Bids below mid price
            bid_price = Decimal(str(mid_price - spread/2 - i))
            bid_qty = Decimal(str(random.uniform(0.1, 5.0)))
            bids.append((bid_price, bid_qty))
            
            # Asks above mid price
            ask_price = Decimal(str(mid_price + spread/2 + i))
            ask_qty = Decimal(str(random.uniform(0.1, 5.0)))
            asks.append((ask_price, ask_qty))
        
        return OrderBook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=datetime.now()
        )
    
    def _generate_ticker(self, symbol: str) -> Ticker:
        """Generate mock ticker"""
        price = self._update_price(symbol)
        
        return Ticker(
            symbol=symbol,
            price=Decimal(str(price)),
            volume_24h=Decimal(str(random.uniform(10000, 100000))),
            change_24h=Decimal(str(random.uniform(-5.0, 5.0))),
            timestamp=datetime.now()
        )
    
    def _generate_ohlcv(self, symbol: str, timeframe: str = "1m") -> OHLCV:
        """Generate mock OHLCV"""
        base_price = self.current_prices.get(symbol, 45000.0)
        
        # Generate OHLCV with some randomness
        open_price = base_price * random.uniform(0.995, 1.005)
        close_price = open_price * random.uniform(0.99, 1.01)
        high_price = max(open_price, close_price) * random.uniform(1.0, 1.005)
        low_price = min(open_price, close_price) * random.uniform(0.995, 1.0)
        volume = random.uniform(100, 1000)
        
        return OHLCV(
            symbol=symbol,
            open=Decimal(str(round(open_price, 2))),
            high=Decimal(str(round(high_price, 2))),
            low=Decimal(str(round(low_price, 2))),
            close=Decimal(str(round(close_price, 2))),
            volume=Decimal(str(round(volume, 2))),
            timestamp=datetime.now(),
            timeframe=timeframe
        )
    
    async def get_ticker(self, symbol: str) -> Optional[Ticker]:
        """Get mock ticker"""
        return self._generate_ticker(symbol)
    
    async def get_orderbook(self, symbol: str, levels: int = 20) -> Optional[OrderBook]:
        """Get mock orderbook"""
        return self._generate_orderbook(symbol, levels)
    
    async def get_historical_klines(
        self, 
        symbol: str, 
        interval: str = "1m", 
        limit: int = 100
    ) -> List[OHLCV]:
        """Generate mock historical klines"""
        klines = []
        base_time = datetime.now() - timedelta(minutes=limit)
        
        for i in range(limit):
            timestamp = base_time + timedelta(minutes=i)
            ohlcv = self._generate_ohlcv(symbol, interval)
            ohlcv = OHLCV(
                symbol=ohlcv.symbol,
                open=ohlcv.open,
                high=ohlcv.high,
                low=ohlcv.low,
                close=ohlcv.close,
                volume=ohlcv.volume,
                timestamp=timestamp,
                timeframe=interval
            )
            klines.append(ohlcv)
        
        return klines
    
    async def get_funding_rate(self, symbol: str) -> Optional[FundingRate]:
        """Generate mock funding rate"""
        return FundingRate(
            symbol=symbol,
            rate=Decimal(str(random.uniform(-0.01, 0.01))),  # ±1% funding
            next_funding_time=datetime.now() + timedelta(hours=8),
            timestamp=datetime.now()
        )
    
    def is_connected(self) -> bool:
        """Mock is always connected when running"""
        return self.running and self.status == ConnectionStatus.CONNECTED
    
    def get_status(self) -> ConnectionStatus:
        """Get mock status"""
        return self.status
    
    async def health_check(self) -> bool:
        """Mock health check always passes"""
        return self.running
    
    # Streaming protocol implementation
    async def stream_ticker(self, symbol: str) -> AsyncIterator[Ticker]:
        """Stream mock ticker updates"""
        while self.is_connected():
            yield self._generate_ticker(symbol)
            await asyncio.sleep(1)
    
    async def stream_orderbook(self, symbol: str) -> AsyncIterator[OrderBook]:
        """Stream mock orderbook updates"""
        while self.is_connected():
            yield self._generate_orderbook(symbol)
            await asyncio.sleep(1)
    
    async def stream_klines(self, symbol: str, interval: str = "1m") -> AsyncIterator[OHLCV]:
        """Stream mock kline updates"""
        while self.is_connected():
            yield self._generate_ohlcv(symbol, interval)
            await asyncio.sleep(60)  # 1 minute intervals
    
    async def stream_funding(self, symbol: str) -> AsyncIterator[FundingRate]:
        """Stream mock funding updates"""
        while self.is_connected():
            funding = await self.get_funding_rate(symbol)
            if funding:
                yield funding
            await asyncio.sleep(3600)  # 1 hour intervals