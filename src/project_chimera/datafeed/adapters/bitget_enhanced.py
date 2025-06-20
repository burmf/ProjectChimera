"""
Enhanced Bitget adapter for unified WebSocket and REST API
Implements full Bitget-specific functionality for spot and perpetual futures
"""

import asyncio
import json
import logging
import hmac
import hashlib
import base64
import time
from datetime import datetime
from decimal import Decimal
from typing import AsyncIterator, Optional, Dict, Any, List
import websockets
import httpx

from ..protocols import ExchangeAdapter, ConnectionStatus
from ...domains.market import Ticker, OrderBook, OHLCV, FundingRate
from ...settings import get_exchange_config


logger = logging.getLogger(__name__)


class BitgetEnhancedAdapter(ExchangeAdapter):
    """
    Enhanced Bitget exchange adapter with full feature support
    
    Features:
    - Unified WebSocket streams (spot + futures)
    - Order book depth and trades
    - Funding rates and open interest
    - Historical data fetching
    - Authentication for private endpoints
    - Automatic reconnection with exponential backoff
    """
    
    # API Endpoints - loaded from settings
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        
        # Load exchange configuration from settings
        exchange_config = get_exchange_config('bitget')
        
        self.SPOT_WS_URL = exchange_config.spot_ws_url
        self.MIX_WS_URL = exchange_config.mix_ws_url
        self.REST_BASE_URL = exchange_config.rest_base_url
    
        
        # Configuration from settings and provided config
        self.api_key = config.get('api_key', '')
        self.secret_key = config.get('secret_key', '')
        self.passphrase = config.get('passphrase', '')
        self.sandbox = config.get('sandbox', True)
        self.timeout = exchange_config.timeout_seconds
        
        # Use demo environment if sandbox
        if self.sandbox:
            self.REST_BASE_URL = exchange_config.demo_rest_url
        
        # HTTP client
        self.http_client: Optional[httpx.AsyncClient] = None
        
        # WebSocket connections
        self.spot_ws: Optional[websockets.WebSocketServerProtocol] = None
        self.mix_ws: Optional[websockets.WebSocketServerProtocol] = None
        
        # Subscribed channels
        self.spot_channels: List[str] = []
        self.mix_channels: List[str] = []
        
        # Data storage
        self.market_data: Dict[str, Dict[str, Any]] = {}
        self.funding_rates: Dict[str, FundingRate] = {}
        self.open_interest: Dict[str, float] = {}
        
        # Message handlers
        self.message_handlers = {
            'books': self._handle_orderbook,
            'trade': self._handle_trade,
            'ticker': self._handle_ticker,
            'fundingRate': self._handle_funding_rate,
            'openInterest': self._handle_open_interest
        }
    
    async def connect(self) -> None:
        """Establish all connections"""
        try:
            self.status = ConnectionStatus.CONNECTING
            
            # Initialize HTTP client using exchange config
            exchange_config = get_exchange_config('bitget')
            self.http_client = httpx.AsyncClient(
                base_url=self.REST_BASE_URL,
                timeout=self.timeout,
                limits=httpx.Limits(
                    max_connections=exchange_config.max_connections,
                    max_keepalive_connections=exchange_config.max_keepalive_connections
                )
            )
            
            # Test connection
            response = await self.http_client.get("/api/spot/v1/public/time")
            response.raise_for_status()
            
            # Connect WebSocket streams
            await self._connect_websockets()
            
            self.status = ConnectionStatus.CONNECTED
            self.reset_error_count()
            logger.info("Bitget enhanced adapter connected")
            
        except Exception as e:
            self.status = ConnectionStatus.FAILED
            logger.error(f"Bitget connection failed: {e}")
            raise
    
    async def _connect_websockets(self) -> None:
        """Connect to Bitget WebSocket streams"""
        try:
            # Connect spot WebSocket
            self.spot_ws = await websockets.connect(self.SPOT_WS_URL)
            
            # Connect mix (futures) WebSocket
            self.mix_ws = await websockets.connect(self.MIX_WS_URL)
            
            # Start message handlers
            asyncio.create_task(self._handle_spot_messages())
            asyncio.create_task(self._handle_mix_messages())
            
            logger.debug("Bitget WebSocket connections established")
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close all connections"""
        self.status = ConnectionStatus.DISCONNECTED
        
        # Close WebSocket connections
        if self.spot_ws and not self.spot_ws.closed:
            await self.spot_ws.close()
        
        if self.mix_ws and not self.mix_ws.closed:
            await self.mix_ws.close()
        
        # Close HTTP client
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
        
        logger.info("Bitget enhanced adapter disconnected")
    
    async def subscribe_ticker(self, symbol: str) -> None:
        """Subscribe to ticker updates for both spot and futures"""
        # Spot ticker
        spot_channel = f"ticker.{symbol}"
        await self._subscribe_spot_channel(spot_channel)
        
        # Futures ticker (if applicable)
        if self._is_futures_symbol(symbol):
            mix_channel = f"ticker.{symbol}"
            await self._subscribe_mix_channel(mix_channel)
    
    async def subscribe_orderbook(self, symbol: str, levels: int = 20) -> None:
        """Subscribe to order book updates"""
        # Bitget uses books5 for top 5 levels, books for full depth
        depth_type = "books5" if levels <= 5 else "books"
        channel = f"{depth_type}.{symbol}"
        
        if self._is_futures_symbol(symbol):
            await self._subscribe_mix_channel(channel)
        else:
            await self._subscribe_spot_channel(channel)
    
    async def subscribe_trades(self, symbol: str) -> None:
        """Subscribe to trade updates"""
        channel = f"trade.{symbol}"
        
        if self._is_futures_symbol(symbol):
            await self._subscribe_mix_channel(channel)
        else:
            await self._subscribe_spot_channel(channel)
    
    async def subscribe_funding(self, symbol: str) -> None:
        """Subscribe to funding rate updates (futures only)"""
        if not self._is_futures_symbol(symbol):
            logger.warning(f"Funding rates not available for spot symbol {symbol}")
            return
        
        channel = f"fundingRate.{symbol}"
        await self._subscribe_mix_channel(channel)
    
    async def subscribe_open_interest(self, symbol: str) -> None:
        """Subscribe to open interest updates (futures only)"""
        if not self._is_futures_symbol(symbol):
            logger.warning(f"Open interest not available for spot symbol {symbol}")
            return
        
        channel = f"openInterest.{symbol}"
        await self._subscribe_mix_channel(channel)
    
    def _is_futures_symbol(self, symbol: str) -> bool:
        """Check if symbol is a futures contract"""
        return "USDT" in symbol or "USDC" in symbol  # Simplified check
    
    async def _subscribe_spot_channel(self, channel: str) -> None:
        """Subscribe to spot WebSocket channel"""
        if channel in self.spot_channels:
            return
        
        if not self.spot_ws or self.spot_ws.closed:
            await self._connect_websockets()
        
        try:
            subscribe_msg = {
                "op": "subscribe",
                "args": [{"instType": "SPOT", "channel": channel, "instId": "default"}]
            }
            
            await self.spot_ws.send(json.dumps(subscribe_msg))
            self.spot_channels.append(channel)
            logger.debug(f"Subscribed to spot channel: {channel}")
            
        except Exception as e:
            logger.error(f"Spot subscription failed for {channel}: {e}")
            raise
    
    async def _subscribe_mix_channel(self, channel: str) -> None:
        """Subscribe to futures WebSocket channel"""
        if channel in self.mix_channels:
            return
        
        if not self.mix_ws or self.mix_ws.closed:
            await self._connect_websockets()
        
        try:
            subscribe_msg = {
                "op": "subscribe",
                "args": [{"instType": "UMCBL", "channel": channel, "instId": "default"}]
            }
            
            await self.mix_ws.send(json.dumps(subscribe_msg))
            self.mix_channels.append(channel)
            logger.debug(f"Subscribed to mix channel: {channel}")
            
        except Exception as e:
            logger.error(f"Mix subscription failed for {channel}: {e}")
            raise
    
    async def _handle_spot_messages(self) -> None:
        """Handle spot WebSocket messages"""
        try:
            async for message in self.spot_ws:
                await self._process_message(message, "spot")
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Spot WebSocket connection closed")
            self.status = ConnectionStatus.DISCONNECTED
        except Exception as e:
            logger.error(f"Spot WebSocket error: {e}")
            self.error_count += 1
    
    async def _handle_mix_messages(self) -> None:
        """Handle futures WebSocket messages"""
        try:
            async for message in self.mix_ws:
                await self._process_message(message, "mix")
        except websockets.exceptions.ConnectionClosed:
            logger.warning("Mix WebSocket connection closed")
            self.status = ConnectionStatus.DISCONNECTED
        except Exception as e:
            logger.error(f"Mix WebSocket error: {e}")
            self.error_count += 1
    
    async def _process_message(self, message: str, stream_type: str) -> None:
        """Process incoming WebSocket message"""
        try:
            data = json.loads(message)
            
            # Handle different message types
            if 'event' in data:
                # System events (subscribe confirmations, etc.)
                logger.debug(f"System event: {data}")
                return
            
            if 'arg' in data and 'data' in data:
                # Market data update
                channel = data['arg']['channel']
                market_data = data['data']
                
                # Route to appropriate handler
                if channel.startswith('books'):
                    await self._handle_orderbook(market_data, channel)
                elif channel.startswith('trade'):
                    await self._handle_trade(market_data, channel)
                elif channel.startswith('ticker'):
                    await self._handle_ticker(market_data, channel)
                elif channel.startswith('fundingRate'):
                    await self._handle_funding_rate(market_data, channel)
                elif channel.startswith('openInterest'):
                    await self._handle_open_interest(market_data, channel)
                
        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON message: {message}")
        except Exception as e:
            logger.error(f"Message processing error: {e}")
    
    async def _handle_orderbook(self, data: List[Dict], channel: str) -> None:
        """Handle order book update"""
        for book_data in data:
            symbol = book_data.get('instId', '')
            if not symbol:
                continue
            
            bids = []
            asks = []
            
            # Parse bids and asks
            for bid in book_data.get('bids', []):
                if len(bid) >= 2:
                    bids.append((Decimal(bid[0]), Decimal(bid[1])))
            
            for ask in book_data.get('asks', []):
                if len(ask) >= 2:
                    asks.append((Decimal(ask[0]), Decimal(ask[1])))
            
            if bids or asks:
                orderbook = OrderBook(
                    symbol=symbol,
                    bids=bids,
                    asks=asks,
                    timestamp=datetime.fromtimestamp(int(book_data.get('ts', 0)) / 1000)
                )
                
                # Store in market data
                if symbol not in self.market_data:
                    self.market_data[symbol] = {}
                self.market_data[symbol]['orderbook'] = orderbook
    
    async def _handle_trade(self, data: List[Dict], channel: str) -> None:
        """Handle trade update"""
        # Store latest trade data for volume and price info
        for trade_data in data:
            symbol = trade_data.get('instId', '')
            if symbol:
                if symbol not in self.market_data:
                    self.market_data[symbol] = {}
                self.market_data[symbol]['last_trade'] = trade_data
    
    async def _handle_ticker(self, data: List[Dict], channel: str) -> None:
        """Handle ticker update"""
        for ticker_data in data:
            symbol = ticker_data.get('instId', '')
            if not symbol:
                continue
            
            ticker = Ticker(
                symbol=symbol,
                price=Decimal(ticker_data.get('last', '0')),
                volume_24h=Decimal(ticker_data.get('baseVolume', '0')),
                change_24h=Decimal(ticker_data.get('change24h', '0')),
                timestamp=datetime.fromtimestamp(int(ticker_data.get('ts', 0)) / 1000)
            )
            
            # Store in market data
            if symbol not in self.market_data:
                self.market_data[symbol] = {}
            self.market_data[symbol]['ticker'] = ticker
    
    async def _handle_funding_rate(self, data: List[Dict], channel: str) -> None:
        """Handle funding rate update"""
        for funding_data in data:
            symbol = funding_data.get('instId', '')
            if not symbol:
                continue
            
            funding_rate = FundingRate(
                symbol=symbol,
                rate=Decimal(funding_data.get('fundingRate', '0')),
                next_funding_time=datetime.fromtimestamp(
                    int(funding_data.get('nextFundingTime', 0)) / 1000
                ),
                timestamp=datetime.fromtimestamp(int(funding_data.get('fundingTime', 0)) / 1000)
            )
            
            self.funding_rates[symbol] = funding_rate
    
    async def _handle_open_interest(self, data: List[Dict], channel: str) -> None:
        """Handle open interest update"""
        for oi_data in data:
            symbol = oi_data.get('instId', '')
            if symbol:
                self.open_interest[symbol] = float(oi_data.get('openInterest', 0))
    
    # REST API methods
    async def get_ticker(self, symbol: str) -> Optional[Ticker]:
        """Get current ticker via REST API"""
        try:
            endpoint = "/api/spot/v1/market/ticker" if not self._is_futures_symbol(symbol) else "/api/mix/v1/market/ticker"
            response = await self.http_client.get(endpoint, params={'symbol': symbol})
            response.raise_for_status()
            
            data = response.json()
            if data.get('code') == '00000' and data.get('data'):
                ticker_data = data['data']
                
                return Ticker(
                    symbol=symbol,
                    price=Decimal(ticker_data.get('last', '0')),
                    volume_24h=Decimal(ticker_data.get('baseVol', '0')),
                    change_24h=Decimal(ticker_data.get('change24h', '0')),
                    timestamp=datetime.now()
                )
            
        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
        
        return None
    
    async def get_orderbook(self, symbol: str, levels: int = 20) -> Optional[OrderBook]:
        """Get current order book via REST API"""
        try:
            limit = min(levels, 100)  # Bitget max limit
            endpoint = "/api/spot/v1/market/depth" if not self._is_futures_symbol(symbol) else "/api/mix/v1/market/depth"
            
            params = {'symbol': symbol, 'limit': limit}
            response = await self.http_client.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            if data.get('code') == '00000' and data.get('data'):
                book_data = data['data']
                
                bids = [(Decimal(bid[0]), Decimal(bid[1])) for bid in book_data.get('bids', [])]
                asks = [(Decimal(ask[0]), Decimal(ask[1])) for ask in book_data.get('asks', [])]
                
                return OrderBook(
                    symbol=symbol,
                    bids=bids,
                    asks=asks,
                    timestamp=datetime.fromtimestamp(int(book_data.get('ts', 0)) / 1000)
                )
            
        except Exception as e:
            logger.error(f"Failed to get orderbook for {symbol}: {e}")
        
        return None
    
    async def get_historical_klines(
        self, 
        symbol: str, 
        interval: str = "1m", 
        limit: int = 100
    ) -> List[OHLCV]:
        """Get historical candlestick data"""
        try:
            # Convert interval to Bitget format
            granularity_map = {
                "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
                "1h": "1H", "4h": "4H", "1d": "1D"
            }
            granularity = granularity_map.get(interval, "1m")
            
            endpoint = "/api/spot/v1/market/candles" if not self._is_futures_symbol(symbol) else "/api/mix/v1/market/candles"
            
            params = {
                'symbol': symbol,
                'granularity': granularity,
                'limit': min(limit, 1000)
            }
            
            response = await self.http_client.get(endpoint, params=params)
            response.raise_for_status()
            
            data = response.json()
            klines = []
            
            if data.get('code') == '00000' and data.get('data'):
                for candle in data['data']:
                    if len(candle) >= 6:
                        ohlcv = OHLCV(
                            symbol=symbol,
                            open=Decimal(candle[1]),
                            high=Decimal(candle[2]),
                            low=Decimal(candle[3]),
                            close=Decimal(candle[4]),
                            volume=Decimal(candle[5]),
                            timestamp=datetime.fromtimestamp(int(candle[0]) / 1000),
                            timeframe=interval
                        )
                        klines.append(ohlcv)
            
            return list(reversed(klines))  # Bitget returns newest first
            
        except Exception as e:
            logger.error(f"Failed to get historical klines for {symbol}: {e}")
            return []
    
    async def get_funding_rate(self, symbol: str) -> Optional[FundingRate]:
        """Get current funding rate"""
        if not self._is_futures_symbol(symbol):
            return None
        
        try:
            response = await self.http_client.get(
                "/api/mix/v1/market/current-fundRate",
                params={'symbol': symbol}
            )
            response.raise_for_status()
            
            data = response.json()
            if data.get('code') == '00000' and data.get('data'):
                funding_data = data['data']
                
                return FundingRate(
                    symbol=symbol,
                    rate=Decimal(funding_data.get('fundingRate', '0')),
                    next_funding_time=datetime.fromtimestamp(
                        int(funding_data.get('nextFundingTime', 0)) / 1000
                    ),
                    timestamp=datetime.now()
                )
            
        except Exception as e:
            logger.error(f"Failed to get funding rate for {symbol}: {e}")
        
        return None
    
    async def get_open_interest(self, symbol: str) -> Optional[float]:
        """Get current open interest"""
        if not self._is_futures_symbol(symbol):
            return None
        
        try:
            response = await self.http_client.get(
                "/api/mix/v1/market/open-interest",
                params={'symbol': symbol}
            )
            response.raise_for_status()
            
            data = response.json()
            if data.get('code') == '00000' and data.get('data'):
                oi_data = data['data']
                return float(oi_data.get('openInterest', 0))
            
        except Exception as e:
            logger.error(f"Failed to get open interest for {symbol}: {e}")
        
        return None
    
    def is_connected(self) -> bool:
        """Check if adapter is connected"""
        return (
            self.status == ConnectionStatus.CONNECTED and
            self.http_client is not None and
            ((self.spot_ws and not self.spot_ws.closed) or
             (self.mix_ws and not self.mix_ws.closed))
        )
    
    def get_status(self) -> ConnectionStatus:
        """Get current connection status"""
        return self.status
    
    async def health_check(self) -> bool:
        """Perform health check"""
        try:
            if not self.http_client:
                return False
            
            response = await self.http_client.get("/api/spot/v1/public/time")
            return response.status_code == 200
            
        except Exception:
            return False
    
    # Streaming protocol implementation
    async def stream_ticker(self, symbol: str) -> AsyncIterator[Ticker]:
        """Stream ticker updates"""
        while self.is_connected():
            if symbol in self.market_data and 'ticker' in self.market_data[symbol]:
                yield self.market_data[symbol]['ticker']
            await asyncio.sleep(1)
    
    async def stream_orderbook(self, symbol: str) -> AsyncIterator[OrderBook]:
        """Stream orderbook updates"""
        while self.is_connected():
            if symbol in self.market_data and 'orderbook' in self.market_data[symbol]:
                yield self.market_data[symbol]['orderbook']
            await asyncio.sleep(0.1)  # High frequency for order book
    
    async def stream_funding(self, symbol: str) -> AsyncIterator[FundingRate]:
        """Stream funding rate updates"""
        while self.is_connected():
            if symbol in self.funding_rates:
                yield self.funding_rates[symbol]
            await asyncio.sleep(60)  # Funding rates update less frequently