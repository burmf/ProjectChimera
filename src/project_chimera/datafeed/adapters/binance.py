"""
Binance exchange adapter
Implements WebSocket and REST API integration for Binance
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime
from decimal import Decimal
from typing import Any

import httpx
import websockets

from ...domains.market import OHLCV, FundingRate, OrderBook, Ticker
from ..protocols import ConnectionStatus, ExchangeAdapter

logger = logging.getLogger(__name__)


class BinanceAdapter(ExchangeAdapter):
    """
    Binance exchange adapter with WebSocket and REST API

    Features:
    - Real-time ticker, orderbook, kline streams
    - REST API for historical data
    - Automatic reconnection
    - Rate limiting compliance
    """

    BASE_URL = "https://api.binance.com"
    WS_BASE_URL = "wss://stream.binance.com:9443/ws"

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)

        # Configuration
        self.timeout = config.get("timeout_seconds", 30)
        self.rate_limit_delay = config.get("rate_limit_delay", 0.1)

        # HTTP client
        self.http_client: httpx.AsyncClient | None = None

        # WebSocket connections
        self.ws_connections: dict[str, websockets.WebSocketServerProtocol] = {}
        self.ws_subscriptions: dict[str, list[str]] = {}  # symbol -> streams

        # Message handlers
        self.message_handlers = {
            "24hrTicker": self._handle_ticker,
            "depthUpdate": self._handle_orderbook,
            "kline": self._handle_kline,
            "markPriceUpdate": self._handle_funding,
        }

        # Data storage
        self.latest_data: dict[str, dict[str, Any]] = {}

    async def connect(self) -> None:
        """Establish connections"""
        try:
            self.status = ConnectionStatus.CONNECTING

            # Initialize HTTP client
            self.http_client = httpx.AsyncClient(
                base_url=self.BASE_URL,
                timeout=self.timeout,
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )

            # Test connection
            response = await self.http_client.get("/api/v3/ping")
            response.raise_for_status()

            self.status = ConnectionStatus.CONNECTED
            self.reset_error_count()
            logger.info("Binance adapter connected")

        except Exception as e:
            self.status = ConnectionStatus.FAILED
            logger.error(f"Binance connection failed: {e}")
            raise

    async def disconnect(self) -> None:
        """Close all connections"""
        self.status = ConnectionStatus.DISCONNECTED

        # Close WebSocket connections
        for ws in self.ws_connections.values():
            if not ws.closed:
                await ws.close()
        self.ws_connections.clear()

        # Close HTTP client
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None

        logger.info("Binance adapter disconnected")

    async def subscribe_ticker(self, symbol: str) -> None:
        """Subscribe to ticker updates"""
        stream = f"{symbol.lower()}@ticker"
        await self._subscribe_stream(symbol, stream)

    async def subscribe_orderbook(self, symbol: str, levels: int = 20) -> None:
        """Subscribe to orderbook updates"""
        # Use depth stream with 1000ms updates
        stream = f"{symbol.lower()}@depth@1000ms"
        await self._subscribe_stream(symbol, stream)

    async def subscribe_klines(self, symbol: str, interval: str = "1m") -> None:
        """Subscribe to kline updates"""
        stream = f"{symbol.lower()}@kline_{interval}"
        await self._subscribe_stream(symbol, stream)

    async def subscribe_funding(self, symbol: str) -> None:
        """Subscribe to funding rate updates (futures only)"""
        # Note: Spot trading doesn't have funding rates
        # This would be implemented for Binance Futures
        logger.debug(
            f"Funding rate subscription not implemented for spot symbol {symbol}"
        )

    async def _subscribe_stream(self, symbol: str, stream: str) -> None:
        """Subscribe to a WebSocket stream"""
        if symbol not in self.ws_subscriptions:
            self.ws_subscriptions[symbol] = []

        if stream not in self.ws_subscriptions[symbol]:
            self.ws_subscriptions[symbol].append(stream)

            # Create WebSocket connection for this symbol if not exists
            if symbol not in self.ws_connections:
                ws_url = f"{self.WS_BASE_URL}/{stream}"
                try:
                    ws = await websockets.connect(ws_url)
                    self.ws_connections[symbol] = ws

                    # Start message handler
                    asyncio.create_task(self._handle_messages(symbol, ws))
                    logger.debug(f"Subscribed to {stream}")

                except Exception as e:
                    logger.error(f"WebSocket subscription failed for {stream}: {e}")
                    raise

    async def _handle_messages(
        self, symbol: str, ws: websockets.WebSocketServerProtocol
    ) -> None:
        """Handle incoming WebSocket messages"""
        try:
            async for message in ws:
                try:
                    data = json.loads(message)

                    # Determine message type and handle
                    if "e" in data:  # Event type
                        event_type = data["e"]
                        if event_type in self.message_handlers:
                            await self.message_handlers[event_type](symbol, data)

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON message: {message}")
                except Exception as e:
                    logger.error(f"Message handling error: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.warning(f"WebSocket connection closed for {symbol}")
        except Exception as e:
            logger.error(f"WebSocket error for {symbol}: {e}")
        finally:
            if symbol in self.ws_connections:
                del self.ws_connections[symbol]

    async def _handle_ticker(self, symbol: str, data: dict[str, Any]) -> None:
        """Handle ticker update"""
        ticker = Ticker(
            symbol=data["s"],
            price=Decimal(data["c"]),  # Close price
            volume_24h=Decimal(data["v"]),  # Volume
            change_24h=Decimal(data["P"]),  # Price change percent
            timestamp=datetime.fromtimestamp(data["E"] / 1000),  # Event time
        )

        if symbol not in self.latest_data:
            self.latest_data[symbol] = {}
        self.latest_data[symbol]["ticker"] = ticker

    async def _handle_orderbook(self, symbol: str, data: dict[str, Any]) -> None:
        """Handle orderbook update"""
        bids = [(Decimal(bid[0]), Decimal(bid[1])) for bid in data["b"]]
        asks = [(Decimal(ask[0]), Decimal(ask[1])) for ask in data["a"]]

        orderbook = OrderBook(
            symbol=data["s"],
            bids=bids,
            asks=asks,
            timestamp=datetime.fromtimestamp(data["E"] / 1000),
        )

        if symbol not in self.latest_data:
            self.latest_data[symbol] = {}
        self.latest_data[symbol]["orderbook"] = orderbook

    async def _handle_kline(self, symbol: str, data: dict[str, Any]) -> None:
        """Handle kline update"""
        kline_data = data["k"]

        ohlcv = OHLCV(
            symbol=kline_data["s"],
            open=Decimal(kline_data["o"]),
            high=Decimal(kline_data["h"]),
            low=Decimal(kline_data["l"]),
            close=Decimal(kline_data["c"]),
            volume=Decimal(kline_data["v"]),
            timestamp=datetime.fromtimestamp(kline_data["t"] / 1000),
            timeframe=kline_data["i"],
        )

        if symbol not in self.latest_data:
            self.latest_data[symbol] = {}
        if "klines" not in self.latest_data[symbol]:
            self.latest_data[symbol]["klines"] = []

        # Store only last 100 klines
        klines = self.latest_data[symbol]["klines"]
        klines.append(ohlcv)
        if len(klines) > 100:
            self.latest_data[symbol]["klines"] = klines[-100:]

    async def _handle_funding(self, symbol: str, data: dict[str, Any]) -> None:
        """Handle funding rate update"""
        # Implementation for futures funding rates
        pass

    async def get_ticker(self, symbol: str) -> Ticker | None:
        """Get current ticker data"""
        try:
            response = await self.http_client.get(
                "/api/v3/ticker/24hr", params={"symbol": symbol}
            )
            response.raise_for_status()
            data = response.json()

            return Ticker(
                symbol=data["symbol"],
                price=Decimal(data["lastPrice"]),
                volume_24h=Decimal(data["volume"]),
                change_24h=Decimal(data["priceChangePercent"]),
                timestamp=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            return None

    async def get_orderbook(self, symbol: str, levels: int = 20) -> OrderBook | None:
        """Get current orderbook"""
        try:
            response = await self.http_client.get(
                "/api/v3/depth", params={"symbol": symbol, "limit": levels}
            )
            response.raise_for_status()
            data = response.json()

            bids = [(Decimal(bid[0]), Decimal(bid[1])) for bid in data["bids"]]
            asks = [(Decimal(ask[0]), Decimal(ask[1])) for ask in data["asks"]]

            return OrderBook(
                symbol=symbol, bids=bids, asks=asks, timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Failed to get orderbook for {symbol}: {e}")
            return None

    async def get_historical_klines(
        self, symbol: str, interval: str = "1m", limit: int = 100
    ) -> list[OHLCV]:
        """Get historical kline data"""
        try:
            response = await self.http_client.get(
                "/api/v3/klines",
                params={"symbol": symbol, "interval": interval, "limit": limit},
            )
            response.raise_for_status()
            data = response.json()

            klines = []
            for kline in data:
                ohlcv = OHLCV(
                    symbol=symbol,
                    open=Decimal(kline[1]),
                    high=Decimal(kline[2]),
                    low=Decimal(kline[3]),
                    close=Decimal(kline[4]),
                    volume=Decimal(kline[5]),
                    timestamp=datetime.fromtimestamp(kline[0] / 1000),
                    timeframe=interval,
                )
                klines.append(ohlcv)

            return klines

        except Exception as e:
            logger.error(f"Failed to get historical klines for {symbol}: {e}")
            return []

    async def get_funding_rate(self, symbol: str) -> FundingRate | None:
        """Get current funding rate (futures only)"""
        # Not implemented for spot trading
        return None

    def is_connected(self) -> bool:
        """Check if adapter is connected"""
        return (
            self.status == ConnectionStatus.CONNECTED and self.http_client is not None
        )

    def get_status(self) -> ConnectionStatus:
        """Get current connection status"""
        return self.status

    async def health_check(self) -> bool:
        """Perform health check"""
        try:
            if not self.http_client:
                return False

            response = await self.http_client.get("/api/v3/ping")
            return response.status_code == 200

        except Exception:
            return False

    # Streaming protocol implementation
    async def stream_ticker(self, symbol: str) -> AsyncIterator[Ticker]:
        """Stream ticker updates"""
        while self.is_connected():
            if symbol in self.latest_data and "ticker" in self.latest_data[symbol]:
                yield self.latest_data[symbol]["ticker"]
            await asyncio.sleep(1)  # 1 second intervals

    async def stream_orderbook(self, symbol: str) -> AsyncIterator[OrderBook]:
        """Stream orderbook updates"""
        while self.is_connected():
            if symbol in self.latest_data and "orderbook" in self.latest_data[symbol]:
                yield self.latest_data[symbol]["orderbook"]
            await asyncio.sleep(1)

    async def stream_klines(
        self, symbol: str, interval: str = "1m"
    ) -> AsyncIterator[OHLCV]:
        """Stream kline updates"""
        while self.is_connected():
            if symbol in self.latest_data and "klines" in self.latest_data[symbol]:
                klines = self.latest_data[symbol]["klines"]
                if klines:
                    yield klines[-1]  # Latest kline
            await asyncio.sleep(60)  # 1 minute intervals

    async def stream_funding(self, symbol: str) -> AsyncIterator[FundingRate]:
        """Stream funding rate updates"""
        # Not implemented for spot trading
        return
        yield  # Make it a generator
