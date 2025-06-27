"""
Bybit exchange adapter
Implements WebSocket and REST API integration for Bybit
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


class BybitAdapter(ExchangeAdapter):
    """
    Bybit exchange adapter with WebSocket and REST API

    Features:
    - Real-time ticker, orderbook, kline streams
    - Funding rate support for futures
    - REST API for historical data
    - Automatic reconnection
    """

    BASE_URL = "https://api.bybit.com"
    WS_BASE_URL = "wss://stream.bybit.com/v5/public/spot"  # Spot trading

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)

        # Configuration
        self.timeout = config.get("timeout_seconds", 30)
        self.category = config.get("category", "spot")  # spot, linear, inverse, option

        # HTTP client
        self.http_client: httpx.AsyncClient | None = None

        # WebSocket connection
        self.ws_connection: websockets.WebSocketServerProtocol | None = None
        self.subscribed_topics: list[str] = []

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
            response = await self.http_client.get("/v5/market/time")
            response.raise_for_status()

            # Connect WebSocket
            await self._connect_websocket()

            self.status = ConnectionStatus.CONNECTED
            self.reset_error_count()
            logger.info("Bybit adapter connected")

        except Exception as e:
            self.status = ConnectionStatus.FAILED
            logger.error(f"Bybit connection failed: {e}")
            raise

    async def _connect_websocket(self) -> None:
        """Connect to Bybit WebSocket"""
        try:
            self.ws_connection = await websockets.connect(self.WS_BASE_URL)

            # Start message handler
            asyncio.create_task(self._handle_websocket_messages())
            logger.debug("Bybit WebSocket connected")

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            raise

    async def disconnect(self) -> None:
        """Close all connections"""
        self.status = ConnectionStatus.DISCONNECTED

        # Close WebSocket connection
        if self.ws_connection and not self.ws_connection.closed:
            await self.ws_connection.close()

        # Close HTTP client
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None

        logger.info("Bybit adapter disconnected")

    async def subscribe_ticker(self, symbol: str) -> None:
        """Subscribe to ticker updates"""
        topic = f"tickers.{symbol}"
        await self._subscribe_topic(topic)

    async def subscribe_orderbook(self, symbol: str, levels: int = 20) -> None:
        """Subscribe to orderbook updates"""
        # Bybit orderbook depth levels: 1, 50, 200
        depth = 50 if levels <= 50 else 200
        topic = f"orderbook.{depth}.{symbol}"
        await self._subscribe_topic(topic)

    async def subscribe_klines(self, symbol: str, interval: str = "1") -> None:
        """Subscribe to kline updates"""
        # Convert interval to Bybit format (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
        bybit_interval = "1" if interval == "1m" else interval
        topic = f"kline.{bybit_interval}.{symbol}"
        await self._subscribe_topic(topic)

    async def subscribe_funding(self, symbol: str) -> None:
        """Subscribe to funding rate updates"""
        if self.category in ["linear", "inverse"]:
            topic = f"funding.{symbol}"
            await self._subscribe_topic(topic)

    async def _subscribe_topic(self, topic: str) -> None:
        """Subscribe to a WebSocket topic"""
        if topic in self.subscribed_topics:
            return

        if not self.ws_connection or self.ws_connection.closed:
            await self._connect_websocket()

        try:
            subscribe_msg = {"op": "subscribe", "args": [topic]}

            await self.ws_connection.send(json.dumps(subscribe_msg))
            self.subscribed_topics.append(topic)
            logger.debug(f"Subscribed to {topic}")

        except Exception as e:
            logger.error(f"WebSocket subscription failed for {topic}: {e}")
            raise

    async def _handle_websocket_messages(self) -> None:
        """Handle incoming WebSocket messages"""
        try:
            async for message in self.ws_connection:
                try:
                    data = json.loads(message)

                    # Handle different message types
                    if "topic" in data and "data" in data:
                        topic = data["topic"]
                        await self._handle_data_message(topic, data["data"])
                    elif "success" in data:
                        # Subscription confirmation
                        logger.debug(f"Subscription confirmed: {data}")
                    elif "op" in data and data["op"] == "ping":
                        # Respond to ping
                        await self.ws_connection.send(json.dumps({"op": "pong"}))

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON message: {message}")
                except Exception as e:
                    logger.error(f"Message handling error: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.warning("Bybit WebSocket connection closed")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")

    async def _handle_data_message(self, topic: str, data: dict[str, Any]) -> None:
        """Handle data messages based on topic"""
        try:
            if topic.startswith("tickers."):
                await self._handle_ticker_data(data)
            elif topic.startswith("orderbook."):
                await self._handle_orderbook_data(data)
            elif topic.startswith("kline."):
                await self._handle_kline_data(data)
            elif topic.startswith("funding."):
                await self._handle_funding_data(data)
        except Exception as e:
            logger.error(f"Error handling {topic} data: {e}")

    async def _handle_ticker_data(self, data: dict[str, Any]) -> None:
        """Handle ticker data"""
        ticker = Ticker(
            symbol=data["symbol"],
            price=Decimal(data["lastPrice"]),
            volume_24h=Decimal(data["volume24h"]),
            change_24h=Decimal(data["price24hPcnt"]),
            timestamp=datetime.fromtimestamp(int(data["ts"]) / 1000),
        )

        symbol = data["symbol"]
        if symbol not in self.latest_data:
            self.latest_data[symbol] = {}
        self.latest_data[symbol]["ticker"] = ticker

    async def _handle_orderbook_data(self, data: dict[str, Any]) -> None:
        """Handle orderbook data"""
        symbol = data["s"]

        bids = [(Decimal(bid[0]), Decimal(bid[1])) for bid in data["b"]]
        asks = [(Decimal(ask[0]), Decimal(ask[1])) for ask in data["a"]]

        orderbook = OrderBook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=datetime.fromtimestamp(int(data["ts"]) / 1000),
        )

        if symbol not in self.latest_data:
            self.latest_data[symbol] = {}
        self.latest_data[symbol]["orderbook"] = orderbook

    async def _handle_kline_data(self, data: list[dict[str, Any]]) -> None:
        """Handle kline data"""
        for kline_data in data:
            symbol = kline_data["symbol"]

            ohlcv = OHLCV(
                symbol=symbol,
                open=Decimal(kline_data["open"]),
                high=Decimal(kline_data["high"]),
                low=Decimal(kline_data["low"]),
                close=Decimal(kline_data["close"]),
                volume=Decimal(kline_data["volume"]),
                timestamp=datetime.fromtimestamp(int(kline_data["start"]) / 1000),
                timeframe=kline_data["interval"],
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

    async def _handle_funding_data(self, data: dict[str, Any]) -> None:
        """Handle funding rate data"""
        symbol = data["symbol"]

        funding_rate = FundingRate(
            symbol=symbol,
            rate=Decimal(data["fundingRate"]),
            next_funding_time=datetime.fromtimestamp(
                int(data["fundingRateTimestamp"]) / 1000
            ),
            timestamp=datetime.now(),
        )

        if symbol not in self.latest_data:
            self.latest_data[symbol] = {}
        self.latest_data[symbol]["funding"] = funding_rate

    # REST API methods
    async def get_ticker(self, symbol: str) -> Ticker | None:
        """Get current ticker data"""
        try:
            response = await self.http_client.get(
                "/v5/market/tickers",
                params={"category": self.category, "symbol": symbol},
            )
            response.raise_for_status()
            data = response.json()

            if data["retCode"] == 0 and data["result"]["list"]:
                ticker_data = data["result"]["list"][0]
                return Ticker(
                    symbol=ticker_data["symbol"],
                    price=Decimal(ticker_data["lastPrice"]),
                    volume_24h=Decimal(ticker_data["volume24h"]),
                    change_24h=Decimal(ticker_data["price24hPcnt"]),
                    timestamp=datetime.now(),
                )

        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")

        return None

    async def get_orderbook(self, symbol: str, levels: int = 20) -> OrderBook | None:
        """Get current orderbook"""
        try:
            limit = min(levels, 200)  # Bybit max limit
            response = await self.http_client.get(
                "/v5/market/orderbook",
                params={"category": self.category, "symbol": symbol, "limit": limit},
            )
            response.raise_for_status()
            data = response.json()

            if data["retCode"] == 0:
                result = data["result"]
                bids = [(Decimal(bid[0]), Decimal(bid[1])) for bid in result["b"]]
                asks = [(Decimal(ask[0]), Decimal(ask[1])) for ask in result["a"]]

                return OrderBook(
                    symbol=symbol,
                    bids=bids,
                    asks=asks,
                    timestamp=datetime.fromtimestamp(int(result["ts"]) / 1000),
                )

        except Exception as e:
            logger.error(f"Failed to get orderbook for {symbol}: {e}")

        return None

    async def get_historical_klines(
        self, symbol: str, interval: str = "1", limit: int = 100
    ) -> list[OHLCV]:
        """Get historical kline data"""
        try:
            response = await self.http_client.get(
                "/v5/market/kline",
                params={
                    "category": self.category,
                    "symbol": symbol,
                    "interval": interval,
                    "limit": min(limit, 1000),  # Bybit max limit
                },
            )
            response.raise_for_status()
            data = response.json()

            klines = []
            if data["retCode"] == 0:
                for kline in data["result"]["list"]:
                    ohlcv = OHLCV(
                        symbol=symbol,
                        open=Decimal(kline[1]),
                        high=Decimal(kline[2]),
                        low=Decimal(kline[3]),
                        close=Decimal(kline[4]),
                        volume=Decimal(kline[5]),
                        timestamp=datetime.fromtimestamp(int(kline[0]) / 1000),
                        timeframe=interval,
                    )
                    klines.append(ohlcv)

            return list(reversed(klines))  # Bybit returns newest first

        except Exception as e:
            logger.error(f"Failed to get historical klines for {symbol}: {e}")
            return []

    async def get_funding_rate(self, symbol: str) -> FundingRate | None:
        """Get current funding rate"""
        if self.category not in ["linear", "inverse"]:
            return None

        try:
            response = await self.http_client.get(
                "/v5/market/funding/history",
                params={"category": self.category, "symbol": symbol, "limit": 1},
            )
            response.raise_for_status()
            data = response.json()

            if data["retCode"] == 0 and data["result"]["list"]:
                funding_data = data["result"]["list"][0]
                return FundingRate(
                    symbol=symbol,
                    rate=Decimal(funding_data["fundingRate"]),
                    next_funding_time=datetime.fromtimestamp(
                        int(funding_data["fundingRateTimestamp"]) / 1000
                    ),
                    timestamp=datetime.now(),
                )

        except Exception as e:
            logger.error(f"Failed to get funding rate for {symbol}: {e}")

        return None

    def is_connected(self) -> bool:
        """Check if adapter is connected"""
        return (
            self.status == ConnectionStatus.CONNECTED
            and self.http_client is not None
            and self.ws_connection is not None
            and not self.ws_connection.closed
        )

    def get_status(self) -> ConnectionStatus:
        """Get current connection status"""
        return self.status

    async def health_check(self) -> bool:
        """Perform health check"""
        try:
            if not self.http_client:
                return False

            response = await self.http_client.get("/v5/market/time")
            return response.status_code == 200

        except Exception:
            return False

    # Streaming protocol implementation
    async def stream_ticker(self, symbol: str) -> AsyncIterator[Ticker]:
        """Stream ticker updates"""
        while self.is_connected():
            if symbol in self.latest_data and "ticker" in self.latest_data[symbol]:
                yield self.latest_data[symbol]["ticker"]
            await asyncio.sleep(1)

    async def stream_orderbook(self, symbol: str) -> AsyncIterator[OrderBook]:
        """Stream orderbook updates"""
        while self.is_connected():
            if symbol in self.latest_data and "orderbook" in self.latest_data[symbol]:
                yield self.latest_data[symbol]["orderbook"]
            await asyncio.sleep(1)

    async def stream_klines(
        self, symbol: str, interval: str = "1"
    ) -> AsyncIterator[OHLCV]:
        """Stream kline updates"""
        while self.is_connected():
            if symbol in self.latest_data and "klines" in self.latest_data[symbol]:
                klines = self.latest_data[symbol]["klines"]
                if klines:
                    yield klines[-1]
            await asyncio.sleep(60)

    async def stream_funding(self, symbol: str) -> AsyncIterator[FundingRate]:
        """Stream funding rate updates"""
        while self.is_connected():
            if symbol in self.latest_data and "funding" in self.latest_data[symbol]:
                yield self.latest_data[symbol]["funding"]
            await asyncio.sleep(3600)  # Funding updates every hour
