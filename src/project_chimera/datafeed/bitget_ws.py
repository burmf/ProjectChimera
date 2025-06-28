"""
Bitget WebSocket Feed Implementation - FEED-02
Real-time data streams with tenacity reconnection and latency monitoring

Design Reference: CLAUDE.md - Data Sources Section 7 (WS Channels: books/trade/ticker/funding/OI)
Related Classes:
- DataFeedBase: Abstract base with tenacity retry logic
- MarketFrame: Unified output structure for strategies
- BitgetAuth: HMAC signing for private channels
- PromExporter: ws_latency_ms metric collection
"""

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx
import structlog
import websockets
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..domains.market import OHLCV, FundingRate, OrderBook, Ticker
from ..settings import get_exchange_config
from .protocols import ConnectionStatus, ExchangeAdapter

# Structured logging for JSON output
logger = structlog.get_logger(__name__)


@dataclass
class LatencyMetrics:
    """WebSocket latency tracking"""

    last_ping_time: float = 0.0
    last_pong_time: float = 0.0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    message_count: int = 0

    def update_latency(self, ping_time: float, pong_time: float) -> None:
        """Update latency metrics with new ping/pong timing"""
        latency_ms = (pong_time - ping_time) * 1000

        if self.message_count == 0:
            self.avg_latency_ms = latency_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.avg_latency_ms = alpha * latency_ms + (1 - alpha) * self.avg_latency_ms

        self.max_latency_ms = max(self.max_latency_ms, latency_ms)
        self.message_count += 1
        self.last_ping_time = ping_time
        self.last_pong_time = pong_time


class BitgetWebSocketFeed(ExchangeAdapter):
    """
    Production-ready Bitget WebSocket adapter

    Features:
    - Unified spot + futures WebSocket streams
    - Automatic reconnection with exponential backoff
    - Latency monitoring and JSON logging
    - Circuit breaker on connection failures
    - Real-time books, trades, funding, OI data
    """

    def __init__(self, name: str, config: dict[str, Any]):
        super().__init__(name, config)

        # Exchange configuration from settings
        self.exchange_config = get_exchange_config("bitget")

        # API credentials and URLs
        self.api_key = config.get("api_key", "")
        self.secret_key = config.get("secret_key", "")
        self.passphrase = config.get("passphrase", "")
        self.sandbox = config.get("sandbox", True)

        # WebSocket URLs - Bitget official WebSocket endpoints
        self.spot_ws_url = "wss://ws.bitget.com/v2/ws/public"
        self.mix_ws_url = "wss://ws.bitget.com/v2/ws/public"
        self.private_ws_url = "wss://ws.bitget.com/v2/ws/private"
        self.rest_base_url = "https://api.bitget.com"

        # Connection state
        self.http_client: httpx.AsyncClient | None = None
        self.spot_ws: websockets.WebSocketServerProtocol | None = None
        self.mix_ws: websockets.WebSocketServerProtocol | None = None
        self.private_ws: websockets.WebSocketServerProtocol | None = None

        # Subscription tracking
        self.spot_channels: list[str] = []
        self.mix_channels: list[str] = []
        self.private_channels: list[str] = []

        # Data storage
        self.market_data: dict[str, dict[str, Any]] = {}
        self.funding_rates: dict[str, FundingRate] = {}
        self.open_interest: dict[str, float] = {}

        # Latency monitoring
        self.spot_latency = LatencyMetrics()
        self.mix_latency = LatencyMetrics()

        # Circuit breaker state
        self.connection_failures = 0
        self.max_connection_failures = 3
        self.circuit_breaker_until = 0.0

        # Message handlers
        self.message_handlers = {
            "ticker": self._handle_ticker,
            "books": self._handle_orderbook,
            "books5": self._handle_orderbook,
            "trade": self._handle_trade,
            "fundingRate": self._handle_funding_rate,
            "openInterest": self._handle_open_interest,
        }

    async def connect(self) -> None:
        """Establish connections with circuit breaker protection"""
        if self._is_circuit_breaker_active():
            raise ConnectionError(
                "Circuit breaker active - too many connection failures"
            )

        try:
            self.status = ConnectionStatus.CONNECTING

            # Initialize HTTP client
            self.http_client = httpx.AsyncClient(
                base_url=self.rest_base_url,
                timeout=self.exchange_config.timeout_seconds,
                limits=httpx.Limits(
                    max_connections=self.exchange_config.max_connections,
                    max_keepalive_connections=self.exchange_config.max_keepalive_connections,
                ),
            )

            # Test REST connection
            response = await self.http_client.get("/api/v2/public/time")
            response.raise_for_status()

            # Connect WebSocket streams with retry logic
            await self._connect_websockets_with_retry()

            self.status = ConnectionStatus.CONNECTED
            self.connection_failures = 0  # Reset on successful connection

            logger.info(
                "bitget_ws_connected",
                spot_url=self.spot_ws_url,
                mix_url=self.mix_ws_url,
                sandbox=self.sandbox,
            )

        except Exception as e:
            self.connection_failures += 1
            if self.connection_failures >= self.max_connection_failures:
                self.circuit_breaker_until = time.time() + 300  # 5 min circuit breaker

            self.status = ConnectionStatus.FAILED
            logger.error(
                "bitget_connection_failed",
                error=str(e),
                failures=self.connection_failures,
            )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(
            (websockets.exceptions.ConnectionClosed, ConnectionError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _connect_websockets_with_retry(self) -> None:
        """Connect WebSockets with tenacity retry logic"""
        try:
            # Connect spot WebSocket
            self.spot_ws = await websockets.connect(
                self.spot_ws_url,
                ping_interval=self.exchange_config.heartbeat_interval,
                ping_timeout=10,
                close_timeout=10,
                max_size=2**20,  # 1MB message size limit
                compression=None,  # Disable compression for latency
            )

            # Connect futures WebSocket
            self.mix_ws = await websockets.connect(
                self.mix_ws_url,
                ping_interval=self.exchange_config.heartbeat_interval,
                ping_timeout=10,
                close_timeout=10,
                max_size=2**20,
                compression=None,
            )

            # Connect private WebSocket if credentials are provided
            if self.api_key and self.secret_key:
                await self._connect_private_websocket()
                asyncio.create_task(self._handle_private_messages())

            # Start message handlers
            asyncio.create_task(self._handle_spot_messages())
            asyncio.create_task(self._handle_mix_messages())
            asyncio.create_task(self._monitor_latency())

            logger.info(
                "bitget_websockets_connected",
                spot_ping=self.exchange_config.heartbeat_interval,
                mix_ping=self.exchange_config.heartbeat_interval,
            )

        except Exception as e:
            logger.error("websocket_connection_failed", error=str(e))
            raise

    async def _connect_private_websocket(self) -> None:
        """Connect to authenticated WebSocket with login"""
        try:
            self.private_ws = await websockets.connect(
                self.private_ws_url,
                ping_interval=self.exchange_config.heartbeat_interval,
                ping_timeout=10,
                close_timeout=10,
                max_size=2**20,
                compression=None,
            )

            # Authenticate with Bitget
            await self._authenticate_private_websocket()

            logger.info("private_websocket_connected")

        except Exception as e:
            logger.error("private_websocket_connection_failed", error=str(e))
            raise

    async def _authenticate_private_websocket(self) -> None:
        """Authenticate private WebSocket connection"""
        timestamp = str(int(time.time() * 1000))
        message = timestamp + "GET" + "/user/verify"

        signature = hmac.new(
            self.secret_key.encode("utf-8"), message.encode("utf-8"), hashlib.sha256
        ).digest()

        sign = base64.b64encode(signature).decode("utf-8")

        login_msg = {
            "op": "login",
            "args": [
                {
                    "apiKey": self.api_key,
                    "passphrase": self.passphrase,
                    "timestamp": timestamp,
                    "sign": sign,
                }
            ],
        }

        await self.private_ws.send(json.dumps(login_msg))
        logger.info("private_websocket_auth_sent")

    async def disconnect(self) -> None:
        """Close all connections gracefully"""
        try:
            self.status = ConnectionStatus.DISCONNECTING

            # Close WebSocket connections
            if self.spot_ws and not self.spot_ws.closed:
                await self.spot_ws.close()
                logger.info("spot_websocket_closed")

            if self.mix_ws and not self.mix_ws.closed:
                await self.mix_ws.close()
                logger.info("mix_websocket_closed")

            if self.private_ws and not self.private_ws.closed:
                await self.private_ws.close()
                logger.info("private_websocket_closed")

            # Close HTTP client
            if self.http_client:
                await self.http_client.aclose()
                logger.info("http_client_closed")

            self.status = ConnectionStatus.DISCONNECTED
            logger.info("bitget_adapter_disconnected")

        except Exception as e:
            logger.error("disconnect_error", error=str(e))
            self.status = ConnectionStatus.ERROR

    async def subscribe_ticker(self, symbol: str) -> None:
        """Subscribe to ticker updates"""
        channel = "ticker"

        if self._is_futures_symbol(symbol):
            await self._subscribe_mix_channel(f"{channel}.{symbol}")
        else:
            await self._subscribe_spot_channel(f"{channel}.{symbol}")

        logger.info(
            "subscribed_ticker", symbol=symbol, futures=self._is_futures_symbol(symbol)
        )

    async def subscribe_orderbook(self, symbol: str, levels: int = 20) -> None:
        """Subscribe to order book updates"""
        # Bitget v2 uses books for orderbook
        channel = "books"

        if self._is_futures_symbol(symbol):
            await self._subscribe_mix_channel(f"{channel}.{symbol}")
        else:
            await self._subscribe_spot_channel(f"{channel}.{symbol}")

        logger.info(
            "subscribed_orderbook", symbol=symbol, levels=levels, type=depth_type
        )

    async def subscribe_trades(self, symbol: str) -> None:
        """Subscribe to trade updates"""
        channel = "trade"

        if self._is_futures_symbol(symbol):
            await self._subscribe_mix_channel(f"{channel}.{symbol}")
        else:
            await self._subscribe_spot_channel(f"{channel}.{symbol}")

        logger.info("subscribed_trades", symbol=symbol)

    async def subscribe_funding(self, symbol: str) -> None:
        """Subscribe to funding rate updates (futures only)"""
        if not self._is_futures_symbol(symbol):
            logger.warning("funding_not_available", symbol=symbol, reason="spot_symbol")
            return

        channel = f"fundingRate.{symbol}"
        await self._subscribe_mix_channel(channel)

        logger.info("subscribed_funding", symbol=symbol)

    async def subscribe_open_interest(self, symbol: str) -> None:
        """Subscribe to open interest updates (futures only)"""
        if not self._is_futures_symbol(symbol):
            logger.warning("oi_not_available", symbol=symbol, reason="spot_symbol")
            return

        channel = f"openInterest.{symbol}"
        await self._subscribe_mix_channel(channel)

        logger.info("subscribed_open_interest", symbol=symbol)

    def _is_futures_symbol(self, symbol: str) -> bool:
        """Check if symbol is a futures contract"""
        return "USDT" in symbol or "USDC" in symbol or symbol.endswith("_UMCBL")

    def _is_circuit_breaker_active(self) -> bool:
        """Check if circuit breaker is currently active"""
        return time.time() < self.circuit_breaker_until

    async def _subscribe_spot_channel(self, channel: str) -> None:
        """Subscribe to spot WebSocket channel"""
        if channel in self.spot_channels:
            return

        if not self.spot_ws or self.spot_ws.closed:
            await self._connect_websockets_with_retry()

        try:
            # Bitget v2 WebSocket subscription format
            subscribe_msg = {
                "op": "subscribe",
                "args": [
                    {
                        "instType": "SPOT",
                        "channel": channel,
                        "instId": (
                            channel.split(".")[1] if "." in channel else "BTCUSDT"
                        ),
                    }
                ],
            }

            await self.spot_ws.send(json.dumps(subscribe_msg))
            self.spot_channels.append(channel)

            logger.info("spot_channel_subscribed", channel=channel)

        except Exception as e:
            logger.error("spot_subscription_failed", channel=channel, error=str(e))
            raise

    async def _subscribe_mix_channel(self, channel: str) -> None:
        """Subscribe to futures WebSocket channel"""
        if channel in self.mix_channels:
            return

        if not self.mix_ws or self.mix_ws.closed:
            await self._connect_websockets_with_retry()

        try:
            # Bitget v2 WebSocket subscription format for futures
            subscribe_msg = {
                "op": "subscribe",
                "args": [
                    {
                        "instType": "USDT-FUTURES",
                        "channel": channel,
                        "instId": (
                            channel.split(".")[1] if "." in channel else "BTCUSDT"
                        ),
                    }
                ],
            }

            await self.mix_ws.send(json.dumps(subscribe_msg))
            self.mix_channels.append(channel)

            logger.info("mix_channel_subscribed", channel=channel)

        except Exception as e:
            logger.error("mix_subscription_failed", channel=channel, error=str(e))
            raise

    async def _handle_spot_messages(self) -> None:
        """Handle incoming spot WebSocket messages"""
        try:
            async for message in self.spot_ws:
                message_time = time.time()
                await self._process_message(message, "spot", message_time)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("spot_websocket_closed", reason="connection_lost")
            await self._reconnect_spot()
        except Exception as e:
            logger.error("spot_message_handler_error", error=str(e))

    async def _handle_mix_messages(self) -> None:
        """Handle incoming futures WebSocket messages"""
        try:
            async for message in self.mix_ws:
                message_time = time.time()
                await self._process_message(message, "mix", message_time)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("mix_websocket_closed", reason="connection_lost")
            await self._reconnect_mix()
        except Exception as e:
            logger.error("mix_message_handler_error", error=str(e))

    async def _handle_private_messages(self) -> None:
        """Handle incoming private WebSocket messages"""
        try:
            async for message in self.private_ws:
                message_time = time.time()
                await self._process_private_message(message, message_time)
        except websockets.exceptions.ConnectionClosed:
            logger.warning("private_websocket_closed", reason="connection_lost")
            # Could implement private reconnection here
        except Exception as e:
            logger.error("private_message_handler_error", error=str(e))

    async def _process_message(
        self, message: str, ws_type: str, receive_time: float
    ) -> None:
        """Process WebSocket message and update latency metrics"""
        try:
            data = json.loads(message)

            # Handle pong responses for latency measurement
            if data.get("event") == "pong":
                ping_time = data.get("ping_time", receive_time)
                if ws_type == "spot":
                    self.spot_latency.update_latency(ping_time, receive_time)
                else:
                    self.mix_latency.update_latency(ping_time, receive_time)
                return

            # Route to appropriate handler - Bitget v2 format
            if "data" in data:
                # Handle subscription confirmation
                if data.get("event") == "subscribe":
                    logger.info("subscription_confirmed", arg=data.get("arg"))
                    return

                # Handle data messages
                for msg in data["data"]:
                    if "arg" in data:
                        channel = data["arg"].get("channel", "")
                        handler_name = (
                            channel.split(".")[0] if "." in channel else channel
                        )

                        if handler_name in self.message_handlers:
                            await self.message_handlers[handler_name](
                                [msg], data["arg"]
                            )
                        else:
                            logger.debug(
                                "unhandled_message", channel=channel, type=ws_type
                            )

        except json.JSONDecodeError as e:
            logger.error("json_decode_error", message=message[:100], error=str(e))
        except Exception as e:
            logger.error("message_processing_error", error=str(e))

    async def _process_private_message(self, message: str, receive_time: float) -> None:
        """Process private WebSocket message"""
        try:
            data = json.loads(message)

            # Handle login response
            if data.get("event") == "login":
                if data.get("code") == "0":
                    logger.info("private_websocket_authenticated", success=True)
                else:
                    logger.error("private_websocket_auth_failed", error=data.get("msg"))
                return

            # Handle private data (orders, positions, account updates)
            if "data" in data and "arg" in data:
                channel = data["arg"].get("channel", "")
                logger.info("private_data_received", channel=channel, data=data["data"])

        except json.JSONDecodeError as e:
            logger.error(
                "private_json_decode_error", message=message[:100], error=str(e)
            )
        except Exception as e:
            logger.error("private_message_processing_error", error=str(e))

    async def _monitor_latency(self) -> None:
        """Monitor WebSocket latency with periodic pings"""
        while self.status == ConnectionStatus.CONNECTED:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                current_time = time.time()

                # Send ping to spot WebSocket
                if self.spot_ws and not self.spot_ws.closed:
                    ping_msg = {"op": "ping", "ping_time": current_time}
                    await self.spot_ws.send(json.dumps(ping_msg))

                # Send ping to mix WebSocket
                if self.mix_ws and not self.mix_ws.closed:
                    ping_msg = {"op": "ping", "ping_time": current_time}
                    await self.mix_ws.send(json.dumps(ping_msg))

                # Log latency metrics
                logger.info(
                    "ws_latency_metrics",
                    spot_latency_ms=round(self.spot_latency.avg_latency_ms, 2),
                    mix_latency_ms=round(self.mix_latency.avg_latency_ms, 2),
                    spot_max_ms=round(self.spot_latency.max_latency_ms, 2),
                    mix_max_ms=round(self.mix_latency.max_latency_ms, 2),
                )

            except Exception as e:
                logger.error("latency_monitor_error", error=str(e))
                await asyncio.sleep(5)

    async def _reconnect_spot(self) -> None:
        """Reconnect spot WebSocket with exponential backoff"""
        for attempt in range(self.exchange_config.max_reconnect_attempts):
            try:
                await asyncio.sleep(self.exchange_config.reconnect_delay * (2**attempt))

                self.spot_ws = await websockets.connect(self.spot_ws_url)

                # Resubscribe to channels
                for channel in self.spot_channels.copy():
                    self.spot_channels.remove(channel)
                    await self._subscribe_spot_channel(channel)

                logger.info("spot_reconnected", attempt=attempt + 1)
                return

            except Exception as e:
                logger.warning(
                    "spot_reconnect_failed", attempt=attempt + 1, error=str(e)
                )

        logger.error(
            "spot_reconnect_exhausted",
            max_attempts=self.exchange_config.max_reconnect_attempts,
        )

    async def _reconnect_mix(self) -> None:
        """Reconnect futures WebSocket with exponential backoff"""
        for attempt in range(self.exchange_config.max_reconnect_attempts):
            try:
                await asyncio.sleep(self.exchange_config.reconnect_delay * (2**attempt))

                self.mix_ws = await websockets.connect(self.mix_ws_url)

                # Resubscribe to channels
                for channel in self.mix_channels.copy():
                    self.mix_channels.remove(channel)
                    await self._subscribe_mix_channel(channel)

                logger.info("mix_reconnected", attempt=attempt + 1)
                return

            except Exception as e:
                logger.warning(
                    "mix_reconnect_failed", attempt=attempt + 1, error=str(e)
                )

        logger.error(
            "mix_reconnect_exhausted",
            max_attempts=self.exchange_config.max_reconnect_attempts,
        )

    # Message handlers
    async def _handle_ticker(self, data: list[dict], arg: dict) -> None:
        """Handle ticker updates"""
        for tick in data:
            symbol = tick.get("instId", "")
            if symbol:
                ticker = Ticker(
                    symbol=symbol,
                    last_price=float(tick.get("last", 0)),
                    bid_price=float(tick.get("bidPx", 0)),
                    ask_price=float(tick.get("askPx", 0)),
                    volume_24h=float(tick.get("vol24h", 0)),
                    change_24h=float(tick.get("change24h", 0)),
                    timestamp=datetime.now(timezone.utc),
                )

                self.market_data[symbol] = {"ticker": ticker}
                logger.debug("ticker_updated", symbol=symbol, price=ticker.last_price)

    async def _handle_orderbook(self, data: list[dict], arg: dict) -> None:
        """Handle order book updates"""
        for book in data:
            symbol = book.get("instId", "")
            if symbol:
                # Convert Bitget format to internal OrderBook
                bids = [[float(bid[0]), float(bid[1])] for bid in book.get("bids", [])]
                asks = [[float(ask[0]), float(ask[1])] for ask in book.get("asks", [])]

                orderbook = OrderBook(
                    symbol=symbol,
                    bids=bids,
                    asks=asks,
                    timestamp=datetime.now(timezone.utc),
                )

                if symbol not in self.market_data:
                    self.market_data[symbol] = {}
                self.market_data[symbol]["orderbook"] = orderbook

                logger.debug(
                    "orderbook_updated",
                    symbol=symbol,
                    bid=bids[0][0] if bids else 0,
                    ask=asks[0][0] if asks else 0,
                )

    async def _handle_trade(self, data: list[dict], arg: dict) -> None:
        """Handle trade updates"""
        for trade in data:
            symbol = trade.get("instId", "")
            if symbol:
                logger.debug(
                    "trade_received",
                    symbol=symbol,
                    price=float(trade.get("px", 0)),
                    size=float(trade.get("sz", 0)),
                    side=trade.get("side", ""),
                )

    async def _handle_funding_rate(self, data: list[dict], arg: dict) -> None:
        """Handle funding rate updates"""
        for rate in data:
            symbol = rate.get("instId", "")
            if symbol:
                funding_rate = FundingRate(
                    symbol=symbol,
                    rate=float(rate.get("fundingRate", 0)),
                    next_funding_time=datetime.fromisoformat(
                        rate.get("fundingTime", "")
                    ),
                    timestamp=datetime.now(timezone.utc),
                )

                self.funding_rates[symbol] = funding_rate
                logger.debug("funding_updated", symbol=symbol, rate=funding_rate.rate)

    async def _handle_open_interest(self, data: list[dict], arg: dict) -> None:
        """Handle open interest updates"""
        for oi in data:
            symbol = oi.get("instId", "")
            if symbol:
                oi_value = float(oi.get("oi", 0))
                self.open_interest[symbol] = oi_value
                logger.debug("oi_updated", symbol=symbol, oi=oi_value)

    # Required abstract method implementations
    async def get_ticker(self, symbol: str) -> Ticker | None:
        """Get current ticker data"""
        if symbol in self.market_data and "ticker" in self.market_data[symbol]:
            return self.market_data[symbol]["ticker"]
        return None

    async def get_orderbook(self, symbol: str, levels: int = 20) -> OrderBook | None:
        """Get current order book"""
        if symbol in self.market_data and "orderbook" in self.market_data[symbol]:
            return self.market_data[symbol]["orderbook"]
        return None

    async def get_historical_klines(
        self, symbol: str, interval: str = "1m", limit: int = 100
    ) -> list[OHLCV]:
        """Get historical kline data via REST API"""
        try:
            if not self.http_client:
                return []

            # Bitget v2 REST endpoint for historical data
            endpoint = "/api/v2/spot/market/candles"
            params = {"symbol": symbol, "granularity": interval, "limit": str(limit)}

            response = await self.http_client.get(endpoint, params=params)
            response.raise_for_status()
            data = response.json()

            # Convert to OHLCV format
            klines = []
            for candle in data.get("data", []):
                ohlcv = OHLCV(
                    symbol=symbol,
                    open=float(candle[1]),
                    high=float(candle[2]),
                    low=float(candle[3]),
                    close=float(candle[4]),
                    volume=float(candle[5]),
                    timestamp=datetime.fromtimestamp(
                        int(candle[0]) / 1000, timezone.utc
                    ),
                )
                klines.append(ohlcv)

            return klines

        except Exception as e:
            logger.error("historical_klines_error", symbol=symbol, error=str(e))
            return []

    async def get_funding_rate(self, symbol: str) -> FundingRate | None:
        """Get current funding rate"""
        if symbol in self.funding_rates:
            return self.funding_rates[symbol]
        return None

    async def subscribe_klines(self, symbol: str, interval: str = "1m") -> None:
        """Subscribe to kline/candlestick updates"""
        channel = f"candle{interval}.{symbol}"

        if self._is_futures_symbol(symbol):
            await self._subscribe_mix_channel(channel)
        else:
            await self._subscribe_spot_channel(channel)

        logger.info("subscribed_klines", symbol=symbol, interval=interval)

    def get_status(self) -> ConnectionStatus:
        """Get current connection status"""
        return self.status

    def is_connected(self) -> bool:
        """Check if adapter is connected"""
        return self.status == ConnectionStatus.CONNECTED

    async def health_check(self) -> bool:
        """Perform health check"""
        try:
            if not self.http_client:
                return False

            response = await self.http_client.get("/api/v2/public/time", timeout=5)
            response.raise_for_status()
            return True

        except Exception:
            return False

    # Streaming protocol implementations (async generators)
    async def stream_ticker(self, symbol: str) -> AsyncIterator[Ticker]:
        """Stream ticker updates"""
        await self.subscribe_ticker(symbol)

        while self.is_connected():
            if symbol in self.market_data and "ticker" in self.market_data[symbol]:
                yield self.market_data[symbol]["ticker"]
            await asyncio.sleep(0.1)  # Small delay to prevent busy loop

    async def stream_orderbook(self, symbol: str) -> AsyncIterator[OrderBook]:
        """Stream order book updates"""
        await self.subscribe_orderbook(symbol)

        while self.is_connected():
            if symbol in self.market_data and "orderbook" in self.market_data[symbol]:
                yield self.market_data[symbol]["orderbook"]
            await asyncio.sleep(0.1)

    async def stream_klines(
        self, symbol: str, interval: str = "1m"
    ) -> AsyncIterator[OHLCV]:
        """Stream kline updates"""
        await self.subscribe_klines(symbol, interval)

        # For WebSocket implementation, this would need additional logic
        # to process kline data as it arrives
        while self.is_connected():
            await asyncio.sleep(1)  # Placeholder
            # In real implementation, yield OHLCV updates

    async def stream_funding(self, symbol: str) -> AsyncIterator[FundingRate]:
        """Stream funding rate updates"""
        await self.subscribe_funding(symbol)

        while self.is_connected():
            if symbol in self.funding_rates:
                yield self.funding_rates[symbol]
            await asyncio.sleep(30)  # Funding updates are less frequent


def create_bitget_ws_feed(config: dict[str, Any]) -> BitgetWebSocketFeed:
    """Factory function to create BitgetWebSocketFeed"""
    return BitgetWebSocketFeed("bitget_ws", config)
