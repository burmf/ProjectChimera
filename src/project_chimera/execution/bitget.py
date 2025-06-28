"""
Bitget Execution Engine - Phase F Implementation
REST order placement + WebSocket fills monitoring with CircuitBreaker
Features: 3 REST failures → pause 5 min, graceful SIGINT shutdown

Design Reference: CLAUDE.md - Coding Guidelines Section 8 (AsyncIO + tenacity)
Related Classes:
- Signal: Input from strategies (buy/sell signals)
- Order/Fill: Output structures for trade tracking
- CircuitBreaker: Fault tolerance (3 failures -> 5min pause)
- BitgetRest: HMAC authentication for order placement
"""

import asyncio
import hashlib
import hmac
import json
import logging
import signal
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from ..settings import get_settings

# Mock aiohttp and websockets for systems without these packages
try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("aiohttp not available, using mock HTTP client")
    AIOHTTP_AVAILABLE = False

try:
    import websockets

    WEBSOCKETS_AVAILABLE = True
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("websockets not available, using mock WebSocket client")
    WEBSOCKETS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderStatus(Enum):
    """Order status enumeration"""

    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class OrderSide(Enum):
    """Order side enumeration"""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration"""

    MARKET = "market"
    LIMIT = "limit"


@dataclass
class BitgetConfig:
    """Bitget API configuration"""

    api_key: str
    secret_key: str
    passphrase: str
    sandbox: bool = True
    base_url: str = "https://api.bitget.com"
    ws_url: str = "wss://ws.bitget.com/spot/v1/stream"
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0

    def __post_init__(self):
        if self.sandbox:
            self.base_url = "https://api.bitget.com"  # Bitget uses same URL for sandbox
            self.ws_url = "wss://ws.bitget.com/spot/v1/stream"

    @classmethod
    def from_settings(cls) -> "BitgetConfig":
        """Create config from application settings"""
        settings = get_settings()
        api_config = settings.api

        return cls(
            api_key=api_config.bitget_key.get_secret_value(),
            secret_key=api_config.bitget_secret.get_secret_value(),
            passphrase=api_config.bitget_passphrase.get_secret_value(),
            sandbox=api_config.bitget_sandbox,
            base_url=api_config.bitget_rest_url,
            ws_url=api_config.bitget_ws_spot_url,
            timeout_seconds=api_config.timeout_seconds,
            max_retries=api_config.max_retries,
            retry_delay=api_config.retry_delay,
        )


@dataclass
class Order:
    """Order representation"""

    symbol: str
    side: OrderSide
    order_type: OrderType
    size: float
    price: float | None = None
    client_order_id: str | None = None

    # Status tracking
    order_id: str | None = None
    status: OrderStatus = OrderStatus.PENDING
    filled_size: float = 0.0
    avg_price: float | None = None
    created_time: datetime | None = None
    updated_time: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data["side"] = self.side.value
        data["order_type"] = self.order_type.value
        data["status"] = self.status.value
        return data


@dataclass
class Fill:
    """Trade fill representation"""

    order_id: str
    trade_id: str
    symbol: str
    side: OrderSide
    size: float
    price: float
    fee: float
    fee_currency: str
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["side"] = self.side.value
        return data


class CircuitBreaker:
    """Enhanced circuit breaker for REST API failures"""

    def __init__(self, failure_threshold: int = 3, pause_duration: float = 300.0):
        self.failure_threshold = failure_threshold
        self.pause_duration = pause_duration  # 5 minutes
        self.consecutive_failures = 0
        self.last_failure_time = None
        self.is_open = False

    def can_execute(self) -> bool:
        """Check if requests can be executed"""
        if not self.is_open:
            return True

        # Check if pause period has elapsed
        if (
            self.last_failure_time
            and time.time() - self.last_failure_time > self.pause_duration
        ):
            self.reset()
            logger.info("Circuit breaker reset after pause period")
            return True

        return False

    def record_success(self):
        """Record successful operation"""
        if self.consecutive_failures > 0:
            logger.info(
                f"Circuit breaker: Reset after {self.consecutive_failures} failures"
            )

        self.consecutive_failures = 0
        self.is_open = False

    def record_failure(self):
        """Record failed operation"""
        self.consecutive_failures += 1
        self.last_failure_time = time.time()

        if self.consecutive_failures >= self.failure_threshold:
            self.is_open = True
            logger.warning(
                f"Circuit breaker OPENED: {self.consecutive_failures} consecutive failures. "
                f"Pausing for {self.pause_duration/60:.1f} minutes"
            )

    def reset(self):
        """Manually reset circuit breaker"""
        self.consecutive_failures = 0
        self.is_open = False
        self.last_failure_time = None

    def get_status(self) -> dict[str, Any]:
        """Get circuit breaker status"""
        remaining_pause = 0
        if self.is_open and self.last_failure_time:
            remaining_pause = max(
                0, self.pause_duration - (time.time() - self.last_failure_time)
            )

        return {
            "is_open": self.is_open,
            "consecutive_failures": self.consecutive_failures,
            "failure_threshold": self.failure_threshold,
            "remaining_pause_seconds": remaining_pause,
            "can_execute": self.can_execute(),
        }


class MockHTTPClient:
    """Mock HTTP client for when aiohttp is not available"""

    def __init__(self):
        self.session_active = False

    async def __aenter__(self):
        self.session_active = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.session_active = False

    async def post(self, url: str, **kwargs) -> "MockResponse":
        await asyncio.sleep(0.1)  # Simulate network delay

        # Simulate occasional failures for testing
        import random

        if random.random() < 0.05:  # 5% failure rate
            raise Exception("Mock network error")

        # Mock successful response
        return MockResponse(
            {
                "code": "00000",
                "msg": "success",
                "data": {
                    "orderId": f"mock_order_{int(time.time())}",
                    "clientOid": kwargs.get("json", {}).get(
                        "clientOid", "mock_client_id"
                    ),
                },
            }
        )

    async def get(self, url: str, **kwargs) -> "MockResponse":
        await asyncio.sleep(0.05)
        return MockResponse({"code": "00000", "msg": "success", "data": []})


class MockResponse:
    """Mock HTTP response"""

    def __init__(self, data: dict[str, Any], status: int = 200):
        self._data = data
        self.status = status

    async def json(self) -> dict[str, Any]:
        return self._data

    async def text(self) -> str:
        return json.dumps(self._data)


class MockWebSocket:
    """Mock WebSocket for when websockets is not available"""

    def __init__(self, uri: str):
        self.uri = uri
        self.connected = False
        self._message_queue = asyncio.Queue()
        self._feed_task = None

    async def __aenter__(self):
        self.connected = True
        # Start mock message feed
        self._feed_task = asyncio.create_task(self._generate_mock_messages())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.connected = False
        if self._feed_task:
            self._feed_task.cancel()

    async def send(self, message: str):
        logger.debug(f"Mock WS send: {message}")

    async def recv(self) -> str:
        return await self._message_queue.get()

    async def _generate_mock_messages(self):
        """Generate mock WebSocket messages"""
        import random

        while self.connected:
            # Generate mock fill message
            mock_fill = {
                "event": "trade",
                "data": {
                    "orderId": f"mock_order_{random.randint(1000, 9999)}",
                    "tradeId": f"trade_{random.randint(10000, 99999)}",
                    "symbol": "BTCUSDT",
                    "side": random.choice(["buy", "sell"]),
                    "size": f"{random.uniform(0.001, 0.1):.6f}",
                    "price": f"{random.uniform(40000, 60000):.2f}",
                    "fee": f"{random.uniform(0.001, 0.01):.6f}",
                    "feeCurrency": "USDT",
                    "timestamp": int(time.time() * 1000),
                },
            }

            await self._message_queue.put(json.dumps(mock_fill))
            await asyncio.sleep(random.uniform(5, 15))  # Random interval


class BitgetExecutionEngine:
    """
    Bitget execution engine with REST API and WebSocket integration

    Features:
    - REST API for order placement
    - WebSocket for real-time fill monitoring
    - Circuit breaker for failure handling
    - Graceful shutdown on SIGINT
    - Comprehensive error handling and logging
    """

    def __init__(self, config: BitgetConfig):
        self.config = config
        self.circuit_breaker = CircuitBreaker()

        # State management
        self.running = False
        self.orders: dict[str, Order] = {}
        self.fills: list[Fill] = []

        # WebSocket connection
        self.ws_connection = None
        self.ws_task = None

        # Callbacks
        self.order_callbacks: list[Callable] = []
        self.fill_callbacks: list[Callable] = []

        # Metrics
        self.metrics = {
            "orders_placed": 0,
            "orders_filled": 0,
            "orders_failed": 0,
            "fills_received": 0,
            "api_errors": 0,
            "ws_reconnects": 0,
            "start_time": None,
        }

        # Setup signal handlers
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.stop())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    def _generate_signature(
        self, timestamp: str, method: str, request_path: str, body: str = ""
    ) -> str:
        """Generate Bitget API signature"""
        message = timestamp + method.upper() + request_path + body
        signature = hmac.new(
            self.config.secret_key.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return signature

    def _get_headers(
        self, method: str, request_path: str, body: str = ""
    ) -> dict[str, str]:
        """Generate API request headers"""
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp, method, request_path, body)

        return {
            "ACCESS-KEY": self.config.api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self.config.passphrase,
            "Content-Type": "application/json",
        }

    async def start(self):
        """Start the execution engine"""
        logger.info("Starting Bitget Execution Engine")
        self.running = True
        self.metrics["start_time"] = datetime.now()

        try:
            # Start WebSocket connection
            self.ws_task = asyncio.create_task(self._ws_connection_loop())
            logger.info("Bitget Execution Engine started successfully")

        except Exception as e:
            logger.error(f"Failed to start execution engine: {e}")
            await self.stop()
            raise

    async def stop(self):
        """Stop the execution engine gracefully"""
        logger.info("Stopping Bitget Execution Engine")
        self.running = False

        from contextlib import suppress

        # Cancel WebSocket task
        if self.ws_task:
            self.ws_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.ws_task

        # Close WebSocket connection
        if self.ws_connection:
            with suppress(Exception):
                await self.ws_connection.close()

        logger.info("Bitget Execution Engine stopped")

    async def place_order(self, order: Order) -> bool:
        """Place order via REST API"""
        if not self.circuit_breaker.can_execute():
            logger.warning("Circuit breaker is open, order rejected")
            return False

        try:
            # Generate client order ID if not provided
            if not order.client_order_id:
                order.client_order_id = f"chimera_{int(time.time() * 1000)}"

            # Prepare order data
            order_data = {
                "symbol": order.symbol,
                "side": order.side.value,
                "orderType": order.order_type.value,
                "size": str(order.size),
                "clientOid": order.client_order_id,
            }

            if order.price:
                order_data["price"] = str(order.price)

            # Make API request
            request_path = "/api/spot/v1/trade/orders"
            body = json.dumps(order_data)
            headers = self._get_headers("POST", request_path, body)

            http_client = (
                aiohttp.ClientSession() if AIOHTTP_AVAILABLE else MockHTTPClient()
            )

            async with http_client as session:
                response = await session.post(
                    self.config.base_url + request_path,
                    json=order_data,
                    headers=headers,
                )

                result = await response.json()

                if result.get("code") == "00000":  # Success
                    order.order_id = result["data"]["orderId"]
                    order.status = OrderStatus.OPEN
                    order.created_time = datetime.now()

                    self.orders[order.order_id] = order
                    self.metrics["orders_placed"] += 1
                    self.circuit_breaker.record_success()

                    logger.info(f"Order placed successfully: {order.order_id}")

                    # Notify callbacks
                    for callback in self.order_callbacks:
                        try:
                            await callback(order)
                        except Exception as e:
                            logger.error(f"Error in order callback: {e}")

                    return True
                else:
                    error_msg = result.get("msg", "Unknown error")
                    logger.error(f"Order placement failed: {error_msg}")
                    self.metrics["orders_failed"] += 1
                    self.circuit_breaker.record_failure()
                    return False

        except Exception as e:
            logger.error(f"Error placing order: {e}")
            self.metrics["api_errors"] += 1
            self.circuit_breaker.record_failure()
            return False

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel order via REST API"""
        if not self.circuit_breaker.can_execute():
            logger.warning("Circuit breaker is open, cancel rejected")
            return False

        try:
            request_path = "/api/spot/v1/trade/cancel-order"
            body = json.dumps({"orderId": order_id})
            headers = self._get_headers("POST", request_path, body)

            http_client = (
                aiohttp.ClientSession() if AIOHTTP_AVAILABLE else MockHTTPClient()
            )

            async with http_client as session:
                response = await session.post(
                    self.config.base_url + request_path,
                    json={"orderId": order_id},
                    headers=headers,
                )

                result = await response.json()

                if result.get("code") == "00000":
                    if order_id in self.orders:
                        self.orders[order_id].status = OrderStatus.CANCELLED
                        self.orders[order_id].updated_time = datetime.now()

                    logger.info(f"Order cancelled: {order_id}")
                    self.circuit_breaker.record_success()
                    return True
                else:
                    logger.error(f"Order cancellation failed: {result.get('msg')}")
                    self.circuit_breaker.record_failure()
                    return False

        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            self.circuit_breaker.record_failure()
            return False

    async def _ws_connection_loop(self):
        """WebSocket connection loop with auto-reconnection"""
        while self.running:
            try:
                await self._connect_websocket()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.metrics["ws_reconnects"] += 1
                await asyncio.sleep(5)  # Wait before reconnecting

    async def _connect_websocket(self):
        """Connect to Bitget WebSocket and handle messages"""
        logger.info("Connecting to Bitget WebSocket")

        websocket_module = websockets if WEBSOCKETS_AVAILABLE else None

        if websocket_module:
            async with websocket_module.connect(self.config.ws_url) as ws:
                self.ws_connection = ws

                # Subscribe to order updates
                subscribe_msg = {
                    "op": "subscribe",
                    "args": [
                        {"instType": "SPOT", "channel": "orders", "instId": "default"}
                    ],
                }
                await ws.send(json.dumps(subscribe_msg))

                logger.info("WebSocket connected and subscribed")

                async for message in ws:
                    await self._handle_ws_message(message)
        else:
            # Use mock WebSocket
            async with MockWebSocket(self.config.ws_url) as ws:
                self.ws_connection = ws
                logger.info("Mock WebSocket connected")

                while self.running:
                    try:
                        message = await ws.recv()
                        await self._handle_ws_message(message)
                    except asyncio.CancelledError:
                        break

    async def _handle_ws_message(self, message: str):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)

            if data.get("event") == "trade":
                # Handle trade fill
                fill_data = data["data"]

                fill = Fill(
                    order_id=fill_data["orderId"],
                    trade_id=fill_data["tradeId"],
                    symbol=fill_data["symbol"],
                    side=OrderSide(fill_data["side"]),
                    size=float(fill_data["size"]),
                    price=float(fill_data["price"]),
                    fee=float(fill_data["fee"]),
                    fee_currency=fill_data["feeCurrency"],
                    timestamp=datetime.fromtimestamp(fill_data["timestamp"] / 1000),
                )

                self.fills.append(fill)
                self.metrics["fills_received"] += 1

                # Update order status
                if fill.order_id in self.orders:
                    order = self.orders[fill.order_id]
                    order.filled_size += fill.size
                    order.avg_price = fill.price  # Simplified
                    order.updated_time = fill.timestamp

                    if (
                        order.filled_size >= order.size * 0.99
                    ):  # Allow for minor rounding
                        order.status = OrderStatus.FILLED
                        self.metrics["orders_filled"] += 1

                logger.info(
                    f"Fill received: {fill.symbol} {fill.side.value} {fill.size} @ {fill.price}"
                )

                # Notify callbacks
                for callback in self.fill_callbacks:
                    try:
                        await callback(fill)
                    except Exception as e:
                        logger.error(f"Error in fill callback: {e}")

        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")

    def add_order_callback(self, callback: Callable):
        """Add order status callback"""
        self.order_callbacks.append(callback)

    def add_fill_callback(self, callback: Callable):
        """Add fill callback"""
        self.fill_callbacks.append(callback)

    def get_order(self, order_id: str) -> Order | None:
        """Get order by ID"""
        return self.orders.get(order_id)

    def get_open_orders(self) -> list[Order]:
        """Get all open orders"""
        return [
            order for order in self.orders.values() if order.status == OrderStatus.OPEN
        ]

    def get_health(self) -> dict[str, Any]:
        """Get execution engine health status"""
        uptime = (
            (datetime.now() - self.metrics["start_time"]).total_seconds()
            if self.metrics["start_time"]
            else 0
        )

        return {
            "status": (
                "healthy"
                if self.running and self.circuit_breaker.can_execute()
                else "unhealthy"
            ),
            "running": self.running,
            "websocket_connected": self.ws_connection is not None,
            "circuit_breaker": self.circuit_breaker.get_status(),
            "metrics": {
                **self.metrics,
                "uptime_seconds": uptime,
                "open_orders": len(self.get_open_orders()),
                "total_orders": len(self.orders),
            },
            "timestamp": datetime.now().isoformat(),
        }

    async def run_health_endpoint(self, port: int = 8080):
        """Run simple health check HTTP server"""
        try:
            from aiohttp import web

            async def health_handler(request):
                health = self.get_health()
                return web.json_response(health)

            app = web.Application()
            app.router.add_get("/health", health_handler)

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "0.0.0.0", port)
            await site.start()

            logger.info(f"Health endpoint running on port {port}")

        except Exception as e:
            logger.error(f"Failed to start health endpoint: {e}")


# CLI interface
async def main():
    """Main entry point for Bitget execution engine"""
    import argparse

    parser = argparse.ArgumentParser(description="Bitget Execution Engine")
    parser.add_argument("--api-key", required=True, help="Bitget API key")
    parser.add_argument("--secret-key", required=True, help="Bitget secret key")
    parser.add_argument("--passphrase", required=True, help="Bitget passphrase")
    parser.add_argument(
        "--sandbox", action="store_true", help="Use sandbox environment"
    )
    parser.add_argument(
        "--health-port", type=int, default=8080, help="Health endpoint port"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create configuration
    config = BitgetConfig(
        api_key=args.api_key,
        secret_key=args.secret_key,
        passphrase=args.passphrase,
        sandbox=args.sandbox,
    )

    # Create and start execution engine
    engine = BitgetExecutionEngine(config)

    # Add some example callbacks
    async def order_callback(order: Order):
        logger.info(f"Order update: {order.order_id} -> {order.status.value}")

    async def fill_callback(fill: Fill):
        logger.info(f"Fill: {fill.symbol} {fill.side.value} {fill.size} @ {fill.price}")

    engine.add_order_callback(order_callback)
    engine.add_fill_callback(fill_callback)

    try:
        await engine.start()

        # Start health endpoint
        await engine.run_health_endpoint(args.health_port)

        # Keep running
        while engine.running:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
        return 1
    finally:
        await engine.stop()

    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
