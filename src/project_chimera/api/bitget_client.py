"""
Bitget API Client for Real Market Data
Provides real-time market data from Bitget exchange
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from datetime import datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class BitgetAPIClient:
    """
    Bitget REST API client for real market data

    Features:
    - Real-time ticker data
    - Market depth (orderbook)
    - Recent trades
    - Account information (if authenticated)
    - Funding rates
    """

    def __init__(
        self,
        api_key: str = "",
        secret_key: str = "",
        passphrase: str = "",
        sandbox: bool = True,
        min_request_interval: float = 0.1,
        timeout_seconds: float = 10.0,
        max_connections: int = 100,
        max_keepalive_connections: int = 20,
    ):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.sandbox = sandbox

        # API endpoints - Bitget uses same URL for both sandbox and production
        # Sandbox mode is controlled by API key permissions
        self.base_url = "https://api.bitget.com"  # Official Bitget API URL

        # HTTP client
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout_seconds,
            limits=httpx.Limits(
                max_connections=max_connections,
                max_keepalive_connections=max_keepalive_connections,
            ),
        )

        # Rate limiting
        self.last_request_time = 0.0
        self.min_request_interval = min_request_interval

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

    def _generate_signature(
        self, timestamp: str, method: str, request_path: str, body: str = ""
    ) -> str:
        """Generate request signature for authenticated endpoints"""
        if not self.secret_key:
            return ""

        # Bitget signature format: timestamp + method + requestPath + body
        message = timestamp + method.upper() + request_path + body
        signature = hmac.new(
            self.secret_key.encode("utf-8"), message.encode("utf-8"), hashlib.sha256
        ).digest()

        # Base64 encode the signature (Bitget requirement)
        import base64

        return base64.b64encode(signature).decode("utf-8")

    def _get_headers(
        self, method: str, request_path: str, body: str = ""
    ) -> dict[str, str]:
        """Get request headers with authentication"""
        timestamp = str(int(time.time() * 1000))

        headers = {
            "Content-Type": "application/json",
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": self._generate_signature(
                timestamp, method, request_path, body
            ),
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self.passphrase,
        }

        return headers

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: dict = None,
        auth_required: bool = False,
    ) -> dict[str, Any]:
        """Make HTTP request with rate limiting and error handling"""

        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)

        self.last_request_time = time.time()

        try:
            # For authenticated requests, include query params in signature
            if auth_required and self.api_key:
                if method.upper() == "GET" and params:
                    # Build query string for signature
                    import urllib.parse

                    query_string = urllib.parse.urlencode(sorted(params.items()))
                    request_path = f"{endpoint}?{query_string}"
                else:
                    request_path = endpoint

                headers = self._get_headers(
                    method,
                    request_path,
                    json.dumps(params) if method.upper() != "GET" and params else "",
                )
            else:
                headers = {"Content-Type": "application/json"}

            if method.upper() == "GET":
                response = await self.client.get(
                    endpoint, params=params, headers=headers
                )
            else:
                response = await self.client.post(
                    endpoint, json=params, headers=headers
                )

            response.raise_for_status()
            data = response.json()

            # Check Bitget API response format
            if data.get("code") == "00000":  # Bitget success code
                return data.get("data", {})
            else:
                logger.error(f"Bitget API error: {data}")
                return {}

        except Exception as e:
            logger.error(f"Request failed: {e}")
            return {}

    async def get_ticker(self, symbol: str) -> dict[str, Any]:
        """Get real-time ticker data"""
        endpoint = "/api/v2/spot/market/tickers"
        params = {"symbol": symbol}

        data = await self._make_request("GET", endpoint, params)

        if data and isinstance(data, list) and len(data) > 0:
            # Bitget v2 API returns list of tickers
            ticker_data = data[0]
            return {
                "symbol": symbol,
                "price": float(ticker_data.get("lastPr", 0)),
                "bid": float(ticker_data.get("bidPr", 0)),
                "ask": float(ticker_data.get("askPr", 0)),
                "volume_24h": float(ticker_data.get("baseVolume", 0)),
                "change_24h": float(ticker_data.get("changeUtc", 0)),
                "high_24h": float(ticker_data.get("high24h", 0)),
                "low_24h": float(ticker_data.get("low24h", 0)),
                "timestamp": datetime.now(),
                "source": "bitget",
            }

        return {}

    async def get_orderbook(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        """Get market depth (orderbook)"""
        endpoint = "/api/v2/spot/market/orderbook"
        params = {"symbol": symbol, "limit": str(limit)}

        data = await self._make_request("GET", endpoint, params)

        if data:
            # Handle list response format
            if isinstance(data, list) and len(data) > 0:
                orderbook_data = data[0]
            else:
                orderbook_data = data

            return {
                "symbol": symbol,
                "bids": [
                    [float(bid[0]), float(bid[1])]
                    for bid in orderbook_data.get("bids", [])
                ],
                "asks": [
                    [float(ask[0]), float(ask[1])]
                    for ask in orderbook_data.get("asks", [])
                ],
                "timestamp": datetime.now(),
            }

        return {}

    async def get_recent_trades(
        self, symbol: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get recent trades"""
        endpoint = "/api/v2/spot/market/fills"
        params = {"symbol": symbol, "limit": str(limit)}

        data = await self._make_request("GET", endpoint, params)

        if data:
            trades = []
            for trade in data:
                trades.append(
                    {
                        "symbol": symbol,
                        "price": float(trade.get("price", 0)),
                        "size": float(trade.get("size", 0)),
                        "side": trade.get("side", "buy"),
                        "timestamp": datetime.fromtimestamp(
                            int(trade.get("ts", 0)) / 1000
                        ),
                        "trade_id": trade.get("tradeId", ""),
                    }
                )
            return trades

        return []

    async def get_kline(
        self, symbol: str, period: str = "1m", limit: int = 200
    ) -> list[dict[str, Any]]:
        """Get candlestick data"""
        endpoint = "/api/v2/spot/market/candles"
        # Convert period format for Bitget API
        period_map = {
            "1m": "1min",
            "5m": "5min",
            "15m": "15min",
            "30m": "30min",
            "1h": "1H",
            "4h": "4H",
            "1d": "1D",
        }
        granularity = period_map.get(period, "1min")
        params = {"symbol": symbol, "granularity": granularity, "limit": str(limit)}

        data = await self._make_request("GET", endpoint, params)

        if data:
            candles = []
            for candle in data:
                candles.append(
                    {
                        "symbol": symbol,
                        "timestamp": datetime.fromtimestamp(int(candle[0]) / 1000),
                        "open": float(candle[1]),
                        "high": float(candle[2]),
                        "low": float(candle[3]),
                        "close": float(candle[4]),
                        "volume": float(candle[5]),
                        "period": period,
                    }
                )
            return sorted(candles, key=lambda x: x["timestamp"])

        return []

    async def get_account_info(self) -> dict[str, Any]:
        """Get account information (requires authentication)"""
        if not self.api_key:
            return {"error": "Authentication required"}

        endpoint = "/api/v2/spot/account/assets"

        data = await self._make_request("GET", endpoint, auth_required=True)

        if data:
            assets = {}
            total_value_usdt = 0.0

            for asset in data:
                coin = asset.get("coin", "")
                available = float(asset.get("available", 0))
                frozen = float(asset.get("frozen", 0))
                total = available + frozen

                if total > 0:
                    assets[coin] = {
                        "available": available,
                        "frozen": frozen,
                        "total": total,
                    }

                    # Estimate USDT value (simplified)
                    if coin == "USDT":
                        total_value_usdt += total
                    elif coin == "BTC":
                        # You could fetch BTC price here for accurate conversion
                        total_value_usdt += total * 50000  # Rough estimate

            return {
                "assets": assets,
                "total_value_usdt": total_value_usdt,
                "timestamp": datetime.now(),
            }

        return {}

    async def get_futures_account(self) -> dict[str, Any]:
        """Get futures account information (requires authentication)"""
        if not self.api_key:
            return {"error": "Authentication required"}

        # For account info, we need to use the correct endpoint format
        endpoint = "/api/v2/mix/account/accounts"
        params = {"productType": "USDT-FUTURES"}

        data = await self._make_request("GET", endpoint, params, auth_required=True)

        if data:
            # Handle list response - take first account or USDT account
            if isinstance(data, list):
                if not data:
                    return {}
                # Find USDT account or use first one
                account_data = None
                for account in data:
                    if account.get("marginCoin") == "USDT":
                        account_data = account
                        break
                if not account_data:
                    account_data = data[0]
            else:
                account_data = data

            return {
                "marginCoin": account_data.get("marginCoin", "USDT"),
                "locked": float(account_data.get("locked", 0)),
                "available": float(account_data.get("available", 0)),
                "crossMaxAvailable": float(account_data.get("crossedMaxAvailable", 0)),
                "fixedMaxAvailable": float(account_data.get("isolatedMaxAvailable", 0)),
                "maxTransferOut": float(account_data.get("maxTransferOut", 0)),
                "equity": float(account_data.get("accountEquity", 0)),
                "usdtEquity": float(account_data.get("usdtEquity", 0)),
                "unrealizedPL": float(account_data.get("unrealizedPL", 0)),
                "timestamp": datetime.now(),
            }

        return {}

    async def get_futures_positions(self) -> list[dict[str, Any]]:
        """Get futures positions (requires authentication)"""
        if not self.api_key:
            return []

        endpoint = "/api/v2/mix/position/all-position"
        params = {"productType": "USDT-FUTURES", "marginCoin": "USDT"}

        data = await self._make_request("GET", endpoint, params, auth_required=True)

        if data:
            positions = []
            for pos in data:
                if float(pos.get("total", 0)) != 0:  # Only active positions
                    positions.append(
                        {
                            "symbol": pos.get("symbol", ""),
                            "size": float(pos.get("total", 0)),
                            "available": float(pos.get("available", 0)),
                            "averageOpenPrice": float(pos.get("averageOpenPrice", 0)),
                            "markPrice": float(pos.get("markPrice", 0)),
                            "unrealizedPL": float(pos.get("unrealizedPL", 0)),
                            "side": pos.get("holdSide", ""),
                            "marginMode": pos.get("marginMode", ""),
                            "leverage": pos.get("leverage", ""),
                            "marginSize": float(pos.get("marginSize", 0)),
                            "timestamp": datetime.now(),
                        }
                    )
            return positions

        return []

    async def get_multi_ticker(self, symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Get ticker data for multiple symbols"""
        tasks = [self.get_ticker(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        tickers = {}
        for i, result in enumerate(results):
            if isinstance(result, dict) and result:
                tickers[symbols[i]] = result
            else:
                logger.error(f"Failed to get ticker for {symbols[i]}: {result}")

        return tickers

    async def get_system_status(self) -> dict[str, Any]:
        """Get Bitget system status"""
        endpoint = "/api/v2/public/time"

        data = await self._make_request("GET", endpoint)

        if data:
            # Handle different response formats
            if isinstance(data, dict):
                server_time_ms = data.get("serverTime", int(time.time() * 1000))
            else:
                server_time_ms = (
                    data if isinstance(data, (int, str)) else int(time.time() * 1000)
                )

            return {
                "status": "online",
                "server_time": datetime.fromtimestamp(int(server_time_ms) / 1000),
                "timestamp": datetime.now(),
            }

        return {"status": "offline", "timestamp": datetime.now()}


class BitgetMarketDataService:
    """
    High-level service for Bitget market data
    Provides simplified interface for dashboard consumption
    """

    def __init__(
        self,
        api_key: str = "",
        secret_key: str = "",
        passphrase: str = "",
        sandbox: bool = True,
    ):
        self.client = BitgetAPIClient(api_key, secret_key, passphrase, sandbox)
        self.cache = {}
        self.cache_ttl = 30  # 30 seconds cache

    async def close(self):
        """Close the service"""
        await self.client.close()

    async def get_market_overview(self, symbols: list[str] = None) -> dict[str, Any]:
        """Get comprehensive market overview"""
        if symbols is None:
            # Use Bitget's standard symbol format (no suffix for v2 API)
            symbols = [
                "BTCUSDT",
                "ETHUSDT",
                "BNBUSDT",
                "ADAUSDT",
                "DOTUSDT",
            ]

        try:
            # Get ticker data for multiple symbols
            tickers = await self.client.get_multi_ticker(symbols)

            # Calculate market statistics
            total_volume = sum(
                ticker.get("volume_24h", 0) for ticker in tickers.values()
            )
            avg_change = sum(
                ticker.get("change_24h", 0) for ticker in tickers.values()
            ) / len(tickers)

            # Get system status
            system_status = await self.client.get_system_status()

            return {
                "tickers": tickers,
                "market_stats": {
                    "total_volume_24h": total_volume,
                    "average_change_24h": avg_change,
                    "active_symbols": len(
                        [t for t in tickers.values() if t.get("price", 0) > 0]
                    ),
                },
                "system_status": system_status,
                "timestamp": datetime.now(),
                "source": "bitget",
            }

        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return {"error": str(e), "timestamp": datetime.now()}

    async def get_portfolio_value(self) -> dict[str, Any]:
        """Get real portfolio value from Bitget account"""
        try:
            # Only try if we have API credentials
            if not self.client.api_key:
                return {
                    "total_value_usdt": 150000.0,
                    "equity": 150000.0,
                    "unrealized_pnl": 0.0,
                    "error": "No API credentials configured",
                    "demo_mode": True,
                    "account_type": "demo",
                    "timestamp": datetime.now(),
                }

            # Try futures account first (primary for ProjectChimera)
            futures_account = await self.client.get_futures_account()

            if futures_account and "error" not in futures_account:
                # Check if we have actual futures data (not empty response)
                equity = futures_account.get("equity", 0)
                usdt_equity = futures_account.get("usdtEquity", 0)

                if equity > 0 or usdt_equity > 0:
                    # Get futures positions for additional P&L data
                    positions = await self.client.get_futures_positions()

                    total_unrealized_pnl = sum(
                        pos.get("unrealizedPL", 0) for pos in positions
                    )

                    return {
                        "total_value_usdt": float(usdt_equity),
                        "equity": float(equity),
                        "available": float(futures_account.get("available", 0)),
                        "locked": float(futures_account.get("locked", 0)),
                        "unrealized_pnl": total_unrealized_pnl,
                        "positions": positions,
                        "account_type": "futures",
                        "demo_mode": False,
                        "timestamp": datetime.now(),
                    }

            # Fallback to spot account
            account_info = await self.client.get_account_info()

            if account_info and "error" not in account_info:
                assets = account_info.get("assets", {})
                total_value = account_info.get("total_value_usdt", 0)

                # Check if we have actual spot assets
                if assets or total_value > 0:
                    return {
                        "total_value_usdt": float(total_value),
                        "assets": assets,
                        "unrealized_pnl": 0.0,  # Spot doesn't have unrealized P&L
                        "account_type": "spot",
                        "demo_mode": False,
                        "timestamp": datetime.now(),
                    }

            # If we got here, we have API credentials but no account data
            return {
                "total_value_usdt": 150000.0,  # Default demo value
                "equity": 150000.0,
                "unrealized_pnl": 0.0,
                "error": "No account data available - API permissions may be limited",
                "demo_mode": True,
                "account_type": "demo",
                "timestamp": datetime.now(),
            }

        except Exception as e:
            logger.error(f"Error getting portfolio value: {e}")
            return {
                "total_value_usdt": 150000.0,
                "equity": 150000.0,
                "unrealized_pnl": 0.0,
                "error": str(e),
                "demo_mode": True,
                "account_type": "demo",
                "timestamp": datetime.now(),
            }

    async def get_price_history(
        self, symbol: str, period: str = "1m", hours: int = 24
    ) -> list[dict[str, Any]]:
        """Get price history for equity curve"""
        try:
            # Calculate number of candles needed
            period_minutes = {
                "1m": 1,
                "5m": 5,
                "15m": 15,
                "1h": 60,
                "4h": 240,
                "1d": 1440,
            }
            minutes = period_minutes.get(period, 1)
            limit = min((hours * 60) // minutes, 1000)  # Bitget limit

            candles = await self.client.get_kline(symbol, period, limit)

            return candles

        except Exception as e:
            logger.error(f"Error getting price history: {e}")
            return []

    async def place_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = "market",
        price: float = None,
    ) -> dict[str, Any]:
        """Place trading order (requires authentication)"""
        if not self.api_key:
            return {"error": "Authentication required"}

        endpoint = "/api/v2/spot/trade/place-order"

        body = {
            "symbol": symbol,
            "side": side,  # "buy" or "sell"
            "orderType": order_type,  # "limit", "market"
            "size": str(size),
        }

        if order_type == "limit" and price:
            body["price"] = str(price)

        data = await self._make_request("POST", endpoint, body, auth_required=True)

        if data:
            return {
                "order_id": data.get("orderId", ""),
                "symbol": symbol,
                "side": side,
                "size": size,
                "status": data.get("status", ""),
                "timestamp": datetime.now(),
            }

        return {}

    async def cancel_order(self, symbol: str, order_id: str) -> dict[str, Any]:
        """Cancel trading order (requires authentication)"""
        if not self.api_key:
            return {"error": "Authentication required"}

        endpoint = "/api/v2/spot/trade/cancel-order"
        params = {"symbol": symbol, "orderId": order_id}

        data = await self._make_request("POST", endpoint, params, auth_required=True)

        if data:
            return {
                "order_id": order_id,
                "symbol": symbol,
                "status": "cancelled",
                "timestamp": datetime.now(),
            }

        return {}

    async def get_open_orders(self, symbol: str = None) -> list[dict[str, Any]]:
        """Get open orders (requires authentication)"""
        if not self.api_key:
            return []

        endpoint = "/api/v2/spot/trade/unfilled-orders"
        params = {}
        if symbol:
            params["symbol"] = symbol

        data = await self._make_request("GET", endpoint, params, auth_required=True)

        if data:
            orders = []
            for order in data:
                orders.append(
                    {
                        "order_id": order.get("orderId", ""),
                        "symbol": order.get("symbol", ""),
                        "side": order.get("side", ""),
                        "size": float(order.get("size", 0)),
                        "price": float(order.get("price", 0)),
                        "filled_size": float(order.get("fillSize", 0)),
                        "status": order.get("status", ""),
                        "timestamp": datetime.fromtimestamp(
                            int(order.get("cTime", 0)) / 1000
                        ),
                    }
                )
            return orders

        return []


# Global instance for easy access
_bitget_service: BitgetMarketDataService | None = None


def get_bitget_service() -> BitgetMarketDataService:
    """Get global Bitget service instance"""
    global _bitget_service
    if _bitget_service is None:
        # Get credentials from environment
        import os

        api_key = os.getenv("BITGET_API_KEY", "")
        secret_key = os.getenv("BITGET_SECRET_KEY", "")
        passphrase = os.getenv("BITGET_PASSPHRASE", "")
        sandbox = os.getenv("BITGET_SANDBOX", "true").lower() == "true"

        _bitget_service = BitgetMarketDataService(
            api_key=api_key,
            secret_key=secret_key,
            passphrase=passphrase,
            sandbox=sandbox,
        )
    return _bitget_service


async def demo_bitget_data():
    """Demo function to test Bitget data fetching"""
    # Reset global service to pick up current environment
    global _bitget_service
    _bitget_service = None

    service = get_bitget_service()

    try:
        print("🔍 Testing Bitget API connection...")
        print(f"🔑 API Key configured: {bool(service.client.api_key)}")

        # Test system status
        overview = await service.get_market_overview()
        print(f"✅ Market Overview: {len(overview.get('tickers', {}))} symbols")

        # Test individual ticker
        btc_ticker = await service.client.get_ticker("BTCUSDT")
        print(f"💰 BTC Price: ${btc_ticker.get('price', 0):,.2f}")

        # Test portfolio (will show demo mode if not authenticated)
        portfolio = await service.get_portfolio_value()
        print(f"💼 Portfolio Value: ${portfolio.get('total_value_usdt', 0):,.2f}")
        print(f"📊 Demo Mode: {portfolio.get('demo_mode', True)}")
        print(f"📈 Account Type: {portfolio.get('account_type', 'unknown')}")
        if portfolio.get("equity"):
            print(f"💰 Real Equity: ${portfolio.get('equity', 0):,.6f}")

        return True

    except Exception as e:
        print(f"❌ Error testing Bitget API: {e}")
        return False
    finally:
        await service.close()


if __name__ == "__main__":
    # Load environment variables for testing
    from dotenv import load_dotenv

    load_dotenv()

    # Test the Bitget API
    asyncio.run(demo_bitget_data())
