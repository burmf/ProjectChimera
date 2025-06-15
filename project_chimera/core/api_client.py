"""
Professional Async API Client for Bitget Futures
Clean architecture with proper error handling and retry strategies
"""

import asyncio
import hmac
import hashlib
import base64
import time
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

import httpx
import websockets
from loguru import logger
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from ..config import Settings, get_settings


class OrderSide(Enum):
    """Order side enumeration"""
    LONG = "long"
    SHORT = "short"


class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"


@dataclass
class TickerData:
    """Structured ticker data"""
    symbol: str
    price: float
    open_24h: float
    high_24h: float
    low_24h: float
    volume: float
    change_24h: float
    ask_price: float
    bid_price: float
    spread: float
    timestamp: datetime


@dataclass
class OrderResult:
    """Order execution result"""
    order_id: str
    client_order_id: Optional[str]
    symbol: str
    side: OrderSide
    size: float
    order_type: OrderType
    status: str
    timestamp: datetime


class APIException(Exception):
    """Custom API exception"""
    
    def __init__(self, message: str, code: Optional[str] = None, response_data: Optional[Dict] = None):
        super().__init__(message)
        self.code = code
        self.response_data = response_data


class ConnectionException(APIException):
    """Connection-related exception"""
    pass


class AuthenticationException(APIException):
    """Authentication-related exception"""
    pass


class RateLimitException(APIException):
    """Rate limit exception"""
    pass


class AsyncBitgetClient:
    """
    Professional async Bitget API client
    Features: retry strategies, connection pooling, proper error handling
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.api_config = self.settings.api
        
        # HTTP client with connection pooling
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.api_config.timeout_seconds),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            headers={"Content-Type": "application/json"}
        )
        
        # Rate limiting
        self._last_request_time = 0.0
        self._request_lock = asyncio.Lock()
        
        # WebSocket connection
        self._ws_connection: Optional[websockets.WebSocketServerProtocol] = None
        self._ws_subscriptions: Dict[str, callable] = {}
        
        # Base URLs
        self.base_url = "https://api.bitget.com"
        self.ws_url = "wss://ws.bitget.com/mix/v1/stream"
        
        logger.info(f"Initialized AsyncBitgetClient (sandbox: {self.api_config.bitget_sandbox})")
    
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = '') -> str:
        """Generate API signature"""
        message = timestamp + method + request_path + body
        signature = base64.b64encode(
            hmac.new(
                self.api_config.bitget_secret_key.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        return signature
    
    def _get_headers(self, method: str, request_path: str, body: str = '') -> Dict[str, str]:
        """Generate request headers with authentication"""
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp, method, request_path, body)
        
        return {
            'ACCESS-KEY': self.api_config.bitget_api_key,
            'ACCESS-SIGN': signature,
            'ACCESS-TIMESTAMP': timestamp,
            'ACCESS-PASSPHRASE': self.api_config.bitget_passphrase,
            'Content-Type': 'application/json'
        }
    
    async def _rate_limit(self) -> None:
        """Apply rate limiting"""
        async with self._request_lock:
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            
            if time_since_last < self.api_config.rate_limit_delay:
                sleep_time = self.api_config.rate_limit_delay - time_since_last
                await asyncio.sleep(sleep_time)
            
            self._last_request_time = time.time()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((ConnectionException, RateLimitException)),
        before_sleep=before_sleep_log(logger, "WARNING")
    )
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        authenticated: bool = True
    ) -> Dict[str, Any]:
        """
        Make HTTP request with retry logic
        Professional error handling and logging
        """
        await self._rate_limit()
        
        url = f"{self.base_url}{endpoint}"
        body = json.dumps(data) if data else ""
        
        headers = {}
        if authenticated:
            headers = self._get_headers(method.upper(), endpoint, body)
        else:
            headers = {"Content-Type": "application/json"}
        
        try:
            logger.debug(f"Making {method} request to {endpoint}")
            
            response = await self.client.request(
                method=method,
                url=url,
                params=params,
                content=body if data else None,
                headers=headers
            )
            
            # Handle different response codes
            if response.status_code == 429:
                raise RateLimitException("Rate limit exceeded")
            
            if response.status_code == 401:
                raise AuthenticationException("Authentication failed")
            
            if response.status_code >= 500:
                raise ConnectionException(f"Server error: {response.status_code}")
            
            if response.status_code >= 400:
                error_data = response.json() if response.content else {}
                raise APIException(f"API error: {response.status_code}", response_data=error_data)
            
            result = response.json()
            
            # Check API response code
            if result.get('code') != '00000':
                raise APIException(f"API returned error: {result.get('msg', 'Unknown error')}", 
                                 code=result.get('code'), response_data=result)
            
            return result
            
        except httpx.RequestError as e:
            logger.error(f"Request error: {e}")
            raise ConnectionException(f"Request failed: {e}")
        
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise APIException(f"Invalid JSON response: {e}")
    
    async def get_futures_ticker(self, symbol: str) -> Optional[TickerData]:
        """
        Get futures ticker data with structured response
        """
        endpoint = f'/api/v2/mix/market/ticker'
        params = {'symbol': symbol, 'productType': 'USDT-FUTURES'}
        
        try:
            result = await self._make_request('GET', endpoint, params=params, authenticated=False)
            
            tickers = result.get('data', [])
            if not tickers:
                logger.warning(f"No ticker data for {symbol}")
                return None
            
            ticker = tickers[0]
            
            return TickerData(
                symbol=symbol,
                price=float(ticker['lastPr']),
                open_24h=float(ticker.get('open24h', ticker['lastPr'])),
                high_24h=float(ticker['high24h']),
                low_24h=float(ticker['low24h']),
                volume=float(ticker.get('baseVolume', 0)),
                change_24h=float(ticker.get('change24h', 0)),
                ask_price=float(ticker['askPr']),
                bid_price=float(ticker['bidPr']),
                spread=float(ticker['askPr']) - float(ticker['bidPr']),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error fetching ticker for {symbol}: {e}")
            return None
    
    async def get_multiple_tickers(self, symbols: List[str]) -> Dict[str, TickerData]:
        """Get multiple tickers concurrently"""
        tasks = [self.get_futures_ticker(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        tickers = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, TickerData):
                tickers[symbol] = result
            elif isinstance(result, Exception):
                logger.error(f"Error fetching {symbol}: {result}")
        
        return tickers
    
    async def get_klines(
        self,
        symbol: str,
        granularity: str = '1m',
        limit: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get kline/candlestick data"""
        endpoint = '/api/v2/mix/market/candles'
        
        params = {
            'symbol': symbol,
            'granularity': granularity,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = int(start_time.timestamp() * 1000)
        if end_time:
            params['endTime'] = int(end_time.timestamp() * 1000)
        
        try:
            result = await self._make_request('GET', endpoint, params=params, authenticated=False)
            
            candles = result.get('data', [])
            klines = []
            
            for candle in candles:
                klines.append({
                    'timestamp': datetime.fromtimestamp(int(candle[0]) / 1000),
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5])
                })
            
            return sorted(klines, key=lambda x: x['timestamp'])
            
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            return []
    
    async def place_order(
        self,
        symbol: str,
        side: OrderSide,
        size: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        client_order_id: Optional[str] = None
    ) -> Optional[OrderResult]:
        """
        Place futures order with proper validation
        """
        endpoint = '/api/mix/v1/order/placeOrder'
        
        order_data = {
            'symbol': f"{symbol}_UMCBL",  # Futures symbol format
            'marginCoin': 'USDT',
            'side': side.value,
            'orderType': order_type.value,
            'size': str(size)
        }
        
        if order_type == OrderType.LIMIT and price:
            order_data['price'] = str(price)
        
        if client_order_id:
            order_data['clientOid'] = client_order_id
        
        try:
            result = await self._make_request('POST', endpoint, data=order_data)
            
            order_info = result.get('data', {})
            
            return OrderResult(
                order_id=order_info['orderId'],
                client_order_id=order_info.get('clientOid'),
                symbol=symbol,
                side=side,
                size=size,
                order_type=order_type,
                status='submitted',
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error placing order for {symbol}: {e}")
            return None
    
    async def get_account_balance(self) -> Optional[Dict[str, Any]]:
        """Get account balance and positions"""
        endpoint = '/api/mix/v1/account/accounts'
        
        try:
            result = await self._make_request('GET', endpoint)
            accounts = result.get('data', [])
            
            balances = {}
            for account in accounts:
                if account['marginCoin'] == 'USDT':
                    balances['USDT'] = {
                        'available': float(account['available']),
                        'frozen': float(account['frozen']),
                        'equity': float(account['equity']),
                        'unrealized_pnl': float(account['unrealizedPL']),
                        'margin_ratio': float(account.get('marginRatio', 0))
                    }
            
            return balances
            
        except Exception as e:
            logger.error(f"Error fetching account balance: {e}")
            return None
    
    async def start_websocket(self, on_message: callable = None) -> None:
        """Start WebSocket connection for real-time data"""
        try:
            self._ws_connection = await websockets.connect(self.ws_url)
            logger.info("WebSocket connected successfully")
            
            # Start message handler
            asyncio.create_task(self._ws_message_handler(on_message))
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
    
    async def _ws_message_handler(self, on_message: callable = None) -> None:
        """Handle WebSocket messages"""
        try:
            async for message in self._ws_connection:
                data = json.loads(message)
                
                if on_message:
                    await on_message(data)
                
                # Handle specific message types
                if 'event' in data:
                    logger.debug(f"WebSocket event: {data['event']}")
                
        except websockets.exceptions.ConnectionClosed:
            logger.warning("WebSocket connection closed")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
    
    async def subscribe_ticker(self, symbols: List[str]) -> None:
        """Subscribe to ticker updates via WebSocket"""
        if not self._ws_connection:
            await self.start_websocket()
        
        for symbol in symbols:
            subscribe_msg = {
                "op": "subscribe",
                "args": [{"instType": "mc", "channel": "ticker", "instId": f"{symbol}_UMCBL"}]
            }
            
            await self._ws_connection.send(json.dumps(subscribe_msg))
            logger.info(f"Subscribed to {symbol} ticker updates")
    
    async def close(self) -> None:
        """Clean shutdown of connections"""
        if self._ws_connection:
            await self._ws_connection.close()
        
        await self.client.aclose()
        logger.info("AsyncBitgetClient closed successfully")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()


# Example usage and testing
async def test_async_client():
    """Test the async client functionality"""
    settings = get_settings()
    
    async with AsyncBitgetClient(settings) as client:
        # Test ticker data
        ticker = await client.get_futures_ticker('BTCUSDT')
        if ticker:
            logger.info(f"BTC Price: ${ticker.price:,.2f}")
        
        # Test multiple tickers
        tickers = await client.get_multiple_tickers(['BTCUSDT', 'ETHUSDT'])
        logger.info(f"Retrieved {len(tickers)} tickers")
        
        # Test klines
        klines = await client.get_klines('BTCUSDT', '1m', 5)
        logger.info(f"Retrieved {len(klines)} klines")


if __name__ == "__main__":
    asyncio.run(test_async_client())