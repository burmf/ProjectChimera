"""
Bitget API Client for Real Market Data
Provides real-time market data from Bitget exchange
"""

import asyncio
import json
import logging
import hmac
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import httpx
from decimal import Decimal

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
    
    def __init__(self, api_key: str = "", secret_key: str = "", passphrase: str = "", sandbox: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.sandbox = sandbox
        
        # API endpoints
        if sandbox:
            self.base_url = "https://api.bitget.com"  # Sandbox URL
        else:
            self.base_url = "https://api.bitget.com"  # Production URL
        
        # HTTP client
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=10.0,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20)
        )
        
        # Rate limiting
        self.last_request_time = 0.0
        self.min_request_interval = 0.1  # 100ms between requests
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()
    
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = "") -> str:
        """Generate request signature for authenticated endpoints"""
        if not self.secret_key:
            return ""
        
        message = timestamp + method.upper() + request_path + body
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _get_headers(self, method: str, request_path: str, body: str = "") -> Dict[str, str]:
        """Get request headers with authentication"""
        timestamp = str(int(time.time() * 1000))
        
        headers = {
            'Content-Type': 'application/json',
            'ACCESS-KEY': self.api_key,
            'ACCESS-SIGN': self._generate_signature(timestamp, method, request_path, body),
            'ACCESS-TIMESTAMP': timestamp,
            'ACCESS-PASSPHRASE': self.passphrase,
        }
        
        return headers
    
    async def _make_request(self, method: str, endpoint: str, params: Dict = None, auth_required: bool = False) -> Dict[str, Any]:
        """Make HTTP request with rate limiting and error handling"""
        
        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
        
        try:
            if auth_required and self.api_key:
                headers = self._get_headers(method, endpoint)
            else:
                headers = {'Content-Type': 'application/json'}
            
            if method.upper() == 'GET':
                response = await self.client.get(endpoint, params=params, headers=headers)
            else:
                response = await self.client.post(endpoint, json=params, headers=headers)
            
            response.raise_for_status()
            data = response.json()
            
            # Check Bitget API response format
            if data.get('code') == '00000':  # Bitget success code
                return data.get('data', {})
            else:
                logger.error(f"Bitget API error: {data}")
                return {}
                
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return {}
    
    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get real-time ticker data"""
        endpoint = "/api/spot/v1/market/ticker"
        params = {"symbol": symbol}
        
        data = await self._make_request("GET", endpoint, params)
        
        if data:
            return {
                'symbol': symbol,
                'price': float(data.get('close', 0)),
                'bid': float(data.get('bidPr', 0)),
                'ask': float(data.get('askPr', 0)),
                'volume_24h': float(data.get('baseVolume', 0)),
                'change_24h': float(data.get('change', 0)),
                'high_24h': float(data.get('high24h', 0)),
                'low_24h': float(data.get('low24h', 0)),
                'timestamp': datetime.now(),
                'source': 'bitget'
            }
        
        return {}
    
    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get market depth (orderbook)"""
        endpoint = "/api/spot/v1/market/depth"
        params = {"symbol": symbol, "limit": str(limit), "type": "step0"}
        
        data = await self._make_request("GET", endpoint, params)
        
        if data:
            return {
                'symbol': symbol,
                'bids': [[float(bid[0]), float(bid[1])] for bid in data.get('bids', [])],
                'asks': [[float(ask[0]), float(ask[1])] for ask in data.get('asks', [])],
                'timestamp': datetime.now()
            }
        
        return {}
    
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades"""
        endpoint = "/api/spot/v1/market/fills"
        params = {"symbol": symbol, "limit": str(limit)}
        
        data = await self._make_request("GET", endpoint, params)
        
        if data:
            trades = []
            for trade in data:
                trades.append({
                    'symbol': symbol,
                    'price': float(trade.get('price', 0)),
                    'size': float(trade.get('size', 0)),
                    'side': trade.get('side', 'buy'),
                    'timestamp': datetime.fromtimestamp(int(trade.get('ts', 0)) / 1000),
                    'trade_id': trade.get('tradeId', '')
                })
            return trades
        
        return []
    
    async def get_kline(self, symbol: str, period: str = "1m", limit: int = 200) -> List[Dict[str, Any]]:
        """Get candlestick data"""
        endpoint = "/api/spot/v1/market/candles"
        params = {
            "symbol": symbol,
            "period": period,
            "limit": str(limit)
        }
        
        data = await self._make_request("GET", endpoint, params)
        
        if data:
            candles = []
            for candle in data:
                candles.append({
                    'symbol': symbol,
                    'timestamp': datetime.fromtimestamp(int(candle[0]) / 1000),
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5]),
                    'period': period
                })
            return sorted(candles, key=lambda x: x['timestamp'])
        
        return []
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information (requires authentication)"""
        if not self.api_key:
            return {'error': 'Authentication required'}
        
        endpoint = "/api/spot/v1/account/assets"
        
        data = await self._make_request("GET", endpoint, auth_required=True)
        
        if data:
            assets = {}
            total_value_usdt = 0.0
            
            for asset in data:
                coin = asset.get('coin', '')
                available = float(asset.get('available', 0))
                frozen = float(asset.get('frozen', 0))
                total = available + frozen
                
                if total > 0:
                    assets[coin] = {
                        'available': available,
                        'frozen': frozen,
                        'total': total
                    }
                    
                    # Estimate USDT value (simplified)
                    if coin == 'USDT':
                        total_value_usdt += total
                    elif coin == 'BTC':
                        # You could fetch BTC price here for accurate conversion
                        total_value_usdt += total * 50000  # Rough estimate
            
            return {
                'assets': assets,
                'total_value_usdt': total_value_usdt,
                'timestamp': datetime.now()
            }
        
        return {}
    
    async def get_futures_account(self) -> Dict[str, Any]:
        """Get futures account information (requires authentication)"""
        if not self.api_key:
            return {'error': 'Authentication required'}
        
        endpoint = "/api/mix/v1/account/account"
        params = {"symbol": "BTCUSDT_UMCBL", "marginCoin": "USDT"}
        
        data = await self._make_request("GET", endpoint, params, auth_required=True)
        
        if data:
            return {
                'margin_coin': data.get('marginCoin', 'USDT'),
                'locked': float(data.get('locked', 0)),
                'available': float(data.get('available', 0)),
                'crossMaxAvailable': float(data.get('crossMaxAvailable', 0)),
                'fixedMaxAvailable': float(data.get('fixedMaxAvailable', 0)),
                'maxTransferOut': float(data.get('maxTransferOut', 0)),
                'equity': float(data.get('equity', 0)),
                'usdtEquity': float(data.get('usdtEquity', 0)),
                'unrealizedPL': float(data.get('unrealizedPL', 0)),
                'timestamp': datetime.now()
            }
        
        return {}
    
    async def get_futures_positions(self) -> List[Dict[str, Any]]:
        """Get futures positions (requires authentication)"""
        if not self.api_key:
            return []
        
        endpoint = "/api/mix/v1/position/allPosition"
        params = {"productType": "umcbl"}
        
        data = await self._make_request("GET", endpoint, params, auth_required=True)
        
        if data:
            positions = []
            for pos in data:
                if float(pos.get('total', 0)) != 0:  # Only active positions
                    positions.append({
                        'symbol': pos.get('symbol', ''),
                        'size': float(pos.get('total', 0)),
                        'available': float(pos.get('available', 0)),
                        'averageOpenPrice': float(pos.get('averageOpenPrice', 0)),
                        'markPrice': float(pos.get('markPrice', 0)),
                        'unrealizedPL': float(pos.get('unrealizedPL', 0)),
                        'side': pos.get('holdSide', ''),
                        'marginMode': pos.get('marginMode', ''),
                        'leverage': pos.get('leverage', ''),
                        'marginSize': float(pos.get('marginSize', 0)),
                        'timestamp': datetime.now()
                    })
            return positions
        
        return []
    
    async def get_multi_ticker(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
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
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get Bitget system status"""
        endpoint = "/api/spot/v1/public/time"
        
        data = await self._make_request("GET", endpoint)
        
        if data:
            return {
                'status': 'online',
                'server_time': datetime.fromtimestamp(int(data) / 1000),
                'timestamp': datetime.now()
            }
        
        return {'status': 'offline', 'timestamp': datetime.now()}


class BitgetMarketDataService:
    """
    High-level service for Bitget market data
    Provides simplified interface for dashboard consumption
    """
    
    def __init__(self, api_key: str = "", secret_key: str = "", passphrase: str = "", sandbox: bool = True):
        self.client = BitgetAPIClient(api_key, secret_key, passphrase, sandbox)
        self.cache = {}
        self.cache_ttl = 30  # 30 seconds cache
        
    async def close(self):
        """Close the service"""
        await self.client.close()
    
    async def get_market_overview(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Get comprehensive market overview"""
        if symbols is None:
            # Use Bitget's actual symbol format with _SPBL suffix for spot trading
            symbols = ['BTCUSDT_SPBL', 'ETHUSDT_SPBL', 'BNBUSDT_SPBL', 'ADAUSDT_SPBL', 'DOTUSDT_SPBL']
        
        try:
            # Get ticker data for multiple symbols
            tickers = await self.client.get_multi_ticker(symbols)
            
            # Calculate market statistics
            total_volume = sum(ticker.get('volume_24h', 0) for ticker in tickers.values())
            avg_change = sum(ticker.get('change_24h', 0) for ticker in tickers.values()) / len(tickers)
            
            # Get system status
            system_status = await self.client.get_system_status()
            
            return {
                'tickers': tickers,
                'market_stats': {
                    'total_volume_24h': total_volume,
                    'average_change_24h': avg_change,
                    'active_symbols': len([t for t in tickers.values() if t.get('price', 0) > 0])
                },
                'system_status': system_status,
                'timestamp': datetime.now(),
                'source': 'bitget'
            }
            
        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    async def get_portfolio_value(self) -> Dict[str, Any]:
        """Get real portfolio value from Bitget account"""
        try:
            # Try futures account first (primary for ProjectChimera)
            futures_account = await self.client.get_futures_account()
            
            if 'error' not in futures_account and futures_account.get('equity', 0) > 0:
                # Get futures positions for additional P&L data
                positions = await self.client.get_futures_positions()
                
                total_unrealized_pnl = sum(pos.get('unrealizedPL', 0) for pos in positions)
                
                return {
                    'total_value_usdt': futures_account.get('usdtEquity', 150000.0),
                    'equity': futures_account.get('equity', 150000.0),
                    'available': futures_account.get('available', 0),
                    'locked': futures_account.get('locked', 0),
                    'unrealized_pnl': total_unrealized_pnl,
                    'positions': positions,
                    'account_type': 'futures',
                    'demo_mode': False,
                    'timestamp': datetime.now()
                }
            
            # Fallback to spot account
            account_info = await self.client.get_account_info()
            
            if 'error' not in account_info:
                return {
                    'total_value_usdt': account_info.get('total_value_usdt', 150000.0),
                    'assets': account_info.get('assets', {}),
                    'unrealized_pnl': 0.0,  # Spot doesn't have unrealized P&L
                    'account_type': 'spot',
                    'demo_mode': False,
                    'timestamp': datetime.now()
                }
            
            # Authentication failed or no data - use demo mode
            return {
                'total_value_usdt': 150000.0,  # Default demo value
                'equity': 150000.0,
                'unrealized_pnl': 0.0,
                'error': 'Authentication required or API unavailable',
                'demo_mode': True,
                'account_type': 'demo',
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio value: {e}")
            return {
                'total_value_usdt': 150000.0,
                'equity': 150000.0,
                'unrealized_pnl': 0.0,
                'error': str(e),
                'demo_mode': True,
                'account_type': 'demo',
                'timestamp': datetime.now()
            }
    
    async def get_price_history(self, symbol: str, period: str = "1m", hours: int = 24) -> List[Dict[str, Any]]:
        """Get price history for equity curve"""
        try:
            # Calculate number of candles needed
            period_minutes = {'1m': 1, '5m': 5, '15m': 15, '1h': 60, '4h': 240, '1d': 1440}
            minutes = period_minutes.get(period, 1)
            limit = min((hours * 60) // minutes, 1000)  # Bitget limit
            
            candles = await self.client.get_kline(symbol, period, limit)
            
            return candles
            
        except Exception as e:
            logger.error(f"Error getting price history: {e}")
            return []


# Global instance for easy access
_bitget_service: Optional[BitgetMarketDataService] = None

def get_bitget_service() -> BitgetMarketDataService:
    """Get global Bitget service instance"""
    global _bitget_service
    if _bitget_service is None:
        # Initialize with demo credentials (replace with real ones for live trading)
        _bitget_service = BitgetMarketDataService(sandbox=True)
    return _bitget_service


async def demo_bitget_data():
    """Demo function to test Bitget data fetching"""
    service = get_bitget_service()
    
    try:
        print("🔍 Testing Bitget API connection...")
        
        # Test system status
        overview = await service.get_market_overview()
        print(f"✅ Market Overview: {len(overview.get('tickers', {}))} symbols")
        
        # Test individual ticker
        btc_ticker = await service.client.get_ticker('BTCUSDT')
        print(f"💰 BTC Price: ${btc_ticker.get('price', 0):,.2f}")
        
        # Test portfolio (will show demo mode if not authenticated)
        portfolio = await service.get_portfolio_value()
        print(f"💼 Portfolio Value: ${portfolio.get('total_value_usdt', 0):,.2f}")
        print(f"📊 Demo Mode: {portfolio.get('demo_mode', True)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing Bitget API: {e}")
        return False
    finally:
        await service.close()


if __name__ == "__main__":
    # Test the Bitget API
    asyncio.run(demo_bitget_data())