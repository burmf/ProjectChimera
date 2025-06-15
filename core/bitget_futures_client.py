#!/usr/bin/env python3
"""
Bitget Futures Trading Client
Bitget先物取引クライアント
"""

import requests
import hmac
import hashlib
import base64
import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import os

# Load environment variables manually
def load_env_vars():
    env_vars = {}
    try:
        with open('/home/ec2-user/ProjectChimera/.env', 'r') as f:
            for line in f:
                if '=' in line and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    env_vars[key] = value
    except FileNotFoundError:
        pass
    return env_vars

env_vars = load_env_vars()


class BitgetFuturesClient:
    """
    Bitget先物取引クライアント
    
    - USDT-M先物取引
    - 高レバレッジ対応
    - リアルタイム価格取得
    - 注文管理
    """
    
    def __init__(self):
        self.api_key = env_vars.get('BITGET_API_KEY', '')
        self.secret_key = env_vars.get('BITGET_SECRET_KEY', '')
        self.passphrase = env_vars.get('BITGET_PASSPHRASE', '')
        self.base_url = 'https://api.bitget.com'
        
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting
        self.request_delay = 0.1  # 100ms between requests
        self.last_request_time = 0
        
        # Futures symbols (v2 API uses clean symbols)
        self.futures_symbols = [
            'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 
            'BNBUSDT', 'ADAUSDT', 'DOGEUSDT'
        ]
    
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = '') -> str:
        """API署名生成"""
        message = timestamp + method + request_path + body
        signature = base64.b64encode(
            hmac.new(
                self.secret_key.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        return signature
    
    def _get_headers(self, method: str, request_path: str, body: str = '') -> Dict[str, str]:
        """APIヘッダー生成"""
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp, method, request_path, body)
        
        return {
            'ACCESS-KEY': self.api_key,
            'ACCESS-SIGN': signature,
            'ACCESS-TIMESTAMP': timestamp,
            'ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }
    
    def _rate_limit(self):
        """レート制限適用"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        self.last_request_time = time.time()
    
    def get_futures_ticker(self, symbol: str) -> Optional[Dict[str, Any]]:
        """先物ティッカー情報取得"""
        self._rate_limit()
        
        # Use v2 API with correct format
        endpoint = f'/api/v2/mix/market/ticker?symbol={symbol}&productType=USDT-FUTURES'
        
        try:
            response = self.session.get(
                self.base_url + endpoint,
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    tickers = data['data']
                    if tickers and len(tickers) > 0:
                        ticker = tickers[0]  # First ticker in list
                        return {
                            'symbol': symbol,
                            'price': float(ticker['lastPr']),
                            'open': float(ticker.get('open24h', ticker['lastPr'])),
                            'high': float(ticker['high24h']),
                            'low': float(ticker['low24h']),
                            'volume': float(ticker.get('baseVolume', 0)),
                            'change_24h': float(ticker.get('change24h', 0)),
                            'ask_price': float(ticker['askPr']),
                            'bid_price': float(ticker['bidPr']),
                            'timestamp': datetime.now()
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching futures ticker for {symbol}: {e}")
            return None
    
    def get_multiple_tickers(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """複数シンボルのティッカー一括取得"""
        tickers = {}
        
        for symbol in symbols:
            ticker = self.get_futures_ticker(symbol)
            if ticker:
                tickers[symbol] = ticker
            
            # 少し間隔を空ける
            time.sleep(0.05)
        
        return tickers
    
    def get_futures_klines(self, symbol: str, granularity: str = '1m', limit: int = 100) -> List[Dict]:
        """先物K線データ取得"""
        self._rate_limit()
        
        # Use v2 API with clean symbol
        endpoint = f'/api/v2/mix/market/candles'
        
        params = {
            'symbol': symbol,
            'granularity': granularity,
            'limit': limit
        }
        
        try:
            response = self.session.get(
                self.base_url + endpoint,
                params=params,
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    candles = data.get('data', [])
                    
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
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error fetching klines for {symbol}: {e}")
            return []
    
    def get_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict[str, Any]]:
        """オーダーブック取得"""
        self._rate_limit()
        
        # Use v2 API
        endpoint = f'/api/v2/mix/market/orderbook'
        
        params = {
            'symbol': symbol,
            'limit': limit
        }
        
        try:
            response = self.session.get(
                self.base_url + endpoint,
                params=params,
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    depth = data['data']
                    return {
                        'symbol': symbol,
                        'bids': [[float(bid[0]), float(bid[1])] for bid in depth['bids']],
                        'asks': [[float(ask[0]), float(ask[1])] for ask in depth['asks']],
                        'timestamp': datetime.now()
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching order book for {symbol}: {e}")
            return None
    
    def get_funding_rate(self, symbol: str) -> Optional[Dict[str, Any]]:
        """資金調達率取得"""
        self._rate_limit()
        
        futures_symbol = self.futures_symbols.get(symbol, symbol + '_UMCBL')
        endpoint = f'/api/mix/v1/market/funding-time'
        
        params = {'symbol': futures_symbol}
        
        try:
            response = self.session.get(
                self.base_url + endpoint,
                params=params,
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    funding = data['data']
                    return {
                        'symbol': symbol,
                        'funding_rate': float(funding['fundingRate']),
                        'next_funding_time': funding['nextFundingTime'],
                        'funding_interval': funding.get('fundingInterval', '8h')
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching funding rate for {symbol}: {e}")
            return None
    
    def get_account_balance(self) -> Optional[Dict[str, Any]]:
        """先物アカウント残高取得"""
        self._rate_limit()
        
        endpoint = '/api/mix/v1/account/accounts'
        method = 'GET'
        
        headers = self._get_headers(method, endpoint)
        
        try:
            response = self.session.get(
                self.base_url + endpoint,
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    accounts = data.get('data', [])
                    
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
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error fetching account balance: {e}")
            return None
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """現在のポジション取得"""
        self._rate_limit()
        
        endpoint = '/api/mix/v1/position/allPosition'
        method = 'GET'
        
        headers = self._get_headers(method, endpoint)
        
        try:
            response = self.session.get(
                self.base_url + endpoint,
                headers=headers,
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    positions = data.get('data', [])
                    
                    active_positions = []
                    for pos in positions:
                        if float(pos['total']) != 0:  # アクティブなポジション
                            active_positions.append({
                                'symbol': pos['symbol'].replace('_UMCBL', ''),
                                'side': pos['holdSide'],
                                'size': float(pos['total']),
                                'available': float(pos['available']),
                                'avg_price': float(pos['averageOpenPrice']),
                                'mark_price': float(pos['markPrice']),
                                'unrealized_pnl': float(pos['unrealizedPL']),
                                'pnl_ratio': float(pos['unrealizedPLR']),
                                'margin': float(pos['margin']),
                                'leverage': int(pos['leverage'])
                            })
                    
                    return active_positions
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error fetching positions: {e}")
            return []
    
    def place_order(self, symbol: str, side: str, size: float, 
                   order_type: str = 'market', price: Optional[float] = None,
                   leverage: int = 10) -> Optional[Dict[str, Any]]:
        """注文発注"""
        self._rate_limit()
        
        futures_symbol = self.futures_symbols.get(symbol, symbol + '_UMCBL')
        endpoint = '/api/mix/v1/order/placeOrder'
        method = 'POST'
        
        order_data = {
            'symbol': futures_symbol,
            'marginCoin': 'USDT',
            'side': side,  # 'long' or 'short'
            'orderType': order_type,  # 'market' or 'limit'
            'size': str(size)
        }
        
        if order_type == 'limit' and price:
            order_data['price'] = str(price)
        
        body = json.dumps(order_data)
        headers = self._get_headers(method, endpoint, body)
        
        try:
            response = self.session.post(
                self.base_url + endpoint,
                headers=headers,
                data=body,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    order_info = data['data']
                    return {
                        'order_id': order_info['orderId'],
                        'client_order_id': order_info.get('clientOid'),
                        'symbol': symbol,
                        'side': side,
                        'size': size,
                        'order_type': order_type,
                        'status': 'submitted',
                        'timestamp': datetime.now()
                    }
                else:
                    self.logger.error(f"Order placement failed: {data}")
                    return None
            else:
                self.logger.error(f"Order placement HTTP error: {response.status_code}, {response.text}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None
    
    def cancel_order(self, symbol: str, order_id: str) -> bool:
        """注文キャンセル"""
        self._rate_limit()
        
        futures_symbol = self.futures_symbols.get(symbol, symbol + '_UMCBL')
        endpoint = '/api/mix/v1/order/cancel-order'
        method = 'POST'
        
        cancel_data = {
            'symbol': futures_symbol,
            'marginCoin': 'USDT',
            'orderId': order_id
        }
        
        body = json.dumps(cancel_data)
        headers = self._get_headers(method, endpoint, body)
        
        try:
            response = self.session.post(
                self.base_url + endpoint,
                headers=headers,
                data=body,
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('code') == '00000'
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error canceling order: {e}")
            return False
    
    def close_position(self, symbol: str, side: str, size: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """ポジション決済"""
        # ポジション情報取得
        positions = self.get_positions()
        target_position = None
        
        for pos in positions:
            if pos['symbol'] == symbol and pos['side'] == side:
                target_position = pos
                break
        
        if not target_position:
            self.logger.warning(f"No position found for {symbol} {side}")
            return None
        
        # 決済サイズ決定
        close_size = size if size else target_position['available']
        
        # 反対方向の注文で決済
        close_side = 'short' if side == 'long' else 'long'
        
        return self.place_order(
            symbol=symbol,
            side=close_side,
            size=close_size,
            order_type='market'
        )
    
    def get_market_summary(self) -> Dict[str, Any]:
        """市場サマリー取得"""
        summary = {
            'timestamp': datetime.now(),
            'symbols': {}
        }
        
        # 主要シンボルの情報取得
        major_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        
        for symbol in major_symbols:
            ticker = self.get_futures_ticker(symbol)
            if ticker:
                summary['symbols'][symbol] = {
                    'price': ticker['price'],
                    'change_24h': ticker['change_24h'],
                    'volume': ticker['volume'],
                    'funding_rate': ticker['funding_rate']
                }
        
        return summary


# テスト関数
def test_futures_client():
    """先物クライアントテスト"""
    client = BitgetFuturesClient()
    
    print("🧪 Testing Bitget Futures Client...")
    
    # ティッカー取得テスト
    print("\\n📊 Testing ticker data...")
    btc_ticker = client.get_futures_ticker('BTCUSDT')
    if btc_ticker:
        print(f"✅ BTC Futures Price: ${btc_ticker['price']:,.2f}")
        print(f"   24h Change: {btc_ticker['change_24h']:+.2f}%")
        print(f"   Ask: ${btc_ticker['ask_price']:,.2f}, Bid: ${btc_ticker['bid_price']:,.2f}")
    else:
        print("❌ Failed to get BTC ticker")
    
    # 複数ティッカー取得テスト
    print("\\n📈 Testing multiple tickers...")
    tickers = client.get_multiple_tickers(['BTCUSDT', 'ETHUSDT'])
    for symbol, ticker in tickers.items():
        print(f"✅ {symbol}: ${ticker['price']:,.2f} ({ticker['change_24h']:+.2f}%)")
    
    # K線データ取得テスト
    print("\\n📊 Testing kline data...")
    klines = client.get_futures_klines('BTCUSDT', '1m', 5)
    if klines:
        print(f"✅ Retrieved {len(klines)} klines")
        latest = klines[-1]
        print(f"   Latest: {latest['timestamp']} - ${latest['close']:,.2f}")
    else:
        print("❌ Failed to get klines")
    
    # オーダーブック取得テスト
    print("\\n📖 Testing order book...")
    order_book = client.get_order_book('BTCUSDT', 5)
    if order_book:
        print("✅ Order book retrieved")
        print(f"   Best bid: ${order_book['bids'][0][0]:,.2f}")
        print(f"   Best ask: ${order_book['asks'][0][0]:,.2f}")
        spread = order_book['asks'][0][0] - order_book['bids'][0][0]
        print(f"   Spread: ${spread:.2f}")
    else:
        print("❌ Failed to get order book")
    
    print("\\n🎯 Futures client test completed!")


if __name__ == "__main__":
    test_futures_client()