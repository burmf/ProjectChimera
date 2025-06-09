import asyncio
import json
import time
import hmac
import hashlib
import base64
import websockets
import logging
from typing import Dict, List, Callable, Optional
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

class BitgetWebSocketClient:
    def __init__(self):
        self.api_key = os.getenv('BITGET_API_KEY')
        self.secret_key = os.getenv('BITGET_SECRET_KEY')
        self.passphrase = os.getenv('BITGET_PASSPHRASE')
        self.ws_url = os.getenv('BITGET_WS_URL', 'wss://ws.bitget.com/spot/v1/stream')
        
        self.websocket = None
        self.is_connected = False
        self.subscriptions = {}
        self.callbacks = {}
        
        self.logger = logging.getLogger(__name__)
        
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = '') -> str:
        """Bitget API署名を生成"""
        message = timestamp + method + request_path + body
        signature = base64.b64encode(
            hmac.new(
                self.secret_key.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        return signature
    
    def _create_auth_message(self) -> Dict:
        """WebSocket認証メッセージを作成"""
        timestamp = str(int(time.time()))
        signature = self._generate_signature(timestamp, 'GET', '/user/verify', '')
        
        return {
            "op": "login",
            "args": [{
                "apiKey": self.api_key,
                "passphrase": self.passphrase,
                "timestamp": timestamp,
                "sign": signature
            }]
        }
    
    async def connect(self) -> bool:
        """WebSocket接続を確立"""
        try:
            self.logger.info(f"Bitget WebSocketに接続中: {self.ws_url}")
            self.websocket = await websockets.connect(self.ws_url)
            self.is_connected = True
            
            # 認証
            auth_msg = self._create_auth_message()
            await self.websocket.send(json.dumps(auth_msg))
            
            # 認証レスポンスを待機
            response = await self.websocket.recv()
            auth_result = json.loads(response)
            
            if auth_result.get('event') == 'login' and auth_result.get('code') == '0':
                self.logger.info("Bitget WebSocket認証成功")
                return True
            else:
                self.logger.error(f"認証失敗: {auth_result}")
                return False
                
        except Exception as e:
            self.logger.error(f"WebSocket接続エラー: {e}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """WebSocket接続を切断"""
        if self.websocket:
            await self.websocket.close()
            self.is_connected = False
            self.logger.info("Bitget WebSocket接続を切断")
    
    async def subscribe_ticker(self, symbol: str, callback: Callable = None):
        """ティッカー情報を購読"""
        channel = f"spot/ticker:{symbol}"
        message = {
            "op": "subscribe",
            "args": [{"instType": "SPOT", "channel": "ticker", "instId": symbol}]
        }
        
        await self._subscribe(channel, message, callback)
    
    async def subscribe_orderbook(self, symbol: str, depth: str = "books5", callback: Callable = None):
        """オーダーブック情報を購読"""
        channel = f"spot/{depth}:{symbol}"
        message = {
            "op": "subscribe", 
            "args": [{"instType": "SPOT", "channel": depth, "instId": symbol}]
        }
        
        await self._subscribe(channel, message, callback)
    
    async def subscribe_trades(self, symbol: str, callback: Callable = None):
        """トレード情報を購読"""
        channel = f"spot/trade:{symbol}"
        message = {
            "op": "subscribe",
            "args": [{"instType": "SPOT", "channel": "trade", "instId": symbol}]
        }
        
        await self._subscribe(channel, message, callback)
    
    async def subscribe_kline(self, symbol: str, interval: str = "1m", callback: Callable = None):
        """ローソク足データを購読"""
        channel = f"spot/candle{interval}:{symbol}"
        message = {
            "op": "subscribe",
            "args": [{"instType": "SPOT", "channel": f"candle{interval}", "instId": symbol}]
        }
        
        await self._subscribe(channel, message, callback)
    
    async def _subscribe(self, channel: str, message: Dict, callback: Callable = None):
        """チャンネル購読の共通処理"""
        if not self.is_connected:
            raise Exception("WebSocket未接続")
        
        await self.websocket.send(json.dumps(message))
        self.subscriptions[channel] = message
        
        if callback:
            self.callbacks[channel] = callback
        
        self.logger.info(f"チャンネル購読: {channel}")
    
    async def unsubscribe(self, channel: str):
        """チャンネル購読を解除"""
        if channel in self.subscriptions:
            message = self.subscriptions[channel].copy()
            message["op"] = "unsubscribe"
            
            await self.websocket.send(json.dumps(message))
            del self.subscriptions[channel]
            
            if channel in self.callbacks:
                del self.callbacks[channel]
            
            self.logger.info(f"チャンネル購読解除: {channel}")
    
    async def listen(self):
        """メッセージを継続的に受信"""
        try:
            while self.is_connected:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                # ピン応答
                if data.get('ping'):
                    pong = {"pong": data['ping']}
                    await self.websocket.send(json.dumps(pong))
                    continue
                
                # データメッセージ処理
                if 'arg' in data and 'data' in data:
                    await self._handle_data_message(data)
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("WebSocket接続が切断されました")
            self.is_connected = False
        except Exception as e:
            self.logger.error(f"メッセージ受信エラー: {e}")
    
    async def _handle_data_message(self, data: Dict):
        """データメッセージを処理"""
        arg = data.get('arg', {})
        channel = arg.get('channel')
        inst_id = arg.get('instId')
        
        if not channel or not inst_id:
            return
        
        # チャンネル識別子を作成
        channel_key = f"spot/{channel}:{inst_id}"
        
        # 対応するコールバックを実行
        if channel_key in self.callbacks:
            try:
                await self.callbacks[channel_key](data)
            except Exception as e:
                self.logger.error(f"コールバック実行エラー: {e}")
        
        # ログ出力
        self.logger.debug(f"受信データ [{channel_key}]: {data}")
    
    async def get_account_info(self):
        """アカウント情報を取得（プライベートチャンネル）"""
        if not self.is_connected:
            raise Exception("WebSocket未接続")
        
        message = {
            "op": "subscribe",
            "args": [{"instType": "SPOT", "channel": "account", "coin": "USDT"}]
        }
        
        await self.websocket.send(json.dumps(message))
        self.logger.info("アカウント情報チャンネルを購読")


class BitgetDataProcessor:
    """Bitgetデータの処理・変換クラス"""
    
    def __init__(self, database_adapter=None):
        self.db = database_adapter
        self.logger = logging.getLogger(__name__)
    
    async def process_ticker_data(self, data: Dict):
        """ティッカーデータを処理"""
        try:
            ticker_data = data['data'][0]
            processed = {
                'symbol': data['arg']['instId'],
                'price': float(ticker_data['last']),
                'bid': float(ticker_data['bidPx']),
                'ask': float(ticker_data['askPx']),
                'volume': float(ticker_data['baseVolume']),
                'change_24h': float(ticker_data['change24h']),
                'timestamp': datetime.now()
            }
            
            self.logger.info(f"ティッカー: {processed['symbol']} - 価格: {processed['price']}")
            
            # データベースに保存
            if self.db:
                await self._save_ticker_to_db(processed)
            
            return processed
            
        except Exception as e:
            self.logger.error(f"ティッカーデータ処理エラー: {e}")
    
    async def process_orderbook_data(self, data: Dict):
        """オーダーブックデータを処理"""
        try:
            book_data = data['data'][0]
            processed = {
                'symbol': data['arg']['instId'],
                'bids': [[float(bid[0]), float(bid[1])] for bid in book_data['bids']],
                'asks': [[float(ask[0]), float(ask[1])] for ask in book_data['asks']],
                'timestamp': datetime.now()
            }
            
            self.logger.debug(f"オーダーブック: {processed['symbol']}")
            return processed
            
        except Exception as e:
            self.logger.error(f"オーダーブックデータ処理エラー: {e}")
    
    async def _save_ticker_to_db(self, ticker_data: Dict):
        """ティッカーデータをデータベースに保存"""
        if not self.db:
            return
        
        try:
            # 既存のprice_dataテーブルに保存
            query = """
            INSERT INTO price_data (pair, timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            await self.db.execute_query(query, (
                ticker_data['symbol'],
                ticker_data['timestamp'],
                ticker_data['price'],  # open として使用
                ticker_data['price'],  # high として使用  
                ticker_data['price'],  # low として使用
                ticker_data['price'],  # close
                ticker_data['volume']
            ))
            
        except Exception as e:
            self.logger.error(f"ティッカーデータ保存エラー: {e}")