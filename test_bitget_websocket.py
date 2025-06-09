#!/usr/bin/env python3
"""
Bitget WebSocket接続テストスクリプト
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.bitget_websocket import BitgetWebSocketClient, BitgetDataProcessor
from core.database_adapter import DatabaseAdapter

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BitgetWebSocketTester:
    def __init__(self):
        self.client = BitgetWebSocketClient()
        self.processor = BitgetDataProcessor()
        self.received_messages = []
        
    async def test_connection(self):
        """基本的な接続テスト"""
        logger.info("=== Bitget WebSocket接続テスト開始 ===")
        
        try:
            # 接続
            success = await self.client.connect()
            if not success:
                logger.error("接続に失敗しました")
                return False
            
            logger.info("✅ WebSocket接続成功")
            
            # 少し待機
            await asyncio.sleep(2)
            
            # 切断
            await self.client.disconnect()
            logger.info("✅ 切断完了")
            
            return True
            
        except Exception as e:
            logger.error(f"接続テストエラー: {e}")
            return False
    
    async def test_ticker_subscription(self, symbol: str = "BTCUSDT"):
        """ティッカー購読テスト"""
        logger.info(f"=== ティッカー購読テスト: {symbol} ===")
        
        try:
            # 接続
            await self.client.connect()
            
            # ティッカー購読
            await self.client.subscribe_ticker(symbol, self.on_ticker_update)
            
            # リスニング開始
            listen_task = asyncio.create_task(self.client.listen())
            
            # 30秒間データを受信
            await asyncio.sleep(30)
            
            # タスクキャンセル
            listen_task.cancel()
            
            # 切断
            await self.client.disconnect()
            
            logger.info(f"✅ 受信メッセージ数: {len(self.received_messages)}")
            
        except Exception as e:
            logger.error(f"ティッカー購読テストエラー: {e}")
    
    async def test_multiple_subscriptions(self):
        """複数チャンネル購読テスト"""
        logger.info("=== 複数チャンネル購読テスト ===")
        
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        
        try:
            # 接続
            await self.client.connect()
            
            # 複数シンボルのティッカーを購読
            for symbol in symbols:
                await self.client.subscribe_ticker(symbol, self.on_ticker_update)
                await asyncio.sleep(1)  # 少し間隔を空ける
            
            # オーダーブックも購読
            await self.client.subscribe_orderbook("BTCUSDT", callback=self.on_orderbook_update)
            
            # リスニング開始
            listen_task = asyncio.create_task(self.client.listen())
            
            # 60秒間データを受信
            await asyncio.sleep(60)
            
            # タスクキャンセル
            listen_task.cancel()
            
            # 切断
            await self.client.disconnect()
            
            logger.info(f"✅ 総受信メッセージ数: {len(self.received_messages)}")
            
        except Exception as e:
            logger.error(f"複数チャンネル購読テストエラー: {e}")
    
    async def on_ticker_update(self, data):
        """ティッカー更新コールバック"""
        try:
            processed = await self.processor.process_ticker_data(data)
            self.received_messages.append(processed)
            
            if processed:
                logger.info(f"ティッカー更新: {processed['symbol']} = ${processed['price']:.4f}")
                
        except Exception as e:
            logger.error(f"ティッカー処理エラー: {e}")
    
    async def on_orderbook_update(self, data):
        """オーダーブック更新コールバック"""
        try:
            processed = await self.processor.process_orderbook_data(data)
            
            if processed:
                best_bid = processed['bids'][0][0] if processed['bids'] else 0
                best_ask = processed['asks'][0][0] if processed['asks'] else 0
                spread = best_ask - best_bid if best_bid and best_ask else 0
                
                logger.info(f"オーダーブック: {processed['symbol']} - Bid: ${best_bid:.4f}, Ask: ${best_ask:.4f}, Spread: ${spread:.4f}")
                
        except Exception as e:
            logger.error(f"オーダーブック処理エラー: {e}")


async def main():
    """メイン実行関数"""
    tester = BitgetWebSocketTester()
    
    try:
        # 1. 基本接続テスト
        connection_ok = await tester.test_connection()
        if not connection_ok:
            logger.error("基本接続テストに失敗しました")
            return
        
        print("\n" + "="*50)
        
        # 2. ティッカー購読テスト
        await tester.test_ticker_subscription("BTCUSDT")
        
        print("\n" + "="*50)
        
        # 3. 複数チャンネル購読テスト
        await tester.test_multiple_subscriptions()
        
        logger.info("🎉 全てのテストが完了しました")
        
    except KeyboardInterrupt:
        logger.info("テストが中断されました")
    except Exception as e:
        logger.error(f"テスト実行エラー: {e}")


if __name__ == "__main__":
    # 環境変数チェック
    required_env_vars = ['BITGET_API_KEY', 'BITGET_SECRET_KEY']
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"必要な環境変数が設定されていません: {missing_vars}")
        logger.error("'.env'ファイルを確認してください")
        sys.exit(1)
    
    # テスト実行
    asyncio.run(main())