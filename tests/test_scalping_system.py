#!/usr/bin/env python3
"""
Bitget スキャルピングシステム統合テスト
データ収集からシグナル生成まで一貫テスト
"""

import sys
import os
import asyncio
import json
import pandas as pd
from datetime import datetime
import logging


from core.bitget_rest_client import BitgetRestClient, ScalpingDataCollector
from core.orderbook_analyzer import OrderbookAnalyzer
from modules.scalping_features import ScalpingFeatureEngine
from core.logging_config import setup_logging

# ログ設定
setup_logging()
logger = logging.getLogger(__name__)

async def test_rest_api_connection():
    """REST API接続テスト"""
    logger.info("🔌 Bitget REST API接続テスト")
    
    client = BitgetRestClient()
    
    # 基本的なマーケットデータ取得テスト
    try:
        ticker = client.get_ticker_24hr('BTCUSDT')
        if ticker:
            logger.info(f"✅ ティッカー取得成功: BTC価格=${float(ticker.get('lastPr', 0)):,.2f}")
        else:
            logger.error("❌ ティッカー取得失敗")
            return False
        
        # オーダーブック取得テスト
        orderbook = client.get_orderbook_snapshot('BTCUSDT', limit=10)
        if orderbook and 'bids' in orderbook and 'asks' in orderbook:
            bids = orderbook['bids']
            asks = orderbook['asks']
            if bids and asks:
                spread = float(asks[0][0]) - float(bids[0][0])
                logger.info(f"✅ オーダーブック取得成功: スプレッド=${spread:.4f}")
            else:
                logger.error("❌ オーダーブック形式エラー")
                return False
        else:
            logger.error("❌ オーダーブック取得失敗")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ REST API接続エラー: {e}")
        return False

async def test_historical_data_collection():
    """歴史データ収集テスト"""
    logger.info("📊 歴史データ収集テスト（短期間）")
    
    client = BitgetRestClient()
    
    try:
        # 1日分の1分足データ取得テスト
        from datetime import timedelta
        end_time = datetime.now()
        start_time = end_time - timedelta(days=1)
        
        df = client.get_kline_data('BTCUSDT', '1m', start_time, end_time)
        
        if not df.empty:
            logger.info(f"✅ 1分足データ取得成功: {len(df)}レコード")
            logger.info(f"   期間: {df['timestamp'].min()} - {df['timestamp'].max()}")
            logger.info(f"   価格範囲: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
            
            # データ品質チェック
            null_count = df.isnull().sum().sum()
            if null_count == 0:
                logger.info("✅ データ品質: 欠損値なし")
            else:
                logger.warning(f"⚠️ データ品質: {null_count}個の欠損値")
            
            return df
        else:
            logger.error("❌ 歴史データ取得失敗")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"❌ 歴史データ収集エラー: {e}")
        return pd.DataFrame()

async def test_feature_engineering(df: pd.DataFrame):
    """特徴量エンジニアリングテスト"""
    if df.empty:
        logger.error("❌ 特徴量テスト: データなし")
        return pd.DataFrame()
    
    logger.info("🔧 特徴量エンジニアリングテスト")
    
    try:
        feature_engine = ScalpingFeatureEngine()
        
        # 特徴量生成
        df_features = feature_engine.engineer_all_features(df.copy())
        
        # ターゲット変数作成
        df_features = feature_engine.create_target_variables(df_features)
        
        if not df_features.empty:
            feature_count = len([col for col in df_features.columns 
                               if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])
            logger.info(f"✅ 特徴量生成成功: {feature_count}個の特徴量")
            
            # 重要な特徴量のサンプル表示
            important_features = ['rsi_14', 'macd_histogram', 'volume_ratio_10', 'atr_ratio_10', 'bb_position_20']
            for feature in important_features:
                if feature in df_features.columns:
                    avg_val = df_features[feature].mean()
                    logger.info(f"   {feature}: 平均値={avg_val:.4f}")
            
            return df_features
        else:
            logger.error("❌ 特徴量生成失敗")
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"❌ 特徴量エンジニアリングエラー: {e}")
        return pd.DataFrame()

async def test_orderbook_analysis():
    """オーダーブック分析テスト"""
    logger.info("📖 オーダーブック分析テスト")
    
    try:
        analyzer = OrderbookAnalyzer()
        
        # 短期間のオーダーブックスナップショット収集
        pairs = ['BTCUSDT', 'ETHUSDT']
        snapshots = analyzer.collect_orderbook_snapshots(
            pairs=pairs, 
            duration_minutes=2,  # 2分間のテスト
            interval_seconds=10   # 10秒間隔
        )
        
        if snapshots:
            total_snapshots = sum(len(snaps) for snaps in snapshots.values())
            logger.info(f"✅ オーダーブック収集成功: {total_snapshots}スナップショット")
            
            # スプレッド分析
            analysis = analyzer.analyze_spread_patterns(snapshots)
            
            for pair, pair_analysis in analysis.items():
                if pair_analysis:
                    avg_spread = pair_analysis.get('spread_statistics', {}).get('avg_spread_pct', 0)
                    logger.info(f"   {pair}: 平均スプレッド={avg_spread:.4f}%")
            
            return True
        else:
            logger.error("❌ オーダーブック分析失敗")
            return False
            
    except Exception as e:
        logger.error(f"❌ オーダーブック分析エラー: {e}")
        return False

async def test_signal_generation():
    """シグナル生成テスト"""
    logger.info("📡 シグナル生成テスト")
    
    try:
        client = BitgetRestClient()
        analyzer = OrderbookAnalyzer()
        
        # 現在のマーケット状況取得
        current_orderbook = analyzer._get_enhanced_orderbook('BTCUSDT')
        
        if current_orderbook:
            logger.info(f"現在の市況: {current_orderbook['pair']}")
            logger.info(f"  価格: ${current_orderbook['mid_price']:.2f}")
            logger.info(f"  スプレッド: {current_orderbook['spread_pct']:.4f}%")
            logger.info(f"  深度不均衡: {current_orderbook['depth_analysis'].get('depth_imbalance_5', 0):.3f}")
            
            # ダミーの歴史パターンでシグナル生成テスト
            historical_patterns = {
                'BTCUSDT': {
                    'optimal_conditions': {
                        'optimal_spread_threshold': 0.005,
                        'optimal_liquidity_threshold': 100,
                        'optimal_hours': [8, 9, 10, 14, 15, 16]
                    },
                    'spread_statistics': {
                        'avg_spread_pct': 0.008
                    }
                }
            }
            
            signal = analyzer.generate_scalping_signals(current_orderbook, historical_patterns)
            
            logger.info(f"✅ シグナル生成: {signal['signal']} (信頼度: {signal['confidence']:.2f})")
            logger.info(f"   要因: {', '.join(signal.get('factors', []))}")
            
            return True
        else:
            logger.error("❌ 現在の市況取得失敗")
            return False
            
    except Exception as e:
        logger.error(f"❌ シグナル生成エラー: {e}")
        return False

async def main():
    """メインテスト実行"""
    logger.info("🚀 Bitget スキャルピングシステム統合テスト開始")
    
    test_results = {}
    
    # 1. REST API接続テスト
    test_results['rest_api'] = await test_rest_api_connection()
    
    # 2. 歴史データ収集テスト
    historical_data = await test_historical_data_collection()
    test_results['historical_data'] = not historical_data.empty
    
    # 3. 特徴量エンジニアリングテスト
    if not historical_data.empty:
        feature_data = await test_feature_engineering(historical_data)
        test_results['feature_engineering'] = not feature_data.empty
    else:
        test_results['feature_engineering'] = False
    
    # 4. オーダーブック分析テスト
    test_results['orderbook_analysis'] = await test_orderbook_analysis()
    
    # 5. シグナル生成テスト
    test_results['signal_generation'] = await test_signal_generation()
    
    # 結果サマリー
    print("\n" + "="*60)
    print("🎯 Bitget スキャルピングシステム テスト結果")
    print("="*60)
    
    for test_name, result in test_results.items():
        status = "✅ 成功" if result else "❌ 失敗"
        print(f"{test_name.replace('_', ' ').title():.<40} {status}")
    
    success_rate = sum(test_results.values()) / len(test_results)
    print(f"\n総合成功率: {success_rate:.1%}")
    
    if success_rate >= 0.8:
        print("\n🎉 システムは正常に動作しています！")
        print("📋 次のステップ:")
        print("   1. python scripts/scalping_data_collector.py で1ヶ月分データ収集")
        print("   2. 収集したデータでバックテスト実行")
        print("   3. ライブ取引環境での小規模テスト")
    else:
        print("\n⚠️ 一部機能に問題があります。エラーを確認してください。")
    
    print("="*60)

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