#!/usr/bin/env python3
"""
Bitget スキャルピングデータ収集実行スクリプト
1ヶ月分の高頻度データ(1分、5分、15分足)を収集し、
テクニカル分析と特徴量エンジニアリングを実行
"""

import sys
import os
import json
import pickle
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# プロジェクトパスを追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.bitget_rest_client import BitgetRestClient, ScalpingDataCollector
from core.database_adapter import db_adapter
from core.logging_config import setup_logging

# ログ設定
setup_logging()
logger = logging.getLogger(__name__)

def main():
    """メイン実行関数"""
    logger.info("🚀 Bitget スキャルピングデータ収集開始")
    
    # データ収集器を初期化
    collector = ScalpingDataCollector(database_adapter=db_adapter)
    
    try:
        # 1ヶ月分のデータを収集
        logger.info("📊 1ヶ月分の高頻度データ収集中...")
        dataset = collector.collect_scalping_dataset(days_back=30)
        
        # データ品質チェック
        total_records = 0
        data_summary = {}
        
        for pair, timeframes in dataset.items():
            pair_records = 0
            pair_summary = {}
            
            for timeframe, df in timeframes.items():
                records = len(df) if not df.empty else 0
                pair_records += records
                pair_summary[timeframe] = {
                    'records': records,
                    'date_range': {
                        'start': df['timestamp'].min().isoformat() if not df.empty else None,
                        'end': df['timestamp'].max().isoformat() if not df.empty else None
                    },
                    'avg_volume': float(df['volume'].mean()) if not df.empty else 0,
                    'price_range': {
                        'min': float(df['low'].min()) if not df.empty else 0,
                        'max': float(df['high'].max()) if not df.empty else 0
                    }
                }
            
            total_records += pair_records
            data_summary[pair] = pair_summary
            logger.info(f"✅ {pair}: {pair_records:,} レコード収集完了")
        
        logger.info(f"🎯 総収集レコード数: {total_records:,}")
        
        # データをファイルに保存
        output_dir = "/home/ec2-user/ProjectChimera/data/scalping"
        os.makedirs(output_dir, exist_ok=True)
        
        # データセット保存
        dataset_file = os.path.join(output_dir, f"scalping_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        with open(dataset_file, 'wb') as f:
            pickle.dump(dataset, f)
        logger.info(f"💾 データセット保存: {dataset_file}")
        
        # サマリー保存
        summary_file = os.path.join(output_dir, f"data_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(summary_file, 'w') as f:
            json.dump(data_summary, f, indent=2, default=str)
        logger.info(f"📋 サマリー保存: {summary_file}")
        
        # スキャルピングパターン分析
        logger.info("🔍 スキャルピングパターン分析中...")
        pattern_analysis = collector.analyze_scalping_patterns(dataset)
        
        # 分析結果保存
        analysis_file = os.path.join(output_dir, f"pattern_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(analysis_file, 'w') as f:
            json.dump(pattern_analysis, f, indent=2, default=str)
        logger.info(f"📈 パターン分析保存: {analysis_file}")
        
        # 結果サマリー表示
        print_collection_summary(data_summary, pattern_analysis)
        
        logger.info("🎉 スキャルピングデータ収集完了!")
        
    except Exception as e:
        logger.error(f"❌ データ収集エラー: {e}")
        raise

def print_collection_summary(data_summary: dict, pattern_analysis: dict):
    """収集結果サマリーを表示"""
    print("\n" + "="*80)
    print("📊 BITGET スキャルピングデータ収集結果")
    print("="*80)
    
    for pair, timeframes in data_summary.items():
        print(f"\n🔸 {pair}")
        total_pair_records = sum(tf['records'] for tf in timeframes.values())
        print(f"   総レコード数: {total_pair_records:,}")
        
        for timeframe, info in timeframes.items():
            print(f"   {timeframe:>3}: {info['records']:>6,} レコード")
            if info['date_range']['start']:
                start_date = datetime.fromisoformat(info['date_range']['start']).strftime('%m/%d %H:%M')
                end_date = datetime.fromisoformat(info['date_range']['end']).strftime('%m/%d %H:%M')
                print(f"        期間: {start_date} - {end_date}")
                print(f"        価格: ${info['price_range']['min']:.4f} - ${info['price_range']['max']:.4f}")
    
    print(f"\n📈 スキャルピング分析結果:")
    
    for pair, timeframes in pattern_analysis.items():
        print(f"\n🎯 {pair} 最適化情報:")
        
        for timeframe, analysis in timeframes.items():
            if not analysis:
                continue
                
            print(f"   {timeframe}:")
            print(f"     平均ボラティリティ: {analysis.get('avg_volatility', 0):.4f}")
            print(f"     高出来高閾値: {analysis.get('high_volume_threshold', 0):.2f}")
            print(f"     大きな値動き: {analysis.get('significant_moves_count', 0):,} 回")
            
            best_hours = analysis.get('best_scalping_hours', [])
            if best_hours:
                print(f"     最適時間帯: {', '.join(f'{h:02d}:00' for h in best_hours)}")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()