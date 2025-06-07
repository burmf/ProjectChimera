#!/usr/bin/env python3
"""
Learning Cycle Runner
バックテスト学習サイクル実行スクリプト

Usage:
    python scripts/learning_cycle_runner.py [options]
    
Options:
    --data-file: バックテストデータファイルパス
    --target-profit: 目標利益率 (default: 0.10)
    --target-accuracy: 目標予測精度 (default: 0.65)
    --max-iterations: 最大学習回数 (default: 1000)
"""

import asyncio
import argparse
import pandas as pd
import logging
import sys
import os
from pathlib import Path

# プロジェクトルートを追加
sys.path.append(str(Path(__file__).parent.parent))

from core.backtest_learning_engine import backtest_learning_engine
from core.database_adapter import database_adapter

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/ec2-user/BOT/logs/learning_cycle.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class LearningCycleRunner:
    """学習サイクル実行器"""
    
    def __init__(self):
        self.engine = backtest_learning_engine
        
    async def run_learning_cycle(self, config: dict) -> bool:
        """学習サイクル実行"""
        
        try:
            logger.info("=== Starting Backtest Learning Cycle ===")
            
            # 1. バックテストデータ読み込み
            backtest_data = await self._load_backtest_data(config['data_file'])
            if backtest_data is None or len(backtest_data) < 100:
                logger.error("Insufficient backtest data for learning")
                return False
            
            logger.info(f"Loaded {len(backtest_data)} data points for learning")
            
            # 2. 学習設定の更新
            self._update_learning_config(config)
            
            # 3. 学習サイクル開始
            logger.info("Starting learning cycle...")
            learning_success = await self.engine.start_learning_cycle(backtest_data)
            
            # 4. 結果サマリー
            await self._print_learning_summary(learning_success)
            
            return learning_success
            
        except Exception as e:
            logger.error(f"Learning cycle failed: {e}")
            return False
    
    async def _load_backtest_data(self, data_file: str) -> pd.DataFrame:
        """バックテストデータ読み込み"""
        
        try:
            if data_file and os.path.exists(data_file):
                # ファイルから読み込み
                logger.info(f"Loading data from file: {data_file}")
                if data_file.endswith('.csv'):
                    data = pd.read_csv(data_file)
                elif data_file.endswith('.json'):
                    data = pd.read_json(data_file)
                else:
                    logger.error(f"Unsupported file format: {data_file}")
                    return None
            else:
                # データベースから読み込み
                logger.info("Loading data from database...")
                data = await self._load_from_database()
            
            # データ検証
            required_columns = ['close', 'high', 'low', 'open']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return None
            
            # 型変換
            for col in ['close', 'high', 'low', 'open']:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # 欠損値除去
            data = data.dropna(subset=required_columns)
            
            logger.info(f"Successfully loaded and validated {len(data)} data points")
            return data
            
        except Exception as e:
            logger.error(f"Data loading failed: {e}")
            return None
    
    async def _load_from_database(self) -> pd.DataFrame:
        """データベースからデータ読み込み"""
        
        try:
            # 過去3ヶ月のデータを取得
            query = """
            SELECT timestamp, open, high, low, close, volume
            FROM price_data 
            WHERE timestamp >= datetime('now', '-3 months')
            ORDER BY timestamp
            """
            
            conn = database_adapter.get_connection()
            data = pd.read_sql_query(query, conn)
            conn.close()
            
            # タイムスタンプ変換
            if 'timestamp' in data.columns:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            return data
            
        except Exception as e:
            logger.error(f"Database loading failed: {e}")
            return pd.DataFrame()  # 空のDataFrame返却
    
    def _update_learning_config(self, config: dict):
        """学習設定更新"""
        
        # エンジンの設定を更新
        if 'target_profit' in config:
            self.engine.learning_config['target_profit'] = config['target_profit']
        
        if 'target_accuracy' in config:
            self.engine.learning_config['target_accuracy'] = config['target_accuracy']
        
        if 'max_iterations' in config:
            self.engine.learning_config['max_learning_iterations'] = config['max_iterations']
        
        logger.info(f"Updated learning config: {self.engine.learning_config}")
    
    async def _print_learning_summary(self, success: bool):
        """学習結果サマリー出力"""
        
        summary = self.engine.get_learning_summary()
        
        print("\n" + "="*60)
        print("BACKTEST LEARNING CYCLE RESULTS")
        print("="*60)
        
        metrics = summary['learning_metrics']
        print(f"Total Inferences: {metrics['total_inferences']}")
        print(f"Prediction Accuracy: {metrics['accuracy_rate']:.2%}")
        print(f"Cumulative Profit: {metrics['cumulative_profit']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"Learning Confidence: {metrics['learning_confidence']:.2%}")
        
        print(f"\nReady for Production: {'YES' if summary['ready_for_production'] else 'NO'}")
        print(f"Recommendation: {summary['recommendation']}")
        
        recent_perf = summary['recent_performance']
        print(f"\nRecent Win Rate: {recent_perf['win_rate']:.2%}")
        print(f"Successful Patterns: {recent_perf['successful_patterns']}")
        
        # 目標基準との比較
        targets = summary['target_criteria']
        print(f"\nTarget vs Actual:")
        print(f"  Accuracy: {targets['target_accuracy']:.1%} vs {metrics['accuracy_rate']:.1%}")
        print(f"  Profit: {targets['target_profit']:.1%} vs {metrics['cumulative_profit']:.1%}")
        print(f"  Confidence: {targets['confidence_threshold']:.1%} vs {metrics['learning_confidence']:.1%}")
        
        if success:
            print(f"\n🎉 LEARNING COMPLETED SUCCESSFULLY!")
            print(f"   System is ready for live trading.")
        else:
            print(f"\n⚠️  LEARNING INCOMPLETE")
            print(f"   Continue learning or adjust parameters.")
        
        print("="*60)

def create_sample_data(num_points: int = 500) -> pd.DataFrame:
    """サンプルデータ生成（テスト用）"""
    
    import numpy as np
    from datetime import datetime, timedelta
    
    # ランダムシード設定
    np.random.seed(42)
    
    # 基本価格系列生成
    base_price = 150.0
    returns = np.random.normal(0.0001, 0.01, num_points)  # 平均0.01%、標準偏差1%のリターン
    
    prices = [base_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    
    # OHLC生成
    data = []
    for i, close in enumerate(prices[1:]):
        open_price = prices[i]
        high_noise = np.random.uniform(0, 0.005)  # 0-0.5%のノイズ
        low_noise = np.random.uniform(0, 0.005)
        
        high = close * (1 + high_noise)
        low = close * (1 - low_noise)
        
        # 時系列調整
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        timestamp = datetime.now() - timedelta(hours=num_points-i)
        
        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.uniform(1000, 5000)
        })
    
    return pd.DataFrame(data)

async def main():
    """メイン関数"""
    
    parser = argparse.ArgumentParser(description='Run backtest learning cycle')
    parser.add_argument('--data-file', type=str, help='Path to backtest data file')
    parser.add_argument('--target-profit', type=float, default=0.10, help='Target profit rate')
    parser.add_argument('--target-accuracy', type=float, default=0.65, help='Target prediction accuracy')
    parser.add_argument('--max-iterations', type=int, default=1000, help='Maximum learning iterations')
    parser.add_argument('--generate-sample', action='store_true', help='Generate sample data for testing')
    
    args = parser.parse_args()
    
    # サンプルデータ生成（テスト用）
    if args.generate_sample:
        logger.info("Generating sample data for testing...")
        sample_data = create_sample_data(500)
        sample_file = '/home/ec2-user/BOT/data/sample_backtest_data.csv'
        sample_data.to_csv(sample_file, index=False)
        logger.info(f"Sample data saved to: {sample_file}")
        args.data_file = sample_file
    
    # 設定準備
    config = {
        'data_file': args.data_file,
        'target_profit': args.target_profit,
        'target_accuracy': args.target_accuracy,
        'max_iterations': args.max_iterations
    }
    
    # 学習サイクル実行
    runner = LearningCycleRunner()
    success = await runner.run_learning_cycle(config)
    
    # 終了コード
    exit_code = 0 if success else 1
    logger.info(f"Learning cycle completed with exit code: {exit_code}")
    
    return exit_code

if __name__ == "__main__":
    # 非同期実行
    exit_code = asyncio.run(main())
    sys.exit(exit_code)