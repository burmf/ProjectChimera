#!/usr/bin/env python3
"""
Demo Learning Cycle - Simplified Version
バックテスト学習システムのデモンストレーション（依存関係最小版）

Purpose: 学習サイクルの概念実証とワークフロー確認
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# 簡易ログ設定
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleLearningDemo:
    """簡易学習デモシステム"""
    
    def __init__(self):
        """初期化"""
        self.learning_config = {
            'target_accuracy': 0.60,
            'target_profit': 0.05,
            'min_samples': 20,
            'max_iterations': 20
        }
        
        self.inference_history = []
        self.metrics = {
            'total_inferences': 0,
            'correct_predictions': 0,
            'accuracy_rate': 0.0,
            'cumulative_profit': 0.0,
            'learning_confidence': 0.0
        }

    def generate_sample_data(self, num_points=100):
        """サンプルデータ生成"""
        logger.info(f"Generating {num_points} sample data points...")
        
        np.random.seed(42)
        base_price = 150.0
        data = []
        
        for i in range(num_points):
            # ランダムウォーク価格生成
            if i == 0:
                price = base_price
            else:
                change = np.random.normal(0, 0.01)  # 1% 標準偏差
                price = data[i-1]['close'] * (1 + change)
            
            # OHLCV データ
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            volume = np.random.uniform(1000, 5000)
            
            timestamp = datetime.now() - timedelta(hours=num_points-i)
            
            data.append({
                'timestamp': timestamp,
                'open': price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume,
                'rsi': 30 + np.random.uniform(0, 40),  # 30-70のRSI
                'macd': np.random.normal(0, 0.5)
            })
        
        logger.info(f"Sample data generated: {len(data)} points")
        return data

    async def execute_learning_cycle(self, data):
        """学習サイクル実行"""
        logger.info("=== Starting Learning Cycle ===")
        
        for iteration in range(self.learning_config['max_iterations']):
            logger.info(f"Learning iteration {iteration + 1}/{self.learning_config['max_iterations']}")
            
            # バッチサイズ
            batch_size = min(10, len(data) - iteration - 1)
            if batch_size <= 0:
                break
            
            start_idx = iteration
            batch_data = data[start_idx:start_idx + batch_size]
            
            # バッチ推論実行
            batch_results = await self.execute_inference_batch(batch_data, iteration)
            
            # 結果検証
            self.validate_batch_results(batch_results)
            
            # 学習進捗確認
            if self.check_learning_completion():
                logger.info(f"Learning target achieved at iteration {iteration + 1}")
                return True
            
            # 進捗ログ
            if (iteration + 1) % 5 == 0:
                self.log_progress(iteration + 1)
        
        # 最終評価
        success = self.check_learning_completion()
        logger.info(f"Learning cycle completed. Success: {success}")
        return success

    async def execute_inference_batch(self, batch_data, iteration):
        """推論バッチ実行"""
        batch_results = []
        
        for i, data_point in enumerate(batch_data[:-1]):  # 最後を除く
            try:
                # 市場コンテキスト準備
                market_context = self.prepare_market_context(data_point)
                
                # AI推論シミュレーション
                ai_prediction = await self.simulate_ai_inference(market_context, iteration)
                
                # 実際の結果計算
                next_data = batch_data[i + 1]
                actual_outcome = self.calculate_actual_outcome(data_point, next_data)
                
                # 結果記録
                result = {
                    'timestamp': data_point['timestamp'],
                    'market_context': market_context,
                    'ai_prediction': ai_prediction,
                    'actual_outcome': actual_outcome,
                    'prediction_accuracy': None,
                    'profit_impact': None
                }
                
                batch_results.append(result)
                
            except Exception as e:
                logger.error(f"Inference failed for batch item {i}: {e}")
                continue
        
        logger.info(f"Batch inference completed: {len(batch_results)} results")
        return batch_results

    async def simulate_ai_inference(self, market_context, iteration):
        """AI推論シミュレーション"""
        
        # 簡易推論ロジック（実際のAI推論の代替）
        current_price = market_context['current_price']
        rsi = market_context.get('rsi', 50)
        macd = market_context.get('macd', 0)
        
        # 学習進捗による精度向上シミュレーション
        learning_bonus = min(iteration * 0.02, 0.3)  # 最大30%の精度向上
        base_accuracy = 0.45 + learning_bonus
        
        # シンプルな予測ロジック
        if rsi < 35:  # 過売り
            action = 'buy'
            confidence = base_accuracy + 0.1
        elif rsi > 65:  # 過買い
            action = 'sell'
            confidence = base_accuracy + 0.1
        elif macd > 0:  # MACD正
            action = 'buy'
            confidence = base_accuracy
        elif macd < 0:  # MACD負
            action = 'sell'
            confidence = base_accuracy
        else:
            action = 'hold'
            confidence = base_accuracy * 0.7
        
        # ランダムノイズ追加
        confidence += np.random.normal(0, 0.05)
        confidence = max(0.3, min(0.9, confidence))
        
        # ポジションサイズ
        position_size = confidence * 50  # 最大50%
        
        return {
            'action': action,
            'confidence': confidence * 100,
            'position_size': position_size,
            'expected_return': np.random.uniform(-2, 3),  # -2% to 3%
            'reasoning': f"RSI: {rsi:.1f}, MACD: {macd:.3f}"
        }

    def prepare_market_context(self, data_point):
        """市場コンテキスト準備"""
        return {
            'current_price': data_point['close'],
            'rsi': data_point.get('rsi', 50),
            'macd': data_point.get('macd', 0),
            'volume': data_point.get('volume', 1000),
            'timestamp': data_point['timestamp']
        }

    def calculate_actual_outcome(self, current_data, future_data):
        """実際の結果計算"""
        current_price = current_data['close']
        future_price = future_data['close']
        
        return (future_price - current_price) / current_price

    def validate_batch_results(self, batch_results):
        """バッチ結果検証"""
        for result in batch_results:
            # 予測精度計算
            accuracy = self.calculate_prediction_accuracy(result)
            result['prediction_accuracy'] = accuracy
            
            # 利益インパクト計算
            profit_impact = self.calculate_profit_impact(result)
            result['profit_impact'] = profit_impact
            
            # 履歴に追加
            self.inference_history.append(result)
        
        # メトリクス更新
        self.update_metrics()

    def calculate_prediction_accuracy(self, result):
        """予測精度計算"""
        ai_prediction = result['ai_prediction']
        actual_outcome = result['actual_outcome']
        
        predicted_action = ai_prediction['action']
        actual_direction = 'buy' if actual_outcome > 0.001 else ('sell' if actual_outcome < -0.001 else 'hold')
        
        # 方向予測の正確性
        direction_correct = predicted_action == actual_direction
        
        # 信頼度による重み付け
        confidence_weight = ai_prediction['confidence'] / 100.0
        
        return (1.0 if direction_correct else 0.0) * confidence_weight

    def calculate_profit_impact(self, result):
        """利益インパクト計算"""
        ai_prediction = result['ai_prediction']
        actual_outcome = result['actual_outcome']
        
        action = ai_prediction['action']
        position_size = ai_prediction['position_size'] / 100.0
        
        if action == 'buy':
            return actual_outcome * position_size
        elif action == 'sell':
            return -actual_outcome * position_size
        else:
            return 0.0

    def update_metrics(self):
        """メトリクス更新"""
        if not self.inference_history:
            return
        
        valid_results = [r for r in self.inference_history if r['prediction_accuracy'] is not None]
        
        self.metrics['total_inferences'] = len(valid_results)
        self.metrics['correct_predictions'] = sum(1 for r in valid_results if r['prediction_accuracy'] > 0.5)
        
        if self.metrics['total_inferences'] > 0:
            self.metrics['accuracy_rate'] = self.metrics['correct_predictions'] / self.metrics['total_inferences']
        
        # 利益計算
        profit_impacts = [r['profit_impact'] for r in valid_results if r['profit_impact'] is not None]
        self.metrics['cumulative_profit'] = sum(profit_impacts) if profit_impacts else 0.0
        
        # 学習信頼度
        self.metrics['learning_confidence'] = min(
            self.metrics['accuracy_rate'] * 1.2,
            (len([p for p in profit_impacts if p > 0]) / len(profit_impacts)) if profit_impacts else 0,
            1.0
        )

    def check_learning_completion(self):
        """学習完了判定"""
        config = self.learning_config
        metrics = self.metrics
        
        criteria = [
            metrics['total_inferences'] >= config['min_samples'],
            metrics['accuracy_rate'] >= config['target_accuracy'],
            metrics['cumulative_profit'] >= config['target_profit']
        ]
        
        return all(criteria)

    def log_progress(self, iteration):
        """進捗ログ"""
        metrics = self.metrics
        logger.info(
            f"Progress [{iteration}]: "
            f"Accuracy: {metrics['accuracy_rate']:.1%}, "
            f"Profit: {metrics['cumulative_profit']:.2%}, "
            f"Confidence: {metrics['learning_confidence']:.1%}"
        )

    def print_final_summary(self, success):
        """最終サマリー出力"""
        metrics = self.metrics
        config = self.learning_config
        
        print("\n" + "="*60)
        print("BACKTEST LEARNING CYCLE DEMO RESULTS")
        print("="*60)
        
        print(f"Total Inferences: {metrics['total_inferences']}")
        print(f"Correct Predictions: {metrics['correct_predictions']}")
        print(f"Prediction Accuracy: {metrics['accuracy_rate']:.1%}")
        print(f"Cumulative Profit: {metrics['cumulative_profit']:.2%}")
        print(f"Learning Confidence: {metrics['learning_confidence']:.1%}")
        
        print(f"\nTarget vs Actual:")
        print(f"  Accuracy: {config['target_accuracy']:.1%} vs {metrics['accuracy_rate']:.1%}")
        print(f"  Profit: {config['target_profit']:.1%} vs {metrics['cumulative_profit']:.1%}")
        
        if success:
            print(f"\n🎉 LEARNING COMPLETED SUCCESSFULLY!")
            print(f"   System is ready for production inference.")
        else:
            print(f"\n⚠️  LEARNING INCOMPLETE")
            print(f"   Continue learning or adjust parameters.")
        
        print("="*60)
        
        # 推奨アクション
        if metrics['accuracy_rate'] < config['target_accuracy']:
            print("💡 Recommendation: Improve prediction accuracy")
        elif metrics['cumulative_profit'] < config['target_profit']:
            print("💡 Recommendation: Optimize profit generation")
        elif success:
            print("💡 Recommendation: Ready for live trading")
        
        return success

async def main():
    """メイン実行関数"""
    logger.info("Starting Backtest Learning Demo...")
    
    # デモシステム初期化
    demo = SimpleLearningDemo()
    
    # サンプルデータ生成
    sample_data = demo.generate_sample_data(100)
    
    # 学習サイクル実行
    success = await demo.execute_learning_cycle(sample_data)
    
    # 最終サマリー
    demo.print_final_summary(success)
    
    # 結果保存
    try:
        os.makedirs('/home/ec2-user/BOT/data', exist_ok=True)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'success': success,
            'metrics': demo.metrics,
            'config': demo.learning_config,
            'inference_count': len(demo.inference_history)
        }
        
        with open('/home/ec2-user/BOT/data/demo_learning_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Demo results saved to data/demo_learning_results.json")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)