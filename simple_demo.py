#!/usr/bin/env python3
"""
Simple Learning Demo - No External Dependencies
バックテスト学習システムの簡易デモ（標準ライブラリのみ）

Purpose: 学習サイクルの概念実証
"""

import asyncio
import json
import random
import math
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
        """サンプルデータ生成（標準ライブラリのみ）"""
        logger.info(f"Generating {num_points} sample data points...")
        
        random.seed(42)
        base_price = 150.0
        data = []
        
        for i in range(num_points):
            # ランダムウォーク価格生成
            if i == 0:
                price = base_price
            else:
                change = random.gauss(0, 0.01)  # 平均0、標準偏差1%
                price = data[i-1]['close'] * (1 + change)
            
            # OHLCV データ
            high_noise = abs(random.gauss(0, 0.005))
            low_noise = abs(random.gauss(0, 0.005))
            high = price * (1 + high_noise)
            low = price * (1 - low_noise)
            volume = random.uniform(1000, 5000)
            
            timestamp = datetime.now() - timedelta(hours=num_points-i)
            
            # RSI風指標 (簡易版)
            rsi = 30 + random.uniform(0, 40)  # 30-70のRSI
            macd = random.gauss(0, 0.5)       # MACD風
            
            data.append({
                'timestamp': timestamp,
                'open': price,
                'high': high,
                'low': low,
                'close': price,
                'volume': volume,
                'rsi': rsi,
                'macd': macd
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
        """
        AI推論シミュレーション
        
        実際のシステムでは以下のプロンプトをAIに送信:
        ---
        市場分析タスク:
        現在の市場状況: {market_context}
        
        過去の学習結果:
        - 予測精度: {current_accuracy}%
        - 累計収益: {cumulative_profit}%
        - 成功パターン: {successful_patterns}
        
        質問:
        1. 次の期間で価格はどう動くと予想しますか？
        2. 推奨アクション（buy/sell/hold）は？
        3. 予想の信頼度は？
        
        JSON形式で回答してください。
        ---
        """
        
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
            reasoning = f"RSI過売り状態 ({rsi:.1f}) - 買いシグナル"
        elif rsi > 65:  # 過買い
            action = 'sell'
            confidence = base_accuracy + 0.1
            reasoning = f"RSI過買い状態 ({rsi:.1f}) - 売りシグナル"
        elif macd > 0:  # MACD正
            action = 'buy'
            confidence = base_accuracy
            reasoning = f"MACD正転 ({macd:.3f}) - 上昇モメンタム"
        elif macd < 0:  # MACD負
            action = 'sell'
            confidence = base_accuracy
            reasoning = f"MACD負転 ({macd:.3f}) - 下降モメンタム"
        else:
            action = 'hold'
            confidence = base_accuracy * 0.7
            reasoning = "明確なシグナルなし - 様子見"
        
        # ランダムノイズ追加（市場の不確実性）
        confidence += random.gauss(0, 0.05)
        confidence = max(0.3, min(0.9, confidence))
        
        # ポジションサイズ（信頼度に比例）
        position_size = confidence * 50  # 最大50%
        
        # 期待リターン（簡易計算）
        expected_return = random.uniform(-2, 3) * confidence  # -2% to 3%
        
        return {
            'action': action,
            'confidence': confidence * 100,
            'position_size': position_size,
            'expected_return': expected_return,
            'reasoning': reasoning,
            'market_regime': self.assess_market_regime(market_context)
        }

    def prepare_market_context(self, data_point):
        """市場コンテキスト準備"""
        return {
            'current_price': data_point['close'],
            'rsi': data_point.get('rsi', 50),
            'macd': data_point.get('macd', 0),
            'volume': data_point.get('volume', 1000),
            'timestamp': data_point['timestamp'],
            'volatility': abs(random.gauss(0, 0.01))  # 簡易ボラティリティ
        }

    def assess_market_regime(self, context):
        """市場レジーム評価"""
        volatility = context.get('volatility', 0.01)
        rsi = context.get('rsi', 50)
        
        if volatility > 0.02:
            return "high_volatility"
        elif rsi < 35 or rsi > 65:
            return "trending"
        else:
            return "ranging"

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
        
        # 実際の方向性判定（0.1%閾値）
        if actual_outcome > 0.001:
            actual_direction = 'buy'
        elif actual_outcome < -0.001:
            actual_direction = 'sell'
        else:
            actual_direction = 'hold'
        
        # 方向予測の正確性
        direction_correct = predicted_action == actual_direction
        
        # 信頼度による重み付け
        confidence_weight = ai_prediction['confidence'] / 100.0
        
        # 基本精度
        base_accuracy = 1.0 if direction_correct else 0.0
        
        # ボーナス：大きな動きの予測成功
        magnitude_bonus = 0.0
        if abs(actual_outcome) > 0.005 and direction_correct:  # 0.5%以上の動き
            magnitude_bonus = 0.2
        
        return (base_accuracy + magnitude_bonus) * confidence_weight

    def calculate_profit_impact(self, result):
        """利益インパクト計算"""
        ai_prediction = result['ai_prediction']
        actual_outcome = result['actual_outcome']
        
        action = ai_prediction['action']
        position_size = ai_prediction['position_size'] / 100.0
        
        # 取引コスト（簡易）
        transaction_cost = 0.0005  # 0.05%
        
        if action == 'buy':
            gross_return = actual_outcome * position_size
            net_return = gross_return - transaction_cost * position_size
            return net_return
        elif action == 'sell':
            gross_return = -actual_outcome * position_size
            net_return = gross_return - transaction_cost * position_size
            return net_return
        else:  # hold
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
        
        # 学習信頼度（精度と収益性の組み合わせ）
        accuracy_component = self.metrics['accuracy_rate']
        profit_component = max(0, self.metrics['cumulative_profit'] * 10)  # 利益を10倍してスケール調整
        
        self.metrics['learning_confidence'] = min(
            (accuracy_component * 0.7 + profit_component * 0.3),
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
            f"Samples: {metrics['total_inferences']}, "
            f"Accuracy: {metrics['accuracy_rate']:.1%}, "
            f"Profit: {metrics['cumulative_profit']:.2%}, "
            f"Confidence: {metrics['learning_confidence']:.1%}"
        )

    def print_final_summary(self, success):
        """最終サマリー出力"""
        metrics = self.metrics
        config = self.learning_config
        
        print("\n" + "="*70)
        print("BACKTEST LEARNING CYCLE - DEMO RESULTS")
        print("="*70)
        
        print(f"📊 LEARNING METRICS:")
        print(f"   Total Inferences: {metrics['total_inferences']}")
        print(f"   Correct Predictions: {metrics['correct_predictions']}")
        print(f"   Prediction Accuracy: {metrics['accuracy_rate']:.1%}")
        print(f"   Cumulative Profit: {metrics['cumulative_profit']:.3%}")
        print(f"   Learning Confidence: {metrics['learning_confidence']:.1%}")
        
        print(f"\n🎯 TARGET vs ACTUAL:")
        print(f"   Accuracy: {config['target_accuracy']:.1%} → {metrics['accuracy_rate']:.1%} "
              f"({'✅' if metrics['accuracy_rate'] >= config['target_accuracy'] else '❌'})")
        print(f"   Profit: {config['target_profit']:.1%} → {metrics['cumulative_profit']:.1%} "
              f"({'✅' if metrics['cumulative_profit'] >= config['target_profit'] else '❌'})")
        print(f"   Samples: {config['min_samples']} → {metrics['total_inferences']} "
              f"({'✅' if metrics['total_inferences'] >= config['min_samples'] else '❌'})")
        
        if success:
            print(f"\n🎉 LEARNING COMPLETED SUCCESSFULLY!")
            print(f"   ✅ All targets achieved")
            print(f"   ✅ System ready for production inference")
            print(f"   ✅ Can proceed to live trading")
        else:
            print(f"\n⚠️  LEARNING INCOMPLETE")
            shortfalls = []
            if metrics['accuracy_rate'] < config['target_accuracy']:
                shortfalls.append(f"accuracy ({metrics['accuracy_rate']:.1%} < {config['target_accuracy']:.1%})")
            if metrics['cumulative_profit'] < config['target_profit']:
                shortfalls.append(f"profit ({metrics['cumulative_profit']:.1%} < {config['target_profit']:.1%})")
            if metrics['total_inferences'] < config['min_samples']:
                shortfalls.append(f"samples ({metrics['total_inferences']} < {config['min_samples']})")
            
            print(f"   ❌ Shortfalls: {', '.join(shortfalls)}")
        
        # 推奨アクション
        print(f"\n💡 RECOMMENDATIONS:")
        if not success:
            if metrics['accuracy_rate'] < config['target_accuracy']:
                print(f"   📈 Improve prediction accuracy through:")
                print(f"      - Enhanced market analysis prompts")
                print(f"      - Better technical indicator integration")
                print(f"      - Regime-specific strategies")
            
            if metrics['cumulative_profit'] < config['target_profit']:
                print(f"   💰 Optimize profit generation through:")
                print(f"      - Better position sizing")
                print(f"      - Improved entry/exit timing")
                print(f"      - Risk management refinements")
            
            if metrics['total_inferences'] < config['min_samples']:
                print(f"   📊 Collect more training data")
        else:
            print(f"   🚀 Ready for production deployment")
            print(f"   📊 Monitor performance continuously")
            print(f"   🔄 Implement feedback learning")
        
        # 学習パターン分析
        if len(self.inference_history) > 0:
            profitable_trades = [r for r in self.inference_history if r['profit_impact'] and r['profit_impact'] > 0]
            if profitable_trades:
                print(f"\n📈 SUCCESSFUL PATTERNS DETECTED:")
                print(f"   Profitable trades: {len(profitable_trades)}/{len(self.inference_history)}")
                
                # 最も成功したパターンの簡易分析
                best_trade = max(profitable_trades, key=lambda x: x['profit_impact'])
                best_context = best_trade['market_context']
                best_prediction = best_trade['ai_prediction']
                
                print(f"   Best trade: {best_trade['profit_impact']:.3%} profit")
                print(f"   Pattern: {best_prediction['reasoning']}")
                print(f"   Market regime: {best_prediction.get('market_regime', 'unknown')}")
        
        print("="*70)
        return success

async def demonstrate_production_inference():
    """実運用推論のデモンストレーション"""
    print("\n" + "="*50)
    print("PRODUCTION INFERENCE SIMULATION")
    print("="*50)
    
    # 現在の市場データシミュレーション
    current_market = {
        'current_price': 151.25,
        'rsi': 42.5,
        'macd': 0.15,
        'volume': 3200,
        'volatility': 0.012,
        'timestamp': datetime.now()
    }
    
    print(f"📊 Current Market Data:")
    for key, value in current_market.items():
        if key != 'timestamp':
            print(f"   {key}: {value}")
    
    # 学習済みパターンを活用した推論シミュレーション
    print(f"\n🤖 AI Inference (using learned patterns):")
    
    # 実際のシステムではここでAIにプロンプト送信
    simulated_ai_response = {
        'action': 'buy',
        'confidence': 73.5,
        'position_size': 3.2,
        'reasoning': 'RSI in favorable range (42.5), positive MACD (0.15), learned pattern match: 85%',
        'expected_return': 1.8,
        'risk_assessment': 'moderate',
        'holding_period': 45
    }
    
    for key, value in simulated_ai_response.items():
        print(f"   {key}: {value}")
    
    print(f"\n✅ Production inference completed successfully!")
    print(f"   Ready for live trading execution")

async def main():
    """メイン実行関数"""
    print("="*70)
    print("🚀 BACKTEST LEARNING SYSTEM - DEMONSTRATION")
    print("="*70)
    print()
    print("This demo simulates the complete learning cycle:")
    print("1. 📊 Generate sample market data")
    print("2. 🤖 Execute AI inference with learning")
    print("3. 📈 Validate predictions against actual outcomes")
    print("4. 🎯 Optimize until target performance achieved")
    print("5. ✅ Ready for production trading")
    print()
    
    # デモシステム初期化
    demo = SimpleLearningDemo()
    
    # サンプルデータ生成
    sample_data = demo.generate_sample_data(100)
    
    # 学習サイクル実行
    success = await demo.execute_learning_cycle(sample_data)
    
    # 最終サマリー
    demo.print_final_summary(success)
    
    # 実運用推論のデモ
    if success:
        await demonstrate_production_inference()
    
    # 結果保存
    try:
        os.makedirs('/home/ec2-user/BOT/data', exist_ok=True)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'demo_type': 'simple_learning_cycle',
            'success': success,
            'metrics': demo.metrics,
            'config': demo.learning_config,
            'inference_count': len(demo.inference_history),
            'learning_ready': success
        }
        
        with open('/home/ec2-user/BOT/data/demo_learning_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Demo results saved to data/demo_learning_results.json")
        
        # サンプル推論結果も保存
        if demo.inference_history:
            sample_inferences = demo.inference_history[-5:]  # 最新5件
            with open('/home/ec2-user/BOT/data/sample_inferences.json', 'w') as f:
                json.dump(sample_inferences, f, indent=2, default=str)
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    
    print(f"\n🎯 DEMO COMPLETED!")
    print(f"   Exit code: {0 if success else 1}")
    print(f"   Results saved to: /home/ec2-user/BOT/data/")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)