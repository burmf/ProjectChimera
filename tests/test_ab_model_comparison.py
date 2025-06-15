#!/usr/bin/env python3
"""
AI Model A/B Test Runner
AIモデルA/Bテスト実行・データ収集システム
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any


from core.ab_test_manager import (
    ab_test_manager, TestConfiguration, ModelType, TestStatus
)
from core.model_comparator import (
    model_comparator, ModelComparisonRequest
)
from core.database_adapter import db_adapter

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ABTestRunner:
    """A/Bテスト実行システム"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_data_samples = self._generate_test_samples()
    
    def _generate_test_samples(self) -> List[Dict[str, Any]]:
        """テスト用サンプルデータ生成"""
        return [
            # 金利関連ニュース
            {
                'news': 'Federal Reserve raises interest rates by 0.25% to combat inflation',
                'pair': 'USD/JPY',
                'expected_direction': 'long',
                'category': 'monetary_policy'
            },
            {
                'news': 'Bank of Japan maintains ultra-loose monetary policy despite inflation',
                'pair': 'USD/JPY', 
                'expected_direction': 'long',
                'category': 'monetary_policy'
            },
            
            # 雇用統計
            {
                'news': 'US non-farm payrolls surge by 350,000, beating expectations',
                'pair': 'USD/JPY',
                'expected_direction': 'long',
                'category': 'employment'
            },
            {
                'news': 'Japan unemployment rate rises to 3.2%, higher than forecast',
                'pair': 'USD/JPY',
                'expected_direction': 'long', 
                'category': 'employment'
            },
            
            # インフレ関連
            {
                'news': 'US inflation accelerates to 4.2% year-on-year in latest reading',
                'pair': 'USD/JPY',
                'expected_direction': 'long',
                'category': 'inflation'
            },
            {
                'news': 'Japanese core CPI remains below 2% target for 12th month',
                'pair': 'USD/JPY',
                'expected_direction': 'long',
                'category': 'inflation'
            },
            
            # GDP・成長率
            {
                'news': 'US GDP growth slows to 1.8% in Q3, below expectations',
                'pair': 'USD/JPY',
                'expected_direction': 'short',
                'category': 'growth'
            },
            {
                'news': 'Japan economy contracts 0.3% as consumer spending weakens',
                'pair': 'USD/JPY',
                'expected_direction': 'short',
                'category': 'growth'
            },
            
            # 地政学リスク
            {
                'news': 'Trade tensions escalate as new tariffs announced',
                'pair': 'USD/JPY',
                'expected_direction': 'short',
                'category': 'geopolitical'
            },
            {
                'news': 'Global supply chain disruptions continue to impact manufacturing',
                'pair': 'USD/JPY',
                'expected_direction': 'short',
                'category': 'geopolitical'
            },
            
            # 中立的ニュース
            {
                'news': 'Central bank officials express mixed views on future policy direction',
                'pair': 'USD/JPY',
                'expected_direction': 'hold',
                'category': 'neutral'
            },
            {
                'news': 'Economic indicators show mixed signals for future growth prospects',
                'pair': 'USD/JPY',
                'expected_direction': 'hold',
                'category': 'neutral'
            },
            
            # 市場データ分析用
            {
                'price_data': {
                    'open': 150.0,
                    'high': 151.5,
                    'low': 149.8,
                    'close': 151.2,
                    'volume': 100000
                },
                'pair': 'USD/JPY',
                'expected_direction': 'long',
                'category': 'technical'
            },
            {
                'price_data': {
                    'open': 150.0,
                    'high': 150.2,
                    'low': 148.5,
                    'close': 148.8,
                    'volume': 80000
                },
                'pair': 'USD/JPY',
                'expected_direction': 'short',
                'category': 'technical'
            },
            
            # EUR/USD用サンプル
            {
                'news': 'European Central Bank hints at potential rate hikes next quarter',
                'pair': 'EUR/USD',
                'expected_direction': 'long',
                'category': 'monetary_policy'
            },
            {
                'news': 'Eurozone manufacturing PMI drops to 18-month low',
                'pair': 'EUR/USD',
                'expected_direction': 'short',
                'category': 'economic_data'
            }
        ]
    
    async def create_comprehensive_test(self) -> str:
        """包括的なA/Bテストを作成"""
        try:
            test_id = f"comprehensive_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            config = TestConfiguration(
                test_id=test_id,
                name="Comprehensive AI Model Performance Test",
                description="GPT-4, GPT-4-Turbo, O3-mini comparison for forex trading signals",
                models_to_test=[ModelType.GPT4, ModelType.GPT4_TURBO, ModelType.O3_MINI],
                traffic_split={
                    "gpt-4": 0.34,
                    "gpt-4-turbo": 0.33,
                    "o3-mini": 0.33
                },
                duration_days=7,
                min_samples=50,
                confidence_level=0.95,
                success_metrics=['accuracy', 'confidence', 'cost_efficiency', 'response_time']
            )
            
            success = await ab_test_manager.create_test(config)
            if success:
                self.logger.info(f"Created comprehensive test: {test_id}")
                return test_id
            else:
                self.logger.error("Failed to create test")
                return ""
                
        except Exception as e:
            self.logger.error(f"Failed to create comprehensive test: {e}")
            return ""
    
    async def run_model_comparison_batch(self, test_id: str, sample_count: int = 20) -> Dict[str, Any]:
        """モデル比較バッチ実行"""
        try:
            # テストサンプルを選択
            selected_samples = self.test_data_samples[:sample_count]
            
            models_to_test = [ModelType.GPT4_TURBO, ModelType.O3_MINI]  # 実際に利用可能なモデル
            
            self.logger.info(f"Starting batch comparison with {len(selected_samples)} samples")
            
            # バッチ比較実行
            results = await model_comparator.run_batch_comparison(
                test_cases=selected_samples,
                models=models_to_test,
                test_id=test_id
            )
            
            # 結果分析
            analysis = model_comparator.analyze_batch_results(results)
            
            self.logger.info(f"Batch comparison completed: {len(results)} comparisons")
            self.logger.info(f"Total cost: ${analysis.get('total_cost', 0):.4f}")
            
            return {
                'test_id': test_id,
                'results_count': len(results),
                'analysis': analysis,
                'individual_results': [r.to_dict() for r in results[:5]]  # 最初の5件のみ
            }
            
        except Exception as e:
            self.logger.error(f"Batch comparison failed: {e}")
            return {'error': str(e)}
    
    async def simulate_actual_outcomes(self, test_id: str) -> int:
        """実際の結果をシミュレート（本来は実際の市場データ）"""
        try:
            # データベースから未完了の結果を取得
            query = """
            SELECT result_id, ai_response, request_data 
            FROM ab_test_results 
            WHERE test_id = ? AND actual_outcome IS NULL
            """
            
            results = await db_adapter.fetch_all_async(query, (test_id,))
            updated_count = 0
            
            for result in results:
                result_id = result[0]
                ai_response = json.loads(result[1]) if result[1] else {}
                request_data = json.loads(result[2]) if result[2] else {}
                
                # 実際の結果をシミュレート
                actual_outcome = self._simulate_market_outcome(ai_response, request_data)
                
                # データベース更新
                success = await ab_test_manager.update_actual_outcome(result_id, actual_outcome)
                if success:
                    updated_count += 1
            
            self.logger.info(f"Updated {updated_count} results with simulated outcomes")
            return updated_count
            
        except Exception as e:
            self.logger.error(f"Failed to simulate outcomes: {e}")
            return 0
    
    def _simulate_market_outcome(self, ai_response: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
        """市場結果のシミュレーション"""
        # 期待される方向性
        expected_direction = request_data.get('expected_direction', 'hold')
        
        # AI予測
        ai_direction = ai_response.get('direction', 'hold')
        ai_trade_warranted = ai_response.get('trade_warranted', False)
        
        # シミュレートされた市場の動き
        if expected_direction == 'long':
            simulated_return = 0.015  # 1.5%の上昇
        elif expected_direction == 'short':
            simulated_return = -0.012  # 1.2%の下落
        else:  # hold
            simulated_return = 0.003  # 0.3%の小さな変動
        
        # P&L計算
        if ai_trade_warranted:
            if ai_direction == 'long':
                profit_loss = simulated_return
            elif ai_direction == 'short':
                profit_loss = -simulated_return
            else:
                profit_loss = 0.0
        else:
            profit_loss = 0.0  # トレードしない場合
        
        # 正確性判定
        direction_correct = (
            (expected_direction == 'long' and ai_direction == 'long') or
            (expected_direction == 'short' and ai_direction == 'short') or
            (expected_direction == 'hold' and not ai_trade_warranted)
        )
        
        return {
            'simulated_return': simulated_return,
            'profit_loss': profit_loss,
            'direction_correct': direction_correct,
            'expected_direction': expected_direction,
            'ai_direction': ai_direction,
            'simulation_timestamp': datetime.now().isoformat()
        }
    
    async def generate_test_report(self, test_id: str) -> Dict[str, Any]:
        """テスト結果レポート生成"""
        try:
            # テストサマリー生成
            summary = await ab_test_manager.generate_test_summary(test_id)
            
            if not summary:
                return {'error': 'Failed to generate summary'}
            
            # 追加分析
            query = """
            SELECT model_used, 
                   COUNT(*) as total_requests,
                   AVG(confidence_score) as avg_confidence,
                   AVG(cost_usd) as avg_cost,
                   AVG(processing_time_ms) as avg_time,
                   COUNT(CASE WHEN success = 1 THEN 1 END) as success_count
            FROM ab_test_results 
            WHERE test_id = ?
            GROUP BY model_used
            """
            
            model_stats = await db_adapter.fetch_all_async(query, (test_id,))
            
            # カテゴリ別分析
            category_query = """
            SELECT json_extract(request_data, '$.category') as category,
                   model_used,
                   AVG(confidence_score) as avg_confidence,
                   COUNT(CASE WHEN success = 1 THEN 1 END) as success_count,
                   COUNT(*) as total_count
            FROM ab_test_results 
            WHERE test_id = ? AND json_extract(request_data, '$.category') IS NOT NULL
            GROUP BY category, model_used
            """
            
            category_stats = await db_adapter.fetch_all_async(category_query, (test_id,))
            
            report = {
                'test_id': test_id,
                'summary': summary.to_dict(),
                'model_statistics': [
                    {
                        'model': stat[0],
                        'total_requests': stat[1],
                        'avg_confidence': stat[2],
                        'avg_cost': stat[3],
                        'avg_time_ms': stat[4],
                        'success_count': stat[5],
                        'success_rate': stat[5] / stat[1] if stat[1] > 0 else 0
                    }
                    for stat in model_stats
                ],
                'category_analysis': [
                    {
                        'category': stat[0],
                        'model': stat[1],
                        'avg_confidence': stat[2],
                        'success_count': stat[3],
                        'total_count': stat[4],
                        'success_rate': stat[3] / stat[4] if stat[4] > 0 else 0
                    }
                    for stat in category_stats
                ],
                'generated_at': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate test report: {e}")
            return {'error': str(e)}
    
    async def run_complete_test_cycle(self, sample_count: int = 15) -> Dict[str, Any]:
        """完全なテストサイクルを実行"""
        try:
            self.logger.info("=== Starting Complete A/B Test Cycle ===")
            
            # 1. テスト作成
            test_id = await self.create_comprehensive_test()
            if not test_id:
                return {'error': 'Failed to create test'}
            
            # 2. モデル比較実行
            comparison_results = await self.run_model_comparison_batch(test_id, sample_count)
            
            # 3. 実際の結果をシミュレート
            await asyncio.sleep(2)  # 少し待機
            updated_outcomes = await self.simulate_actual_outcomes(test_id)
            
            # 4. レポート生成
            report = await self.generate_test_report(test_id)
            
            self.logger.info("=== A/B Test Cycle Complete ===")
            
            return {
                'test_id': test_id,
                'comparison_results': comparison_results,
                'updated_outcomes': updated_outcomes,
                'final_report': report,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Complete test cycle failed: {e}")
            return {'error': str(e), 'success': False}


async def initialize_database():
    """データベース初期化"""
    try:
        # スキーマ作成
        with open('sql/ab_test_schema.sql', 'r') as f:
            schema_sql = f.read()
        
        # セミコロンで分割して実行
        statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
        
        for statement in statements:
            if statement and not statement.startswith('--'):
                try:
                    await db_adapter.execute_async(statement)
                except Exception as e:
                    if "already exists" not in str(e):
                        logger.warning(f"Failed to execute SQL statement: {e}")
        
        logger.info("Database initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


async def main():
    """メイン実行関数"""
    logger.info("AI Model A/B Test Runner Starting...")
    
    try:
        # データベース初期化
        db_success = await initialize_database()
        if not db_success:
            logger.error("Database initialization failed")
            return False
        
        # テストランナー作成
        test_runner = ABTestRunner()
        
        # 完全なテストサイクル実行
        results = await test_runner.run_complete_test_cycle(sample_count=12)
        
        if results.get('success', False):
            logger.info("🎉 A/B Test completed successfully!")
            
            # 結果保存
            output_file = f"ab_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {output_file}")
            
            # サマリー表示
            if 'final_report' in results and 'model_statistics' in results['final_report']:
                logger.info("\n=== MODEL PERFORMANCE SUMMARY ===")
                for stat in results['final_report']['model_statistics']:
                    logger.info(
                        f"{stat['model']}: "
                        f"Success Rate: {stat['success_rate']:.1%}, "
                        f"Avg Cost: ${stat['avg_cost']:.4f}, "
                        f"Avg Confidence: {stat['avg_confidence']:.3f}"
                    )
            
            return True
        else:
            logger.error("❌ A/B Test failed")
            logger.error(f"Error: {results.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)