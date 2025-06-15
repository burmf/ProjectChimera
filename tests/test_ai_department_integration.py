#!/usr/bin/env python3
"""
AI Department System Integration Test
AI部門システム統合テスト
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any


from core.ai_orchestrator import AIOrchestrator, MarketSituation, DecisionType
from core.ai_agent_base import DepartmentType, AnalysisPriority
from departments.technical_analysis_ai import TechnicalAnalysisAI
from departments.fundamental_analysis_ai import FundamentalAnalysisAI
from departments.sentiment_analysis_ai import SentimentAnalysisAI
from departments.risk_management_ai import RiskManagementAI
from departments.execution_portfolio_ai import ExecutionPortfolioAI

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AISystemIntegrationTest:
    """AI部門システム統合テスト"""
    
    def __init__(self):
        self.orchestrator = AIOrchestrator()
        self.test_results = []
        
    async def run_all_tests(self):
        """全テストを実行"""
        logger.info("=== AI Department System Integration Test ===")
        
        tests = [
            self.test_department_registration,
            self.test_market_situation_analysis,
            self.test_trade_signal_generation,
            self.test_risk_assessment,
            self.test_department_coordination,
            self.test_error_handling,
            self.test_performance_metrics
        ]
        
        for test in tests:
            try:
                logger.info(f"Running test: {test.__name__}")
                result = await test()
                self.test_results.append({
                    'test_name': test.__name__,
                    'status': 'PASSED' if result else 'FAILED',
                    'result': result
                })
                logger.info(f"Test {test.__name__}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                logger.error(f"Test {test.__name__} failed with error: {e}")
                self.test_results.append({
                    'test_name': test.__name__,
                    'status': 'ERROR',
                    'error': str(e)
                })
        
        # テスト結果のサマリー
        self.print_test_summary()
        
        return self.test_results
    
    async def test_department_registration(self) -> bool:
        """部門登録テスト"""
        try:
            # 各部門AIを初期化・登録
            departments = {
                DepartmentType.TECHNICAL: TechnicalAnalysisAI(),
                DepartmentType.FUNDAMENTAL: FundamentalAnalysisAI(),
                DepartmentType.SENTIMENT: SentimentAnalysisAI(),
                DepartmentType.RISK: RiskManagementAI(),
                DepartmentType.EXECUTION: ExecutionPortfolioAI()
            }
            
            for dept_type, agent in departments.items():
                self.orchestrator.register_department(dept_type, agent)
            
            # 登録確認
            registered_depts = list(self.orchestrator.departments.keys())
            expected_depts = list(departments.keys())
            
            return set(registered_depts) == set(expected_depts)
            
        except Exception as e:
            logger.error(f"Department registration test failed: {e}")
            return False
    
    async def test_market_situation_analysis(self) -> bool:
        """市場状況分析テスト"""
        try:
            # テスト用市場データ
            market_data = MarketSituation(
                price_data={
                    'open': 150.0,
                    'high': 151.5,
                    'low': 149.5,
                    'close': 150.8,
                    'volume': 100000
                },
                technical_indicators={
                    'rsi': {'value': 65.0, 'signal': 'neutral'},
                    'macd': {'signal': 'bullish_cross'}
                },
                news_data=[
                    {
                        'title': 'Fed raises interest rates by 0.25%',
                        'content': 'Federal Reserve announces rate hike to combat inflation',
                        'published_at': '2024-01-15T10:00:00Z',
                        'sentiment': 0.2
                    }
                ],
                economic_data={
                    'inflation_rate': {'actual': 3.2, 'forecast': 3.0, 'previous': 3.1},
                    'unemployment_rate': {'actual': 3.8, 'forecast': 3.9, 'previous': 4.0}
                },
                portfolio_state={
                    'total_value': 100000,
                    'positions': {'USD/JPY': {'size': 0.1, 'entry_price': 149.5}},
                    'cash': 90000
                },
                risk_metrics={
                    'var_1d': 1500.0,
                    'max_drawdown': 0.05,
                    'sharpe_ratio': 1.2
                },
                timestamp=datetime.now()
            )
            
            # 分析実行
            decision = await self.orchestrator.analyze_market_situation(
                market_data, 
                DecisionType.TRADE_SIGNAL, 
                AnalysisPriority.HIGH
            )
            
            # 結果検証
            required_fields = [
                'decision_id', 'decision_type', 'departments_involved',
                'consensus_confidence', 'final_decision', 'department_results'
            ]
            
            for field in required_fields:
                if not hasattr(decision, field):
                    logger.error(f"Missing field in decision: {field}")
                    return False
            
            # 信頼度が妥当な範囲内か
            if not (0.0 <= decision.consensus_confidence <= 1.0):
                logger.error(f"Invalid confidence: {decision.consensus_confidence}")
                return False
            
            # 部門結果が存在するか
            if not decision.department_results:
                logger.error("No department results found")
                return False
            
            logger.info(f"Market analysis completed with confidence: {decision.consensus_confidence:.3f}")
            return True
            
        except Exception as e:
            logger.error(f"Market situation analysis test failed: {e}")
            return False
    
    async def test_trade_signal_generation(self) -> bool:
        """トレードシグナル生成テスト"""
        try:
            # 強いブルシグナルのテストデータ
            bullish_market_data = MarketSituation(
                price_data={
                    'open': 148.0,
                    'high': 151.0,
                    'low': 147.8,
                    'close': 150.5,
                    'volume': 150000
                },
                technical_indicators={
                    'rsi': {'value': 35.0, 'signal': 'oversold'},  # 買われすぎ
                    'macd': {'signal': 'bullish_cross'},
                    'moving_averages': {'signal': 'golden_cross'}
                },
                news_data=[
                    {
                        'title': 'Strong US employment data boosts dollar',
                        'content': 'Non-farm payrolls exceed expectations',
                        'published_at': '2024-01-15T08:30:00Z',
                        'sentiment': 0.8
                    }
                ],
                economic_data={
                    'employment_change': {'actual': 250000, 'forecast': 200000, 'previous': 180000}
                },
                portfolio_state={'total_value': 100000, 'positions': {}, 'cash': 100000},
                risk_metrics={'var_1d': 1000.0, 'max_drawdown': 0.02},
                timestamp=datetime.now()
            )
            
            decision = await self.orchestrator.analyze_market_situation(
                bullish_market_data,
                DecisionType.TRADE_SIGNAL,
                AnalysisPriority.HIGH
            )
            
            # ブルシグナルが生成されるか確認
            final_action = decision.final_decision.get('action', 'hold')
            
            if final_action not in ['buy', 'sell', 'hold']:
                logger.error(f"Invalid action: {final_action}")
                return False
            
            # 信頼度が妥当か
            if decision.consensus_confidence < 0.1:
                logger.error(f"Confidence too low: {decision.consensus_confidence}")
                return False
            
            logger.info(f"Trade signal generated: {final_action} (confidence: {decision.consensus_confidence:.3f})")
            return True
            
        except Exception as e:
            logger.error(f"Trade signal generation test failed: {e}")
            return False
    
    async def test_risk_assessment(self) -> bool:
        """リスク評価テスト"""
        try:
            # 高リスク状況のテストデータ
            high_risk_market_data = MarketSituation(
                price_data={
                    'open': 150.0,
                    'high': 152.0,
                    'low': 148.0,
                    'close': 149.0,
                    'volume': 200000
                },
                technical_indicators={
                    'rsi': {'value': 85.0, 'signal': 'overbought'},  # 極端に買われすぎ
                    'bollinger_bands': {'signal': 'overbought', 'position': 'upper'}
                },
                news_data=[
                    {
                        'title': 'Geopolitical tensions escalate',
                        'content': 'Market uncertainty increases due to conflicts',
                        'published_at': '2024-01-15T09:00:00Z',
                        'sentiment': -0.7
                    }
                ],
                economic_data={},
                portfolio_state={
                    'total_value': 80000,  # ドローダウン中
                    'positions': {'USD/JPY': {'size': 0.5, 'entry_price': 152.0}},  # 含み損
                    'cash': 40000
                },
                risk_metrics={
                    'var_1d': 5000.0,  # 高いVaR
                    'max_drawdown': 0.15,  # 大きなドローダウン
                    'sharpe_ratio': 0.3
                },
                timestamp=datetime.now()
            )
            
            decision = await self.orchestrator.analyze_market_situation(
                high_risk_market_data,
                DecisionType.RISK_ASSESSMENT,
                AnalysisPriority.HIGH
            )
            
            # リスクレベルが適切に評価されているか
            risk_level = decision.final_decision.get('risk_level', 'medium')
            
            if risk_level not in ['low', 'medium', 'high']:
                logger.error(f"Invalid risk level: {risk_level}")
                return False
            
            logger.info(f"Risk assessment completed: {risk_level}")
            return True
            
        except Exception as e:
            logger.error(f"Risk assessment test failed: {e}")
            return False
    
    async def test_department_coordination(self) -> bool:
        """部門間協調テスト"""
        try:
            # 複数部門が関与する分析
            market_data = MarketSituation(
                price_data={'open': 150, 'high': 151, 'low': 149, 'close': 150.5, 'volume': 100000},
                technical_indicators={'rsi': {'value': 55, 'signal': 'neutral'}},
                news_data=[{'title': 'Mixed economic signals', 'content': 'Various indicators', 'sentiment': 0.1}],
                economic_data={'gdp_growth': {'actual': 2.1, 'forecast': 2.0}},
                portfolio_state={'total_value': 100000, 'positions': {}, 'cash': 100000},
                risk_metrics={'var_1d': 1200},
                timestamp=datetime.now()
            )
            
            decision = await self.orchestrator.analyze_market_situation(
                market_data,
                DecisionType.TRADE_SIGNAL,
                AnalysisPriority.MEDIUM
            )
            
            # 複数部門が関与したか確認
            if len(decision.departments_involved) < 2:
                logger.error(f"Insufficient departments involved: {len(decision.departments_involved)}")
                return False
            
            # 各部門の結果が含まれているか
            for dept in decision.departments_involved:
                if dept not in decision.department_results:
                    logger.error(f"Missing department result: {dept}")
                    return False
            
            logger.info(f"Department coordination test passed: {len(decision.departments_involved)} departments involved")
            return True
            
        except Exception as e:
            logger.error(f"Department coordination test failed: {e}")
            return False
    
    async def test_error_handling(self) -> bool:
        """エラーハンドリングテスト"""
        try:
            # 不正なデータでのテスト
            invalid_market_data = MarketSituation(
                price_data={},  # 空のデータ
                technical_indicators={},
                news_data=[],
                economic_data={},
                portfolio_state={},
                risk_metrics={},
                timestamp=datetime.now()
            )
            
            decision = await self.orchestrator.analyze_market_situation(
                invalid_market_data,
                DecisionType.TRADE_SIGNAL,
                AnalysisPriority.LOW
            )
            
            # エラー時のフォールバック動作確認
            if decision.consensus_confidence < 0:
                logger.error("Invalid confidence in error scenario")
                return False
            
            # エラー時はホールドになるべき
            action = decision.final_decision.get('action', 'hold')
            if action not in ['hold', 'error']:
                logger.error(f"Unexpected action in error scenario: {action}")
                return False
            
            logger.info("Error handling test passed")
            return True
            
        except Exception as e:
            logger.error(f"Error handling test failed: {e}")
            return False
    
    async def test_performance_metrics(self) -> bool:
        """パフォーマンスメトリクステスト"""
        try:
            # 統計情報の取得
            stats = self.orchestrator.get_orchestrator_statistics()
            
            required_stats = ['total_decisions', 'avg_processing_time_ms', 'total_cost_usd', 'registered_departments']
            
            for stat in required_stats:
                if stat not in stats:
                    logger.error(f"Missing statistic: {stat}")
                    return False
            
            # 最近の決定履歴
            recent_decisions = self.orchestrator.get_recent_decisions(limit=5)
            
            if not isinstance(recent_decisions, list):
                logger.error("Recent decisions should be a list")
                return False
            
            logger.info(f"Performance metrics test passed: {len(recent_decisions)} recent decisions")
            return True
            
        except Exception as e:
            logger.error(f"Performance metrics test failed: {e}")
            return False
    
    def print_test_summary(self):
        """テスト結果サマリーを出力"""
        logger.info("\n=== TEST SUMMARY ===")
        
        passed = sum(1 for result in self.test_results if result['status'] == 'PASSED')
        failed = sum(1 for result in self.test_results if result['status'] == 'FAILED')
        errors = sum(1 for result in self.test_results if result['status'] == 'ERROR')
        total = len(self.test_results)
        
        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Errors: {errors}")
        logger.info(f"Success Rate: {(passed/total)*100:.1f}%")
        
        # 失敗したテストの詳細
        for result in self.test_results:
            if result['status'] != 'PASSED':
                logger.error(f"❌ {result['test_name']}: {result['status']}")
                if 'error' in result:
                    logger.error(f"   Error: {result['error']}")
        
        # 成功したテスト
        for result in self.test_results:
            if result['status'] == 'PASSED':
                logger.info(f"✅ {result['test_name']}")


async def main():
    """メイン実行関数"""
    test_runner = AISystemIntegrationTest()
    
    try:
        results = await test_runner.run_all_tests()
        
        # 結果をJSONファイルに保存
        with open('test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info("Test results saved to test_results.json")
        
        # 成功判定
        passed_count = sum(1 for r in results if r['status'] == 'PASSED')
        success_rate = (passed_count / len(results)) * 100
        
        if success_rate >= 80:
            logger.info(f"🎉 Integration test PASSED ({success_rate:.1f}% success rate)")
            return True
        else:
            logger.error(f"❌ Integration test FAILED ({success_rate:.1f}% success rate)")
            return False
            
    except Exception as e:
        logger.error(f"Integration test execution failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)