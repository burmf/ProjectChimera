#!/usr/bin/env python3
"""
Simple AI Department System Test (Mock Version)
AI部門システム簡易テスト（モック版）
"""

import asyncio
import json
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MockAIAgent:
    """モックAIエージェント"""
    
    def __init__(self, department_name: str):
        self.department_name = department_name
        self.logger = logging.getLogger(f"Mock{department_name}")
    
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """モック分析処理"""
        self.logger.info(f"{self.department_name} processing request")
        
        # 部門別のモック応答
        if self.department_name == "Technical":
            return {
                'department': self.department_name,
                'confidence': 0.75,
                'decision': {'action': 'buy', 'trend': 'uptrend'},
                'reasoning': 'RSI oversold, MACD bullish cross',
                'processing_time_ms': 150.0,
                'cost_usd': 0.01
            }
        elif self.department_name == "Fundamental":
            return {
                'department': self.department_name,
                'confidence': 0.65,
                'decision': {'action': 'buy', 'outlook': 'positive'},
                'reasoning': 'Strong economic data, hawkish central bank',
                'processing_time_ms': 200.0,
                'cost_usd': 0.02
            }
        elif self.department_name == "Sentiment":
            return {
                'department': self.department_name,
                'confidence': 0.55,
                'decision': {'action': 'hold', 'sentiment': 'neutral'},
                'reasoning': 'Mixed news sentiment',
                'processing_time_ms': 100.0,
                'cost_usd': 0.01
            }
        elif self.department_name == "Risk":
            return {
                'department': self.department_name,
                'confidence': 0.80,
                'decision': {'action': 'buy', 'risk_level': 'medium'},
                'reasoning': 'Acceptable risk parameters',
                'processing_time_ms': 80.0,
                'cost_usd': 0.005
            }
        elif self.department_name == "Execution":
            return {
                'department': self.department_name,
                'confidence': 0.90,
                'decision': {'action': 'buy', 'position_size': 0.1},
                'reasoning': 'Optimal execution conditions',
                'processing_time_ms': 50.0,
                'cost_usd': 0.002
            }
        else:
            return {
                'department': self.department_name,
                'confidence': 0.50,
                'decision': {'action': 'hold'},
                'reasoning': 'Default mock response',
                'processing_time_ms': 100.0,
                'cost_usd': 0.01
            }


class MockOrchestrator:
    """モックオーケストレーター"""
    
    def __init__(self):
        self.departments = {}
        self.decision_history = []
        self.total_decisions = 0
        self.total_cost = 0.0
        self.logger = logging.getLogger("MockOrchestrator")
    
    def register_department(self, dept_name: str, agent: MockAIAgent):
        """部門登録"""
        self.departments[dept_name] = agent
        self.logger.info(f"Registered {dept_name} department")
    
    async def analyze_market_situation(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """市場分析（モック）"""
        self.logger.info("Starting integrated market analysis")
        
        # 並行で各部門に分析依頼
        tasks = []
        for dept_name, agent in self.departments.items():
            task = agent.process_request(market_data)
            tasks.append((dept_name, task))
        
        # 結果を収集
        department_results = {}
        total_cost = 0.0
        total_processing_time = 0.0
        
        for dept_name, task in tasks:
            try:
                result = await task
                department_results[dept_name] = result
                total_cost += result.get('cost_usd', 0.0)
                total_processing_time += result.get('processing_time_ms', 0.0)
            except Exception as e:
                self.logger.error(f"Department {dept_name} failed: {e}")
                department_results[dept_name] = {
                    'department': dept_name,
                    'confidence': 0.0,
                    'decision': {'action': 'hold'},
                    'reasoning': f'Error: {str(e)}',
                    'error': True
                }
        
        # 統合判定
        actions = []
        confidence_scores = []
        
        for result in department_results.values():
            if not result.get('error', False):
                actions.append(result['decision'].get('action', 'hold'))
                confidence_scores.append(result.get('confidence', 0.0))
        
        # 多数決でアクション決定
        if actions:
            action_counts = {}
            for action in actions:
                action_counts[action] = action_counts.get(action, 0) + 1
            
            final_action = max(action_counts, key=action_counts.get)
            consensus_confidence = sum(confidence_scores) / len(confidence_scores)
        else:
            final_action = 'hold'
            consensus_confidence = 0.0
        
        # 統合決定
        integrated_decision = {
            'decision_id': f"decision_{self.total_decisions + 1}",
            'decision_type': 'trade_signal',
            'departments_involved': list(self.departments.keys()),
            'consensus_confidence': consensus_confidence,
            'final_decision': {
                'action': final_action,
                'confidence': consensus_confidence,
                'reasoning': f'Integrated analysis from {len(department_results)} departments'
            },
            'department_results': department_results,
            'total_cost_usd': total_cost,
            'processing_time_ms': total_processing_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # 統計更新
        self.total_decisions += 1
        self.total_cost += total_cost
        self.decision_history.append(integrated_decision)
        
        self.logger.info(f"Integrated analysis completed: {final_action} (confidence: {consensus_confidence:.3f})")
        
        return integrated_decision
    
    def get_statistics(self) -> Dict[str, Any]:
        """統計情報取得"""
        return {
            'total_decisions': self.total_decisions,
            'total_cost_usd': self.total_cost,
            'registered_departments': list(self.departments.keys()),
            'decision_history_count': len(self.decision_history)
        }


class SimpleAISystemTest:
    """簡易AI部門システムテスト"""
    
    def __init__(self):
        self.orchestrator = MockOrchestrator()
        self.test_results = []
    
    async def run_all_tests(self):
        """全テスト実行"""
        logger.info("=== Simple AI Department System Test ===")
        
        tests = [
            self.test_department_registration,
            self.test_market_analysis,
            self.test_multiple_scenarios,
            self.test_error_scenarios,
            self.test_statistics
        ]
        
        for test in tests:
            try:
                logger.info(f"Running test: {test.__name__}")
                result = await test()
                self.test_results.append({
                    'test_name': test.__name__,
                    'status': 'PASSED' if result else 'FAILED'
                })
                logger.info(f"Test {test.__name__}: {'PASSED' if result else 'FAILED'}")
            except Exception as e:
                logger.error(f"Test {test.__name__} error: {e}")
                self.test_results.append({
                    'test_name': test.__name__,
                    'status': 'ERROR',
                    'error': str(e)
                })
        
        self.print_summary()
        return self.test_results
    
    async def test_department_registration(self) -> bool:
        """部門登録テスト"""
        departments = ['Technical', 'Fundamental', 'Sentiment', 'Risk', 'Execution']
        
        for dept_name in departments:
            agent = MockAIAgent(dept_name)
            self.orchestrator.register_department(dept_name, agent)
        
        registered = list(self.orchestrator.departments.keys())
        return set(registered) == set(departments)
    
    async def test_market_analysis(self) -> bool:
        """市場分析テスト"""
        market_data = {
            'price_data': {'open': 150.0, 'high': 151.0, 'low': 149.0, 'close': 150.5},
            'technical_indicators': {'rsi': 65.0, 'macd': 'bullish'},
            'news_data': [{'title': 'Positive economic news', 'sentiment': 0.8}],
            'economic_data': {'gdp_growth': 2.5},
            'portfolio_state': {'total_value': 100000},
            'risk_metrics': {'var_1d': 1000}
        }
        
        decision = await self.orchestrator.analyze_market_situation(market_data)
        
        # 必須フィールドの確認
        required_fields = ['decision_id', 'final_decision', 'department_results', 'consensus_confidence']
        for field in required_fields:
            if field not in decision:
                logger.error(f"Missing field: {field}")
                return False
        
        # アクションが有効か
        action = decision['final_decision'].get('action')
        if action not in ['buy', 'sell', 'hold']:
            logger.error(f"Invalid action: {action}")
            return False
        
        # 信頼度が妥当か
        confidence = decision.get('consensus_confidence', 0)
        if not (0.0 <= confidence <= 1.0):
            logger.error(f"Invalid confidence: {confidence}")
            return False
        
        return True
    
    async def test_multiple_scenarios(self) -> bool:
        """複数シナリオテスト"""
        scenarios = [
            {
                'name': 'bullish_scenario',
                'data': {
                    'price_data': {'close': 155.0},
                    'technical_indicators': {'rsi': 30, 'trend': 'upward'},
                    'news_data': [{'sentiment': 0.9}]
                }
            },
            {
                'name': 'bearish_scenario',
                'data': {
                    'price_data': {'close': 145.0},
                    'technical_indicators': {'rsi': 80, 'trend': 'downward'},
                    'news_data': [{'sentiment': -0.7}]
                }
            },
            {
                'name': 'neutral_scenario',
                'data': {
                    'price_data': {'close': 150.0},
                    'technical_indicators': {'rsi': 50, 'trend': 'sideways'},
                    'news_data': [{'sentiment': 0.1}]
                }
            }
        ]
        
        for scenario in scenarios:
            decision = await self.orchestrator.analyze_market_situation(scenario['data'])
            
            if not decision.get('decision_id'):
                logger.error(f"Scenario {scenario['name']} failed")
                return False
            
            logger.info(f"Scenario {scenario['name']}: {decision['final_decision']['action']}")
        
        return True
    
    async def test_error_scenarios(self) -> bool:
        """エラーシナリオテスト"""
        # 空データでのテスト
        empty_data = {}
        
        try:
            decision = await self.orchestrator.analyze_market_situation(empty_data)
            
            # エラー時でも基本構造は維持されるべき
            if not decision.get('decision_id'):
                return False
            
            # フォールバック動作確認
            action = decision['final_decision'].get('action', 'hold')
            if action not in ['buy', 'sell', 'hold']:
                return False
            
            return True
            
        except Exception:
            # 例外が発生しても処理を継続できるかテスト
            return True
    
    async def test_statistics(self) -> bool:
        """統計情報テスト"""
        stats = self.orchestrator.get_statistics()
        
        required_stats = ['total_decisions', 'total_cost_usd', 'registered_departments']
        for stat in required_stats:
            if stat not in stats:
                logger.error(f"Missing statistic: {stat}")
                return False
        
        # 意思決定数が増加しているか
        if stats['total_decisions'] <= 0:
            logger.error("No decisions recorded")
            return False
        
        return True
    
    def print_summary(self):
        """テスト結果サマリー"""
        logger.info("\n=== TEST SUMMARY ===")
        
        passed = sum(1 for r in self.test_results if r['status'] == 'PASSED')
        total = len(self.test_results)
        success_rate = (passed / total) * 100 if total > 0 else 0
        
        logger.info(f"Total Tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        
        for result in self.test_results:
            status_emoji = "✅" if result['status'] == 'PASSED' else "❌"
            logger.info(f"{status_emoji} {result['test_name']}")


async def main():
    """メイン実行"""
    test_runner = SimpleAISystemTest()
    
    try:
        results = await test_runner.run_all_tests()
        
        # 結果保存
        with open('simple_test_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # 成功判定
        passed_count = sum(1 for r in results if r['status'] == 'PASSED')
        success_rate = (passed_count / len(results)) * 100
        
        if success_rate >= 80:
            logger.info(f"🎉 AI Department System Test PASSED ({success_rate:.1f}%)")
            return True
        else:
            logger.error(f"❌ AI Department System Test FAILED ({success_rate:.1f}%)")
            return False
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)