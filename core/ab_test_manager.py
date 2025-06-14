#!/usr/bin/env python3
"""
AI Model A/B Test Manager
AIモデルA/Bテスト管理システム
"""

import asyncio
import datetime
import json
import logging
import uuid
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sys
import os

# Add core modules to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database_adapter import db_adapter
from core.redis_manager import redis_manager


class TestStatus(Enum):
    """テストステータス"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ModelType(Enum):
    """AIモデルタイプ"""
    GPT4 = "gpt-4"
    GPT4_TURBO = "gpt-4-turbo"
    O3_MINI = "o3-mini"
    O3 = "o3"
    CLAUDE_SONNET = "claude-3-5-sonnet"
    GEMINI_PRO = "gemini-pro"


@dataclass
class TestConfiguration:
    """テスト設定"""
    test_id: str
    name: str
    description: str
    models_to_test: List[ModelType]
    traffic_split: Dict[str, float]  # モデル名 -> トラフィック割合
    duration_days: int
    min_samples: int
    confidence_level: float = 0.95
    success_metrics: List[str] = None
    created_at: datetime.datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.datetime.now()
        if self.success_metrics is None:
            self.success_metrics = ['accuracy', 'confidence', 'cost_efficiency', 'response_time']


@dataclass
class TestResult:
    """個別テスト結果"""
    result_id: str
    test_id: str
    model_used: ModelType
    request_data: Dict[str, Any]
    ai_response: Dict[str, Any]
    actual_outcome: Optional[Dict[str, Any]]
    metrics: Dict[str, float]
    timestamp: datetime.datetime
    processing_time_ms: float
    cost_usd: float
    confidence_score: float
    success: Optional[bool] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['model_used'] = self.model_used.value
        return result


@dataclass
class TestSummary:
    """テスト総合結果"""
    test_id: str
    total_samples: int
    model_performance: Dict[str, Dict[str, float]]
    statistical_significance: Dict[str, bool]
    winning_model: Optional[ModelType]
    cost_analysis: Dict[str, float]
    recommendations: List[str]
    generated_at: datetime.datetime
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['generated_at'] = self.generated_at.isoformat()
        if self.winning_model:
            result['winning_model'] = self.winning_model.value
        return result


class ABTestManager:
    """A/Bテスト管理システム"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_tests: Dict[str, TestConfiguration] = {}
        self.test_results: Dict[str, List[TestResult]] = {}
        
        # 統計計算用
        self.min_effect_size = 0.05  # 5%以上の差を有意とする
        
        self.logger.info("AB Test Manager initialized")
    
    async def create_test(self, config: TestConfiguration) -> bool:
        """新しいA/Bテストを作成"""
        try:
            # 設定検証
            if not self._validate_test_config(config):
                return False
            
            # データベースに保存
            await self._save_test_config(config)
            
            # アクティブテストに追加
            self.active_tests[config.test_id] = config
            self.test_results[config.test_id] = []
            
            self.logger.info(f"Created A/B test: {config.name} ({config.test_id})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create test: {e}")
            return False
    
    def _validate_test_config(self, config: TestConfiguration) -> bool:
        """テスト設定の妥当性チェック"""
        # トラフィック分割の合計が1.0になるかチェック
        total_traffic = sum(config.traffic_split.values())
        if abs(total_traffic - 1.0) > 0.001:
            self.logger.error(f"Traffic split sum is {total_traffic}, should be 1.0")
            return False
        
        # テスト対象モデルがトラフィック分割に含まれているかチェック
        for model in config.models_to_test:
            if model.value not in config.traffic_split:
                self.logger.error(f"Model {model.value} not in traffic split")
                return False
        
        # 最小サンプル数の妥当性チェック
        if config.min_samples < 30:
            self.logger.warning("Minimum sample size less than 30, results may not be statistically significant")
        
        return True
    
    async def _save_test_config(self, config: TestConfiguration):
        """テスト設定をデータベースに保存"""
        try:
            # SQLiteに保存（PostgreSQLが使える場合はそちらを使用）
            query = """
            INSERT OR REPLACE INTO ab_test_configs 
            (test_id, name, description, models_to_test, traffic_split, 
             duration_days, min_samples, confidence_level, success_metrics, 
             status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                config.test_id,
                config.name,
                config.description,
                json.dumps([model.value for model in config.models_to_test]),
                json.dumps(config.traffic_split),
                config.duration_days,
                config.min_samples,
                config.confidence_level,
                json.dumps(config.success_metrics),
                TestStatus.ACTIVE.value,
                config.created_at.isoformat()
            )
            
            await db_adapter.execute_async(query, params)
            
        except Exception as e:
            self.logger.error(f"Failed to save test config: {e}")
            raise
    
    def select_model_for_request(self, test_id: str, request_context: Dict[str, Any] = None) -> ModelType:
        """リクエストに対してテスト用モデルを選択"""
        try:
            if test_id not in self.active_tests:
                # デフォルトモデルを返す
                return ModelType.O3_MINI
            
            config = self.active_tests[test_id]
            
            # リクエストコンテキストに基づく選択ロジック（必要に応じて）
            # 現在はランダム選択
            return self._random_model_selection(config)
            
        except Exception as e:
            self.logger.error(f"Model selection failed: {e}")
            return ModelType.O3_MINI
    
    def _random_model_selection(self, config: TestConfiguration) -> ModelType:
        """トラフィック分割に基づくランダム選択"""
        rand_value = random.random()
        cumulative = 0.0
        
        for model_name, probability in config.traffic_split.items():
            cumulative += probability
            if rand_value <= cumulative:
                return ModelType(model_name)
        
        # フォールバック
        return config.models_to_test[0]
    
    async def record_test_result(
        self, 
        test_id: str, 
        model_used: ModelType,
        request_data: Dict[str, Any],
        ai_response: Dict[str, Any],
        processing_time_ms: float,
        cost_usd: float,
        confidence_score: float
    ) -> str:
        """テスト結果を記録"""
        try:
            result_id = str(uuid.uuid4())
            
            # メトリクス計算
            metrics = self._calculate_metrics(ai_response, confidence_score, processing_time_ms, cost_usd)
            
            result = TestResult(
                result_id=result_id,
                test_id=test_id,
                model_used=model_used,
                request_data=request_data,
                ai_response=ai_response,
                actual_outcome=None,  # 後で更新
                metrics=metrics,
                timestamp=datetime.datetime.now(),
                processing_time_ms=processing_time_ms,
                cost_usd=cost_usd,
                confidence_score=confidence_score
            )
            
            # メモリに保存
            if test_id in self.test_results:
                self.test_results[test_id].append(result)
            else:
                self.test_results[test_id] = [result]
            
            # データベースに保存
            await self._save_test_result(result)
            
            # Redisに即座通知
            await self._publish_test_result(result)
            
            self.logger.debug(f"Recorded test result {result_id} for test {test_id}")
            return result_id
            
        except Exception as e:
            self.logger.error(f"Failed to record test result: {e}")
            return ""
    
    def _calculate_metrics(
        self, 
        ai_response: Dict[str, Any], 
        confidence_score: float,
        processing_time_ms: float,
        cost_usd: float
    ) -> Dict[str, float]:
        """結果からメトリクスを計算"""
        metrics = {}
        
        # 基本メトリクス
        metrics['confidence'] = confidence_score
        metrics['response_time'] = processing_time_ms
        metrics['cost'] = cost_usd
        
        # 応答品質メトリクス
        if 'trade_warranted' in ai_response:
            metrics['decisiveness'] = 1.0 if ai_response['trade_warranted'] else 0.5
        else:
            metrics['decisiveness'] = 0.0
        
        # コスト効率メトリクス
        if cost_usd > 0:
            metrics['cost_efficiency'] = confidence_score / cost_usd
        else:
            metrics['cost_efficiency'] = confidence_score
        
        # 応答時間効率
        if processing_time_ms > 0:
            metrics['time_efficiency'] = confidence_score / (processing_time_ms / 1000.0)
        else:
            metrics['time_efficiency'] = confidence_score
        
        return metrics
    
    async def _save_test_result(self, result: TestResult):
        """テスト結果をデータベースに保存"""
        try:
            query = """
            INSERT INTO ab_test_results 
            (result_id, test_id, model_used, request_data, ai_response, 
             metrics, timestamp, processing_time_ms, cost_usd, confidence_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                result.result_id,
                result.test_id,
                result.model_used.value,
                json.dumps(result.request_data),
                json.dumps(result.ai_response),
                json.dumps(result.metrics),
                result.timestamp.isoformat(),
                result.processing_time_ms,
                result.cost_usd,
                result.confidence_score
            )
            
            await db_adapter.execute_async(query, params)
            
        except Exception as e:
            self.logger.error(f"Failed to save test result: {e}")
    
    async def _publish_test_result(self, result: TestResult):
        """テスト結果をRedisに通知"""
        try:
            data = result.to_dict()
            
            # ストリームに追加
            await asyncio.to_thread(
                redis_manager.add_to_stream,
                'ab_test_results',
                data
            )
            
            # 即座通知
            await asyncio.to_thread(
                redis_manager.publish,
                'ab_test_updates',
                data
            )
            
        except Exception as e:
            self.logger.error(f"Failed to publish test result: {e}")
    
    async def update_actual_outcome(
        self, 
        result_id: str, 
        actual_outcome: Dict[str, Any]
    ) -> bool:
        """実際の結果でテスト結果を更新"""
        try:
            # メモリ内の結果を更新
            for test_id, results in self.test_results.items():
                for result in results:
                    if result.result_id == result_id:
                        result.actual_outcome = actual_outcome
                        result.success = self._evaluate_success(result.ai_response, actual_outcome)
                        break
            
            # データベース更新
            query = """
            UPDATE ab_test_results 
            SET actual_outcome = ?, success = ?
            WHERE result_id = ?
            """
            
            success = self._evaluate_success_from_data(actual_outcome)
            params = (json.dumps(actual_outcome), success, result_id)
            
            await db_adapter.execute_async(query, params)
            
            self.logger.debug(f"Updated actual outcome for result {result_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update actual outcome: {e}")
            return False
    
    def _evaluate_success(self, ai_response: Dict[str, Any], actual_outcome: Dict[str, Any]) -> bool:
        """AI予測と実際の結果を比較して成功判定"""
        try:
            # トレード推奨の正確性
            predicted_trade = ai_response.get('trade_warranted', False)
            actual_profit = actual_outcome.get('profit_loss', 0.0)
            
            if predicted_trade:
                # トレードを推奨した場合、利益が出たかチェック
                return actual_profit > 0
            else:
                # トレードを推奨しなかった場合、大きな機会損失がなかったかチェック
                return abs(actual_profit) < 0.01  # 1%未満の変動
                
        except Exception:
            return False
    
    def _evaluate_success_from_data(self, actual_outcome: Dict[str, Any]) -> bool:
        """実際の結果データから成功判定（簡易版）"""
        try:
            profit_loss = actual_outcome.get('profit_loss', 0.0)
            return profit_loss > 0
        except Exception:
            return False
    
    async def generate_test_summary(self, test_id: str) -> Optional[TestSummary]:
        """テスト結果の統計分析とサマリー生成"""
        try:
            if test_id not in self.test_results:
                return None
            
            results = self.test_results[test_id]
            if not results:
                return None
            
            # モデル別パフォーマンス計算
            model_performance = self._calculate_model_performance(results)
            
            # 統計的有意性検定
            statistical_significance = self._test_statistical_significance(results)
            
            # 勝者決定
            winning_model = self._determine_winning_model(model_performance)
            
            # コスト分析
            cost_analysis = self._analyze_costs(results)
            
            # 推奨事項生成
            recommendations = self._generate_recommendations(
                model_performance, statistical_significance, cost_analysis
            )
            
            summary = TestSummary(
                test_id=test_id,
                total_samples=len(results),
                model_performance=model_performance,
                statistical_significance=statistical_significance,
                winning_model=winning_model,
                cost_analysis=cost_analysis,
                recommendations=recommendations,
                generated_at=datetime.datetime.now()
            )
            
            # データベースに保存
            await self._save_test_summary(summary)
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate test summary: {e}")
            return None
    
    def _calculate_model_performance(self, results: List[TestResult]) -> Dict[str, Dict[str, float]]:
        """モデル別パフォーマンス計算"""
        model_stats = {}
        
        for result in results:
            model_name = result.model_used.value
            
            if model_name not in model_stats:
                model_stats[model_name] = {
                    'count': 0,
                    'total_confidence': 0.0,
                    'total_cost': 0.0,
                    'total_time': 0.0,
                    'successes': 0,
                    'cost_efficiency_sum': 0.0,
                    'time_efficiency_sum': 0.0
                }
            
            stats = model_stats[model_name]
            stats['count'] += 1
            stats['total_confidence'] += result.confidence_score
            stats['total_cost'] += result.cost_usd
            stats['total_time'] += result.processing_time_ms
            
            if result.success:
                stats['successes'] += 1
            
            stats['cost_efficiency_sum'] += result.metrics.get('cost_efficiency', 0.0)
            stats['time_efficiency_sum'] += result.metrics.get('time_efficiency', 0.0)
        
        # 平均値計算
        performance = {}
        for model_name, stats in model_stats.items():
            count = stats['count']
            if count > 0:
                performance[model_name] = {
                    'sample_count': count,
                    'avg_confidence': stats['total_confidence'] / count,
                    'avg_cost': stats['total_cost'] / count,
                    'avg_response_time': stats['total_time'] / count,
                    'success_rate': stats['successes'] / count,
                    'avg_cost_efficiency': stats['cost_efficiency_sum'] / count,
                    'avg_time_efficiency': stats['time_efficiency_sum'] / count,
                    'total_cost': stats['total_cost']
                }
        
        return performance
    
    def _test_statistical_significance(self, results: List[TestResult]) -> Dict[str, bool]:
        """統計的有意性検定（簡易版）"""
        significance = {}
        
        # モデル別にグループ化
        model_groups = {}
        for result in results:
            model_name = result.model_used.value
            if model_name not in model_groups:
                model_groups[model_name] = []
            model_groups[model_name].append(result)
        
        # ペアワイズ比較
        model_names = list(model_groups.keys())
        for i, model_a in enumerate(model_names):
            for model_b in model_names[i+1:]:
                pair_key = f"{model_a}_vs_{model_b}"
                
                # 簡易的な効果サイズ計算
                group_a = model_groups[model_a]
                group_b = model_groups[model_b]
                
                if len(group_a) >= 10 and len(group_b) >= 10:
                    avg_a = sum(r.confidence_score for r in group_a) / len(group_a)
                    avg_b = sum(r.confidence_score for r in group_b) / len(group_b)
                    
                    effect_size = abs(avg_a - avg_b)
                    significance[pair_key] = effect_size > self.min_effect_size
                else:
                    significance[pair_key] = False
        
        return significance
    
    def _determine_winning_model(self, model_performance: Dict[str, Dict[str, float]]) -> Optional[ModelType]:
        """総合スコアで勝者モデルを決定"""
        if not model_performance:
            return None
        
        best_model = None
        best_score = -1.0
        
        for model_name, perf in model_performance.items():
            # 総合スコア計算（重み付き）
            score = (
                perf['success_rate'] * 0.4 +
                perf['avg_confidence'] * 0.3 +
                perf['avg_cost_efficiency'] * 0.2 +
                perf['avg_time_efficiency'] * 0.1
            )
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return ModelType(best_model) if best_model else None
    
    def _analyze_costs(self, results: List[TestResult]) -> Dict[str, float]:
        """コスト分析"""
        total_cost = sum(r.cost_usd for r in results)
        avg_cost = total_cost / len(results) if results else 0.0
        
        # モデル別コスト
        model_costs = {}
        for result in results:
            model_name = result.model_used.value
            if model_name not in model_costs:
                model_costs[model_name] = []
            model_costs[model_name].append(result.cost_usd)
        
        cost_analysis = {
            'total_cost': total_cost,
            'avg_cost_per_request': avg_cost,
            'cost_variance': 0.0
        }
        
        # モデル別平均コスト
        for model_name, costs in model_costs.items():
            if costs:
                cost_analysis[f'{model_name}_avg_cost'] = sum(costs) / len(costs)
        
        return cost_analysis
    
    def _generate_recommendations(
        self,
        model_performance: Dict[str, Dict[str, float]],
        statistical_significance: Dict[str, bool],
        cost_analysis: Dict[str, float]
    ) -> List[str]:
        """推奨事項生成"""
        recommendations = []
        
        if not model_performance:
            recommendations.append("データが不足しています。より多くのサンプルを収集してください。")
            return recommendations
        
        # 成功率が最も高いモデル
        best_success_model = max(
            model_performance.items(),
            key=lambda x: x[1]['success_rate']
        )
        recommendations.append(
            f"成功率が最も高い: {best_success_model[0]} "
            f"({best_success_model[1]['success_rate']:.1%})"
        )
        
        # コスト効率が最も良いモデル
        best_cost_model = max(
            model_performance.items(),
            key=lambda x: x[1]['avg_cost_efficiency']
        )
        recommendations.append(
            f"コスト効率が最も良い: {best_cost_model[0]} "
            f"(効率: {best_cost_model[1]['avg_cost_efficiency']:.2f})"
        )
        
        # 統計的有意性についてのコメント
        significant_differences = sum(statistical_significance.values())
        if significant_differences > 0:
            recommendations.append(f"{significant_differences}組のモデル間で統計的有意差あり")
        else:
            recommendations.append("モデル間で統計的有意差なし - より多くのデータが必要")
        
        # コストに関する推奨
        total_cost = cost_analysis.get('total_cost', 0.0)
        if total_cost > 10.0:  # $10以上
            recommendations.append("コストが高額になっています。コスト効率の良いモデルの使用を検討してください。")
        
        return recommendations
    
    async def _save_test_summary(self, summary: TestSummary):
        """テストサマリーをデータベースに保存"""
        try:
            query = """
            INSERT OR REPLACE INTO ab_test_summaries 
            (test_id, total_samples, model_performance, statistical_significance,
             winning_model, cost_analysis, recommendations, generated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            params = (
                summary.test_id,
                summary.total_samples,
                json.dumps(summary.model_performance),
                json.dumps(summary.statistical_significance),
                summary.winning_model.value if summary.winning_model else None,
                json.dumps(summary.cost_analysis),
                json.dumps(summary.recommendations),
                summary.generated_at.isoformat()
            )
            
            await db_adapter.execute_async(query, params)
            
        except Exception as e:
            self.logger.error(f"Failed to save test summary: {e}")
    
    async def get_active_tests(self) -> List[TestConfiguration]:
        """アクティブなテスト一覧を取得"""
        return list(self.active_tests.values())
    
    async def get_test_results(self, test_id: str) -> List[TestResult]:
        """特定テストの結果を取得"""
        return self.test_results.get(test_id, [])
    
    async def stop_test(self, test_id: str) -> bool:
        """テストを停止"""
        try:
            if test_id in self.active_tests:
                del self.active_tests[test_id]
                
                # データベースのステータス更新
                query = "UPDATE ab_test_configs SET status = ? WHERE test_id = ?"
                await db_adapter.execute_async(query, (TestStatus.COMPLETED.value, test_id))
                
                self.logger.info(f"Stopped test {test_id}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to stop test: {e}")
            return False


# シングルトンインスタンス
ab_test_manager = ABTestManager()