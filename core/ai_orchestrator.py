#!/usr/bin/env python3
"""
AI Orchestrator for Department-Based AI System
部門別AIシステムの中央オーケストレーター
"""

import asyncio
import datetime
import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sys
import os


from core.ai_agent_base import (
    AIAgentBase, DepartmentType, AnalysisRequest, AnalysisResult, AnalysisPriority
)
from core.redis_manager import redis_manager


class DecisionType(Enum):
    """意思決定タイプ"""
    TRADE_SIGNAL = "trade_signal"
    RISK_ASSESSMENT = "risk_assessment" 
    MARKET_ANALYSIS = "market_analysis"
    PORTFOLIO_REBALANCE = "portfolio_rebalance"
    EMERGENCY_ACTION = "emergency_action"


@dataclass
class MarketSituation:
    """市場状況データ"""
    price_data: Dict[str, Any]
    technical_indicators: Dict[str, Any]
    news_data: List[Dict[str, Any]]
    economic_data: Dict[str, Any]
    portfolio_state: Dict[str, Any]
    risk_metrics: Dict[str, Any]
    timestamp: datetime.datetime
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class IntegratedDecision:
    """統合された意思決定"""
    decision_id: str
    decision_type: DecisionType
    departments_involved: List[DepartmentType]
    consensus_confidence: float
    final_decision: Dict[str, Any]
    department_results: Dict[str, AnalysisResult]
    integration_logic: str
    timestamp: datetime.datetime
    processing_time_ms: float
    total_cost_usd: float
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['decision_type'] = self.decision_type.value
        result['departments_involved'] = [dept.value for dept in self.departments_involved]
        result['department_results'] = {
            dept: res.to_dict() for dept, res in self.department_results.items()
        }
        return result


class AIOrchestrator:
    """
    部門別AIシステムの中央オーケストレーター
    
    各部門AIエージェントを統括し、協調的な意思決定を実現
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.departments: Dict[DepartmentType, AIAgentBase] = {}
        
        # 決定履歴
        self.decision_history: List[IntegratedDecision] = []
        
        # 部門間協調ルール
        self.collaboration_rules = self._initialize_collaboration_rules()
        
        # 統計情報
        self.total_decisions = 0
        self.avg_processing_time = 0.0
        self.total_cost = 0.0
        
        self.logger.info("AI Orchestrator initialized")
    
    def register_department(self, department: DepartmentType, agent: AIAgentBase):
        """部門AIエージェントを登録"""
        self.departments[department] = agent
        self.logger.info(f"Registered {department.value} department")
    
    def _initialize_collaboration_rules(self) -> Dict[str, Any]:
        """部門間協調ルールの初期化"""
        return {
            # トレードシグナル生成時の部門協調
            DecisionType.TRADE_SIGNAL: {
                'required_departments': [
                    DepartmentType.TECHNICAL,
                    DepartmentType.SENTIMENT,
                    DepartmentType.RISK
                ],
                'optional_departments': [
                    DepartmentType.FUNDAMENTAL
                ],
                'consensus_threshold': 0.6,
                'weights': {
                    DepartmentType.TECHNICAL: 0.35,
                    DepartmentType.FUNDAMENTAL: 0.25,
                    DepartmentType.SENTIMENT: 0.20,
                    DepartmentType.RISK: 0.20
                }
            },
            
            # リスク評価時の協調
            DecisionType.RISK_ASSESSMENT: {
                'required_departments': [
                    DepartmentType.RISK,
                    DepartmentType.PORTFOLIO
                ],
                'optional_departments': [
                    DepartmentType.TECHNICAL,
                    DepartmentType.FUNDAMENTAL
                ],
                'consensus_threshold': 0.7,
                'weights': {
                    DepartmentType.RISK: 0.50,
                    DepartmentType.PORTFOLIO: 0.30,
                    DepartmentType.TECHNICAL: 0.10,
                    DepartmentType.FUNDAMENTAL: 0.10
                }
            },
            
            # 市場分析時の協調
            DecisionType.MARKET_ANALYSIS: {
                'required_departments': [
                    DepartmentType.TECHNICAL,
                    DepartmentType.FUNDAMENTAL,
                    DepartmentType.SENTIMENT
                ],
                'optional_departments': [],
                'consensus_threshold': 0.5,
                'weights': {
                    DepartmentType.TECHNICAL: 0.35,
                    DepartmentType.FUNDAMENTAL: 0.35,
                    DepartmentType.SENTIMENT: 0.30
                }
            },
            
            # ポートフォリオリバランス時の協調
            DecisionType.PORTFOLIO_REBALANCE: {
                'required_departments': [
                    DepartmentType.PORTFOLIO,
                    DepartmentType.RISK,
                    DepartmentType.EXECUTION
                ],
                'optional_departments': [
                    DepartmentType.TECHNICAL,
                    DepartmentType.FUNDAMENTAL
                ],
                'consensus_threshold': 0.75,
                'weights': {
                    DepartmentType.PORTFOLIO: 0.40,
                    DepartmentType.RISK: 0.30,
                    DepartmentType.EXECUTION: 0.20,
                    DepartmentType.TECHNICAL: 0.05,
                    DepartmentType.FUNDAMENTAL: 0.05
                }
            },
            
            # 緊急時の協調
            DecisionType.EMERGENCY_ACTION: {
                'required_departments': [
                    DepartmentType.RISK,
                    DepartmentType.EXECUTION
                ],
                'optional_departments': [],
                'consensus_threshold': 0.8,
                'weights': {
                    DepartmentType.RISK: 0.70,
                    DepartmentType.EXECUTION: 0.30
                }
            }
        }
    
    async def analyze_market_situation(
        self, 
        market_data: MarketSituation,
        decision_type: DecisionType = DecisionType.TRADE_SIGNAL,
        priority: AnalysisPriority = AnalysisPriority.MEDIUM
    ) -> IntegratedDecision:
        """
        市場状況を総合分析して統合された意思決定を返す
        """
        start_time = datetime.datetime.now()
        decision_id = str(uuid.uuid4())
        
        try:
            # 協調ルール取得
            rules = self.collaboration_rules.get(decision_type)
            if not rules:
                raise ValueError(f"No collaboration rules for {decision_type}")
            
            # 必要部門と任意部門を特定
            required_depts = rules['required_departments']
            optional_depts = rules.get('optional_departments', [])
            all_depts = required_depts + optional_depts
            
            # 利用可能な部門をフィルタ
            available_depts = [dept for dept in all_depts if dept in self.departments]
            missing_required = [dept for dept in required_depts if dept not in self.departments]
            
            if missing_required:
                raise RuntimeError(f"Missing required departments: {missing_required}")
            
            # 部門別分析リクエスト作成
            requests = self._create_department_requests(
                decision_id, market_data, available_depts, priority
            )
            
            # 並行分析実行
            results = await self._execute_parallel_analysis(requests)
            
            # 結果統合
            integrated_decision = self._integrate_department_results(
                decision_id, decision_type, results, rules, start_time
            )
            
            # 履歴保存
            self._save_decision_history(integrated_decision)
            
            # Redis通知
            await self._publish_decision_to_redis(integrated_decision)
            
            # 統計更新
            self._update_orchestrator_statistics(integrated_decision)
            
            self.logger.info(
                f"Completed integrated analysis {decision_id} "
                f"in {integrated_decision.processing_time_ms:.2f}ms "
                f"with confidence {integrated_decision.consensus_confidence:.3f}"
            )
            
            return integrated_decision
            
        except Exception as e:
            self.logger.error(f"Failed to analyze market situation: {e}")
            
            # エラー時のフォールバック決定
            processing_time = (datetime.datetime.now() - start_time).total_seconds() * 1000
            return IntegratedDecision(
                decision_id=decision_id,
                decision_type=decision_type,
                departments_involved=[],
                consensus_confidence=0.0,
                final_decision={'error': str(e), 'action': 'hold'},
                department_results={},
                integration_logic=f"Error fallback: {str(e)}",
                timestamp=datetime.datetime.now(),
                processing_time_ms=processing_time,
                total_cost_usd=0.0
            )
    
    def _create_department_requests(
        self,
        decision_id: str,
        market_data: MarketSituation,
        departments: List[DepartmentType],
        priority: AnalysisPriority
    ) -> List[AnalysisRequest]:
        """部門別分析リクエストを作成"""
        requests = []
        
        for dept in departments:
            # 部門別にデータをフィルタリング
            dept_data = self._filter_data_for_department(dept, market_data)
            
            request = AnalysisRequest(
                request_id=f"{decision_id}_{dept.value}",
                department=dept,
                priority=priority,
                data=dept_data,
                context={
                    'decision_id': decision_id,
                    'market_timestamp': market_data.timestamp.isoformat(),
                    'other_departments': [d.value for d in departments if d != dept]
                }
            )
            requests.append(request)
        
        return requests
    
    def _filter_data_for_department(
        self, 
        department: DepartmentType, 
        market_data: MarketSituation
    ) -> Dict[str, Any]:
        """部門に関連するデータのみを抽出"""
        base_data = {
            'timestamp': market_data.timestamp.isoformat(),
            'symbol': 'USD/JPY'  # デフォルト
        }
        
        if department == DepartmentType.TECHNICAL:
            return {
                **base_data,
                'price_data': market_data.price_data,
                'technical_indicators': market_data.technical_indicators
            }
        
        elif department == DepartmentType.FUNDAMENTAL:
            return {
                **base_data,
                'economic_data': market_data.economic_data,
                'price_data': market_data.price_data,  # 価格トレンド参考
                'news_data': [
                    news for news in market_data.news_data 
                    if self._is_fundamental_news(news)
                ]
            }
        
        elif department == DepartmentType.SENTIMENT:
            return {
                **base_data,
                'news_data': market_data.news_data,
                'price_data': market_data.price_data  # 価格反応参考
            }
        
        elif department == DepartmentType.RISK:
            return {
                **base_data,
                'portfolio_state': market_data.portfolio_state,
                'risk_metrics': market_data.risk_metrics,
                'price_data': market_data.price_data
            }
        
        elif department == DepartmentType.EXECUTION:
            return {
                **base_data,
                'price_data': market_data.price_data,
                'portfolio_state': market_data.portfolio_state
            }
        
        elif department == DepartmentType.PORTFOLIO:
            return {
                **base_data,
                'portfolio_state': market_data.portfolio_state,
                'risk_metrics': market_data.risk_metrics,
                'price_data': market_data.price_data
            }
        
        return base_data
    
    def _is_fundamental_news(self, news: Dict[str, Any]) -> bool:
        """ニュースがファンダメンタル分析に関連するかチェック"""
        fundamental_keywords = [
            'central bank', 'interest rate', 'inflation', 'gdp', 'employment',
            'unemployment', 'policy', 'economic', 'monetary', 'fiscal'
        ]
        
        content = news.get('title', '') + ' ' + news.get('content', '')
        content_lower = content.lower()
        
        return any(keyword in content_lower for keyword in fundamental_keywords)
    
    async def _execute_parallel_analysis(
        self, 
        requests: List[AnalysisRequest]
    ) -> Dict[DepartmentType, AnalysisResult]:
        """並行して部門分析を実行"""
        tasks = []
        request_map = {}
        
        for request in requests:
            department = request.department
            agent = self.departments.get(department)
            
            if agent:
                task = agent.process_request(request)
                tasks.append(task)
                request_map[task] = department
            else:
                self.logger.warning(f"No agent available for {department.value}")
        
        # 並行実行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 結果をマッピング
        department_results = {}
        for task, result in zip(tasks, results):
            department = request_map[task]
            
            if isinstance(result, Exception):
                self.logger.error(f"Department {department.value} analysis failed: {result}")
                # エラー時のダミー結果
                department_results[department] = AnalysisResult(
                    request_id=f"error_{department.value}",
                    department=department,
                    confidence=0.0,
                    decision={'error': str(result)},
                    reasoning=f"Analysis failed: {str(result)}",
                    metadata={'error': True},
                    timestamp=datetime.datetime.now(),
                    processing_time_ms=0.0,
                    cost_usd=0.0,
                    model_used='none'
                )
            else:
                department_results[department] = result
        
        return department_results
    
    def _integrate_department_results(
        self,
        decision_id: str,
        decision_type: DecisionType,
        results: Dict[DepartmentType, AnalysisResult],
        rules: Dict[str, Any],
        start_time: datetime.datetime
    ) -> IntegratedDecision:
        """部門結果を統合して最終決定を作成"""
        
        # 重み付きコンセンサス計算
        weights = rules['weights']
        consensus_confidence = 0.0
        total_weight = 0.0
        
        valid_results = {
            dept: result for dept, result in results.items()
            if not result.metadata.get('error', False)
        }
        
        for dept, result in valid_results.items():
            weight = weights.get(dept, 0.0)
            consensus_confidence += result.confidence * weight
            total_weight += weight
        
        # 正規化
        if total_weight > 0:
            consensus_confidence /= total_weight
        
        # 最終決定ロジック
        final_decision = self._create_final_decision(
            decision_type, valid_results, consensus_confidence, rules
        )
        
        # 統合ロジックの説明
        integration_logic = self._generate_integration_explanation(
            valid_results, weights, consensus_confidence
        )
        
        # コスト計算
        total_cost = sum(result.cost_usd for result in results.values())
        
        # 処理時間計算
        processing_time = (datetime.datetime.now() - start_time).total_seconds() * 1000
        
        return IntegratedDecision(
            decision_id=decision_id,
            decision_type=decision_type,
            departments_involved=list(results.keys()),
            consensus_confidence=consensus_confidence,
            final_decision=final_decision,
            department_results=results,
            integration_logic=integration_logic,
            timestamp=datetime.datetime.now(),
            processing_time_ms=processing_time,
            total_cost_usd=total_cost
        )
    
    def _create_final_decision(
        self,
        decision_type: DecisionType,
        results: Dict[DepartmentType, AnalysisResult],
        consensus_confidence: float,
        rules: Dict[str, Any]
    ) -> Dict[str, Any]:
        """最終決定を作成"""
        
        consensus_threshold = rules['consensus_threshold']
        
        # コンセンサス閾値チェック
        if consensus_confidence < consensus_threshold:
            return {
                'action': 'hold',
                'reason': 'consensus_below_threshold',
                'consensus_confidence': consensus_confidence,
                'threshold': consensus_threshold
            }
        
        if decision_type == DecisionType.TRADE_SIGNAL:
            return self._create_trade_signal_decision(results, consensus_confidence)
        elif decision_type == DecisionType.RISK_ASSESSMENT:
            return self._create_risk_assessment_decision(results, consensus_confidence)
        elif decision_type == DecisionType.MARKET_ANALYSIS:
            return self._create_market_analysis_decision(results, consensus_confidence)
        elif decision_type == DecisionType.PORTFOLIO_REBALANCE:
            return self._create_portfolio_rebalance_decision(results, consensus_confidence)
        elif decision_type == DecisionType.EMERGENCY_ACTION:
            return self._create_emergency_action_decision(results, consensus_confidence)
        else:
            return {'action': 'unknown', 'decision_type': decision_type.value}
    
    def _create_trade_signal_decision(
        self, 
        results: Dict[DepartmentType, AnalysisResult], 
        confidence: float
    ) -> Dict[str, Any]:
        """トレードシグナル決定を作成"""
        
        # 各部門の推奨アクションを集計
        actions = []
        for dept, result in results.items():
            action = result.decision.get('action', 'hold')
            actions.append(action)
        
        # 多数決ロジック
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # 最多得票のアクション
        final_action = max(action_counts, key=action_counts.get)
        
        # リスク部門の承認チェック
        risk_result = results.get(DepartmentType.RISK)
        if risk_result and risk_result.decision.get('risk_level', 'medium') == 'high':
            final_action = 'hold'
        
        return {
            'action': final_action,
            'confidence': confidence,
            'department_votes': action_counts,
            'risk_approved': risk_result.confidence > 0.6 if risk_result else False,
            'position_size': self._calculate_position_size(results, confidence)
        }
    
    def _create_risk_assessment_decision(
        self, 
        results: Dict[DepartmentType, AnalysisResult], 
        confidence: float
    ) -> Dict[str, Any]:
        """リスク評価決定を作成"""
        
        risk_result = results.get(DepartmentType.RISK)
        portfolio_result = results.get(DepartmentType.PORTFOLIO)
        
        risk_level = 'medium'
        if risk_result:
            risk_score = risk_result.decision.get('risk_score', 0.5)
            if risk_score > 0.7:
                risk_level = 'high'
            elif risk_score < 0.3:
                risk_level = 'low'
        
        return {
            'risk_level': risk_level,
            'confidence': confidence,
            'recommendations': self._get_risk_recommendations(results),
            'portfolio_health': portfolio_result.decision.get('health_score', 0.5) if portfolio_result else 0.5
        }
    
    def _create_market_analysis_decision(
        self, 
        results: Dict[DepartmentType, AnalysisResult], 
        confidence: float
    ) -> Dict[str, Any]:
        """市場分析決定を作成"""
        
        trend_consensus = self._analyze_trend_consensus(results)
        
        return {
            'market_trend': trend_consensus['trend'],
            'trend_strength': trend_consensus['strength'],
            'confidence': confidence,
            'key_factors': self._extract_key_factors(results),
            'outlook_timeframe': self._determine_outlook_timeframe(results)
        }
    
    def _create_portfolio_rebalance_decision(
        self, 
        results: Dict[DepartmentType, AnalysisResult], 
        confidence: float
    ) -> Dict[str, Any]:
        """ポートフォリオリバランス決定を作成"""
        
        portfolio_result = results.get(DepartmentType.PORTFOLIO)
        
        return {
            'rebalance_needed': confidence > 0.6,
            'confidence': confidence,
            'suggested_allocations': portfolio_result.decision.get('allocations', {}) if portfolio_result else {},
            'rebalance_urgency': self._assess_rebalance_urgency(results)
        }
    
    def _create_emergency_action_decision(
        self, 
        results: Dict[DepartmentType, AnalysisResult], 
        confidence: float
    ) -> Dict[str, Any]:
        """緊急アクション決定を作成"""
        
        risk_result = results.get(DepartmentType.RISK)
        execution_result = results.get(DepartmentType.EXECUTION)
        
        return {
            'emergency_level': 'critical' if confidence > 0.8 else 'warning',
            'immediate_actions': self._get_immediate_actions(results),
            'confidence': confidence,
            'risk_mitigation': risk_result.decision.get('mitigation_steps', []) if risk_result else []
        }
    
    def _calculate_position_size(
        self, 
        results: Dict[DepartmentType, AnalysisResult], 
        confidence: float
    ) -> float:
        """ポジションサイズを計算"""
        
        base_size = 0.1  # ベースサイズ
        
        # 信頼度調整
        confidence_multiplier = min(confidence * 2, 1.0)
        
        # リスク調整
        risk_result = results.get(DepartmentType.RISK)
        if risk_result:
            risk_multiplier = 1.0 - risk_result.decision.get('risk_score', 0.5)
        else:
            risk_multiplier = 0.5
        
        return base_size * confidence_multiplier * risk_multiplier
    
    def _get_risk_recommendations(
        self, 
        results: Dict[DepartmentType, AnalysisResult]
    ) -> List[str]:
        """リスク推奨事項を取得"""
        recommendations = []
        
        risk_result = results.get(DepartmentType.RISK)
        if risk_result:
            recommendations.extend(risk_result.decision.get('recommendations', []))
        
        return recommendations
    
    def _analyze_trend_consensus(
        self, 
        results: Dict[DepartmentType, AnalysisResult]
    ) -> Dict[str, Any]:
        """トレンドコンセンサスを分析"""
        
        trends = []
        for dept, result in results.items():
            trend = result.decision.get('trend', 'neutral')
            trends.append(trend)
        
        # 最頻値でトレンド決定
        from collections import Counter
        trend_counts = Counter(trends)
        dominant_trend = trend_counts.most_common(1)[0][0]
        strength = trend_counts[dominant_trend] / len(trends)
        
        return {
            'trend': dominant_trend,
            'strength': strength
        }
    
    def _extract_key_factors(
        self, 
        results: Dict[DepartmentType, AnalysisResult]
    ) -> List[str]:
        """主要要因を抽出"""
        factors = []
        
        for dept, result in results.items():
            dept_factors = result.decision.get('key_factors', [])
            factors.extend(dept_factors)
        
        return list(set(factors))  # 重複除去
    
    def _determine_outlook_timeframe(
        self, 
        results: Dict[DepartmentType, AnalysisResult]
    ) -> str:
        """見通し期間を決定"""
        
        # テクニカル：短期、ファンダメンタル：長期の重み付け
        technical_result = results.get(DepartmentType.TECHNICAL)
        fundamental_result = results.get(DepartmentType.FUNDAMENTAL)
        
        if technical_result and technical_result.confidence > 0.7:
            return 'short_term'
        elif fundamental_result and fundamental_result.confidence > 0.7:
            return 'long_term'
        else:
            return 'medium_term'
    
    def _assess_rebalance_urgency(
        self, 
        results: Dict[DepartmentType, AnalysisResult]
    ) -> str:
        """リバランス緊急度を評価"""
        
        risk_result = results.get(DepartmentType.RISK)
        if risk_result:
            risk_score = risk_result.decision.get('risk_score', 0.5)
            if risk_score > 0.8:
                return 'urgent'
            elif risk_score > 0.6:
                return 'moderate'
        
        return 'low'
    
    def _get_immediate_actions(
        self, 
        results: Dict[DepartmentType, AnalysisResult]
    ) -> List[str]:
        """即座に取るべきアクションを取得"""
        actions = []
        
        risk_result = results.get(DepartmentType.RISK)
        if risk_result:
            actions.extend(risk_result.decision.get('immediate_actions', []))
        
        execution_result = results.get(DepartmentType.EXECUTION)
        if execution_result:
            actions.extend(execution_result.decision.get('immediate_actions', []))
        
        return actions
    
    def _generate_integration_explanation(
        self,
        results: Dict[DepartmentType, AnalysisResult],
        weights: Dict[DepartmentType, float],
        consensus_confidence: float
    ) -> str:
        """統合ロジックの説明を生成"""
        
        explanations = []
        explanations.append(f"統合信頼度: {consensus_confidence:.3f}")
        
        for dept, result in results.items():
            weight = weights.get(dept, 0.0)
            explanations.append(
                f"{dept.value}: 信頼度{result.confidence:.3f} "
                f"(重み{weight:.2f}) - {result.reasoning[:50]}..."
            )
        
        return " | ".join(explanations)
    
    def _save_decision_history(self, decision: IntegratedDecision):
        """決定履歴を保存"""
        self.decision_history.append(decision)
        
        # 履歴サイズ制限（最新1000件）
        if len(self.decision_history) > 1000:
            self.decision_history = self.decision_history[-1000:]
    
    async def _publish_decision_to_redis(self, decision: IntegratedDecision):
        """決定をRedisに通知"""
        try:
            decision_data = decision.to_dict()
            
            # ストリームに追加
            await asyncio.to_thread(
                redis_manager.add_to_stream,
                'ai_integrated_decisions',
                decision_data
            )
            
            # 即座通知用チャンネル
            await asyncio.to_thread(
                redis_manager.publish,
                'ai_decision_updates',
                decision_data
            )
            
        except Exception as e:
            self.logger.error(f"Failed to publish decision to Redis: {e}")
    
    def _update_orchestrator_statistics(self, decision: IntegratedDecision):
        """オーケストレーター統計を更新"""
        self.total_decisions += 1
        self.total_cost += decision.total_cost_usd
        
        # 移動平均で処理時間更新
        alpha = 0.1
        self.avg_processing_time = (
            alpha * decision.processing_time_ms + 
            (1 - alpha) * self.avg_processing_time
        )
    
    def get_orchestrator_statistics(self) -> Dict[str, Any]:
        """オーケストレーター統計を取得"""
        return {
            'total_decisions': self.total_decisions,
            'avg_processing_time_ms': self.avg_processing_time,
            'total_cost_usd': self.total_cost,
            'registered_departments': list(self.departments.keys()),
            'department_statistics': {
                dept.value: agent.get_statistics()
                for dept, agent in self.departments.items()
            }
        }
    
    def get_recent_decisions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """最近の決定を取得"""
        recent = self.decision_history[-limit:] if self.decision_history else []
        return [decision.to_dict() for decision in reversed(recent)]