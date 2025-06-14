#!/usr/bin/env python3
"""
Department Coordination System
部門間協調システム
"""

import asyncio
import json
import logging
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import sys
import os

# Add core modules to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ai_agent_base import DepartmentType, AnalysisRequest, AnalysisResult
from core.ai_orchestrator import AIOrchestrator, IntegratedDecision, MarketSituation


@dataclass
class DepartmentWeight:
    """部門別の重み設定"""
    department: DepartmentType
    base_weight: float
    dynamic_weight: float
    confidence_multiplier: float


@dataclass
class CoordinationResult:
    """協調結果"""
    final_decision: str
    confidence: float
    reasoning: str
    department_contributions: Dict[str, Dict[str, Any]]
    coordination_score: float
    dissent_analysis: Dict[str, Any]


class DepartmentCoordination:
    """
    部門間協調システム
    
    各部門AIの分析結果を統合し、競合や不整合を解決して
    最終的な取引判断を生成する
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 部門間の基本重み配分
        self.base_weights = {
            DepartmentType.TECHNICAL: 0.25,      # テクニカル分析
            DepartmentType.FUNDAMENTAL: 0.30,    # ファンダメンタル分析（最重要）
            DepartmentType.SENTIMENT: 0.20,      # センチメント分析
            DepartmentType.RISK: 0.15,           # リスク管理
            DepartmentType.EXECUTION: 0.10       # 執行・ポートフォリオ
        }
        
        # 市場状況別の重み調整係数
        self.situation_adjustments = {
            'high_volatility': {
                DepartmentType.RISK: 1.5,        # リスク管理重視
                DepartmentType.TECHNICAL: 1.2,   # テクニカル重視
                DepartmentType.SENTIMENT: 0.8    # センチメント軽視
            },
            'low_volatility': {
                DepartmentType.FUNDAMENTAL: 1.3,  # ファンダメンタル重視
                DepartmentType.SENTIMENT: 1.1,   # センチメント重視
                DepartmentType.RISK: 0.8         # リスク軽視
            },
            'news_driven': {
                DepartmentType.SENTIMENT: 1.4,   # センチメント最重要
                DepartmentType.FUNDAMENTAL: 1.2, # ファンダメンタル重視
                DepartmentType.TECHNICAL: 0.8    # テクニカル軽視
            },
            'technical_breakout': {
                DepartmentType.TECHNICAL: 1.5,   # テクニカル最重要
                DepartmentType.EXECUTION: 1.2,   # 執行重視
                DepartmentType.SENTIMENT: 0.9    # センチメント軽視
            }
        }
        
        # 意見の一致度評価閾値
        self.consensus_thresholds = {
            'strong_consensus': 0.8,    # 80%以上一致
            'moderate_consensus': 0.6,  # 60%以上一致
            'weak_consensus': 0.4,      # 40%以上一致
            'no_consensus': 0.0         # 40%未満一致
        }
        
        # 部門間の相互作用パターン
        self.interaction_patterns = {
            # テクニカル分析との相互作用
            (DepartmentType.TECHNICAL, DepartmentType.FUNDAMENTAL): {
                'conflict_resolution': 'time_horizon_based',
                'synergy_boost': 1.2
            },
            (DepartmentType.TECHNICAL, DepartmentType.SENTIMENT): {
                'conflict_resolution': 'confirmation_based',
                'synergy_boost': 1.1
            },
            
            # ファンダメンタル分析との相互作用
            (DepartmentType.FUNDAMENTAL, DepartmentType.SENTIMENT): {
                'conflict_resolution': 'strength_based',
                'synergy_boost': 1.15
            },
            (DepartmentType.FUNDAMENTAL, DepartmentType.RISK): {
                'conflict_resolution': 'conservative_bias',
                'synergy_boost': 1.1
            },
            
            # リスク管理との相互作用
            (DepartmentType.RISK, DepartmentType.EXECUTION): {
                'conflict_resolution': 'safety_first',
                'synergy_boost': 1.3
            }
        }
        
        self.logger.info("Department Coordination System initialized")
    
    async def coordinate_departments(
        self,
        department_results: Dict[DepartmentType, AnalysisResult],
        market_situation: MarketSituation
    ) -> CoordinationResult:
        """
        部門間協調処理のメインエントリーポイント
        """
        try:
            # 1. 市場状況に基づく重み調整
            adjusted_weights = self._calculate_dynamic_weights(
                department_results, market_situation
            )
            
            # 2. 部門間の意見一致度分析
            consensus_analysis = self._analyze_consensus(department_results)
            
            # 3. 競合解決処理
            conflict_resolution = await self._resolve_conflicts(
                department_results, adjusted_weights, consensus_analysis
            )
            
            # 4. 協調スコア計算
            coordination_score = self._calculate_coordination_score(
                consensus_analysis, conflict_resolution
            )
            
            # 5. 最終判断生成
            final_decision = self._generate_final_decision(
                department_results, adjusted_weights, conflict_resolution
            )
            
            # 6. 異議分析
            dissent_analysis = self._analyze_dissenting_opinions(
                department_results, final_decision
            )
            
            return CoordinationResult(
                final_decision=final_decision['action'],
                confidence=final_decision['confidence'],
                reasoning=final_decision['reasoning'],
                department_contributions=self._summarize_contributions(
                    department_results, adjusted_weights
                ),
                coordination_score=coordination_score,
                dissent_analysis=dissent_analysis
            )
            
        except Exception as e:
            self.logger.error(f"Department coordination failed: {e}")
            return self._generate_fallback_result(e)
    
    def _calculate_dynamic_weights(
        self,
        department_results: Dict[DepartmentType, AnalysisResult],
        market_situation: MarketSituation
    ) -> Dict[DepartmentType, DepartmentWeight]:
        """市場状況に基づく動的重み計算"""
        try:
            adjusted_weights = {}
            
            # 市場状況の特定
            situation_type = self._identify_market_situation_type(market_situation)
            
            for dept_type in department_results.keys():
                base_weight = self.base_weights.get(dept_type, 0.2)
                
                # 市場状況による調整
                situation_multiplier = self.situation_adjustments.get(
                    situation_type, {}
                ).get(dept_type, 1.0)
                
                # 信頼度による調整
                result = department_results[dept_type]
                confidence_multiplier = result.confidence
                
                # 動的重み計算
                dynamic_weight = base_weight * situation_multiplier
                
                adjusted_weights[dept_type] = DepartmentWeight(
                    department=dept_type,
                    base_weight=base_weight,
                    dynamic_weight=dynamic_weight,
                    confidence_multiplier=confidence_multiplier
                )
            
            # 正規化
            total_weight = sum(w.dynamic_weight for w in adjusted_weights.values())
            if total_weight > 0:
                for weight in adjusted_weights.values():
                    weight.dynamic_weight /= total_weight
            
            return adjusted_weights
            
        except Exception as e:
            self.logger.error(f"Dynamic weight calculation failed: {e}")
            return self._get_default_weights(department_results.keys())
    
    def _identify_market_situation_type(self, market_situation: MarketSituation) -> str:
        """市場状況タイプの特定"""
        try:
            volatility = market_situation.volatility_level
            news_impact = market_situation.news_impact_level
            trend_strength = market_situation.trend_strength
            
            # 高ボラティリティ環境
            if volatility > 0.8:
                return 'high_volatility'
            
            # 低ボラティリティ環境
            elif volatility < 0.3:
                return 'low_volatility'
            
            # ニュース主導の市場
            elif news_impact > 0.7:
                return 'news_driven'
            
            # テクニカル・ブレイクアウト
            elif trend_strength > 0.8:
                return 'technical_breakout'
            
            else:
                return 'normal'
                
        except Exception:
            return 'normal'
    
    def _get_default_weights(self, department_types: List[DepartmentType]) -> Dict[DepartmentType, DepartmentWeight]:
        """デフォルト重み設定の取得"""
        weights = {}
        equal_weight = 1.0 / len(department_types)
        
        for dept_type in department_types:
            weights[dept_type] = DepartmentWeight(
                department=dept_type,
                base_weight=equal_weight,
                dynamic_weight=equal_weight,
                confidence_multiplier=0.5
            )
        
        return weights
    
    def _analyze_consensus(
        self,
        department_results: Dict[DepartmentType, AnalysisResult]
    ) -> Dict[str, Any]:
        """部門間の意見一致度分析"""
        try:
            actions = [result.action for result in department_results.values()]
            confidences = [result.confidence for result in department_results.values()]
            
            # アクション別の集計
            action_counts = {}
            action_confidences = {}
            
            for action, confidence in zip(actions, confidences):
                if action not in action_counts:
                    action_counts[action] = 0
                    action_confidences[action] = []
                
                action_counts[action] += 1
                action_confidences[action].append(confidence)
            
            # 最多数のアクション
            majority_action = max(action_counts.items(), key=lambda x: x[1])[0]
            majority_count = action_counts[majority_action]
            
            # 一致度計算
            consensus_ratio = majority_count / len(actions)
            
            # 一致度レベルの決定
            if consensus_ratio >= self.consensus_thresholds['strong_consensus']:
                consensus_level = 'strong_consensus'
            elif consensus_ratio >= self.consensus_thresholds['moderate_consensus']:
                consensus_level = 'moderate_consensus'
            elif consensus_ratio >= self.consensus_thresholds['weak_consensus']:
                consensus_level = 'weak_consensus'
            else:
                consensus_level = 'no_consensus'
            
            # 信頼度統計
            avg_confidence = np.mean(confidences)
            confidence_std = np.std(confidences)
            
            return {
                'consensus_level': consensus_level,
                'consensus_ratio': float(consensus_ratio),
                'majority_action': majority_action,
                'action_distribution': action_counts,
                'average_confidence': float(avg_confidence),
                'confidence_deviation': float(confidence_std),
                'unanimous': len(set(actions)) == 1
            }
            
        except Exception as e:
            self.logger.error(f"Consensus analysis failed: {e}")
            return {
                'consensus_level': 'no_consensus',
                'consensus_ratio': 0.0,
                'majority_action': 'hold'
            }
    
    async def _resolve_conflicts(
        self,
        department_results: Dict[DepartmentType, AnalysisResult],
        adjusted_weights: Dict[DepartmentType, DepartmentWeight],
        consensus_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """部門間の競合解決"""
        try:
            conflict_resolutions = {}
            
            # 強い一致がある場合は競合解決不要
            if consensus_analysis['consensus_level'] == 'strong_consensus':
                return {'resolution_needed': False, 'method': 'consensus'}
            
            # 部門ペアごとの競合チェック
            departments = list(department_results.keys())
            
            for i, dept1 in enumerate(departments):
                for j, dept2 in enumerate(departments[i+1:], i+1):
                    conflict = self._detect_conflict(
                        department_results[dept1],
                        department_results[dept2]
                    )
                    
                    if conflict['has_conflict']:
                        resolution = await self._resolve_department_pair_conflict(
                            dept1, dept2,
                            department_results[dept1],
                            department_results[dept2],
                            adjusted_weights
                        )
                        
                        conflict_resolutions[f"{dept1.name}_{dept2.name}"] = resolution
            
            return {
                'resolution_needed': len(conflict_resolutions) > 0,
                'method': 'pairwise_resolution',
                'resolutions': conflict_resolutions
            }
            
        except Exception as e:
            self.logger.error(f"Conflict resolution failed: {e}")
            return {'resolution_needed': False, 'method': 'fallback'}
    
    def _detect_conflict(
        self,
        result1: AnalysisResult,
        result2: AnalysisResult
    ) -> Dict[str, Any]:
        """2つの部門結果間の競合検出"""
        try:
            # アクションの競合
            action_conflict = (
                result1.action == 'buy' and result2.action == 'sell'
            ) or (
                result1.action == 'sell' and result2.action == 'buy'
            )
            
            # 信頼度の差異
            confidence_gap = abs(result1.confidence - result2.confidence)
            
            # 競合の強度評価
            conflict_strength = 0.0
            if action_conflict:
                conflict_strength = min(result1.confidence, result2.confidence)
            
            return {
                'has_conflict': action_conflict,
                'conflict_strength': float(conflict_strength),
                'confidence_gap': float(confidence_gap),
                'severity': 'high' if conflict_strength > 0.7 else 'medium' if conflict_strength > 0.4 else 'low'
            }
            
        except Exception:
            return {'has_conflict': False, 'conflict_strength': 0.0}
    
    async def _resolve_department_pair_conflict(
        self,
        dept1: DepartmentType,
        dept2: DepartmentType,
        result1: AnalysisResult,
        result2: AnalysisResult,
        adjusted_weights: Dict[DepartmentType, DepartmentWeight]
    ) -> Dict[str, Any]:
        """部門ペア間の競合解決"""
        try:
            # 相互作用パターンの取得
            interaction_key = (dept1, dept2)
            if interaction_key not in self.interaction_patterns:
                interaction_key = (dept2, dept1)
            
            interaction = self.interaction_patterns.get(interaction_key, {
                'conflict_resolution': 'weight_based',
                'synergy_boost': 1.0
            })
            
            resolution_method = interaction['conflict_resolution']
            
            if resolution_method == 'weight_based':
                # 重み基準での解決
                weight1 = adjusted_weights[dept1].dynamic_weight
                weight2 = adjusted_weights[dept2].dynamic_weight
                
                winning_dept = dept1 if weight1 > weight2 else dept2
                winning_result = result1 if weight1 > weight2 else result2
                
            elif resolution_method == 'confidence_based':
                # 信頼度基準での解決
                winning_dept = dept1 if result1.confidence > result2.confidence else dept2
                winning_result = result1 if result1.confidence > result2.confidence else result2
                
            elif resolution_method == 'conservative_bias':
                # 保守的バイアスでの解決（よりリスク回避的な判断を採用）
                if result1.action == 'hold' or result2.action == 'hold':
                    winning_dept = dept1 if result1.action == 'hold' else dept2
                    winning_result = result1 if result1.action == 'hold' else result2
                else:
                    # 信頼度の低い方を採用（保守的）
                    winning_dept = dept1 if result1.confidence < result2.confidence else dept2
                    winning_result = result1 if result1.confidence < result2.confidence else result2
                    
            elif resolution_method == 'time_horizon_based':
                # 時間軸基準での解決（短期 vs 長期の観点）
                # 実装簡略化：より高い信頼度を採用
                winning_dept = dept1 if result1.confidence > result2.confidence else dept2
                winning_result = result1 if result1.confidence > result2.confidence else result2
                
            else:
                # デフォルト：重み基準
                weight1 = adjusted_weights[dept1].dynamic_weight
                weight2 = adjusted_weights[dept2].dynamic_weight
                
                winning_dept = dept1 if weight1 > weight2 else dept2
                winning_result = result1 if weight1 > weight2 else result2
            
            return {
                'resolution_method': resolution_method,
                'winning_department': winning_dept.name,
                'winning_action': winning_result.action,
                'resolution_confidence': winning_result.confidence * 0.8  # 競合解決による信頼度減少
            }
            
        except Exception as e:
            self.logger.error(f"Pairwise conflict resolution failed: {e}")
            return {
                'resolution_method': 'fallback',
                'winning_department': dept1.name,
                'winning_action': 'hold',
                'resolution_confidence': 0.3
            }
    
    def _calculate_coordination_score(
        self,
        consensus_analysis: Dict[str, Any],
        conflict_resolution: Dict[str, Any]
    ) -> float:
        """協調スコアの計算"""
        try:
            base_score = 0.5
            
            # 一致度による加点
            consensus_ratio = consensus_analysis.get('consensus_ratio', 0.0)
            consensus_bonus = consensus_ratio * 0.3
            
            # 信頼度の一貫性による加点
            confidence_deviation = consensus_analysis.get('confidence_deviation', 1.0)
            consistency_bonus = max(0, (1.0 - confidence_deviation)) * 0.2
            
            # 競合解決の必要性による減点
            if conflict_resolution.get('resolution_needed', False):
                conflict_penalty = 0.1
            else:
                conflict_penalty = 0.0
            
            # 全員一致による特別ボーナス
            unanimity_bonus = 0.2 if consensus_analysis.get('unanimous', False) else 0.0
            
            coordination_score = (
                base_score + 
                consensus_bonus + 
                consistency_bonus + 
                unanimity_bonus - 
                conflict_penalty
            )
            
            return float(max(0.0, min(1.0, coordination_score)))
            
        except Exception:
            return 0.5
    
    def _generate_final_decision(
        self,
        department_results: Dict[DepartmentType, AnalysisResult],
        adjusted_weights: Dict[DepartmentType, DepartmentWeight],
        conflict_resolution: Dict[str, Any]
    ) -> Dict[str, Any]:
        """最終判断の生成"""
        try:
            # 重み付き投票による最終判断
            action_scores = {'buy': 0.0, 'sell': 0.0, 'hold': 0.0}
            weighted_confidence = 0.0
            reasoning_components = []
            
            for dept_type, result in department_results.items():
                weight = adjusted_weights[dept_type].dynamic_weight
                confidence = result.confidence
                
                # 重み付きスコア加算
                action_scores[result.action] += weight * confidence
                weighted_confidence += weight * confidence
                
                # 推論コンポーネント追加
                reasoning_components.append(
                    f"{dept_type.name}: {result.action} (信頼度: {confidence:.2f}, 重み: {weight:.2f})"
                )
            
            # 最高スコアのアクションを採用
            final_action = max(action_scores.items(), key=lambda x: x[1])[0]
            
            # 最終信頼度の計算
            total_weight = sum(w.dynamic_weight for w in adjusted_weights.values())
            if total_weight > 0:
                final_confidence = weighted_confidence / total_weight
            else:
                final_confidence = 0.5
            
            # 推論文の生成
            top_score = action_scores[final_action]
            score_difference = top_score - max(score for action, score in action_scores.items() if action != final_action)
            
            reasoning = (
                f"重み付き協調による判断: {final_action} "
                f"(スコア差: {score_difference:.3f})\n"
                f"主要根拠: {'; '.join(reasoning_components[:3])}"
            )
            
            # 競合解決の影響を反映
            if conflict_resolution.get('resolution_needed', False):
                final_confidence *= 0.9  # 競合があった場合は信頼度を若干下げる
                reasoning += f"\n競合解決: {len(conflict_resolution.get('resolutions', {}))}件の競合を解決"
            
            return {
                'action': final_action,
                'confidence': float(final_confidence),
                'reasoning': reasoning,
                'action_scores': {k: float(v) for k, v in action_scores.items()},
                'score_difference': float(score_difference)
            }
            
        except Exception as e:
            self.logger.error(f"Final decision generation failed: {e}")
            return {
                'action': 'hold',
                'confidence': 0.3,
                'reasoning': f'最終判断生成エラー: {str(e)}',
                'action_scores': {'buy': 0.0, 'sell': 0.0, 'hold': 1.0}
            }
    
    def _analyze_dissenting_opinions(
        self,
        department_results: Dict[DepartmentType, AnalysisResult],
        final_decision: Dict[str, Any]
    ) -> Dict[str, Any]:
        """異議意見の分析"""
        try:
            final_action = final_decision['action']
            dissenting_departments = []
            dissent_strength = 0.0
            
            for dept_type, result in department_results.items():
                if result.action != final_action:
                    dissenting_departments.append({
                        'department': dept_type.name,
                        'action': result.action,
                        'confidence': result.confidence,
                        'reasoning': result.reasoning
                    })
                    dissent_strength += result.confidence
            
            # 異議の強度評価
            if dissent_strength > 1.5:
                dissent_level = 'strong'
            elif dissent_strength > 0.8:
                dissent_level = 'moderate'
            elif dissent_strength > 0.3:
                dissent_level = 'weak'
            else:
                dissent_level = 'minimal'
            
            return {
                'dissent_level': dissent_level,
                'dissent_strength': float(dissent_strength),
                'dissenting_departments': dissenting_departments,
                'dissent_count': len(dissenting_departments),
                'requires_attention': dissent_level in ['strong', 'moderate']
            }
            
        except Exception as e:
            self.logger.error(f"Dissent analysis failed: {e}")
            return {
                'dissent_level': 'unknown',
                'dissent_strength': 0.0,
                'dissenting_departments': []
            }
    
    def _summarize_contributions(
        self,
        department_results: Dict[DepartmentType, AnalysisResult],
        adjusted_weights: Dict[DepartmentType, DepartmentWeight]
    ) -> Dict[str, Dict[str, Any]]:
        """各部門の貢献度まとめ"""
        try:
            contributions = {}
            
            for dept_type, result in department_results.items():
                weight = adjusted_weights[dept_type]
                
                contributions[dept_type.name] = {
                    'action': result.action,
                    'confidence': result.confidence,
                    'reasoning': result.reasoning,
                    'base_weight': weight.base_weight,
                    'dynamic_weight': weight.dynamic_weight,
                    'final_contribution': weight.dynamic_weight * result.confidence,
                    'key_factors': getattr(result, 'key_factors', [])
                }
            
            return contributions
            
        except Exception as e:
            self.logger.error(f"Contribution summary failed: {e}")
            return {}
    
    def _generate_fallback_result(self, error: Exception) -> CoordinationResult:
        """フォールバック結果の生成"""
        return CoordinationResult(
            final_decision='hold',
            confidence=0.2,
            reasoning=f'協調システムエラーにより保守的判断: {str(error)}',
            department_contributions={},
            coordination_score=0.0,
            dissent_analysis={'dissent_level': 'unknown'}
        )


class CoordinationWorkflow:
    """
    協調ワークフロー管理
    
    複雑な協調プロセスの実行順序とタイミングを管理
    """
    
    def __init__(self, coordination_system: DepartmentCoordination):
        self.coordination = coordination_system
        self.logger = logging.getLogger(__name__)
        
        # ワークフロー設定
        self.workflow_config = {
            'max_coordination_iterations': 3,
            'convergence_threshold': 0.1,
            'timeout_seconds': 30
        }
    
    async def execute_coordination_workflow(
        self,
        department_results: Dict[DepartmentType, AnalysisResult],
        market_situation: MarketSituation
    ) -> CoordinationResult:
        """協調ワークフローの実行"""
        try:
            start_time = datetime.now()
            timeout = timedelta(seconds=self.workflow_config['timeout_seconds'])
            
            # 初回協調
            coordination_result = await self.coordination.coordinate_departments(
                department_results, market_situation
            )
            
            # 反復改善プロセス
            iteration = 1
            while iteration < self.workflow_config['max_coordination_iterations']:
                if datetime.now() - start_time > timeout:
                    self.logger.warning("Coordination workflow timeout")
                    break
                
                # 改善の必要性チェック
                if not self._needs_improvement(coordination_result):
                    break
                
                # 協調結果の改善
                improved_result = await self._improve_coordination(
                    department_results, coordination_result, market_situation
                )
                
                # 収束チェック
                if self._has_converged(coordination_result, improved_result):
                    coordination_result = improved_result
                    break
                
                coordination_result = improved_result
                iteration += 1
            
            return coordination_result
            
        except Exception as e:
            self.logger.error(f"Coordination workflow failed: {e}")
            return self.coordination._generate_fallback_result(e)
    
    def _needs_improvement(self, result: CoordinationResult) -> bool:
        """改善の必要性チェック"""
        return (
            result.coordination_score < 0.7 or
            result.dissent_analysis.get('dissent_level') == 'strong' or
            result.confidence < 0.5
        )
    
    async def _improve_coordination(
        self,
        department_results: Dict[DepartmentType, AnalysisResult],
        current_result: CoordinationResult,
        market_situation: MarketSituation
    ) -> CoordinationResult:
        """協調結果の改善"""
        try:
            # 異議の強い部門に対する重み調整
            dissenting_depts = current_result.dissent_analysis.get('dissenting_departments', [])
            
            if dissenting_depts:
                # 異議のある部門の重みを一時的に増加
                adjusted_market_situation = self._adjust_market_situation_for_dissent(
                    market_situation, dissenting_depts
                )
                
                return await self.coordination.coordinate_departments(
                    department_results, adjusted_market_situation
                )
            
            return current_result
            
        except Exception:
            return current_result
    
    def _adjust_market_situation_for_dissent(
        self,
        market_situation: MarketSituation,
        dissenting_departments: List[Dict[str, Any]]
    ) -> MarketSituation:
        """異議に基づく市場状況調整"""
        # 簡略化：市場状況を微調整して異議のある部門の重みを増加
        adjusted_situation = MarketSituation(
            volatility_level=market_situation.volatility_level * 1.1,
            trend_strength=market_situation.trend_strength,
            news_impact_level=market_situation.news_impact_level * 1.05,
            liquidity_condition=market_situation.liquidity_condition
        )
        
        return adjusted_situation
    
    def _has_converged(
        self,
        result1: CoordinationResult,
        result2: CoordinationResult
    ) -> bool:
        """収束判定"""
        try:
            # 信頼度の変化が閾値以下なら収束
            confidence_change = abs(result1.confidence - result2.confidence)
            
            # 協調スコアの変化が閾値以下なら収束
            score_change = abs(result1.coordination_score - result2.coordination_score)
            
            threshold = self.workflow_config['convergence_threshold']
            
            return (confidence_change < threshold and score_change < threshold)
            
        except Exception:
            return True  # エラー時は収束として処理