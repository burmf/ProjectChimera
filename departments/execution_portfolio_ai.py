#!/usr/bin/env python3
"""
Execution & Portfolio Management AI Department
執行・ポートフォリオ管理部門AI
"""

import json
import logging
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import sys
import os
from scipy import optimize


from core.ai_agent_base import AIAgentBase, DepartmentType, AnalysisRequest, AnalysisResult
from core.department_prompts import DepartmentPrompts, PromptFormatter


class ExecutionPortfolioAI(AIAgentBase):
    """
    執行・ポートフォリオ管理専門AIエージェント
    
    最適執行戦略、ポートフォリオ最適化、アセットアロケーション、
    リバランシング戦略を専門とする
    """
    
    def __init__(self, model_config: Dict[str, Any] = None):
        super().__init__(DepartmentType.EXECUTION, model_config)
        
        # 執行戦略パラメータ
        self.execution_params = {
            'twap_time_slices': 10,                # TWAP分割数
            'vwap_participation_rate': 0.15,      # VWAP参加率
            'implementation_shortfall_lambda': 0.5, # IS最適化パラメータ
            'max_market_impact_bps': 50,          # 最大マーケットインパクト
            'min_execution_time_minutes': 5,      # 最小執行時間
            'max_execution_time_minutes': 240     # 最大執行時間
        }
        
        # 流動性評価パラメータ
        self.liquidity_params = {
            'excellent_spread_bps': 1,   # 優良スプレッド
            'good_spread_bps': 2,        # 良好スプレッド
            'normal_spread_bps': 3,      # 通常スプレッド
            'poor_spread_bps': 5,        # 不良スプレッド
            'volume_threshold_ratio': 0.1 # 出来高閾値比率
        }
        
        # ポートフォリオ最適化パラメータ
        self.portfolio_params = {
            'risk_aversion': 3.0,              # リスク回避度
            'transaction_cost_bps': 5,         # 取引コスト
            'rebalance_threshold_pct': 5.0,    # リバランシング閾値
            'min_position_pct': 0.5,           # 最小ポジション比率
            'max_position_pct': 40.0,          # 最大ポジション比率
            'cash_target_pct': 5.0             # 目標キャッシュ比率
        }
        
        # 通貨ペア別執行特性
        self.execution_characteristics = {
            'USD/JPY': {
                'typical_spread_bps': 0.5,
                'market_impact_coefficient': 0.1,
                'optimal_slice_size': 10000000,  # 1000万通貨単位
                'peak_liquidity_hours': [8, 9, 13, 14, 15],  # UTC
                'weekend_liquidity_factor': 0.3
            },
            'EUR/USD': {
                'typical_spread_bps': 0.4,
                'market_impact_coefficient': 0.08,
                'optimal_slice_size': 15000000,
                'peak_liquidity_hours': [7, 8, 13, 14, 15],
                'weekend_liquidity_factor': 0.25
            },
            'GBP/USD': {
                'typical_spread_bps': 0.8,
                'market_impact_coefficient': 0.15,
                'optimal_slice_size': 8000000,
                'peak_liquidity_hours': [8, 9, 13, 14],
                'weekend_liquidity_factor': 0.4
            }
        }
        
        # ポートフォリオ配分モデル
        self.allocation_models = {
            'equal_weight': self._equal_weight_allocation,
            'risk_parity': self._risk_parity_allocation,
            'mean_variance': self._mean_variance_allocation,
            'momentum': self._momentum_allocation,
            'mean_reversion': self._mean_reversion_allocation
        }
        
        self.logger.info("Execution & Portfolio Management AI initialized")
    
    def _get_system_prompt(self) -> str:
        """執行・ポートフォリオ管理専用システムプロンプトを取得"""
        # 執行とポートフォリオの複合プロンプトを返す
        execution_prompt = DepartmentPrompts.get_system_prompt(DepartmentType.EXECUTION)
        portfolio_prompt = DepartmentPrompts.get_system_prompt(DepartmentType.PORTFOLIO)
        
        combined_prompt = f"""あなたは取引執行とポートフォリオ管理の両方を専門とするAIエージェントです。

## 執行管理の専門分野
{execution_prompt}

## ポートフォリオ管理の専門分野  
{portfolio_prompt}

## 統合的な分析アプローチ
両分野を統合し、執行効率性とポートフォリオ最適化の両方を考慮した総合的な推奨事項を提供してください。"""
        
        return combined_prompt
    
    def _validate_request_data(self, request: AnalysisRequest) -> bool:
        """リクエストデータの妥当性チェック"""
        try:
            data = request.data
            
            # ポートフォリオまたは価格データの存在確認
            has_portfolio = 'portfolio_state' in data
            has_price_data = 'price_data' in data
            
            if not (has_portfolio or has_price_data):
                self.logger.error("Portfolio state or price data required for execution/portfolio analysis")
                return False
            
            # 基本的なデータ構造チェック
            if has_portfolio:
                portfolio = data['portfolio_state']
                if not isinstance(portfolio, dict):
                    self.logger.error("Portfolio state must be a dictionary")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Execution/Portfolio data validation failed: {e}")
            return False
    
    async def _analyze_data(self, request: AnalysisRequest) -> Dict[str, Any]:
        """執行・ポートフォリオ分析のメインロジック"""
        try:
            data = request.data
            portfolio_state = data.get('portfolio_state', {})
            price_data = data.get('price_data', {})
            symbol = data.get('symbol', 'USD/JPY')
            
            # 執行分析（新規注文がある場合）
            execution_analysis = self._analyze_execution_strategy(
                portfolio_state, price_data, symbol
            )
            
            # 流動性・タイミング分析
            liquidity_analysis = self._analyze_market_liquidity(
                price_data, symbol
            )
            
            # ポートフォリオ健全性分析
            portfolio_health = self._analyze_portfolio_health(portfolio_state)
            
            # アセット配分分析
            allocation_analysis = self._analyze_asset_allocation(portfolio_state)
            
            # リバランシング需要分析
            rebalancing_analysis = self._analyze_rebalancing_needs(
                portfolio_state, allocation_analysis
            )
            
            # パフォーマンス・アトリビューション分析
            performance_analysis = self._analyze_portfolio_performance(portfolio_state)
            
            # ポートフォリオ最適化推奨
            optimization_recommendations = self._generate_optimization_recommendations(
                portfolio_health, allocation_analysis, rebalancing_analysis
            )
            
            # 総合判定・アクション決定
            integrated_decision = self._generate_integrated_decision(
                execution_analysis,
                liquidity_analysis,
                portfolio_health,
                rebalancing_analysis,
                optimization_recommendations
            )
            
            # 実装優先順位の決定
            implementation_priority = self._determine_implementation_priority(
                integrated_decision, rebalancing_analysis, execution_analysis
            )
            
            return {
                'portfolio_health_score': portfolio_health['health_score'],
                'current_allocation': allocation_analysis['current_allocation'],
                'performance_metrics': performance_analysis,
                'rebalancing_analysis': rebalancing_analysis,
                'optimization_recommendations': optimization_recommendations,
                'execution_strategy': execution_analysis['recommended_strategy'],
                'optimal_timing': liquidity_analysis['optimal_timing'],
                'liquidity_analysis': liquidity_analysis,
                'action': integrated_decision['action'],
                'confidence': integrated_decision['confidence'],
                'reasoning': integrated_decision['reasoning'],
                'implementation_priority': implementation_priority,
                'monitoring_metrics': self._define_monitoring_metrics(portfolio_state)
            }
            
        except Exception as e:
            self.logger.error(f"Execution/Portfolio analysis failed: {e}")
            return {
                'error': str(e),
                'action': 'maintain',
                'confidence': 0.5,
                'reasoning': f'分析エラーのため現状維持: {str(e)}'
            }
    
    def _analyze_execution_strategy(
        self,
        portfolio_state: Dict[str, Any],
        price_data: Dict[str, Any],
        symbol: str
    ) -> Dict[str, Any]:
        """執行戦略の分析"""
        try:
            analysis = {
                'recommended_strategy': 'immediate',
                'execution_plan': {},
                'cost_analysis': {},
                'risk_factors': [],
                'alternative_strategies': []
            }
            
            # 取引サイズの推定
            proposed_size = self._estimate_trade_size(portfolio_state)
            
            if proposed_size == 0:
                analysis['recommended_strategy'] = 'no_execution'
                return analysis
            
            # 市場特性の取得
            char = self.execution_characteristics.get(symbol, {})
            
            # 現在の市場状況評価
            current_spread = price_data.get('spread', char.get('typical_spread_bps', 1.0)) * 10000  # bps変換
            current_volume = price_data.get('volume', 1000000)
            
            # 流動性状況の判定
            liquidity_condition = self._assess_current_liquidity(
                current_spread, current_volume, char
            )
            
            # 執行戦略の選択
            if proposed_size < char.get('optimal_slice_size', 1000000) * 0.1:
                # 小口取引：即座執行
                strategy = 'immediate'
                execution_plan = self._plan_immediate_execution(proposed_size, current_spread)
            
            elif liquidity_condition == 'excellent' and proposed_size < char.get('optimal_slice_size', 1000000):
                # 良好な流動性：TWAP
                strategy = 'twap'
                execution_plan = self._plan_twap_execution(proposed_size, char)
            
            elif liquidity_condition in ['good', 'normal']:
                # 通常流動性：VWAP
                strategy = 'vwap' 
                execution_plan = self._plan_vwap_execution(proposed_size, char)
            
            else:
                # 流動性不足：Implementation Shortfall
                strategy = 'implementation_shortfall'
                execution_plan = self._plan_is_execution(proposed_size, char, liquidity_condition)
            
            analysis['recommended_strategy'] = strategy
            analysis['execution_plan'] = execution_plan
            analysis['cost_analysis'] = self._calculate_execution_costs(
                strategy, proposed_size, current_spread, char
            )
            analysis['risk_factors'] = self._identify_execution_risks(
                strategy, proposed_size, liquidity_condition
            )
            analysis['alternative_strategies'] = self._suggest_alternative_strategies(
                strategy, proposed_size, liquidity_condition
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Execution strategy analysis failed: {e}")
            return {
                'recommended_strategy': 'immediate',
                'execution_plan': {'error': str(e)}
            }
    
    def _estimate_trade_size(self, portfolio_state: Dict[str, Any]) -> float:
        """取引サイズの推定"""
        # 簡略化：ポートフォリオ価値の1%を想定
        total_value = portfolio_state.get('total_value', 1000000)
        if isinstance(total_value, str):
            total_value = float(total_value.replace(',', '').replace('$', ''))
        
        return total_value * 0.01
    
    def _assess_current_liquidity(
        self,
        current_spread: float,
        current_volume: float,
        char: Dict[str, Any]
    ) -> str:
        """現在の流動性状況評価"""
        typical_spread = char.get('typical_spread_bps', 1.0)
        
        # スプレッドによる流動性判定
        if current_spread <= typical_spread * 1.2:
            return 'excellent'
        elif current_spread <= typical_spread * 1.5:
            return 'good'
        elif current_spread <= typical_spread * 2.0:
            return 'normal'
        elif current_spread <= typical_spread * 3.0:
            return 'poor'
        else:
            return 'very_poor'
    
    def _plan_immediate_execution(self, size: float, spread: float) -> Dict[str, Any]:
        """即座執行プランの作成"""
        return {
            'execution_method': 'market_order',
            'total_quantity': size,
            'number_of_slices': 1,
            'execution_time_minutes': 1,
            'expected_slippage_bps': spread
        }
    
    def _plan_twap_execution(self, size: float, char: Dict[str, Any]) -> Dict[str, Any]:
        """TWAPプランの作成"""
        slices = self.execution_params['twap_time_slices']
        slice_size = size / slices
        interval_minutes = 30  # 30分間隔
        
        return {
            'execution_method': 'twap',
            'total_quantity': size,
            'number_of_slices': slices,
            'slice_size': slice_size,
            'execution_interval_minutes': interval_minutes,
            'total_execution_time_minutes': slices * interval_minutes,
            'expected_slippage_bps': char.get('typical_spread_bps', 1.0) * 1.1
        }
    
    def _plan_vwap_execution(self, size: float, char: Dict[str, Any]) -> Dict[str, Any]:
        """VWAPプランの作成"""
        participation_rate = self.execution_params['vwap_participation_rate']
        estimated_duration = 120  # 2時間
        
        return {
            'execution_method': 'vwap',
            'total_quantity': size,
            'participation_rate': participation_rate,
            'estimated_duration_minutes': estimated_duration,
            'expected_slippage_bps': char.get('typical_spread_bps', 1.0) * 1.3
        }
    
    def _plan_is_execution(
        self,
        size: float,
        char: Dict[str, Any],
        liquidity_condition: str
    ) -> Dict[str, Any]:
        """Implementation Shortfall プランの作成"""
        
        # 流動性に応じた執行期間調整
        duration_multiplier = {
            'poor': 2.0,
            'very_poor': 3.0
        }.get(liquidity_condition, 1.5)
        
        estimated_duration = 180 * duration_multiplier  # 基準3時間
        
        return {
            'execution_method': 'implementation_shortfall',
            'total_quantity': size,
            'estimated_duration_minutes': estimated_duration,
            'urgency_factor': 0.3,  # 急がない
            'expected_slippage_bps': char.get('typical_spread_bps', 1.0) * 2.0
        }
    
    def _calculate_execution_costs(
        self,
        strategy: str,
        size: float,
        spread: float,
        char: Dict[str, Any]
    ) -> Dict[str, Any]:
        """執行コストの計算"""
        try:
            # 基本スプレッドコスト
            spread_cost = size * spread / 10000  # bps -> 実額
            
            # マーケットインパクト
            impact_coeff = char.get('market_impact_coefficient', 0.1)
            market_impact = size * impact_coeff * 0.0001  # 簡易計算
            
            # 戦略別追加コスト
            if strategy == 'immediate':
                timing_cost = 0
                opportunity_cost = size * 0.0001  # 機会コスト
            elif strategy == 'twap':
                timing_cost = size * 0.0002  # タイミングリスク
                opportunity_cost = 0
            elif strategy == 'vwap':
                timing_cost = size * 0.0003
                opportunity_cost = 0
            else:  # implementation_shortfall
                timing_cost = size * 0.0005
                opportunity_cost = 0
            
            total_cost = spread_cost + market_impact + timing_cost + opportunity_cost
            
            return {
                'spread_cost': float(spread_cost),
                'market_impact': float(market_impact),
                'timing_cost': float(timing_cost),
                'opportunity_cost': float(opportunity_cost),
                'total_execution_cost': float(total_cost),
                'cost_bps': float(total_cost / size * 10000) if size > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Execution cost calculation failed: {e}")
            return {'total_execution_cost': 0, 'cost_bps': 0}
    
    def _identify_execution_risks(
        self,
        strategy: str,
        size: float,
        liquidity_condition: str
    ) -> List[str]:
        """執行リスクの特定"""
        risks = []
        
        if liquidity_condition in ['poor', 'very_poor']:
            risks.append('流動性不足による大きなマーケットインパクト')
        
        if strategy == 'immediate' and size > 1000000:
            risks.append('大口即座執行による価格インパクト')
        
        if strategy in ['twap', 'vwap'] and liquidity_condition == 'very_poor':
            risks.append('分割執行中の市場環境悪化リスク')
        
        if strategy == 'implementation_shortfall':
            risks.append('長期執行による市場リスクエクスポージャー')
        
        return risks
    
    def _suggest_alternative_strategies(
        self,
        primary_strategy: str,
        size: float,
        liquidity_condition: str
    ) -> List[str]:
        """代替戦略の提案"""
        alternatives = []
        
        if primary_strategy == 'immediate':
            alternatives.extend(['twap', 'vwap'])
        elif primary_strategy == 'twap':
            alternatives.extend(['vwap', 'implementation_shortfall'])
        elif primary_strategy == 'vwap':
            alternatives.extend(['twap', 'implementation_shortfall'])
        else:
            alternatives.extend(['twap', 'vwap'])
        
        # 流動性状況に応じた調整
        if liquidity_condition == 'very_poor':
            alternatives.append('wait_for_liquidity')
        
        return alternatives
    
    def _analyze_market_liquidity(
        self,
        price_data: Dict[str, Any],
        symbol: str
    ) -> Dict[str, Any]:
        """市場流動性の分析"""
        try:
            char = self.execution_characteristics.get(symbol, {})
            
            # 現在時刻の流動性評価
            current_hour = datetime.now().hour
            peak_hours = char.get('peak_liquidity_hours', [])
            
            is_peak_hour = current_hour in peak_hours
            is_weekend = datetime.now().weekday() >= 5
            
            # 流動性スコア計算
            base_liquidity = 1.0
            
            if is_peak_hour:
                base_liquidity *= 1.3
            else:
                base_liquidity *= 0.8
            
            if is_weekend:
                weekend_factor = char.get('weekend_liquidity_factor', 0.3)
                base_liquidity *= weekend_factor
            
            # スプレッド分析
            current_spread = price_data.get('spread', char.get('typical_spread_bps', 1.0)) * 10000
            typical_spread = char.get('typical_spread_bps', 1.0)
            
            spread_condition = self._classify_spread_condition(current_spread, typical_spread)
            
            # 最適タイミング評価
            optimal_timing = self._assess_optimal_timing(
                is_peak_hour, is_weekend, spread_condition
            )
            
            return {
                'current_liquidity': spread_condition,
                'liquidity_score': float(base_liquidity),
                'optimal_timing': optimal_timing,
                'spread_analysis': {
                    'current_spread_bps': float(current_spread),
                    'typical_spread_bps': float(typical_spread),
                    'spread_ratio': float(current_spread / typical_spread) if typical_spread > 0 else 1.0
                },
                'market_hours_analysis': {
                    'is_peak_hour': is_peak_hour,
                    'is_weekend': is_weekend,
                    'recommended_hours': peak_hours
                }
            }
            
        except Exception as e:
            self.logger.error(f"Market liquidity analysis failed: {e}")
            return {
                'current_liquidity': 'normal',
                'optimal_timing': {'immediate_execution': True}
            }
    
    def _classify_spread_condition(self, current_spread: float, typical_spread: float) -> str:
        """スプレッド状況の分類"""
        ratio = current_spread / typical_spread if typical_spread > 0 else 1.0
        
        if ratio <= 1.2:
            return 'excellent'
        elif ratio <= 1.5:
            return 'good'
        elif ratio <= 2.0:
            return 'normal'
        elif ratio <= 3.0:
            return 'poor'
        else:
            return 'very_poor'
    
    def _assess_optimal_timing(
        self,
        is_peak_hour: bool,
        is_weekend: bool,
        spread_condition: str
    ) -> Dict[str, Any]:
        """最適タイミングの評価"""
        
        immediate_execution = True
        recommended_delay = 0
        
        # 即座執行の判定
        if is_weekend and spread_condition in ['poor', 'very_poor']:
            immediate_execution = False
            recommended_delay = 60  # 週末は平日まで待機
        
        elif not is_peak_hour and spread_condition in ['poor', 'very_poor']:
            immediate_execution = False
            recommended_delay = 30  # ピーク時間まで待機
        
        # 実行時間窓の推奨
        if immediate_execution:
            execution_window = '即座実行推奨'
        elif recommended_delay > 0:
            execution_window = f'{recommended_delay}分後の実行を推奨'
        else:
            execution_window = 'ピーク時間帯での実行を推奨'
        
        return {
            'immediate_execution': immediate_execution,
            'recommended_delay_minutes': recommended_delay,
            'execution_window': execution_window,
            'timing_confidence': 0.8 if is_peak_hour and not is_weekend else 0.6
        }
    
    def _analyze_portfolio_health(self, portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """ポートフォリオ健全性の分析"""
        try:
            total_value = portfolio_state.get('total_value', 1000000)
            if isinstance(total_value, str):
                total_value = float(total_value.replace(',', '').replace('$', ''))
            
            positions = portfolio_state.get('positions', [])
            cash = portfolio_state.get('cash', total_value * 0.05)
            
            # 基本健全性指標
            health_indicators = {
                'position_count': len(positions),
                'cash_ratio': cash / total_value if total_value > 0 else 0,
                'avg_position_size': total_value / len(positions) if positions else 0,
                'portfolio_utilization': (total_value - cash) / total_value if total_value > 0 else 0
            }
            
            # リターン分析
            returns_analysis = self._analyze_returns(portfolio_state)
            
            # リスク指標
            risk_metrics = self._calculate_portfolio_risk_metrics(portfolio_state)
            
            # 健全性スコア計算
            health_score = self._calculate_health_score(
                health_indicators, returns_analysis, risk_metrics
            )
            
            return {
                'health_score': health_score,
                'health_indicators': health_indicators,
                'returns_analysis': returns_analysis,
                'risk_metrics': risk_metrics,
                'health_assessment': self._assess_health_level(health_score)
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio health analysis failed: {e}")
            return {
                'health_score': 0.6,
                'health_assessment': 'moderate'
            }
    
    def _analyze_returns(self, portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """リターン分析"""
        try:
            # 簡易的なリターン分析（実際の実装では履歴データを使用）
            total_return_ytd = portfolio_state.get('total_return_ytd', 0.05)
            total_return_1m = portfolio_state.get('total_return_1m', 0.01)
            total_return_1w = portfolio_state.get('total_return_1w', 0.002)
            
            # 年率化リターン
            annualized_return = total_return_ytd  # 既に年率と仮定
            
            # ベンチマーク対比（簡略化）
            benchmark_return = 0.04  # 4%ベンチマーク
            excess_return = annualized_return - benchmark_return
            
            return {
                'total_return_ytd': float(total_return_ytd),
                'total_return_1m': float(total_return_1m),
                'total_return_1w': float(total_return_1w),
                'annualized_return': float(annualized_return),
                'excess_return': float(excess_return),
                'information_ratio': float(excess_return / 0.05) if excess_return != 0 else 0  # 簡易IR
            }
            
        except Exception:
            return {
                'total_return_ytd': 0.0,
                'annualized_return': 0.0,
                'excess_return': 0.0
            }
    
    def _calculate_portfolio_risk_metrics(self, portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """ポートフォリオリスク指標の計算"""
        try:
            # 簡易的なリスク指標計算
            total_return_ytd = portfolio_state.get('total_return_ytd', 0.05)
            volatility = portfolio_state.get('volatility', 0.12)  # 12%ボラティリティ
            max_drawdown = portfolio_state.get('max_drawdown', 0.08)  # 8%ドローダウン
            
            # シャープレシオ
            risk_free_rate = 0.02  # 2%リスクフリーレート
            sharpe_ratio = (total_return_ytd - risk_free_rate) / volatility if volatility > 0 else 0
            
            # カルマーレシオ
            calmar_ratio = total_return_ytd / max_drawdown if max_drawdown > 0 else 0
            
            return {
                'volatility_annualized': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'calmar_ratio': float(calmar_ratio),
                'downside_deviation': float(volatility * 0.7)  # 簡易推定
            }
            
        except Exception:
            return {
                'volatility_annualized': 0.12,
                'sharpe_ratio': 0.5,
                'max_drawdown': 0.08
            }
    
    def _calculate_health_score(
        self,
        health_indicators: Dict[str, Any],
        returns_analysis: Dict[str, Any],
        risk_metrics: Dict[str, Any]
    ) -> float:
        """健全性スコアの計算"""
        try:
            score_components = []
            
            # リターン要素（40%）
            annualized_return = returns_analysis.get('annualized_return', 0)
            return_score = min(max(annualized_return / 0.15, 0), 1)  # 15%で満点
            score_components.append((return_score, 0.4))
            
            # リスク調整リターン要素（30%）
            sharpe_ratio = risk_metrics.get('sharpe_ratio', 0)
            sharpe_score = min(max(sharpe_ratio / 2.0, 0), 1)  # シャープレシオ2.0で満点
            score_components.append((sharpe_score, 0.3))
            
            # ドローダウン要素（20%）
            max_drawdown = risk_metrics.get('max_drawdown', 0)
            dd_score = max(1 - max_drawdown / 0.2, 0)  # 20%ドローダウンで0点
            score_components.append((dd_score, 0.2))
            
            # 分散化要素（10%）
            position_count = health_indicators.get('position_count', 1)
            diversification_score = min(position_count / 10, 1)  # 10ポジションで満点
            score_components.append((diversification_score, 0.1))
            
            # 重み付き平均
            weighted_score = sum(score * weight for score, weight in score_components)
            
            return float(weighted_score)
            
        except Exception:
            return 0.6
    
    def _assess_health_level(self, health_score: float) -> str:
        """健全性レベルの評価"""
        if health_score >= 0.8:
            return 'excellent'
        elif health_score >= 0.6:
            return 'good'
        elif health_score >= 0.4:
            return 'moderate'
        elif health_score >= 0.2:
            return 'poor'
        else:
            return 'critical'
    
    def _analyze_asset_allocation(self, portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """アセット配分の分析"""
        try:
            positions = portfolio_state.get('positions', [])
            total_value = portfolio_state.get('total_value', 1000000)
            if isinstance(total_value, str):
                total_value = float(total_value.replace(',', '').replace('$', ''))
            
            # 現在の配分計算
            current_allocation = {}
            geographic_allocation = {}
            
            for position in positions:
                symbol = position.get('symbol', 'USD/JPY')
                value = position.get('market_value', position.get('size', 0))
                weight = value / total_value if total_value > 0 else 0
                
                current_allocation[symbol] = weight
                
                # 地域別配分（簡略化）
                if 'USD' in symbol:
                    geographic_allocation['USD'] = geographic_allocation.get('USD', 0) + weight
                if 'EUR' in symbol:
                    geographic_allocation['EUR'] = geographic_allocation.get('EUR', 0) + weight
                if 'JPY' in symbol:
                    geographic_allocation['JPY'] = geographic_allocation.get('JPY', 0) + weight
                if 'GBP' in symbol:
                    geographic_allocation['GBP'] = geographic_allocation.get('GBP', 0) + weight
            
            # 理想的な配分との比較
            target_allocation = self._get_target_allocation(portfolio_state)
            
            # 配分乖離の計算
            allocation_drift = self._calculate_allocation_drift(
                current_allocation, target_allocation
            )
            
            return {
                'current_allocation': current_allocation,
                'geographic_allocation': geographic_allocation,
                'target_allocation': target_allocation,
                'allocation_drift': allocation_drift,
                'allocation_concentration': self._calculate_concentration_index(current_allocation)
            }
            
        except Exception as e:
            self.logger.error(f"Asset allocation analysis failed: {e}")
            return {
                'current_allocation': {},
                'allocation_drift': {}
            }
    
    def _get_target_allocation(self, portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """目標アセット配分の取得"""
        # 簡略化：均等配分を目標とする
        positions = portfolio_state.get('positions', [])
        if not positions:
            return {}
        
        target_weight = 1.0 / len(positions)
        return {pos.get('symbol', f'pos_{i}'): target_weight for i, pos in enumerate(positions)}
    
    def _calculate_allocation_drift(
        self,
        current_allocation: Dict[str, float],
        target_allocation: Dict[str, float]
    ) -> Dict[str, float]:
        """配分乖離の計算"""
        drift = {}
        
        all_assets = set(current_allocation.keys()) | set(target_allocation.keys())
        
        for asset in all_assets:
            current_weight = current_allocation.get(asset, 0)
            target_weight = target_allocation.get(asset, 0)
            drift[asset] = current_weight - target_weight
        
        return drift
    
    def _calculate_concentration_index(self, allocation: Dict[str, float]) -> float:
        """集中度指数の計算（ハーフィンダール指数）"""
        if not allocation:
            return 0.0
        
        return sum(weight**2 for weight in allocation.values())
    
    def _analyze_rebalancing_needs(
        self,
        portfolio_state: Dict[str, Any],
        allocation_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """リバランシング需要の分析"""
        try:
            allocation_drift = allocation_analysis.get('allocation_drift', {})
            
            # 最大乖離の計算
            max_drift = max((abs(drift) for drift in allocation_drift.values()), default=0)
            
            # リバランシング閾値との比較
            threshold = self.portfolio_params['rebalance_threshold_pct'] / 100
            
            rebalancing_needed = max_drift > threshold
            
            # 緊急度の評価
            if max_drift > threshold * 2:
                urgency = 'urgent'
            elif max_drift > threshold * 1.5:
                urgency = 'high'
            elif max_drift > threshold:
                urgency = 'medium'
            else:
                urgency = 'low'
            
            # 取引コスト推定
            total_value = portfolio_state.get('total_value', 1000000)
            if isinstance(total_value, str):
                total_value = float(total_value.replace(',', '').replace('$', ''))
            
            estimated_turnover = sum(abs(drift) for drift in allocation_drift.values()) / 2
            transaction_cost = estimated_turnover * total_value * (self.portfolio_params['transaction_cost_bps'] / 10000)
            
            # リバランシング効果の推定
            expected_benefit = self._estimate_rebalancing_benefit(
                allocation_drift, portfolio_state
            )
            
            return {
                'rebalancing_needed': rebalancing_needed,
                'urgency_level': urgency,
                'max_deviation_pct': float(max_drift * 100),
                'threshold_pct': float(threshold * 100),
                'transaction_cost_estimate': float(transaction_cost),
                'expected_benefit': expected_benefit,
                'recommended_trades': self._generate_rebalancing_trades(
                    allocation_drift, total_value
                )
            }
            
        except Exception as e:
            self.logger.error(f"Rebalancing analysis failed: {e}")
            return {
                'rebalancing_needed': False,
                'urgency_level': 'low'
            }
    
    def _estimate_rebalancing_benefit(
        self,
        allocation_drift: Dict[str, float],
        portfolio_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """リバランシング効果の推定"""
        try:
            # 簡易的な効果推定
            total_drift = sum(abs(drift) for drift in allocation_drift.values())
            
            # リスク削減効果
            risk_reduction = total_drift * 0.1  # 仮定：乖離1%につき10%のリスク削減
            
            # リターン改善効果
            return_improvement = total_drift * 0.05  # 仮定：乖離1%につき5%のリターン改善
            
            return {
                'risk_reduction_pct': float(risk_reduction * 100),
                'return_improvement_pct': float(return_improvement * 100),
                'sharpe_improvement': float(risk_reduction + return_improvement)
            }
            
        except Exception:
            return {
                'risk_reduction_pct': 0.0,
                'return_improvement_pct': 0.0
            }
    
    def _generate_rebalancing_trades(
        self,
        allocation_drift: Dict[str, float],
        total_value: float
    ) -> List[Dict[str, Any]]:
        """リバランシングトレードの生成"""
        trades = []
        
        try:
            for asset, drift in allocation_drift.items():
                if abs(drift) > 0.01:  # 1%以上の乖離のみ対象
                    trade_amount = drift * total_value
                    
                    trades.append({
                        'asset': asset,
                        'action': 'sell' if drift > 0 else 'buy',
                        'amount': abs(trade_amount),
                        'priority': 'high' if abs(drift) > 0.05 else 'medium'
                    })
            
            # 優先度順にソート
            trades.sort(key=lambda x: abs(allocation_drift[x['asset']]), reverse=True)
            
            return trades[:10]  # 最大10トレード
            
        except Exception:
            return []
    
    def _analyze_portfolio_performance(self, portfolio_state: Dict[str, Any]) -> Dict[str, Any]:
        """ポートフォリオパフォーマンス分析"""
        try:
            # 基本パフォーマンス指標
            total_return_ytd = portfolio_state.get('total_return_ytd', 0.05)
            volatility = portfolio_state.get('volatility', 0.12)
            max_drawdown = portfolio_state.get('max_drawdown', 0.08)
            
            # リスクフリーレート
            risk_free_rate = 0.02
            
            # パフォーマンス指標計算
            sharpe_ratio = (total_return_ytd - risk_free_rate) / volatility if volatility > 0 else 0
            calmar_ratio = total_return_ytd / max_drawdown if max_drawdown > 0 else 0
            
            # 勝率・負率（簡略化）
            win_rate = 0.6  # 仮定
            avg_win = 0.03
            avg_loss = -0.02
            
            return {
                'total_return_ytd': float(total_return_ytd),
                'volatility_annualized': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'calmar_ratio': float(calmar_ratio),
                'win_rate': float(win_rate),
                'profit_factor': float(abs(avg_win * win_rate / (avg_loss * (1 - win_rate)))) if avg_loss != 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio performance analysis failed: {e}")
            return {
                'total_return_ytd': 0.0,
                'sharpe_ratio': 0.0
            }
    
    def _generate_optimization_recommendations(
        self,
        portfolio_health: Dict[str, Any],
        allocation_analysis: Dict[str, Any],
        rebalancing_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """ポートフォリオ最適化推奨の生成"""
        try:
            recommendations = {
                'target_allocation': {},
                'expected_improvement': {},
                'optimization_method': 'mean_variance',
                'constraints': {}
            }
            
            # 健全性スコアに基づく最適化方針
            health_score = portfolio_health.get('health_score', 0.5)
            
            if health_score < 0.4:
                # 低健全性：リスク重視の最適化
                recommendations['optimization_method'] = 'risk_parity'
                recommendations['target_allocation'] = self._risk_parity_allocation(allocation_analysis)
            elif health_score > 0.8:
                # 高健全性：リターン重視の最適化
                recommendations['optimization_method'] = 'momentum'
                recommendations['target_allocation'] = self._momentum_allocation(allocation_analysis)
            else:
                # 中間：バランス重視
                recommendations['optimization_method'] = 'mean_variance'
                recommendations['target_allocation'] = self._mean_variance_allocation(allocation_analysis)
            
            # 期待改善効果
            recommendations['expected_improvement'] = self._calculate_optimization_benefit(
                allocation_analysis, recommendations['target_allocation']
            )
            
            # 制約条件
            recommendations['constraints'] = {
                'min_position_pct': self.portfolio_params['min_position_pct'],
                'max_position_pct': self.portfolio_params['max_position_pct'],
                'max_turnover_pct': 20.0,
                'liquidity_requirement': 'high'
            }
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Optimization recommendations generation failed: {e}")
            return {
                'optimization_method': 'equal_weight',
                'target_allocation': {}
            }
    
    def _risk_parity_allocation(self, allocation_analysis: Dict[str, Any]) -> Dict[str, float]:
        """リスクパリティ配分の計算"""
        current_allocation = allocation_analysis.get('current_allocation', {})
        if not current_allocation:
            return {}
        
        # 簡略化：等リスク寄与配分（均等配分に近似）
        n_assets = len(current_allocation)
        target_weight = 1.0 / n_assets if n_assets > 0 else 0
        
        return {asset: target_weight for asset in current_allocation.keys()}
    
    def _momentum_allocation(self, allocation_analysis: Dict[str, Any]) -> Dict[str, float]:
        """モメンタム配分の計算"""
        current_allocation = allocation_analysis.get('current_allocation', {})
        if not current_allocation:
            return {}
        
        # 簡略化：現在の配分を基に強化
        enhanced_allocation = {}
        total_weight = 0
        
        for asset, weight in current_allocation.items():
            # 上位資産に重みを追加
            if weight > 1.0 / len(current_allocation):
                enhanced_weight = weight * 1.2
            else:
                enhanced_weight = weight * 0.8
            
            enhanced_allocation[asset] = enhanced_weight
            total_weight += enhanced_weight
        
        # 正規化
        if total_weight > 0:
            enhanced_allocation = {asset: weight / total_weight for asset, weight in enhanced_allocation.items()}
        
        return enhanced_allocation
    
    def _mean_variance_allocation(self, allocation_analysis: Dict[str, Any]) -> Dict[str, float]:
        """平均分散最適化配分の計算"""
        current_allocation = allocation_analysis.get('current_allocation', {})
        if not current_allocation:
            return {}
        
        # 簡略化：わずかに調整した均等配分
        n_assets = len(current_allocation)
        base_weight = 1.0 / n_assets if n_assets > 0 else 0
        
        # 小さなランダム調整を加える（実際の実装では最適化アルゴリズムを使用）
        target_allocation = {}
        total_adjustment = 0
        
        for i, asset in enumerate(current_allocation.keys()):
            adjustment = 0.02 * (i % 3 - 1)  # -0.02, 0, 0.02の調整
            adjusted_weight = base_weight + adjustment
            target_allocation[asset] = max(0.01, adjusted_weight)  # 最小1%
            total_adjustment += adjusted_weight
        
        # 正規化
        if total_adjustment > 0:
            target_allocation = {asset: weight / total_adjustment for asset, weight in target_allocation.items()}
        
        return target_allocation
    
    def _equal_weight_allocation(self, allocation_analysis: Dict[str, Any]) -> Dict[str, float]:
        """均等配分の計算"""
        current_allocation = allocation_analysis.get('current_allocation', {})
        if not current_allocation:
            return {}
        
        n_assets = len(current_allocation)
        target_weight = 1.0 / n_assets if n_assets > 0 else 0
        
        return {asset: target_weight for asset in current_allocation.keys()}
    
    def _mean_reversion_allocation(self, allocation_analysis: Dict[str, Any]) -> Dict[str, float]:
        """平均回帰配分の計算"""
        current_allocation = allocation_analysis.get('current_allocation', {})
        if not current_allocation:
            return {}
        
        # 簡略化：パフォーマンスの低い資産により多く配分
        reversion_allocation = {}
        total_weight = 0
        
        for asset, weight in current_allocation.items():
            # パフォーマンスが低い（配分が少ない）資産を強化
            if weight < 1.0 / len(current_allocation):
                reversion_weight = weight * 1.3
            else:
                reversion_weight = weight * 0.7
            
            reversion_allocation[asset] = reversion_weight
            total_weight += reversion_weight
        
        # 正規化
        if total_weight > 0:
            reversion_allocation = {asset: weight / total_weight for asset, weight in reversion_allocation.items()}
        
        return reversion_allocation
    
    def _calculate_optimization_benefit(
        self,
        current_analysis: Dict[str, Any],
        target_allocation: Dict[str, float]
    ) -> Dict[str, Any]:
        """最適化効果の計算"""
        try:
            current_allocation = current_analysis.get('current_allocation', {})
            
            # 配分変更量の計算
            total_change = sum(
                abs(target_allocation.get(asset, 0) - current_allocation.get(asset, 0))
                for asset in set(current_allocation.keys()) | set(target_allocation.keys())
            )
            
            # 簡易的な効果推定
            return_uplift = total_change * 0.02  # 配分変更1%につき2%のリターン改善
            risk_reduction = total_change * 0.01  # 配分変更1%につき1%のリスク削減
            sharpe_improvement = return_uplift + risk_reduction
            
            return {
                'return_uplift': float(return_uplift),
                'risk_reduction': float(risk_reduction),
                'sharpe_improvement': float(sharpe_improvement),
                'total_turnover': float(total_change)
            }
            
        except Exception:
            return {
                'return_uplift': 0.0,
                'risk_reduction': 0.0,
                'sharpe_improvement': 0.0
            }
    
    def _generate_integrated_decision(
        self,
        execution_analysis: Dict[str, Any],
        liquidity_analysis: Dict[str, Any],
        portfolio_health: Dict[str, Any],
        rebalancing_analysis: Dict[str, Any],
        optimization_recommendations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """統合的な意思決定の生成"""
        try:
            # 各分析からの優先度評価
            execution_urgency = self._assess_execution_urgency(execution_analysis, liquidity_analysis)
            portfolio_urgency = self._assess_portfolio_urgency(portfolio_health, rebalancing_analysis)
            
            # 総合アクションの決定
            if portfolio_urgency == 'critical':
                action = 'emergency_rebalance'
                confidence = 0.9
                reasoning = 'ポートフォリオ健全性が危険水準のため緊急リバランシング'
            elif rebalancing_analysis.get('urgency_level') == 'urgent':
                action = 'rebalance'
                confidence = 0.8
                reasoning = 'アセット配分の大幅な乖離によりリバランシングが必要'
            elif execution_urgency == 'high':
                action = 'optimize_execution'
                confidence = 0.7
                reasoning = '流動性状況を考慮した最適執行の実施'
            elif portfolio_health.get('health_score', 0.5) < 0.5:
                action = 'improve_portfolio'
                confidence = 0.6
                reasoning = 'ポートフォリオ健全性改善のための最適化'
            else:
                action = 'maintain'
                confidence = 0.7
                reasoning = '現在の状況は良好のため現状維持'
            
            # 追加的な考慮事項
            key_factors = []
            
            if rebalancing_analysis.get('rebalancing_needed', False):
                key_factors.append(f"配分乖離: {rebalancing_analysis.get('max_deviation_pct', 0):.1f}%")
            
            if portfolio_health.get('health_score', 0.5) < 0.6:
                key_factors.append(f"健全性スコア: {portfolio_health.get('health_score', 0.5):.2f}")
            
            liquidity_condition = liquidity_analysis.get('current_liquidity', 'normal')
            if liquidity_condition in ['poor', 'very_poor']:
                key_factors.append(f"流動性状況: {liquidity_condition}")
            
            return {
                'action': action,
                'confidence': float(confidence),
                'reasoning': reasoning,
                'key_factors': key_factors,
                'execution_priority': execution_urgency,
                'portfolio_priority': portfolio_urgency
            }
            
        except Exception as e:
            self.logger.error(f"Integrated decision generation failed: {e}")
            return {
                'action': 'maintain',
                'confidence': 0.5,
                'reasoning': f'判定エラーのため現状維持: {str(e)}',
                'key_factors': []
            }
    
    def _assess_execution_urgency(
        self,
        execution_analysis: Dict[str, Any],
        liquidity_analysis: Dict[str, Any]
    ) -> str:
        """執行緊急度の評価"""
        strategy = execution_analysis.get('recommended_strategy', 'immediate')
        liquidity = liquidity_analysis.get('current_liquidity', 'normal')
        
        if strategy == 'immediate' and liquidity == 'very_poor':
            return 'high'
        elif strategy in ['implementation_shortfall'] and liquidity in ['poor', 'very_poor']:
            return 'medium'
        else:
            return 'low'
    
    def _assess_portfolio_urgency(
        self,
        portfolio_health: Dict[str, Any],
        rebalancing_analysis: Dict[str, Any]
    ) -> str:
        """ポートフォリオ緊急度の評価"""
        health_score = portfolio_health.get('health_score', 0.5)
        urgency_level = rebalancing_analysis.get('urgency_level', 'low')
        
        if health_score < 0.3 or urgency_level == 'urgent':
            return 'critical'
        elif health_score < 0.5 or urgency_level == 'high':
            return 'high'
        elif urgency_level == 'medium':
            return 'medium'
        else:
            return 'low'
    
    def _determine_implementation_priority(
        self,
        integrated_decision: Dict[str, Any],
        rebalancing_analysis: Dict[str, Any],
        execution_analysis: Dict[str, Any]
    ) -> List[str]:
        """実装優先順位の決定"""
        priority_list = []
        
        try:
            action = integrated_decision.get('action', 'maintain')
            
            if action == 'emergency_rebalance':
                priority_list.extend([
                    'ポートフォリオリスクの緊急評価',
                    '高リスクポジションの即座縮小',
                    '安全資産への一時避難',
                    'リバランシング実行'
                ])
            
            elif action == 'rebalance':
                recommended_trades = rebalancing_analysis.get('recommended_trades', [])
                high_priority_trades = [t for t in recommended_trades if t.get('priority') == 'high']
                
                if high_priority_trades:
                    priority_list.append(f'高優先度トレード実行（{len(high_priority_trades)}件）')
                
                priority_list.extend([
                    'トランザクションコスト最適化',
                    '段階的リバランシング実行',
                    'パフォーマンス監視強化'
                ])
            
            elif action == 'optimize_execution':
                strategy = execution_analysis.get('recommended_strategy', 'immediate')
                priority_list.extend([
                    f'{strategy}戦略による最適執行',
                    '市場インパクト最小化',
                    '執行コスト監視'
                ])
            
            elif action == 'improve_portfolio':
                priority_list.extend([
                    'ポートフォリオ分析の深化',
                    '最適化アルゴリズムの適用',
                    'リスク指標の改善',
                    'パフォーマンス向上施策'
                ])
            
            else:  # maintain
                priority_list.extend([
                    '定期的なポートフォリオ監視',
                    'リバランシング閾値の確認',
                    'パフォーマンス追跡'
                ])
            
            return priority_list[:5]  # 最大5つの優先事項
            
        except Exception as e:
            self.logger.error(f"Implementation priority determination failed: {e}")
            return ['ポートフォリオ状況の再確認']
    
    def _define_monitoring_metrics(self, portfolio_state: Dict[str, Any]) -> List[str]:
        """監視すべき指標の定義"""
        metrics = [
            'ポートフォリオ総価値',
            'アセット配分乖離度',
            'シャープレシオ',
            '最大ドローダウン',
            'VaR（99%信頼区間）'
        ]
        
        # ポートフォリオ特性に応じた追加メトリクス
        positions = portfolio_state.get('positions', [])
        if len(positions) > 5:
            metrics.append('分散化効果')
        
        if any('USD' in pos.get('symbol', '') for pos in positions):
            metrics.append('USD エクスポージャー')
        
        return metrics