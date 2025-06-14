#!/usr/bin/env python3
"""
Risk Management AI Department
リスク管理部門AI
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
from scipy import stats

# Add core modules to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ai_agent_base import AIAgentBase, DepartmentType, AnalysisRequest, AnalysisResult
from core.department_prompts import DepartmentPrompts, PromptFormatter


class RiskManagementAI(AIAgentBase):
    """
    リスク管理専門AIエージェント
    
    VaR計算、ドローダウン分析、ポジションサイジング、相関リスク管理を専門とする
    """
    
    def __init__(self, model_config: Dict[str, Any] = None):
        super().__init__(DepartmentType.RISK, model_config)
        
        # リスク管理の基本パラメータ
        self.risk_limits = {
            'max_portfolio_var_pct': 3.0,      # 最大ポートフォリオVaR（日次）
            'max_single_position_pct': 2.0,    # 単一ポジション最大リスク
            'max_drawdown_pct': 8.0,           # 最大許容ドローダウン
            'max_correlation_exposure': 0.7,   # 最大相関エクスポージャー
            'min_diversification_score': 0.6  # 最小分散化スコア
        }
        
        # VaR計算パラメータ
        self.var_params = {
            'confidence_level': 0.99,          # 99%信頼区間
            'lookback_days': 252,              # 1年間のデータ
            'decay_factor': 0.94,              # 指数重み付け減衰率
            'min_observations': 30             # 最小観測数
        }
        
        # 通貨ペア別の特性
        self.currency_characteristics = {
            'USD/JPY': {
                'typical_volatility': 0.008,    # 日次ボラティリティ
                'liquidity_score': 1.0,         # 流動性スコア
                'correlation_group': 'majors',
                'safe_haven': True
            },
            'EUR/USD': {
                'typical_volatility': 0.007,
                'liquidity_score': 1.0,
                'correlation_group': 'majors',
                'safe_haven': False
            },
            'GBP/USD': {
                'typical_volatility': 0.009,
                'liquidity_score': 0.9,
                'correlation_group': 'majors',
                'safe_haven': False
            },
            'AUD/USD': {
                'typical_volatility': 0.010,
                'liquidity_score': 0.8,
                'correlation_group': 'commodities',
                'safe_haven': False
            }
        }
        
        # リスクイベントパターン
        self.risk_event_patterns = {
            'volatility_spike': {
                'threshold': 2.0,  # 通常ボラティリティの2倍
                'warning_level': 1.5
            },
            'correlation_breakdown': {
                'threshold': 0.8,  # 相関係数の急変
                'warning_level': 0.6
            },
            'liquidity_crisis': {
                'spread_threshold': 3.0,  # スプレッド拡大倍率
                'volume_threshold': 0.5   # 出来高減少
            }
        }
        
        self.logger.info("Risk Management AI initialized")
    
    def _get_system_prompt(self) -> str:
        """リスク管理専用システムプロンプトを取得"""
        return DepartmentPrompts.get_system_prompt(DepartmentType.RISK)
    
    def _validate_request_data(self, request: AnalysisRequest) -> bool:
        """リクエストデータの妥当性チェック"""
        try:
            data = request.data
            
            # ポートフォリオまたはリスク指標の存在確認
            has_portfolio = 'portfolio_state' in data
            has_risk_metrics = 'risk_metrics' in data
            has_price_data = 'price_data' in data
            
            if not (has_portfolio or has_risk_metrics or has_price_data):
                self.logger.error("No portfolio, risk metrics, or price data for risk analysis")
                return False
            
            # ポートフォリオデータの基本構造チェック
            if has_portfolio:
                portfolio = data['portfolio_state']
                if not isinstance(portfolio, dict):
                    self.logger.error("Portfolio state must be a dictionary")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Risk data validation failed: {e}")
            return False
    
    async def _analyze_data(self, request: AnalysisRequest) -> Dict[str, Any]:
        """リスク分析のメインロジック"""
        try:
            data = request.data
            portfolio_state = data.get('portfolio_state', {})
            risk_metrics = data.get('risk_metrics', {})
            price_data = data.get('price_data', {})
            
            # VaR計算・分析
            var_analysis = self._calculate_var_analysis(
                portfolio_state, price_data, risk_metrics
            )
            
            # ドローダウン分析
            drawdown_analysis = self._analyze_drawdown_risk(
                portfolio_state, risk_metrics
            )
            
            # ポジション集中度分析
            position_analysis = self._analyze_position_concentration(
                portfolio_state
            )
            
            # 市場リスクファクター分析
            market_risk_analysis = self._analyze_market_risk_factors(
                price_data, risk_metrics
            )
            
            # 流動性リスク評価
            liquidity_analysis = self._assess_liquidity_risk(
                portfolio_state, price_data
            )
            
            # 相関・クラスターリスク分析
            correlation_analysis = self._analyze_correlation_risk(
                portfolio_state, price_data
            )
            
            # ストレステスト実行
            stress_test_results = self._run_stress_tests(
                portfolio_state, var_analysis
            )
            
            # 総合リスクスコア計算
            overall_risk_assessment = self._calculate_overall_risk_score(
                var_analysis,
                drawdown_analysis,
                position_analysis,
                market_risk_analysis,
                liquidity_analysis,
                correlation_analysis
            )
            
            # リスク推奨事項生成
            recommendations = self._generate_risk_recommendations(
                overall_risk_assessment,
                var_analysis,
                drawdown_analysis,
                position_analysis,
                stress_test_results
            )
            
            # 緊急対応アクション
            immediate_actions = self._assess_immediate_actions(
                overall_risk_assessment,
                stress_test_results
            )
            
            return {
                'risk_score': overall_risk_assessment['risk_score'],
                'risk_level': overall_risk_assessment['risk_level'],
                'var_analysis': var_analysis,
                'drawdown_analysis': drawdown_analysis,
                'position_analysis': position_analysis,
                'market_risk_factors': market_risk_analysis,
                'recommendations': recommendations,
                'position_sizing': self._calculate_position_sizing_guidance(
                    overall_risk_assessment, portfolio_state
                ),
                'action': overall_risk_assessment['recommended_action'],
                'confidence': overall_risk_assessment['confidence'],
                'reasoning': overall_risk_assessment['reasoning'],
                'immediate_actions': immediate_actions,
                'stress_test_results': stress_test_results
            }
            
        except Exception as e:
            self.logger.error(f"Risk analysis failed: {e}")
            return {
                'error': str(e),
                'action': 'reduce_risk',
                'confidence': 0.8,
                'reasoning': f'リスク分析エラーのため保守的対応: {str(e)}'
            }
    
    def _calculate_var_analysis(
        self,
        portfolio_state: Dict[str, Any],
        price_data: Dict[str, Any],
        risk_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """VaR計算・分析"""
        try:
            analysis = {
                '1_day_var_99': 0.0,
                '1_week_var_99': 0.0,
                '1_month_var_99': 0.0,
                'var_utilization': 0.0,
                'var_method': 'parametric',
                'confidence_level': self.var_params['confidence_level']
            }
            
            # ポートフォリオ価値取得
            total_value = portfolio_state.get('total_value', 1000000)
            if isinstance(total_value, str):
                total_value = float(total_value.replace(',', '').replace('$', ''))
            
            # 既存のリスク指標から推定
            current_var = risk_metrics.get('current_var', 0)
            if current_var:
                analysis['1_day_var_99'] = current_var
            else:
                # ポートフォリオボラティリティから推定
                portfolio_volatility = self._estimate_portfolio_volatility(
                    portfolio_state, price_data
                )
                # 99%VaR ≈ 2.33 * σ * √t (正規分布仮定)
                z_score = stats.norm.ppf(self.var_params['confidence_level'])
                analysis['1_day_var_99'] = total_value * portfolio_volatility * z_score
            
            # 1週間・1ヶ月VaRの計算（時間調整）
            analysis['1_week_var_99'] = analysis['1_day_var_99'] * math.sqrt(7)
            analysis['1_month_var_99'] = analysis['1_day_var_99'] * math.sqrt(30)
            
            # VaR利用率（限度に対する使用率）
            max_var_limit = total_value * (self.risk_limits['max_portfolio_var_pct'] / 100)
            if max_var_limit > 0:
                analysis['var_utilization'] = analysis['1_day_var_99'] / max_var_limit
            
            # VaR検証（バックテスト）
            analysis['var_validation'] = self._validate_var_model(
                analysis['1_day_var_99'], portfolio_state
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"VaR calculation failed: {e}")
            return {
                '1_day_var_99': 0.0,
                'var_utilization': 0.0,
                'var_method': 'fallback'
            }
    
    def _estimate_portfolio_volatility(
        self,
        portfolio_state: Dict[str, Any],
        price_data: Dict[str, Any]
    ) -> float:
        """ポートフォリオボラティリティの推定"""
        try:
            # 既存のボラティリティ指標
            if 'volatility' in portfolio_state:
                return float(portfolio_state['volatility'])
            
            # 価格データからボラティリティ推定
            if 'close' in price_data and 'historical_prices' in price_data:
                prices = price_data['historical_prices']
                if len(prices) > 1:
                    returns = np.diff(np.log(prices))
                    return np.std(returns) * math.sqrt(252)  # 年率化
            
            # フォールバック：デフォルトボラティリティ
            return 0.15  # 15% 年率ボラティリティ
            
        except Exception:
            return 0.15
    
    def _validate_var_model(
        self,
        predicted_var: float,
        portfolio_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """VaRモデルの検証"""
        try:
            # 簡易的なバックテスト結果
            # 実際の実装では過去のP&L履歴と比較
            validation = {
                'violation_rate': 0.01,  # 実際の損失がVaRを超えた割合
                'expected_rate': 0.01,   # 期待される違反率
                'model_accuracy': 'good'
            }
            
            # モデル精度判定
            if validation['violation_rate'] > validation['expected_rate'] * 1.5:
                validation['model_accuracy'] = 'poor'
            elif validation['violation_rate'] < validation['expected_rate'] * 0.5:
                validation['model_accuracy'] = 'conservative'
            
            return validation
            
        except Exception:
            return {'model_accuracy': 'unknown'}
    
    def _analyze_drawdown_risk(
        self,
        portfolio_state: Dict[str, Any],
        risk_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """ドローダウンリスク分析"""
        try:
            total_value = portfolio_state.get('total_value', 1000000)
            if isinstance(total_value, str):
                total_value = float(total_value.replace(',', '').replace('$', ''))
            
            # 現在のドローダウン
            peak_value = portfolio_state.get('peak_value', total_value)
            if isinstance(peak_value, str):
                peak_value = float(peak_value.replace(',', '').replace('$', ''))
            
            current_drawdown_pct = ((peak_value - total_value) / peak_value * 100) if peak_value > 0 else 0
            
            # 最大許容ドローダウン
            max_acceptable_dd = self.risk_limits['max_drawdown_pct']
            
            # ドローダウンリスクレベル
            dd_risk_level = 'low'
            if current_drawdown_pct > max_acceptable_dd * 0.8:
                dd_risk_level = 'high'
            elif current_drawdown_pct > max_acceptable_dd * 0.5:
                dd_risk_level = 'medium'
            
            # ドローダウン回復期間の推定
            recovery_estimate = self._estimate_recovery_time(
                current_drawdown_pct, portfolio_state
            )
            
            return {
                'current_drawdown_pct': float(current_drawdown_pct),
                'max_acceptable_dd': float(max_acceptable_dd),
                'dd_risk_level': dd_risk_level,
                'drawdown_utilization': current_drawdown_pct / max_acceptable_dd if max_acceptable_dd > 0 else 0,
                'recovery_time_estimate': recovery_estimate,
                'underwater_period': self._calculate_underwater_period(portfolio_state)
            }
            
        except Exception as e:
            self.logger.error(f"Drawdown analysis failed: {e}")
            return {
                'current_drawdown_pct': 0.0,
                'dd_risk_level': 'medium'
            }
    
    def _estimate_recovery_time(
        self,
        current_drawdown_pct: float,
        portfolio_state: Dict[str, Any]
    ) -> int:
        """ドローダウン回復期間の推定"""
        try:
            if current_drawdown_pct <= 0:
                return 0
            
            # 過去のリターン実績から推定
            monthly_return = portfolio_state.get('avg_monthly_return', 0.01)  # 1%デフォルト
            
            if monthly_return <= 0:
                return 999  # 回復不可能
            
            # 複利計算での回復期間
            recovery_months = math.log(1 / (1 - current_drawdown_pct / 100)) / math.log(1 + monthly_return)
            return int(recovery_months)
            
        except Exception:
            return 12  # デフォルト12ヶ月
    
    def _calculate_underwater_period(self, portfolio_state: Dict[str, Any]) -> int:
        """現在の水面下期間（高値更新していない期間）"""
        try:
            # 最後の高値更新からの日数
            last_high_date = portfolio_state.get('last_high_date')
            if last_high_date:
                # 簡易計算（実装では実際の日付計算）
                return portfolio_state.get('days_since_high', 0)
            return 0
        except Exception:
            return 0
    
    def _analyze_position_concentration(
        self,
        portfolio_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """ポジション集中度分析"""
        try:
            positions = portfolio_state.get('positions', [])
            total_value = portfolio_state.get('total_value', 1000000)
            if isinstance(total_value, str):
                total_value = float(total_value.replace(',', '').replace('$', ''))
            
            if not positions:
                return {
                    'portfolio_concentration': 0.0,
                    'largest_position_risk': 0.0,
                    'correlation_cluster_risk': 0.0,
                    'diversification_score': 1.0
                }
            
            # ポジションサイズの分析
            position_sizes = []
            position_risks = []
            
            for position in positions:
                size = position.get('size', 0)
                value = position.get('market_value', size)
                position_sizes.append(value)
                
                # リスク寄与度（サイズ × ボラティリティ）
                symbol = position.get('symbol', 'USD/JPY')
                volatility = self._get_symbol_volatility(symbol)
                risk_contrib = value * volatility
                position_risks.append(risk_contrib)
            
            # 集中度指標
            if position_sizes:
                # ハーフィンダール指数
                total_size = sum(position_sizes)
                weights = [size / total_size for size in position_sizes] if total_size > 0 else []
                herfindahl_index = sum(w**2 for w in weights)
                
                # 最大ポジションリスク
                largest_position_risk = max(position_sizes) / total_value * 100 if total_value > 0 else 0
                
                # 相関クラスターリスク
                cluster_risk = self._calculate_correlation_cluster_risk(positions)
                
                # 分散化スコア（1 - ハーフィンダール指数）
                diversification_score = 1 - herfindahl_index
                
                return {
                    'portfolio_concentration': float(herfindahl_index),
                    'largest_position_risk': float(largest_position_risk),
                    'correlation_cluster_risk': float(cluster_risk),
                    'diversification_score': float(diversification_score),
                    'position_count': len(positions),
                    'effective_positions': 1 / herfindahl_index if herfindahl_index > 0 else len(positions)
                }
            
            return {
                'portfolio_concentration': 0.0,
                'largest_position_risk': 0.0,
                'correlation_cluster_risk': 0.0,
                'diversification_score': 1.0
            }
            
        except Exception as e:
            self.logger.error(f"Position concentration analysis failed: {e}")
            return {
                'portfolio_concentration': 0.5,
                'largest_position_risk': 0.0
            }
    
    def _get_symbol_volatility(self, symbol: str) -> float:
        """通貨ペア別ボラティリティ取得"""
        char = self.currency_characteristics.get(symbol, {})
        return char.get('typical_volatility', 0.01)
    
    def _calculate_correlation_cluster_risk(self, positions: List[Dict[str, Any]]) -> float:
        """相関クラスターリスク計算"""
        try:
            if len(positions) <= 1:
                return 0.0
            
            # 通貨ペアのグループ分類
            group_exposure = {}
            total_exposure = 0
            
            for position in positions:
                symbol = position.get('symbol', 'USD/JPY')
                size = position.get('market_value', position.get('size', 0))
                
                char = self.currency_characteristics.get(symbol, {})
                group = char.get('correlation_group', 'other')
                
                group_exposure[group] = group_exposure.get(group, 0) + size
                total_exposure += size
            
            # 最大グループ集中度
            if total_exposure > 0:
                max_group_weight = max(group_exposure.values()) / total_exposure
                return max_group_weight
            
            return 0.0
            
        except Exception:
            return 0.5
    
    def _analyze_market_risk_factors(
        self,
        price_data: Dict[str, Any],
        risk_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """市場リスクファクター分析"""
        try:
            analysis = {
                'volatility_regime': 'medium',
                'liquidity_conditions': 'normal',
                'tail_risk_level': 'medium',
                'market_stress_indicator': 0.5
            }
            
            # ボラティリティレジーム分析
            current_volatility = risk_metrics.get('current_volatility', 0.01)
            historical_volatility = risk_metrics.get('historical_volatility', 0.01)
            
            vol_ratio = current_volatility / historical_volatility if historical_volatility > 0 else 1.0
            
            if vol_ratio > 1.5:
                analysis['volatility_regime'] = 'high'
            elif vol_ratio < 0.7:
                analysis['volatility_regime'] = 'low'
            
            # 流動性状況
            current_spread = price_data.get('spread', 0.0002)
            normal_spread = 0.0002  # 通常スプレッド
            
            spread_ratio = current_spread / normal_spread if normal_spread > 0 else 1.0
            
            if spread_ratio > 2.0:
                analysis['liquidity_conditions'] = 'poor'
            elif spread_ratio < 0.8:
                analysis['liquidity_conditions'] = 'good'
            
            # テールリスク評価
            skewness = risk_metrics.get('skewness', 0)
            kurtosis = risk_metrics.get('kurtosis', 3)
            
            # 歪度・尖度からテールリスク判定
            if abs(skewness) > 1.0 or kurtosis > 5:
                analysis['tail_risk_level'] = 'high'
            elif abs(skewness) < 0.5 and kurtosis < 4:
                analysis['tail_risk_level'] = 'low'
            
            # 市場ストレス指標（複合指標）
            stress_score = (vol_ratio - 1) * 0.4 + (spread_ratio - 1) * 0.3 + abs(skewness) * 0.3
            analysis['market_stress_indicator'] = max(0, min(1, stress_score))
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Market risk factor analysis failed: {e}")
            return {
                'volatility_regime': 'medium',
                'liquidity_conditions': 'normal',
                'tail_risk_level': 'medium'
            }
    
    def _assess_liquidity_risk(
        self,
        portfolio_state: Dict[str, Any],
        price_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """流動性リスク評価"""
        try:
            positions = portfolio_state.get('positions', [])
            
            if not positions:
                return {
                    'overall_liquidity_score': 1.0,
                    'liquidity_risk_level': 'low',
                    'illiquid_position_pct': 0.0
                }
            
            liquidity_scores = []
            illiquid_value = 0
            total_value = 0
            
            for position in positions:
                symbol = position.get('symbol', 'USD/JPY')
                value = position.get('market_value', position.get('size', 0))
                total_value += value
                
                # 通貨ペア別流動性スコア
                char = self.currency_characteristics.get(symbol, {})
                liquidity_score = char.get('liquidity_score', 0.5)
                liquidity_scores.append(liquidity_score)
                
                # 流動性が低いポジション
                if liquidity_score < 0.7:
                    illiquid_value += value
            
            # 全体的な流動性スコア
            overall_score = sum(liquidity_scores) / len(liquidity_scores) if liquidity_scores else 1.0
            
            # 非流動ポジション比率
            illiquid_pct = illiquid_value / total_value * 100 if total_value > 0 else 0
            
            # リスクレベル判定
            if overall_score > 0.8 and illiquid_pct < 10:
                risk_level = 'low'
            elif overall_score > 0.6 and illiquid_pct < 25:
                risk_level = 'medium'
            else:
                risk_level = 'high'
            
            return {
                'overall_liquidity_score': float(overall_score),
                'liquidity_risk_level': risk_level,
                'illiquid_position_pct': float(illiquid_pct),
                'liquidity_breakdown': self._analyze_liquidity_breakdown(positions)
            }
            
        except Exception as e:
            self.logger.error(f"Liquidity risk assessment failed: {e}")
            return {
                'overall_liquidity_score': 0.8,
                'liquidity_risk_level': 'medium'
            }
    
    def _analyze_liquidity_breakdown(self, positions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """流動性内訳分析"""
        try:
            breakdown = {
                'high_liquidity': 0,
                'medium_liquidity': 0,
                'low_liquidity': 0
            }
            
            for position in positions:
                symbol = position.get('symbol', 'USD/JPY')
                char = self.currency_characteristics.get(symbol, {})
                liquidity_score = char.get('liquidity_score', 0.5)
                
                if liquidity_score > 0.8:
                    breakdown['high_liquidity'] += 1
                elif liquidity_score > 0.6:
                    breakdown['medium_liquidity'] += 1
                else:
                    breakdown['low_liquidity'] += 1
            
            return breakdown
            
        except Exception:
            return {'high_liquidity': 0, 'medium_liquidity': 0, 'low_liquidity': 0}
    
    def _analyze_correlation_risk(
        self,
        portfolio_state: Dict[str, Any],
        price_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """相関・クラスターリスク分析"""
        try:
            positions = portfolio_state.get('positions', [])
            
            if len(positions) <= 1:
                return {
                    'correlation_risk_score': 0.0,
                    'cluster_analysis': {},
                    'diversification_benefit': 1.0
                }
            
            # 通貨グループ別の分析
            group_clusters = {}
            total_exposure = 0
            
            for position in positions:
                symbol = position.get('symbol', 'USD/JPY')
                value = position.get('market_value', position.get('size', 0))
                total_exposure += value
                
                char = self.currency_characteristics.get(symbol, {})
                group = char.get('correlation_group', 'other')
                
                if group not in group_clusters:
                    group_clusters[group] = {'exposure': 0, 'count': 0}
                
                group_clusters[group]['exposure'] += value
                group_clusters[group]['count'] += 1
            
            # クラスターリスクスコア計算
            cluster_risks = []
            for group, data in group_clusters.items():
                weight = data['exposure'] / total_exposure if total_exposure > 0 else 0
                cluster_risks.append(weight**2)
            
            correlation_risk_score = sum(cluster_risks)
            
            # 分散化効果
            diversification_benefit = 1 - correlation_risk_score
            
            return {
                'correlation_risk_score': float(correlation_risk_score),
                'cluster_analysis': group_clusters,
                'diversification_benefit': float(diversification_benefit),
                'correlation_matrix': self._estimate_correlation_matrix(positions)
            }
            
        except Exception as e:
            self.logger.error(f"Correlation risk analysis failed: {e}")
            return {
                'correlation_risk_score': 0.5,
                'diversification_benefit': 0.5
            }
    
    def _estimate_correlation_matrix(self, positions: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """相関行列の推定"""
        try:
            symbols = [pos.get('symbol', 'USD/JPY') for pos in positions]
            
            # 簡易的な相関係数（実際の実装では価格データから計算）
            correlation_matrix = {}
            
            for i, symbol1 in enumerate(symbols):
                correlation_matrix[symbol1] = {}
                for j, symbol2 in enumerate(symbols):
                    if i == j:
                        corr = 1.0
                    else:
                        # 通貨グループ間の典型的な相関
                        char1 = self.currency_characteristics.get(symbol1, {})
                        char2 = self.currency_characteristics.get(symbol2, {})
                        
                        if char1.get('correlation_group') == char2.get('correlation_group'):
                            corr = 0.7  # 同じグループ内は高い相関
                        else:
                            corr = 0.3  # 異なるグループは低い相関
                    
                    correlation_matrix[symbol1][symbol2] = corr
            
            return correlation_matrix
            
        except Exception:
            return {}
    
    def _run_stress_tests(
        self,
        portfolio_state: Dict[str, Any],
        var_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """ストレステスト実行"""
        try:
            total_value = portfolio_state.get('total_value', 1000000)
            if isinstance(total_value, str):
                total_value = float(total_value.replace(',', '').replace('$', ''))
            
            stress_scenarios = {
                '2008_crisis_scenario': {
                    'description': '2008年金融危機相当のボラティリティショック',
                    'volatility_multiplier': 3.0,
                    'correlation_increase': 0.9,
                    'liquidity_impact': 0.5
                },
                'volatility_spike_scenario': {
                    'description': 'ボラティリティ2倍シナリオ',
                    'volatility_multiplier': 2.0,
                    'correlation_increase': 0.7,
                    'liquidity_impact': 0.3
                },
                'flash_crash_scenario': {
                    'description': 'フラッシュクラッシュシナリオ',
                    'volatility_multiplier': 5.0,
                    'correlation_increase': 0.95,
                    'liquidity_impact': 0.8
                }
            }
            
            stress_results = {}
            
            for scenario_name, scenario in stress_scenarios.items():
                # ベースVaRにストレス乗数を適用
                base_var = var_analysis.get('1_day_var_99', total_value * 0.02)
                stressed_var = base_var * scenario['volatility_multiplier']
                
                # ポートフォリオへの影響
                portfolio_impact_pct = stressed_var / total_value * 100
                
                # 流動性インパクト
                liquidity_cost = total_value * scenario['liquidity_impact'] * 0.01
                
                # 総合ストレス損失
                total_stress_loss = stressed_var + liquidity_cost
                total_stress_pct = total_stress_loss / total_value * 100
                
                stress_results[scenario_name] = {
                    'description': scenario['description'],
                    'stressed_var': float(stressed_var),
                    'portfolio_impact_pct': float(portfolio_impact_pct),
                    'liquidity_cost': float(liquidity_cost),
                    'total_stress_loss': float(total_stress_loss),
                    'total_stress_pct': float(total_stress_pct),
                    'survivability': 'pass' if total_stress_pct < 15 else 'fail'
                }
            
            return stress_results
            
        except Exception as e:
            self.logger.error(f"Stress test failed: {e}")
            return {}
    
    def _calculate_overall_risk_score(
        self,
        var_analysis: Dict[str, Any],
        drawdown_analysis: Dict[str, Any],
        position_analysis: Dict[str, Any],
        market_risk_analysis: Dict[str, Any],
        liquidity_analysis: Dict[str, Any],
        correlation_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """総合リスクスコア計算"""
        try:
            risk_components = []
            
            # VaRリスクスコア
            var_utilization = var_analysis.get('var_utilization', 0)
            var_score = min(var_utilization, 1.0)
            risk_components.append(('var_risk', var_score, 0.25))
            
            # ドローダウンリスクスコア
            dd_utilization = drawdown_analysis.get('drawdown_utilization', 0)
            dd_score = min(dd_utilization, 1.0)
            risk_components.append(('drawdown_risk', dd_score, 0.20))
            
            # 集中度リスクスコア
            concentration = position_analysis.get('portfolio_concentration', 0)
            concentration_score = min(concentration, 1.0)
            risk_components.append(('concentration_risk', concentration_score, 0.15))
            
            # 市場リスクスコア
            market_stress = market_risk_analysis.get('market_stress_indicator', 0.5)
            risk_components.append(('market_risk', market_stress, 0.20))
            
            # 流動性リスクスコア
            liquidity_score = 1 - liquidity_analysis.get('overall_liquidity_score', 0.8)
            risk_components.append(('liquidity_risk', liquidity_score, 0.10))
            
            # 相関リスクスコア
            correlation_score = correlation_analysis.get('correlation_risk_score', 0.5)
            risk_components.append(('correlation_risk', correlation_score, 0.10))
            
            # 重み付き総合スコア
            total_risk_score = sum(score * weight for _, score, weight in risk_components)
            
            # リスクレベル分類
            if total_risk_score < 0.3:
                risk_level = 'low'
                recommended_action = 'maintain'
                confidence = 0.8
            elif total_risk_score < 0.6:
                risk_level = 'medium'
                recommended_action = 'monitor'
                confidence = 0.7
            elif total_risk_score < 0.8:
                risk_level = 'high'
                recommended_action = 'reduce_risk'
                confidence = 0.8
            else:
                risk_level = 'critical'
                recommended_action = 'emergency_reduction'
                confidence = 0.9
            
            # 主要リスク要因の特定
            top_risks = sorted(risk_components, key=lambda x: x[1], reverse=True)[:3]
            
            reasoning = f"総合リスクスコア {total_risk_score:.2f} - 主要リスク: {', '.join([r[0] for r in top_risks])}"
            
            return {
                'risk_score': float(total_risk_score),
                'risk_level': risk_level,
                'recommended_action': recommended_action,
                'confidence': confidence,
                'reasoning': reasoning,
                'risk_components': {name: score for name, score, _ in risk_components},
                'top_risk_factors': [{'factor': name, 'score': score} for name, score, _ in top_risks]
            }
            
        except Exception as e:
            self.logger.error(f"Overall risk score calculation failed: {e}")
            return {
                'risk_score': 0.7,
                'risk_level': 'high',
                'recommended_action': 'reduce_risk',
                'confidence': 0.6,
                'reasoning': f'リスク計算エラーのため保守的評価: {str(e)}'
            }
    
    def _generate_risk_recommendations(
        self,
        overall_risk_assessment: Dict[str, Any],
        var_analysis: Dict[str, Any],
        drawdown_analysis: Dict[str, Any],
        position_analysis: Dict[str, Any],
        stress_test_results: Dict[str, Any]
    ) -> List[str]:
        """リスク軽減推奨事項生成"""
        recommendations = []
        
        try:
            risk_score = overall_risk_assessment.get('risk_score', 0.5)
            risk_components = overall_risk_assessment.get('risk_components', {})
            
            # VaRリスク対応
            if risk_components.get('var_risk', 0) > 0.7:
                recommendations.append('VaR利用率が高いため、ポジションサイズを縮小してください')
            
            # ドローダウンリスク対応
            if risk_components.get('drawdown_risk', 0) > 0.6:
                recommendations.append('ドローダウンが限界に近づいています。新規ポジションを制限してください')
            
            # 集中度リスク対応
            concentration = position_analysis.get('portfolio_concentration', 0)
            if concentration > 0.5:
                recommendations.append('ポートフォリオの集中度が高すぎます。分散化を進めてください')
            
            largest_position = position_analysis.get('largest_position_risk', 0)
            if largest_position > self.risk_limits['max_single_position_pct']:
                recommendations.append(f'最大ポジションが制限（{self.risk_limits["max_single_position_pct"]}%）を超過しています')
            
            # 相関リスク対応
            correlation_risk = risk_components.get('correlation_risk', 0)
            if correlation_risk > 0.6:
                recommendations.append('相関の高いポジションが集中しています。異なる資産クラスへの分散を検討してください')
            
            # 流動性リスク対応
            liquidity_risk = risk_components.get('liquidity_risk', 0)
            if liquidity_risk > 0.5:
                recommendations.append('流動性の低いポジションを削減し、主要通貨ペアに集中してください')
            
            # ストレステスト結果に基づく推奨
            for scenario_name, result in stress_test_results.items():
                if result.get('survivability') == 'fail':
                    recommendations.append(f'{scenario_name}で大きな損失リスク。リスク削減が必要です')
            
            # 全般的な推奨事項
            if risk_score > 0.8:
                recommendations.append('緊急：全ポジションを見直し、リスクエクスポージャーを大幅に削減してください')
            elif risk_score > 0.6:
                recommendations.append('注意：リスク管理体制を強化し、定期的な見直しを実施してください')
            
            return recommendations[:10]  # 最大10個の推奨事項
            
        except Exception as e:
            self.logger.error(f"Risk recommendations generation failed: {e}")
            return ['リスク評価エラーのため、すべてのポジションを保守的に管理してください']
    
    def _calculate_position_sizing_guidance(
        self,
        overall_risk_assessment: Dict[str, Any],
        portfolio_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """ポジションサイジングガイダンス"""
        try:
            risk_score = overall_risk_assessment.get('risk_score', 0.5)
            total_value = portfolio_state.get('total_value', 1000000)
            if isinstance(total_value, str):
                total_value = float(total_value.replace(',', '').replace('$', ''))
            
            # リスクスコアに基づくサイジング調整
            risk_multiplier = max(0.1, 1.0 - risk_score)
            
            # 基準ポジションサイズ
            base_position_size = total_value * (self.risk_limits['max_single_position_pct'] / 100)
            
            # 調整後最大ポジションサイズ
            max_new_position_size = base_position_size * risk_multiplier
            
            # 利用可能リスク予算
            used_risk_budget = risk_score * 100
            available_risk_budget = max(0, 100 - used_risk_budget)
            
            return {
                'max_new_position_size': float(max_new_position_size),
                'risk_budget_available': float(available_risk_budget),
                'position_size_multiplier': float(risk_multiplier),
                'recommended_sizing_method': self._recommend_sizing_method(risk_score)
            }
            
        except Exception as e:
            self.logger.error(f"Position sizing guidance calculation failed: {e}")
            return {
                'max_new_position_size': 10000,
                'risk_budget_available': 50.0
            }
    
    def _recommend_sizing_method(self, risk_score: float) -> str:
        """ポジションサイジング手法の推奨"""
        if risk_score > 0.7:
            return 'fixed_fractional_conservative'
        elif risk_score > 0.4:
            return 'kelly_criterion_adjusted'
        else:
            return 'optimal_f'
    
    def _assess_immediate_actions(
        self,
        overall_risk_assessment: Dict[str, Any],
        stress_test_results: Dict[str, Any]
    ) -> List[str]:
        """緊急対応アクション評価"""
        immediate_actions = []
        
        try:
            risk_score = overall_risk_assessment.get('risk_score', 0.5)
            risk_level = overall_risk_assessment.get('risk_level', 'medium')
            
            # 緊急度に応じたアクション
            if risk_level == 'critical':
                immediate_actions.extend([
                    '全ポジションの緊急見直し実施',
                    'リスク限度額の一時的縮小',
                    '新規ポジション取得の一時停止',
                    'ストップロス設定の厳格化'
                ])
            elif risk_level == 'high':
                immediate_actions.extend([
                    '大型ポジションの段階的縮小',
                    'VaR限度の見直し',
                    '相関の高いポジション整理'
                ])
            
            # ストレステスト結果に基づく緊急アクション
            failed_scenarios = [
                name for name, result in stress_test_results.items() 
                if result.get('survivability') == 'fail'
            ]
            
            if failed_scenarios:
                immediate_actions.append(f'ストレステスト不合格（{len(failed_scenarios)}シナリオ）につきリスク削減')
            
            return immediate_actions
            
        except Exception as e:
            self.logger.error(f"Immediate actions assessment failed: {e}")
            return ['リスク評価エラーのため緊急見直し実施']