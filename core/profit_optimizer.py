#!/usr/bin/env python3
"""
Profit-Based AI Optimization System
利益ベースAI最適化システム
"""

import asyncio
import datetime
import json
import logging
try:
    import numpy as np
except ImportError:
    # Numpyが利用できない場合のフォールバック
    class MockNumpy:
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0
        
        @staticmethod
        def std(data):
            if not data:
                return 0
            mean_val = sum(data) / len(data)
            variance = sum((x - mean_val) ** 2 for x in data) / len(data)
            return variance ** 0.5
        
        @staticmethod
        def sqrt(x):
            return x ** 0.5
    
    np = MockNumpy()
import random
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sys
import os


from core.ai_orchestrator import DepartmentType
from core.database_adapter import db_adapter


class ProfitMetric(Enum):
    """利益メトリクス"""
    TOTAL_RETURN = "total_return"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    ROI = "roi"
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"


@dataclass
class DepartmentConfiguration:
    """部門設定"""
    department: DepartmentType
    model_name: str
    weight: float
    enabled: bool = True
    cost_per_analysis: float = 0.01
    confidence_threshold: float = 0.5


@dataclass
class TradingSession:
    """取引セッション"""
    session_id: str
    department_config: List[DepartmentConfiguration]
    start_balance: float
    current_balance: float
    trades: List[Dict[str, Any]]
    session_start: datetime.datetime
    session_end: Optional[datetime.datetime] = None
    total_cost: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['session_start'] = self.session_start.isoformat()
        if self.session_end:
            result['session_end'] = self.session_end.isoformat()
        result['department_config'] = [
            {**config.__dict__, 'department': config.department.value}
            for config in self.department_config
        ]
        return result


@dataclass
class ProfitAnalysis:
    """利益分析結果"""
    configuration_id: str
    total_return: float
    roi_percentage: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    profitable_trades: int
    total_cost: float
    net_profit: float
    risk_adjusted_return: float
    analysis_timestamp: datetime.datetime
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['analysis_timestamp'] = self.analysis_timestamp.isoformat()
        return result


class ProfitOptimizer:
    """利益最適化システム"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 取引パラメータ
        self.base_position_size = 10000  # $10,000
        self.leverage = 1.0  # レバレッジなし（安全重視）
        self.spread_cost = 0.0002  # 0.02% (2 pips for USD/JPY)
        self.commission_rate = 0.0001  # 0.01%
        
        # リスク管理パラメータ
        self.max_risk_per_trade = 0.02  # 2%
        self.max_daily_drawdown = 0.05  # 5%
        self.stop_loss_ratio = 0.015  # 1.5%
        self.take_profit_ratio = 0.03  # 3% (Risk:Reward = 1:2)
        
        # AI部門設定
        self.department_models = {
            DepartmentType.TECHNICAL: ["gpt-4-turbo", "o3-mini", "claude-sonnet"],
            DepartmentType.FUNDAMENTAL: ["gpt-4", "o3-mini", "claude-sonnet"],
            DepartmentType.SENTIMENT: ["gpt-4-turbo", "o3-mini"],
            DepartmentType.RISK: ["o3", "gpt-4"],
            DepartmentType.EXECUTION: ["o3-mini", "gpt-4-turbo"]
        }
        
        # モデルコスト設定
        self.model_costs = {
            "gpt-4": 0.08,
            "gpt-4-turbo": 0.04,
            "o3-mini": 0.03,
            "o3": 0.15,
            "claude-sonnet": 0.05
        }
        
        self.logger.info("Profit Optimizer initialized")
    
    async def generate_department_combinations(self) -> List[List[DepartmentConfiguration]]:
        """部門組み合わせパターン生成"""
        combinations = []
        
        # パターン1: 全部門フル活用（高コスト・高精度想定）
        full_config = [
            DepartmentConfiguration(DepartmentType.TECHNICAL, "o3-mini", 0.35, True, 0.03, 0.7),
            DepartmentConfiguration(DepartmentType.FUNDAMENTAL, "gpt-4", 0.25, True, 0.08, 0.6),
            DepartmentConfiguration(DepartmentType.SENTIMENT, "o3-mini", 0.20, True, 0.03, 0.5),
            DepartmentConfiguration(DepartmentType.RISK, "o3", 0.15, True, 0.15, 0.8),
            DepartmentConfiguration(DepartmentType.EXECUTION, "o3-mini", 0.05, True, 0.03, 0.6)
        ]
        combinations.append(full_config)
        
        # パターン2: コスト効率重視（低コスト・中精度）
        cost_efficient = [
            DepartmentConfiguration(DepartmentType.TECHNICAL, "o3-mini", 0.45, True, 0.03, 0.6),
            DepartmentConfiguration(DepartmentType.FUNDAMENTAL, "o3-mini", 0.25, True, 0.03, 0.5),
            DepartmentConfiguration(DepartmentType.SENTIMENT, "gpt-4-turbo", 0.20, True, 0.04, 0.5),
            DepartmentConfiguration(DepartmentType.RISK, "gpt-4", 0.10, True, 0.08, 0.7)
        ]
        combinations.append(cost_efficient)
        
        # パターン3: テクニカル重視
        technical_focused = [
            DepartmentConfiguration(DepartmentType.TECHNICAL, "o3", 0.60, True, 0.15, 0.8),
            DepartmentConfiguration(DepartmentType.RISK, "o3", 0.25, True, 0.15, 0.8),
            DepartmentConfiguration(DepartmentType.EXECUTION, "o3-mini", 0.15, True, 0.03, 0.6)
        ]
        combinations.append(technical_focused)
        
        # パターン4: ファンダメンタル重視
        fundamental_focused = [
            DepartmentConfiguration(DepartmentType.FUNDAMENTAL, "gpt-4", 0.50, True, 0.08, 0.7),
            DepartmentConfiguration(DepartmentType.SENTIMENT, "o3-mini", 0.30, True, 0.03, 0.6),
            DepartmentConfiguration(DepartmentType.RISK, "gpt-4", 0.20, True, 0.08, 0.7)
        ]
        combinations.append(fundamental_focused)
        
        # パターン5: バランス型（中コスト・中精度）
        balanced = [
            DepartmentConfiguration(DepartmentType.TECHNICAL, "o3-mini", 0.30, True, 0.03, 0.65),
            DepartmentConfiguration(DepartmentType.FUNDAMENTAL, "o3-mini", 0.25, True, 0.03, 0.6),
            DepartmentConfiguration(DepartmentType.SENTIMENT, "gpt-4-turbo", 0.25, True, 0.04, 0.55),
            DepartmentConfiguration(DepartmentType.RISK, "gpt-4-turbo", 0.20, True, 0.04, 0.75)
        ]
        combinations.append(balanced)
        
        # パターン6: 超低コスト
        ultra_low_cost = [
            DepartmentConfiguration(DepartmentType.TECHNICAL, "o3-mini", 0.70, True, 0.03, 0.5),
            DepartmentConfiguration(DepartmentType.RISK, "o3-mini", 0.30, True, 0.03, 0.6)
        ]
        combinations.append(ultra_low_cost)
        
        self.logger.info(f"Generated {len(combinations)} department combinations")
        return combinations
    
    async def simulate_trading_session(
        self, 
        config: List[DepartmentConfiguration],
        market_scenarios: List[Dict[str, Any]],
        initial_balance: float = 100000
    ) -> TradingSession:
        """取引セッションシミュレーション"""
        
        session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000,9999)}"
        
        session = TradingSession(
            session_id=session_id,
            department_config=config,
            start_balance=initial_balance,
            current_balance=initial_balance,
            trades=[],
            session_start=datetime.datetime.now()
        )
        
        for i, scenario in enumerate(market_scenarios):
            trade_result = await self._execute_trade_simulation(session, scenario, config)
            
            if trade_result:
                session.trades.append(trade_result)
                session.current_balance += trade_result['profit_loss']
                session.total_cost += trade_result['ai_cost']
                
                # ドローダウンチェック
                drawdown = (session.start_balance - session.current_balance) / session.start_balance
                if drawdown > self.max_daily_drawdown:
                    self.logger.warning(f"Session {session_id} stopped due to max drawdown")
                    break
        
        session.session_end = datetime.datetime.now()
        return session
    
    async def _execute_trade_simulation(
        self,
        session: TradingSession,
        scenario: Dict[str, Any],
        config: List[DepartmentConfiguration]
    ) -> Optional[Dict[str, Any]]:
        """個別取引シミュレーション"""
        
        try:
            # AI部門分析シミュレーション
            department_decisions = []
            total_ai_cost = 0.0
            
            for dept_config in config:
                if not dept_config.enabled:
                    continue
                
                # 部門別分析結果をシミュレート
                decision = self._simulate_department_decision(dept_config, scenario)
                department_decisions.append(decision)
                total_ai_cost += dept_config.cost_per_analysis
            
            # 統合判定
            final_decision = self._integrate_department_decisions(department_decisions, config)
            
            if not final_decision['trade_warranted']:
                return {
                    'scenario_id': scenario.get('id', 'unknown'),
                    'action': 'hold',
                    'profit_loss': 0.0,
                    'ai_cost': total_ai_cost,
                    'net_result': -total_ai_cost,
                    'reason': 'no_trade_signal'
                }
            
            # 取引実行
            return await self._execute_simulated_trade(
                final_decision, scenario, total_ai_cost, session.current_balance
            )
            
        except Exception as e:
            self.logger.error(f"Trade simulation failed: {e}")
            return None
    
    def _simulate_department_decision(
        self, 
        dept_config: DepartmentConfiguration, 
        scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """部門判定シミュレーション"""
        
        # シナリオタイプによる成功確率
        scenario_type = scenario.get('type', 'neutral')
        base_accuracy = self._get_model_accuracy(dept_config.model_name, dept_config.department)
        
        # シナリオ難易度調整
        difficulty_multiplier = {
            'clear_bullish': 1.2,
            'clear_bearish': 1.2,
            'neutral': 0.6,
            'mixed_signals': 0.4,
            'high_volatility': 0.7
        }.get(scenario_type, 0.8)
        
        success_probability = base_accuracy * difficulty_multiplier
        success_probability = max(0.1, min(0.95, success_probability))
        
        is_correct = random.random() < success_probability
        confidence = random.uniform(0.4, 0.9) if is_correct else random.uniform(0.2, 0.6)
        
        # 期待される方向性
        expected_direction = scenario.get('expected_direction', 'hold')
        
        if is_correct and confidence > dept_config.confidence_threshold:
            decision_direction = expected_direction
            trade_warranted = expected_direction != 'hold'
        else:
            decision_direction = random.choice(['long', 'short', 'hold'])
            trade_warranted = decision_direction != 'hold' and confidence > dept_config.confidence_threshold
        
        return {
            'department': dept_config.department,
            'model': dept_config.model_name,
            'trade_warranted': trade_warranted,
            'direction': decision_direction,
            'confidence': confidence,
            'weight': dept_config.weight,
            'cost': dept_config.cost_per_analysis,
            'is_correct': is_correct
        }
    
    def _get_model_accuracy(self, model_name: str, department: DepartmentType) -> float:
        """モデル・部門別の精度設定"""
        base_accuracies = {
            "gpt-4": 0.72,
            "gpt-4-turbo": 0.75,
            "o3-mini": 0.78,
            "o3": 0.82,
            "claude-sonnet": 0.76
        }
        
        # 部門別調整
        department_multipliers = {
            DepartmentType.TECHNICAL: 1.0,
            DepartmentType.FUNDAMENTAL: 0.9,  # 予測が難しい
            DepartmentType.SENTIMENT: 0.85,   # ノイズが多い
            DepartmentType.RISK: 1.1,         # 比較的正確
            DepartmentType.EXECUTION: 1.05    # 実行は得意
        }
        
        base_acc = base_accuracies.get(model_name, 0.7)
        dept_mult = department_multipliers.get(department, 1.0)
        
        return base_acc * dept_mult
    
    def _integrate_department_decisions(
        self, 
        decisions: List[Dict[str, Any]], 
        config: List[DepartmentConfiguration]
    ) -> Dict[str, Any]:
        """部門判定統合"""
        
        if not decisions:
            return {'trade_warranted': False, 'direction': 'hold', 'confidence': 0.0}
        
        # 重み付き投票
        total_weight = 0.0
        weighted_confidence = 0.0
        direction_votes = {'long': 0, 'short': 0, 'hold': 0}
        trade_votes = []
        
        for decision in decisions:
            weight = decision['weight']
            confidence = decision['confidence']
            direction = decision['direction']
            trade_warranted = decision['trade_warranted']
            
            total_weight += weight
            weighted_confidence += confidence * weight
            direction_votes[direction] += weight
            
            if trade_warranted:
                trade_votes.append(weight)
        
        # 最終判定
        avg_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0
        
        # 方向決定（重み付き多数決）
        final_direction = max(direction_votes, key=direction_votes.get)
        
        # 取引実行判定
        trade_weight_sum = sum(trade_votes)
        trade_warranted = (
            trade_weight_sum > total_weight * 0.6 and  # 60%以上の重みで取引推奨
            avg_confidence > 0.6 and  # 信頼度60%以上
            final_direction != 'hold'
        )
        
        return {
            'trade_warranted': trade_warranted,
            'direction': final_direction,
            'confidence': avg_confidence,
            'department_count': len(decisions),
            'consensus_strength': direction_votes[final_direction] / total_weight
        }
    
    async def _execute_simulated_trade(
        self,
        decision: Dict[str, Any],
        scenario: Dict[str, Any],
        ai_cost: float,
        current_balance: float
    ) -> Dict[str, Any]:
        """取引実行シミュレーション"""
        
        direction = decision['direction']
        confidence = decision['confidence']
        
        # ポジションサイズ計算（信頼度ベース）
        risk_amount = current_balance * self.max_risk_per_trade
        position_size = risk_amount * confidence  # 信頼度でサイズ調整
        position_size = min(position_size, self.base_position_size)
        
        # 市場結果シミュレーション
        market_move = self._simulate_market_movement(scenario)
        
        # P&L計算
        if direction == 'long':
            profit_loss = position_size * market_move
        elif direction == 'short':
            profit_loss = position_size * (-market_move)
        else:
            profit_loss = 0.0
        
        # コスト差し引き
        trading_cost = position_size * (self.spread_cost + self.commission_rate)
        net_profit_loss = profit_loss - trading_cost
        
        return {
            'scenario_id': scenario.get('id', 'unknown'),
            'action': direction,
            'position_size': position_size,
            'market_move': market_move,
            'gross_profit_loss': profit_loss,
            'trading_cost': trading_cost,
            'profit_loss': net_profit_loss,
            'ai_cost': ai_cost,
            'net_result': net_profit_loss - ai_cost,
            'confidence': confidence,
            'decision_details': decision
        }
    
    def _simulate_market_movement(self, scenario: Dict[str, Any]) -> float:
        """市場動きシミュレーション"""
        
        scenario_type = scenario.get('type', 'neutral')
        volatility = scenario.get('volatility', 0.01)  # デフォルト1%
        
        # シナリオ別の期待リターン
        expected_moves = {
            'clear_bullish': 0.015,    # 1.5%上昇
            'clear_bearish': -0.012,   # 1.2%下落
            'neutral': 0.002,          # 0.2%の小動き
            'mixed_signals': 0.005,    # 0.5%
            'high_volatility': 0.008   # 0.8%（方向はランダム）
        }
        
        base_move = expected_moves.get(scenario_type, 0.0)
        
        # ボラティリティ追加
        noise = random.gauss(0, volatility)
        actual_move = base_move + noise
        
        # 極端な動きを制限（-5%〜+5%）
        return max(-0.05, min(0.05, actual_move))
    
    async def calculate_profit_analysis(self, session: TradingSession) -> ProfitAnalysis:
        """利益分析計算"""
        
        if not session.trades:
            return ProfitAnalysis(
                configuration_id=session.session_id,
                total_return=0.0,
                roi_percentage=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                total_trades=0,
                profitable_trades=0,
                total_cost=session.total_cost,
                net_profit=-session.total_cost,
                risk_adjusted_return=0.0,
                analysis_timestamp=datetime.datetime.now()
            )
        
        # 基本統計
        total_trades = len(session.trades)
        profitable_trades = sum(1 for trade in session.trades if trade['net_result'] > 0)
        
        # 収益計算
        gross_profits = sum(trade['profit_loss'] for trade in session.trades if trade['profit_loss'] > 0)
        gross_losses = abs(sum(trade['profit_loss'] for trade in session.trades if trade['profit_loss'] < 0))
        
        total_return = session.current_balance - session.start_balance
        roi_percentage = (total_return / session.start_balance) * 100
        net_profit = total_return - session.total_cost
        
        # 勝率
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0.0
        
        # プロフィットファクター
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf') if gross_profits > 0 else 0.0
        
        # 最大ドローダウン
        max_drawdown = self._calculate_max_drawdown(session)
        
        # シャープレシオ
        sharpe_ratio = self._calculate_sharpe_ratio(session)
        
        # リスク調整後リターン
        risk_adjusted_return = roi_percentage / max(max_drawdown, 0.01) if max_drawdown > 0 else roi_percentage
        
        return ProfitAnalysis(
            configuration_id=session.session_id,
            total_return=total_return,
            roi_percentage=roi_percentage,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=total_trades,
            profitable_trades=profitable_trades,
            total_cost=session.total_cost,
            net_profit=net_profit,
            risk_adjusted_return=risk_adjusted_return,
            analysis_timestamp=datetime.datetime.now()
        )
    
    def _calculate_max_drawdown(self, session: TradingSession) -> float:
        """最大ドローダウン計算"""
        if not session.trades:
            return 0.0
        
        running_balance = session.start_balance
        peak_balance = session.start_balance
        max_drawdown = 0.0
        
        for trade in session.trades:
            running_balance += trade['net_result']
            peak_balance = max(peak_balance, running_balance)
            
            if peak_balance > 0:
                drawdown = (peak_balance - running_balance) / peak_balance
                max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown * 100  # パーセント表示
    
    def _calculate_sharpe_ratio(self, session: TradingSession) -> float:
        """シャープレシオ計算"""
        if len(session.trades) < 2:
            return 0.0
        
        returns = [trade['net_result'] / session.start_balance for trade in session.trades]
        
        if len(returns) == 0:
            return 0.0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # 年率換算（仮に252営業日）
        sharpe = (mean_return * 252) / (std_return * np.sqrt(252))
        return sharpe
    
    async def generate_market_scenarios(self, count: int = 50) -> List[Dict[str, Any]]:
        """市場シナリオ生成"""
        scenarios = []
        
        scenario_types = [
            ('clear_bullish', 0.2),    # 20%
            ('clear_bearish', 0.2),    # 20%
            ('neutral', 0.3),          # 30%
            ('mixed_signals', 0.2),    # 20%
            ('high_volatility', 0.1)   # 10%
        ]
        
        for i in range(count):
            # シナリオタイプ選択
            scenario_type = random.choices(
                [t[0] for t in scenario_types], 
                weights=[t[1] for t in scenario_types]
            )[0]
            
            # ボラティリティ設定
            volatility = random.uniform(0.005, 0.025)  # 0.5%〜2.5%
            
            # 期待方向設定
            if scenario_type == 'clear_bullish':
                expected_direction = 'long'
            elif scenario_type == 'clear_bearish':
                expected_direction = 'short'
            else:
                expected_direction = 'hold'
            
            scenario = {
                'id': f'scenario_{i+1}',
                'type': scenario_type,
                'expected_direction': expected_direction,
                'volatility': volatility,
                'timestamp': datetime.datetime.now() + datetime.timedelta(hours=i),
                'news_sentiment': random.uniform(-1, 1),
                'economic_strength': random.uniform(0, 1)
            }
            
            scenarios.append(scenario)
        
        return scenarios
    
    async def run_profit_optimization_test(self, scenario_count: int = 30) -> Dict[str, Any]:
        """利益最適化テスト実行"""
        self.logger.info("Starting profit optimization test...")
        
        try:
            # 部門組み合わせ生成
            combinations = await self.generate_department_combinations()
            
            # 市場シナリオ生成
            scenarios = await self.generate_market_scenarios(scenario_count)
            
            # 各組み合わせでテスト実行
            results = []
            
            for i, config in enumerate(combinations):
                self.logger.info(f"Testing configuration {i+1}/{len(combinations)}")
                
                # 取引セッション実行
                session = await self.simulate_trading_session(config, scenarios)
                
                # 利益分析
                analysis = await self.calculate_profit_analysis(session)
                
                # 設定情報追加
                config_info = {
                    'config_id': i+1,
                    'departments': [
                        {'department': cfg.department.value, 'model': cfg.model_name, 'weight': cfg.weight}
                        for cfg in config
                    ],
                    'total_departments': len(config),
                    'avg_cost_per_analysis': sum(cfg.cost_per_analysis for cfg in config) / len(config)
                }
                
                result = {
                    'configuration': config_info,
                    'session': session.to_dict(),
                    'analysis': analysis.to_dict()
                }
                
                results.append(result)
            
            # 最適化分析
            optimization_summary = self._analyze_optimization_results(results)
            
            return {
                'test_timestamp': datetime.datetime.now().isoformat(),
                'scenario_count': scenario_count,
                'configuration_count': len(combinations),
                'results': results,
                'optimization_summary': optimization_summary,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Profit optimization test failed: {e}")
            return {'error': str(e), 'success': False}
    
    def _analyze_optimization_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """最適化結果分析"""
        
        if not results:
            return {'error': 'No results to analyze'}
        
        # 各メトリクスでのランキング
        rankings = {}
        
        metrics = ['roi_percentage', 'sharpe_ratio', 'win_rate', 'profit_factor', 'risk_adjusted_return']
        
        for metric in metrics:
            sorted_results = sorted(
                results, 
                key=lambda x: x['analysis'][metric] if x['analysis'][metric] != float('inf') else 999,
                reverse=True
            )
            rankings[metric] = [
                {
                    'rank': i+1,
                    'config_id': r['configuration']['config_id'],
                    'value': r['analysis'][metric],
                    'departments': [d['department'] for d in r['configuration']['departments']]
                }
                for i, r in enumerate(sorted_results[:5])  # Top 5
            ]
        
        # 総合スコア計算
        comprehensive_scores = []
        for result in results:
            analysis = result['analysis']
            
            # 正規化スコア計算（負の値も考慮）
            roi_score = max(0, analysis['roi_percentage'] / 100)  # 0-1スケール
            sharpe_score = max(0, min(1, analysis['sharpe_ratio'] / 3))  # 3を上限とする
            win_rate_score = analysis['win_rate']
            risk_adj_score = max(0, min(1, analysis['risk_adjusted_return'] / 50))  # 50%を上限
            
            # 重み付き総合スコア
            comprehensive_score = (
                roi_score * 0.3 +
                sharpe_score * 0.25 +
                win_rate_score * 0.2 +
                risk_adj_score * 0.25
            )
            
            comprehensive_scores.append({
                'config_id': result['configuration']['config_id'],
                'comprehensive_score': comprehensive_score,
                'roi_percentage': analysis['roi_percentage'],
                'sharpe_ratio': analysis['sharpe_ratio'],
                'win_rate': analysis['win_rate'],
                'total_cost': analysis['total_cost'],
                'departments': result['configuration']['departments']
            })
        
        # 総合ランキング
        comprehensive_scores.sort(key=lambda x: x['comprehensive_score'], reverse=True)
        
        # コスト効率分析
        cost_efficiency = []
        for result in results:
            analysis = result['analysis']
            if analysis['total_cost'] > 0:
                efficiency = analysis['net_profit'] / analysis['total_cost']
                cost_efficiency.append({
                    'config_id': result['configuration']['config_id'],
                    'cost_efficiency': efficiency,
                    'total_cost': analysis['total_cost'],
                    'net_profit': analysis['net_profit'],
                    'departments': [d['department'] for d in result['configuration']['departments']]
                })
        
        cost_efficiency.sort(key=lambda x: x['cost_efficiency'], reverse=True)
        
        return {
            'metric_rankings': rankings,
            'comprehensive_ranking': comprehensive_scores[:10],  # Top 10
            'cost_efficiency_ranking': cost_efficiency[:10],
            'best_configuration': comprehensive_scores[0] if comprehensive_scores else None,
            'summary_statistics': {
                'avg_roi': sum(r['analysis']['roi_percentage'] for r in results) / len(results),
                'avg_win_rate': sum(r['analysis']['win_rate'] for r in results) / len(results),
                'avg_total_cost': sum(r['analysis']['total_cost'] for r in results) / len(results),
                'best_roi': max(r['analysis']['roi_percentage'] for r in results),
                'best_sharpe': max(r['analysis']['sharpe_ratio'] for r in results if r['analysis']['sharpe_ratio'] != float('inf')),
                'configurations_tested': len(results)
            }
        }


# シングルトンインスタンス
profit_optimizer = ProfitOptimizer()