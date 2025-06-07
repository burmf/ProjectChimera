"""
Alpha Integration Engine
既存システムとUltra Alpha Optimizerの統合モジュール

Purpose: 
- 既存alpha_engine.pyとultra_alpha_optimizer.pyの統合
- 実取引向け信号生成とポートフォリオ最適化
- リアルタイム・エッジ監視とリスク管理
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

# Import our engines
from .alpha_engine import alpha_engine
from .ultra_alpha_optimizer import ultra_optimizer, OptimalSignal, EdgeMetrics

@dataclass
class IntegratedAlphaSignal:
    """統合アルファ信号"""
    # Core signal data
    signal_strength: float      # -1.0 to 1.0 (主信号)
    confidence: float          # 0.0 to 1.0 (信頼度)
    
    # Edge optimization data
    expected_edge_bps: float   # 期待エッジ (basis points)
    optimal_position_size: float  # 最適ポジションサイズ
    risk_adjusted_return: float   # リスク調整済みリターン
    
    # Execution parameters
    entry_price: float         # 推奨エントリー価格
    stop_loss_price: float     # ストップロス価格
    take_profit_price: float   # テイクプロフィット価格
    holding_period_minutes: int # 推奨保有期間
    
    # Strategy breakdown
    timeframe_components: Dict[str, float]  # 時間軸別信号強度
    strategy_components: Dict[str, float]   # 戦略別信号強度
    
    # Risk metrics
    sharpe_estimate: float     # 推定シャープレシオ
    max_drawdown_estimate: float # 推定最大ドローダウン
    
    # Meta data
    timestamp: datetime
    expiry: datetime
    market_regime: str

class AlphaIntegrationEngine:
    """統合アルファエンジン - 全戦略の統合と最適化"""
    
    def __init__(self):
        """初期化"""
        self.logger = logging.getLogger(__name__)
        
        # Integration parameters
        self.integration_config = {
            'base_engine_weight': 0.6,      # 既存エンジンの重み
            'ultra_optimizer_weight': 0.4,   # Ultra optimizerの重み
            'minimum_signal_threshold': 0.15, # 最小信号閾値
            'maximum_position_size': 0.05,   # 最大ポジションサイズ (5%)
            'risk_budget_per_trade': 0.01    # 1取引あたりリスク予算 (1%)
        }
        
        # Performance tracking
        self.performance_metrics = {
            'total_signals_generated': 0,
            'profitable_signals': 0,
            'cumulative_edge': 0.0,
            'average_holding_period': 0.0,
            'best_sharpe_achieved': 0.0
        }
        
    def generate_integrated_alpha_signal(self, market_data: Dict, 
                                       current_price: float) -> Optional[IntegratedAlphaSignal]:
        """
        統合アルファ信号生成
        
        Args:
            market_data: 市場データ (価格、ボリューム、ニュース等)
            current_price: 現在価格
            
        Returns:
            統合最適化済みアルファ信号 or None
        """
        try:
            # 1. 既存アルファエンジンから信号取得
            base_signals = self._get_base_engine_signals(market_data)
            
            # 2. Ultra optimizerから最適化信号取得
            ultra_signal = self._get_ultra_optimized_signal(market_data)
            
            # 3. 信号品質チェック
            if not self._validate_signal_quality(base_signals, ultra_signal):
                return None
                
            # 4. 信号統合と重み最適化
            integrated_signal = self._integrate_signals(base_signals, ultra_signal)
            
            # 5. リスク調整とポジション最適化
            risk_adjusted_signal = self._apply_risk_optimization(
                integrated_signal, current_price, market_data
            )
            
            # 6. 実行パラメータ計算
            execution_params = self._calculate_execution_parameters(
                risk_adjusted_signal, current_price, market_data
            )
            
            # 7. 統合信号組み立て
            final_signal = self._build_integrated_signal(
                risk_adjusted_signal, execution_params, base_signals, 
                ultra_signal, current_price, market_data
            )
            
            # 8. パフォーマンス追跡更新
            self._update_performance_tracking(final_signal)
            
            return final_signal
            
        except Exception as e:
            self.logger.error(f"Integrated alpha signal generation failed: {e}")
            return None
    
    def _get_base_engine_signals(self, market_data: Dict) -> Dict[str, Any]:
        """既存アルファエンジンから信号取得"""
        
        # Convert market_data to DataFrame format for alpha_engine
        df = self._convert_to_dataframe(market_data)
        
        # Generate multi-timeframe signals
        base_signals = alpha_engine.generate_multi_timeframe_signals(df, None)
        
        return base_signals
    
    def _get_ultra_optimized_signal(self, market_data: Dict) -> OptimalSignal:
        """Ultra optimizerから最適化信号取得"""
        
        # Generate ultra-optimized signal
        ultra_signal = ultra_optimizer.optimize_short_term_alpha(market_data)
        
        return ultra_signal
    
    def _validate_signal_quality(self, base_signals: Dict, ultra_signal: OptimalSignal) -> bool:
        """信号品質検証"""
        
        # Base signal validation
        master_signal = base_signals.get('master_signal', {})
        base_confidence = master_signal.get('confidence', 0.0)
        
        # Ultra signal validation
        ultra_confidence = ultra_signal.confidence
        ultra_edge = ultra_signal.expected_edge
        
        # Quality criteria
        criteria = [
            base_confidence > 0.3,                    # Base signal minimum confidence
            ultra_confidence > 0.4,                   # Ultra signal minimum confidence
            ultra_edge > 0.05,                        # Minimum 5bps edge
            ultra_signal.edge_metrics.net_alpha > 0   # Positive net alpha
        ]
        
        return all(criteria)
    
    def _integrate_signals(self, base_signals: Dict, ultra_signal: OptimalSignal) -> Dict[str, float]:
        """信号統合処理"""
        
        # Extract signals
        master_signal = base_signals.get('master_signal', {})
        base_signal_strength = master_signal.get('signal', 0.0)
        base_confidence = master_signal.get('confidence', 0.0)
        
        ultra_signal_strength = ultra_signal.signal_strength
        ultra_confidence = ultra_signal.confidence
        
        # Confidence-weighted integration
        total_confidence = base_confidence + ultra_confidence
        
        if total_confidence > 0:
            integrated_strength = (
                base_signal_strength * base_confidence * self.integration_config['base_engine_weight'] +
                ultra_signal_strength * ultra_confidence * self.integration_config['ultra_optimizer_weight']
            ) / total_confidence
            
            integrated_confidence = (base_confidence + ultra_confidence) / 2
        else:
            integrated_strength = 0.0
            integrated_confidence = 0.0
        
        # Signal direction consensus bonus
        if np.sign(base_signal_strength) == np.sign(ultra_signal_strength) and abs(integrated_strength) > 0.1:
            consensus_bonus = 1.2
        else:
            consensus_bonus = 0.8
        
        integrated_strength *= consensus_bonus
        integrated_confidence *= consensus_bonus
        
        return {
            'signal_strength': np.clip(integrated_strength, -1.0, 1.0),
            'confidence': min(integrated_confidence, 1.0),
            'base_component': base_signal_strength,
            'ultra_component': ultra_signal_strength,
            'consensus_bonus': consensus_bonus
        }
    
    def _apply_risk_optimization(self, integrated_signal: Dict, current_price: float, 
                               market_data: Dict) -> Dict[str, float]:
        """リスク最適化適用"""
        
        signal_strength = integrated_signal['signal_strength']
        confidence = integrated_signal['confidence']
        
        # Volatility estimation for risk scaling
        prices = market_data.get('price_series', [])
        if len(prices) >= 20:
            returns = np.diff(prices[-20:]) / prices[-20:-1]
            current_volatility = np.std(returns) * np.sqrt(1440)  # Daily volatility
        else:
            current_volatility = 0.01  # Default 1% daily volatility
        
        # Risk-adjusted position sizing
        base_position_size = abs(signal_strength) * confidence
        
        # Volatility adjustment
        vol_adjustment = min(0.01 / current_volatility, 1.5) if current_volatility > 0 else 1.0
        
        risk_adjusted_size = base_position_size * vol_adjustment
        risk_adjusted_size = min(risk_adjusted_size, self.integration_config['maximum_position_size'])
        
        # Risk-adjusted signal strength
        risk_adjusted_strength = np.sign(signal_strength) * risk_adjusted_size
        
        return {
            'risk_adjusted_strength': risk_adjusted_strength,
            'risk_adjusted_size': risk_adjusted_size,
            'volatility_adjustment': vol_adjustment,
            'current_volatility': current_volatility
        }
    
    def _calculate_execution_parameters(self, risk_adjusted_signal: Dict, 
                                      current_price: float, market_data: Dict) -> Dict[str, float]:
        """実行パラメータ計算"""
        
        signal_strength = risk_adjusted_signal['risk_adjusted_strength']
        position_size = risk_adjusted_signal['risk_adjusted_size']
        volatility = risk_adjusted_signal['current_volatility']
        
        # Entry price (with slippage consideration)
        if signal_strength > 0:  # Buy signal
            slippage_bps = 0.5  # 0.5 pips slippage for buy
            entry_price = current_price * (1 + slippage_bps / 10000)
        elif signal_strength < 0:  # Sell signal
            slippage_bps = 0.5
            entry_price = current_price * (1 - slippage_bps / 10000)
        else:
            entry_price = current_price
        
        # Stop loss (volatility-based)
        stop_loss_distance = max(volatility * 0.5, 0.0015)  # Min 15bps stop
        
        if signal_strength > 0:
            stop_loss_price = entry_price * (1 - stop_loss_distance)
        else:
            stop_loss_price = entry_price * (1 + stop_loss_distance)
        
        # Take profit (risk-reward ratio based)
        risk_reward_ratio = 2.0  # 2:1 risk-reward
        take_profit_distance = stop_loss_distance * risk_reward_ratio
        
        if signal_strength > 0:
            take_profit_price = entry_price * (1 + take_profit_distance)
        else:
            take_profit_price = entry_price * (1 - take_profit_distance)
        
        # Optimal holding period (based on signal decay)
        base_holding_minutes = 30  # Base 30 minutes
        confidence_adjustment = abs(signal_strength) * 2  # Higher signal = longer hold
        optimal_holding = int(base_holding_minutes * (1 + confidence_adjustment))
        optimal_holding = min(max(optimal_holding, 5), 120)  # 5-120 minutes range
        
        return {
            'entry_price': entry_price,
            'stop_loss_price': stop_loss_price,
            'take_profit_price': take_profit_price,
            'holding_period_minutes': optimal_holding,
            'stop_loss_distance': stop_loss_distance,
            'take_profit_distance': take_profit_distance
        }
    
    def _build_integrated_signal(self, risk_adjusted_signal: Dict, execution_params: Dict,
                               base_signals: Dict, ultra_signal: OptimalSignal,
                               current_price: float, market_data: Dict) -> IntegratedAlphaSignal:
        """統合信号組み立て"""
        
        # Extract timeframe components from base signals
        timeframe_signals = base_signals.get('timeframe_signals', {})
        timeframe_components = {
            tf: sig.get('signal', 0.0) for tf, sig in timeframe_signals.items()
        }
        
        # Extract strategy components from ultra signal
        strategy_components = {
            'microstructure': ultra_signal.edge_metrics.expected_return * 0.3,
            'statistical_arbitrage': ultra_signal.edge_metrics.expected_return * 0.25,
            'momentum': ultra_signal.edge_metrics.expected_return * 0.25,
            'mean_reversion': ultra_signal.edge_metrics.expected_return * 0.2
        }
        
        # Market regime
        market_regime = base_signals.get('regime_state', 'unknown')
        
        # Signal expiry (shorter of ultra signal expiry and holding period)
        signal_expiry = min(
            datetime.now() + timedelta(minutes=execution_params['holding_period_minutes']),
            datetime.now() + timedelta(hours=2)  # Max 2 hours
        )
        
        return IntegratedAlphaSignal(
            # Core signal
            signal_strength=risk_adjusted_signal['risk_adjusted_strength'],
            confidence=min(ultra_signal.confidence * 1.1, 1.0),
            
            # Edge data
            expected_edge_bps=ultra_signal.expected_edge * 100,  # Convert to bps
            optimal_position_size=risk_adjusted_signal['risk_adjusted_size'],
            risk_adjusted_return=ultra_signal.edge_metrics.net_alpha,
            
            # Execution parameters
            entry_price=execution_params['entry_price'],
            stop_loss_price=execution_params['stop_loss_price'],
            take_profit_price=execution_params['take_profit_price'],
            holding_period_minutes=execution_params['holding_period_minutes'],
            
            # Strategy breakdown
            timeframe_components=timeframe_components,
            strategy_components=strategy_components,
            
            # Risk metrics
            sharpe_estimate=ultra_signal.edge_metrics.sharpe_ratio,
            max_drawdown_estimate=ultra_signal.edge_metrics.max_drawdown,
            
            # Meta data
            timestamp=datetime.now(),
            expiry=signal_expiry,
            market_regime=market_regime
        )
    
    def _convert_to_dataframe(self, market_data: Dict) -> pd.DataFrame:
        """市場データをDataFrame形式に変換"""
        
        prices = market_data.get('price_series', [])
        volumes = market_data.get('volume_series', [])
        highs = market_data.get('high_series', [])
        lows = market_data.get('low_series', [])
        
        # Ensure all series have same length
        min_length = min(len(prices), len(volumes), len(highs), len(lows)) if all([prices, volumes, highs, lows]) else len(prices)
        
        if min_length == 0:
            return pd.DataFrame()
        
        df_data = {
            'close': prices[-min_length:],
            'volume': volumes[-min_length:] if volumes else [1.0] * min_length,
            'high': highs[-min_length:] if highs else prices[-min_length:],
            'low': lows[-min_length:] if lows else prices[-min_length:]
        }
        
        df = pd.DataFrame(df_data)
        df.index = pd.date_range(end=datetime.now(), periods=len(df), freq='1min')
        
        return df
    
    def _update_performance_tracking(self, signal: IntegratedAlphaSignal):
        """パフォーマンス追跡更新"""
        
        self.performance_metrics['total_signals_generated'] += 1
        self.performance_metrics['cumulative_edge'] += signal.expected_edge_bps
        
        # Update average holding period
        total_signals = self.performance_metrics['total_signals_generated']
        current_avg = self.performance_metrics['average_holding_period']
        new_avg = ((current_avg * (total_signals - 1)) + signal.holding_period_minutes) / total_signals
        self.performance_metrics['average_holding_period'] = new_avg
        
        # Update best Sharpe
        if signal.sharpe_estimate > self.performance_metrics['best_sharpe_achieved']:
            self.performance_metrics['best_sharpe_achieved'] = signal.sharpe_estimate
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """パフォーマンス要約取得"""
        
        total_signals = self.performance_metrics['total_signals_generated']
        
        return {
            'total_signals': total_signals,
            'average_edge_bps': self.performance_metrics['cumulative_edge'] / max(total_signals, 1),
            'average_holding_period_minutes': self.performance_metrics['average_holding_period'],
            'best_sharpe_achieved': self.performance_metrics['best_sharpe_achieved'],
            'signal_frequency_per_hour': total_signals / max(1, total_signals / 60),  # Rough estimate
            'estimated_daily_alpha_bps': (self.performance_metrics['cumulative_edge'] / max(total_signals, 1)) * 24
        }

# Global integration engine instance
integration_engine = AlphaIntegrationEngine()