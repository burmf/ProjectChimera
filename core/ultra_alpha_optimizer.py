"""
Ultra Alpha Optimizer - Maximum Edge Extraction Engine
超高度α最大化エンジン - エッジ抽出特化

Philosophy: 短期テクニカル重視でα最大化
- Microstructure Edge: Bid-Ask dynamics, Order flow imbalance
- Mean Reversion Edge: Statistical arbitrage opportunities  
- Momentum Edge: Breakout confirmation with volume
- Regime Edge: Dynamic strategy switching based on market state
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

@dataclass
class EdgeMetrics:
    """エッジ計測データクラス"""
    expected_return: float      # 期待収益率
    hit_ratio: float           # 勝率
    profit_loss_ratio: float   # 損益比率
    sharpe_ratio: float        # シャープレシオ
    max_drawdown: float        # 最大ドローダウン
    edge_decay_rate: float     # エッジ減衰速度
    transaction_costs: float   # 取引コスト
    net_alpha: float          # 純α (コスト後)

@dataclass  
class OptimalSignal:
    """最適化済み信号"""
    signal_strength: float     # -1.0 to 1.0
    confidence: float         # 0.0 to 1.0  
    expected_edge: float      # 期待エッジ (%)
    optimal_holding_period: int  # 最適保有期間 (minutes)
    risk_adjusted_size: float # リスク調整ポジションサイズ
    stop_loss: float         # 最適ストップロス
    take_profit: float       # 最適テイクプロフィット
    edge_metrics: EdgeMetrics

class UltraAlphaOptimizer:
    """超高度α最大化エンジン"""
    
    def __init__(self):
        """初期化 - エッジ最大化パラメータ設定"""
        
        # 取引コスト設定 (USD/JPY)
        self.transaction_costs = {
            'spread_bps': 0.2,        # 0.2 pips spread
            'slippage_bps': 0.1,      # 0.1 pips slippage  
            'commission_bps': 0.05,   # 0.05 pips commission
            'funding_rate_daily': 0.0001  # Daily funding cost
        }
        
        # エッジ最適化ターゲット
        self.optimization_targets = {
            'minimum_sharpe': 1.5,       # 最小シャープレシオ
            'minimum_hit_ratio': 0.52,   # 最小勝率 52%
            'maximum_drawdown': 0.03,    # 最大ドローダウン 3%
            'minimum_net_alpha': 0.001,  # 最小純α 0.1%
            'target_holding_minutes': 15 # 目標保有期間 15分
        }
        
        # マイクロストラクチャー・パラメータ
        self.microstructure_params = {
            'order_flow_lookback': 10,    # オーダーフロー分析期間
            'volume_profile_bins': 20,    # 価格帯別出来高分析
            'bid_ask_sensitivity': 2.0,   # Bid-Ask感度係数
            'liquidity_threshold': 0.8    # 流動性閾値
        }
        
        self.logger = logging.getLogger(__name__)

    def optimize_short_term_alpha(self, market_data: Dict) -> OptimalSignal:
        """
        短期α最適化 - Ultra-Think Level
        
        Args:
            market_data: 高頻度市場データ (1分足、ティック、オーダーブック)
            
        Returns:
            最適化済み取引信号
        """
        
        # 1. マイクロストラクチャー・エッジ分析
        microstructure_edge = self._analyze_microstructure_edge(market_data)
        
        # 2. 統計的裁定機会の検出
        statistical_arbitrage_edge = self._detect_statistical_arbitrage(market_data)
        
        # 3. モメンタム・ブレイクアウトエッジ
        momentum_edge = self._analyze_momentum_edge(market_data)
        
        # 4. 平均回帰エッジ
        mean_reversion_edge = self._analyze_mean_reversion_edge(market_data)
        
        # 5. レジーム適応型重み最適化
        optimal_weights = self._optimize_dynamic_weights(
            microstructure_edge, statistical_arbitrage_edge, 
            momentum_edge, mean_reversion_edge, market_data
        )
        
        # 6. エッジ統合と最適化
        raw_signal = (
            microstructure_edge['signal'] * optimal_weights['microstructure'] +
            statistical_arbitrage_edge['signal'] * optimal_weights['statistical'] +
            momentum_edge['signal'] * optimal_weights['momentum'] +
            mean_reversion_edge['signal'] * optimal_weights['mean_reversion']
        )
        
        # 7. 取引コスト調整とエッジ計算
        edge_metrics = self._calculate_comprehensive_edge_metrics(
            raw_signal, market_data, optimal_weights
        )
        
        # 8. 最適保有期間・ポジションサイズ計算
        optimal_params = self._optimize_execution_parameters(edge_metrics, market_data)
        
        return OptimalSignal(
            signal_strength=np.clip(raw_signal, -1.0, 1.0),
            confidence=edge_metrics.hit_ratio,
            expected_edge=edge_metrics.net_alpha * 100,  # Percentage
            optimal_holding_period=optimal_params['holding_period'],
            risk_adjusted_size=optimal_params['position_size'],
            stop_loss=optimal_params['stop_loss'],
            take_profit=optimal_params['take_profit'],
            edge_metrics=edge_metrics
        )

    def _analyze_microstructure_edge(self, market_data: Dict) -> Dict:
        """
        マイクロストラクチャー・エッジ分析
        
        Focus: Bid-Ask dynamics, Order flow imbalance, Volume clustering
        """
        
        prices = market_data.get('price_series', [])
        volumes = market_data.get('volume_series', [])
        highs = market_data.get('high_series', [])
        lows = market_data.get('low_series', [])
        
        if len(prices) < 20:
            return {'signal': 0.0, 'confidence': 0.0, 'edge': 0.0}
        
        signals = {}
        
        # 1. Bid-Ask Spread Proxy Analysis
        # High-Low ranges as spread proxy
        spreads = [(h - l) / ((h + l) / 2) * 10000 for h, l in zip(highs[-10:], lows[-10:])]  # in bps
        avg_spread = np.mean(spreads) if spreads else 1.0
        current_spread = spreads[-1] if spreads else avg_spread
        
        # Spread compression = liquidity increase = continuation signal  
        # Spread expansion = liquidity decrease = reversal signal
        spread_signal = -np.tanh((current_spread - avg_spread) / avg_spread * 3) if avg_spread > 0 else 0
        signals['spread_dynamics'] = spread_signal * 0.6
        
        # 2. Volume Profile Analysis (Price-Volume clustering)
        if len(volumes) >= 20:
            # Volume-Weighted Average Price deviation
            vwap = sum(p * v for p, v in zip(prices[-20:], volumes[-20:])) / sum(volumes[-20:])
            current_price = prices[-1]
            vwap_deviation = (current_price - vwap) / vwap if vwap > 0 else 0
            
            # Mean reversion signal when price deviates significantly from VWAP
            signals['vwap_reversion'] = -np.tanh(vwap_deviation * 50) * 0.7
        else:
            signals['vwap_reversion'] = 0.0
        
        # 3. Order Flow Imbalance Proxy
        # Using price-volume correlation as order flow proxy
        if len(prices) >= 10 and len(volumes) >= 10:
            price_changes = np.diff(prices[-11:])
            volume_changes = volumes[-10:]
            
            # Correlation between price moves and volume
            if len(price_changes) == len(volume_changes) and np.std(price_changes) > 0:
                order_flow_correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
                if not np.isnan(order_flow_correlation):
                    # Strong correlation = momentum, weak = mean reversion
                    signals['order_flow'] = np.tanh(order_flow_correlation * 2) * 0.5
                else:
                    signals['order_flow'] = 0.0
            else:
                signals['order_flow'] = 0.0
        else:
            signals['order_flow'] = 0.0
        
        # 4. Tick-level momentum (using minute-level as proxy)
        if len(prices) >= 5:
            recent_momentum = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] > 0 else 0
            # Short-term momentum confirmation
            signals['tick_momentum'] = np.tanh(recent_momentum * 100) * 0.4
        else:
            signals['tick_momentum'] = 0.0
        
        # 5. Liquidity stress indicator
        if len(volumes) >= 10:
            volume_volatility = np.std(volumes[-10:]) / np.mean(volumes[-10:]) if np.mean(volumes[-10:]) > 0 else 0
            # High volume volatility = liquidity stress = mean reversion
            signals['liquidity_stress'] = -np.tanh(volume_volatility * 2) * 0.3
        else:
            signals['liquidity_stress'] = 0.0
        
        # Weighted microstructure signal
        total_signal = sum(signals.values())
        confidence = min(abs(total_signal) * 1.5, 1.0)
        
        # Expected edge calculation (simplified)
        hit_ratio = 0.48 + confidence * 0.08  # 48-56% depending on confidence
        avg_win = 0.0015  # 1.5 bps average win
        avg_loss = 0.0012  # 1.2 bps average loss
        expected_edge = hit_ratio * avg_win - (1 - hit_ratio) * avg_loss
        
        return {
            'signal': np.tanh(total_signal),
            'confidence': confidence,
            'edge': expected_edge,
            'components': signals,
            'hit_ratio': hit_ratio
        }

    def _detect_statistical_arbitrage(self, market_data: Dict) -> Dict:
        """
        統計的裁定機会検出
        
        Focus: Z-score reversion, cointegration deviations, pattern completion
        """
        
        prices = market_data.get('price_series', [])
        if len(prices) < 50:
            return {'signal': 0.0, 'confidence': 0.0, 'edge': 0.0}
        
        signals = {}
        
        # 1. Multi-timeframe Z-score mean reversion
        for period, weight in [(10, 0.4), (20, 0.35), (50, 0.25)]:
            if len(prices) >= period:
                recent_prices = prices[-period:]
                mean_price = np.mean(recent_prices)
                std_price = np.std(recent_prices)
                
                if std_price > 0:
                    z_score = (prices[-1] - mean_price) / std_price
                    # Mean reversion signal
                    reversion_signal = -np.tanh(z_score / 2) * weight
                    signals[f'z_revert_{period}'] = reversion_signal
        
        # 2. Bollinger Band statistical arbitrage
        if len(prices) >= 20:
            bb_period = 20
            bb_std = 2.0
            sma = np.mean(prices[-bb_period:])
            std = np.std(prices[-bb_period:])
            
            upper_band = sma + bb_std * std
            lower_band = sma - bb_std * std
            current_price = prices[-1]
            
            # Statistical arbitrage at band extremes
            if current_price > upper_band:
                signals['bb_arbitrage'] = -0.8  # Strong sell signal
            elif current_price < lower_band:
                signals['bb_arbitrage'] = 0.8   # Strong buy signal
            else:
                # Linear signal within bands
                band_position = (current_price - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5
                signals['bb_arbitrage'] = (0.5 - band_position) * 1.6  # Contrarian
        
        # 3. Autocorrelation reversion
        if len(prices) >= 30:
            returns = np.diff(prices[-30:]) / prices[-30:-1]
            returns = returns[~np.isnan(returns)]
            
            if len(returns) >= 10:
                # Serial correlation test
                autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                if not np.isnan(autocorr):
                    # Negative autocorr = momentum, positive = mean reversion opportunity
                    signals['autocorr_reversion'] = np.tanh(autocorr * 3) * 0.5
        
        # 4. Support/Resistance statistical bounce
        if len(prices) >= 50:
            # Find recent significant levels
            highs = [max(prices[i:i+5]) for i in range(len(prices)-5)]
            lows = [min(prices[i:i+5]) for i in range(len(prices)-5)]
            
            # Resistance level (recent high)
            resistance = max(highs[-10:]) if len(highs) >= 10 else prices[-1]
            support = min(lows[-10:]) if len(lows) >= 10 else prices[-1]
            current_price = prices[-1]
            
            # Distance-based bounce probability
            resistance_distance = abs(current_price - resistance) / resistance if resistance > 0 else 1
            support_distance = abs(current_price - support) / support if support > 0 else 1
            
            if resistance_distance < 0.001:  # Within 0.1% of resistance
                signals['level_bounce'] = -0.6  # Resistance bounce down
            elif support_distance < 0.001:  # Within 0.1% of support
                signals['level_bounce'] = 0.6   # Support bounce up
            else:
                signals['level_bounce'] = 0.0
        
        # Aggregate statistical arbitrage signal
        total_signal = sum(signals.values())
        confidence = min(abs(total_signal) * 1.2, 1.0)
        
        # Higher hit ratio for statistical arbitrage
        hit_ratio = 0.52 + confidence * 0.06  # 52-58%
        avg_win = 0.0018
        avg_loss = 0.0010
        expected_edge = hit_ratio * avg_win - (1 - hit_ratio) * avg_loss
        
        return {
            'signal': np.tanh(total_signal),
            'confidence': confidence,
            'edge': expected_edge,
            'components': signals,
            'hit_ratio': hit_ratio
        }

    def _analyze_momentum_edge(self, market_data: Dict) -> Dict:
        """
        モメンタム・ブレイクアウトエッジ分析
        
        Focus: Volume-confirmed breakouts, multi-timeframe alignment
        """
        
        prices = market_data.get('price_series', [])
        volumes = market_data.get('volume_series', [])
        highs = market_data.get('high_series', [])
        lows = market_data.get('low_series', [])
        
        if len(prices) < 30:
            return {'signal': 0.0, 'confidence': 0.0, 'edge': 0.0}
        
        signals = {}
        
        # 1. Volume-confirmed price breakout
        if len(highs) >= 20 and len(lows) >= 20:
            # Recent range
            recent_high = max(highs[-20:])
            recent_low = min(lows[-20:])
            current_price = prices[-1]
            
            # Breakout detection
            upper_breakout = current_price > recent_high * 1.0001  # 0.01% buffer
            lower_breakout = current_price < recent_low * 0.9999
            
            # Volume confirmation
            if len(volumes) >= 10:
                avg_volume = np.mean(volumes[-10:])
                current_volume = volumes[-1]
                volume_confirmation = min(current_volume / avg_volume, 2.0) if avg_volume > 0 else 1.0
                
                if upper_breakout and volume_confirmation > 1.2:
                    signals['volume_breakout'] = 0.8 * volume_confirmation
                elif lower_breakout and volume_confirmation > 1.2:
                    signals['volume_breakout'] = -0.8 * volume_confirmation
                else:
                    signals['volume_breakout'] = 0.0
            else:
                signals['volume_breakout'] = 0.0
        
        # 2. Multi-timeframe momentum alignment
        momentum_signals = []
        for period in [3, 5, 10]:
            if len(prices) > period:
                momentum = (prices[-1] - prices[-period]) / prices[-period] if prices[-period] > 0 else 0
                momentum_signals.append(np.tanh(momentum * 100))
        
        if momentum_signals:
            # Bonus for alignment across timeframes
            alignment_bonus = 1.0 if all(s > 0 for s in momentum_signals) or all(s < 0 for s in momentum_signals) else 0.6
            signals['momentum_alignment'] = np.mean(momentum_signals) * alignment_bonus
        
        # 3. Volatility expansion momentum
        if len(prices) >= 20:
            returns = np.diff(prices[-21:]) / prices[-21:-1]
            current_vol = np.std(returns[-5:]) if len(returns) >= 5 else 0
            baseline_vol = np.std(returns[-20:]) if len(returns) >= 20 else current_vol
            
            if baseline_vol > 0:
                vol_expansion = current_vol / baseline_vol
                recent_return = returns[-1] if len(returns) > 0 else 0
                
                # Momentum when volatility expanding
                if vol_expansion > 1.3:
                    signals['vol_momentum'] = np.tanh(recent_return * 100) * min(vol_expansion / 2, 1.0)
                else:
                    signals['vol_momentum'] = 0.0
        
        # 4. RSI momentum (trend following in extreme zones)
        if len(prices) >= 14:
            # Simple RSI calculation
            returns = np.diff(prices[-15:]) / prices[-15:-1]
            gains = [r if r > 0 else 0 for r in returns]
            losses = [-r if r < 0 else 0 for r in returns]
            
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                
                # Trend following in extreme zones
                if rsi > 75:  # Strong uptrend
                    signals['rsi_momentum'] = 0.5
                elif rsi < 25:  # Strong downtrend
                    signals['rsi_momentum'] = -0.5
                else:
                    signals['rsi_momentum'] = 0.0
        
        # Aggregate momentum signal
        total_signal = sum(signals.values())
        confidence = min(abs(total_signal) * 1.1, 1.0)
        
        # Momentum typically has lower hit ratio but higher payoff
        hit_ratio = 0.45 + confidence * 0.08  # 45-53%
        avg_win = 0.0025  # Higher average win
        avg_loss = 0.0015  # Higher average loss
        expected_edge = hit_ratio * avg_win - (1 - hit_ratio) * avg_loss
        
        return {
            'signal': np.tanh(total_signal),
            'confidence': confidence,
            'edge': expected_edge,
            'components': signals,
            'hit_ratio': hit_ratio
        }

    def _analyze_mean_reversion_edge(self, market_data: Dict) -> Dict:
        """
        平均回帰エッジ分析
        
        Focus: Extreme deviation reversals, overshooting corrections
        """
        
        prices = market_data.get('price_series', [])
        if len(prices) < 20:
            return {'signal': 0.0, 'confidence': 0.0, 'edge': 0.0}
        
        signals = {}
        
        # 1. Price overshooting detection
        for period in [5, 10, 20]:
            if len(prices) > period:
                sma = np.mean(prices[-period:])
                deviation = (prices[-1] - sma) / sma if sma > 0 else 0
                
                # Mean reversion signal increases with deviation
                reversion_strength = -np.tanh(deviation * 50) * (0.5 if period == 5 else 0.3 if period == 10 else 0.2)
                signals[f'sma_reversion_{period}'] = reversion_strength
        
        # 2. Extreme move fade
        if len(prices) >= 5:
            recent_moves = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(-4, 0)]
            recent_moves = [m for m in recent_moves if not np.isnan(m)]
            
            if recent_moves:
                # Large moves tend to partially reverse
                latest_move = recent_moves[-1]
                if abs(latest_move) > 0.002:  # > 0.2% move
                    fade_strength = -np.sign(latest_move) * min(abs(latest_move) * 200, 1.0)
                    signals['extreme_fade'] = fade_strength * 0.6
        
        # 3. Volatility mean reversion
        if len(prices) >= 30:
            returns = np.diff(prices[-31:]) / prices[-31:-1]
            current_vol = np.std(returns[-5:]) if len(returns) >= 5 else 0
            long_vol = np.std(returns) if len(returns) > 0 else current_vol
            
            if long_vol > 0:
                vol_ratio = current_vol / long_vol
                
                # When volatility is extreme, expect reversion
                if vol_ratio > 2.0:  # High volatility
                    latest_return = returns[-1] if len(returns) > 0 else 0
                    signals['vol_reversion'] = -np.tanh(latest_return * 50) * 0.7
                elif vol_ratio < 0.5:  # Low volatility - expect breakout
                    signals['vol_reversion'] = 0.0  # No mean reversion signal
        
        # 4. Intraday patterns (placeholder for hour-of-day effects)
        # In real implementation, this would use hour-specific patterns
        current_hour = datetime.now().hour
        if 8 <= current_hour <= 10 or 14 <= current_hour <= 16:  # Active trading hours
            # Slightly favor mean reversion during active hours
            signals['intraday_pattern'] = sum(signals.values()) * 0.1
        else:
            signals['intraday_pattern'] = 0.0
        
        # Aggregate mean reversion signal
        total_signal = sum(signals.values())
        confidence = min(abs(total_signal) * 1.3, 1.0)
        
        # Mean reversion typically has higher hit ratio
        hit_ratio = 0.55 + confidence * 0.05  # 55-60%
        avg_win = 0.0012  # Smaller average win
        avg_loss = 0.0015  # Smaller average loss
        expected_edge = hit_ratio * avg_win - (1 - hit_ratio) * avg_loss
        
        return {
            'signal': np.tanh(total_signal),
            'confidence': confidence,
            'edge': expected_edge,
            'components': signals,
            'hit_ratio': hit_ratio
        }

    def _optimize_dynamic_weights(self, microstructure_edge: Dict, statistical_edge: Dict, 
                                 momentum_edge: Dict, mean_reversion_edge: Dict, 
                                 market_data: Dict) -> Dict[str, float]:
        """
        動的重み最適化 - Kelly Criterion & Risk-Adjusted Returns
        """
        
        edges = {
            'microstructure': microstructure_edge,
            'statistical': statistical_edge,
            'momentum': momentum_edge,
            'mean_reversion': mean_reversion_edge
        }
        
        # Kelly weight calculation for each strategy
        kelly_weights = {}
        total_kelly = 0
        
        for strategy, edge_data in edges.items():
            hit_ratio = edge_data.get('hit_ratio', 0.5)
            edge = edge_data.get('edge', 0.0)
            
            if edge > 0 and hit_ratio > 0.5:
                # Simplified Kelly criterion
                avg_win = edge / hit_ratio if hit_ratio > 0 else 0
                avg_loss = edge / (hit_ratio - 1) if hit_ratio < 1 else avg_win
                
                if avg_loss != 0:
                    kelly_fraction = (hit_ratio * avg_win - (1 - hit_ratio) * abs(avg_loss)) / abs(avg_loss)
                    kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
                else:
                    kelly_fraction = 0
            else:
                kelly_fraction = 0
            
            kelly_weights[strategy] = kelly_fraction
            total_kelly += kelly_fraction
        
        # Normalize weights
        if total_kelly > 0:
            optimized_weights = {k: v / total_kelly for k, v in kelly_weights.items()}
        else:
            # Fallback to equal weights
            optimized_weights = {k: 0.25 for k in kelly_weights.keys()}
        
        # Market regime adjustment
        regime_adjustments = self._get_regime_weight_adjustments(market_data)
        
        for strategy in optimized_weights:
            adjustment = regime_adjustments.get(strategy, 1.0)
            optimized_weights[strategy] *= adjustment
        
        # Re-normalize after regime adjustment
        total_weight = sum(optimized_weights.values())
        if total_weight > 0:
            optimized_weights = {k: v / total_weight for k, v in optimized_weights.items()}
        
        return optimized_weights

    def _get_regime_weight_adjustments(self, market_data: Dict) -> Dict[str, float]:
        """レジーム別重み調整係数"""
        
        prices = market_data.get('price_series', [])
        if len(prices) < 50:
            return {'microstructure': 1.0, 'statistical': 1.0, 'momentum': 1.0, 'mean_reversion': 1.0}
        
        # Volatility regime
        returns = np.diff(prices[-21:]) / prices[-21:-1]
        current_vol = np.std(returns[-5:]) if len(returns) >= 5 else 0
        baseline_vol = np.std(returns) if len(returns) > 0 else current_vol
        
        vol_ratio = current_vol / baseline_vol if baseline_vol > 0 else 1.0
        
        if vol_ratio > 1.5:  # High volatility regime
            return {
                'microstructure': 1.3,  # Microstructure edges work well
                'statistical': 0.7,     # Statistical arbitrage less reliable
                'momentum': 1.2,        # Momentum strategies work
                'mean_reversion': 1.1   # Some mean reversion
            }
        elif vol_ratio < 0.7:  # Low volatility regime
            return {
                'microstructure': 0.8,
                'statistical': 1.3,     # Statistical arbitrage more reliable
                'momentum': 0.6,        # Less momentum
                'mean_reversion': 1.2   # More mean reversion
            }
        else:  # Normal regime
            return {
                'microstructure': 1.0,
                'statistical': 1.0,
                'momentum': 1.0,
                'mean_reversion': 1.0
            }

    def _calculate_comprehensive_edge_metrics(self, signal: float, market_data: Dict, 
                                           weights: Dict[str, float]) -> EdgeMetrics:
        """包括的エッジ計測"""
        
        # Expected returns based on signal strength and historical performance
        base_expected_return = abs(signal) * 0.0015  # 15 bps for full strength signal
        
        # Hit ratio estimation
        confidence = min(abs(signal) * 1.2, 1.0)
        base_hit_ratio = 0.52 + confidence * 0.06  # 52-58%
        
        # Transaction costs
        total_cost_bps = (
            self.transaction_costs['spread_bps'] + 
            self.transaction_costs['slippage_bps'] + 
            self.transaction_costs['commission_bps']
        )
        transaction_cost_rate = total_cost_bps / 10000  # Convert to decimal
        
        # Net expected return after costs
        net_expected_return = base_expected_return - transaction_cost_rate
        
        # Profit/Loss ratio estimation
        avg_win = net_expected_return / base_hit_ratio if base_hit_ratio > 0 else 0
        avg_loss = net_expected_return / (base_hit_ratio - 1) if base_hit_ratio < 1 else avg_win
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 1.0
        
        # Sharpe ratio estimation (simplified)
        expected_vol = 0.002  # 20 bps volatility assumption
        sharpe_ratio = net_expected_return / expected_vol if expected_vol > 0 else 0
        
        # Max drawdown estimation
        max_drawdown = expected_vol * 2.5  # Simplified estimate
        
        # Edge decay rate (how quickly edge deteriorates)
        edge_decay_rate = 0.05  # 5% per hour assumption
        
        return EdgeMetrics(
            expected_return=base_expected_return,
            hit_ratio=base_hit_ratio,
            profit_loss_ratio=profit_loss_ratio,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            edge_decay_rate=edge_decay_rate,
            transaction_costs=transaction_cost_rate,
            net_alpha=net_expected_return
        )

    def _optimize_execution_parameters(self, edge_metrics: EdgeMetrics, 
                                      market_data: Dict) -> Dict[str, float]:
        """実行パラメータ最適化"""
        
        # Optimal holding period based on edge decay
        decay_rate = edge_metrics.edge_decay_rate
        optimal_holding_minutes = -np.log(0.5) / decay_rate * 60  # Half-life in minutes
        optimal_holding_minutes = min(max(optimal_holding_minutes, 5), 60)  # 5-60 minutes
        
        # Kelly-based position sizing
        if edge_metrics.net_alpha > 0 and edge_metrics.hit_ratio > 0.5:
            win_size = edge_metrics.net_alpha / edge_metrics.hit_ratio
            loss_size = edge_metrics.net_alpha / (edge_metrics.hit_ratio - 1) if edge_metrics.hit_ratio < 1 else win_size
            
            kelly_fraction = (edge_metrics.hit_ratio * win_size - (1 - edge_metrics.hit_ratio) * abs(loss_size)) / abs(loss_size)
            kelly_fraction = max(0, min(kelly_fraction, 0.1))  # Cap at 10%
        else:
            kelly_fraction = 0
        
        position_size = kelly_fraction * 0.5  # Conservative Kelly
        
        # Optimal stop loss and take profit
        volatility = 0.002  # Estimated volatility
        
        # Stop loss: 1.5x volatility or 50% of expected edge
        stop_loss = min(1.5 * volatility, edge_metrics.net_alpha * 0.5)
        
        # Take profit: 2x expected edge or based on profit/loss ratio
        take_profit = max(edge_metrics.net_alpha * 2, stop_loss * edge_metrics.profit_loss_ratio)
        
        return {
            'holding_period': int(optimal_holding_minutes),
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }

# Global optimizer instance
ultra_optimizer = UltraAlphaOptimizer()