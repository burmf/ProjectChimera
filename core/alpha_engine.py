#!/usr/bin/env python3
"""
Alpha Engine - Time-Horizon Optimized Trading Strategy Framework
Implements multi-timeframe alpha generation with regime-aware signal fusion
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import talib
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AlphaEngine:
    """
    Multi-timeframe alpha generation engine optimized for different market regimes.
    
    Time Horizons:
    - Ultra-Short (1-5min): Mean reversion, microstructure signals
    - Short (5-60min): Technical momentum, volatility breakouts  
    - Medium (1-24hr): Factor signals, fundamental momentum
    - Long (1-7 days): Regime changes, macro factor rotations
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        
        # Alpha decay parameters by timeframe
        self.alpha_decay = {
            'ultra_short': 0.95,  # Very fast decay
            'short': 0.90,        # Fast decay
            'medium': 0.85,       # Moderate decay
            'long': 0.80          # Slow decay
        }
        
        # Signal confidence thresholds
        self.confidence_thresholds = {
            'ultra_short': 0.75,
            'short': 0.70,
            'medium': 0.65,
            'long': 0.60
        }
        
    def generate_multi_timeframe_signals(self, price_data: pd.DataFrame, 
                                       macro_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate alpha signals across multiple timeframes"""
        try:
            signals = {}
            
            # 1. Ultra-Short Timeframe (1-5min): Mean Reversion Focus
            signals['ultra_short'] = self._generate_ultra_short_alpha(price_data)
            
            # 2. Short Timeframe (5-60min): Technical Momentum
            signals['short'] = self._generate_short_alpha(price_data)
            
            # 3. Medium Timeframe (1-24hr): Factor Signals
            signals['medium'] = self._generate_medium_alpha(price_data, macro_data)
            
            # 4. Long Timeframe (1-7 days): Regime & Macro
            signals['long'] = self._generate_long_alpha(price_data, macro_data)
            
            # 5. Master Signal Fusion
            master_signal = self._fuse_timeframe_signals(signals, price_data)
            
            return {
                'timeframe_signals': signals,
                'master_signal': master_signal,
                'regime_state': self._detect_market_regime(price_data),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Multi-timeframe signal generation failed: {e}")
            return self._get_null_signals()
    
    def _generate_ultra_short_alpha(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Ultra-Short Timeframe Alpha (1-5min)
        Focus: Mean reversion, order flow imbalances, microstructure patterns
        """
        try:
            if len(df) < 50:
                return self._get_null_signal()
                
            # Microstructure indicators
            signals = {}
            
            # 1. Short-term mean reversion (Z-score)
            returns_1min = df['close'].pct_change()
            returns_zscore = (returns_1min - returns_1min.rolling(20).mean()) / returns_1min.rolling(20).std()
            
            # Mean reversion signal (contrarian)
            signals['mean_reversion'] = -np.tanh(returns_zscore.iloc[-1])  # Contrarian
            
            # 2. Bid-Ask spread proxy (volatility spikes)
            volatility = returns_1min.rolling(10).std()
            vol_zscore = (volatility - volatility.rolling(50).mean()) / volatility.rolling(50).std()
            signals['liquidity_stress'] = -np.tanh(vol_zscore.iloc[-1] * 0.5)  # Contrarian to vol spikes
            
            # 3. Support/Resistance bounce
            current_price = df['close'].iloc[-1]
            recent_high = df['high'].rolling(20).max().iloc[-1]
            recent_low = df['low'].rolling(20).min().iloc[-1]
            
            price_position = (current_price - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5
            
            # Mean reversion at extremes
            if price_position > 0.85:
                signals['level_reversion'] = -0.8  # Sell at resistance
            elif price_position < 0.15:
                signals['level_reversion'] = 0.8   # Buy at support
            else:
                signals['level_reversion'] = 0.0
            
            # 4. Volume-price divergence
            volume = df.get('volume', pd.Series([1] * len(df)))
            price_change = df['close'].pct_change()
            volume_change = volume.pct_change()
            
            # Volume-price correlation (last 20 periods)
            vol_price_corr = price_change.rolling(20).corr(volume_change).iloc[-1]
            if pd.notna(vol_price_corr):
                signals['volume_divergence'] = np.tanh(vol_price_corr * 2)
            else:
                signals['volume_divergence'] = 0.0
            
            # Aggregate ultra-short signal
            weights = {
                'mean_reversion': 0.4,
                'liquidity_stress': 0.2,
                'level_reversion': 0.3,
                'volume_divergence': 0.1
            }
            
            weighted_signal = sum(signals[key] * weights[key] for key in signals.keys())
            confidence = min(abs(weighted_signal) * 1.5, 1.0)
            
            return {
                'signal': np.tanh(weighted_signal),
                'confidence': confidence,
                'components': signals,
                'timeframe': 'ultra_short',
                'strategy': 'mean_reversion'
            }
            
        except Exception as e:
            self.logger.error(f"Ultra-short alpha generation failed: {e}")
            return self._get_null_signal()
    
    def _generate_short_alpha(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Short Timeframe Alpha (5-60min)
        Focus: Technical momentum, volatility breakouts, pattern recognition
        """
        try:
            if len(df) < 100:
                return self._get_null_signal()
                
            signals = {}
            
            # 1. Multi-timeframe momentum
            returns_5min = df['close'].pct_change(5)
            returns_15min = df['close'].pct_change(15)
            returns_60min = df['close'].pct_change(60)
            
            # Momentum consistency across timeframes
            momentum_signals = [
                np.tanh(returns_5min.iloc[-1] * 100),
                np.tanh(returns_15min.iloc[-1] * 50),
                np.tanh(returns_60min.iloc[-1] * 25)
            ]
            
            # Momentum alignment bonus
            momentum_alignment = 1.0 if all(s > 0 for s in momentum_signals) or all(s < 0 for s in momentum_signals) else 0.5
            signals['momentum'] = np.mean(momentum_signals) * momentum_alignment
            
            # 2. Volatility breakout
            returns = df['close'].pct_change()
            vol_20 = returns.rolling(20).std()
            vol_5 = returns.rolling(5).std()
            
            vol_expansion = (vol_5.iloc[-1] / vol_20.iloc[-1]) if vol_20.iloc[-1] > 0 else 1.0
            price_move = returns.iloc[-1]
            
            # Breakout signal (directional volatility)
            if vol_expansion > 1.5:  # Volatility expanding
                signals['volatility_breakout'] = np.tanh(price_move * 50) * min(vol_expansion / 2, 1.0)
            else:
                signals['volatility_breakout'] = 0.0
            
            # 3. RSI with dynamic thresholds
            rsi = talib.RSI(df['close'].values, timeperiod=14)
            if len(rsi) > 0 and not pd.isna(rsi[-1]):
                current_rsi = rsi[-1]
                
                # Adaptive RSI thresholds based on volatility
                vol_adj = min(vol_20.iloc[-1] * 1000, 0.3)  # Scale to reasonable range
                rsi_upper = 70 + vol_adj * 15  # Higher threshold in high vol
                rsi_lower = 30 - vol_adj * 15  # Lower threshold in high vol
                
                if current_rsi > rsi_upper:
                    signals['rsi_momentum'] = -0.6  # Overbought
                elif current_rsi < rsi_lower:
                    signals['rsi_momentum'] = 0.6   # Oversold
                else:
                    signals['rsi_momentum'] = (current_rsi - 50) / 50 * 0.3  # Mild momentum
            else:
                signals['rsi_momentum'] = 0.0
            
            # 4. MACD with histogram divergence
            macd, macd_signal, macd_hist = talib.MACD(df['close'].values)
            if len(macd_hist) > 5 and not pd.isna(macd_hist[-1]):
                # MACD histogram momentum
                hist_momentum = (macd_hist[-1] - macd_hist[-3]) if not pd.isna(macd_hist[-3]) else 0
                signals['macd_momentum'] = np.tanh(hist_momentum * 1000)
            else:
                signals['macd_momentum'] = 0.0
            
            # 5. Bollinger Band squeeze and expansion
            bb_upper, bb_middle, bb_lower = talib.BBANDS(df['close'].values, timeperiod=20)
            if len(bb_upper) > 0:
                bb_width = (bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
                bb_width_ma = pd.Series(bb_upper - bb_lower).rolling(10).mean().iloc[-1] / bb_middle[-1]
                
                # Band expansion after squeeze
                if bb_width > bb_width_ma * 1.2:  # Expanding
                    price_pos = (df['close'].iloc[-1] - bb_lower[-1]) / (bb_upper[-1] - bb_lower[-1])
                    signals['bb_expansion'] = (price_pos - 0.5) * 2 * 0.7  # Directional
                else:
                    signals['bb_expansion'] = 0.0
            else:
                signals['bb_expansion'] = 0.0
            
            # Aggregate short-term signal
            weights = {
                'momentum': 0.35,
                'volatility_breakout': 0.25,
                'rsi_momentum': 0.15,
                'macd_momentum': 0.15,
                'bb_expansion': 0.10
            }
            
            weighted_signal = sum(signals[key] * weights[key] for key in signals.keys())
            confidence = min(abs(weighted_signal) * 1.2, 1.0)
            
            return {
                'signal': np.tanh(weighted_signal),
                'confidence': confidence,
                'components': signals,
                'timeframe': 'short',
                'strategy': 'technical_momentum'
            }
            
        except Exception as e:
            self.logger.error(f"Short alpha generation failed: {e}")
            return self._get_null_signal()
    
    def _generate_medium_alpha(self, df: pd.DataFrame, macro_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Medium Timeframe Alpha (1-24hr)
        Focus: Factor signals, cross-asset momentum, fundamental indicators
        """
        try:
            if len(df) < 200:
                return self._get_null_signal()
                
            signals = {}
            
            # 1. Multi-timeframe trend alignment
            sma_10 = talib.SMA(df['close'].values, 10)
            sma_50 = talib.SMA(df['close'].values, 50)
            sma_200 = talib.SMA(df['close'].values, 200)
            
            if len(sma_200) > 0:
                current_price = df['close'].iloc[-1]
                trend_signals = [
                    1 if current_price > sma_10[-1] else -1,
                    1 if current_price > sma_50[-1] else -1,
                    1 if current_price > sma_200[-1] else -1
                ]
                
                # Trend strength based on alignment
                trend_strength = sum(trend_signals) / 3
                
                # Distance from key levels
                if sma_200[-1] > 0:
                    distance_200 = (current_price / sma_200[-1] - 1) * 100
                    signals['trend_momentum'] = np.tanh(distance_200 / 5) * abs(trend_strength)
                else:
                    signals['trend_momentum'] = 0.0
            else:
                signals['trend_momentum'] = 0.0
            
            # 2. ADX trend strength with directional bias
            adx = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            plus_di = talib.PLUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            minus_di = talib.MINUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            
            if len(adx) > 0 and not pd.isna(adx[-1]):
                trend_strength = min(adx[-1] / 50, 1.0)  # Normalize ADX
                if not pd.isna(plus_di[-1]) and not pd.isna(minus_di[-1]):
                    directional_bias = 1 if plus_di[-1] > minus_di[-1] else -1
                    signals['adx_trend'] = directional_bias * trend_strength * 0.7
                else:
                    signals['adx_trend'] = 0.0
            else:
                signals['adx_trend'] = 0.0
            
            # 3. Volume-based signals (if available)
            if 'volume' in df.columns:
                volume = df['volume']
                vol_sma = volume.rolling(20).mean()
                price_change = df['close'].pct_change()
                
                # Volume confirmation
                recent_volume = volume.iloc[-5:].mean()
                avg_volume = vol_sma.iloc[-1] if not pd.isna(vol_sma.iloc[-1]) else recent_volume
                
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
                price_momentum = price_change.iloc[-5:].mean()
                
                # Volume-confirmed momentum
                signals['volume_momentum'] = np.tanh(price_momentum * 50) * min(volume_ratio, 2.0) * 0.5
            else:
                signals['volume_momentum'] = 0.0
            
            # 4. Volatility regime detection
            returns = df['close'].pct_change()
            current_vol = returns.rolling(20).std().iloc[-1]
            long_vol = returns.rolling(100).std().iloc[-1]
            
            if pd.notna(current_vol) and pd.notna(long_vol) and long_vol > 0:
                vol_regime = current_vol / long_vol
                
                # Volatility momentum (trend following in high vol, mean reversion in low vol)
                if vol_regime > 1.3:  # High volatility - trend following
                    signals['vol_regime'] = signals.get('trend_momentum', 0) * 0.8
                elif vol_regime < 0.7:  # Low volatility - prepare for breakout
                    signals['vol_regime'] = signals.get('trend_momentum', 0) * 1.2
                else:
                    signals['vol_regime'] = 0.0
            else:
                signals['vol_regime'] = 0.0
            
            # 5. Macro factor integration (if available)
            if macro_data:
                macro_signal = self._process_macro_factors(macro_data)
                signals['macro_factor'] = macro_signal * 0.6
            else:
                signals['macro_factor'] = 0.0
            
            # Aggregate medium-term signal
            weights = {
                'trend_momentum': 0.30,
                'adx_trend': 0.25,
                'volume_momentum': 0.15,
                'vol_regime': 0.15,
                'macro_factor': 0.15
            }
            
            weighted_signal = sum(signals[key] * weights[key] for key in signals.keys())
            confidence = min(abs(weighted_signal) * 1.1, 1.0)
            
            return {
                'signal': np.tanh(weighted_signal),
                'confidence': confidence,
                'components': signals,
                'timeframe': 'medium',
                'strategy': 'factor_momentum'
            }
            
        except Exception as e:
            self.logger.error(f"Medium alpha generation failed: {e}")
            return self._get_null_signal()
    
    def _generate_long_alpha(self, df: pd.DataFrame, macro_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Long Timeframe Alpha (1-7 days)
        Focus: Regime changes, macro factor rotations, fundamental momentum
        """
        try:
            if len(df) < 500:
                return self._get_null_signal()
                
            signals = {}
            
            # 1. Regime change detection
            returns = df['close'].pct_change()
            
            # Volatility regime changes
            short_vol = returns.rolling(20).std()
            long_vol = returns.rolling(100).std()
            vol_ratio = short_vol / long_vol
            
            # Detect regime shifts (significant volatility changes)
            vol_change = vol_ratio.pct_change(5).iloc[-1]
            if pd.notna(vol_change):
                # Anticipate mean reversion after extreme regime changes
                if abs(vol_change) > 0.3:  # Significant regime change
                    signals['regime_shift'] = -np.tanh(vol_change * 2) * 0.6  # Contrarian
                else:
                    signals['regime_shift'] = 0.0
            else:
                signals['regime_shift'] = 0.0
            
            # 2. Long-term trend momentum (monthly)
            if len(df) >= 30:
                monthly_return = (df['close'].iloc[-1] / df['close'].iloc[-30] - 1)
                monthly_momentum = np.tanh(monthly_return * 10) * 0.7
                signals['monthly_momentum'] = monthly_momentum
            else:
                signals['monthly_momentum'] = 0.0
            
            # 3. Moving average convergence/divergence
            sma_50 = talib.SMA(df['close'].values, 50)
            sma_200 = talib.SMA(df['close'].values, 200)
            
            if len(sma_200) > 0 and not pd.isna(sma_200[-1]) and sma_200[-1] > 0:
                ma_ratio = sma_50[-1] / sma_200[-1] - 1
                signals['ma_divergence'] = np.tanh(ma_ratio * 20) * 0.8
            else:
                signals['ma_divergence'] = 0.0
            
            # 4. Correlation regime analysis (placeholder for multi-asset)
            # In real implementation, this would analyze USD/JPY vs other major pairs
            correlation_proxy = returns.rolling(50).corr(returns.shift(1)).iloc[-1]
            if pd.notna(correlation_proxy):
                # Mean reversion signal when autocorrelation is extreme
                signals['correlation_regime'] = -np.tanh(correlation_proxy * 3) * 0.4
            else:
                signals['correlation_regime'] = 0.0
            
            # 5. Seasonal and calendar effects
            current_date = df.index[-1] if hasattr(df.index[-1], 'month') else datetime.now()
            if hasattr(current_date, 'month'):
                # USD/JPY seasonal patterns (simplified)
                month = current_date.month
                seasonal_bias = {
                    1: 0.1, 2: -0.1, 3: 0.2, 4: 0.0, 5: -0.1, 6: 0.1,
                    7: 0.0, 8: -0.2, 9: 0.1, 10: 0.0, 11: 0.1, 12: 0.0
                }.get(month, 0.0)
                signals['seasonal'] = seasonal_bias
            else:
                signals['seasonal'] = 0.0
            
            # Aggregate long-term signal
            weights = {
                'regime_shift': 0.25,
                'monthly_momentum': 0.25,
                'ma_divergence': 0.20,
                'correlation_regime': 0.15,
                'seasonal': 0.15
            }
            
            weighted_signal = sum(signals[key] * weights[key] for key in signals.keys())
            confidence = min(abs(weighted_signal) * 0.9, 1.0)  # Conservative confidence for long-term
            
            return {
                'signal': np.tanh(weighted_signal),
                'confidence': confidence,
                'components': signals,
                'timeframe': 'long',
                'strategy': 'regime_momentum'
            }
            
        except Exception as e:
            self.logger.error(f"Long alpha generation failed: {e}")
            return self._get_null_signal()
    
    def _fuse_timeframe_signals(self, signals: Dict, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Master signal fusion with regime-aware weighting
        """
        try:
            # Detect current market regime
            regime = self._detect_market_regime(df)
            
            # Regime-based timeframe weights
            if regime == 'trending':
                weights = {'ultra_short': 0.10, 'short': 0.30, 'medium': 0.40, 'long': 0.20}
            elif regime == 'mean_reverting':
                weights = {'ultra_short': 0.40, 'short': 0.30, 'medium': 0.20, 'long': 0.10}
            elif regime == 'volatile':
                weights = {'ultra_short': 0.20, 'short': 0.50, 'medium': 0.20, 'long': 0.10}
            else:  # neutral
                weights = {'ultra_short': 0.20, 'short': 0.30, 'medium': 0.30, 'long': 0.20}
            
            # Weighted signal fusion
            master_signal = 0.0
            total_confidence = 0.0
            active_signals = 0
            
            for timeframe, weight in weights.items():
                if timeframe in signals and signals[timeframe].get('confidence', 0) > self.confidence_thresholds.get(timeframe, 0.6):
                    signal_value = signals[timeframe].get('signal', 0)
                    confidence = signals[timeframe].get('confidence', 0)
                    
                    # Apply alpha decay
                    decayed_signal = signal_value * (1 - self.alpha_decay.get(timeframe, 0.85))
                    
                    master_signal += decayed_signal * weight * confidence
                    total_confidence += confidence * weight
                    active_signals += 1
            
            # Normalize by active signals
            if active_signals > 0:
                master_signal = master_signal / max(total_confidence, 0.1)
                final_confidence = total_confidence / active_signals
            else:
                master_signal = 0.0
                final_confidence = 0.0
            
            return {
                'signal': np.tanh(master_signal),
                'confidence': min(final_confidence, 1.0),
                'regime': regime,
                'active_timeframes': active_signals,
                'signal_components': {tf: signals[tf].get('signal', 0) for tf in signals.keys()},
                'weights_used': weights
            }
            
        except Exception as e:
            self.logger.error(f"Signal fusion failed: {e}")
            return {'signal': 0.0, 'confidence': 0.0, 'regime': 'unknown'}
    
    def _detect_market_regime(self, df: pd.DataFrame) -> str:
        """
        Detect current market regime: trending, mean_reverting, volatile, neutral
        """
        try:
            if len(df) < 100:
                return 'unknown'
            
            returns = df['close'].pct_change()
            
            # Volatility measures
            current_vol = returns.rolling(20).std().iloc[-1]
            long_vol = returns.rolling(100).std().iloc[-1]
            
            # Trend measures
            sma_20 = talib.SMA(df['close'].values, 20)
            sma_50 = talib.SMA(df['close'].values, 50)
            
            # ADX for trend strength
            adx = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            
            if pd.notna(current_vol) and pd.notna(long_vol) and long_vol > 0:
                vol_ratio = current_vol / long_vol
                
                # High volatility regime
                if vol_ratio > 1.5:
                    return 'volatile'
                
                # Check trend strength
                if len(adx) > 0 and not pd.isna(adx[-1]):
                    if adx[-1] > 25 and len(sma_50) > 0:  # Strong trend
                        return 'trending'
                    elif adx[-1] < 15:  # Weak trend
                        return 'mean_reverting'
            
            return 'neutral'
            
        except Exception as e:
            self.logger.error(f"Regime detection failed: {e}")
            return 'unknown'
    
    def _process_macro_factors(self, macro_data: Dict) -> float:
        """
        Process macroeconomic factors for signal generation
        """
        try:
            signal = 0.0
            
            # Interest rate differentials
            if 'usd_rate' in macro_data and 'jpy_rate' in macro_data:
                rate_diff = macro_data['usd_rate'] - macro_data['jpy_rate']
                signal += np.tanh(rate_diff) * 0.4
            
            # Economic surprise indices
            if 'us_surprise' in macro_data:
                signal += np.tanh(macro_data['us_surprise'] / 100) * 0.3
            
            # Risk sentiment (VIX proxy)
            if 'risk_sentiment' in macro_data:
                signal += -np.tanh(macro_data['risk_sentiment'] / 50) * 0.3  # Negative correlation with risk
            
            return signal
            
        except Exception as e:
            self.logger.error(f"Macro factor processing failed: {e}")
            return 0.0
    
    def _get_null_signal(self) -> Dict[str, Any]:
        """Return null signal structure"""
        return {
            'signal': 0.0,
            'confidence': 0.0,
            'components': {},
            'timeframe': 'unknown',
            'strategy': 'none'
        }
    
    def _get_null_signals(self) -> Dict[str, Any]:
        """Return null signals structure"""
        return {
            'timeframe_signals': {},
            'master_signal': self._get_null_signal(),
            'regime_state': 'unknown',
            'timestamp': datetime.now().isoformat()
        }

# Global alpha engine instance
alpha_engine = AlphaEngine()