#!/usr/bin/env python3
"""
Technical Analysis Module for Trading Bot
Provides technical indicators and signal generation
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime


class TechnicalAnalyzer:
    """Technical analysis with multiple indicators and signal fusion."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators."""
        if df.empty or not self._validate_ohlcv(df):
            self.logger.warning("Invalid OHLCV data")
            return df
        
        try:
            # Trend indicators
            df = self._add_moving_averages(df)
            df = self._add_macd(df)
            df = self._add_adx(df)
            
            # Momentum indicators  
            df = self._add_rsi(df)
            df = self._add_stochastic(df)
            
            # Volatility indicators
            df = self._add_bollinger_bands(df)
            df = self._add_atr(df)
            
            # Support/Resistance
            df = self._add_pivot_points(df)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Indicator calculation failed: {e}")
            return df
    
    def _validate_ohlcv(self, df: pd.DataFrame) -> bool:
        """Validate OHLCV data structure."""
        required_cols = ['open', 'high', 'low', 'close']
        return all(col in df.columns for col in required_cols) and len(df) > 0
    
    def _add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Simple and Exponential Moving Averages."""
        # SMA
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
        df['sma_200'] = talib.SMA(df['close'], timeperiod=200)
        
        # EMA 
        df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
        df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
        df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
        
        return df
    
    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicator."""
        macd, macd_signal, macd_hist = talib.MACD(df['close'])
        df['macd'] = macd
        df['macd_signal'] = macd_signal  
        df['macd_hist'] = macd_hist
        
        return df
    
    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI indicator."""
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        return df
    
    def _add_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Stochastic oscillator."""
        slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'],
                                   fastk_period=14, slowk_period=3, slowd_period=3)
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        
        return df
    
    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands."""
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_width'] = (upper - lower) / middle
        df['bb_position'] = (df['close'] - lower) / (upper - lower)
        
        return df
    
    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Average True Range."""
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        return df
    
    def _add_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Average Directional Index."""
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        return df
    
    def _add_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pivot points (daily)."""
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['r1'] = 2 * df['pivot'] - df['low']
        df['s1'] = 2 * df['pivot'] - df['high']
        df['r2'] = df['pivot'] + (df['high'] - df['low'])
        df['s2'] = df['pivot'] - (df['high'] - df['low'])
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from technical indicators."""
        if df.empty:
            return df
        
        # Calculate all indicators first
        df = self.calculate_indicators(df)
        
        # Individual signal components
        df = self._trend_signals(df)
        df = self._momentum_signals(df)
        df = self._volatility_signals(df)
        
        # Combine signals
        df = self._combine_signals(df)
        
        return df
    
    def _trend_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trend-based signals."""
        # SMA crossover
        df['sma_signal'] = 0
        df.loc[df['sma_20'] > df['sma_50'], 'sma_signal'] = 1
        df.loc[df['sma_20'] < df['sma_50'], 'sma_signal'] = -1
        
        # MACD signal
        df['macd_signal_direction'] = 0
        df.loc[df['macd'] > df['macd_signal'], 'macd_signal_direction'] = 1
        df.loc[df['macd'] < df['macd_signal'], 'macd_signal_direction'] = -1
        
        # ADX trend strength
        df['adx_strength'] = 0
        df.loc[df['adx'] > 25, 'adx_strength'] = 1  # Strong trend
        df.loc[df['adx'] < 20, 'adx_strength'] = -1  # Weak trend
        
        return df
    
    def _momentum_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum-based signals."""
        # RSI signals
        df['rsi_signal'] = 0
        df.loc[df['rsi'] < 30, 'rsi_signal'] = 1  # Oversold
        df.loc[df['rsi'] > 70, 'rsi_signal'] = -1  # Overbought
        
        # Stochastic signals
        df['stoch_signal'] = 0
        df.loc[(df['stoch_k'] < 20) & (df['stoch_d'] < 20), 'stoch_signal'] = 1
        df.loc[(df['stoch_k'] > 80) & (df['stoch_d'] > 80), 'stoch_signal'] = -1
        
        return df
    
    def _volatility_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate volatility-based signals."""
        # Bollinger Band signals
        df['bb_signal'] = 0
        df.loc[df['close'] <= df['bb_lower'], 'bb_signal'] = 1
        df.loc[df['close'] >= df['bb_upper'], 'bb_signal'] = -1
        
        # ATR-based volatility filter
        df['atr_filter'] = 1
        atr_percentile_80 = df['atr'].quantile(0.8)
        df.loc[df['atr'] > atr_percentile_80, 'atr_filter'] = 0  # High volatility filter
        
        return df
    
    def _combine_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine individual signals into final trading signal."""
        signal_cols = ['sma_signal', 'macd_signal_direction', 'rsi_signal', 
                      'stoch_signal', 'bb_signal']
        
        # Weight the signals (trend gets higher weight)
        weights = {
            'sma_signal': 0.3,
            'macd_signal_direction': 0.25,
            'rsi_signal': 0.2,
            'stoch_signal': 0.15,
            'bb_signal': 0.1
        }
        
        # Calculate weighted signal
        df['weighted_signal'] = 0
        for col, weight in weights.items():
            if col in df.columns:
                df['weighted_signal'] += df[col] * weight
        
        # Apply volatility filter
        df['filtered_signal'] = df['weighted_signal'] * df.get('atr_filter', 1)
        
        # Convert to discrete signals
        df['signal'] = 0
        df.loc[df['filtered_signal'] > 0.3, 'signal'] = 1
        df.loc[df['filtered_signal'] < -0.3, 'signal'] = -1
        
        # Calculate confidence
        df['confidence'] = abs(df['filtered_signal']).clip(0, 1)
        
        return df
    
    def get_latest_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get latest technical analysis summary."""
        if df.empty:
            return {'signal': 0, 'confidence': 0, 'analysis': {}}
        
        df = self.generate_signals(df)
        latest = df.iloc[-1]
        
        analysis = {
            'signal': int(latest.get('signal', 0)),
            'confidence': float(latest.get('confidence', 0)),
            'analysis': {
                'trend': {
                    'sma_20_50_cross': 'bullish' if latest.get('sma_signal', 0) > 0 else 'bearish',
                    'macd': 'bullish' if latest.get('macd_signal_direction', 0) > 0 else 'bearish',
                    'adx_strength': latest.get('adx', 0)
                },
                'momentum': {
                    'rsi': latest.get('rsi', 0),
                    'rsi_condition': self._rsi_condition(latest.get('rsi', 50)),
                    'stochastic': {
                        'k': latest.get('stoch_k', 0),
                        'd': latest.get('stoch_d', 0)
                    }
                },
                'volatility': {
                    'atr': latest.get('atr', 0),
                    'bb_position': latest.get('bb_position', 0.5),
                    'bb_width': latest.get('bb_width', 0)
                },
                'support_resistance': {
                    'pivot': latest.get('pivot', 0),
                    'r1': latest.get('r1', 0),
                    's1': latest.get('s1', 0)
                }
            },
            'timestamp': latest.name if hasattr(latest, 'name') else datetime.now()
        }
        
        return analysis
    
    def _rsi_condition(self, rsi_value: float) -> str:
        """Get RSI condition description."""
        if rsi_value < 30:
            return 'oversold'
        elif rsi_value > 70:
            return 'overbought'
        elif rsi_value < 40:
            return 'weak'
        elif rsi_value > 60:
            return 'strong'
        else:
            return 'neutral'
    
    def calculate_position_size(self, df: pd.DataFrame, account_equity: float, 
                               risk_per_trade: float = 0.01) -> Dict[str, float]:
        """Calculate position size based on ATR."""
        if df.empty or 'atr' not in df.columns:
            return {'position_size': 0, 'stop_loss_distance': 0}
        
        latest_atr = df['atr'].iloc[-1]
        latest_close = df['close'].iloc[-1]
        
        # Use 2x ATR as stop loss distance
        stop_loss_distance = latest_atr * 2
        
        # Calculate position size
        risk_amount = account_equity * risk_per_trade
        position_size = risk_amount / stop_loss_distance
        
        return {
            'position_size': position_size,
            'stop_loss_distance': stop_loss_distance,
            'stop_loss_price_long': latest_close - stop_loss_distance,
            'stop_loss_price_short': latest_close + stop_loss_distance,
            'atr': latest_atr
        }


# Legacy function for backward compatibility
def generate_sma_crossover_signals(price_df, short_window=10, long_window=50):
    """Legacy SMA crossover function for backward compatibility."""
    analyzer = TechnicalAnalyzer()
    df = analyzer.generate_signals(price_df.copy())
    
    signals = {}
    if 'signal' in df.columns:
        for timestamp, row in df.iterrows():
            signal = row['signal']
            if signal != 0:
                signals[timestamp] = int(signal)
    
    return signals