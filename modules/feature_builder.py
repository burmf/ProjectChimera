# modules/feature_builder.py
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional
import sys
import os

# Try to import TA-Lib, fall back to manual calculations if not available
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    logging.warning("TA-Lib not available, using manual calculations")

from core.redis_manager import redis_manager

logger = logging.getLogger(__name__)

class FeatureBuilder:
    def __init__(self):
        self.feature_cache = {}
        
    def calculate_sma(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        if HAS_TALIB:
            return pd.Series(talib.SMA(prices.values, timeperiod=window), index=prices.index)
        else:
            return prices.rolling(window=window, min_periods=window).mean()
    
    def calculate_ema(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        if HAS_TALIB:
            return pd.Series(talib.EMA(prices.values, timeperiod=window), index=prices.index)
        else:
            return prices.ewm(span=window, adjust=False).mean()
    
    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        if HAS_TALIB:
            return pd.Series(talib.RSI(prices.values, timeperiod=window), index=prices.index)
        else:
            # Manual RSI calculation
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if HAS_TALIB:
            macd, signal_line, histogram = talib.MACD(prices.values, fastperiod=fast, slowperiod=slow, signalperiod=signal)
            return {
                'macd': pd.Series(macd, index=prices.index),
                'signal': pd.Series(signal_line, index=prices.index),
                'histogram': pd.Series(histogram, index=prices.index)
            }
        else:
            # Manual MACD calculation
            ema_fast = self.calculate_ema(prices, fast)
            ema_slow = self.calculate_ema(prices, slow)
            macd_line = ema_fast - ema_slow
            signal_line = self.calculate_ema(macd_line, signal)
            histogram = macd_line - signal_line
            
            return {
                'macd': macd_line,
                'signal': signal_line,
                'histogram': histogram
            }
    
    def calculate_bollinger_bands(self, prices: pd.Series, window: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        if HAS_TALIB:
            upper, middle, lower = talib.BBANDS(prices.values, timeperiod=window, nbdevup=std_dev, nbdevdn=std_dev)
            return {
                'upper': pd.Series(upper, index=prices.index),
                'middle': pd.Series(middle, index=prices.index),
                'lower': pd.Series(lower, index=prices.index)
            }
        else:
            # Manual calculation
            sma = self.calculate_sma(prices, window)
            std = prices.rolling(window=window).std()
            
            return {
                'upper': sma + (std * std_dev),
                'middle': sma,
                'lower': sma - (std * std_dev)
            }
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        if HAS_TALIB:
            return pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=window), index=close.index)
        else:
            # Manual ATR calculation
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=window).mean()
            return atr
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                           k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator"""
        if HAS_TALIB:
            slowk, slowd = talib.STOCH(high.values, low.values, close.values, 
                                     fastk_period=k_period, slowk_period=d_period, slowd_period=d_period)
            return {
                'k': pd.Series(slowk, index=close.index),
                'd': pd.Series(slowd, index=close.index)
            }
        else:
            # Manual calculation
            lowest_low = low.rolling(window=k_period).min()
            highest_high = high.rolling(window=k_period).max()
            
            k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
            d_percent = k_percent.rolling(window=d_period).mean()
            
            return {
                'k': k_percent,
                'd': d_percent
            }
    
    def build_features_for_pair(self, pair: str, ohlcv_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Build all technical features for a currency pair"""
        if ohlcv_data.empty:
            return {}
        
        try:
            close = ohlcv_data['close']
            high = ohlcv_data['high']
            low = ohlcv_data['low']
            volume = ohlcv_data.get('volume', pd.Series())
            
            features = {}
            
            # Moving Averages
            features['sma_5'] = self.calculate_sma(close, 5)
            features['sma_10'] = self.calculate_sma(close, 10)
            features['sma_20'] = self.calculate_sma(close, 20)
            features['sma_50'] = self.calculate_sma(close, 50)
            
            features['ema_12'] = self.calculate_ema(close, 12)
            features['ema_26'] = self.calculate_ema(close, 26)
            
            # Momentum Indicators
            features['rsi'] = self.calculate_rsi(close)
            
            # MACD
            macd_data = self.calculate_macd(close)
            features.update(macd_data)
            
            # Bollinger Bands
            bb_data = self.calculate_bollinger_bands(close)
            features['bb_upper'] = bb_data['upper']
            features['bb_middle'] = bb_data['middle']
            features['bb_lower'] = bb_data['lower']
            
            # Volatility
            features['atr'] = self.calculate_atr(high, low, close)
            
            # Stochastic
            stoch_data = self.calculate_stochastic(high, low, close)
            features['stoch_k'] = stoch_data['k']
            features['stoch_d'] = stoch_data['d']
            
            # Price-based features
            features['returns'] = close.pct_change()
            features['log_returns'] = np.log(close / close.shift(1))
            features['price_change'] = close.diff()
            
            # Volume features (if available)
            if not volume.empty:
                features['volume'] = volume
                features['volume_sma'] = self.calculate_sma(volume, 20)
                features['volume_ratio'] = volume / features['volume_sma']
            
            # Cross-over signals
            features['sma_cross_5_20'] = (features['sma_5'] > features['sma_20']).astype(int)
            features['sma_cross_10_50'] = (features['sma_10'] > features['sma_50']).astype(int)
            features['ema_cross'] = (features['ema_12'] > features['ema_26']).astype(int)
            
            # Store features in Redis cache
            self.cache_features_to_redis(pair, features)
            
            logger.info(f"Built {len(features)} features for {pair}")
            return features
            
        except Exception as e:
            logger.error(f"Failed to build features for {pair}: {e}")
            return {}
    
    def cache_features_to_redis(self, pair: str, features: Dict[str, pd.Series]):
        """Cache latest feature values to Redis"""
        try:
            if not redis_manager.is_connected():
                return
            
            # Store only the latest values for real-time access
            latest_features = {}
            for feature_name, series in features.items():
                if not series.empty and not pd.isna(series.iloc[-1]):
                    latest_features[feature_name] = float(series.iloc[-1])
            
            cache_key = f"features:{pair}"
            redis_manager.set_cache(cache_key, latest_features, ttl=3600)
            
            logger.debug(f"Cached {len(latest_features)} features for {pair}")
            
        except Exception as e:
            logger.error(f"Failed to cache features for {pair}: {e}")
    
    def get_cached_features(self, pair: str) -> Optional[Dict]:
        """Get cached features from Redis"""
        try:
            cache_key = f"features:{pair}"
            return redis_manager.get_cache(cache_key)
        except Exception as e:
            logger.error(f"Failed to get cached features for {pair}: {e}")
            return None
    
    def generate_trading_signals(self, pair: str, features: Dict[str, pd.Series]) -> Dict[str, int]:
        """Generate basic trading signals from features"""
        signals = {}
        
        try:
            # RSI signals
            rsi = features.get('rsi')
            if rsi is not None and not rsi.empty:
                latest_rsi = rsi.iloc[-1]
                if latest_rsi < 30:
                    signals['rsi_oversold'] = 1  # Buy signal
                elif latest_rsi > 70:
                    signals['rsi_overbought'] = -1  # Sell signal
                else:
                    signals['rsi_neutral'] = 0
            
            # MACD signals
            macd = features.get('macd')
            signal_line = features.get('signal')
            if macd is not None and signal_line is not None:
                if len(macd) > 1 and len(signal_line) > 1:
                    # MACD crossover
                    if macd.iloc[-1] > signal_line.iloc[-1] and macd.iloc[-2] <= signal_line.iloc[-2]:
                        signals['macd_bullish_cross'] = 1
                    elif macd.iloc[-1] < signal_line.iloc[-1] and macd.iloc[-2] >= signal_line.iloc[-2]:
                        signals['macd_bearish_cross'] = -1
            
            # Bollinger Bands signals
            bb_upper = features.get('bb_upper')
            bb_lower = features.get('bb_lower')
            close = features.get('returns')  # Using returns as proxy for close
            
            if bb_upper is not None and bb_lower is not None and close is not None:
                # This is simplified - you'd normally use actual close prices
                pass
            
            # SMA crossover signals
            sma_cross_5_20 = features.get('sma_cross_5_20')
            if sma_cross_5_20 is not None and len(sma_cross_5_20) > 1:
                if sma_cross_5_20.iloc[-1] == 1 and sma_cross_5_20.iloc[-2] == 0:
                    signals['sma_golden_cross'] = 1
                elif sma_cross_5_20.iloc[-1] == 0 and sma_cross_5_20.iloc[-2] == 1:
                    signals['sma_death_cross'] = -1
            
            # Store signals in Redis
            if signals:
                signal_key = f"signals:{pair}"
                redis_manager.set_cache(signal_key, signals, ttl=300)  # 5 minutes
            
            return signals
            
        except Exception as e:
            logger.error(f"Failed to generate signals for {pair}: {e}")
            return {}

# Global feature builder instance
feature_builder = FeatureBuilder()