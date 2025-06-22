"""
Advanced Technical Analysis Module using pandas-ta
Provides comprehensive technical indicators for ProjectChimera
"""

import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class TechnicalSignal:
    """Technical analysis signal output"""
    indicator_name: str
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float  # 0.0 to 1.0
    value: float
    threshold: Optional[float] = None
    timestamp: datetime = None
    metadata: Dict = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


class TechnicalAnalyzer:
    """
    Advanced technical analysis using pandas-ta
    
    Features:
    - Momentum indicators (RSI, MACD, Stochastic)
    - Trend indicators (SMA, EMA, ADX)
    - Volatility indicators (Bollinger Bands, ATR)
    - Volume indicators (OBV, Volume SMA)
    - Custom composite signals
    """
    
    def __init__(self, lookback_periods: int = 200):
        self.lookback_periods = lookback_periods
        self.indicators_config = {
            'rsi': {'length': 14, 'overbought': 70, 'oversold': 30},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bb': {'length': 20, 'std': 2},
            'sma': {'length': 50},
            'ema': {'length': 21},
            'atr': {'length': 14},
            'stoch': {'k': 14, 'd': 3, 'overbought': 80, 'oversold': 20},
            'adx': {'length': 14, 'strong_trend': 25}
        }
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators
        
        Args:
            df: DataFrame with OHLCV columns
            
        Returns:
            DataFrame with added technical indicators
        """
        df = df.copy()
        
        try:
            # Momentum Indicators
            df = self._add_momentum_indicators(df)
            
            # Trend Indicators
            df = self._add_trend_indicators(df)
            
            # Volatility Indicators
            df = self._add_volatility_indicators(df)
            
            # Volume Indicators
            df = self._add_volume_indicators(df)
            
            # Custom Composite Indicators
            df = self._add_composite_indicators(df)
            
            logger.info(f"Calculated {len([col for col in df.columns if any(x in col.lower() for x in ['rsi', 'macd', 'bb', 'sma', 'ema', 'atr'])])} technical indicators")
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based indicators"""
        
        # RSI
        rsi_config = self.indicators_config['rsi']
        df['rsi'] = ta.rsi(df['close'], length=rsi_config['length'])
        
        # MACD
        macd_config = self.indicators_config['macd']
        macd_data = ta.macd(df['close'], 
                           fast=macd_config['fast'], 
                           slow=macd_config['slow'], 
                           signal=macd_config['signal'])
        if macd_data is not None:
            df['macd'] = macd_data['MACD_12_26_9']
            df['macd_signal'] = macd_data['MACDs_12_26_9']
            df['macd_histogram'] = macd_data['MACDh_12_26_9']
        
        # Stochastic
        stoch_config = self.indicators_config['stoch']
        stoch_data = ta.stoch(df['high'], df['low'], df['close'], 
                             k=stoch_config['k'], d=stoch_config['d'])
        if stoch_data is not None:
            df['stoch_k'] = stoch_data[f'STOCHk_{stoch_config["k"]}_{stoch_config["d"]}_3']
            df['stoch_d'] = stoch_data[f'STOCHd_{stoch_config["k"]}_{stoch_config["d"]}_3']
        
        # Williams %R
        df['willr'] = ta.willr(df['high'], df['low'], df['close'], length=14)
        
        # ROC (Rate of Change)
        df['roc'] = ta.roc(df['close'], length=10)
        
        return df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add trend-based indicators"""
        
        # Moving Averages
        df['sma_20'] = ta.sma(df['close'], length=20)
        df['sma_50'] = ta.sma(df['close'], length=50)
        df['sma_200'] = ta.sma(df['close'], length=200)
        
        df['ema_12'] = ta.ema(df['close'], length=12)
        df['ema_26'] = ta.ema(df['close'], length=26)
        df['ema_50'] = ta.ema(df['close'], length=50)
        
        # ADX (Average Directional Index)
        adx_data = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx_data is not None:
            df['adx'] = adx_data['ADX_14']
            df['dmp'] = adx_data['DMP_14']  # Directional Movement Positive
            df['dmn'] = adx_data['DMN_14']  # Directional Movement Negative
        
        # Parabolic SAR
        psar_data = ta.psar(df['high'], df['low'], acceleration=0.02, maximum=0.2)
        if psar_data is not None:
            if isinstance(psar_data, pd.DataFrame):
                df['psar'] = psar_data.iloc[:, 0]  # Take first column
            else:
                df['psar'] = psar_data
        
        # Supertrend
        try:
            supertrend_data = ta.supertrend(df['high'], df['low'], df['close'], length=7, multiplier=3.0)
            if supertrend_data is not None and isinstance(supertrend_data, pd.DataFrame):
                # Get actual column names
                cols = supertrend_data.columns.tolist()
                if len(cols) >= 2:
                    df['supertrend'] = supertrend_data.iloc[:, 0]
                    df['supertrend_direction'] = supertrend_data.iloc[:, 1]
        except Exception as e:
            logger.warning(f"Supertrend calculation failed: {e}")
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based indicators"""
        
        # Bollinger Bands
        bb_config = self.indicators_config['bb']
        bb_data = ta.bbands(df['close'], length=bb_config['length'], std=bb_config['std'])
        if bb_data is not None:
            df['bb_lower'] = bb_data[f'BBL_{bb_config["length"]}_{bb_config["std"]}.0']
            df['bb_middle'] = bb_data[f'BBM_{bb_config["length"]}_{bb_config["std"]}.0']
            df['bb_upper'] = bb_data[f'BBU_{bb_config["length"]}_{bb_config["std"]}.0']
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_percent'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR (Average True Range)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        # Keltner Channels
        try:
            kc_data = ta.kc(df['high'], df['low'], df['close'], length=20, scalar=2)
            if kc_data is not None and isinstance(kc_data, pd.DataFrame):
                # Get actual column names dynamically
                cols = kc_data.columns.tolist()
                if len(cols) >= 3:
                    df['kc_lower'] = kc_data.iloc[:, 0]
                    df['kc_middle'] = kc_data.iloc[:, 1]
                    df['kc_upper'] = kc_data.iloc[:, 2]
        except Exception as e:
            logger.warning(f"Keltner Channels calculation failed: {e}")
        
        # Volatility Index
        df['volatility'] = ta.true_range(df['high'], df['low'], df['close']).rolling(20).mean()
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        
        if 'volume' in df.columns:
            # On Balance Volume
            df['obv'] = ta.obv(df['close'], df['volume'])
            
            # Volume SMA
            df['volume_sma'] = ta.sma(df['volume'], length=20)
            
            # Accumulation/Distribution Line
            df['ad'] = ta.ad(df['high'], df['low'], df['close'], df['volume'])
            
            # Chaikin Money Flow
            df['cmf'] = ta.cmf(df['high'], df['low'], df['close'], df['volume'], length=20)
            
            # Volume Price Trend (alternative calculation if vpt not available)
            try:
                df['vpt'] = ta.vpt(df['close'], df['volume'])
            except AttributeError:
                # Calculate VPT manually if not available in pandas-ta
                df['vpt'] = (df['volume'] * ((df['close'] - df['close'].shift(1)) / df['close'].shift(1))).cumsum()
        
        return df
    
    def _add_composite_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add custom composite indicators"""
        
        # Trend Strength (combination of ADX and moving averages)
        if 'adx' in df.columns and 'sma_20' in df.columns and 'sma_50' in df.columns:
            ma_trend = np.where(df['sma_20'] > df['sma_50'], 1, -1)
            df['trend_strength'] = df['adx'] * ma_trend
        
        # Momentum Oscillator (combination of RSI and Stochastic)
        if 'rsi' in df.columns and 'stoch_k' in df.columns:
            df['momentum_composite'] = (df['rsi'] + df['stoch_k']) / 2
        
        # Volatility Breakout Signal
        if 'bb_width' in df.columns and 'atr' in df.columns:
            df['volatility_squeeze'] = df['bb_width'] < df['bb_width'].rolling(20).mean() * 0.8
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> List[TechnicalSignal]:
        """
        Generate trading signals based on technical indicators
        
        Args:
            df: DataFrame with calculated technical indicators
            
        Returns:
            List of TechnicalSignal objects
        """
        signals = []
        
        if len(df) == 0:
            return signals
        
        latest = df.iloc[-1]
        
        try:
            # RSI Signals
            if 'rsi' in df.columns and not pd.isna(latest['rsi']):
                signals.extend(self._generate_rsi_signals(latest))
            
            # MACD Signals
            if all(col in df.columns for col in ['macd', 'macd_signal']):
                signals.extend(self._generate_macd_signals(df))
            
            # Bollinger Bands Signals
            if all(col in df.columns for col in ['bb_lower', 'bb_upper', 'close']):
                signals.extend(self._generate_bb_signals(latest))
            
            # Moving Average Signals
            if all(col in df.columns for col in ['close', 'sma_20', 'sma_50']):
                signals.extend(self._generate_ma_signals(latest))
            
            # ADX Trend Signals
            if 'adx' in df.columns and not pd.isna(latest['adx']):
                signals.extend(self._generate_adx_signals(latest))
            
            logger.info(f"Generated {len(signals)} technical signals")
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
        
        return signals
    
    def _generate_rsi_signals(self, latest: pd.Series) -> List[TechnicalSignal]:
        """Generate RSI-based signals"""
        signals = []
        rsi_config = self.indicators_config['rsi']
        
        rsi_value = latest['rsi']
        
        if rsi_value >= rsi_config['overbought']:
            signals.append(TechnicalSignal(
                indicator_name='RSI',
                signal_type='sell',
                strength=min((rsi_value - rsi_config['overbought']) / (100 - rsi_config['overbought']), 1.0),
                value=rsi_value,
                threshold=rsi_config['overbought'],
                metadata={'condition': 'overbought'}
            ))
        elif rsi_value <= rsi_config['oversold']:
            signals.append(TechnicalSignal(
                indicator_name='RSI',
                signal_type='buy',
                strength=min((rsi_config['oversold'] - rsi_value) / rsi_config['oversold'], 1.0),
                value=rsi_value,
                threshold=rsi_config['oversold'],
                metadata={'condition': 'oversold'}
            ))
        
        return signals
    
    def _generate_macd_signals(self, df: pd.DataFrame) -> List[TechnicalSignal]:
        """Generate MACD-based signals"""
        signals = []
        
        if len(df) < 2:
            return signals
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        # MACD line crosses signal line
        if (previous['macd'] <= previous['macd_signal'] and 
            current['macd'] > current['macd_signal']):
            signals.append(TechnicalSignal(
                indicator_name='MACD',
                signal_type='buy',
                strength=0.7,
                value=current['macd'],
                metadata={'condition': 'bullish_crossover', 'signal_line': current['macd_signal']}
            ))
        elif (previous['macd'] >= previous['macd_signal'] and 
              current['macd'] < current['macd_signal']):
            signals.append(TechnicalSignal(
                indicator_name='MACD',
                signal_type='sell',
                strength=0.7,
                value=current['macd'],
                metadata={'condition': 'bearish_crossover', 'signal_line': current['macd_signal']}
            ))
        
        return signals
    
    def _generate_bb_signals(self, latest: pd.Series) -> List[TechnicalSignal]:
        """Generate Bollinger Bands signals"""
        signals = []
        
        close_price = latest['close']
        bb_lower = latest['bb_lower']
        bb_upper = latest['bb_upper']
        
        if close_price <= bb_lower:
            signals.append(TechnicalSignal(
                indicator_name='BB',
                signal_type='buy',
                strength=0.6,
                value=close_price,
                threshold=bb_lower,
                metadata={'condition': 'price_at_lower_band'}
            ))
        elif close_price >= bb_upper:
            signals.append(TechnicalSignal(
                indicator_name='BB',
                signal_type='sell',
                strength=0.6,
                value=close_price,
                threshold=bb_upper,
                metadata={'condition': 'price_at_upper_band'}
            ))
        
        return signals
    
    def _generate_ma_signals(self, latest: pd.Series) -> List[TechnicalSignal]:
        """Generate Moving Average signals"""
        signals = []
        
        close_price = latest['close']
        sma_20 = latest['sma_20']
        sma_50 = latest['sma_50']
        
        # Golden Cross / Death Cross
        if sma_20 > sma_50 and close_price > sma_20:
            signals.append(TechnicalSignal(
                indicator_name='MA_Cross',
                signal_type='buy',
                strength=0.8,
                value=close_price,
                metadata={'condition': 'golden_cross', 'sma_20': sma_20, 'sma_50': sma_50}
            ))
        elif sma_20 < sma_50 and close_price < sma_20:
            signals.append(TechnicalSignal(
                indicator_name='MA_Cross',
                signal_type='sell',
                strength=0.8,
                value=close_price,
                metadata={'condition': 'death_cross', 'sma_20': sma_20, 'sma_50': sma_50}
            ))
        
        return signals
    
    def _generate_adx_signals(self, latest: pd.Series) -> List[TechnicalSignal]:
        """Generate ADX trend strength signals"""
        signals = []
        adx_config = self.indicators_config['adx']
        
        adx_value = latest['adx']
        
        if adx_value >= adx_config['strong_trend']:
            # Strong trend detected
            if 'dmp' in latest and 'dmn' in latest:
                if latest['dmp'] > latest['dmn']:
                    signals.append(TechnicalSignal(
                        indicator_name='ADX',
                        signal_type='buy',
                        strength=min(adx_value / 50, 1.0),
                        value=adx_value,
                        threshold=adx_config['strong_trend'],
                        metadata={'condition': 'strong_uptrend', 'dmp': latest['dmp'], 'dmn': latest['dmn']}
                    ))
                else:
                    signals.append(TechnicalSignal(
                        indicator_name='ADX',
                        signal_type='sell',
                        strength=min(adx_value / 50, 1.0),
                        value=adx_value,
                        threshold=adx_config['strong_trend'],
                        metadata={'condition': 'strong_downtrend', 'dmp': latest['dmp'], 'dmn': latest['dmn']}
                    ))
        
        return signals
    
    def get_market_regime(self, df: pd.DataFrame) -> Dict[str, Union[str, float]]:
        """
        Determine current market regime (trending, ranging, volatile)
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            Dictionary with regime information
        """
        if len(df) == 0:
            return {'regime': 'unknown', 'confidence': 0.0}
        
        latest = df.iloc[-1]
        
        try:
            # Trend strength from ADX
            trend_strength = latest.get('adx', 0)
            
            # Volatility from BB width
            volatility = latest.get('bb_width', 0)
            
            # Price position in BB
            bb_position = latest.get('bb_percent', 0.5)
            
            if trend_strength > 25:
                if bb_position > 0.8:
                    regime = 'strong_uptrend'
                    confidence = min(trend_strength / 50, 1.0)
                elif bb_position < 0.2:
                    regime = 'strong_downtrend'
                    confidence = min(trend_strength / 50, 1.0)
                else:
                    regime = 'trending'
                    confidence = trend_strength / 50
            elif volatility > df['bb_width'].rolling(20).mean().iloc[-1] * 1.5:
                regime = 'volatile'
                confidence = 0.7
            else:
                regime = 'ranging'
                confidence = 0.6
            
            return {
                'regime': regime,
                'confidence': confidence,
                'trend_strength': trend_strength,
                'volatility': volatility,
                'bb_position': bb_position
            }
            
        except Exception as e:
            logger.error(f"Error determining market regime: {e}")
            return {'regime': 'unknown', 'confidence': 0.0}


# Convenience functions for quick analysis
def quick_rsi(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Quick RSI calculation"""
    return ta.rsi(df['close'], length=length)


def quick_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Quick MACD calculation"""
    return ta.macd(df['close'], fast=fast, slow=slow, signal=signal)


def quick_bbands(df: pd.DataFrame, length: int = 20, std: float = 2) -> pd.DataFrame:
    """Quick Bollinger Bands calculation"""
    return ta.bbands(df['close'], length=length, std=std)


def demo_technical_analysis():
    """Demo function to test technical analysis capabilities"""
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1h')
    np.random.seed(42)
    
    # Generate realistic OHLCV data with trend and signals
    base_price = 50000
    trend = np.linspace(0, 5000, 100)  # Strong uptrend
    noise = np.random.randn(100) * 100
    close_prices = base_price + trend + noise
    
    high_prices = close_prices + np.random.uniform(50, 300, 100)
    low_prices = close_prices - np.random.uniform(50, 300, 100)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    volumes = np.random.uniform(1000, 10000, 100)
    
    df = pd.DataFrame({
        'datetime': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    # Initialize analyzer
    analyzer = TechnicalAnalyzer()
    
    # Calculate all indicators
    df_with_indicators = analyzer.calculate_all_indicators(df)
    
    # Generate signals
    signals = analyzer.generate_signals(df_with_indicators)
    
    # Get market regime
    regime = analyzer.get_market_regime(df_with_indicators)
    
    print("🔬 Technical Analysis Demo Results:")
    print(f"📊 Calculated indicators for {len(df)} periods")
    print(f"📈 Generated {len(signals)} signals")
    print(f"🎯 Market regime: {regime['regime']} (confidence: {regime['confidence']:.2f})")
    
    if signals:
        print("\n🚨 Latest Signals:")
        for signal in signals[-3:]:  # Show last 3 signals
            print(f"  {signal.indicator_name}: {signal.signal_type.upper()} "
                  f"(strength: {signal.strength:.2f}, value: {signal.value:.2f})")
    
    return df_with_indicators, signals, regime


if __name__ == "__main__":
    demo_technical_analysis()