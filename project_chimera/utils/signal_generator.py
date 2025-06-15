"""
Signal Generation System
Advanced technical analysis and signal generation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger

from ..config import Settings, get_settings
from ..core.api_client import TickerData


class SignalType(Enum):
    """Signal type enumeration"""
    TECHNICAL = "technical"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALPING = "scalping"


@dataclass
class TradingSignal:
    """Trading signal with full context"""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    size: float  # Position size in USD
    leverage: Optional[int] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    reasoning: List[str] = field(default_factory=list)
    expected_return: float = 0.0
    win_probability: float = 0.5
    avg_win: float = 0.02
    avg_loss: float = 0.01
    timestamp: datetime = field(default_factory=datetime.now)


class TechnicalIndicators:
    """Technical analysis indicators"""
    
    @staticmethod
    def sma(prices: List[float], period: int) -> List[float]:
        """Simple Moving Average"""
        if len(prices) < period:
            return []
        
        sma_values = []
        for i in range(period - 1, len(prices)):
            avg = sum(prices[i - period + 1:i + 1]) / period
            sma_values.append(avg)
        
        return sma_values
    
    @staticmethod
    def ema(prices: List[float], period: int) -> List[float]:
        """Exponential Moving Average"""
        if len(prices) < period:
            return []
        
        multiplier = 2 / (period + 1)
        ema_values = []
        
        # Start with SMA
        ema = sum(prices[:period]) / period
        ema_values.append(ema)
        
        # Calculate EMA
        for price in prices[period:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
            ema_values.append(ema)
        
        return ema_values
    
    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> List[float]:
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return []
        
        # Calculate price changes
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        
        gains = [max(0, change) for change in changes]
        losses = [max(0, -change) for change in changes]
        
        rsi_values = []
        
        # Initial average gain and loss
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        if avg_loss == 0:
            rsi_values.append(100)
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            rsi_values.append(rsi)
        
        # Calculate subsequent RSI values
        for i in range(period, len(changes)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
            if avg_loss == 0:
                rsi_values.append(100)
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
                rsi_values.append(rsi)
        
        return rsi_values
    
    @staticmethod
    def macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[List[float], List[float], List[float]]:
        """MACD (Moving Average Convergence Divergence)"""
        if len(prices) < slow:
            return [], [], []
        
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)
        
        # Align lengths
        start_idx = slow - fast
        ema_fast = ema_fast[start_idx:]
        
        # Calculate MACD line
        macd_line = [fast_val - slow_val for fast_val, slow_val in zip(ema_fast, ema_slow)]
        
        # Calculate signal line
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        
        # Calculate histogram
        signal_start = len(macd_line) - len(signal_line)
        histogram = [macd_line[i + signal_start] - signal_line[i] for i in range(len(signal_line))]
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2) -> Tuple[List[float], List[float], List[float]]:
        """Bollinger Bands"""
        if len(prices) < period:
            return [], [], []
        
        sma_values = TechnicalIndicators.sma(prices, period)
        
        upper_band = []
        lower_band = []
        
        for i in range(len(sma_values)):
            price_slice = prices[i:i + period]
            std = np.std(price_slice)
            
            upper_band.append(sma_values[i] + (std_dev * std))
            lower_band.append(sma_values[i] - (std_dev * std))
        
        return upper_band, sma_values, lower_band


class SignalGenerator:
    """
    Advanced signal generation system
    Combines multiple technical indicators for robust signals
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.indicators = TechnicalIndicators()
        
        # Signal parameters
        self.min_confidence = 0.6
        self.position_size_base = 10000  # Base position size in USD
        
        # Historical signal performance
        self.signal_history: Dict[str, List[Dict]] = {}
        
        logger.info("SignalGenerator initialized")
    
    async def generate_signal(
        self,
        symbol: str,
        klines: List[Dict],
        current_ticker: TickerData
    ) -> Optional[TradingSignal]:
        """
        Generate trading signal based on multiple indicators
        """
        if len(klines) < 50:  # Need sufficient data
            return None
        
        try:
            # Extract price data
            closes = [k['close'] for k in klines]
            highs = [k['high'] for k in klines]
            lows = [k['low'] for k in klines]
            volumes = [k['volume'] for k in klines]
            
            # Calculate indicators
            indicators = self._calculate_indicators(closes, highs, lows, volumes)
            
            # Generate signals from different strategies
            signals = []
            
            # 1. Moving Average Crossover
            ma_signal = self._moving_average_signal(indicators, closes)
            if ma_signal:
                signals.append(ma_signal)
            
            # 2. RSI Oversold/Overbought
            rsi_signal = self._rsi_signal(indicators, closes)
            if rsi_signal:
                signals.append(rsi_signal)
            
            # 3. MACD Signal
            macd_signal = self._macd_signal(indicators, closes)
            if macd_signal:
                signals.append(macd_signal)
            
            # 4. Bollinger Band Signal
            bb_signal = self._bollinger_signal(indicators, closes)
            if bb_signal:
                signals.append(bb_signal)
            
            # 5. Momentum/Breakout Signal
            momentum_signal = self._momentum_signal(indicators, closes, volumes)
            if momentum_signal:
                signals.append(momentum_signal)
            
            # Combine signals
            final_signal = self._combine_signals(symbol, signals, current_ticker)
            
            if final_signal and final_signal.confidence >= self.min_confidence:
                logger.info(f"Generated {final_signal.action} signal for {symbol} (confidence: {final_signal.confidence:.2f})")
                return final_signal
            
            return None
            
        except Exception as e:
            logger.error(f"Signal generation error for {symbol}: {e}")
            return None
    
    def _calculate_indicators(self, closes: List[float], highs: List[float], lows: List[float], volumes: List[float]) -> Dict:
        """Calculate all technical indicators"""
        return {
            'sma_20': self.indicators.sma(closes, 20),
            'sma_50': self.indicators.sma(closes, 50),
            'ema_12': self.indicators.ema(closes, 12),
            'ema_26': self.indicators.ema(closes, 26),
            'rsi': self.indicators.rsi(closes),
            'macd': self.indicators.macd(closes),
            'bb': self.indicators.bollinger_bands(closes),
            'volume_sma': self.indicators.sma(volumes, 20) if volumes else [],
            'atr': self._calculate_atr(highs, lows, closes),
            'momentum': self._calculate_momentum(closes)
        }
    
    def _calculate_atr(self, highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> List[float]:
        """Calculate Average True Range"""
        if len(closes) < period + 1:
            return []
        
        true_ranges = []
        for i in range(1, len(closes)):
            hl = highs[i] - lows[i]
            hc = abs(highs[i] - closes[i-1])
            lc = abs(lows[i] - closes[i-1])
            true_ranges.append(max(hl, hc, lc))
        
        return self.indicators.sma(true_ranges, period)
    
    def _calculate_momentum(self, closes: List[float], period: int = 10) -> List[float]:
        """Calculate price momentum"""
        if len(closes) < period:
            return []
        
        momentum = []
        for i in range(period, len(closes)):
            mom = (closes[i] - closes[i - period]) / closes[i - period]
            momentum.append(mom)
        
        return momentum
    
    def _moving_average_signal(self, indicators: Dict, closes: List[float]) -> Optional[Dict]:
        """Generate signal based on moving average crossover"""
        sma_20 = indicators['sma_20']
        sma_50 = indicators['sma_50']
        
        if len(sma_20) < 2 or len(sma_50) < 2:
            return None
        
        current_price = closes[-1]
        
        # Check for crossover
        if sma_20[-1] > sma_50[-1] and sma_20[-2] <= sma_50[-2]:
            # Golden cross - bullish
            return {
                'action': 'BUY',
                'confidence': 0.7,
                'reasoning': ['SMA20 crossed above SMA50 (Golden Cross)'],
                'signal_type': SignalType.TECHNICAL
            }
        elif sma_20[-1] < sma_50[-1] and sma_20[-2] >= sma_50[-2]:
            # Death cross - bearish
            return {
                'action': 'SELL',
                'confidence': 0.7,
                'reasoning': ['SMA20 crossed below SMA50 (Death Cross)'],
                'signal_type': SignalType.TECHNICAL
            }
        
        return None
    
    def _rsi_signal(self, indicators: Dict, closes: List[float]) -> Optional[Dict]:
        """Generate signal based on RSI"""
        rsi = indicators['rsi']
        
        if not rsi:
            return None
        
        current_rsi = rsi[-1]
        
        if current_rsi < 30:
            # Oversold - potential buy
            return {
                'action': 'BUY',
                'confidence': 0.6,
                'reasoning': [f'RSI oversold at {current_rsi:.1f}'],
                'signal_type': SignalType.MEAN_REVERSION
            }
        elif current_rsi > 70:
            # Overbought - potential sell
            return {
                'action': 'SELL',
                'confidence': 0.6,
                'reasoning': [f'RSI overbought at {current_rsi:.1f}'],
                'signal_type': SignalType.MEAN_REVERSION
            }
        
        return None
    
    def _macd_signal(self, indicators: Dict, closes: List[float]) -> Optional[Dict]:
        """Generate signal based on MACD"""
        macd_line, signal_line, histogram = indicators['macd']
        
        if len(macd_line) < 2 or len(signal_line) < 2:
            return None
        
        # MACD line crossing signal line
        if macd_line[-1] > signal_line[-1] and macd_line[-2] <= signal_line[-2]:
            # Bullish crossover
            return {
                'action': 'BUY',
                'confidence': 0.65,
                'reasoning': ['MACD bullish crossover'],
                'signal_type': SignalType.MOMENTUM
            }
        elif macd_line[-1] < signal_line[-1] and macd_line[-2] >= signal_line[-2]:
            # Bearish crossover
            return {
                'action': 'SELL',
                'confidence': 0.65,
                'reasoning': ['MACD bearish crossover'],
                'signal_type': SignalType.MOMENTUM
            }
        
        return None
    
    def _bollinger_signal(self, indicators: Dict, closes: List[float]) -> Optional[Dict]:
        """Generate signal based on Bollinger Bands"""
        upper_band, middle_band, lower_band = indicators['bb']
        
        if not upper_band or not lower_band:
            return None
        
        current_price = closes[-1]
        
        if current_price <= lower_band[-1]:
            # Price at lower band - potential buy
            return {
                'action': 'BUY',
                'confidence': 0.55,
                'reasoning': ['Price at Bollinger lower band'],
                'signal_type': SignalType.MEAN_REVERSION
            }
        elif current_price >= upper_band[-1]:
            # Price at upper band - potential sell
            return {
                'action': 'SELL',
                'confidence': 0.55,
                'reasoning': ['Price at Bollinger upper band'],
                'signal_type': SignalType.MEAN_REVERSION
            }
        
        return None
    
    def _momentum_signal(self, indicators: Dict, closes: List[float], volumes: List[float]) -> Optional[Dict]:
        """Generate signal based on momentum and volume"""
        momentum = indicators['momentum']
        volume_sma = indicators['volume_sma']
        
        if not momentum or not volumes:
            return None
        
        current_momentum = momentum[-1]
        current_volume = volumes[-1]
        
        # Volume confirmation
        volume_above_average = True
        if volume_sma:
            volume_above_average = current_volume > volume_sma[-1] * 1.2
        
        # Strong momentum with volume confirmation
        if current_momentum > 0.02 and volume_above_average:  # 2% momentum
            return {
                'action': 'BUY',
                'confidence': 0.75,
                'reasoning': ['Strong upward momentum with volume'],
                'signal_type': SignalType.BREAKOUT
            }
        elif current_momentum < -0.02 and volume_above_average:  # -2% momentum
            return {
                'action': 'SELL',
                'confidence': 0.75,
                'reasoning': ['Strong downward momentum with volume'],
                'signal_type': SignalType.BREAKOUT
            }
        
        return None
    
    def _combine_signals(self, symbol: str, signals: List[Dict], ticker: TickerData) -> Optional[TradingSignal]:
        """Combine multiple signals into final decision"""
        if not signals:
            return TradingSignal(
                symbol=symbol,
                action='HOLD',
                signal_type=SignalType.TECHNICAL,
                confidence=0.0,
                size=0.0
            )
        
        # Count buy and sell signals
        buy_signals = [s for s in signals if s['action'] == 'BUY']
        sell_signals = [s for s in signals if s['action'] == 'SELL']
        
        # Calculate weighted confidence
        if len(buy_signals) > len(sell_signals):
            action = 'BUY'
            confidence = np.mean([s['confidence'] for s in buy_signals])
            reasoning = []
            for s in buy_signals:
                reasoning.extend(s['reasoning'])
        elif len(sell_signals) > len(buy_signals):
            action = 'SELL'
            confidence = np.mean([s['confidence'] for s in sell_signals])
            reasoning = []
            for s in sell_signals:
                reasoning.extend(s['reasoning'])
        else:
            # Equal signals - use strongest
            all_signals = buy_signals + sell_signals
            strongest = max(all_signals, key=lambda x: x['confidence'])
            action = strongest['action']
            confidence = strongest['confidence'] * 0.8  # Reduce confidence for conflicting signals
            reasoning = strongest['reasoning']
        
        # Adjust confidence based on signal agreement
        signal_agreement = max(len(buy_signals), len(sell_signals)) / len(signals)
        confidence *= signal_agreement
        
        # Calculate position size based on confidence
        base_size = self.position_size_base
        size = base_size * confidence
        
        # Risk/reward estimates
        atr_pct = ticker.spread / ticker.price * 2  # Approximate ATR as 2x spread
        expected_return = atr_pct * 2  # Target 2:1 reward/risk
        
        return TradingSignal(
            symbol=symbol,
            action=action,
            signal_type=SignalType.TECHNICAL,
            confidence=confidence,
            size=size,
            entry_price=ticker.price,
            stop_loss=ticker.price * (0.98 if action == 'BUY' else 1.02),
            take_profit=ticker.price * (1.04 if action == 'BUY' else 0.96),
            reasoning=reasoning,
            expected_return=expected_return,
            win_probability=0.55 + (confidence - 0.5) * 0.2,  # 55-75% based on confidence
            avg_win=0.025,
            avg_loss=0.015
        )
    
    def update_signal_performance(self, symbol: str, signal: TradingSignal, outcome: float) -> None:
        """Update signal performance tracking"""
        if symbol not in self.signal_history:
            self.signal_history[symbol] = []
        
        self.signal_history[symbol].append({
            'timestamp': signal.timestamp,
            'action': signal.action,
            'confidence': signal.confidence,
            'signal_type': signal.signal_type.value,
            'outcome': outcome,
            'reasoning': signal.reasoning
        })
        
        # Limit history
        if len(self.signal_history[symbol]) > 100:
            self.signal_history[symbol] = self.signal_history[symbol][-50:]
    
    def get_signal_statistics(self, symbol: str) -> Dict[str, float]:
        """Get signal performance statistics"""
        if symbol not in self.signal_history:
            return {}
        
        history = self.signal_history[symbol]
        
        if not history:
            return {}
        
        # Calculate statistics
        total_signals = len(history)
        successful_signals = len([h for h in history if h['outcome'] > 0])
        win_rate = successful_signals / total_signals if total_signals > 0 else 0
        
        avg_outcome = np.mean([h['outcome'] for h in history])
        avg_confidence = np.mean([h['confidence'] for h in history])
        
        return {
            'total_signals': total_signals,
            'win_rate': win_rate,
            'avg_outcome': avg_outcome,
            'avg_confidence': avg_confidence
        }


if __name__ == "__main__":
    # Test signal generation
    from datetime import datetime
    
    # Create dummy kline data
    klines = []
    base_price = 50000
    
    for i in range(100):
        price_change = np.random.normal(0, 0.01)
        new_price = base_price * (1 + price_change)
        
        klines.append({
            'timestamp': datetime.now(),
            'open': base_price,
            'high': max(base_price, new_price),
            'low': min(base_price, new_price),
            'close': new_price,
            'volume': 100
        })
        
        base_price = new_price
    
    # Create dummy ticker
    ticker = TickerData(
        symbol='BTCUSDT',
        price=base_price,
        open_24h=49000,
        high_24h=51000,
        low_24h=48000,
        volume=1000,
        change_24h=2.0,
        ask_price=base_price + 1,
        bid_price=base_price - 1,
        spread=2,
        timestamp=datetime.now()
    )
    
    # Test signal generation
    signal_gen = SignalGenerator()
    
    import asyncio
    
    async def test():
        signal = await signal_gen.generate_signal('BTCUSDT', klines, ticker)
        if signal:
            print(f"Signal: {signal.action} {signal.symbol}")
            print(f"Confidence: {signal.confidence:.2f}")
            print(f"Reasoning: {signal.reasoning}")
        else:
            print("No signal generated")
    
    asyncio.run(test())