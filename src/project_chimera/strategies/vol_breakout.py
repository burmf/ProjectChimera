"""
Volatility Breakout Strategy
Detects Bollinger Band squeeze and triggers on 2% threshold breakouts
"""

from typing import Optional, Dict, Any, List
from decimal import Decimal
from datetime import datetime
import math

from .base import TechnicalStrategy, StrategyConfig
from ..domains.market import MarketFrame, Signal, SignalType, SignalStrength


class VolatilityBreakoutStrategy(TechnicalStrategy):
    """
    Bollinger Band Squeeze Breakout Strategy
    
    Logic:
    1. Calculate Bollinger Bands (20-period, 2 std dev)
    2. Detect "squeeze" when bandwidth < threshold (default 2%)
    3. Trigger BUY on price breaking above upper band
    4. Trigger SELL on price breaking below lower band
    5. Require sufficient volume confirmation
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        
        # Strategy parameters with defaults
        self.bb_period = self.params.get('bb_period', 20)
        self.bb_std_dev = self.params.get('bb_std_dev', 2.0)
        self.squeeze_threshold = self.params.get('squeeze_threshold', 0.02)  # 2%
        self.breakout_threshold = self.params.get('breakout_threshold', 0.005)  # 0.5%
        self.volume_multiplier = self.params.get('volume_multiplier', 1.5)
        self.min_lookback = self.params.get('min_lookback', 50)
    
    def validate_config(self) -> None:
        """Validate strategy configuration"""
        if self.bb_period < 5 or self.bb_period > 100:
            raise ValueError("bb_period must be between 5 and 100")
        
        if self.bb_std_dev < 1.0 or self.bb_std_dev > 3.0:
            raise ValueError("bb_std_dev must be between 1.0 and 3.0")
        
        if self.squeeze_threshold <= 0 or self.squeeze_threshold > 0.1:
            raise ValueError("squeeze_threshold must be between 0 and 0.1 (10%)")
        
        if self.breakout_threshold <= 0 or self.breakout_threshold > 0.05:
            raise ValueError("breakout_threshold must be between 0 and 0.05 (5%)")
    
    def get_required_data(self) -> Dict[str, Any]:
        """Specify required market data"""
        return {
            'ohlcv_timeframes': ['1m'],
            'orderbook_levels': 5,
            'indicators': ['bollinger_bands', 'volume_profile'],
            'lookback_periods': max(self.min_lookback, self.bb_period * 2)
        }
    
    def generate_signal(self, market_data: MarketFrame) -> Optional[Signal]:
        """Generate volatility breakout signal"""
        if not market_data.ohlcv_1m or len(market_data.ohlcv_1m) < self.min_lookback:
            return None
        
        # Extract price and volume data
        candles = market_data.ohlcv_1m
        closes = [float(c.close) for c in candles]
        volumes = [float(c.volume) for c in candles]
        highs = [float(c.high) for c in candles]
        lows = [float(c.low) for c in candles]
        
        current_price = float(market_data.current_price)
        current_volume = volumes[-1]
        avg_volume = sum(volumes[-20:]) / min(20, len(volumes))
        
        # Calculate Bollinger Bands
        bb_data = self.calculate_bollinger_bands(closes, self.bb_period, self.bb_std_dev)
        if not bb_data:
            return None
        
        upper_band = bb_data['upper']
        lower_band = bb_data['lower']
        middle_band = bb_data['middle']
        bandwidth = bb_data['bandwidth']
        
        # Check for squeeze condition
        is_squeeze = bandwidth < self.squeeze_threshold
        
        if not is_squeeze:
            return None
        
        # Check for breakout
        signal_type = None
        strength = SignalStrength.MEDIUM
        confidence = 0.6
        
        # Bullish breakout
        if current_price > upper_band * (1 + self.breakout_threshold):
            signal_type = SignalType.BUY
            # Higher confidence if volume is strong
            if current_volume > avg_volume * self.volume_multiplier:
                strength = SignalStrength.STRONG
                confidence = 0.8
        
        # Bearish breakout  
        elif current_price < lower_band * (1 - self.breakout_threshold):
            signal_type = SignalType.SELL
            if current_volume > avg_volume * self.volume_multiplier:
                strength = SignalStrength.STRONG
                confidence = 0.8
        
        if signal_type is None:
            return None
        
        # Calculate targets
        atr = self.calculate_atr(highs, lows, closes, 14)
        atr_factor = atr if atr else (current_price * 0.01)  # 1% fallback
        
        if signal_type == SignalType.BUY:
            target_price = Decimal(str(current_price + (atr_factor * 2)))
            stop_loss = Decimal(str(lower_band))
        else:
            target_price = Decimal(str(current_price - (atr_factor * 2)))
            stop_loss = Decimal(str(upper_band))
        
        # Create signal
        signal = Signal(
            symbol=market_data.symbol,
            signal_type=signal_type,
            strength=strength,
            price=Decimal(str(current_price)),
            timestamp=market_data.timestamp,
            strategy_name=self.name,
            confidence=confidence,
            target_price=target_price,
            stop_loss=stop_loss,
            indicators_used={
                'bb_upper': upper_band,
                'bb_lower': lower_band,
                'bb_middle': middle_band,
                'bandwidth': bandwidth,
                'squeeze_threshold': self.squeeze_threshold,
                'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 0,
                'atr': atr
            },
            reasoning=f"BB Squeeze breakout: bandwidth={bandwidth:.4f} < {self.squeeze_threshold}, "
                     f"price break {'above' if signal_type == SignalType.BUY else 'below'} "
                     f"{'upper' if signal_type == SignalType.BUY else 'lower'} band with volume confirmation"
        )
        
        return signal if signal.is_valid() else None


def create_vol_breakout_strategy(
    name: str = "volatility_breakout",
    bb_period: int = 20,
    bb_std_dev: float = 2.0,
    squeeze_threshold: float = 0.02,
    breakout_threshold: float = 0.005,
    volume_multiplier: float = 1.5
) -> VolatilityBreakoutStrategy:
    """Factory function to create volatility breakout strategy"""
    
    config = StrategyConfig(
        name=name,
        enabled=True,
        params={
            'bb_period': bb_period,
            'bb_std_dev': bb_std_dev,
            'squeeze_threshold': squeeze_threshold,
            'breakout_threshold': breakout_threshold,
            'volume_multiplier': volume_multiplier
        }
    )
    
    return VolatilityBreakoutStrategy(config)