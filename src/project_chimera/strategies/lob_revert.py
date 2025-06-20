"""
Limit Order Book Reversion Strategy (LOB_REV)
Exploits order flow imbalances: Order-flow RSI >70/<30 → mean reversion
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
import statistics

from .base import TechnicalStrategy, StrategyConfig
from ..domains.market import MarketFrame, Signal, SignalType


class LimitOrderBookReversionStrategy(TechnicalStrategy):
    """
    Limit Order Book Reversion Strategy
    
    Core trigger: Order-flow RSI >70/<30 → mean reversion
    Exploits orderbook imbalances and flow exhaustion
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
    
    def validate_config(self) -> None:
        """Validate strategy configuration"""
        # Set default parameters
        self.params.setdefault('rsi_overbought', 70.0)  # RSI overbought threshold
        self.params.setdefault('rsi_oversold', 30.0)  # RSI oversold threshold
        self.params.setdefault('rsi_period', 14)  # RSI calculation period
        self.params.setdefault('orderbook_levels', 10)  # Orderbook depth levels
        self.params.setdefault('imbalance_threshold', 2.0)  # Bid/Ask imbalance ratio
        self.params.setdefault('volume_confirmation', True)  # Require volume confirmation
        self.params.setdefault('min_volume_ratio', 1.5)  # Min volume vs average
        self.params.setdefault('max_position_minutes', 30)  # Max hold time
        self.params.setdefault('stop_loss_pct', 1.0)  # Stop loss %
        self.params.setdefault('take_profit_pct', 1.5)  # Take profit %
        self.params.setdefault('timeframe', '1m')  # Primary timeframe
        
        # Validate ranges
        if not (50 < self.params['rsi_overbought'] <= 100):
            raise ValueError("rsi_overbought must be between 50-100")
        if not (0 <= self.params['rsi_oversold'] < 50):
            raise ValueError("rsi_oversold must be between 0-50")
        if self.params['rsi_period'] < 5:
            raise ValueError("rsi_period must be >= 5")
        if self.params['imbalance_threshold'] <= 1.0:
            raise ValueError("imbalance_threshold must be > 1.0")
    
    def get_required_data(self) -> Dict[str, Any]:
        """Specify required market data"""
        return {
            'ohlcv_timeframes': ['1m'],
            'orderbook_levels': self.params['orderbook_levels'],
            'indicators': ['rsi'],
            'trade_flow_data': True,
            'lookback_periods': max(self.params['rsi_period'] * 2, 50)
        }
    
    def generate_signal(self, market_data: MarketFrame) -> Optional[Signal]:
        """Generate limit order book reversion signal"""
        if not market_data.ohlcv_1m or len(market_data.ohlcv_1m) < self.params['rsi_period'] + 5:
            return None
        
        # Calculate RSI
        prices = [float(c.close) for c in market_data.ohlcv_1m]
        rsi = self.calculate_rsi(prices, self.params['rsi_period'])
        if rsi is None:
            return None
        
        # Check RSI conditions
        is_overbought = rsi >= self.params['rsi_overbought']
        is_oversold = rsi <= self.params['rsi_oversold']
        
        if not (is_overbought or is_oversold):
            return None
        
        # Check orderbook imbalance
        orderbook_imbalance = self._calculate_orderbook_imbalance(market_data)
        if orderbook_imbalance is None:
            return None
        
        # Check volume confirmation if required
        if self.params['volume_confirmation']:
            volume_ratio = self._check_volume_confirmation(market_data.ohlcv_1m)
            if volume_ratio < self.params['min_volume_ratio']:
                return None
        else:
            volume_ratio = 1.0
        
        # Check trade flow direction
        trade_flow_pressure = self._calculate_trade_flow_pressure(market_data)
        
        # Determine signal direction (contrarian to current pressure)
        if is_overbought and orderbook_imbalance['type'] == 'bid_heavy':
            # Overbought + bid heavy → expect reversal down
            signal_type = SignalType.SELL
            reasoning = f"LOB reversion SHORT: RSI {rsi:.1f}, bid heavy {orderbook_imbalance['ratio']:.2f}x"
        elif is_oversold and orderbook_imbalance['type'] == 'ask_heavy':
            # Oversold + ask heavy → expect reversal up
            signal_type = SignalType.BUY
            reasoning = f"LOB reversion LONG: RSI {rsi:.1f}, ask heavy {orderbook_imbalance['ratio']:.2f}x"
        else:
            # Mixed signals, skip
            return None
        
        # Calculate confidence
        rsi_extremity = self._calculate_rsi_extremity(rsi)
        imbalance_strength = min(1.0, orderbook_imbalance['ratio'] / 3.0)
        confidence = min(0.85, 0.5 + rsi_extremity * 0.2 + imbalance_strength * 0.15)
        
        # Adjust stops based on direction
        if signal_type == SignalType.BUY:
            stop_loss = market_data.current_price * (1 - self.params['stop_loss_pct'] / 100)
            take_profit = market_data.current_price * (1 + self.params['take_profit_pct'] / 100)
        else:
            stop_loss = market_data.current_price * (1 + self.params['stop_loss_pct'] / 100)
            take_profit = market_data.current_price * (1 - self.params['take_profit_pct'] / 100)
        
        return Signal(
            symbol=market_data.symbol,
            signal_type=signal_type,
            confidence=confidence,
            target_size=0.025,  # 2.5% position size
            entry_price=market_data.current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timeframe="1m",
            strategy_id="LOB_REV",
            reasoning=reasoning,
            metadata={
                'rsi': rsi,
                'orderbook_imbalance': orderbook_imbalance,
                'volume_ratio': volume_ratio,
                'trade_flow_pressure': trade_flow_pressure,
                'pattern_type': 'orderbook_reversion'
            },
            timestamp=market_data.timestamp
        )
    
    def _calculate_orderbook_imbalance(self, market_data: MarketFrame) -> Optional[Dict[str, Any]]:
        """Calculate orderbook bid/ask imbalance"""
        if not hasattr(market_data, 'orderbook') or not market_data.orderbook:
            return None
        
        orderbook = market_data.orderbook
        
        # Calculate total bid and ask liquidity
        total_bid_size = sum(float(level.size) for level in orderbook.bids[:self.params['orderbook_levels']])
        total_ask_size = sum(float(level.size) for level in orderbook.asks[:self.params['orderbook_levels']])
        
        if total_bid_size == 0 or total_ask_size == 0:
            return None
        
        # Calculate imbalance ratio
        if total_bid_size > total_ask_size:
            ratio = total_bid_size / total_ask_size
            imbalance_type = 'bid_heavy'
        else:
            ratio = total_ask_size / total_bid_size
            imbalance_type = 'ask_heavy'
        
        # Only signal if imbalance is significant
        if ratio < self.params['imbalance_threshold']:
            return None
        
        return {
            'type': imbalance_type,
            'ratio': ratio,
            'total_bid_size': total_bid_size,
            'total_ask_size': total_ask_size
        }
    
    def _check_volume_confirmation(self, candles: List) -> float:
        """Check if current volume supports the signal"""
        if len(candles) < 20:
            return 1.0
        
        current_volume = float(candles[-1].volume)
        recent_volumes = [float(c.volume) for c in candles[-20:-1]]
        
        if not recent_volumes or current_volume == 0:
            return 1.0
        
        avg_volume = statistics.mean(recent_volumes)
        if avg_volume == 0:
            return 1.0
        
        return current_volume / avg_volume
    
    def _calculate_trade_flow_pressure(self, market_data: MarketFrame) -> Dict[str, float]:
        """Calculate recent trade flow pressure"""
        # This would typically use trade data (buy vs sell volume)
        # For now, use price action as proxy
        if not market_data.ohlcv_1m or len(market_data.ohlcv_1m) < 5:
            return {'buy_pressure': 0.5, 'sell_pressure': 0.5}
        
        recent_candles = market_data.ohlcv_1m[-5:]
        buy_pressure = 0
        sell_pressure = 0
        
        for candle in recent_candles:
            open_price = float(candle.open)
            close_price = float(candle.close)
            high_price = float(candle.high)
            low_price = float(candle.low)
            
            # Calculate where close is relative to range
            if high_price != low_price:
                close_position = (close_price - low_price) / (high_price - low_price)
                buy_pressure += close_position
                sell_pressure += (1 - close_position)
        
        total = buy_pressure + sell_pressure
        if total > 0:
            return {
                'buy_pressure': buy_pressure / total,
                'sell_pressure': sell_pressure / total
            }
        
        return {'buy_pressure': 0.5, 'sell_pressure': 0.5}
    
    def _calculate_rsi_extremity(self, rsi: float) -> float:
        """Calculate how extreme the RSI is (0-1 scale)"""
        if rsi >= self.params['rsi_overbought']:
            # Distance from overbought threshold
            extremity = (rsi - self.params['rsi_overbought']) / (100 - self.params['rsi_overbought'])
        elif rsi <= self.params['rsi_oversold']:
            # Distance from oversold threshold
            extremity = (self.params['rsi_oversold'] - rsi) / self.params['rsi_oversold']
        else:
            extremity = 0.0
        
        return min(1.0, extremity)
    
    def _calculate_orderbook_depth_ratio(self, market_data: MarketFrame) -> Optional[float]:
        """Calculate ratio of deep vs shallow orderbook liquidity"""
        if not hasattr(market_data, 'orderbook') or not market_data.orderbook:
            return None
        
        orderbook = market_data.orderbook
        
        # Compare first 3 levels vs next 7 levels
        shallow_levels = 3
        deep_levels = 7
        
        if len(orderbook.bids) < shallow_levels + deep_levels or len(orderbook.asks) < shallow_levels + deep_levels:
            return None
        
        # Calculate shallow liquidity
        shallow_bid = sum(float(level.size) for level in orderbook.bids[:shallow_levels])
        shallow_ask = sum(float(level.size) for level in orderbook.asks[:shallow_levels])
        shallow_total = shallow_bid + shallow_ask
        
        # Calculate deep liquidity
        deep_bid = sum(float(level.size) for level in orderbook.bids[shallow_levels:shallow_levels+deep_levels])
        deep_ask = sum(float(level.size) for level in orderbook.asks[shallow_levels:shallow_levels+deep_levels])
        deep_total = deep_bid + deep_ask
        
        if shallow_total == 0 or deep_total == 0:
            return None
        
        return deep_total / shallow_total


def create_lob_reversion_strategy(config: Optional[Dict[str, Any]] = None) -> LimitOrderBookReversionStrategy:
    """Factory function to create LimitOrderBookReversionStrategy"""
    if config is None:
        config = {}
    
    strategy_config = StrategyConfig(
        name="lob_reversion",
        enabled=True,
        params=config
    )
    
    return LimitOrderBookReversionStrategy(strategy_config)