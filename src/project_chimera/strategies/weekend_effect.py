"""
Weekend Effect Strategy (WKND_EFF)
Exploits weekly patterns: Fri 23:00 UTC buy → Mon 01:00 sell
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any

from .base import Strategy, StrategyConfig
from ..domains.market import MarketFrame, Signal, SignalType
from ..settings import get_strategy_config


class WeekendEffectStrategy(Strategy):
    """
    Weekend Effect Strategy
    
    Core trigger: Fri 23:00 UTC buy → Mon 01:00 sell
    Exploits systematic weekend patterns in crypto markets
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        # Load strategy-specific settings
        self.strategy_settings = get_strategy_config('weekend_effect')
    
    def validate_config(self) -> None:
        """Validate strategy configuration"""
        required_params = []  # No additional params required for basic weekend effect
        
        # Load parameters from settings with fallbacks to config params
        self.params.setdefault('enable_friday_buy', getattr(self.strategy_settings, 'enable_friday_buy', True))
        self.params.setdefault('enable_monday_sell', getattr(self.strategy_settings, 'enable_monday_sell', True))
        self.params.setdefault('friday_entry_hour', getattr(self.strategy_settings, 'friday_entry_hour', 23))
        self.params.setdefault('monday_exit_hour', getattr(self.strategy_settings, 'monday_exit_hour', 1))
        self.params.setdefault('max_position_hours', getattr(self.strategy_settings, 'max_position_hours', 60))
        self.params.setdefault('min_volatility', getattr(self.strategy_settings, 'min_volatility', 0.001))
        self.params.setdefault('confidence', getattr(self.strategy_settings, 'confidence', 0.7))
        self.params.setdefault('target_size', getattr(self.strategy_settings, 'target_size', 0.05))
        self.params.setdefault('stop_loss_pct', getattr(self.strategy_settings, 'stop_loss_pct', 2.0))
        self.params.setdefault('take_profit_pct', getattr(self.strategy_settings, 'take_profit_pct', 1.5))
        
        # Validate ranges
        if not (0 <= self.params['friday_entry_hour'] <= 23):
            raise ValueError("friday_entry_hour must be between 0-23")
        if not (0 <= self.params['monday_exit_hour'] <= 23):
            raise ValueError("monday_exit_hour must be between 0-23")
        if self.params['max_position_hours'] <= 0:
            raise ValueError("max_position_hours must be positive")
    
    def get_required_data(self) -> Dict[str, Any]:
        """Specify required market data"""
        return {
            'ohlcv_timeframes': ['1h'],
            'orderbook_levels': 0,  # Not needed
            'indicators': [],
            'lookback_periods': 168  # 1 week of hourly data
        }
    
    def generate_signal(self, market_data: MarketFrame) -> Optional[Signal]:
        """Generate weekend effect trading signal"""
        current_time = market_data.timestamp
        
        # Convert to UTC for consistent weekend detection
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        else:
            current_time = current_time.astimezone(timezone.utc)
        
        # Get current day and hour
        weekday = current_time.weekday()  # 0=Monday, 6=Sunday
        hour = current_time.hour
        
        # Friday is 4, Monday is 0
        is_friday = weekday == 4
        is_monday = weekday == 0
        
        # Check entry conditions (Friday evening)
        if (is_friday and 
            hour == self.params['friday_entry_hour'] and
            self.params['enable_friday_buy']):
            
            # Check minimum volatility requirement
            if not self._has_sufficient_volatility(market_data):
                return None
            
            return Signal(
                symbol=market_data.symbol,
                signal_type=SignalType.BUY,
                confidence=self.params['confidence'],
                target_size=self.params['target_size'],
                entry_price=market_data.current_price,
                stop_loss=market_data.current_price * (1 - self.params['stop_loss_pct'] / 100),
                take_profit=market_data.current_price * (1 + self.params['take_profit_pct'] / 100),
                timeframe="1h",
                strategy_id="WKND_EFF",
                reasoning=f"Weekend effect: Friday {hour:02d}:00 UTC entry signal",
                metadata={
                    'entry_day': 'Friday',
                    'entry_hour': hour,
                    'expected_exit': 'Monday 01:00 UTC',
                    'pattern_type': 'weekend_effect'
                },
                timestamp=current_time
            )
        
        # Check exit conditions (Monday early morning)
        elif (is_monday and 
              hour == self.params['monday_exit_hour'] and
              self.params['enable_monday_sell']):
            
            return Signal(
                symbol=market_data.symbol,
                signal_type=SignalType.SELL,
                confidence=min(0.9, self.params['confidence'] + 0.1),  # Higher confidence for exit
                target_size=1.0,  # Close full position
                entry_price=market_data.current_price,
                timeframe="1h",
                strategy_id="WKND_EFF",
                reasoning=f"Weekend effect: Monday {hour:02d}:00 UTC exit signal",
                metadata={
                    'exit_day': 'Monday',
                    'exit_hour': hour,
                    'pattern_type': 'weekend_effect_close'
                },
                timestamp=current_time
            )
        
        return None
    
    def _has_sufficient_volatility(self, market_data: MarketFrame) -> bool:
        """Check if market has sufficient volatility to trade"""
        if not market_data.ohlcv_1h or len(market_data.ohlcv_1h) < 24:
            return True  # Assume sufficient if we can't calculate
        
        # Calculate 24-hour price volatility
        recent_candles = market_data.ohlcv_1h[-24:]
        prices = [float(candle.close) for candle in recent_candles]
        
        if len(prices) < 2:
            return True
        
        # Calculate price changes
        price_changes = []
        for i in range(1, len(prices)):
            change = abs(prices[i] - prices[i-1]) / prices[i-1]
            price_changes.append(change)
        
        avg_volatility = sum(price_changes) / len(price_changes)
        return avg_volatility >= self.params['min_volatility']
    
    def get_weekend_schedule(self, current_time: datetime) -> Dict[str, datetime]:
        """Get next weekend trading schedule"""
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)
        else:
            current_time = current_time.astimezone(timezone.utc)
        
        # Find next Friday
        days_until_friday = (4 - current_time.weekday()) % 7
        if days_until_friday == 0 and current_time.hour >= self.params['friday_entry_hour']:
            days_until_friday = 7  # Next Friday
        
        next_friday = current_time + timedelta(days=days_until_friday)
        next_friday = next_friday.replace(
            hour=self.params['friday_entry_hour'], 
            minute=0, 
            second=0, 
            microsecond=0
        )
        
        # Find corresponding Monday
        days_until_monday = (7 - 4 + 0) % 7  # From Friday to Monday
        if days_until_monday == 0:
            days_until_monday = 3  # Friday to Monday is 3 days
        else:
            days_until_monday = 3
        
        next_monday = next_friday + timedelta(days=days_until_monday)
        next_monday = next_monday.replace(
            hour=self.params['monday_exit_hour'],
            minute=0,
            second=0,
            microsecond=0
        )
        
        return {
            'next_entry': next_friday,
            'next_exit': next_monday,
            'hold_duration_hours': (next_monday - next_friday).total_seconds() / 3600
        }


def create_weekend_effect_strategy(config: Optional[Dict[str, Any]] = None) -> WeekendEffectStrategy:
    """Factory function to create WeekendEffectStrategy"""
    if config is None:
        config = {}
    
    strategy_config = StrategyConfig(
        name="weekend_effect",
        enabled=True,
        params=config
    )
    
    return WeekendEffectStrategy(strategy_config)