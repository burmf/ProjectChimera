"""
Funding Rate Contrarian Strategy (FUND_CONTRA)
Exploits funding rate extremes: Funding ±0.03% & OI spike → contrarian position
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
import statistics

from .base import Strategy, StrategyConfig
from ..domains.market import MarketFrame, Signal, SignalType
from ..settings import get_strategy_config


class FundingContraStrategy(Strategy):
    """
    Funding Rate Contrarian Strategy
    
    Core trigger: Funding ±0.03% & OI spike → contrarian position
    Exploits overextended funding rates and high leverage positions
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        # Load strategy-specific settings
        self.strategy_settings = get_strategy_config('funding_contrarian')
    
    def validate_config(self) -> None:
        """Validate strategy configuration"""
        # Load parameters from settings with fallbacks
        self.params.setdefault('funding_threshold_pct', getattr(self.strategy_settings, 'funding_threshold_pct', 0.03))
        self.params.setdefault('oi_spike_multiplier', getattr(self.strategy_settings, 'oi_spike_multiplier', 2.0))
        self.params.setdefault('lookback_periods', getattr(self.strategy_settings, 'lookback_periods', 24))
        self.params.setdefault('min_funding_history', getattr(self.strategy_settings, 'min_funding_history', 8))
        self.params.setdefault('max_position_hours', getattr(self.strategy_settings, 'max_position_hours', 8))
        self.params.setdefault('stop_loss_pct', getattr(self.strategy_settings, 'stop_loss_pct', 1.5))
        self.params.setdefault('take_profit_pct', getattr(self.strategy_settings, 'take_profit_pct', 2.0))
        self.params.setdefault('funding_momentum_periods', getattr(self.strategy_settings, 'funding_momentum_periods', 3))
        self.params.setdefault('confidence_base', getattr(self.strategy_settings, 'confidence_base', 0.6))
        self.params.setdefault('target_size', getattr(self.strategy_settings, 'target_size', 0.04))
        
        # Validate ranges
        if self.params['funding_threshold_pct'] <= 0:
            raise ValueError("funding_threshold_pct must be positive")
        if self.params['oi_spike_multiplier'] <= 1.0:
            raise ValueError("oi_spike_multiplier must be > 1.0")
        if self.params['lookback_periods'] < 5:
            raise ValueError("lookback_periods must be >= 5")
    
    def get_required_data(self) -> Dict[str, Any]:
        """Specify required market data"""
        return {
            'ohlcv_timeframes': ['1h'],
            'orderbook_levels': 0,
            'indicators': [],
            'funding_rate_history': True,
            'open_interest_history': True,
            'lookback_periods': max(self.params['lookback_periods'], 48)
        }
    
    def generate_signal(self, market_data: MarketFrame) -> Optional[Signal]:
        """Generate funding contrarian signal"""
        # Check if we have required funding and OI data
        if not hasattr(market_data, 'funding_rate') or not hasattr(market_data, 'open_interest'):
            return None
        
        if not market_data.funding_rate or not market_data.open_interest:
            return None
            
        current_funding = market_data.funding_rate
        current_oi = market_data.open_interest
        
        # Check funding rate threshold
        funding_extreme = self._check_funding_extreme(current_funding)
        if not funding_extreme:
            return None
        
        # Check OI spike
        oi_data = getattr(market_data, 'oi_history', [])
        if not oi_data or len(oi_data) < self.params['lookback_periods']:
            return None
            
        oi_spike = self._check_oi_spike(current_oi, oi_data)
        if not oi_spike:
            return None
        
        # Check funding momentum (is funding getting more extreme?)
        funding_data = getattr(market_data, 'funding_history', [])
        funding_momentum = self._check_funding_momentum(funding_data)
        
        # Determine signal direction (contrarian to funding)
        if current_funding > self.params['funding_threshold_pct'] / 100:
            # High positive funding (longs pay shorts) → go short
            signal_type = SignalType.SELL
            reasoning = f"Funding contrarian SHORT: {current_funding*100:.3f}% funding, OI spike"
        else:
            # High negative funding (shorts pay longs) → go long  
            signal_type = SignalType.BUY
            reasoning = f"Funding contrarian LONG: {current_funding*100:.3f}% funding, OI spike"
        
        # Calculate confidence based on funding extremity and OI spike
        funding_extremity = abs(current_funding) / (self.params['funding_threshold_pct'] / 100)
        confidence = min(0.9, self.params['confidence_base'] + (funding_extremity - 1) * 0.1 + (oi_spike - 2) * 0.05)
        
        # Adjust stop/take based on signal direction
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
            target_size=self.params['target_size'],
            entry_price=market_data.current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timeframe="1h",
            strategy_id="FUND_CONTRA",
            reasoning=reasoning,
            metadata={
                'funding_rate': current_funding,
                'funding_threshold': self.params['funding_threshold_pct'] / 100,
                'oi_spike_multiplier': oi_spike,
                'funding_momentum': funding_momentum,
                'pattern_type': 'funding_contrarian'
            },
            timestamp=market_data.timestamp
        )
    
    def _check_funding_extreme(self, current_funding: float) -> bool:
        """Check if funding rate is extreme enough"""
        threshold = self.params['funding_threshold_pct'] / 100
        return abs(current_funding) >= threshold
    
    def _check_oi_spike(self, current_oi: float, oi_history: List[float]) -> float:
        """Check for open interest spike"""
        if len(oi_history) < self.params['lookback_periods']:
            return 0.0
        
        # Use recent history excluding current value
        recent_oi = oi_history[-self.params['lookback_periods']:]
        
        if not recent_oi or current_oi == 0:
            return 0.0
        
        avg_oi = statistics.mean(recent_oi)
        if avg_oi == 0:
            return 0.0
        
        spike_multiplier = current_oi / avg_oi
        return spike_multiplier if spike_multiplier >= self.params['oi_spike_multiplier'] else 0.0
    
    def _check_funding_momentum(self, funding_history: List[float]) -> float:
        """Check if funding is trending more extreme"""
        if not funding_history or len(funding_history) < self.params['funding_momentum_periods']:
            return 0.0
        
        recent_funding = funding_history[-self.params['funding_momentum_periods']:]
        
        # Calculate momentum (are we getting more extreme?)
        if len(recent_funding) < 2:
            return 0.0
        
        # Check if magnitude is increasing
        momentum_score = 0.0
        for i in range(1, len(recent_funding)):
            if abs(recent_funding[i]) > abs(recent_funding[i-1]):
                momentum_score += 1
        
        return momentum_score / (len(recent_funding) - 1)
    
    def _get_next_funding_time(self) -> datetime:
        """Calculate next funding time (every 8 hours at 00:00, 08:00, 16:00 UTC)"""
        now = datetime.now(timezone.utc)
        
        # Funding times: 00:00, 08:00, 16:00 UTC
        funding_hours = [0, 8, 16]
        
        for hour in funding_hours:
            next_funding = now.replace(hour=hour, minute=0, second=0, microsecond=0)
            if next_funding > now:
                return next_funding
        
        # Next day 00:00
        next_day = now + timedelta(days=1)
        return next_day.replace(hour=0, minute=0, second=0, microsecond=0)
    
    def _calculate_funding_pressure(self, funding_history: List[float]) -> Dict[str, float]:
        """Calculate funding pressure metrics"""
        if not funding_history or len(funding_history) < 3:
            return {'trend': 0.0, 'volatility': 0.0}
        
        # Calculate trend (are we moving toward extreme?)
        recent = funding_history[-3:]
        trend = (recent[-1] - recent[0]) / len(recent)
        
        # Calculate volatility
        volatility = statistics.stdev(recent) if len(recent) > 1 else 0.0
        
        return {'trend': trend, 'volatility': volatility}


def create_funding_contra_strategy(config: Optional[Dict[str, Any]] = None) -> FundingContraStrategy:
    """Factory function to create FundingContraStrategy"""
    if config is None:
        config = {}
    
    strategy_config = StrategyConfig(
        name="funding_contrarian",
        enabled=True,
        params=config
    )
    
    return FundingContraStrategy(strategy_config)