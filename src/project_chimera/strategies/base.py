"""
Abstract base class for trading strategies
Defines the common interface for all trading strategies
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from ..domains.market import MarketFrame, Signal


@dataclass
class StrategyConfig:
    """Base configuration for trading strategies"""
    name: str
    enabled: bool = True
    params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}


@dataclass
class StrategyResult:
    """Result from strategy execution"""
    signal: Optional[Signal]
    metadata: Dict[str, Any]
    execution_time_ms: float
    success: bool
    error_message: Optional[str] = None


class Strategy(ABC):
    """
    Abstract base class for all trading strategies
    
    Each strategy must implement:
    1. generate_signal() - core signal generation logic
    2. validate_config() - parameter validation
    3. get_required_data() - specify required market data
    """
    
    def __init__(self, config: StrategyConfig):
        self.config = config
        self.name = config.name
        self.enabled = config.enabled
        self.params = config.params
        
        # Validate configuration
        self.validate_config()
    
    @abstractmethod
    def generate_signal(self, market_data: MarketFrame) -> Optional[Signal]:
        """
        Generate trading signal based on market data
        
        Args:
            market_data: Current market state with OHLCV, orderbook, etc.
            
        Returns:
            Signal object if conditions are met, None otherwise
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate strategy configuration parameters
        Raise ValueError if configuration is invalid
        """
        pass
    
    @abstractmethod
    def get_required_data(self) -> Dict[str, Any]:
        """
        Specify what market data this strategy requires
        
        Returns:
            Dict specifying required data:
            {
                'ohlcv_timeframes': ['1m', '5m'],
                'orderbook_levels': 10,
                'indicators': ['sma_20', 'rsi_14'],
                'lookback_periods': 100
            }
        """
        pass
    
    def get_description(self) -> str:
        """Get strategy description"""
        return f"{self.name} - {self.__class__.__doc__ or 'No description'}"
    
    def is_enabled(self) -> bool:
        """Check if strategy is enabled"""
        return self.enabled
    
    def update_config(self, new_params: Dict[str, Any]) -> None:
        """Update strategy parameters"""
        self.params.update(new_params)
        self.validate_config()


class TechnicalStrategy(Strategy):
    """
    Base class for technical analysis strategies
    Provides common technical indicator calculations
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self._indicator_cache = {}
    
    def calculate_sma(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period
    
    def calculate_ema(self, prices: List[float], period: int) -> Optional[float]:
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return None
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        
        return ema
    
    def calculate_rsi(self, prices: List[float], period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index"""
        if len(prices) < period + 1:
            return None
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change >= 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(-change)
        
        if len(gains) < period:
            return None
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_bollinger_bands(self, prices: List[float], period: int = 20, std_dev: float = 2.0) -> Optional[Dict[str, float]]:
        """Calculate Bollinger Bands"""
        if len(prices) < period:
            return None
        
        sma = self.calculate_sma(prices, period)
        if sma is None:
            return None
        
        recent_prices = prices[-period:]
        variance = sum((price - sma) ** 2 for price in recent_prices) / period
        std = variance ** 0.5
        
        return {
            'upper': sma + (std_dev * std),
            'middle': sma,
            'lower': sma - (std_dev * std),
            'bandwidth': (std_dev * std * 2) / sma if sma != 0 else 0
        }
    
    def calculate_atr(self, high: List[float], low: List[float], close: List[float], period: int = 14) -> Optional[float]:
        """Calculate Average True Range"""
        if len(high) < period + 1 or len(low) < period + 1 or len(close) < period + 1:
            return None
        
        true_ranges = []
        for i in range(1, len(close)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i-1])
            tr3 = abs(low[i] - close[i-1])
            true_ranges.append(max(tr1, tr2, tr3))
        
        if len(true_ranges) < period:
            return None
        
        return sum(true_ranges[-period:]) / period