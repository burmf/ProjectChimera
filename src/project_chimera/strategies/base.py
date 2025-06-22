"""
Abstract base class for trading strategies
Defines the common interface for all trading strategies
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import pandas as pd
from ..domains.market import MarketFrame, Signal
from ..analysis import TechnicalAnalyzer, TechnicalSignal


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
    Uses pandas-ta for advanced technical analysis
    """
    
    def __init__(self, config: StrategyConfig):
        super().__init__(config)
        self.technical_analyzer = TechnicalAnalyzer()
        self._price_history = []
        self._indicators_cache = {}
        
        # パフォーマンス測定の初期化
        from .performance_mixin import PerformanceMixin
        if not isinstance(self, PerformanceMixin):
            # ミックスインの機能を動的に追加
            self.performance_logger = None
            try:
                from ..monitor.performance_logger import get_performance_logger
                self.performance_logger = get_performance_logger()
            except:
                pass
    
    def update_price_history(self, market_data: MarketFrame) -> None:
        """Update price history for technical analysis"""
        price_point = {
            'timestamp': market_data.timestamp,
            'open': market_data.ohlcv.open,
            'high': market_data.ohlcv.high,
            'low': market_data.ohlcv.low,
            'close': market_data.ohlcv.close,
            'volume': market_data.ohlcv.volume
        }
        
        self._price_history.append(price_point)
        
        # Keep only last 200 periods for efficiency
        max_history = 200
        if len(self._price_history) > max_history:
            self._price_history = self._price_history[-max_history:]
    
    def get_price_dataframe(self) -> pd.DataFrame:
        """Convert price history to DataFrame for technical analysis"""
        if not self._price_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self._price_history)
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.set_index('datetime')
        return df[['open', 'high', 'low', 'close', 'volume']]
    
    def calculate_indicators(self) -> pd.DataFrame:
        """Calculate all technical indicators using pandas-ta"""
        price_df = self.get_price_dataframe()
        if price_df.empty:
            return pd.DataFrame()
        
        # Cache key based on the latest timestamp
        cache_key = str(price_df.index[-1]) if len(price_df) > 0 else "empty"
        
        if cache_key in self._indicators_cache:
            return self._indicators_cache[cache_key]
        
        # Calculate indicators using pandas-ta
        indicators_df = self.technical_analyzer.calculate_all_indicators(price_df)
        
        # Cache the result
        self._indicators_cache[cache_key] = indicators_df
        
        # Limit cache size
        if len(self._indicators_cache) > 10:
            oldest_key = list(self._indicators_cache.keys())[0]
            del self._indicators_cache[oldest_key]
        
        return indicators_df
    
    def generate_technical_signals(self) -> List[TechnicalSignal]:
        """Generate technical signals using pandas-ta"""
        indicators_df = self.calculate_indicators()
        if indicators_df.empty:
            return []
        
        return self.technical_analyzer.generate_signals(indicators_df)
    
    def get_latest_indicators(self) -> Dict[str, float]:
        """Get latest indicator values as dictionary"""
        indicators_df = self.calculate_indicators()
        if indicators_df.empty:
            return {}
        
        latest_row = indicators_df.iloc[-1]
        return {col: val for col, val in latest_row.items() if pd.notna(val)}