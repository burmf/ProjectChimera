"""
Enhanced parameterized tests for all trading strategies
Comprehensive test coverage using pytest.mark.parametrize
"""

import pytest
from decimal import Decimal
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple

from src.project_chimera.strategies.base import StrategyConfig
from src.project_chimera.strategies.vol_breakout import VolatilityBreakoutStrategy
from src.project_chimera.strategies.mini_momo import MiniMomentumStrategy  
from src.project_chimera.strategies.ob_revert import OrderBookMeanReversionStrategy
from src.project_chimera.domains.market import (
    MarketFrame, OHLCV, OrderBook, Ticker, SignalType, SignalStrength
)


class ParameterizedTestData:
    """Test data generators with parameterized scenarios"""
    
    @staticmethod
    def generate_market_scenarios() -> List[Tuple[str, Dict[str, Any]]]:
        """Generate various market scenarios for testing"""
        return [
            ("strong_uptrend", {
                "trend_pct": 0.05, "volatility": 0.02, "periods": 100,
                "expected_signals": ["BUY"], "min_confidence": 0.6
            }),
            ("strong_downtrend", {
                "trend_pct": -0.05, "volatility": 0.02, "periods": 100,
                "expected_signals": ["SELL"], "min_confidence": 0.6
            }),
            ("sideways_low_vol", {
                "trend_pct": 0.001, "volatility": 0.005, "periods": 100,
                "expected_signals": [], "min_confidence": 0.0
            }),
            ("sideways_high_vol", {
                "trend_pct": 0.001, "volatility": 0.03, "periods": 100,
                "expected_signals": ["BUY", "SELL"], "min_confidence": 0.4
            }),
            ("choppy_market", {
                "trend_pct": 0.0, "volatility": 0.025, "periods": 100,
                "expected_signals": [], "min_confidence": 0.0
            })
        ]
    
    @staticmethod
    def generate_timeframe_scenarios() -> List[Tuple[str, int]]:
        """Generate different timeframe scenarios"""
        return [
            ("1m", 1),
            ("5m", 5),
            ("15m", 15),
            ("1h", 60),
            ("4h", 240),
            ("1d", 1440)
        ]
    
    @staticmethod
    def generate_symbol_scenarios() -> List[str]:
        """Generate different trading symbols"""
        return [
            "BTCUSDT", "ETHUSDT", "ADAUSDT", "DOGEUSDT", 
            "SOLUSDT", "MATICUSDT", "AVAXUSDT", "DOTUSDT"
        ]
    
    @staticmethod
    def generate_price_levels() -> List[float]:
        """Generate different price levels for testing"""
        return [
            1.0,      # Low price (like DOGE)
            100.0,    # Medium price (like ADA)
            3000.0,   # High price (like ETH)
            50000.0,  # Very high price (like BTC)
            0.001     # Very low price (micro-cap)
        ]
    
    @staticmethod 
    def create_ohlcv_with_pattern(
        symbol: str,
        base_price: float,
        pattern_type: str,
        periods: int = 100
    ) -> List[OHLCV]:
        """Create OHLCV data with specific patterns"""
        start_time = datetime.now() - timedelta(minutes=periods)
        candles = []
        price = base_price
        
        for i in range(periods):
            timestamp = start_time + timedelta(minutes=i)
            
            if pattern_type == "uptrend":
                trend_factor = 1 + (0.001 * i)  # Gradual uptrend
                vol_factor = 0.01
            elif pattern_type == "downtrend":
                trend_factor = 1 - (0.001 * i)  # Gradual downtrend
                vol_factor = 0.01
            elif pattern_type == "volatile":
                trend_factor = 1 + (0.002 * (i % 10 - 5))  # Oscillating
                vol_factor = 0.03
            elif pattern_type == "squeeze":
                # Bollinger Band squeeze pattern
                if i < periods * 0.7:
                    vol_factor = 0.002 * (1 - i / (periods * 0.7))  # Decreasing volatility
                    trend_factor = 1
                else:
                    vol_factor = 0.02  # Breakout volatility
                    trend_factor = 1 + 0.002 * (i - periods * 0.7)
            else:  # flat
                trend_factor = 1
                vol_factor = 0.005
            
            new_price = price * trend_factor
            
            open_price = price
            close_price = new_price
            high_price = max(open_price, close_price) * (1 + vol_factor)
            low_price = min(open_price, close_price) * (1 - vol_factor)
            
            volume = Decimal(str(1000 + (i % 100) * 10))
            
            candle = OHLCV(
                symbol=symbol,
                open=Decimal(str(round(open_price, 6))),
                high=Decimal(str(round(high_price, 6))),
                low=Decimal(str(round(low_price, 6))),
                close=Decimal(str(round(close_price, 6))),
                volume=volume,
                timestamp=timestamp,
                timeframe="1m"
            )
            candles.append(candle)
            price = new_price
        
        return candles


class TestStrategyParameterized:
    """Parameterized tests for all strategies"""
    
    @pytest.mark.parametrize("symbol", ParameterizedTestData.generate_symbol_scenarios())
    @pytest.mark.parametrize("base_price", ParameterizedTestData.generate_price_levels())
    def test_all_strategies_different_symbols_and_prices(self, symbol: str, base_price: float):
        """Test all strategies work with different symbols and price levels"""
        # Create standard trending data
        candles = ParameterizedTestData.create_ohlcv_with_pattern(
            symbol=symbol,
            base_price=base_price,
            pattern_type="uptrend",
            periods=60
        )
        
        market_frame = MarketFrame(
            symbol=symbol,
            timestamp=datetime.now(),
            ohlcv_1m=candles
        )
        
        strategies = [
            VolatilityBreakoutStrategy(StrategyConfig("vol_breakout")),
            MiniMomentumStrategy(StrategyConfig("mini_momentum")),
            OrderBookMeanReversionStrategy(StrategyConfig("ob_revert"))
        ]
        
        for strategy in strategies:
            try:
                signal = strategy.generate_signal(market_frame)
                # Should not crash regardless of symbol/price
                if signal:
                    assert signal.symbol == symbol
                    assert signal.is_valid()
                    assert 0.0 <= signal.confidence <= 1.0
            except Exception as e:
                pytest.fail(f"Strategy {strategy.name} failed for {symbol} at ${base_price}: {e}")
    
    @pytest.mark.parametrize("scenario_name,scenario_params", ParameterizedTestData.generate_market_scenarios())
    def test_volatility_breakout_market_scenarios(self, scenario_name: str, scenario_params: Dict[str, Any]):
        """Test volatility breakout strategy across different market scenarios"""
        config = StrategyConfig(
            name="test_vol_breakout",
            params={
                'bb_period': 20,
                'squeeze_threshold': 0.02,
                'breakout_threshold': 0.005
            }
        )
        strategy = VolatilityBreakoutStrategy(config)
        
        if scenario_name == "sideways_low_vol":
            pattern_type = "squeeze"
        elif "uptrend" in scenario_name:
            pattern_type = "uptrend"
        elif "downtrend" in scenario_name:
            pattern_type = "downtrend"
        else:
            pattern_type = "volatile"
        
        candles = ParameterizedTestData.create_ohlcv_with_pattern(
            symbol="BTCUSDT",
            base_price=45000.0,
            pattern_type=pattern_type,
            periods=scenario_params["periods"]
        )
        
        market_frame = MarketFrame(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            ohlcv_1m=candles
        )
        
        signal = strategy.generate_signal(market_frame)
        
        # Check if signal matches expected behavior
        expected_signals = scenario_params["expected_signals"]
        min_confidence = scenario_params["min_confidence"]
        
        if expected_signals:
            if signal:
                assert signal.signal_type.value.upper() in expected_signals
                assert signal.confidence >= min_confidence
        else:
            # No signal expected for this scenario
            assert signal is None or signal.confidence < 0.5
    
    @pytest.mark.parametrize("momentum_strength", [-0.1, -0.05, -0.02, 0.02, 0.05, 0.1])
    @pytest.mark.parametrize("periods", [30, 60, 100, 200])
    def test_mini_momentum_sensitivity(self, momentum_strength: float, periods: int):
        """Test mini momentum strategy sensitivity to different momentum strengths and periods"""
        config = StrategyConfig(
            name="test_mini_momentum",
            params={
                'momentum_period': 7,
                'momentum_threshold': 0.02
            }
        )
        strategy = MiniMomentumStrategy(config)
        
        pattern_type = "uptrend" if momentum_strength > 0 else "downtrend"
        candles = ParameterizedTestData.create_ohlcv_with_pattern(
            symbol="BTCUSDT",
            base_price=45000.0,
            pattern_type=pattern_type,
            periods=periods
        )
        
        market_frame = MarketFrame(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            ohlcv_1m=candles
        )
        
        signal = strategy.generate_signal(market_frame)
        
        # Test signal characteristics based on momentum strength
        if abs(momentum_strength) >= 0.02 and periods >= 60:
            # Strong momentum with sufficient data should generate signal
            assert signal is not None
            expected_direction = SignalType.BUY if momentum_strength > 0 else SignalType.SELL
            assert signal.signal_type == expected_direction
            assert signal.confidence >= 0.3
        elif periods < 30:
            # Insufficient data should not generate signal
            assert signal is None
    
    @pytest.mark.parametrize("imbalance_ratio", [-0.6, -0.3, -0.1, 0.1, 0.3, 0.6])
    @pytest.mark.parametrize("price_deviation", [0.001, 0.005, 0.01, 0.02])
    def test_orderbook_reversion_parameters(self, imbalance_ratio: float, price_deviation: float):
        """Test order book mean reversion with different imbalance ratios and price deviations"""
        config = StrategyConfig(
            name="test_ob_revert",
            params={
                'imbalance_threshold': 0.3,
                'price_deviation_threshold': 0.005
            }
        )
        strategy = OrderBookMeanReversionStrategy(config)
        
        # Create trending data
        candles = ParameterizedTestData.create_ohlcv_with_pattern(
            symbol="BTCUSDT",
            base_price=45000.0,
            pattern_type="uptrend" if price_deviation > 0 else "flat",
            periods=60
        )
        
        # Create order book with specific imbalance
        mid_price = 45000.0 * (1 + price_deviation)
        bids = []
        asks = []
        
        for i in range(10):
            bid_price = Decimal(str(mid_price - 1 - i))
            ask_price = Decimal(str(mid_price + 1 + i))
            
            if imbalance_ratio > 0:  # Bid heavy
                bid_qty = Decimal(str(1000 * (1 + abs(imbalance_ratio))))
                ask_qty = Decimal(str(1000 * (1 - abs(imbalance_ratio))))
            else:  # Ask heavy
                bid_qty = Decimal(str(1000 * (1 - abs(imbalance_ratio))))
                ask_qty = Decimal(str(1000 * (1 + abs(imbalance_ratio))))
            
            bids.append((bid_price, bid_qty))
            asks.append((ask_price, ask_qty))
        
        orderbook = OrderBook(
            symbol="BTCUSDT",
            bids=bids,
            asks=asks,
            timestamp=datetime.now()
        )
        
        market_frame = MarketFrame(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            ohlcv_1m=candles,
            orderbook=orderbook
        )
        
        signal = strategy.generate_signal(market_frame)
        
        # Check signal based on imbalance and deviation thresholds
        should_signal = (
            abs(imbalance_ratio) >= config.params['imbalance_threshold'] and
            price_deviation >= config.params['price_deviation_threshold']
        )
        
        if should_signal:
            assert signal is not None
            # Bid heavy (positive imbalance) should generate SELL signal (reversion)
            # Ask heavy (negative imbalance) should generate BUY signal (reversion)
            expected_direction = SignalType.SELL if imbalance_ratio > 0 else SignalType.BUY
            assert signal.signal_type == expected_direction
        else:
            assert signal is None or signal.confidence < 0.5
    
    @pytest.mark.parametrize("strategy_class,config_params", [
        (VolatilityBreakoutStrategy, {'bb_period': 10, 'squeeze_threshold': 0.01}),
        (VolatilityBreakoutStrategy, {'bb_period': 30, 'squeeze_threshold': 0.03}),
        (MiniMomentumStrategy, {'momentum_period': 5, 'momentum_threshold': 0.01}),
        (MiniMomentumStrategy, {'momentum_period': 14, 'momentum_threshold': 0.03}),
        (OrderBookMeanReversionStrategy, {'imbalance_threshold': 0.2, 'price_deviation_threshold': 0.003}),
        (OrderBookMeanReversionStrategy, {'imbalance_threshold': 0.4, 'price_deviation_threshold': 0.01})
    ])
    def test_strategy_configuration_variations(self, strategy_class, config_params: Dict[str, Any]):
        """Test strategies with different configuration parameters"""
        config = StrategyConfig(
            name=f"test_{strategy_class.__name__}",
            params=config_params
        )
        
        strategy = strategy_class(config)
        
        # Create suitable test data
        candles = ParameterizedTestData.create_ohlcv_with_pattern(
            symbol="BTCUSDT",
            base_price=45000.0,
            pattern_type="squeeze" if strategy_class == VolatilityBreakoutStrategy else "uptrend",
            periods=100
        )
        
        market_frame = MarketFrame(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            ohlcv_1m=candles
        )
        
        # Should not crash with different configurations
        try:
            signal = strategy.generate_signal(market_frame)
            if signal:
                assert signal.is_valid()
                assert isinstance(signal.confidence, float)
                assert 0.0 <= signal.confidence <= 1.0
        except Exception as e:
            pytest.fail(f"Strategy {strategy_class.__name__} failed with config {config_params}: {e}")
    
    @pytest.mark.parametrize("insufficient_periods", [1, 5, 10, 15])
    def test_strategies_insufficient_data(self, insufficient_periods: int):
        """Test all strategies handle insufficient data gracefully"""
        candles = ParameterizedTestData.create_ohlcv_with_pattern(
            symbol="BTCUSDT",
            base_price=45000.0,
            pattern_type="uptrend",
            periods=insufficient_periods
        )
        
        market_frame = MarketFrame(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            ohlcv_1m=candles
        )
        
        strategies = [
            VolatilityBreakoutStrategy(StrategyConfig("vol_breakout")),
            MiniMomentumStrategy(StrategyConfig("mini_momentum")),
            OrderBookMeanReversionStrategy(StrategyConfig("ob_revert"))
        ]
        
        for strategy in strategies:
            # Should either return None or handle gracefully
            try:
                signal = strategy.generate_signal(market_frame)
                # With insufficient data, either no signal or low confidence
                if signal:
                    assert signal.confidence <= 0.7  # Lower confidence expected
            except Exception as e:
                # Should not crash, but if it does, it should be a clear error message
                assert "insufficient" in str(e).lower() or "data" in str(e).lower()
    
    @pytest.mark.parametrize("error_scenario", [
        "empty_ohlcv",
        "single_candle", 
        "invalid_prices",
        "future_timestamps",
        "missing_orderbook"
    ])
    def test_strategy_error_handling(self, error_scenario: str):
        """Test strategy error handling for various edge cases"""
        strategies = [
            VolatilityBreakoutStrategy(StrategyConfig("vol_breakout")),
            MiniMomentumStrategy(StrategyConfig("mini_momentum")),
            OrderBookMeanReversionStrategy(StrategyConfig("ob_revert"))
        ]
        
        # Create problematic market frame based on scenario
        if error_scenario == "empty_ohlcv":
            market_frame = MarketFrame(
                symbol="BTCUSDT",
                timestamp=datetime.now(),
                ohlcv_1m=[]
            )
        elif error_scenario == "single_candle":
            market_frame = MarketFrame(
                symbol="BTCUSDT",
                timestamp=datetime.now(),
                ohlcv_1m=ParameterizedTestData.create_ohlcv_with_pattern(
                    "BTCUSDT", 45000.0, "uptrend", 1
                )
            )
        elif error_scenario == "invalid_prices":
            # Create candle with invalid price (negative)
            invalid_candle = OHLCV(
                symbol="BTCUSDT",
                open=Decimal("-100"),
                high=Decimal("0"),
                low=Decimal("-200"),
                close=Decimal("-50"),
                volume=Decimal("1000"),
                timestamp=datetime.now(),
                timeframe="1m"
            )
            market_frame = MarketFrame(
                symbol="BTCUSDT",
                timestamp=datetime.now(),
                ohlcv_1m=[invalid_candle]
            )
        elif error_scenario == "future_timestamps":
            future_candles = ParameterizedTestData.create_ohlcv_with_pattern(
                "BTCUSDT", 45000.0, "uptrend", 10
            )
            # Set timestamps in the future
            for i, candle in enumerate(future_candles):
                candle.timestamp = datetime.now() + timedelta(minutes=i)
            
            market_frame = MarketFrame(
                symbol="BTCUSDT",
                timestamp=datetime.now(),
                ohlcv_1m=future_candles
            )
        else:  # missing_orderbook
            market_frame = MarketFrame(
                symbol="BTCUSDT",
                timestamp=datetime.now(),
                ohlcv_1m=ParameterizedTestData.create_ohlcv_with_pattern(
                    "BTCUSDT", 45000.0, "uptrend", 60
                ),
                orderbook=None  # Missing orderbook for OB strategy
            )
        
        for strategy in strategies:
            try:
                signal = strategy.generate_signal(market_frame)
                # Should handle gracefully - either None or valid signal
                if signal:
                    assert signal.is_valid()
            except Exception as e:
                # Exceptions are acceptable for truly invalid data
                # But should be meaningful error messages
                assert len(str(e)) > 0
                # Should not be generic errors
                assert "AttributeError" not in str(type(e))
                assert "KeyError" not in str(type(e))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])