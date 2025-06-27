"""
Comprehensive tests for all trading strategies
Tests strategy signal generation with synthetic market data fixtures
"""

from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from project_chimera.domains.market import (
    OHLCV,
    MarketFrame,
    OrderBook,
    SignalType,
)
from project_chimera.strategies.base import StrategyConfig
from project_chimera.strategies.mini_momo import MiniMomentumStrategy
from project_chimera.strategies.ob_revert import OrderBookMeanReversionStrategy
from project_chimera.strategies.vol_breakout import VolatilityBreakoutStrategy


class TestMarketDataFixtures:
    """Synthetic market data generators for testing"""

    @staticmethod
    def create_ohlcv_trend(
        symbol: str = "BTCUSDT",
        base_price: float = 45000.0,
        trend_pct: float = 0.02,
        volatility: float = 0.01,
        periods: int = 100,
        start_time: datetime = None
    ) -> list[OHLCV]:
        """Generate trending OHLCV data"""
        if start_time is None:
            start_time = datetime.now() - timedelta(minutes=periods)

        candles = []
        price = base_price

        for i in range(periods):
            timestamp = start_time + timedelta(minutes=i)

            # Apply trend
            trend_factor = 1 + (trend_pct * i / periods)
            base = price * trend_factor

            # Add volatility
            vol_factor = volatility * (0.5 - abs(0.5 - (i % 20) / 20))  # Cyclical volatility

            open_price = base * (1 + vol_factor)
            high_price = base * (1 + abs(vol_factor) * 1.5)
            low_price = base * (1 - abs(vol_factor) * 1.5)
            close_price = base * (1 + vol_factor * 0.8)

            # Ensure OHLC consistency
            high_price = max(open_price, high_price, low_price, close_price)
            low_price = min(open_price, high_price, low_price, close_price)

            volume = Decimal(str(1000 + (i % 100) * 10))

            candle = OHLCV(
                symbol=symbol,
                open=Decimal(str(round(open_price, 2))),
                high=Decimal(str(round(high_price, 2))),
                low=Decimal(str(round(low_price, 2))),
                close=Decimal(str(round(close_price, 2))),
                volume=volume,
                timestamp=timestamp
            )
            candles.append(candle)
            price = float(close_price)

        return candles

    @staticmethod
    def create_squeeze_breakout_data(
        symbol: str = "BTCUSDT",
        base_price: float = 45000.0,
        periods: int = 50
    ) -> list[OHLCV]:
        """Generate BB squeeze followed by breakout"""
        start_time = datetime.now() - timedelta(minutes=periods)
        candles = []

        # Phase 1: Squeeze (low volatility)
        squeeze_periods = periods // 2
        for i in range(squeeze_periods):
            timestamp = start_time + timedelta(minutes=i)
            vol_factor = 0.002 * (1 - i / squeeze_periods)  # Decreasing volatility

            open_price = base_price * (1 + vol_factor)
            high_price = base_price * (1 + vol_factor * 1.2)
            low_price = base_price * (1 - vol_factor * 1.2)
            close_price = base_price * (1 + vol_factor * 0.5)

            candle = OHLCV(
                symbol=symbol,
                open=Decimal(str(round(open_price, 2))),
                high=Decimal(str(round(high_price, 2))),
                low=Decimal(str(round(low_price, 2))),
                close=Decimal(str(round(close_price, 2))),
                volume=Decimal("1000"),
                timestamp=timestamp
            )
            candles.append(candle)

        # Phase 2: Breakout (high volatility)
        breakout_periods = periods - squeeze_periods
        for i in range(breakout_periods):
            timestamp = start_time + timedelta(minutes=squeeze_periods + i)
            trend_factor = 1 + (0.03 * i / breakout_periods)  # 3% uptrend
            vol_factor = 0.015  # High volatility

            base = base_price * trend_factor
            open_price = base * (1 + vol_factor)
            high_price = base * (1 + vol_factor * 1.5)
            low_price = base * (1 - vol_factor * 0.5)
            close_price = base * (1 + vol_factor * 1.2)

            candle = OHLCV(
                symbol=symbol,
                open=Decimal(str(round(open_price, 2))),
                high=Decimal(str(round(high_price, 2))),
                low=Decimal(str(round(low_price, 2))),
                close=Decimal(str(round(close_price, 2))),
                volume=Decimal("2000"),  # Higher volume on breakout
                timestamp=timestamp
            )
            candles.append(candle)

        return candles

    @staticmethod
    def create_momentum_data(
        symbol: str = "BTCUSDT",
        base_price: float = 45000.0,
        momentum_strength: float = 0.05,
        periods: int = 50
    ) -> list[OHLCV]:
        """Generate momentum pattern data"""
        start_time = datetime.now() - timedelta(minutes=periods)
        candles = []
        price = base_price

        for i in range(periods):
            timestamp = start_time + timedelta(minutes=i)

            # Accelerating momentum
            momentum_factor = momentum_strength * (i / periods) ** 1.5
            vol_factor = 0.01

            price_change = price * momentum_factor
            new_price = price + price_change

            open_price = price
            close_price = new_price
            high_price = max(open_price, close_price) * (1 + vol_factor)
            low_price = min(open_price, close_price) * (1 - vol_factor)

            volume = Decimal(str(1000 + i * 50))  # Increasing volume

            candle = OHLCV(
                symbol=symbol,
                open=Decimal(str(round(open_price, 2))),
                high=Decimal(str(round(high_price, 2))),
                low=Decimal(str(round(low_price, 2))),
                close=Decimal(str(round(close_price, 2))),
                volume=volume,
                timestamp=timestamp
            )
            candles.append(candle)
            price = new_price

        return candles

    @staticmethod
    def create_orderbook_imbalanced(
        symbol: str = "BTCUSDT",
        mid_price: float = 45000.0,
        imbalance_ratio: float = 0.4  # Positive = bid heavy
    ) -> OrderBook:
        """Generate imbalanced order book"""
        spread = mid_price * 0.0001  # 0.01% spread

        bids = []
        asks = []

        # Generate 10 levels each side
        for i in range(10):
            bid_price = Decimal(str(mid_price - spread/2 - i))
            ask_price = Decimal(str(mid_price + spread/2 + i))

            # Apply imbalance
            if imbalance_ratio > 0:  # Bid heavy
                bid_qty = Decimal(str(1000 + i * 100))
                ask_qty = Decimal(str(500 + i * 50))
            else:  # Ask heavy
                bid_qty = Decimal(str(500 + i * 50))
                ask_qty = Decimal(str(1000 + i * 100))

            bids.append((bid_price, bid_qty))
            asks.append((ask_price, ask_qty))

        return OrderBook(
            symbol=symbol,
            bids=bids,
            asks=asks,
            timestamp=datetime.now()
        )


class TestVolatilityBreakoutStrategy:
    """Test volatility breakout strategy"""

    def setup_method(self):
        config = StrategyConfig(
            name="test_vol_breakout",
            params={
                'bb_period': 20,
                'squeeze_threshold': 0.02,
                'breakout_threshold': 0.005
            }
        )
        self.strategy = VolatilityBreakoutStrategy(config)

    def test_squeeze_detection(self):
        """Test Bollinger Band squeeze detection"""
        candles = TestMarketDataFixtures.create_squeeze_breakout_data(periods=60)

        market_frame = MarketFrame(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            ohlcv_1m=candles
        )

        signal = self.strategy.generate_signal(market_frame)

        # Should generate signal on breakout
        assert signal is not None
        assert signal.signal_type == SignalType.BUY  # Upward breakout
        assert signal.confidence > 0.5
        assert "squeeze" in signal.reasoning.lower()

    def test_no_signal_without_squeeze(self):
        """Test no signal when no squeeze present"""
        candles = TestMarketDataFixtures.create_ohlcv_trend(
            trend_pct=0.01, volatility=0.02, periods=60
        )

        market_frame = MarketFrame(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            ohlcv_1m=candles
        )

        signal = self.strategy.generate_signal(market_frame)

        # Should not generate signal without squeeze
        assert signal is None

    def test_config_validation(self):
        """Test configuration validation"""
        # Invalid squeeze threshold
        with pytest.raises(ValueError):
            config = StrategyConfig(
                name="test",
                params={'squeeze_threshold': 0.15}  # Too high
            )
            VolatilityBreakoutStrategy(config)


class TestMiniMomentumStrategy:
    """Test mini-momentum strategy"""

    def setup_method(self):
        config = StrategyConfig(
            name="test_mini_momentum",
            params={
                'momentum_period': 7,
                'momentum_threshold': 0.02
            }
        )
        self.strategy = MiniMomentumStrategy(config)

    def test_bullish_momentum_signal(self):
        """Test bullish momentum signal generation"""
        candles = TestMarketDataFixtures.create_momentum_data(
            momentum_strength=0.03, periods=60
        )

        market_frame = MarketFrame(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            ohlcv_1m=candles
        )

        signal = self.strategy.generate_signal(market_frame)

        assert signal is not None
        assert signal.signal_type == SignalType.BUY
        assert signal.confidence > 0.5
        assert "momentum" in signal.reasoning.lower()

    def test_bearish_momentum_signal(self):
        """Test bearish momentum signal generation"""
        candles = TestMarketDataFixtures.create_momentum_data(
            momentum_strength=-0.03, periods=60
        )

        market_frame = MarketFrame(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            ohlcv_1m=candles
        )

        signal = self.strategy.generate_signal(market_frame)

        assert signal is not None
        assert signal.signal_type == SignalType.SELL
        assert signal.confidence > 0.5

    def test_insufficient_data(self):
        """Test behavior with insufficient data"""
        candles = TestMarketDataFixtures.create_ohlcv_trend(periods=10)  # Too few

        market_frame = MarketFrame(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            ohlcv_1m=candles
        )

        signal = self.strategy.generate_signal(market_frame)
        assert signal is None


class TestOrderBookMeanReversionStrategy:
    """Test order book mean-reversion strategy"""

    def setup_method(self):
        config = StrategyConfig(
            name="test_ob_revert",
            params={
                'imbalance_threshold': 0.3,
                'price_deviation_threshold': 0.005
            }
        )
        self.strategy = OrderBookMeanReversionStrategy(config)

    def test_bid_heavy_reversion_signal(self):
        """Test sell signal on bid-heavy order book"""
        candles = TestMarketDataFixtures.create_ohlcv_trend(
            base_price=45000.0, trend_pct=0.01, periods=60
        )

        # Price above SMA with bid-heavy order book
        orderbook = TestMarketDataFixtures.create_orderbook_imbalanced(
            mid_price=45500.0, imbalance_ratio=0.4
        )

        market_frame = MarketFrame(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            ohlcv_1m=candles,
            orderbook=orderbook
        )

        signal = self.strategy.generate_signal(market_frame)

        assert signal is not None
        assert signal.signal_type == SignalType.SELL
        assert signal.confidence > 0.5
        assert "reversion" in signal.reasoning.lower()

    def test_ask_heavy_reversion_signal(self):
        """Test buy signal on ask-heavy order book"""
        candles = TestMarketDataFixtures.create_ohlcv_trend(
            base_price=45000.0, trend_pct=-0.01, periods=60
        )

        # Price below SMA with ask-heavy order book
        orderbook = TestMarketDataFixtures.create_orderbook_imbalanced(
            mid_price=44500.0, imbalance_ratio=-0.4
        )

        market_frame = MarketFrame(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            ohlcv_1m=candles,
            orderbook=orderbook
        )

        signal = self.strategy.generate_signal(market_frame)

        assert signal is not None
        assert signal.signal_type == SignalType.BUY
        assert signal.confidence > 0.5

    def test_no_signal_balanced_orderbook(self):
        """Test no signal when order book is balanced"""
        candles = TestMarketDataFixtures.create_ohlcv_trend(periods=60)

        orderbook = TestMarketDataFixtures.create_orderbook_imbalanced(
            imbalance_ratio=0.1  # Low imbalance
        )

        market_frame = MarketFrame(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            ohlcv_1m=candles,
            orderbook=orderbook
        )

        signal = self.strategy.generate_signal(market_frame)
        assert signal is None


class TestStrategyIntegration:
    """Integration tests for strategy system"""

    def test_all_strategies_with_same_data(self):
        """Test all strategies with the same market data"""
        candles = TestMarketDataFixtures.create_ohlcv_trend(
            trend_pct=0.02, volatility=0.015, periods=100
        )

        orderbook = TestMarketDataFixtures.create_orderbook_imbalanced(
            imbalance_ratio=0.35
        )

        market_frame = MarketFrame(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            ohlcv_1m=candles,
            orderbook=orderbook
        )

        # Test all strategies
        strategies = [
            VolatilityBreakoutStrategy(StrategyConfig("vol_breakout")),
            MiniMomentumStrategy(StrategyConfig("mini_momentum")),
            OrderBookMeanReversionStrategy(StrategyConfig("ob_revert"))
        ]

        signals = []
        for strategy in strategies:
            try:
                signal = strategy.generate_signal(market_frame)
                if signal:
                    assert signal.is_valid()
                    signals.append(signal)
            except Exception as e:
                pytest.fail(f"Strategy {strategy.name} failed: {e}")

        # At least one strategy should generate a signal with this data
        assert len(signals) >= 1

    def test_strategy_required_data_specification(self):
        """Test that all strategies specify their data requirements"""
        strategies = [
            VolatilityBreakoutStrategy(StrategyConfig("vol_breakout")),
            MiniMomentumStrategy(StrategyConfig("mini_momentum")),
            OrderBookMeanReversionStrategy(StrategyConfig("ob_revert"))
        ]

        for strategy in strategies:
            required_data = strategy.get_required_data()

            assert isinstance(required_data, dict)
            assert 'ohlcv_timeframes' in required_data
            assert 'lookback_periods' in required_data
            assert isinstance(required_data['ohlcv_timeframes'], list)
            assert isinstance(required_data['lookback_periods'], int)
            assert required_data['lookback_periods'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
