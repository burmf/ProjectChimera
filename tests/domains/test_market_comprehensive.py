"""
Comprehensive tests for Market domain objects - targeting coverage improvement
Tests for all market data structures, enums, and utility functions
"""

from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
import pytest

from project_chimera.domains.market import (
    OHLCV,
    FundingRate,
    MarketFrame,
    OrderBook,
    Signal,
    SignalStrength,
    SignalType,
    Ticker,
)


class TestEnums:
    """Test enum functionality"""

    def test_signal_type_enum(self):
        """Test SignalType enum values"""
        assert SignalType.BUY.value == "BUY"
        assert SignalType.SELL.value == "SELL"
        assert SignalType.HOLD.value == "HOLD"

        # Test enum membership
        assert SignalType.BUY in SignalType
        assert "INVALID" not in [s.value for s in SignalType]

    def test_signal_strength_enum(self):
        """Test SignalStrength enum values"""
        assert SignalStrength.WEAK.value == 0.3
        assert SignalStrength.MEDIUM.value == 0.6
        assert SignalStrength.STRONG.value == 0.9

        # Test ordering
        assert SignalStrength.WEAK.value < SignalStrength.MEDIUM.value
        assert SignalStrength.MEDIUM.value < SignalStrength.STRONG.value


class TestTicker:
    """Test Ticker data structure"""

    @pytest.fixture
    def sample_ticker(self):
        """Sample ticker for testing"""
        return Ticker(
            symbol="BTCUSDT",
            price=Decimal("50000.50"),
            volume_24h=Decimal("1000.25"),
            change_24h=Decimal("500.75"),
            timestamp=datetime.now(timezone.utc)
        )

    def test_ticker_creation(self, sample_ticker):
        """Test ticker creation and properties"""
        assert sample_ticker.symbol == "BTCUSDT"
        assert sample_ticker.price == Decimal("50000.50")
        assert sample_ticker.volume_24h == Decimal("1000.25")
        assert sample_ticker.change_24h == Decimal("500.75")
        assert isinstance(sample_ticker.timestamp, datetime)

    def test_ticker_immutable(self, sample_ticker):
        """Test ticker is immutable (frozen dataclass)"""
        with pytest.raises(AttributeError):
            sample_ticker.price = Decimal("60000")

    def test_ticker_decimal_precision(self):
        """Test ticker handles decimal precision correctly"""
        ticker = Ticker(
            symbol="BTCUSDT",
            price=Decimal("50000.123456789"),
            volume_24h=Decimal("1000.987654321"),
            change_24h=Decimal("-500.111"),
            timestamp=datetime.now(timezone.utc)
        )

        assert ticker.price == Decimal("50000.123456789")
        assert ticker.volume_24h == Decimal("1000.987654321")
        assert ticker.change_24h == Decimal("-500.111")

    def test_ticker_zero_values(self):
        """Test ticker with zero values"""
        ticker = Ticker(
            symbol="TESTUSDT",
            price=Decimal("0"),
            volume_24h=Decimal("0"),
            change_24h=Decimal("0"),
            timestamp=datetime.now(timezone.utc)
        )

        assert ticker.price == Decimal("0")
        assert ticker.volume_24h == Decimal("0")
        assert ticker.change_24h == Decimal("0")


class TestOrderBook:
    """Test OrderBook data structure"""

    @pytest.fixture
    def sample_orderbook(self):
        """Sample orderbook for testing"""
        return OrderBook(
            symbol="BTCUSDT",
            bids=[
                (Decimal("49900"), Decimal("1.5")),
                (Decimal("49800"), Decimal("2.0")),
                (Decimal("49700"), Decimal("0.5"))
            ],
            asks=[
                (Decimal("50100"), Decimal("1.0")),
                (Decimal("50200"), Decimal("1.5")),
                (Decimal("50300"), Decimal("2.0"))
            ],
            timestamp=datetime.now(timezone.utc)
        )

    def test_orderbook_creation(self, sample_orderbook):
        """Test orderbook creation and properties"""
        assert sample_orderbook.symbol == "BTCUSDT"
        assert len(sample_orderbook.bids) == 3
        assert len(sample_orderbook.asks) == 3
        assert sample_orderbook.bids[0][0] == Decimal("49900")  # Price
        assert sample_orderbook.bids[0][1] == Decimal("1.5")    # Quantity

    def test_best_bid_ask(self, sample_orderbook):
        """Test best bid and ask properties"""
        assert sample_orderbook.best_bid == Decimal("49900")
        assert sample_orderbook.best_ask == Decimal("50100")

    def test_spread_calculation(self, sample_orderbook):
        """Test spread calculation"""
        expected_spread = Decimal("50100") - Decimal("49900")
        assert sample_orderbook.spread == expected_spread
        assert sample_orderbook.spread == Decimal("200")

    def test_imbalance_calculation(self, sample_orderbook):
        """Test order book imbalance calculation"""
        # Bid volumes: 1.5 + 2.0 + 0.5 = 4.0
        # Ask volumes: 1.0 + 1.5 + 2.0 = 4.5
        # Total: 8.5
        # Imbalance: (4.0 - 4.5) / 8.5 = -0.5 / 8.5 ≈ -0.0588
        imbalance = sample_orderbook.imbalance
        assert imbalance is not None
        assert abs(imbalance - (-0.5 / 8.5)) < 0.001

    def test_empty_orderbook(self):
        """Test orderbook with empty bids/asks"""
        empty_book = OrderBook(
            symbol="BTCUSDT",
            bids=[],
            asks=[],
            timestamp=datetime.now(timezone.utc)
        )

        assert empty_book.best_bid is None
        assert empty_book.best_ask is None
        assert empty_book.spread is None
        assert empty_book.imbalance is None

    def test_partial_empty_orderbook(self):
        """Test orderbook with empty bids or asks"""
        bids_only = OrderBook(
            symbol="BTCUSDT",
            bids=[(Decimal("49900"), Decimal("1.0"))],
            asks=[],
            timestamp=datetime.now(timezone.utc)
        )

        assert bids_only.best_bid == Decimal("49900")
        assert bids_only.best_ask is None
        assert bids_only.spread is None
        assert bids_only.imbalance is None

    def test_zero_volume_levels(self):
        """Test orderbook with zero volume levels"""
        zero_vol_book = OrderBook(
            symbol="BTCUSDT",
            bids=[(Decimal("49900"), Decimal("0"))],
            asks=[(Decimal("50100"), Decimal("0"))],
            timestamp=datetime.now(timezone.utc)
        )

        assert zero_vol_book.imbalance == 0.0  # 0 / 0 should be handled

    def test_orderbook_immutable(self, sample_orderbook):
        """Test orderbook is immutable"""
        with pytest.raises(AttributeError):
            sample_orderbook.symbol = "ETHUSDT"


class TestOHLCV:
    """Test OHLCV data structure"""

    @pytest.fixture
    def sample_ohlcv(self):
        """Sample OHLCV for testing"""
        return OHLCV(
            symbol="BTCUSDT",
            open=Decimal("49000"),
            high=Decimal("51000"),
            low=Decimal("48000"),
            close=Decimal("50000"),
            volume=Decimal("1000"),
            timestamp=datetime.now(timezone.utc),
            timeframe="1m"
        )

    def test_ohlcv_creation(self, sample_ohlcv):
        """Test OHLCV creation and properties"""
        assert sample_ohlcv.symbol == "BTCUSDT"
        assert sample_ohlcv.open == Decimal("49000")
        assert sample_ohlcv.high == Decimal("51000")
        assert sample_ohlcv.low == Decimal("48000")
        assert sample_ohlcv.close == Decimal("50000")
        assert sample_ohlcv.volume == Decimal("1000")
        assert sample_ohlcv.timeframe == "1m"

    def test_ohlcv_default_timeframe(self):
        """Test OHLCV with default timeframe"""
        ohlcv = OHLCV(
            symbol="BTCUSDT",
            open=Decimal("49000"),
            high=Decimal("51000"),
            low=Decimal("48000"),
            close=Decimal("50000"),
            volume=Decimal("1000"),
            timestamp=datetime.now(timezone.utc)
        )

        assert ohlcv.timeframe == "1m"  # Default value

    def test_ohlcv_different_timeframes(self):
        """Test OHLCV with different timeframes"""
        timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]

        for tf in timeframes:
            ohlcv = OHLCV(
                symbol="BTCUSDT",
                open=Decimal("49000"),
                high=Decimal("51000"),
                low=Decimal("48000"),
                close=Decimal("50000"),
                volume=Decimal("1000"),
                timestamp=datetime.now(timezone.utc),
                timeframe=tf
            )
            assert ohlcv.timeframe == tf

    def test_ohlcv_edge_cases(self):
        """Test OHLCV edge cases"""
        # All prices the same (no movement)
        flat_ohlcv = OHLCV(
            symbol="BTCUSDT",
            open=Decimal("50000"),
            high=Decimal("50000"),
            low=Decimal("50000"),
            close=Decimal("50000"),
            volume=Decimal("0"),
            timestamp=datetime.now(timezone.utc)
        )

        assert flat_ohlcv.open == flat_ohlcv.high == flat_ohlcv.low == flat_ohlcv.close
        assert flat_ohlcv.volume == Decimal("0")


class TestFundingRate:
    """Test FundingRate data structure"""

    @pytest.fixture
    def sample_funding_rate(self):
        """Sample funding rate for testing"""
        return FundingRate(
            symbol="BTCUSDT",
            rate=Decimal("0.0001"),
            next_funding_time=datetime.now(timezone.utc),
            timestamp=datetime.now(timezone.utc)
        )

    def test_funding_rate_creation(self, sample_funding_rate):
        """Test funding rate creation and properties"""
        assert sample_funding_rate.symbol == "BTCUSDT"
        assert sample_funding_rate.rate == Decimal("0.0001")
        assert isinstance(sample_funding_rate.next_funding_time, datetime)
        assert isinstance(sample_funding_rate.timestamp, datetime)

    def test_funding_rate_negative(self):
        """Test funding rate with negative rate"""
        negative_funding = FundingRate(
            symbol="BTCUSDT",
            rate=Decimal("-0.0001"),
            next_funding_time=datetime.now(timezone.utc),
            timestamp=datetime.now(timezone.utc)
        )

        assert negative_funding.rate == Decimal("-0.0001")

    def test_funding_rate_zero(self):
        """Test funding rate with zero rate"""
        zero_funding = FundingRate(
            symbol="BTCUSDT",
            rate=Decimal("0"),
            next_funding_time=datetime.now(timezone.utc),
            timestamp=datetime.now(timezone.utc)
        )

        assert zero_funding.rate == Decimal("0")


class TestMarketFrame:
    """Test MarketFrame data structure"""

    @pytest.fixture
    def sample_ticker(self):
        return Ticker(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            volume_24h=Decimal("1000"),
            change_24h=Decimal("500"),
            timestamp=datetime.now(timezone.utc)
        )

    @pytest.fixture
    def sample_orderbook(self):
        return OrderBook(
            symbol="BTCUSDT",
            bids=[(Decimal("49900"), Decimal("1.0"))],
            asks=[(Decimal("50100"), Decimal("1.0"))],
            timestamp=datetime.now(timezone.utc)
        )

    @pytest.fixture
    def sample_ohlcv_list(self):
        return [
            OHLCV(
                symbol="BTCUSDT",
                open=Decimal("49000"),
                high=Decimal("51000"),
                low=Decimal("48000"),
                close=Decimal("50000"),
                volume=Decimal("1000"),
                timestamp=datetime.now(timezone.utc) - pd.Timedelta(minutes=i),
                timeframe="1m"
            ) for i in range(5)
        ]

    @pytest.fixture
    def sample_funding_rate(self):
        return FundingRate(
            symbol="BTCUSDT",
            rate=Decimal("0.0001"),
            next_funding_time=datetime.now(timezone.utc),
            timestamp=datetime.now(timezone.utc)
        )

    def test_market_frame_basic(self, sample_ticker):
        """Test basic MarketFrame creation"""
        frame = MarketFrame(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            ticker=sample_ticker
        )

        assert frame.symbol == "BTCUSDT"
        assert frame.ticker == sample_ticker
        assert frame.orderbook is None
        assert frame.ohlcv_1m is None

    def test_market_frame_full(self, sample_ticker, sample_orderbook, sample_ohlcv_list, sample_funding_rate):
        """Test MarketFrame with all data"""
        frame = MarketFrame(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            ticker=sample_ticker,
            orderbook=sample_orderbook,
            ohlcv_1m=sample_ohlcv_list,
            ohlcv_5m=sample_ohlcv_list,
            ohlcv_1h=sample_ohlcv_list,
            funding_rate=sample_funding_rate,
            indicators={"rsi": 65.0, "macd": 0.5}
        )

        assert frame.ticker == sample_ticker
        assert frame.orderbook == sample_orderbook
        assert frame.ohlcv_1m == sample_ohlcv_list
        assert frame.funding_rate == sample_funding_rate
        assert frame.indicators["rsi"] == 65.0

    def test_current_price_from_ticker(self, sample_ticker):
        """Test current_price property from ticker"""
        frame = MarketFrame(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            ticker=sample_ticker
        )

        assert frame.current_price == sample_ticker.price

    def test_current_price_from_ohlcv(self, sample_ohlcv_list):
        """Test current_price property from OHLCV when no ticker"""
        frame = MarketFrame(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            ohlcv_1m=sample_ohlcv_list
        )

        # Should use the last (most recent) OHLCV close price
        assert frame.current_price == sample_ohlcv_list[-1].close

    def test_current_price_from_orderbook(self, sample_orderbook):
        """Test current_price property from orderbook when no ticker/ohlcv"""
        frame = MarketFrame(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            orderbook=sample_orderbook
        )

        # Should calculate mid price from best bid/ask
        expected_mid = (sample_orderbook.best_bid + sample_orderbook.best_ask) / 2
        assert frame.current_price == expected_mid

    def test_current_price_none(self):
        """Test current_price property when no data available"""
        frame = MarketFrame(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc)
        )

        assert frame.current_price is None

    def test_get_ohlcv_df_1m(self, sample_ohlcv_list):
        """Test get_ohlcv_df method for 1m timeframe"""
        frame = MarketFrame(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            ohlcv_1m=sample_ohlcv_list
        )

        df = frame.get_ohlcv_df("1m")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_ohlcv_list)
        assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume']
        assert df.index.name == 'timestamp'

    def test_get_ohlcv_df_5m(self, sample_ohlcv_list):
        """Test get_ohlcv_df method for 5m timeframe"""
        frame = MarketFrame(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            ohlcv_5m=sample_ohlcv_list
        )

        df = frame.get_ohlcv_df("5m")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_ohlcv_list)

    def test_get_ohlcv_df_1h(self, sample_ohlcv_list):
        """Test get_ohlcv_df method for 1h timeframe"""
        frame = MarketFrame(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            ohlcv_1h=sample_ohlcv_list
        )

        df = frame.get_ohlcv_df("1h")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_ohlcv_list)

    def test_get_ohlcv_df_no_data(self):
        """Test get_ohlcv_df method when no data available"""
        frame = MarketFrame(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc)
        )

        df = frame.get_ohlcv_df("1m")
        assert df is None

    def test_get_ohlcv_df_empty_list(self):
        """Test get_ohlcv_df method with empty OHLCV list"""
        frame = MarketFrame(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            ohlcv_1m=[]
        )

        df = frame.get_ohlcv_df("1m")
        assert df is None

    def test_get_ohlcv_df_invalid_timeframe(self, sample_ohlcv_list):
        """Test get_ohlcv_df method with invalid timeframe"""
        frame = MarketFrame(
            symbol="BTCUSDT",
            timestamp=datetime.now(timezone.utc),
            ohlcv_1m=sample_ohlcv_list
        )

        df = frame.get_ohlcv_df("invalid")
        assert df is None


class TestSignal:
    """Test Signal data structure"""

    @pytest.fixture
    def sample_signal(self):
        """Sample signal for testing"""
        return Signal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            price=Decimal("50000"),
            timestamp=datetime.now(timezone.utc),
            strategy_name="test_strategy",
            confidence=0.8,
            target_price=Decimal("52000"),
            stop_loss=Decimal("48000"),
            take_profit=Decimal("54000"),
            indicators_used={"rsi": 30, "macd": 0.5},
            reasoning="Strong bullish signal detected"
        )

    def test_signal_creation(self, sample_signal):
        """Test signal creation and properties"""
        assert sample_signal.symbol == "BTCUSDT"
        assert sample_signal.signal_type == SignalType.BUY
        assert sample_signal.strength == SignalStrength.STRONG
        assert sample_signal.price == Decimal("50000")
        assert sample_signal.strategy_name == "test_strategy"
        assert sample_signal.confidence == 0.8
        assert sample_signal.target_price == Decimal("52000")
        assert sample_signal.stop_loss == Decimal("48000")
        assert sample_signal.take_profit == Decimal("54000")
        assert sample_signal.indicators_used["rsi"] == 30
        assert sample_signal.reasoning == "Strong bullish signal detected"

    def test_signal_minimal(self):
        """Test signal with minimal required fields"""
        signal = Signal(
            symbol="BTCUSDT",
            signal_type=SignalType.SELL,
            strength=SignalStrength.WEAK,
            price=Decimal("50000"),
            timestamp=datetime.now(timezone.utc),
            strategy_name="minimal_strategy",
            confidence=0.4
        )

        assert signal.target_price is None
        assert signal.stop_loss is None
        assert signal.take_profit is None
        assert signal.indicators_used == {}
        assert signal.reasoning == ""

    def test_signal_validation_valid(self, sample_signal):
        """Test signal validation with valid signal"""
        assert sample_signal.is_valid() is True

    def test_signal_validation_invalid_confidence_low(self):
        """Test signal validation with low confidence"""
        signal = Signal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=SignalStrength.WEAK,
            price=Decimal("50000"),
            timestamp=datetime.now(timezone.utc),
            strategy_name="test_strategy",
            confidence=-0.1  # Invalid: < 0
        )

        assert signal.is_valid() is False

    def test_signal_validation_invalid_confidence_high(self):
        """Test signal validation with high confidence"""
        signal = Signal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            price=Decimal("50000"),
            timestamp=datetime.now(timezone.utc),
            strategy_name="test_strategy",
            confidence=1.5  # Invalid: > 1
        )

        assert signal.is_valid() is False

    def test_signal_validation_invalid_price(self):
        """Test signal validation with invalid price"""
        signal = Signal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            price=Decimal("0"),  # Invalid: <= 0
            timestamp=datetime.now(timezone.utc),
            strategy_name="test_strategy",
            confidence=0.8
        )

        assert signal.is_valid() is False

    def test_signal_validation_invalid_signal_type(self):
        """Test signal validation handles signal type membership"""
        signal = Signal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            price=Decimal("50000"),
            timestamp=datetime.now(timezone.utc),
            strategy_name="test_strategy",
            confidence=0.8
        )

        # Valid signal type should pass
        assert signal.is_valid() is True

        # The enum validation is handled by the type system
        # so we can't easily test invalid enum values

    def test_signal_immutable(self, sample_signal):
        """Test signal is immutable"""
        with pytest.raises(AttributeError):
            sample_signal.price = Decimal("60000")

    def test_signal_different_types(self):
        """Test signals with different types and strengths"""
        buy_strong = Signal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            price=Decimal("50000"),
            timestamp=datetime.now(timezone.utc),
            strategy_name="test",
            confidence=0.9
        )

        sell_weak = Signal(
            symbol="ETHUSDT",
            signal_type=SignalType.SELL,
            strength=SignalStrength.WEAK,
            price=Decimal("3000"),
            timestamp=datetime.now(timezone.utc),
            strategy_name="test",
            confidence=0.3
        )

        hold_medium = Signal(
            symbol="ADAUSDT",
            signal_type=SignalType.HOLD,
            strength=SignalStrength.MEDIUM,
            price=Decimal("1"),
            timestamp=datetime.now(timezone.utc),
            strategy_name="test",
            confidence=0.6
        )

        assert buy_strong.signal_type == SignalType.BUY
        assert sell_weak.signal_type == SignalType.SELL
        assert hold_medium.signal_type == SignalType.HOLD

        assert buy_strong.strength.value > sell_weak.strength.value
        assert hold_medium.strength.value > sell_weak.strength.value
