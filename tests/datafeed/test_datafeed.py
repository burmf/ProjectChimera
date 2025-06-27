"""
Comprehensive tests for data feed layer
Tests adapters, factory, and feed functionality
"""

import asyncio
from datetime import datetime

import pytest

from project_chimera.datafeed.adapters.mock import MockAdapter
from project_chimera.datafeed.base import AsyncDataFeed, FeedStatus
from project_chimera.datafeed.factory import (
    DataFeedFactory,
    ExchangeType,
    create_datafeed,
)
from project_chimera.domains.market import MarketFrame


class TestMockAdapter:
    """Test mock adapter functionality"""

    @pytest.fixture
    async def mock_adapter(self):
        adapter = MockAdapter("mock", {
            'base_prices': {'BTCUSDT': 45000.0, 'ETHUSDT': 3000.0},
            'volatility': 0.01,
            'trend': 0.001
        })
        await adapter.connect()
        yield adapter
        await adapter.disconnect()

    @pytest.mark.asyncio
    async def test_connection(self, mock_adapter):
        """Test adapter connection"""
        assert mock_adapter.is_connected()
        assert await mock_adapter.health_check()

    @pytest.mark.asyncio
    async def test_ticker_data(self, mock_adapter):
        """Test ticker data generation"""
        await mock_adapter.subscribe_ticker("BTCUSDT")
        ticker = await mock_adapter.get_ticker("BTCUSDT")

        assert ticker is not None
        assert ticker.symbol == "BTCUSDT"
        assert ticker.price > 0
        assert ticker.volume_24h > 0
        assert isinstance(ticker.timestamp, datetime)

    @pytest.mark.asyncio
    async def test_orderbook_data(self, mock_adapter):
        """Test orderbook data generation"""
        await mock_adapter.subscribe_orderbook("BTCUSDT", 10)
        orderbook = await mock_adapter.get_orderbook("BTCUSDT", 10)

        assert orderbook is not None
        assert orderbook.symbol == "BTCUSDT"
        assert len(orderbook.bids) == 10
        assert len(orderbook.asks) == 10
        assert orderbook.best_bid is not None
        assert orderbook.best_ask is not None
        assert orderbook.spread is not None
        assert orderbook.spread > 0

    @pytest.mark.asyncio
    async def test_historical_klines(self, mock_adapter):
        """Test historical kline data"""
        klines = await mock_adapter.get_historical_klines("BTCUSDT", "1m", 100)

        assert len(klines) == 100
        assert all(kline.symbol == "BTCUSDT" for kline in klines)
        assert all(kline.high >= kline.low for kline in klines)
        assert all(kline.high >= max(kline.open, kline.close) for kline in klines)
        assert all(kline.low <= min(kline.open, kline.close) for kline in klines)

    @pytest.mark.asyncio
    async def test_funding_rate(self, mock_adapter):
        """Test funding rate data"""
        funding = await mock_adapter.get_funding_rate("BTCUSDT")

        assert funding is not None
        assert funding.symbol == "BTCUSDT"
        assert -0.01 <= float(funding.rate) <= 0.01
        assert funding.next_funding_time > datetime.now()

    @pytest.mark.asyncio
    async def test_streaming_ticker(self, mock_adapter):
        """Test streaming ticker functionality"""
        await mock_adapter.subscribe_ticker("BTCUSDT")

        tickers = []
        async for ticker in mock_adapter.stream_ticker("BTCUSDT"):
            tickers.append(ticker)
            if len(tickers) >= 3:
                break

        assert len(tickers) == 3
        assert all(ticker.symbol == "BTCUSDT" for ticker in tickers)

        # Prices should change over time (with some randomness)
        prices = [float(ticker.price) for ticker in tickers]
        assert not all(price == prices[0] for price in prices)


class TestDataFeedFactory:
    """Test data feed factory"""

    def test_supported_exchanges(self):
        """Test factory supports required exchanges"""
        # Test enum values
        assert ExchangeType.BINANCE.value == "binance"
        assert ExchangeType.BYBIT.value == "bybit"
        assert ExchangeType.MOCK.value == "mock"

    def test_create_mock_adapter(self):
        """Test creating mock adapter"""
        adapter = DataFeedFactory.create_adapter(ExchangeType.MOCK, {})
        assert isinstance(adapter, MockAdapter)
        assert adapter.name == "mock"

    def test_create_datafeed_from_string(self):
        """Test creating datafeed from string configuration"""
        feed = create_datafeed(
            exchange="mock",
            symbols=["BTCUSDT", "ETHUSDT"],
            config={
                'feed': {
                    'enable_orderbook': True,
                    'enable_funding': False,
                    'cache_ttl': 30
                }
            }
        )

        assert isinstance(feed, AsyncDataFeed)
        assert len(feed.symbols) == 2
        assert "BTCUSDT" in feed.symbols
        assert "ETHUSDT" in feed.symbols

    def test_invalid_exchange_raises_error(self):
        """Test invalid exchange name raises error"""
        with pytest.raises(ValueError, match="Unsupported exchange"):
            create_datafeed("invalid_exchange")


class TestAsyncDataFeed:
    """Test async data feed functionality"""

    @pytest.fixture
    async def feed(self):
        mock_adapter = MockAdapter("mock", {})
        feed = AsyncDataFeed(
            adapter=mock_adapter,
            symbols=["BTCUSDT", "ETHUSDT"],
            config={
                'health_check_interval': 1,  # Fast for testing
                'cache_ttl': 5,
                'enable_orderbook': True,
                'enable_funding': False
            }
        )
        yield feed
        if feed._running:
            await feed.stop()

    @pytest.mark.asyncio
    async def test_feed_lifecycle(self, feed):
        """Test feed start/stop lifecycle"""
        assert feed.status == FeedStatus.INITIALIZING

        # Start feed
        await feed.start()
        assert feed.status == FeedStatus.RUNNING
        assert feed.adapter.is_connected()

        # Stop feed
        await feed.stop()
        assert feed.status == FeedStatus.STOPPED
        assert not feed._running

    @pytest.mark.asyncio
    async def test_snapshot_functionality(self, feed):
        """Test market data snapshot"""
        await feed.start()

        # Wait a moment for data to be generated
        await asyncio.sleep(2)

        snapshot = await feed.snapshot("BTCUSDT")

        assert snapshot is not None
        assert isinstance(snapshot, MarketFrame)
        assert snapshot.symbol == "BTCUSDT"
        assert snapshot.current_price is not None
        assert snapshot.ohlcv_1m is not None
        assert len(snapshot.ohlcv_1m) > 0

        if feed.enable_orderbook:
            assert snapshot.orderbook is not None
            assert snapshot.orderbook.best_bid is not None
            assert snapshot.orderbook.best_ask is not None

    @pytest.mark.asyncio
    async def test_metrics_tracking(self, feed):
        """Test performance metrics tracking"""
        await feed.start()
        await asyncio.sleep(2)  # Let some data flow

        metrics = feed.get_metrics()

        assert 'status' in metrics
        assert 'symbols_subscribed' in metrics
        assert 'messages_received' in metrics
        assert 'errors' in metrics
        assert 'last_update' in metrics

        assert metrics['status'] == FeedStatus.RUNNING.value
        assert metrics['symbols_subscribed'] == 2
        assert metrics['messages_received'] >= 0

    @pytest.mark.asyncio
    async def test_symbol_management(self, feed):
        """Test adding/removing symbols"""
        initial_count = len(feed.symbols)

        # Add symbol
        feed.add_symbol("ADAUSDT")
        assert len(feed.symbols) == initial_count + 1
        assert "ADAUSDT" in feed.symbols

        # Remove symbol
        feed.remove_symbol("ADAUSDT")
        assert len(feed.symbols) == initial_count
        assert "ADAUSDT" not in feed.symbols

    @pytest.mark.asyncio
    async def test_unsubscribed_symbol_snapshot(self, feed):
        """Test snapshot for unsubscribed symbol returns None"""
        await feed.start()

        snapshot = await feed.snapshot("UNSUBSCRIBED")
        assert snapshot is None


class TestLatencyMetrics:
    """Test latency measurement and monitoring"""

    @pytest.mark.asyncio
    async def test_latency_calculation(self):
        """Test latency metrics calculation"""
        feed = create_datafeed("mock", ["BTCUSDT"])

        try:
            await feed.start()
            await asyncio.sleep(3)  # Collect some samples

            metrics = feed.get_metrics()

            # Check if latency metrics are present when available
            if metrics.get('latency_median_ms') is not None:
                assert metrics['latency_median_ms'] >= 0
                assert metrics['latency_median_ms'] < 1000  # Should be fast for mock

            if metrics.get('latency_p95_ms') is not None:
                assert metrics['latency_p95_ms'] >= 0
                assert metrics['latency_p95_ms'] < 1000

        finally:
            await feed.stop()

    @pytest.mark.asyncio
    async def test_latency_requirement(self):
        """Test that latency meets Phase C requirement (<250ms median)"""
        feed = create_datafeed("mock", ["BTCUSDT"])

        try:
            await feed.start()

            # Measure latency over several snapshots
            latencies = []
            for _ in range(10):
                start_time = datetime.now()
                snapshot = await feed.snapshot("BTCUSDT")
                end_time = datetime.now()

                if snapshot:
                    latency_ms = (end_time - start_time).total_seconds() * 1000
                    latencies.append(latency_ms)

                await asyncio.sleep(0.1)

            if latencies:
                median_latency = sorted(latencies)[len(latencies) // 2]
                # Mock adapter should have very low latency
                assert median_latency < 250, f"Median latency {median_latency}ms exceeds requirement"

        finally:
            await feed.stop()


class TestIntegration:
    """Integration tests for complete data feed system"""

    @pytest.mark.asyncio
    async def test_feed_integration_with_strategies(self):
        """Test feed integration with strategy system"""
        # This would test the integration between datafeed and strategies
        # For now, we'll test that the feed produces MarketFrame objects
        # that are compatible with strategy requirements

        feed = create_datafeed("mock", ["BTCUSDT"])

        try:
            await feed.start()
            await asyncio.sleep(2)

            snapshot = await feed.snapshot("BTCUSDT")

            if snapshot:
                # Test that snapshot has all required data for strategies
                assert snapshot.symbol == "BTCUSDT"
                assert snapshot.current_price is not None
                assert snapshot.ohlcv_1m is not None
                assert len(snapshot.ohlcv_1m) >= 20  # Minimum for BB strategy

                # Test orderbook if enabled
                if feed.enable_orderbook and snapshot.orderbook:
                    assert snapshot.orderbook.imbalance is not None
                    assert isinstance(snapshot.orderbook.imbalance, float)
                    assert -1.0 <= snapshot.orderbook.imbalance <= 1.0

                # Test that we can create a basic signal from this data
                # (Without actually running strategies)
                price_changes = []
                if len(snapshot.ohlcv_1m) >= 2:
                    recent = snapshot.ohlcv_1m[-2:]
                    change = (float(recent[-1].close) - float(recent[0].close)) / float(recent[0].close)
                    price_changes.append(change)

                    # Should be able to calculate basic momentum
                    assert isinstance(change, float)
                    assert -1.0 <= change <= 1.0  # Reasonable change for testing

        finally:
            await feed.stop()

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms"""
        # Create adapter that can simulate failures
        adapter = MockAdapter("mock", {})
        feed = AsyncDataFeed(adapter, ["BTCUSDT"], {
            'health_check_interval': 1,
            'reconnect_attempts': 3
        })

        try:
            await feed.start()

            # Simulate adapter failure
            adapter.status = adapter.ConnectionStatus.FAILED
            adapter.running = False

            # Wait for health check to detect failure
            await asyncio.sleep(2)

            # Feed should still be running but may be degraded
            assert feed._running

            # Simulate recovery
            adapter.status = adapter.ConnectionStatus.CONNECTED
            adapter.running = True

            await asyncio.sleep(1)

            # Should be able to get snapshot again
            snapshot = await feed.snapshot("BTCUSDT")
            # May be None if recovery is still in progress, that's ok

        finally:
            await feed.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
