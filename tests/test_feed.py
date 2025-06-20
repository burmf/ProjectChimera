"""
Tests for Bitget WebSocket Feed Implementation - FEED-02
Comprehensive test coverage for real-time data streams
"""

import pytest
import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from src.project_chimera.datafeed.bitget_ws import (
    BitgetWebSocketFeed, 
    LatencyMetrics,
    create_bitget_ws_feed
)
from src.project_chimera.datafeed.protocols import ConnectionStatus


class TestLatencyMetrics:
    """Test latency tracking functionality"""
    
    def test_initial_state(self):
        """Test initial latency metrics state"""
        metrics = LatencyMetrics()
        assert metrics.avg_latency_ms == 0.0
        assert metrics.max_latency_ms == 0.0
        assert metrics.message_count == 0
    
    def test_first_latency_update(self):
        """Test first latency measurement"""
        metrics = LatencyMetrics()
        ping_time = time.time()
        pong_time = ping_time + 0.05  # 50ms latency
        
        metrics.update_latency(ping_time, pong_time)
        
        assert metrics.avg_latency_ms == 50.0
        assert metrics.max_latency_ms == 50.0
        assert metrics.message_count == 1
    
    def test_multiple_latency_updates(self):
        """Test exponential moving average calculation"""
        metrics = LatencyMetrics()
        
        # First measurement: 50ms
        ping1 = time.time()
        pong1 = ping1 + 0.05
        metrics.update_latency(ping1, pong1)
        
        # Second measurement: 100ms
        ping2 = time.time()
        pong2 = ping2 + 0.1
        metrics.update_latency(ping2, pong2)
        
        # EMA should be: 0.1 * 100 + 0.9 * 50 = 55ms
        assert abs(metrics.avg_latency_ms - 55.0) < 0.1
        assert metrics.max_latency_ms == 100.0
        assert metrics.message_count == 2


class TestBitgetWebSocketFeed:
    """Test Bitget WebSocket feed implementation"""
    
    @pytest.fixture
    def feed_config(self):
        """Standard feed configuration for testing"""
        return {
            'api_key': 'test_key',
            'secret_key': 'test_secret', 
            'passphrase': 'test_passphrase',
            'sandbox': True
        }
    
    @pytest.fixture
    def ws_feed(self, feed_config):
        """Create WebSocket feed instance for testing"""
        return BitgetWebSocketFeed("test_feed", feed_config)
    
    def test_initialization(self, ws_feed):
        """Test feed initialization with correct settings"""
        assert ws_feed.name == "test_feed"
        assert ws_feed.sandbox is True
        assert ws_feed.status == ConnectionStatus.DISCONNECTED
        assert len(ws_feed.spot_channels) == 0
        assert len(ws_feed.mix_channels) == 0
        assert ws_feed.connection_failures == 0
    
    def test_futures_symbol_detection(self, ws_feed):
        """Test futures symbol identification logic"""
        assert ws_feed._is_futures_symbol("BTCUSDT") is True
        assert ws_feed._is_futures_symbol("ETHUSDC") is True
        assert ws_feed._is_futures_symbol("BTC_UMCBL") is True
        assert ws_feed._is_futures_symbol("BTCEUR") is False
    
    def test_circuit_breaker_activation(self, ws_feed):
        """Test circuit breaker protection mechanism"""
        # Initially not active
        assert ws_feed._is_circuit_breaker_active() is False
        
        # Trigger circuit breaker
        ws_feed.circuit_breaker_until = time.time() + 300
        assert ws_feed._is_circuit_breaker_active() is True
        
        # Should deactivate after time passes
        ws_feed.circuit_breaker_until = time.time() - 1
        assert ws_feed._is_circuit_breaker_active() is False
    
    @pytest.mark.asyncio
    async def test_connect_circuit_breaker_blocked(self, ws_feed):
        """Test connection blocked by circuit breaker"""
        ws_feed.circuit_breaker_until = time.time() + 300
        
        with pytest.raises(ConnectionError, match="Circuit breaker active"):
            await ws_feed.connect()
    
    @pytest.mark.asyncio
    @patch('websockets.connect')
    @patch('httpx.AsyncClient')
    async def test_successful_connection(self, mock_http, mock_ws, ws_feed):
        """Test successful WebSocket connection"""
        # Mock HTTP client
        mock_http_instance = AsyncMock()
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_http_instance.get.return_value = mock_response
        mock_http.return_value = mock_http_instance
        
        # Mock WebSocket connections
        mock_spot_ws = AsyncMock()
        mock_mix_ws = AsyncMock()
        mock_ws.side_effect = [mock_spot_ws, mock_mix_ws]
        
        await ws_feed.connect()
        
        assert ws_feed.status == ConnectionStatus.CONNECTED
        assert ws_feed.connection_failures == 0
        assert ws_feed.http_client is not None
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_connection_failure_increments_counter(self, mock_http, ws_feed):
        """Test connection failure handling"""
        # Mock HTTP client to raise exception
        mock_http.side_effect = Exception("Connection failed")
        
        with pytest.raises(Exception):
            await ws_feed.connect()
        
        assert ws_feed.connection_failures == 1
        assert ws_feed.status == ConnectionStatus.FAILED
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_circuit_breaker_activation_after_failures(self, mock_http, ws_feed):
        """Test circuit breaker activation after max failures"""
        # Mock HTTP client to always fail
        mock_http.side_effect = Exception("Connection failed")
        
        # Fail 3 times to trigger circuit breaker
        for i in range(3):
            with pytest.raises(Exception):
                await ws_feed.connect()
        
        assert ws_feed.connection_failures == 3
        assert ws_feed._is_circuit_breaker_active() is True
    
    @pytest.mark.asyncio
    async def test_graceful_disconnect(self, ws_feed):
        """Test graceful disconnection"""
        # Mock connections
        ws_feed.spot_ws = AsyncMock()
        ws_feed.spot_ws.closed = False
        ws_feed.mix_ws = AsyncMock()
        ws_feed.mix_ws.closed = False
        ws_feed.http_client = AsyncMock()
        
        await ws_feed.disconnect()
        
        # Verify all connections were closed
        ws_feed.spot_ws.close.assert_called_once()
        ws_feed.mix_ws.close.assert_called_once()
        ws_feed.http_client.aclose.assert_called_once()
        assert ws_feed.status == ConnectionStatus.DISCONNECTED
    
    @pytest.mark.asyncio
    async def test_spot_channel_subscription(self, ws_feed):
        """Test spot channel subscription"""
        # Mock WebSocket
        mock_ws = AsyncMock()
        ws_feed.spot_ws = mock_ws
        
        await ws_feed._subscribe_spot_channel("ticker.BTCUSDT")
        
        # Verify subscription message was sent
        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["op"] == "subscribe"
        assert "ticker.BTCUSDT" in ws_feed.spot_channels
    
    @pytest.mark.asyncio
    async def test_mix_channel_subscription(self, ws_feed):
        """Test futures channel subscription"""
        # Mock WebSocket
        mock_ws = AsyncMock()
        ws_feed.mix_ws = mock_ws
        
        await ws_feed._subscribe_mix_channel("fundingRate.BTCUSDT")
        
        # Verify subscription message was sent
        mock_ws.send.assert_called_once()
        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["op"] == "subscribe"
        assert "fundingRate.BTCUSDT" in ws_feed.mix_channels
    
    @pytest.mark.asyncio
    async def test_duplicate_subscription_ignored(self, ws_feed):
        """Test that duplicate subscriptions are ignored"""
        # Mock WebSocket
        mock_ws = AsyncMock()
        ws_feed.spot_ws = mock_ws
        
        # Subscribe twice to same channel
        await ws_feed._subscribe_spot_channel("ticker.BTCUSDT")
        await ws_feed._subscribe_spot_channel("ticker.BTCUSDT")
        
        # Should only send one subscription
        assert mock_ws.send.call_count == 1
        assert len(ws_feed.spot_channels) == 1
    
    def test_message_processing_ticker(self, ws_feed):
        """Test ticker message processing"""
        ticker_data = [{
            "instId": "BTCUSDT",
            "last": "50000.0",
            "bidPx": "49990.0", 
            "askPx": "50010.0",
            "vol24h": "1000.0",
            "change24h": "0.05"
        }]
        
        # Process ticker message
        asyncio.run(ws_feed._handle_ticker(ticker_data, {}))
        
        # Verify ticker was stored
        assert "BTCUSDT" in ws_feed.market_data
        ticker = ws_feed.market_data["BTCUSDT"]["ticker"]
        assert ticker.last_price == 50000.0
        assert ticker.bid_price == 49990.0
        assert ticker.ask_price == 50010.0
    
    def test_message_processing_orderbook(self, ws_feed):
        """Test orderbook message processing"""
        book_data = [{
            "instId": "BTCUSDT",
            "bids": [["49990.0", "1.5"], ["49980.0", "2.0"]],
            "asks": [["50010.0", "1.2"], ["50020.0", "1.8"]]
        }]
        
        # Process orderbook message
        asyncio.run(ws_feed._handle_orderbook(book_data, {}))
        
        # Verify orderbook was stored
        assert "BTCUSDT" in ws_feed.market_data
        orderbook = ws_feed.market_data["BTCUSDT"]["orderbook"]
        assert len(orderbook.bids) == 2
        assert len(orderbook.asks) == 2
        assert orderbook.bids[0][0] == 49990.0  # Best bid price
        assert orderbook.asks[0][0] == 50010.0  # Best ask price
    
    def test_message_processing_funding_rate(self, ws_feed):
        """Test funding rate message processing"""
        funding_data = [{
            "instId": "BTCUSDT",
            "fundingRate": "0.0001",
            "fundingTime": "2024-01-01T08:00:00.000Z"
        }]
        
        # Process funding rate message
        asyncio.run(ws_feed._handle_funding_rate(funding_data, {}))
        
        # Verify funding rate was stored
        assert "BTCUSDT" in ws_feed.funding_rates
        funding = ws_feed.funding_rates["BTCUSDT"]
        assert funding.rate == 0.0001
    
    def test_message_processing_open_interest(self, ws_feed):
        """Test open interest message processing"""
        oi_data = [{
            "instId": "BTCUSDT", 
            "oi": "1000000.0"
        }]
        
        # Process open interest message
        asyncio.run(ws_feed._handle_open_interest(oi_data, {}))
        
        # Verify open interest was stored
        assert "BTCUSDT" in ws_feed.open_interest
        assert ws_feed.open_interest["BTCUSDT"] == 1000000.0
    
    @pytest.mark.asyncio
    async def test_pong_latency_update(self, ws_feed):
        """Test latency update from pong messages"""
        ping_time = time.time()
        pong_time = ping_time + 0.05  # 50ms latency
        
        pong_message = json.dumps({
            "event": "pong",
            "ping_time": ping_time
        })
        
        await ws_feed._process_message(pong_message, "spot", pong_time)
        
        # Verify latency was updated
        assert ws_feed.spot_latency.avg_latency_ms == 50.0
        assert ws_feed.spot_latency.message_count == 1
    
    @pytest.mark.asyncio
    async def test_invalid_json_handling(self, ws_feed):
        """Test handling of invalid JSON messages"""
        invalid_json = "{'invalid': json message}"
        
        # Should not raise exception
        await ws_feed._process_message(invalid_json, "spot", time.time())
        
        # No market data should be created
        assert len(ws_feed.market_data) == 0


class TestBitgetSubscriptionMethods:
    """Test high-level subscription methods"""
    
    @pytest.fixture
    def ws_feed(self):
        """Create WebSocket feed for subscription testing"""
        config = {'sandbox': True}
        return BitgetWebSocketFeed("test", config)
    
    @pytest.mark.asyncio
    async def test_subscribe_ticker_spot(self, ws_feed):
        """Test ticker subscription for spot symbol"""
        with patch.object(ws_feed, '_subscribe_spot_channel') as mock_spot:
            await ws_feed.subscribe_ticker("BTCEUR")
            mock_spot.assert_called_once_with("ticker.BTCEUR")
    
    @pytest.mark.asyncio
    async def test_subscribe_ticker_futures(self, ws_feed):
        """Test ticker subscription for futures symbol"""
        with patch.object(ws_feed, '_subscribe_mix_channel') as mock_mix:
            await ws_feed.subscribe_ticker("BTCUSDT")
            mock_mix.assert_called_once_with("ticker.BTCUSDT")
    
    @pytest.mark.asyncio
    async def test_subscribe_orderbook_depth_selection(self, ws_feed):
        """Test orderbook subscription depth selection"""
        with patch.object(ws_feed, '_subscribe_spot_channel') as mock_spot:
            # 5 levels should use books5
            await ws_feed.subscribe_orderbook("BTCEUR", levels=5)
            mock_spot.assert_called_once_with("books5.BTCEUR")
            
            mock_spot.reset_mock()
            
            # 20 levels should use books
            await ws_feed.subscribe_orderbook("BTCEUR", levels=20)
            mock_spot.assert_called_once_with("books.BTCEUR")
    
    @pytest.mark.asyncio
    async def test_subscribe_funding_spot_warning(self, ws_feed):
        """Test funding subscription warning for spot symbols"""
        # Should not raise exception, but log warning
        await ws_feed.subscribe_funding("BTCEUR")
        
        # No channels should be subscribed
        assert len(ws_feed.mix_channels) == 0
    
    @pytest.mark.asyncio
    async def test_subscribe_funding_futures(self, ws_feed):
        """Test funding subscription for futures symbols"""
        with patch.object(ws_feed, '_subscribe_mix_channel') as mock_mix:
            await ws_feed.subscribe_funding("BTCUSDT")
            mock_mix.assert_called_once_with("fundingRate.BTCUSDT")


def test_factory_function():
    """Test create_bitget_ws_feed factory function"""
    config = {'sandbox': True, 'api_key': 'test'}
    feed = create_bitget_ws_feed(config)
    
    assert isinstance(feed, BitgetWebSocketFeed)
    assert feed.name == "bitget_ws"
    assert feed.sandbox is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])