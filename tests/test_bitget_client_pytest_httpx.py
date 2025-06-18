"""
Modern test suite using pytest-httpx for Bitget API client
Demonstrates best practices for HTTP mocking without unittest.mock
"""

import pytest
import pytest_httpx
import httpx
from datetime import datetime
from decimal import Decimal

from src.project_chimera.datafeed.adapters.bitget_enhanced import BitgetEnhancedAdapter
from src.project_chimera.domains.market import Ticker, OrderBook, OHLCV


@pytest.fixture
def bitget_config():
    """Test configuration for Bitget adapter"""
    return {
        'api_key': 'test_api_key',
        'secret_key': 'test_secret_key', 
        'passphrase': 'test_passphrase',
        'sandbox': True,
        'timeout_seconds': 30
    }


@pytest.fixture
async def bitget_adapter(bitget_config):
    """Create Bitget adapter for testing"""
    adapter = BitgetEnhancedAdapter("test_bitget", bitget_config)
    yield adapter
    # Cleanup
    if adapter.http_client:
        await adapter.http_client.aclose()


class TestBitgetAdapterWithHttpx:
    """Test suite using pytest-httpx for HTTP mocking"""
    
    @pytest.mark.asyncio
    async def test_connection_success(self, bitget_adapter, httpx_mock: pytest_httpx.HTTPXMock):
        """Test successful connection to Bitget API"""
        # Mock the time endpoint for connection test
        httpx_mock.add_response(
            method="GET",
            url="https://api.bitgetapi.com/api/spot/v1/public/time",
            json={"code": "00000", "data": {"serverTime": "1700000000000"}},
            status_code=200
        )
        
        await bitget_adapter.connect()
        
        assert bitget_adapter.is_connected()
        assert await bitget_adapter.health_check()
    
    @pytest.mark.asyncio
    async def test_get_ticker_success(self, bitget_adapter, httpx_mock: pytest_httpx.HTTPXMock):
        """Test successful ticker retrieval"""
        # Mock connection
        httpx_mock.add_response(
            method="GET", 
            url="https://api.bitgetapi.com/api/spot/v1/public/time",
            json={"code": "00000", "data": {"serverTime": "1700000000000"}},
            status_code=200
        )
        
        # Mock ticker endpoint
        ticker_response = {
            "code": "00000",
            "data": {
                "last": "50000.00",
                "baseVol": "1000.50",
                "change24h": "2.5",
                "ts": "1700000000000"
            }
        }
        
        httpx_mock.add_response(
            method="GET",
            url="https://api.bitgetapi.com/api/spot/v1/market/ticker",
            json=ticker_response,
            status_code=200
        )
        
        await bitget_adapter.connect()
        ticker = await bitget_adapter.get_ticker("BTCUSDT")
        
        assert ticker is not None
        assert isinstance(ticker, Ticker)
        assert ticker.symbol == "BTCUSDT"
        assert ticker.price == Decimal("50000.00")
        assert ticker.volume_24h == Decimal("1000.50")
        assert ticker.change_24h == Decimal("2.5")
    
    @pytest.mark.asyncio
    async def test_get_orderbook_success(self, bitget_adapter, httpx_mock: pytest_httpx.HTTPXMock):
        """Test successful orderbook retrieval"""
        # Mock connection
        httpx_mock.add_response(
            method="GET",
            url="https://api.bitgetapi.com/api/spot/v1/public/time", 
            json={"code": "00000", "data": {"serverTime": "1700000000000"}},
            status_code=200
        )
        
        # Mock orderbook endpoint
        orderbook_response = {
            "code": "00000",
            "data": {
                "bids": [
                    ["49990.00", "1.0000"],
                    ["49980.00", "2.0000"]
                ],
                "asks": [
                    ["50010.00", "1.5000"],
                    ["50020.00", "0.5000"]
                ],
                "ts": "1700000000000"
            }
        }
        
        httpx_mock.add_response(
            method="GET",
            url="https://api.bitgetapi.com/api/spot/v1/market/depth",
            json=orderbook_response,
            status_code=200
        )
        
        await bitget_adapter.connect()
        orderbook = await bitget_adapter.get_orderbook("BTCUSDT", levels=20)
        
        assert orderbook is not None
        assert isinstance(orderbook, OrderBook)
        assert orderbook.symbol == "BTCUSDT"
        assert len(orderbook.bids) == 2
        assert len(orderbook.asks) == 2
        assert orderbook.bids[0] == (Decimal("49990.00"), Decimal("1.0000"))
        assert orderbook.asks[0] == (Decimal("50010.00"), Decimal("1.5000"))
    
    @pytest.mark.asyncio
    async def test_get_historical_klines(self, bitget_adapter, httpx_mock: pytest_httpx.HTTPXMock):
        """Test historical candlestick data retrieval"""
        # Mock connection
        httpx_mock.add_response(
            method="GET",
            url="https://api.bitgetapi.com/api/spot/v1/public/time",
            json={"code": "00000", "data": {"serverTime": "1700000000000"}},
            status_code=200
        )
        
        # Mock klines endpoint
        klines_response = {
            "code": "00000",
            "data": [
                ["1700000000000", "49000.00", "50000.00", "48000.00", "49500.00", "100.0"],
                ["1700000060000", "49500.00", "50500.00", "49000.00", "50000.00", "150.0"]
            ]
        }
        
        httpx_mock.add_response(
            method="GET",
            url="https://api.bitgetapi.com/api/spot/v1/market/candles",
            json=klines_response,
            status_code=200
        )
        
        await bitget_adapter.connect()
        klines = await bitget_adapter.get_historical_klines("BTCUSDT", "1m", 100)
        
        assert len(klines) == 2
        assert all(isinstance(kline, OHLCV) for kline in klines)
        
        # Check first candle
        kline = klines[0]
        assert kline.symbol == "BTCUSDT"
        assert kline.open == Decimal("49000.00")
        assert kline.high == Decimal("50000.00")
        assert kline.low == Decimal("48000.00")
        assert kline.close == Decimal("49500.00")
        assert kline.volume == Decimal("100.0")
        assert kline.timeframe == "1m"
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, bitget_adapter, httpx_mock: pytest_httpx.HTTPXMock):
        """Test API error response handling"""
        # Mock connection
        httpx_mock.add_response(
            method="GET",
            url="https://api.bitgetapi.com/api/spot/v1/public/time",
            json={"code": "00000", "data": {"serverTime": "1700000000000"}},
            status_code=200
        )
        
        # Mock error response
        httpx_mock.add_response(
            method="GET",
            url="https://api.bitgetapi.com/api/spot/v1/market/ticker",
            json={"code": "40001", "msg": "Invalid symbol"},
            status_code=200  # Bitget returns 200 even for API errors
        )
        
        await bitget_adapter.connect()
        ticker = await bitget_adapter.get_ticker("INVALID")
        
        # Should return None for invalid response
        assert ticker is None
    
    @pytest.mark.asyncio  
    async def test_http_error_handling(self, bitget_adapter, httpx_mock: pytest_httpx.HTTPXMock):
        """Test HTTP error status handling"""
        # Mock connection failure
        httpx_mock.add_response(
            method="GET",
            url="https://api.bitgetapi.com/api/spot/v1/public/time",
            status_code=500
        )
        
        with pytest.raises(Exception):
            await bitget_adapter.connect()
    
    @pytest.mark.asyncio
    async def test_rate_limiting_headers(self, bitget_adapter, httpx_mock: pytest_httpx.HTTPXMock):
        """Test proper handling of rate limiting headers"""
        # Mock connection
        httpx_mock.add_response(
            method="GET",
            url="https://api.bitgetapi.com/api/spot/v1/public/time",
            json={"code": "00000", "data": {"serverTime": "1700000000000"}},
            status_code=200
        )
        
        # Mock rate limited response
        httpx_mock.add_response(
            method="GET",
            url="https://api.bitgetapi.com/api/spot/v1/market/ticker",
            json={"code": "40005", "msg": "Rate limit exceeded"},
            status_code=429,
            headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "60"}
        )
        
        await bitget_adapter.connect()
        
        # Should handle rate limiting gracefully
        ticker = await bitget_adapter.get_ticker("BTCUSDT")
        assert ticker is None
    
    @pytest.mark.asyncio
    async def test_futures_endpoints(self, bitget_adapter, httpx_mock: pytest_httpx.HTTPXMock):
        """Test futures-specific endpoints"""
        # Mock connection
        httpx_mock.add_response(
            method="GET",
            url="https://api.bitgetapi.com/api/spot/v1/public/time",
            json={"code": "00000", "data": {"serverTime": "1700000000000"}},
            status_code=200
        )
        
        # Mock funding rate endpoint
        funding_response = {
            "code": "00000",
            "data": {
                "fundingRate": "0.0001",
                "nextFundingTime": "1700003600000",
                "fundingTime": "1700000000000"
            }
        }
        
        httpx_mock.add_response(
            method="GET",
            url="https://api.bitgetapi.com/api/mix/v1/market/current-fundRate",
            json=funding_response,
            status_code=200
        )
        
        # Mock open interest endpoint
        oi_response = {
            "code": "00000",
            "data": {
                "openInterest": "123456.78"
            }
        }
        
        httpx_mock.add_response(
            method="GET",
            url="https://api.bitgetapi.com/api/mix/v1/market/open-interest",
            json=oi_response,
            status_code=200
        )
        
        await bitget_adapter.connect()
        
        # Test funding rate
        funding = await bitget_adapter.get_funding_rate("BTCUSDT")
        assert funding is not None
        assert funding.symbol == "BTCUSDT"
        assert funding.rate == Decimal("0.0001")
        
        # Test open interest
        oi = await bitget_adapter.get_open_interest("BTCUSDT")
        assert oi is not None
        assert oi == 123456.78


class TestParameterizedRequests:
    """Test various request parameter combinations"""
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("symbol,interval,limit", [
        ("BTCUSDT", "1m", 100),
        ("ETHUSDT", "5m", 50),
        ("ADAUSDT", "1h", 200)
    ])
    async def test_klines_parameters(self, bitget_adapter, httpx_mock: pytest_httpx.HTTPXMock, 
                                   symbol, interval, limit):
        """Test klines with different parameters"""
        # Mock connection
        httpx_mock.add_response(
            method="GET",
            url="https://api.bitgetapi.com/api/spot/v1/public/time",
            json={"code": "00000", "data": {"serverTime": "1700000000000"}},
            status_code=200
        )
        
        # Mock klines response
        httpx_mock.add_response(
            method="GET",
            url="https://api.bitgetapi.com/api/spot/v1/market/candles",
            json={"code": "00000", "data": []},
            status_code=200
        )
        
        await bitget_adapter.connect()
        klines = await bitget_adapter.get_historical_klines(symbol, interval, limit)
        
        # Verify the request was made with correct parameters
        assert len(klines) == 0  # Empty response for test
        
        # Check that the mock was called with expected parameters
        request = httpx_mock.get_requests()[-1]  # Get last request
        assert symbol in str(request.url) or symbol in str(request.content)


class TestConcurrentRequests:
    """Test concurrent request handling"""
    
    @pytest.mark.asyncio
    async def test_concurrent_ticker_requests(self, bitget_adapter, httpx_mock: pytest_httpx.HTTPXMock):
        """Test multiple concurrent ticker requests"""
        symbols = ["BTCUSDT", "ETHUSDT", "ADAUSDT"]
        
        # Mock connection
        httpx_mock.add_response(
            method="GET",
            url="https://api.bitgetapi.com/api/spot/v1/public/time",
            json={"code": "00000", "data": {"serverTime": "1700000000000"}},
            status_code=200
        )
        
        # Mock ticker responses for each symbol
        for i, symbol in enumerate(symbols):
            ticker_response = {
                "code": "00000",
                "data": {
                    "last": f"{(i+1)*10000}.00",
                    "baseVol": f"{(i+1)*100}.00",
                    "change24h": f"{i+1}.0",
                    "ts": "1700000000000"
                }
            }
            
            httpx_mock.add_response(
                method="GET",
                url="https://api.bitgetapi.com/api/spot/v1/market/ticker",
                json=ticker_response,
                status_code=200
            )
        
        await bitget_adapter.connect()
        
        # Make concurrent requests
        import asyncio
        tasks = [bitget_adapter.get_ticker(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        
        # Verify all requests succeeded
        assert len(results) == 3
        assert all(ticker is not None for ticker in results)
        
        for i, ticker in enumerate(results):
            expected_price = Decimal(f"{(i+1)*10000}.00")
            assert ticker.price == expected_price


@pytest.fixture
def httpx_client():
    """Create httpx client for direct HTTP testing"""
    return httpx.AsyncClient()


class TestDirectHttpxUsage:
    """Test direct httpx usage patterns for reference"""
    
    @pytest.mark.asyncio
    async def test_direct_httpx_request(self, httpx_client, httpx_mock: pytest_httpx.HTTPXMock):
        """Example of direct httpx testing"""
        mock_response = {"message": "Hello, world!"}
        
        httpx_mock.add_response(
            method="GET",
            url="https://api.example.com/test",
            json=mock_response,
            status_code=200
        )
        
        response = await httpx_client.get("https://api.example.com/test")
        
        assert response.status_code == 200
        assert response.json() == mock_response
    
    @pytest.mark.asyncio
    async def test_request_matching_patterns(self, httpx_client, httpx_mock: pytest_httpx.HTTPXMock):
        """Test various request matching patterns"""
        # Match exact URL
        httpx_mock.add_response(url="https://api.example.com/exact", json={"type": "exact"})
        
        # Match URL pattern
        httpx_mock.add_response(url__regex=r"https://api\.example\.com/users/\d+", json={"type": "pattern"})
        
        # Match with query parameters
        httpx_mock.add_response(
            url="https://api.example.com/search",
            match_query_params={"q": "test"},
            json={"type": "query"}
        )
        
        # Test exact match
        response = await httpx_client.get("https://api.example.com/exact")
        assert response.json()["type"] == "exact"
        
        # Test pattern match  
        response = await httpx_client.get("https://api.example.com/users/123")
        assert response.json()["type"] == "pattern"
        
        # Test query parameter match
        response = await httpx_client.get("https://api.example.com/search?q=test")
        assert response.json()["type"] == "query"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])