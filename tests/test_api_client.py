"""
Professional test suite for AsyncBitgetClient
Comprehensive testing with pytest-httpx fixtures instead of mocks
"""

import pytest
import asyncio
import httpx
import pytest_httpx
from datetime import datetime

from project_chimera.core.api_client import (
    AsyncBitgetClient,
    TickerData,
    OrderResult,
    OrderSide,
    OrderType,
    APIException,
    ConnectionException,
    AuthenticationException,
    RateLimitException
)
from project_chimera.config import Settings


@pytest.fixture
def mock_settings():
    """Mock settings for testing"""
    settings = Settings()
    settings.api.bitget_api_key = "test_key"
    settings.api.bitget_secret_key = "test_secret"
    settings.api.bitget_passphrase = "test_passphrase"
    settings.api.bitget_sandbox = True
    settings.api.timeout_seconds = 5
    settings.api.max_retries = 2
    return settings


@pytest.fixture
async def api_client(mock_settings):
    """Create API client for testing"""
    client = AsyncBitgetClient(mock_settings)
    yield client
    await client.close()


class TestAsyncBitgetClient:
    """Test suite for AsyncBitgetClient"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mock_settings):
        """Test client initialization"""
        client = AsyncBitgetClient(mock_settings)
        
        assert client.settings == mock_settings
        assert client.api_config.bitget_api_key == "test_key"
        assert client.base_url == "https://api.bitget.com"
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_signature_generation(self, api_client):
        """Test API signature generation"""
        timestamp = "1234567890000"
        method = "GET"
        path = "/api/test"
        
        signature = api_client._generate_signature(timestamp, method, path)
        
        assert isinstance(signature, str)
        assert len(signature) > 0
        # Signature should be base64 encoded
        import base64
        try:
            base64.b64decode(signature)
            assert True
        except Exception:
            assert False, "Signature is not valid base64"
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, api_client):
        """Test rate limiting functionality"""
        import time
        
        start_time = time.time()
        
        # Make multiple requests quickly
        await api_client._rate_limit()
        await api_client._rate_limit()
        
        elapsed = time.time() - start_time
        
        # Should have delayed due to rate limiting
        assert elapsed >= api_client.api_config.rate_limit_delay
    
    @pytest.mark.asyncio
    async def test_get_futures_ticker_success(self, api_client, httpx_mock: pytest_httpx.HTTPXMock):
        """Test successful ticker data retrieval using pytest-httpx"""
        mock_response = {
            "code": "00000",
            "data": [{
                "lastPr": "50000.00",
                "open24h": "49000.00",
                "high24h": "51000.00",
                "low24h": "48000.00",
                "baseVolume": "1000.00",
                "change24h": "2.04",
                "askPr": "50001.00",
                "bidPr": "49999.00"
            }]
        }
        
        # Mock the HTTP request using pytest-httpx
        httpx_mock.add_response(
            method="GET",
            url="https://api.bitget.com/api/mix/v1/market/ticker",
            json=mock_response,
            status_code=200
        )
        
        ticker = await api_client.get_futures_ticker('BTCUSDT')
        
        assert isinstance(ticker, TickerData)
        assert ticker.symbol == 'BTCUSDT'
        assert ticker.price == 50000.00
        assert ticker.change_24h == 2.04
        assert ticker.spread == 2.00  # askPr - bidPr
    
    @pytest.mark.asyncio
    async def test_get_futures_ticker_no_data(self, api_client, httpx_mock: pytest_httpx.HTTPXMock):
        """Test ticker retrieval with no data using pytest-httpx"""
        mock_response = {
            "code": "00000",
            "data": []
        }
        
        httpx_mock.add_response(
            method="GET",
            url="https://api.bitget.com/api/mix/v1/market/ticker",
            json=mock_response,
            status_code=200
        )
        
        ticker = await api_client.get_futures_ticker('INVALID')
        
        assert ticker is None
    
    @pytest.mark.asyncio
    async def test_get_multiple_tickers(self, api_client):
        """Test multiple ticker retrieval"""
        mock_ticker_data = TickerData(
            symbol="BTCUSDT",
            price=50000.0,
            open_24h=49000.0,
            high_24h=51000.0,
            low_24h=48000.0,
            volume=1000.0,
            change_24h=2.04,
            ask_price=50001.0,
            bid_price=49999.0,
            spread=2.0,
            timestamp=datetime.now()
        )
        
        with patch.object(api_client, 'get_futures_ticker', return_value=mock_ticker_data):
            tickers = await api_client.get_multiple_tickers(['BTCUSDT', 'ETHUSDT'])
            
            assert len(tickers) == 2
            assert 'BTCUSDT' in tickers
            assert 'ETHUSDT' in tickers
            assert isinstance(tickers['BTCUSDT'], TickerData)
    
    @pytest.mark.asyncio
    async def test_place_order_success(self, api_client, httpx_mock: pytest_httpx.HTTPXMock):
        """Test successful order placement using pytest-httpx"""
        mock_response = {
            "code": "00000",
            "data": {
                "orderId": "123456789",
                "clientOid": "client_123"
            }
        }
        
        httpx_mock.add_response(
            method="POST",
            url="https://api.bitget.com/api/mix/v1/order/placeOrder",
            json=mock_response,
            status_code=200
        )
        
        result = await api_client.place_order(
            symbol='BTCUSDT',
            side=OrderSide.LONG,
            size=0.01,
            order_type=OrderType.MARKET
        )
        
        assert isinstance(result, OrderResult)
        assert result.order_id == "123456789"
        assert result.symbol == "BTCUSDT"
        assert result.side == OrderSide.LONG
        assert result.size == 0.01
    
    @pytest.mark.asyncio
    async def test_place_order_with_price(self, api_client):
        """Test limit order placement with price"""
        mock_response = {
            "code": "00000",
            "data": {
                "orderId": "123456789",
                "clientOid": None
            }
        }
        
        with patch.object(api_client, '_make_request', return_value=mock_response):
            result = await api_client.place_order(
                symbol='BTCUSDT',
                side=OrderSide.SHORT,
                size=0.01,
                order_type=OrderType.LIMIT,
                price=49000.0
            )
            
            assert result is not None
            assert result.order_type == OrderType.LIMIT
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, api_client, httpx_mock: pytest_httpx.HTTPXMock):
        """Test API error handling using pytest-httpx"""
        # Test different error responses
        error_test_cases = [
            (401, AuthenticationException, {"error": "Unauthorized"}),
            (429, RateLimitException, {"error": "Rate limit exceeded"}),
            (500, ConnectionException, {"error": "Internal server error"}),
            (400, APIException, {"error": "Bad request"})
        ]
        
        for status_code, expected_exception, error_body in error_test_cases:
            # Clear previous mocks
            httpx_mock.reset(assert_all_responses_have_been_requested=False)
            
            httpx_mock.add_response(
                method="GET",
                url="https://api.bitget.com/test",
                json=error_body,
                status_code=status_code
            )
            
            with pytest.raises(expected_exception):
                await api_client._make_request('GET', '/test')
    
    @pytest.mark.asyncio
    async def test_connection_error(self, api_client):
        """Test connection error handling"""
        with patch.object(api_client.client, 'request', side_effect=httpx.RequestError("Connection failed")):
            with pytest.raises(ConnectionException):
                await api_client._make_request('GET', '/test')
    
    @pytest.mark.asyncio
    async def test_invalid_json_response(self, api_client):
        """Test invalid JSON response handling"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        
        with patch.object(api_client.client, 'request', return_value=mock_response):
            with pytest.raises(APIException):
                await api_client._make_request('GET', '/test')
    
    @pytest.mark.asyncio
    async def test_api_response_error_code(self, api_client):
        """Test API response with error code"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "code": "40001",
            "msg": "Invalid parameter"
        }
        
        with patch.object(api_client.client, 'request', return_value=mock_response):
            with pytest.raises(APIException) as exc_info:
                await api_client._make_request('GET', '/test')
            
            assert "Invalid parameter" in str(exc_info.value)
            assert exc_info.value.code == "40001"
    
    @pytest.mark.asyncio
    async def test_get_klines(self, api_client):
        """Test kline data retrieval"""
        mock_response = {
            "code": "00000",
            "data": [
                ["1609459200000", "29000.00", "29500.00", "28500.00", "29200.00", "100.00"],
                ["1609462800000", "29200.00", "29800.00", "29000.00", "29600.00", "150.00"]
            ]
        }
        
        with patch.object(api_client, '_make_request', return_value=mock_response):
            klines = await api_client.get_klines('BTCUSDT', '1m', 2)
            
            assert len(klines) == 2
            assert klines[0]['open'] == 29000.00
            assert klines[0]['high'] == 29500.00
            assert klines[0]['low'] == 28500.00
            assert klines[0]['close'] == 29200.00
            assert klines[0]['volume'] == 100.00
            assert isinstance(klines[0]['timestamp'], datetime)
    
    @pytest.mark.asyncio
    async def test_get_account_balance(self, api_client):
        """Test account balance retrieval"""
        mock_response = {
            "code": "00000",
            "data": [{
                "marginCoin": "USDT",
                "available": "10000.00",
                "frozen": "500.00",
                "equity": "10500.00",
                "unrealizedPL": "100.00",
                "marginRatio": "0.05"
            }]
        }
        
        with patch.object(api_client, '_make_request', return_value=mock_response):
            balance = await api_client.get_account_balance()
            
            assert balance is not None
            assert 'USDT' in balance
            assert balance['USDT']['available'] == 10000.00
            assert balance['USDT']['equity'] == 10500.00
            assert balance['USDT']['unrealized_pnl'] == 100.00
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, api_client):
        """Test WebSocket connection (mocked)"""
        mock_websocket = AsyncMock()
        
        with patch('websockets.connect', return_value=mock_websocket):
            await api_client.start_websocket()
            
            assert api_client._ws_connection == mock_websocket
    
    @pytest.mark.asyncio
    async def test_context_manager(self, mock_settings):
        """Test async context manager usage"""
        async with AsyncBitgetClient(mock_settings) as client:
            assert isinstance(client, AsyncBitgetClient)
            assert client.client is not None
        
        # Client should be closed after exiting context
        assert client.client.is_closed


class TestRetryMechanism:
    """Test retry mechanism functionality"""
    
    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self, api_client):
        """Test retry mechanism on rate limit"""
        call_count = 0
        
        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                mock_response = MagicMock()
                mock_response.status_code = 429
                return mock_response
            else:
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"code": "00000", "data": {}}
                return mock_response
        
        with patch.object(api_client.client, 'request', side_effect=mock_request):
            result = await api_client._make_request('GET', '/test', authenticated=False)
            
            assert call_count == 3  # Should retry twice before success
            assert result["code"] == "00000"
    
    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self, api_client):
        """Test behavior when max retries exceeded"""
        async def mock_request(*args, **kwargs):
            mock_response = MagicMock()
            mock_response.status_code = 429
            return mock_response
        
        with patch.object(api_client.client, 'request', side_effect=mock_request):
            with pytest.raises(RateLimitException):
                await api_client._make_request('GET', '/test', authenticated=False)


@pytest.mark.integration
class TestIntegrationScenarios:
    """Integration test scenarios"""
    
    @pytest.mark.asyncio
    async def test_complete_trading_flow(self, api_client):
        """Test complete trading flow simulation"""
        # Mock responses for each step
        ticker_response = {
            "code": "00000",
            "data": [{
                "lastPr": "50000.00",
                "open24h": "49000.00",
                "high24h": "51000.00",
                "low24h": "48000.00",
                "baseVolume": "1000.00",
                "change24h": "2.04",
                "askPr": "50001.00",
                "bidPr": "49999.00"
            }]
        }
        
        balance_response = {
            "code": "00000",
            "data": [{
                "marginCoin": "USDT",
                "available": "10000.00",
                "frozen": "500.00",
                "equity": "10500.00",
                "unrealizedPL": "100.00"
            }]
        }
        
        order_response = {
            "code": "00000",
            "data": {
                "orderId": "123456789",
                "clientOid": "client_123"
            }
        }
        
        responses = [ticker_response, balance_response, order_response]
        response_iter = iter(responses)
        
        with patch.object(api_client, '_make_request', side_effect=lambda *args, **kwargs: next(response_iter)):
            # Step 1: Get market data
            ticker = await api_client.get_futures_ticker('BTCUSDT')
            assert ticker.price == 50000.00
            
            # Step 2: Check account balance
            balance = await api_client.get_account_balance()
            assert balance['USDT']['available'] == 10000.00
            
            # Step 3: Place order
            order = await api_client.place_order(
                symbol='BTCUSDT',
                side=OrderSide.LONG,
                size=0.01
            )
            assert order.order_id == "123456789"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])