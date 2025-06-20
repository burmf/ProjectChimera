"""
Comprehensive tests for Bitget execution engine - targeting high coverage improvement
Tests for BitgetConfig, Order, Fill, CircuitBreaker, and BitgetExecutionEngine
"""

import pytest
import asyncio
import time
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from project_chimera.execution.bitget import (
    BitgetConfig, Order, Fill, CircuitBreaker, BitgetExecutionEngine,
    OrderStatus, OrderSide, OrderType, MockHTTPClient, MockResponse, MockWebSocket
)


class TestEnums:
    """Test enumeration classes"""
    
    def test_order_status_values(self):
        """Test OrderStatus enum values"""
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.OPEN.value == "open"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.REJECTED.value == "rejected"
    
    def test_order_side_values(self):
        """Test OrderSide enum values"""
        assert OrderSide.BUY.value == "buy"
        assert OrderSide.SELL.value == "sell"
    
    def test_order_type_values(self):
        """Test OrderType enum values"""
        assert OrderType.MARKET.value == "market"
        assert OrderType.LIMIT.value == "limit"


class TestBitgetConfig:
    """Test BitgetConfig class"""
    
    def test_config_creation(self):
        """Test basic config creation"""
        config = BitgetConfig(
            api_key="test_key",
            secret_key="test_secret",
            passphrase="test_pass"
        )
        
        assert config.api_key == "test_key"
        assert config.secret_key == "test_secret"
        assert config.passphrase == "test_pass"
        assert config.sandbox is True  # default
        assert config.timeout_seconds == 30  # default
    
    def test_config_sandbox_urls(self):
        """Test sandbox URL configuration"""
        config = BitgetConfig(
            api_key="test_key",
            secret_key="test_secret",
            passphrase="test_pass",
            sandbox=True
        )
        
        # Post-init should set sandbox URLs
        assert "bitget.com" in config.base_url
        assert "bitget.com" in config.ws_url
    
    def test_config_production_urls(self):
        """Test production URL configuration"""
        config = BitgetConfig(
            api_key="test_key",
            secret_key="test_secret",
            passphrase="test_pass",
            sandbox=False,
            base_url="https://api.bitget.com",
            ws_url="wss://ws.bitget.com/spot/v1/stream"
        )
        
        assert config.base_url == "https://api.bitget.com"
        assert config.ws_url == "wss://ws.bitget.com/spot/v1/stream"
    
    @patch('project_chimera.execution.bitget.get_settings')
    def test_config_from_settings(self, mock_settings):
        """Test creating config from application settings"""
        # Mock settings structure
        mock_api_config = MagicMock()
        mock_api_config.bitget_key.get_secret_value.return_value = "settings_key"
        mock_api_config.bitget_secret.get_secret_value.return_value = "settings_secret"
        mock_api_config.bitget_passphrase.get_secret_value.return_value = "settings_pass"
        mock_api_config.bitget_sandbox = False
        mock_api_config.bitget_rest_url = "https://api.bitget.com"
        mock_api_config.bitget_ws_spot_url = "wss://ws.bitget.com/spot/v1/stream"
        mock_api_config.timeout_seconds = 60
        mock_api_config.max_retries = 5
        mock_api_config.retry_delay = 2.0
        
        mock_settings_obj = MagicMock()
        mock_settings_obj.api = mock_api_config
        mock_settings.return_value = mock_settings_obj
        
        config = BitgetConfig.from_settings()
        
        assert config.api_key == "settings_key"
        assert config.secret_key == "settings_secret"
        assert config.passphrase == "settings_pass"
        assert config.sandbox is False
        assert config.timeout_seconds == 60


class TestOrder:
    """Test Order dataclass"""
    
    @pytest.fixture
    def sample_order(self):
        """Sample order for testing"""
        return Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            size=0.1,
            price=50000.0,
            client_order_id="test_order_123"
        )
    
    def test_order_creation(self, sample_order):
        """Test order creation and properties"""
        assert sample_order.symbol == "BTCUSDT"
        assert sample_order.side == OrderSide.BUY
        assert sample_order.order_type == OrderType.LIMIT
        assert sample_order.size == 0.1
        assert sample_order.price == 50000.0
        assert sample_order.client_order_id == "test_order_123"
        
        # Default values
        assert sample_order.order_id is None
        assert sample_order.status == OrderStatus.PENDING
        assert sample_order.filled_size == 0.0
        assert sample_order.avg_price is None
    
    def test_order_to_dict(self, sample_order):
        """Test order conversion to dictionary"""
        order_dict = sample_order.to_dict()
        
        assert order_dict['symbol'] == "BTCUSDT"
        assert order_dict['side'] == "buy"  # Enum value
        assert order_dict['order_type'] == "limit"  # Enum value
        assert order_dict['status'] == "pending"  # Enum value
        assert order_dict['size'] == 0.1
        assert order_dict['price'] == 50000.0
    
    def test_order_market_type(self):
        """Test market order creation"""
        market_order = Order(
            symbol="ETHUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            size=1.0
        )
        
        assert market_order.order_type == OrderType.MARKET
        assert market_order.price is None  # Market orders don't have price
    
    def test_order_status_tracking(self, sample_order):
        """Test order status tracking"""
        # Simulate order lifecycle
        sample_order.order_id = "12345"
        sample_order.status = OrderStatus.OPEN
        sample_order.created_time = datetime.now()
        
        assert sample_order.order_id == "12345"
        assert sample_order.status == OrderStatus.OPEN
        assert sample_order.created_time is not None
        
        # Simulate partial fill
        sample_order.filled_size = 0.05
        sample_order.status = OrderStatus.FILLED
        sample_order.avg_price = 50100.0
        sample_order.updated_time = datetime.now()
        
        assert sample_order.filled_size == 0.05
        assert sample_order.avg_price == 50100.0


class TestFill:
    """Test Fill dataclass"""
    
    @pytest.fixture
    def sample_fill(self):
        """Sample fill for testing"""
        return Fill(
            order_id="12345",
            trade_id="trade_67890",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            size=0.1,
            price=50000.0,
            fee=0.001,
            fee_currency="USDT",
            timestamp=datetime.now()
        )
    
    def test_fill_creation(self, sample_fill):
        """Test fill creation and properties"""
        assert sample_fill.order_id == "12345"
        assert sample_fill.trade_id == "trade_67890"
        assert sample_fill.symbol == "BTCUSDT"
        assert sample_fill.side == OrderSide.BUY
        assert sample_fill.size == 0.1
        assert sample_fill.price == 50000.0
        assert sample_fill.fee == 0.001
        assert sample_fill.fee_currency == "USDT"
        assert isinstance(sample_fill.timestamp, datetime)
    
    def test_fill_to_dict(self, sample_fill):
        """Test fill conversion to dictionary"""
        fill_dict = sample_fill.to_dict()
        
        assert fill_dict['order_id'] == "12345"
        assert fill_dict['trade_id'] == "trade_67890"
        assert fill_dict['side'] == "buy"  # Enum value
        assert fill_dict['size'] == 0.1
        assert fill_dict['price'] == 50000.0
        assert fill_dict['fee'] == 0.001
    
    def test_fill_different_sides(self):
        """Test fills with different order sides"""
        buy_fill = Fill(
            order_id="buy_order",
            trade_id="buy_trade",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            size=0.1,
            price=50000.0,
            fee=0.001,
            fee_currency="USDT",
            timestamp=datetime.now()
        )
        
        sell_fill = Fill(
            order_id="sell_order",
            trade_id="sell_trade",
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            size=0.2,
            price=51000.0,
            fee=0.002,
            fee_currency="USDT",
            timestamp=datetime.now()
        )
        
        assert buy_fill.side == OrderSide.BUY
        assert sell_fill.side == OrderSide.SELL
        assert buy_fill.to_dict()['side'] == "buy"
        assert sell_fill.to_dict()['side'] == "sell"


class TestCircuitBreaker:
    """Test CircuitBreaker class"""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization"""
        cb = CircuitBreaker(failure_threshold=5, pause_duration=60.0)
        
        assert cb.failure_threshold == 5
        assert cb.pause_duration == 60.0
        assert cb.consecutive_failures == 0
        assert cb.last_failure_time is None
        assert cb.is_open is False
    
    def test_circuit_breaker_can_execute_initially(self):
        """Test circuit breaker allows execution initially"""
        cb = CircuitBreaker()
        assert cb.can_execute() is True
    
    def test_circuit_breaker_record_success(self):
        """Test recording successful operations"""
        cb = CircuitBreaker()
        
        # Simulate some failures first
        cb.consecutive_failures = 2
        cb.record_success()
        
        assert cb.consecutive_failures == 0
        assert cb.is_open is False
    
    def test_circuit_breaker_record_failure(self):
        """Test recording failed operations"""
        cb = CircuitBreaker(failure_threshold=3)
        
        # Record failures up to threshold
        cb.record_failure()
        assert cb.consecutive_failures == 1
        assert cb.is_open is False
        assert cb.can_execute() is True
        
        cb.record_failure()
        assert cb.consecutive_failures == 2
        assert cb.is_open is False
        
        cb.record_failure()
        assert cb.consecutive_failures == 3
        assert cb.is_open is True  # Should open after threshold
        assert cb.can_execute() is False
    
    def test_circuit_breaker_pause_duration(self):
        """Test circuit breaker pause duration"""
        cb = CircuitBreaker(failure_threshold=1, pause_duration=1.0)  # 1 second pause
        
        # Trigger circuit breaker
        cb.record_failure()
        assert cb.is_open is True
        assert cb.can_execute() is False
        
        # Wait for pause to expire
        time.sleep(1.1)
        assert cb.can_execute() is True  # Should reset automatically
    
    def test_circuit_breaker_manual_reset(self):
        """Test manual circuit breaker reset"""
        cb = CircuitBreaker(failure_threshold=1)
        
        # Trigger circuit breaker
        cb.record_failure()
        assert cb.is_open is True
        
        # Manual reset
        cb.reset()
        assert cb.consecutive_failures == 0
        assert cb.is_open is False
        assert cb.last_failure_time is None
        assert cb.can_execute() is True
    
    def test_circuit_breaker_status(self):
        """Test circuit breaker status reporting"""
        cb = CircuitBreaker(failure_threshold=3, pause_duration=300.0)
        
        status = cb.get_status()
        assert status['is_open'] is False
        assert status['consecutive_failures'] == 0
        assert status['failure_threshold'] == 3
        assert status['can_execute'] is True
        
        # Trigger circuit breaker
        for _ in range(3):
            cb.record_failure()
        
        status = cb.get_status()
        assert status['is_open'] is True
        assert status['consecutive_failures'] == 3
        assert status['can_execute'] is False
        assert status['remaining_pause_seconds'] > 0


class TestMockClasses:
    """Test mock HTTP and WebSocket classes"""
    
    @pytest.mark.asyncio
    async def test_mock_http_client(self):
        """Test MockHTTPClient functionality"""
        async with MockHTTPClient() as client:
            assert client.session_active is True
            
            # Test POST request
            response = await client.post("https://test.com", json={"test": "data"})
            assert isinstance(response, MockResponse)
            
            # Test GET request
            response = await client.get("https://test.com")
            assert isinstance(response, MockResponse)
        
        assert client.session_active is False
    
    @pytest.mark.asyncio
    async def test_mock_response(self):
        """Test MockResponse functionality"""
        data = {"code": "00000", "msg": "success", "data": {"orderId": "test123"}}
        response = MockResponse(data, status=200)
        
        assert response.status == 200
        
        json_data = await response.json()
        assert json_data == data
        
        text_data = await response.text()
        assert json.loads(text_data) == data
    
    @pytest.mark.asyncio
    async def test_mock_websocket(self):
        """Test MockWebSocket functionality"""
        async with MockWebSocket("wss://test.com") as ws:
            assert ws.connected is True
            
            # Test send
            await ws.send('{"test": "message"}')
            
            # Test receive (should get mock messages)
            message = await asyncio.wait_for(ws.recv(), timeout=2.0)
            assert isinstance(message, str)
            data = json.loads(message)
            assert "event" in data
        
        assert ws.connected is False


class TestBitgetExecutionEngine:
    """Test BitgetExecutionEngine class"""
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing"""
        return BitgetConfig(
            api_key="test_key",
            secret_key="test_secret",
            passphrase="test_pass",
            sandbox=True
        )
    
    @pytest.fixture
    def execution_engine(self, sample_config):
        """Sample execution engine for testing"""
        return BitgetExecutionEngine(sample_config)
    
    def test_engine_initialization(self, execution_engine, sample_config):
        """Test execution engine initialization"""
        assert execution_engine.config == sample_config
        assert isinstance(execution_engine.circuit_breaker, CircuitBreaker)
        assert execution_engine.running is False
        assert len(execution_engine.orders) == 0
        assert len(execution_engine.fills) == 0
        assert execution_engine.ws_connection is None
        assert execution_engine.ws_task is None
        assert execution_engine.metrics['orders_placed'] == 0
    
    def test_engine_signature_generation(self, execution_engine):
        """Test API signature generation"""
        timestamp = "1234567890000"
        method = "POST"
        request_path = "/api/spot/v1/trade/orders"
        body = '{"symbol":"BTCUSDT"}'
        
        signature = execution_engine._generate_signature(timestamp, method, request_path, body)
        
        assert isinstance(signature, str)
        assert len(signature) == 64  # SHA256 hex digest length
    
    def test_engine_headers_generation(self, execution_engine):
        """Test API headers generation"""
        method = "POST"
        request_path = "/api/spot/v1/trade/orders"
        body = '{"symbol":"BTCUSDT"}'
        
        headers = execution_engine._get_headers(method, request_path, body)
        
        assert 'ACCESS-KEY' in headers
        assert 'ACCESS-SIGN' in headers
        assert 'ACCESS-TIMESTAMP' in headers
        assert 'ACCESS-PASSPHRASE' in headers
        assert 'Content-Type' in headers
        
        assert headers['ACCESS-KEY'] == execution_engine.config.api_key
        assert headers['ACCESS-PASSPHRASE'] == execution_engine.config.passphrase
        assert headers['Content-Type'] == 'application/json'
    
    @pytest.mark.asyncio
    async def test_engine_start_stop(self, execution_engine):
        """Test engine start and stop operations"""
        # Mock the WebSocket connection loop to avoid actual connections
        with patch.object(execution_engine, '_ws_connection_loop', new_callable=AsyncMock):
            await execution_engine.start()
            
            assert execution_engine.running is True
            assert execution_engine.metrics['start_time'] is not None
            assert execution_engine.ws_task is not None
            
            await execution_engine.stop()
            
            assert execution_engine.running is False
    
    @pytest.mark.asyncio
    async def test_engine_place_order_circuit_breaker_open(self, execution_engine):
        """Test order placement when circuit breaker is open"""
        # Open circuit breaker
        execution_engine.circuit_breaker.is_open = True
        
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=0.1
        )
        
        result = await execution_engine.place_order(order)
        assert result is False
    
    @pytest.mark.asyncio
    async def test_engine_place_order_success(self, execution_engine):
        """Test successful order placement"""
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=0.1
        )
        
        # Mock successful HTTP response
        mock_response_data = {
            'code': '00000',
            'msg': 'success',
            'data': {
                'orderId': 'test_order_123',
                'clientOid': 'test_client_123'
            }
        }
        
        with patch('project_chimera.execution.bitget.AIOHTTP_AVAILABLE', False):
            with patch('project_chimera.execution.bitget.MockHTTPClient') as mock_client_class:
                mock_session = AsyncMock()
                mock_response = AsyncMock()
                mock_response.json.return_value = mock_response_data
                mock_session.post.return_value = mock_response
                mock_session.__aenter__.return_value = mock_session
                mock_session.__aexit__.return_value = None
                mock_client_class.return_value = mock_session
                
                result = await execution_engine.place_order(order)
                
                assert result is True
                assert order.order_id == 'test_order_123'
                assert order.status == OrderStatus.OPEN
                assert order.created_time is not None
                assert execution_engine.metrics['orders_placed'] == 1
                assert 'test_order_123' in execution_engine.orders
    
    @pytest.mark.asyncio 
    async def test_engine_place_order_failure(self, execution_engine):
        """Test failed order placement"""
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=0.1
        )
        
        # Mock failed HTTP response
        mock_response_data = {
            'code': '40001',
            'msg': 'Invalid symbol',
            'data': None
        }
        
        with patch('project_chimera.execution.bitget.AIOHTTP_AVAILABLE', False):
            with patch('project_chimera.execution.bitget.MockHTTPClient') as mock_client_class:
                mock_session = AsyncMock()
                mock_response = AsyncMock()
                mock_response.json.return_value = mock_response_data
                mock_session.post.return_value = mock_response
                mock_session.__aenter__.return_value = mock_session
                mock_session.__aexit__.return_value = None
                mock_client_class.return_value = mock_session
                
                result = await execution_engine.place_order(order)
                
                assert result is False
                assert execution_engine.metrics['orders_failed'] == 1
    
    @pytest.mark.asyncio
    async def test_engine_order_callbacks(self, execution_engine):
        """Test order callback execution"""
        callback_called = False
        callback_order = None
        
        async def test_callback(order):
            nonlocal callback_called, callback_order
            callback_called = True
            callback_order = order
        
        execution_engine.order_callbacks.append(test_callback)
        
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            size=0.1
        )
        
        # Mock successful response to trigger callback
        mock_response_data = {
            'code': '00000',
            'msg': 'success',
            'data': {
                'orderId': 'test_order_callback',
                'clientOid': 'test_client_callback'
            }
        }
        
        with patch('project_chimera.execution.bitget.AIOHTTP_AVAILABLE', False):
            with patch('project_chimera.execution.bitget.MockHTTPClient') as mock_client_class:
                mock_session = AsyncMock()
                mock_response = AsyncMock()
                mock_response.json.return_value = mock_response_data
                mock_session.post.return_value = mock_response
                mock_session.__aenter__.return_value = mock_session
                mock_session.__aexit__.return_value = None
                mock_client_class.return_value = mock_session
                
                await execution_engine.place_order(order)
                
                assert callback_called is True
                assert callback_order == order
    
    def test_engine_metrics_tracking(self, execution_engine):
        """Test metrics tracking functionality"""
        initial_metrics = execution_engine.metrics.copy()
        
        # Test metric updates
        execution_engine.metrics['orders_placed'] += 1
        execution_engine.metrics['orders_filled'] += 1
        execution_engine.metrics['api_errors'] += 1
        
        assert execution_engine.metrics['orders_placed'] == initial_metrics['orders_placed'] + 1
        assert execution_engine.metrics['orders_filled'] == initial_metrics['orders_filled'] + 1
        assert execution_engine.metrics['api_errors'] == initial_metrics['api_errors'] + 1
    
    def test_engine_order_management(self, execution_engine):
        """Test order management functionality"""
        order = Order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            size=0.1,
            price=50000.0
        )
        order.order_id = "test_order_123"
        
        # Add order to engine
        execution_engine.orders[order.order_id] = order
        
        assert len(execution_engine.orders) == 1
        assert execution_engine.orders["test_order_123"] == order
        
        # Remove order
        del execution_engine.orders["test_order_123"]
        assert len(execution_engine.orders) == 0
    
    def test_engine_fill_management(self, execution_engine):
        """Test fill management functionality"""
        fill = Fill(
            order_id="test_order_123",
            trade_id="test_trade_456",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            size=0.1,
            price=50000.0,
            fee=0.001,
            fee_currency="USDT",
            timestamp=datetime.now()
        )
        
        # Add fill to engine
        execution_engine.fills.append(fill)
        
        assert len(execution_engine.fills) == 1
        assert execution_engine.fills[0] == fill