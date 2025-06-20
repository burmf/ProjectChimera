"""
Tests for datafeed factory functions - targeting coverage improvement
"""

import pytest
from unittest.mock import MagicMock, patch

from project_chimera.datafeed.factory import (
    create_datafeed, ExchangeType, DataFeedFactory
)


class TestExchangeType:
    """Test ExchangeType enum"""
    
    def test_exchange_type_values(self):
        """Test exchange type enum values"""
        assert ExchangeType.MOCK.value == "mock"
        assert ExchangeType.BINANCE.value == "binance" 
        assert ExchangeType.BITGET.value == "bitget"
        assert ExchangeType.BYBIT.value == "bybit"


class TestDataFeedFactory:
    """Test DataFeedFactory functionality"""
    
    def test_factory_initialization(self):
        """Test factory initialization"""
        factory = DataFeedFactory()
        assert factory is not None
    
    def test_create_mock_adapter(self):
        """Test creating mock adapter"""
        factory = DataFeedFactory()
        
        mock_adapter = factory.create_adapter(ExchangeType.MOCK, "test_mock", {})
        
        assert mock_adapter is not None
        assert mock_adapter.name == "test_mock"
    
    def test_create_binance_adapter(self):
        """Test creating Binance adapter"""
        factory = DataFeedFactory()
        config = {"api_key": "test", "secret": "test"}
        
        binance_adapter = factory.create_adapter(ExchangeType.BINANCE, "test_binance", config)
        
        assert binance_adapter is not None
        assert binance_adapter.name == "test_binance"
    
    def test_create_bitget_adapter(self):
        """Test creating Bitget adapter"""
        factory = DataFeedFactory()
        config = {"api_key": "test", "secret": "test", "passphrase": "test"}
        
        bitget_adapter = factory.create_adapter(ExchangeType.BITGET, "test_bitget", config)
        
        assert bitget_adapter is not None
        assert bitget_adapter.name == "test_bitget"
    
    def test_create_bybit_adapter(self):
        """Test creating ByBit adapter"""
        factory = DataFeedFactory()
        config = {"api_key": "test", "secret": "test"}
        
        bybit_adapter = factory.create_adapter(ExchangeType.BYBIT, "test_bybit", config)
        
        assert bybit_adapter is not None
        assert bybit_adapter.name == "test_bybit"
    
    def test_invalid_exchange_type(self):
        """Test handling invalid exchange type"""
        factory = DataFeedFactory()
        
        with pytest.raises(ValueError, match="Unsupported exchange type"):
            factory.create_adapter("invalid_exchange", "test", {})


class TestCreateDatafeedFunction:
    """Test create_datafeed convenience function"""
    
    def test_create_datafeed_mock(self):
        """Test create_datafeed with mock exchange"""
        adapter = create_datafeed(ExchangeType.MOCK, "test_mock", {})
        
        assert adapter is not None
        assert adapter.name == "test_mock"
    
    def test_create_datafeed_string_type(self):
        """Test create_datafeed with string exchange type"""
        adapter = create_datafeed("mock", "test_mock_string", {})
        
        assert adapter is not None
        assert adapter.name == "test_mock_string"
    
    def test_create_datafeed_binance(self):
        """Test create_datafeed with Binance"""
        config = {"api_key": "test", "secret": "test"}
        adapter = create_datafeed(ExchangeType.BINANCE, "test_binance", config)
        
        assert adapter is not None
        assert adapter.name == "test_binance"
    
    def test_create_datafeed_bitget(self):
        """Test create_datafeed with Bitget"""
        config = {"api_key": "test", "secret": "test", "passphrase": "test"}
        adapter = create_datafeed(ExchangeType.BITGET, "test_bitget", config)
        
        assert adapter is not None
        assert adapter.name == "test_bitget"
    
    def test_create_datafeed_bybit(self):
        """Test create_datafeed with ByBit"""
        config = {"api_key": "test", "secret": "test"}
        adapter = create_datafeed(ExchangeType.BYBIT, "test_bybit", config)
        
        assert adapter is not None
        assert adapter.name == "test_bybit"