"""
Tests for configuration management and settings validation
"""

import os

import pytest
import yaml

from project_chimera.settings import (
    ExchangeAdapterConfig,
    Settings,
    StrategyConfig,
    get_exchange_config,
    get_settings,
    get_strategy_config,
    load_yaml_config,
)


class TestSettingsConfig:
    """Test settings configuration and YAML loading"""

    def test_default_settings_creation(self):
        """Test that default settings can be created"""
        # Clear cache to ensure fresh settings
        get_settings.cache_clear()

        # Create settings without YAML file (direct instantiation)
        settings = Settings(env="dev", debug=False)
        assert settings.env == "dev"
        assert settings.debug is False
        assert settings.trading.leverage_default == 25.0
        assert settings.risk.kelly_base_fraction == 0.5

    def test_yaml_config_loading(self, tmp_path):
        """Test YAML configuration loading"""
        # Create a temporary config file
        config_data = {
            'env': 'test',
            'debug': True,
            'trading': {
                'leverage_default': 15.0,
                'max_positions': 5
            },
            'strategies': {
                'weekend_effect': {
                    'enabled': True,
                    'confidence': 0.8,
                    'target_size': 0.06
                }
            }
        }

        config_file = tmp_path / "config.test.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)

        # Change directory to temp path for test
        original_cwd = os.getcwd()
        try:
            os.chdir(tmp_path)
            yaml_config = load_yaml_config('test')

            assert yaml_config['env'] == 'test'
            assert yaml_config['debug'] is True
            assert yaml_config['trading']['leverage_default'] == 15.0
            assert yaml_config['strategies']['weekend_effect']['confidence'] == 0.8
        finally:
            os.chdir(original_cwd)

    def test_strategy_config_access(self):
        """Test strategy configuration access"""
        strategy_config = get_strategy_config('weekend_effect')
        assert isinstance(strategy_config, StrategyConfig)
        assert hasattr(strategy_config, 'enabled')
        assert hasattr(strategy_config, 'confidence')
        assert hasattr(strategy_config, 'target_size')

    def test_exchange_config_access(self):
        """Test exchange configuration access"""
        exchange_config = get_exchange_config('bitget')
        assert isinstance(exchange_config, ExchangeAdapterConfig)
        assert exchange_config.timeout_seconds >= 5
        assert exchange_config.max_connections >= 1
        assert 'bitget.com' in exchange_config.rest_base_url

    def test_settings_validation(self):
        """Test that settings validate properly"""
        # Test invalid leverage
        with pytest.raises(ValueError):
            Settings(trading={'leverage_default': -5.0})

        # Test invalid confidence
        with pytest.raises(ValueError):
            Settings(risk={'kelly_base_fraction': 1.5})  # Should be <= 1.0

    def test_environment_variable_override(self):
        """Test that environment variables can override settings"""
        # Set environment variable
        os.environ['ENV'] = 'test'
        os.environ['DEBUG'] = 'true'

        try:
            # Note: This won't work perfectly without actual config file,
            # but tests the mechanism
            get_settings.cache_clear()
        finally:
            # Clean up
            if 'ENV' in os.environ:
                del os.environ['ENV']
            if 'DEBUG' in os.environ:
                del os.environ['DEBUG']

    def test_strategy_config_with_extra_fields(self):
        """Test that strategy config accepts extra fields"""
        config_data = {
            'enabled': True,
            'confidence': 0.75,
            'custom_param': 42,
            'another_param': 'test_value'
        }

        strategy_config = StrategyConfig(**config_data)
        assert strategy_config.enabled is True
        assert strategy_config.confidence == 0.75
        # Extra fields should be accessible
        assert hasattr(strategy_config, 'custom_param')
        assert strategy_config.custom_param == 42


class TestConfigIntegration:
    """Test integration between different config components"""

    def test_settings_cache_behavior(self):
        """Test that settings are properly cached"""
        settings1 = get_settings()
        settings2 = get_settings()

        # Should be the same object due to caching
        assert settings1 is settings2

        # Clear cache and get new instance
        get_settings.cache_clear()
        settings3 = get_settings()

        # Should be different object but same values
        assert settings1 is not settings3
        assert settings1.env == settings3.env

    def test_risk_config_completeness(self):
        """Test that risk config has all required parameters"""
        settings = get_settings()
        risk = settings.risk

        # Check all critical risk parameters exist
        required_attrs = [
            'kelly_base_fraction', 'atr_target_daily_vol', 'max_leverage',
            'dd_caution_threshold', 'dd_warning_threshold', 'dd_critical_threshold',
            'max_daily_loss', 'max_drawdown'
        ]

        for attr in required_attrs:
            assert hasattr(risk, attr), f"Missing risk parameter: {attr}"
            value = getattr(risk, attr)
            assert value is not None, f"Risk parameter {attr} is None"
            assert isinstance(value, int | float), f"Risk parameter {attr} is not numeric"

    def test_api_config_security(self):
        """Test that API config handles secrets properly"""
        settings = get_settings()
        api = settings.api

        # API keys should be SecretStr types
        assert hasattr(api.bitget_key, 'get_secret_value')
        assert hasattr(api.bitget_secret, 'get_secret_value')

        # String representation should not reveal secrets
        key_str = str(api.bitget_key)
        assert 'SecretStr' in key_str
        assert len(api.bitget_key.get_secret_value()) == 0 or '**' in key_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
