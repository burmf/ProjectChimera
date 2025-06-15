"""
Professional test suite for ProfessionalRiskManager
Comprehensive testing of risk calculations and edge cases
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from project_chimera.core.risk_manager import (
    ProfessionalRiskManager,
    PortfolioMetrics,
    PositionRisk,
    RiskLimits,
    MarketRegime
)
from project_chimera.config import Settings


@pytest.fixture
def mock_settings():
    """Mock settings for testing"""
    settings = Settings()
    settings.risk.max_portfolio_risk = 0.15
    settings.risk.max_daily_loss = 0.05
    settings.risk.max_drawdown = 0.10
    settings.risk.kelly_fraction = 0.25
    settings.trading.max_positions = 8
    settings.trading.max_leverage = 50
    return settings


@pytest.fixture
def risk_manager(mock_settings):
    """Create risk manager for testing"""
    return ProfessionalRiskManager(mock_settings)


class TestProfessionalRiskManager:
    """Test suite for ProfessionalRiskManager"""
    
    def test_initialization(self, mock_settings):
        """Test risk manager initialization"""
        rm = ProfessionalRiskManager(mock_settings)
        
        assert rm.settings == mock_settings
        assert rm.portfolio_value == 100000.0
        assert rm.peak_value == 100000.0
        assert rm.current_regime == MarketRegime.NORMAL
        assert len(rm.price_history) == 0
        assert len(rm.return_history) == 0
    
    def test_update_price_data(self, risk_manager):
        """Test price data updates and return calculations"""
        symbol = 'BTCUSDT'
        
        # Add first price
        risk_manager.update_price_data(symbol, 50000.0, datetime.now())
        assert len(risk_manager.price_history[symbol]) == 1
        assert len(risk_manager.return_history[symbol]) == 0
        
        # Add second price
        risk_manager.update_price_data(symbol, 51000.0, datetime.now())
        assert len(risk_manager.price_history[symbol]) == 2
        assert len(risk_manager.return_history[symbol]) == 1
        
        # Check return calculation
        expected_return = (51000.0 - 50000.0) / 50000.0
        assert abs(risk_manager.return_history[symbol][0] - expected_return) < 1e-6
    
    def test_volatility_model_update(self, risk_manager):
        """Test volatility model updates"""
        symbol = 'BTCUSDT'
        
        # Generate price series with known volatility
        np.random.seed(42)
        base_price = 50000.0
        prices = [base_price]
        
        for i in range(50):
            return_val = np.random.normal(0, 0.02)  # 2% daily volatility
            new_price = prices[-1] * (1 + return_val)
            prices.append(new_price)
            
            risk_manager.update_price_data(symbol, new_price, datetime.now())
        
        # Check that volatility model was updated
        assert symbol in risk_manager.volatility_model
        assert risk_manager.volatility_model[symbol] > 0
        
        # Volatility should be reasonable (around 2% +/- some tolerance)
        assert 0.01 < risk_manager.volatility_model[symbol] < 0.05
    
    def test_kelly_position_sizing(self, risk_manager):
        """Test Kelly criterion position sizing"""
        symbol = 'BTCUSDT'
        
        # Test with profitable strategy
        kelly_size = risk_manager.calculate_kelly_position_size(
            symbol=symbol,
            expected_return=0.01,
            win_probability=0.6,
            loss_probability=0.4,
            avg_win=0.02,
            avg_loss=0.01
        )
        
        assert kelly_size > 0
        assert kelly_size <= risk_manager.portfolio_value * 0.2  # Max concentration
        
        # Test with unprofitable strategy
        kelly_size_bad = risk_manager.calculate_kelly_position_size(
            symbol=symbol,
            expected_return=-0.01,
            win_probability=0.3,
            loss_probability=0.7,
            avg_win=0.01,
            avg_loss=0.02
        )
        
        assert kelly_size_bad == 0.0
    
    def test_monte_carlo_var(self, risk_manager):
        """Test Monte Carlo VaR calculation"""
        # Set up portfolio with some history
        symbol = 'BTCUSDT'
        
        # Generate return history
        np.random.seed(42)
        for i in range(100):
            return_val = np.random.normal(0, 0.02)
            risk_manager.return_history[symbol] = risk_manager.return_history.get(symbol, [])
            risk_manager.return_history[symbol].append(return_val)
        
        # Update volatility model
        risk_manager.volatility_model[symbol] = 0.02
        
        # Set up positions
        risk_manager.current_positions = {
            symbol: {'value': 50000, 'leverage': 2}
        }
        
        var_metrics = risk_manager._monte_carlo_var(0.95, 1)
        
        assert 'var_95' in var_metrics
        assert 'var_99' in var_metrics
        assert 'expected_shortfall' in var_metrics
        
        assert var_metrics['var_95'] > 0
        assert var_metrics['var_99'] > var_metrics['var_95']  # 99% VaR should be higher
        assert var_metrics['expected_shortfall'] >= var_metrics['var_95']
    
    def test_parametric_var(self, risk_manager):
        """Test parametric VaR calculation"""
        symbol = 'BTCUSDT'
        
        # Set up return history with known distribution
        returns = np.random.normal(0, 0.02, 100)
        risk_manager.return_history[symbol] = returns.tolist()
        
        risk_manager.current_positions = {
            symbol: {'value': 50000, 'leverage': 1}
        }
        
        var_metrics = risk_manager._parametric_var(0.95, 1)
        
        assert var_metrics['var_95'] > 0
        assert var_metrics['var_99'] > var_metrics['var_95']
        
        # For normal distribution, 99% VaR should be roughly 2.33/1.65 times 95% VaR
        ratio = var_metrics['var_99'] / var_metrics['var_95']
        assert 2.0 < ratio < 2.5  # Rough check
    
    def test_historical_var(self, risk_manager):
        """Test historical VaR calculation"""
        # Generate daily returns
        np.random.seed(42)
        returns = np.random.normal(0, 0.02, 100)
        risk_manager.daily_returns = returns.tolist()
        risk_manager.portfolio_value = 100000
        
        var_metrics = risk_manager._historical_var(0.95, 1)
        
        assert var_metrics['var_95'] > 0
        assert var_metrics['var_99'] > var_metrics['var_95']
        
        # Historical VaR should match empirical quantiles
        expected_var_95 = abs(np.percentile(returns, 5)) * 100000
        assert abs(var_metrics['var_95'] - expected_var_95) < 1000  # Within $1000
    
    def test_correlation_matrix_update(self, risk_manager):
        """Test correlation matrix calculation"""
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        
        # Generate correlated returns
        np.random.seed(42)
        n_periods = 100
        
        # Create correlated returns
        base_returns = np.random.normal(0, 0.02, n_periods)
        
        for i, symbol in enumerate(symbols):
            correlation = 0.8 if i < 2 else 0.6  # High correlation between BTC/ETH
            noise = np.random.normal(0, 0.01, n_periods)
            correlated_returns = correlation * base_returns + (1 - correlation) * noise
            
            risk_manager.return_history[symbol] = correlated_returns.tolist()
        
        risk_manager._update_correlation_matrix()
        
        assert risk_manager.correlation_matrix is not None
        assert risk_manager.correlation_matrix.shape == (3, 3)
        
        # Check that diagonal is 1 (self-correlation)
        for symbol in symbols:
            assert abs(risk_manager.correlation_matrix.loc[symbol, symbol] - 1.0) < 0.01
        
        # Check that BTC-ETH correlation is higher than BTC-SOL
        btc_eth_corr = abs(risk_manager.correlation_matrix.loc['BTCUSDT', 'ETHUSDT'])
        btc_sol_corr = abs(risk_manager.correlation_matrix.loc['BTCUSDT', 'SOLUSDT'])
        assert btc_eth_corr > btc_sol_corr
    
    def test_market_regime_detection(self, risk_manager):
        """Test market regime detection"""
        # Test normal regime
        normal_returns = np.random.normal(0, 0.01, 60)
        risk_manager.daily_returns = normal_returns.tolist()
        
        regime = risk_manager.detect_market_regime()
        assert regime in [MarketRegime.NORMAL, MarketRegime.TRENDING]
        
        # Test volatile regime
        volatile_returns = np.random.normal(0, 0.05, 60)  # High volatility
        risk_manager.daily_returns = volatile_returns.tolist()
        
        regime = risk_manager.detect_market_regime()
        assert regime == MarketRegime.VOLATILE
        
        # Test crisis regime
        crisis_returns = [-0.06, -0.07, -0.08] + list(np.random.normal(0, 0.02, 57))
        risk_manager.daily_returns = crisis_returns
        
        regime = risk_manager.detect_market_regime()
        assert regime == MarketRegime.CRISIS
    
    def test_dynamic_risk_limits(self, risk_manager):
        """Test dynamic risk limit calculation"""
        # Test normal regime limits
        risk_manager.current_regime = MarketRegime.NORMAL
        limits = risk_manager.get_dynamic_risk_limits()
        
        assert isinstance(limits, RiskLimits)
        assert limits.max_position_size > 0
        assert limits.max_leverage > 0
        assert limits.position_limit > 0
        
        # Test crisis regime limits (should be more restrictive)
        risk_manager.current_regime = MarketRegime.CRISIS
        crisis_limits = risk_manager.get_dynamic_risk_limits()
        
        assert crisis_limits.max_position_size < limits.max_position_size
        assert crisis_limits.max_leverage < limits.max_leverage
        assert crisis_limits.position_limit <= limits.position_limit
    
    def test_position_validation(self, risk_manager):
        """Test new position validation"""
        # Set up reasonable portfolio state
        risk_manager.current_positions = {
            'BTCUSDT': {'value': 20000, 'leverage': 10}
        }
        risk_manager.portfolio_value = 100000
        
        # Test valid position
        is_valid, warnings = risk_manager.validate_new_position('ETHUSDT', 15000, 5)
        assert is_valid or len(warnings) == 0  # Should pass or have minor warnings
        
        # Test oversized position
        is_valid, warnings = risk_manager.validate_new_position('ETHUSDT', 50000, 20)
        assert not is_valid
        assert len(warnings) > 0
        assert any('size' in warning.lower() or 'leverage' in warning.lower() for warning in warnings)
    
    def test_portfolio_metrics_calculation(self, risk_manager):
        """Test comprehensive portfolio metrics"""
        # Generate realistic return series
        np.random.seed(42)
        
        # Generate returns with some drawdown periods
        returns = []
        for i in range(252):  # One year of data
            if 50 <= i <= 70:  # Drawdown period
                ret = np.random.normal(-0.01, 0.03)
            else:
                ret = np.random.normal(0.001, 0.02)
            returns.append(ret)
        
        risk_manager.daily_returns = returns
        risk_manager.portfolio_value = 110000
        
        # Mock VaR calculation
        with patch.object(risk_manager, 'calculate_portfolio_var') as mock_var:
            mock_var.return_value = {
                'var_95': 5000,
                'var_99': 8000,
                'expected_shortfall': 6000
            }
            
            metrics = risk_manager.calculate_portfolio_metrics()
        
        assert isinstance(metrics, PortfolioMetrics)
        assert metrics.var_95 == 5000
        assert metrics.sharpe_ratio != 0
        assert metrics.max_drawdown < 0  # Should be negative
        assert metrics.volatility > 0
        assert 0 <= metrics.correlation_risk <= 1
    
    def test_portfolio_state_update(self, risk_manager):
        """Test portfolio state updates"""
        initial_value = risk_manager.portfolio_value
        
        positions = {
            'BTCUSDT': {'value': 30000, 'leverage': 5},
            'ETHUSDT': {'value': 20000, 'leverage': 3}
        }
        
        new_value = 105000
        risk_manager.update_portfolio_state(positions, new_value)
        
        assert risk_manager.current_positions == positions
        assert risk_manager.portfolio_value == new_value
        assert risk_manager.peak_value == max(initial_value, new_value)
        assert len(risk_manager.portfolio_history) == 1
    
    def test_correlation_risk_calculation(self, risk_manager):
        """Test correlation risk calculation"""
        # Set up correlation matrix
        symbols = ['BTCUSDT', 'ETHUSDT']
        corr_data = {
            'BTCUSDT': [1.0, 0.8],
            'ETHUSDT': [0.8, 1.0]
        }
        risk_manager.correlation_matrix = pd.DataFrame(corr_data, index=symbols, columns=symbols)
        
        # Set up positions
        risk_manager.current_positions = {
            'BTCUSDT': {'value': 40000},
            'ETHUSDT': {'value': 30000}
        }
        
        correlation_risk = risk_manager._calculate_correlation_risk()
        
        assert 0 <= correlation_risk <= 1
        assert correlation_risk > 0  # Should have some correlation risk
    
    def test_edge_cases(self, risk_manager):
        """Test edge cases and error conditions"""
        # Test empty portfolio
        metrics = risk_manager.calculate_portfolio_metrics()
        assert isinstance(metrics, PortfolioMetrics)
        assert metrics.var_95 == 0
        
        # Test VaR with no data
        var_metrics = risk_manager.calculate_portfolio_var()
        assert var_metrics['var_95'] == 0
        
        # Test Kelly sizing with zero loss
        kelly_size = risk_manager.calculate_kelly_position_size(
            'BTCUSDT', 0.01, 0.6, 0.4, 0.02, 0.0
        )
        assert kelly_size == 0.0
        
        # Test regime detection with insufficient data
        risk_manager.daily_returns = [0.01, -0.01]
        regime = risk_manager.detect_market_regime()
        assert regime == MarketRegime.NORMAL


class TestIntegrationScenarios:
    """Integration test scenarios"""
    
    def test_complete_risk_assessment_flow(self, risk_manager):
        """Test complete risk assessment workflow"""
        # Step 1: Build price history
        symbols = ['BTCUSDT', 'ETHUSDT']
        np.random.seed(42)
        
        for symbol in symbols:
            base_price = 50000 if symbol == 'BTCUSDT' else 3000
            for i in range(100):
                return_val = np.random.normal(0, 0.02)
                new_price = base_price * (1 + return_val)
                risk_manager.update_price_data(symbol, new_price, datetime.now())
                base_price = new_price
        
        # Step 2: Set up portfolio
        positions = {
            'BTCUSDT': {'value': 40000, 'leverage': 5},
            'ETHUSDT': {'value': 30000, 'leverage': 3}
        }
        risk_manager.update_portfolio_state(positions, 100000)
        
        # Step 3: Calculate risk metrics
        var_metrics = risk_manager.calculate_portfolio_var()
        assert var_metrics['var_95'] > 0
        
        # Step 4: Check limits
        is_valid, warnings = risk_manager.validate_new_position('SOLUSDT', 20000, 10)
        # Should work or have reasonable warnings
        
        # Step 5: Generate report
        report = risk_manager.generate_risk_report()
        assert 'PORTFOLIO METRICS' in report
        assert 'RISK MEASURES' in report
        assert 'VaR' in report
    
    def test_stress_testing_scenario(self, risk_manager):
        """Test system under stress conditions"""
        # Simulate market crash scenario
        crash_returns = [-0.10, -0.15, -0.08, -0.05] + list(np.random.normal(0, 0.03, 46))
        risk_manager.daily_returns = crash_returns
        
        # Large leveraged positions
        risk_manager.current_positions = {
            'BTCUSDT': {'value': 50000, 'leverage': 20},
            'ETHUSDT': {'value': 40000, 'leverage': 15}
        }
        risk_manager.portfolio_value = 90000  # Portfolio down 10%
        
        # System should detect crisis regime
        regime = risk_manager.detect_market_regime()
        assert regime == MarketRegime.CRISIS
        
        # Limits should be restrictive
        limits = risk_manager.get_dynamic_risk_limits()
        assert limits.max_leverage < 25  # Should reduce leverage
        
        # New position should be rejected or heavily restricted
        is_valid, warnings = risk_manager.validate_new_position('SOLUSDT', 30000, 25)
        assert not is_valid or len(warnings) > 2


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])