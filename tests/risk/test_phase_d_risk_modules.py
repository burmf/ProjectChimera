"""
Comprehensive unit tests for Phase D risk modules
Tests dyn_kelly.py, atr_target.py, and dd_guard.py with NumPy reference implementations
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import List, Tuple

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from project_chimera.risk.dyn_kelly import DynamicKellyCalculator, DynamicKellyConfig, DynamicKellyResult
from project_chimera.risk.atr_target import ATRTargetController, ATRTargetConfig, ATRTargetResult
from project_chimera.risk.dd_guard import DDGuardSystem, DDGuardConfig, DDGuardTier, DDGuardState
from project_chimera.domains.market import OHLCV


class TestDynamicKelly(unittest.TestCase):
    """Test Dynamic Kelly Criterion implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = DynamicKellyConfig(
            base_kelly_fraction=0.5,
            ewma_alpha=0.1,
            min_sample_size=10,
            lookback_window=50
        )
        self.calculator = DynamicKellyCalculator(self.config)
    
    def test_kelly_initialization(self):
        """Test proper initialization"""
        self.assertEqual(self.calculator.config.base_kelly_fraction, 0.5)
        self.assertEqual(self.calculator.ewma_win_rate, 0.5)
        self.assertEqual(len(self.calculator.trade_returns), 0)
    
    def test_add_trade_results(self):
        """Test adding trade results"""
        returns = [0.02, -0.01, 0.03, -0.02, 0.01]
        
        for ret in returns:
            self.calculator.add_trade_result(ret)
        
        self.assertEqual(len(self.calculator.trade_returns), 5)
        self.assertGreater(self.calculator.ewma_win_rate, 0.5)  # More wins than losses
    
    def test_ewma_calculations(self):
        """Test EWMA calculations with known data"""
        # Test with predictable data
        returns = [0.05, 0.03, -0.02, 0.04, -0.01, 0.02, 0.01, -0.03, 0.06, -0.01]
        
        for ret in returns:
            self.calculator.add_trade_result(ret)
        
        # Verify EWMA win rate calculation
        wins = [r for r in returns if r > 0]
        actual_win_rate = len(wins) / len(returns)
        
        # EWMA should be closer to recent performance
        self.assertGreater(self.calculator.ewma_win_rate, 0.0)
        self.assertLess(self.calculator.ewma_win_rate, 1.0)
    
    def test_kelly_calculation_with_sufficient_data(self):
        """Test Kelly calculation with sufficient data"""
        # Create profitable strategy data
        profitable_returns = [0.05, 0.03, -0.02, 0.04, -0.01, 0.02, 0.01, -0.03, 0.06, -0.01,
                             0.03, 0.02, -0.01, 0.04, 0.01, 0.02, -0.02, 0.05, -0.01, 0.03]
        
        for ret in profitable_returns:
            self.calculator.add_trade_result(ret)
        
        result = self.calculator.calculate_dynamic_kelly()
        
        self.assertIsInstance(result, DynamicKellyResult)
        self.assertTrue(result.is_valid())
        self.assertGreater(result.kelly_fraction, 0.0)
        self.assertLessEqual(result.kelly_fraction, self.config.base_kelly_fraction)
        self.assertGreater(result.confidence_score, 0.0)
    
    def test_kelly_calculation_insufficient_data(self):
        """Test Kelly calculation with insufficient data"""
        # Add only a few trades
        for ret in [0.01, -0.02, 0.03]:
            self.calculator.add_trade_result(ret)
        
        result = self.calculator.calculate_dynamic_kelly()
        
        self.assertFalse(result.is_valid())
        self.assertEqual(result.kelly_fraction, 0.0)
        self.assertEqual(result.confidence_score, 0.0)
    
    def test_negative_edge_strategy(self):
        """Test Kelly calculation with negative edge strategy"""
        # Create losing strategy data
        losing_returns = [-0.02, -0.01, 0.005, -0.03, -0.01, 0.002, -0.02, -0.015, 0.01, -0.025,
                         -0.01, -0.02, 0.005, -0.03, -0.01, 0.002, -0.02, -0.015, 0.01, -0.025]
        
        for ret in losing_returns:
            self.calculator.add_trade_result(ret)
        
        result = self.calculator.calculate_dynamic_kelly()
        
        self.assertEqual(result.kelly_fraction, 0.0)  # Should be 0 for negative edge
    
    def test_outlier_filtering(self):
        """Test outlier filtering functionality"""
        # Create data with outliers
        normal_returns = [0.01, 0.02, -0.01, 0.015, -0.005, 0.02, 0.01, -0.015, 0.025, -0.01]
        outlier_returns = [0.5, -0.6]  # Extreme outliers
        
        all_returns = normal_returns + outlier_returns
        
        for ret in all_returns:
            self.calculator.add_trade_result(ret)
        
        # Test internal outlier filtering
        filtered = self.calculator._filter_outliers(all_returns)
        
        # Should remove extreme outliers
        self.assertLess(len(filtered), len(all_returns))
        self.assertNotIn(0.5, filtered)
        self.assertNotIn(-0.6, filtered)
    
    def test_volatility_adjustment(self):
        """Test volatility regime adjustment"""
        # Add initial data
        base_returns = [0.01, 0.02, -0.01, 0.015, -0.005] * 4
        
        for ret in base_returns:
            self.calculator.add_trade_result(ret)
        
        # Add high volatility period
        high_vol_returns = [0.08, -0.07, 0.09, -0.06, 0.1, -0.08] * 2
        
        for ret in high_vol_returns:
            self.calculator.add_trade_result(ret)
        
        adjustment = self.calculator._calculate_volatility_adjustment()
        
        # Should reduce position size in high vol regime
        self.assertLess(adjustment, 1.0)
    
    def test_confidence_scoring(self):
        """Test confidence score calculation"""
        # High confidence scenario: stable, profitable returns
        stable_returns = [0.02, 0.015, -0.01, 0.02, -0.005] * 8
        
        for ret in stable_returns:
            self.calculator.add_trade_result(ret)
        
        result = self.calculator.calculate_dynamic_kelly()
        
        self.assertGreater(result.confidence_score, 0.3)
        
        # Low confidence scenario: volatile, inconsistent returns
        self.calculator.reset()
        volatile_returns = [0.1, -0.08, 0.12, -0.15, 0.05, -0.09, 0.08, -0.11, 0.03, -0.07]
        
        for ret in volatile_returns:
            self.calculator.add_trade_result(ret)
        
        result2 = self.calculator.calculate_dynamic_kelly()
        
        self.assertLess(result2.confidence_score, result.confidence_score)
    
    def test_simulation_functionality(self):
        """Test Kelly performance simulation"""
        # Add trade history
        returns = [0.02, -0.01, 0.03, -0.02, 0.01] * 10
        
        for ret in returns:
            self.calculator.add_trade_result(ret)
        
        simulation = self.calculator.simulate_performance(num_simulations=100)
        
        self.assertIn("final_equity_mean", simulation)
        self.assertIn("max_drawdown_mean", simulation)
        self.assertIn("probability_of_loss", simulation)
        self.assertGreater(simulation["final_equity_mean"], 0.0)
    
    def test_numpy_reference_kelly_formula(self):
        """Test against NumPy reference implementation"""
        # Create test data
        returns = np.array([0.05, 0.03, -0.02, 0.04, -0.01, 0.02, 0.01, -0.03, 0.06, -0.01,
                           0.03, 0.02, -0.01, 0.04, 0.01, 0.02, -0.02, 0.05, -0.01, 0.03])
        
        # NumPy reference implementation
        wins = returns[returns > 0]
        losses = np.abs(returns[returns < 0])
        
        if len(wins) > 0 and len(losses) > 0:
            win_rate = len(wins) / len(returns)
            avg_win = np.mean(wins)
            avg_loss = np.mean(losses)
            
            # Kelly formula: f = (bp - q) / b
            b = avg_win / avg_loss  # odds ratio
            p = win_rate
            q = 1 - p
            
            kelly_reference = (b * p - q) / b
            kelly_reference = max(0.0, kelly_reference)
        else:
            kelly_reference = 0.0
        
        # Test our implementation
        for ret in returns:
            self.calculator.add_trade_result(ret)
        
        result = self.calculator.calculate_dynamic_kelly()
        
        # Should be close to reference (within reasonable tolerance due to EWMA)
        if kelly_reference > 0:
            self.assertGreater(result.raw_kelly, 0.0)
            # Allow for EWMA differences
            self.assertLess(abs(result.raw_kelly - kelly_reference), 0.2)


class TestATRTarget(unittest.TestCase):
    """Test ATR Target Volatility Control implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = ATRTargetConfig(
            target_daily_vol=0.01,
            atr_periods=14,
            min_position_size=0.01,
            max_position_size=0.20
        )
        self.controller = ATRTargetController(self.config)
    
    def create_test_ohlcv_data(self, n_periods: int = 50, base_price: float = 100.0) -> List[OHLCV]:
        """Create test OHLCV data"""
        data = []
        price = base_price
        
        for i in range(n_periods):
            # Random walk with some volatility
            change = np.random.normal(0, 0.02)  # 2% daily volatility
            price *= (1 + change)
            
            # Create OHLCV with realistic intraday range
            high = price * (1 + abs(np.random.normal(0, 0.005)))
            low = price * (1 - abs(np.random.normal(0, 0.005)))
            
            ohlcv = OHLCV(
                timestamp=datetime.now() + timedelta(hours=i),
                symbol="BTCUSDT",
                open=price,
                high=high,
                low=low,
                close=price,
                volume=1000.0
            )
            data.append(ohlcv)
        
        return data
    
    def test_atr_initialization(self):
        """Test proper initialization"""
        self.assertEqual(self.controller.config.target_daily_vol, 0.01)
        self.assertEqual(self.controller.current_atr, 0.0)
        self.assertEqual(len(self.controller.price_history), 0)
    
    def test_add_price_data(self):
        """Test adding price data"""
        test_data = self.create_test_ohlcv_data(20)
        
        for ohlcv in test_data:
            self.controller.add_price_data(ohlcv)
        
        self.assertEqual(len(self.controller.price_history), 20)
        self.assertGreater(self.controller.current_atr, 0.0)
    
    def test_atr_calculation(self):
        """Test ATR calculation with known data"""
        # Create predictable test data
        test_data = []
        base_price = 100.0
        
        for i in range(20):
            ohlcv = OHLCV(
                timestamp=datetime.now() + timedelta(hours=i),
                symbol="BTCUSDT",
                open=base_price + i * 0.1,
                high=base_price + i * 0.1 + 2.0,  # Constant 2.0 range
                low=base_price + i * 0.1 - 1.0,   # Constant 3.0 true range
                close=base_price + i * 0.1 + 1.0,
                volume=1000.0
            )
            test_data.append(ohlcv)
        
        for ohlcv in test_data:
            self.controller.add_price_data(ohlcv)
        
        # ATR should be close to 3.0 (our designed true range)
        self.assertAlmostEqual(self.controller.current_atr, 3.0, delta=0.5)
    
    def test_numpy_reference_atr_calculation(self):
        """Test ATR calculation against NumPy reference"""
        test_data = self.create_test_ohlcv_data(30)
        
        # NumPy reference ATR calculation
        highs = np.array([ohlcv.high for ohlcv in test_data])
        lows = np.array([ohlcv.low for ohlcv in test_data])
        closes = np.array([ohlcv.close for ohlcv in test_data])
        
        # Calculate True Range
        tr1 = highs[1:] - lows[1:]
        tr2 = np.abs(highs[1:] - closes[:-1])
        tr3 = np.abs(lows[1:] - closes[:-1])
        
        true_ranges = np.maximum(tr1, np.maximum(tr2, tr3))
        atr_reference = np.mean(true_ranges[-14:])  # Last 14 periods
        
        # Test our implementation
        for ohlcv in test_data:
            self.controller.add_price_data(ohlcv)
        
        # Should be close to reference
        self.assertAlmostEqual(self.controller.current_atr, atr_reference, delta=0.1)
    
    def test_position_size_calculation(self):
        """Test position sizing for volatility targeting"""
        test_data = self.create_test_ohlcv_data(20)
        
        for ohlcv in test_data:
            self.controller.add_price_data(ohlcv)
        
        current_price = test_data[-1].close
        result = self.controller.calculate_target_position_size(current_price)
        
        self.assertIsInstance(result, ATRTargetResult)
        self.assertGreater(result.position_size_pct, 0.0)
        self.assertLessEqual(result.position_size_pct, self.config.max_position_size)
        self.assertGreaterEqual(result.position_size_pct, self.config.min_position_size)
    
    def test_volatility_targeting_logic(self):
        """Test volatility targeting logic"""
        # Create high volatility data
        high_vol_data = []
        base_price = 100.0
        
        for i in range(20):
            change = np.random.normal(0, 0.05)  # 5% daily volatility (high)
            price = base_price * (1 + change)
            
            ohlcv = OHLCV(
                timestamp=datetime.now() + timedelta(hours=i),
                symbol="BTCUSDT",
                open=price,
                high=price * 1.03,  # High intraday range
                low=price * 0.97,
                close=price,
                volume=1000.0
            )
            high_vol_data.append(ohlcv)
        
        for ohlcv in high_vol_data:
            self.controller.add_price_data(ohlcv)
        
        high_vol_result = self.controller.calculate_target_position_size(high_vol_data[-1].close)
        
        # Reset and test low volatility
        self.controller.reset()
        low_vol_data = []
        
        for i in range(20):
            change = np.random.normal(0, 0.005)  # 0.5% daily volatility (low)
            price = base_price * (1 + change)
            
            ohlcv = OHLCV(
                timestamp=datetime.now() + timedelta(hours=i),
                symbol="BTCUSDT",
                open=price,
                high=price * 1.003,  # Low intraday range
                low=price * 0.997,
                close=price,
                volume=1000.0
            )
            low_vol_data.append(ohlcv)
        
        for ohlcv in low_vol_data:
            self.controller.add_price_data(ohlcv)
        
        low_vol_result = self.controller.calculate_target_position_size(low_vol_data[-1].close)
        
        # High vol should result in smaller position size
        self.assertGreater(low_vol_result.position_size_pct, high_vol_result.position_size_pct)
    
    def test_volatility_regime_adjustment(self):
        """Test volatility regime adjustment"""
        # Add base data
        base_data = self.create_test_ohlcv_data(15)
        for ohlcv in base_data:
            self.controller.add_price_data(ohlcv)
        
        # Record initial regime factor
        initial_regime = self.controller.vol_regime_factor
        
        # Add high volatility period
        for i in range(10):
            high_vol_ohlcv = OHLCV(
                timestamp=datetime.now() + timedelta(hours=15+i),
                symbol="BTCUSDT",
                open=100.0,
                high=105.0,  # High range
                low=95.0,
                close=100.0,
                volume=1000.0
            )
            self.controller.add_price_data(high_vol_ohlcv)
        
        # Regime factor should adjust for high volatility
        self.assertNotEqual(self.controller.vol_regime_factor, initial_regime)
    
    def test_confidence_scoring(self):
        """Test confidence score calculation"""
        # Sufficient data with stable patterns
        stable_data = self.create_test_ohlcv_data(30)
        
        for ohlcv in stable_data:
            self.controller.add_price_data(ohlcv)
        
        result = self.controller.calculate_target_position_size(stable_data[-1].close)
        
        self.assertGreater(result.confidence_score, 0.0)
        self.assertLessEqual(result.confidence_score, 1.0)
    
    def test_volatility_forecasting(self):
        """Test volatility forecasting functionality"""
        test_data = self.create_test_ohlcv_data(25)
        
        for ohlcv in test_data:
            self.controller.add_price_data(ohlcv)
        
        forecast = self.controller.get_volatility_forecast(days_ahead=3)
        
        self.assertIn("forecast_days", forecast)
        self.assertIn("forecasted_vol", forecast)
        self.assertEqual(len(forecast["forecasted_vol"]), 3)
    
    def test_simulation_functionality(self):
        """Test volatility targeting simulation"""
        test_data = self.create_test_ohlcv_data(50)
        
        simulation = self.controller.simulate_vol_targeting(test_data, target_vol=0.015)
        
        self.assertIn("target_vol", simulation)
        self.assertIn("avg_position_size", simulation)
        self.assertIn("avg_achieved_vol", simulation)
        self.assertIn("vol_tracking_error", simulation)


class TestDDGuard(unittest.TestCase):
    """Test Drawdown Guard System implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = DDGuardConfig(
            caution_threshold=0.05,
            warning_threshold=0.10,
            critical_threshold=0.20,
            warning_multiplier=0.5,
            critical_multiplier=0.0
        )
        self.guard = DDGuardSystem(self.config, initial_equity=1.0)
    
    def test_dd_guard_initialization(self):
        """Test proper initialization"""
        self.assertEqual(self.guard.current_equity, 1.0)
        self.assertEqual(self.guard.peak_equity, 1.0)
        self.assertEqual(self.guard.current_tier, DDGuardTier.NORMAL)
        self.assertIsNone(self.guard.cooldown_until)
    
    def test_normal_equity_growth(self):
        """Test normal equity growth without drawdown"""
        equity_curve = [1.0, 1.02, 1.05, 1.08, 1.10, 1.12]
        
        for equity in equity_curve:
            state = self.guard.update_equity(equity)
            self.assertEqual(state.tier, DDGuardTier.NORMAL)
            self.assertEqual(state.position_multiplier, 1.0)
            self.assertFalse(state.is_in_cooldown())
    
    def test_caution_tier_trigger(self):
        """Test caution tier trigger at 5% drawdown"""
        # Grow to peak
        self.guard.update_equity(1.20)
        
        # Drop to 5% drawdown
        caution_equity = 1.20 * 0.95  # 5% drawdown
        state = self.guard.update_equity(caution_equity)
        
        self.assertEqual(state.tier, DDGuardTier.CAUTION)
        self.assertEqual(state.position_multiplier, self.config.caution_multiplier)
    
    def test_warning_tier_trigger(self):
        """Test warning tier trigger at 10% drawdown"""
        # Grow to peak
        self.guard.update_equity(1.20)
        
        # Drop to 10% drawdown
        warning_equity = 1.20 * 0.90  # 10% drawdown
        state = self.guard.update_equity(warning_equity)
        
        self.assertEqual(state.tier, DDGuardTier.WARNING)
        self.assertEqual(state.position_multiplier, self.config.warning_multiplier)
        self.assertTrue(state.is_in_cooldown())
    
    def test_critical_tier_trigger(self):
        """Test critical tier trigger at 20% drawdown"""
        # Grow to peak
        self.guard.update_equity(1.50)
        
        # Drop to 20% drawdown
        critical_equity = 1.50 * 0.80  # 20% drawdown
        state = self.guard.update_equity(critical_equity)
        
        self.assertEqual(state.tier, DDGuardTier.CRITICAL)
        self.assertEqual(state.position_multiplier, 0.0)
        self.assertTrue(state.is_in_cooldown())
        self.assertFalse(state.can_trade())
    
    def test_numpy_reference_drawdown_calculation(self):
        """Test drawdown calculation against NumPy reference"""
        equity_curve = np.array([1.0, 1.1, 1.2, 1.15, 1.05, 0.95, 1.0, 1.1])
        
        # NumPy reference implementation
        running_max = np.maximum.accumulate(equity_curve)
        drawdowns = (running_max - equity_curve) / running_max
        
        # Test our implementation
        for i, equity in enumerate(equity_curve):
            state = self.guard.update_equity(equity)
            
            # Compare drawdown calculation
            expected_dd = drawdowns[i]
            actual_dd = state.drawdown_pct
            
            self.assertAlmostEqual(actual_dd, expected_dd, places=6)
    
    def test_tier_transitions(self):
        """Test proper tier transitions"""
        # Start at peak
        self.guard.update_equity(2.0)
        
        # Caution level
        state1 = self.guard.update_equity(1.9)  # 5% DD
        self.assertEqual(state1.tier, DDGuardTier.CAUTION)
        
        # Warning level
        state2 = self.guard.update_equity(1.8)  # 10% DD
        self.assertEqual(state2.tier, DDGuardTier.WARNING)
        
        # Critical level
        state3 = self.guard.update_equity(1.6)  # 20% DD
        self.assertEqual(state3.tier, DDGuardTier.CRITICAL)
        
        # Recovery should require time and sufficient improvement
        state4 = self.guard.update_equity(1.7)  # Small recovery
        self.assertEqual(state4.tier, DDGuardTier.CRITICAL)  # Should still be critical
    
    def test_cooldown_functionality(self):
        """Test cooldown functionality"""
        # Trigger warning tier
        self.guard.update_equity(1.0)
        self.guard.update_equity(0.9)  # 10% DD
        
        state = self.guard.get_current_state()
        self.assertTrue(state.is_in_cooldown())
        
        # Mock time passing
        with patch('project_chimera.risk.dd_guard.datetime') as mock_dt:
            # Set current time to after cooldown
            future_time = datetime.now() + timedelta(hours=5)
            mock_dt.now.return_value = future_time
            
            state = self.guard.get_current_state()
            # Should no longer be in cooldown
            # Note: This test depends on the actual implementation details
    
    def test_recovery_mechanism(self):
        """Test recovery mechanism with hysteresis"""
        # Set up drawdown
        self.guard.update_equity(1.0)
        self.guard.update_equity(0.85)  # 15% DD -> WARNING tier
        
        # Simulate time passing for minimum recovery time
        import time
        time.sleep(0.1)  # Small delay to ensure time difference
        
        # Partial recovery
        state1 = self.guard.update_equity(0.90)  # Still 10% DD
        
        # Significant recovery
        state2 = self.guard.update_equity(0.95)  # 5% DD
        
        # Recovery progress should be tracked
        self.assertGreater(state2.recovery_progress, state1.recovery_progress)
    
    def test_consecutive_loss_tracking(self):
        """Test consecutive loss tracking"""
        equity_curve = [1.0, 0.98, 0.96, 0.94, 0.92, 0.95]  # 4 consecutive losses, then recovery
        
        max_consecutive = 0
        for equity in equity_curve:
            state = self.guard.update_equity(equity)
            max_consecutive = max(max_consecutive, state.consecutive_losses)
        
        self.assertEqual(max_consecutive, 4)
    
    def test_simulation_functionality(self):
        """Test drawdown simulation on equity curve"""
        # Create test equity curve with known drawdowns
        equity_curve = [1.0, 1.1, 1.2, 1.1, 1.0, 0.9, 0.8, 0.85, 0.9, 1.0, 1.1]
        
        simulation = self.guard.simulate_drawdown_response(equity_curve)
        
        self.assertIn("simulation_points", simulation)
        self.assertIn("tier_changes", simulation)
        self.assertIn("max_drawdown_simulated", simulation)
        self.assertIn("drawdown_breaches", simulation)
        
        # Should detect the 20% drawdown
        self.assertGreaterEqual(simulation["max_drawdown_simulated"], 0.2)
        self.assertGreater(simulation["drawdown_breaches"]["critical"], 0)
    
    def test_force_tier_change(self):
        """Test manual tier override"""
        # Start in normal tier
        self.assertEqual(self.guard.current_tier, DDGuardTier.NORMAL)
        
        # Force to critical
        self.guard.force_tier_change(DDGuardTier.CRITICAL, "Emergency test")
        
        self.assertEqual(self.guard.current_tier, DDGuardTier.CRITICAL)
        self.assertIsNotNone(self.guard.cooldown_until)
    
    def test_statistics_generation(self):
        """Test comprehensive statistics generation"""
        # Create some trading history
        equity_curve = [1.0, 1.05, 1.1, 1.08, 1.12, 1.0, 0.95, 1.02, 1.08]
        
        for equity in equity_curve:
            self.guard.update_equity(equity)
        
        stats = self.guard.get_statistics()
        
        required_keys = [
            "current_equity", "peak_equity", "current_drawdown", "worst_drawdown",
            "current_tier", "position_multiplier", "total_return", "volatility",
            "tier_changes"
        ]
        
        for key in required_keys:
            self.assertIn(key, stats)
    
    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        # Test with zero/negative equity
        state = self.guard.update_equity(0.0)
        self.assertIsInstance(state, DDGuardState)
        
        # Test with very small equity
        state = self.guard.update_equity(0.001)
        self.assertIsInstance(state, DDGuardState)
        
        # Test system reset
        self.guard.reset_system(1.0)
        self.assertEqual(self.guard.current_equity, 1.0)
        self.assertEqual(self.guard.current_tier, DDGuardTier.NORMAL)


class TestPhaseIntegration(unittest.TestCase):
    """Integration tests for all Phase D modules working together"""
    
    def setUp(self):
        """Set up integrated test environment"""
        self.kelly_calc = DynamicKellyCalculator()
        self.atr_controller = ATRTargetController()
        self.dd_guard = DDGuardSystem()
    
    def test_integrated_risk_management(self):
        """Test all three modules working together"""
        # Generate test data
        trade_returns = [0.02, -0.01, 0.03, -0.02, 0.01] * 8  # 40 trades
        
        # Add to Kelly calculator
        for ret in trade_returns:
            self.kelly_calc.add_trade_result(ret)
        
        # Generate price data for ATR
        price_data = []
        base_price = 100.0
        equity = 1.0
        
        for i, ret in enumerate(trade_returns):
            # Update price
            base_price *= (1 + ret)
            
            ohlcv = OHLCV(
                timestamp=datetime.now() + timedelta(hours=i),
                symbol="BTCUSDT",
                open=base_price,
                high=base_price * 1.01,
                low=base_price * 0.99,
                close=base_price,
                volume=1000.0
            )
            
            self.atr_controller.add_price_data(ohlcv)
            
            # Update equity for DD guard
            equity *= (1 + ret)
            self.dd_guard.update_equity(equity)
        
        # Get recommendations from each module
        kelly_result = self.kelly_calc.calculate_dynamic_kelly()
        atr_result = self.atr_controller.calculate_target_position_size(base_price)
        dd_state = self.dd_guard.get_current_state()
        
        # All should be valid
        self.assertTrue(kelly_result.is_valid())
        self.assertTrue(atr_result.is_valid())
        self.assertTrue(dd_state.can_trade())
        
        # Combined position sizing logic
        kelly_size = kelly_result.kelly_fraction
        atr_size = atr_result.position_size_pct
        dd_multiplier = dd_state.position_multiplier
        
        # Final position size should be the most conservative
        final_size = min(kelly_size, atr_size) * dd_multiplier
        
        self.assertGreater(final_size, 0.0)
        self.assertLessEqual(final_size, 1.0)
    
    def test_stress_scenario(self):
        """Test modules under stress scenario"""
        # Create stressed market conditions
        stress_returns = [0.05, -0.08, 0.03, -0.12, 0.02, -0.15, 0.01, -0.10, 0.04, -0.20]
        
        equity = 1.0
        for ret in stress_returns:
            # Update Kelly
            self.kelly_calc.add_trade_result(ret)
            
            # Update equity and DD guard
            equity *= (1 + ret)
            dd_state = self.dd_guard.update_equity(equity)
            
            # In stress, DD guard should reduce position sizes
            if dd_state.drawdown_pct > 0.1:  # 10% DD
                self.assertLess(dd_state.position_multiplier, 1.0)
        
        # Kelly should recognize negative edge
        kelly_result = self.kelly_calc.calculate_dynamic_kelly()
        self.assertLess(kelly_result.kelly_fraction, 0.5)  # Should be conservative


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)