"""
Simple unit tests for Phase D risk modules without external dependencies
Tests core logic of dyn_kelly.py, atr_target.py, and dd_guard.py
"""

import os
import sys
import unittest
from datetime import datetime, timedelta

# Add project to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

try:
    from project_chimera.risk.atr_target import ATRTargetConfig, ATRTargetController
    from project_chimera.risk.dd_guard import DDGuardConfig, DDGuardSystem, DDGuardTier
    from project_chimera.risk.dyn_kelly import (
        DynamicKellyCalculator,
        DynamicKellyConfig,
    )

    # Simple OHLCV class for testing
    class SimpleOHLCV:
        def __init__(self, timestamp, symbol, open_price, high, low, close, volume):
            self.timestamp = timestamp
            self.symbol = symbol
            self.open = open_price
            self.high = high
            self.low = low
            self.close = close
            self.volume = volume

    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    MODULES_AVAILABLE = False


@unittest.skipUnless(MODULES_AVAILABLE, "Phase D modules not available")
class TestDynamicKellyBasic(unittest.TestCase):
    """Basic tests for Dynamic Kelly without NumPy dependencies"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = DynamicKellyConfig(
            base_kelly_fraction=0.5, min_sample_size=5  # Lower for testing
        )
        self.calculator = DynamicKellyCalculator(self.config)

    def test_initialization(self):
        """Test basic initialization"""
        self.assertEqual(self.calculator.config.base_kelly_fraction, 0.5)
        self.assertEqual(len(self.calculator.trade_returns), 0)

    def test_add_trades(self):
        """Test adding trade results"""
        returns = [0.02, -0.01, 0.03, -0.02, 0.01]

        for ret in returns:
            self.calculator.add_trade_result(ret)

        self.assertEqual(len(self.calculator.trade_returns), 5)

    def test_kelly_with_profitable_trades(self):
        """Test Kelly calculation with profitable strategy"""
        # Profitable strategy: 70% win rate, good risk/reward
        profitable_returns = [
            0.05,
            0.03,
            -0.02,
            0.04,
            0.01,
            0.02,
            -0.01,
            0.03,
            0.02,
            0.01,
        ]

        for ret in profitable_returns:
            self.calculator.add_trade_result(ret)

        result = self.calculator.calculate_dynamic_kelly()

        self.assertGreater(result.kelly_fraction, 0.0)
        self.assertLessEqual(result.kelly_fraction, 0.5)  # Should be <= base fraction

    def test_kelly_with_losing_trades(self):
        """Test Kelly calculation with losing strategy"""
        # Losing strategy
        losing_returns = [-0.02, -0.01, 0.005, -0.03, -0.01, 0.002, -0.02, -0.015]

        for ret in losing_returns:
            self.calculator.add_trade_result(ret)

        result = self.calculator.calculate_dynamic_kelly()

        self.assertEqual(result.kelly_fraction, 0.0)  # Should be 0 for negative edge


@unittest.skipUnless(MODULES_AVAILABLE, "Phase D modules not available")
class TestATRTargetBasic(unittest.TestCase):
    """Basic tests for ATR Target without NumPy dependencies"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = ATRTargetConfig(
            target_daily_vol=0.01, atr_periods=5  # Lower for testing
        )
        self.controller = ATRTargetController(self.config)

    def test_initialization(self):
        """Test basic initialization"""
        self.assertEqual(self.controller.config.target_daily_vol, 0.01)
        self.assertEqual(len(self.controller.price_history), 0)

    def test_add_price_data(self):
        """Test adding price data"""
        ohlcv = SimpleOHLCV(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            open_price=100.0,
            high=102.0,
            low=98.0,
            close=101.0,
            volume=1000.0,
        )

        self.controller.add_price_data(ohlcv)
        self.assertEqual(len(self.controller.price_history), 1)

    def test_position_sizing_with_data(self):
        """Test position sizing calculation"""
        # Add enough price data
        base_price = 100.0
        for i in range(10):
            ohlcv = SimpleOHLCV(
                timestamp=datetime.now() + timedelta(hours=i),
                symbol="BTCUSDT",
                open_price=base_price + i * 0.1,
                high=base_price + i * 0.1 + 2.0,
                low=base_price + i * 0.1 - 1.0,
                close=base_price + i * 0.1 + 1.0,
                volume=1000.0,
            )
            self.controller.add_price_data(ohlcv)

        result = self.controller.calculate_target_position_size(base_price + 10)

        self.assertGreater(result.position_size_pct, 0.0)
        self.assertLessEqual(result.position_size_pct, 1.0)


@unittest.skipUnless(MODULES_AVAILABLE, "Phase D modules not available")
class TestDDGuardBasic(unittest.TestCase):
    """Basic tests for DD Guard without NumPy dependencies"""

    def setUp(self):
        """Set up test fixtures"""
        self.config = DDGuardConfig(
            caution_threshold=0.05, warning_threshold=0.10, critical_threshold=0.20
        )
        self.guard = DDGuardSystem(self.config, initial_equity=1.0)

    def test_initialization(self):
        """Test basic initialization"""
        self.assertEqual(self.guard.current_equity, 1.0)
        self.assertEqual(self.guard.current_tier, DDGuardTier.NORMAL)

    def test_normal_operations(self):
        """Test normal operations without drawdown"""
        equity_curve = [1.0, 1.02, 1.05, 1.08, 1.10]

        for equity in equity_curve:
            state = self.guard.update_equity(equity)
            self.assertEqual(state.tier, DDGuardTier.NORMAL)
            self.assertEqual(state.position_multiplier, 1.0)

    def test_drawdown_tiers(self):
        """Test drawdown tier transitions"""
        # Set peak
        self.guard.update_equity(1.20)

        # Test caution tier (5% DD)
        state1 = self.guard.update_equity(1.14)  # 5% DD
        self.assertEqual(state1.tier, DDGuardTier.CAUTION)

        # Test warning tier (10% DD)
        state2 = self.guard.update_equity(1.08)  # 10% DD
        self.assertEqual(state2.tier, DDGuardTier.WARNING)
        self.assertEqual(state2.position_multiplier, 0.5)  # 50% reduction

        # Test critical tier (20% DD)
        state3 = self.guard.update_equity(0.96)  # 20% DD
        self.assertEqual(state3.tier, DDGuardTier.CRITICAL)
        self.assertEqual(state3.position_multiplier, 0.0)  # No trading

    def test_drawdown_calculation(self):
        """Test drawdown percentage calculation"""
        # Set peak at 2.0
        self.guard.update_equity(2.0)

        # Test specific drawdown levels
        test_cases = [
            (1.9, 0.05),  # 5% DD
            (1.8, 0.10),  # 10% DD
            (1.6, 0.20),  # 20% DD
            (1.0, 0.50),  # 50% DD
        ]

        for equity, expected_dd in test_cases:
            state = self.guard.update_equity(equity)
            self.assertAlmostEqual(state.drawdown_pct, expected_dd, places=2)


class TestManualKellyReference(unittest.TestCase):
    """Manual Kelly calculation reference implementation"""

    def test_kelly_formula_reference(self):
        """Test Kelly formula with manual calculation"""
        # Test data: [wins, losses] with known statistics
        returns = [0.05, 0.03, -0.02, 0.04, -0.01, 0.02, 0.01, -0.03, 0.06, -0.01]

        # Manual calculation
        wins = [r for r in returns if r > 0]
        losses = [abs(r) for r in returns if r < 0]

        win_rate = len(wins) / len(returns)
        avg_win = sum(wins) / len(wins)
        avg_loss = sum(losses) / len(losses)

        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1-win_rate
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p

        kelly_fraction = (b * p - q) / b
        kelly_fraction = max(0.0, kelly_fraction)

        # Verify our manual calculation makes sense
        self.assertGreater(kelly_fraction, 0.0)
        self.assertLess(kelly_fraction, 1.0)

        # With 70% win rate and 2:1 avg win/loss, Kelly should be positive
        print(f"Manual Kelly calculation: {kelly_fraction:.4f}")
        print(
            f"Win rate: {win_rate:.3f}, Avg win: {avg_win:.4f}, Avg loss: {avg_loss:.4f}"
        )


class TestManualATRReference(unittest.TestCase):
    """Manual ATR calculation reference implementation"""

    def test_atr_calculation_reference(self):
        """Test ATR calculation with manual implementation"""
        # Create test price data
        price_data = [
            (100.0, 102.0, 98.0, 101.0),  # (open, high, low, close)
            (101.0, 103.0, 99.0, 102.0),
            (102.0, 104.0, 100.0, 103.0),
            (103.0, 105.0, 101.0, 104.0),
            (104.0, 106.0, 102.0, 105.0),
        ]

        # Manual ATR calculation
        true_ranges = []

        for i in range(1, len(price_data)):
            curr_high = price_data[i][1]
            curr_low = price_data[i][2]
            prev_close = price_data[i - 1][3]

            # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
            tr1 = curr_high - curr_low
            tr2 = abs(curr_high - prev_close)
            tr3 = abs(curr_low - prev_close)

            true_range = max(tr1, tr2, tr3)
            true_ranges.append(true_range)

        atr = sum(true_ranges) / len(true_ranges)

        print(f"Manual ATR calculation: {atr:.4f}")
        print(f"True ranges: {true_ranges}")

        # ATR should be positive and reasonable
        self.assertGreater(atr, 0.0)
        self.assertLess(atr, 10.0)  # Reasonable for our test data


class TestManualDDReference(unittest.TestCase):
    """Manual drawdown calculation reference implementation"""

    def test_drawdown_calculation_reference(self):
        """Test drawdown calculation with manual implementation"""
        equity_curve = [1.0, 1.1, 1.2, 1.15, 1.05, 0.95, 1.0, 1.1]

        # Manual drawdown calculation
        running_max = [equity_curve[0]]
        drawdowns = [0.0]

        for i in range(1, len(equity_curve)):
            current_max = max(running_max[-1], equity_curve[i])
            running_max.append(current_max)

            drawdown = (current_max - equity_curve[i]) / current_max
            drawdowns.append(drawdown)

        max_drawdown = max(drawdowns)

        print(f"Equity curve: {equity_curve}")
        print(f"Running max: {running_max}")
        print(f"Drawdowns: {[f'{dd:.3f}' for dd in drawdowns]}")
        print(f"Max drawdown: {max_drawdown:.3f}")

        # Verify reasonable results
        self.assertGreaterEqual(max_drawdown, 0.0)
        self.assertLessEqual(max_drawdown, 1.0)

        # We know the max DD should be around 20.8% (from peak 1.2 to trough 0.95)
        expected_max_dd = (1.2 - 0.95) / 1.2
        self.assertAlmostEqual(max_drawdown, expected_max_dd, places=2)


if __name__ == "__main__":
    if MODULES_AVAILABLE:
        print("Running Phase D risk module tests...")
    else:
        print("Running manual reference implementation tests only...")

    unittest.main(verbosity=2)
