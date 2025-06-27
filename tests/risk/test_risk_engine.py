"""
Comprehensive tests for risk management components
Tests Kelly calculator, drawdown manager, ATR sizer, and unified engine
"""

import random
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from project_chimera.domains.market import (
    OHLCV,
    MarketFrame,
    Signal,
    SignalStrength,
    SignalType,
)
from project_chimera.risk.atr_size import ATRPositionSizer
from project_chimera.risk.drawdown import DrawdownManager, DrawdownTier
from project_chimera.risk.engine import RiskDecision, RiskEngine
from project_chimera.risk.equity_cache import EquityCache
from project_chimera.risk.kelly import KellyCalculator


class TestKellyCalculator:
    """Test Kelly Criterion implementation with reference math validation"""

    def setup_method(self):
        self.kelly = KellyCalculator(
            lookback_trades=100,
            kelly_fraction=0.5,
            min_trades=10
        )

    def test_kelly_reference_math(self):
        """Test Kelly calculation against reference mathematical formula"""

        # Known test case: 60% win rate, avg win 2%, avg loss 1%
        # Kelly = (p*b - q) / b = (0.6*2 - 0.4) / 2 = 0.4

        # Generate precise test data
        random.seed(42)  # Reproducible results
        for _ in range(1000):
            if random.random() < 0.6:  # 60% win rate
                self.kelly.add_trade_result(0.02)  # 2% win
            else:
                self.kelly.add_trade_result(-0.01)  # 1% loss

        result = self.kelly.calculate_kelly()

        # Kelly formula: f = (bp - q) / b
        # where b = avg_win/avg_loss, p = win_rate, q = 1-p
        expected_kelly = (0.6 * 2.0 - 0.4) / 2.0  # = 0.4
        expected_fractional = expected_kelly * 0.5  # 50% of Kelly

        # Allow ±1e-6 tolerance as required
        assert abs(result.fraction - expected_fractional) < 1e-6
        assert abs(result.win_rate - 0.6) < 0.05  # 5% tolerance for win rate
        assert abs(result.avg_win / result.avg_loss - 2.0) < 0.1  # 10% tolerance for ratio

    def test_kelly_with_random_pnl(self):
        """Test Kelly calculation reproduces reference math with random P&L"""

        # Test different scenarios
        scenarios = [
            # (win_rate, avg_win, avg_loss, expected_kelly_approx)
            (0.7, 0.015, 0.01, 0.55),  # Good system
            (0.4, 0.05, 0.02, 0.0),    # Bad system (negative Kelly)
            (0.55, 0.02, 0.02, 0.1),   # Marginal system
        ]

        for win_rate, avg_win, avg_loss, expected_kelly in scenarios:
            kelly = KellyCalculator(kelly_fraction=1.0)  # Use full Kelly for test

            # Generate data
            random.seed(123)
            for _ in range(500):
                if random.random() < win_rate:
                    kelly.add_trade_result(avg_win * random.uniform(0.5, 1.5))
                else:
                    kelly.add_trade_result(-avg_loss * random.uniform(0.5, 1.5))

            result = kelly.calculate_kelly()

            # Verify Kelly calculation is reasonable
            if expected_kelly > 0:
                assert result.fraction > 0
                assert abs(result.fraction - expected_kelly) < 0.2  # 20% tolerance
            else:
                assert result.fraction == 0.0  # Negative Kelly should be capped at 0

    def test_insufficient_data_handling(self):
        """Test behavior with insufficient trade data"""
        kelly = KellyCalculator(min_trades=20)

        # Add only 5 trades
        for i in range(5):
            kelly.add_trade_result(0.01 if i % 2 == 0 else -0.005)

        result = kelly.calculate_kelly()

        assert result.fraction == 0.0
        assert result.confidence == 0.0
        assert result.sample_size == 5
        assert not result.is_valid()

    def test_kelly_confidence_scoring(self):
        """Test Kelly confidence calculation"""
        kelly = KellyCalculator()

        # Add consistent profitable trades
        for _ in range(100):
            kelly.add_trade_result(0.02 if random.random() < 0.6 else -0.01)

        result = kelly.calculate_kelly()

        assert 0.0 <= result.confidence <= 1.0
        assert result.confidence > 0.3  # Should have reasonable confidence

    def test_kelly_simulation(self):
        """Test Kelly strategy simulation"""
        kelly = KellyCalculator()

        # Add trade history
        returns = [0.02, -0.01, 0.02, 0.02, -0.01, 0.03, -0.01, 0.02]
        for ret in returns:
            kelly.add_trade_result(ret)

        # Simulate performance
        final_equity, max_dd, sharpe = kelly.simulate_kelly_performance(0.25, 100)

        assert final_equity > 0
        assert 0.0 <= max_dd <= 1.0
        assert isinstance(sharpe, float)


class TestDrawdownManager:
    """Test drawdown management and tier adjustments"""

    def setup_method(self):
        self.dd_mgr = DrawdownManager(
            warning_threshold=0.10,  # 10%
            critical_threshold=0.20,  # 20%
            warning_multiplier=0.5,
            critical_multiplier=0.0
        )

    def test_drawdown_tier_thresholds(self):
        """Test drawdown tier transitions"""

        # Start with normal equity
        state = self.dd_mgr.update_equity(1.0)
        assert state.tier == DrawdownTier.NORMAL
        assert state.position_multiplier == 1.0

        # 5% drawdown - still normal
        state = self.dd_mgr.update_equity(0.95)
        assert state.tier == DrawdownTier.CAUTION

        # 12% drawdown - warning tier
        state = self.dd_mgr.update_equity(0.88)
        assert state.tier == DrawdownTier.WARNING
        assert state.position_multiplier == 0.5

        # 25% drawdown - critical tier
        state = self.dd_mgr.update_equity(0.75)
        assert state.tier == DrawdownTier.CRITICAL
        assert state.position_multiplier == 0.0

    def test_drawdown_cooldown_periods(self):
        """Test trading cooldown functionality"""

        # Trigger critical drawdown
        timestamp = datetime.now()
        state = self.dd_mgr.update_equity(0.75, timestamp)

        assert state.tier == DrawdownTier.CRITICAL
        assert not self.dd_mgr.can_trade()
        assert state.is_in_cooldown()

        # Check cooldown persists
        future_time = timestamp + timedelta(hours=12)
        state = self.dd_mgr.update_equity(0.85, future_time)  # Partial recovery
        assert not self.dd_mgr.can_trade()  # Still in cooldown

        # Check cooldown expires
        far_future = timestamp + timedelta(hours=25)
        state = self.dd_mgr.update_equity(0.85, far_future)
        assert self.dd_mgr.can_trade()  # Cooldown expired

    def test_peak_tracking(self):
        """Test peak equity tracking over time"""

        # Equity progression: up, down, up higher, down
        equities = [1.0, 1.1, 1.05, 1.2, 0.9]

        for equity in equities:
            state = self.dd_mgr.update_equity(equity)

        assert state.peak_equity == 1.2  # Highest point
        assert abs(state.drawdown_pct - (1.2 - 0.9) / 1.2) < 1e-6  # 25% DD

    def test_volatility_adjustment(self):
        """Test volatility-adjusted thresholds"""

        dd_mgr = DrawdownManager(volatility_adjustment=True)

        # Simulate high volatility environment
        for i in range(30):
            daily_return = random.gauss(0, 0.08)  # 8% daily vol
            equity = 1.0 + daily_return * (i + 1) / 30
            dd_mgr.update_equity(equity)

        # High volatility should make thresholds more lenient
        # (This is somewhat stochastic, so we just check it runs)
        metrics = dd_mgr.get_metrics()
        assert 'daily_volatility' in metrics


class TestATRPositionSizer:
    """Test ATR-based position sizing"""

    def setup_method(self):
        self.atr_sizer = ATRPositionSizer(
            target_daily_vol=0.01,  # 1% target
            max_position_pct=0.5
        )

    def generate_ohlcv_data(self, periods: int = 50, base_price: float = 100.0) -> list:
        """Generate synthetic OHLCV data"""
        candles = []
        price = base_price

        for i in range(periods):
            # Random walk with some volatility
            change_pct = random.gauss(0, 0.02)  # 2% daily vol
            price *= (1 + change_pct)

            high = price * 1.01
            low = price * 0.99
            open_price = price * random.uniform(0.995, 1.005)
            close_price = price

            candle = OHLCV(
                symbol="BTCUSDT",
                open=Decimal(str(round(open_price, 2))),
                high=Decimal(str(round(high, 2))),
                low=Decimal(str(round(low, 2))),
                close=Decimal(str(round(close_price, 2))),
                volume=Decimal("1000"),
                timestamp=datetime.now() - timedelta(minutes=periods-i)
            )
            candles.append(candle)

        return candles

    def test_atr_calculation(self):
        """Test ATR calculation accuracy"""

        ohlcv_data = self.generate_ohlcv_data(50, 45000)

        result = self.atr_sizer.calculate_position_size(
            current_price=45000.0,
            ohlcv_data=ohlcv_data,
            portfolio_value=10000.0
        )

        assert result.atr_value > 0
        assert result.price_volatility > 0
        assert 0.0 <= result.position_size_pct <= 0.5
        assert result.is_valid()

    def test_volatility_targeting(self):
        """Test that position sizing targets correct volatility"""

        # Generate high volatility data
        high_vol_data = []
        price = 45000.0
        for i in range(30):
            change = random.gauss(0, 0.05)  # 5% daily vol
            price *= (1 + change)

            candle = OHLCV(
                symbol="BTCUSDT",
                open=Decimal(str(price)),
                high=Decimal(str(price * 1.02)),
                low=Decimal(str(price * 0.98)),
                close=Decimal(str(price)),
                volume=Decimal("1000"),
                timestamp=datetime.now() - timedelta(minutes=30-i)
            )
            high_vol_data.append(candle)

        result = self.atr_sizer.calculate_position_size(
            current_price=price,
            ohlcv_data=high_vol_data,
            portfolio_value=10000.0
        )

        # High volatility should result in smaller position size
        # to target the same portfolio volatility
        assert result.position_size_pct < 0.3  # Should be relatively small

        # Estimate portfolio vol
        portfolio_vol = self.atr_sizer.estimate_daily_portfolio_vol(
            result.position_size_pct,
            result.price_volatility
        )

        # Should be close to target (1%)
        assert abs(portfolio_vol - 0.01) < 0.02  # Within 2% tolerance

    def test_leverage_calculation(self):
        """Test leverage calculation"""

        # Test normal case
        leverage = self.atr_sizer.calculate_leverage(
            position_size_pct=0.20,  # Want 20% position
            available_margin=0.10,   # Have 10% margin
            max_leverage=10.0
        )

        assert leverage == 2.0  # 20% / 10% = 2x leverage

        # Test leverage capping
        high_leverage = self.atr_sizer.calculate_leverage(
            position_size_pct=0.50,
            available_margin=0.02,
            max_leverage=5.0
        )

        assert high_leverage == 5.0  # Capped at max


class TestEquityCache:
    """Test equity curve caching and statistics"""

    def setup_method(self):
        self.cache = EquityCache(initial_equity=1.0)

    def test_equity_tracking(self):
        """Test basic equity point addition and retrieval"""

        # Add some equity points
        points = [
            (1.0, 0.0),
            (1.05, 0.05),
            (0.98, -0.07),
            (1.02, 0.04)
        ]

        for i, (equity, pnl) in enumerate(points):
            timestamp = datetime.now() + timedelta(minutes=i)
            self.cache.add_equity_point(equity, pnl, timestamp)

        assert self.cache.get_current_equity() == 1.02
        assert len(self.cache.equity_points) == 4

    def test_drawdown_calculation(self):
        """Test drawdown calculation accuracy"""

        # Create equity curve: 1.0 -> 1.2 -> 0.9 -> 1.1
        equities = [1.0, 1.2, 0.9, 1.1]

        for i, equity in enumerate(equities):
            timestamp = datetime.now() + timedelta(hours=i)
            self.cache.add_equity_point(equity, 0.0, timestamp)

        # Max drawdown should be (1.2 - 0.9) / 1.2 = 0.25 (25%)
        stats = self.cache.calculate_statistics()

        assert abs(stats.max_drawdown - 0.25) < 1e-6
        assert stats.peak_equity == 1.2
        assert stats.current_equity == 1.1

    def test_statistics_calculation(self):
        """Test comprehensive statistics calculation"""

        # Add a series of trades
        trade_results = [0.02, -0.01, 0.03, -0.015, 0.025, -0.01, 0.02]
        equity = 1.0

        for i, ret in enumerate(trade_results):
            pnl = equity * ret
            equity += pnl
            timestamp = datetime.now() + timedelta(days=i)
            self.cache.add_equity_point(equity, pnl, timestamp)

        stats = self.cache.calculate_statistics()

        assert stats.total_trades == len(trade_results)
        assert stats.winning_trades == len([r for r in trade_results if r > 0])
        assert stats.losing_trades == len([r for r in trade_results if r < 0])
        assert 0.0 <= stats.win_rate <= 1.0
        assert stats.profit_factor > 0

    def test_persistence(self, tmp_path):
        """Test saving and loading equity cache"""

        cache_file = tmp_path / "test_cache.json"
        cache = EquityCache(persistence_file=str(cache_file))

        # Add some data
        cache.add_equity_point(1.05, 0.05)
        cache.add_equity_point(1.10, 0.05)

        # Save
        cache.save_to_file()

        # Load in new instance
        cache2 = EquityCache(persistence_file=str(cache_file))
        cache2.load_from_file()

        assert cache2.get_current_equity() == 1.10
        assert len(cache2.equity_points) == 2


class TestRiskEngine:
    """Test unified risk engine integration"""

    def setup_method(self):
        self.engine = RiskEngine()

    def create_test_signal(self) -> Signal:
        """Create test trading signal"""
        return Signal(
            symbol="BTCUSDT",
            signal_type=SignalType.BUY,
            strength=SignalStrength.MEDIUM,
            price=Decimal("45000"),
            timestamp=datetime.now(),
            strategy_name="test_strategy",
            confidence=0.7
        )

    def create_test_market_frame(self) -> MarketFrame:
        """Create test market data"""
        # Generate synthetic OHLCV data
        ohlcv_data = []
        price = 45000.0

        for i in range(50):
            change = random.gauss(0, 0.02)
            price *= (1 + change)

            candle = OHLCV(
                symbol="BTCUSDT",
                open=Decimal(str(price)),
                high=Decimal(str(price * 1.01)),
                low=Decimal(str(price * 0.99)),
                close=Decimal(str(price)),
                volume=Decimal("1000"),
                timestamp=datetime.now() - timedelta(minutes=50-i)
            )
            ohlcv_data.append(candle)

        return MarketFrame(
            symbol="BTCUSDT",
            timestamp=datetime.now(),
            ohlcv_1m=ohlcv_data
        )

    def test_position_sizing_integration(self):
        """Test integrated position sizing across all components"""

        signal = self.create_test_signal()
        market_data = self.create_test_market_frame()

        # Add some trade history for Kelly
        for _ in range(30):
            ret = random.gauss(0.01, 0.02)  # 1% avg return, 2% vol
            self.engine.update_trade_result(ret, 10000.0)

        # Calculate position size
        decision = self.engine.calculate_position_size(
            signal=signal,
            market_data=market_data,
            portfolio_value=10000.0
        )

        assert isinstance(decision, RiskDecision)
        assert 0.0 <= decision.position_size_pct <= 1.0
        assert decision.leverage >= 1.0
        assert isinstance(decision.reasoning, str)

        if decision.can_trade:
            assert decision.is_valid()

    def test_drawdown_position_reduction(self):
        """Test that drawdowns reduce position sizes"""

        signal = self.create_test_signal()
        market_data = self.create_test_market_frame()

        # Start with normal portfolio
        decision1 = self.engine.calculate_position_size(
            signal=signal,
            market_data=market_data,
            portfolio_value=10000.0
        )

        # Simulate large loss (15% drawdown)
        self.engine.update_trade_result(-0.15, 8500.0)

        decision2 = self.engine.calculate_position_size(
            signal=signal,
            market_data=market_data,
            portfolio_value=8500.0
        )

        # Position size should be reduced due to drawdown
        if decision1.can_trade and decision2.can_trade:
            assert decision2.position_size_pct <= decision1.position_size_pct

    def test_risk_metrics(self):
        """Test risk metrics reporting"""

        # Add some trade history
        for i in range(20):
            ret = random.gauss(0.005, 0.02)
            portfolio_value = 10000 * (1 + ret * i / 20)
            self.engine.update_trade_result(ret, portfolio_value)

        metrics = self.engine.get_risk_metrics()

        # Check structure
        assert 'kelly' in metrics
        assert 'drawdown' in metrics
        assert 'atr' in metrics
        assert 'equity' in metrics

        # Check Kelly metrics
        kelly_metrics = metrics['kelly']
        assert 'total_trades' in kelly_metrics
        assert 'win_rate' in kelly_metrics

        # Check drawdown metrics
        dd_metrics = metrics['drawdown']
        assert 'current_pct' in dd_metrics
        assert 'can_trade' in dd_metrics

    def test_state_persistence(self, tmp_path):
        """Test saving and loading engine state"""

        # Configure engine with persistence
        cache_file = tmp_path / "engine_state.json"
        engine = RiskEngine(equity_cache_file=str(cache_file))

        # Add some history
        engine.update_trade_result(0.02, 10200.0)
        engine.update_trade_result(-0.01, 10098.0)

        # Save state
        engine.save_state()

        # Create new engine and load
        engine2 = RiskEngine(equity_cache_file=str(cache_file))
        success = engine2.load_state()

        assert success
        assert engine2.equity_cache.get_current_equity() > 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
