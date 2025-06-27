"""
Tests for Strategy Performance Tracking System
"""

import asyncio
import os
import tempfile
from datetime import datetime, timedelta
from decimal import Decimal

import pytest

from project_chimera.domains.market import (
    MarketFrame, Signal, SignalStrength, SignalType, Ticker
)
from project_chimera.monitor.strategy_performance import (
    StrategyPerformanceTracker,
    StrategyStats,
    TradeRecord,
    TradeStatus,
)


@pytest.fixture
def temp_db():
    """Create temporary database for testing"""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_performance.db")
    yield db_path
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)
    os.rmdir(temp_dir)


@pytest.fixture
def performance_tracker(temp_db):
    """Create performance tracker with temporary database"""
    return StrategyPerformanceTracker(db_path=temp_db)


@pytest.fixture
def sample_signal():
    """Create sample trading signal"""
    return Signal(
        symbol="BTCUSDT",
        signal_type=SignalType.BUY,
        strength=SignalStrength.MEDIUM,
        price=Decimal("50000.0"),
        timestamp=datetime.now(),
        strategy_name="test_strategy",
        confidence=0.75
    )


@pytest.fixture
def sample_market_frame():
    """Create sample market frame"""
    return MarketFrame(
        symbol="BTCUSDT",
        timestamp=datetime.now(),
        ticker=Ticker(
            symbol="BTCUSDT",
            price=Decimal("50000.0"),
            volume_24h=Decimal("1000.0"),
            change_24h=Decimal("0.05"),
            timestamp=datetime.now(),
        )
    )


class TestTradeRecord:
    """Test TradeRecord functionality"""

    def test_trade_record_creation(self):
        """Test creating a trade record"""
        trade = TradeRecord(
            strategy_id="test_strategy",
            signal_id="signal_123",
            symbol="BTCUSDT",
            side="buy",
            entry_time=datetime.now(),
            entry_price=50000.0,
            size_usd=1000.0,
            size_native=0.02,
            confidence=0.75
        )

        assert trade.strategy_id == "test_strategy"
        assert trade.signal_id == "signal_123"
        assert trade.symbol == "BTCUSDT"
        assert trade.side == "buy"
        assert trade.pnl_usd == 0.0
        assert trade.status == TradeStatus.PENDING

    def test_pnl_calculation_buy_profit(self):
        """Test P&L calculation for profitable buy trade"""
        trade = TradeRecord(
            strategy_id="test_strategy",
            signal_id="signal_123",
            symbol="BTCUSDT",
            side="buy",
            entry_time=datetime.now(),
            entry_price=50000.0,
            size_usd=1000.0,
            size_native=0.02,
            confidence=0.75,
            exit_price=52000.0,
            exit_time=datetime.now() + timedelta(hours=1)
        )

        trade.calculate_pnl()

        # P&L = (52000 - 50000) * 0.02 = 40 USD
        assert trade.pnl_usd == 40.0
        assert abs(trade.pnl_pct - 4.0) < 0.01  # 4% profit
        assert trade.holding_time_seconds == 3600.0  # 1 hour

    def test_pnl_calculation_sell_profit(self):
        """Test P&L calculation for profitable sell trade"""
        trade = TradeRecord(
            strategy_id="test_strategy",
            signal_id="signal_123",
            symbol="BTCUSDT",
            side="sell",
            entry_time=datetime.now(),
            entry_price=50000.0,
            size_usd=1000.0,
            size_native=0.02,
            confidence=0.75,
            exit_price=48000.0,
            exit_time=datetime.now() + timedelta(hours=2)
        )

        trade.calculate_pnl()

        # P&L = (50000 - 48000) * 0.02 = 40 USD
        assert trade.pnl_usd == 40.0
        assert abs(trade.pnl_pct - 4.17) < 0.01  # ~4.17% profit
        assert trade.holding_time_seconds == 7200.0  # 2 hours

    def test_pnl_calculation_with_commission(self):
        """Test P&L calculation including commission"""
        trade = TradeRecord(
            strategy_id="test_strategy",
            signal_id="signal_123",
            symbol="BTCUSDT",
            side="buy",
            entry_time=datetime.now(),
            entry_price=50000.0,
            size_usd=1000.0,
            size_native=0.02,
            confidence=0.75,
            exit_price=52000.0,
            exit_time=datetime.now() + timedelta(hours=1),
            commission_usd=5.0
        )

        trade.calculate_pnl()

        # P&L = (52000 - 50000) * 0.02 - 5 = 35 USD
        assert trade.pnl_usd == 35.0


class TestStrategyPerformanceTracker:
    """Test StrategyPerformanceTracker functionality"""

    @pytest.mark.asyncio
    async def test_initialization(self, performance_tracker):
        """Test tracker initialization"""
        assert performance_tracker.db_path is not None
        assert len(performance_tracker.trade_records) == 0
        assert len(performance_tracker.strategy_stats) == 0
        assert len(performance_tracker.open_positions) == 0

    @pytest.mark.asyncio
    async def test_record_signal_generated(self, performance_tracker, sample_signal, sample_market_frame):
        """Test recording signal generation"""
        signal_id = await performance_tracker.record_signal_generated(sample_signal, sample_market_frame)

        assert signal_id is not None
        assert sample_signal.metadata['signal_id'] == signal_id
        assert 'generation_time' in sample_signal.metadata
        assert 'market_price' in sample_signal.metadata

    @pytest.mark.asyncio
    async def test_record_trade_entry(self, performance_tracker, sample_signal):
        """Test recording trade entry"""
        signal_id = await performance_tracker.record_trade_entry(
            strategy_id="test_strategy",
            signal=sample_signal,
            entry_price=50000.0,
            size_usd=1000.0,
            size_native=0.02,
            slippage_bps=2.0,
            commission_usd=1.0
        )

        assert signal_id is not None
        assert len(performance_tracker.open_positions["test_strategy"]) == 1

        trade = performance_tracker.open_positions["test_strategy"][0]
        assert trade.strategy_id == "test_strategy"
        assert trade.entry_price == 50000.0
        assert trade.size_usd == 1000.0
        assert trade.slippage_bps == 2.0
        assert trade.commission_usd == 1.0

    @pytest.mark.asyncio
    async def test_record_trade_exit(self, performance_tracker, sample_signal):
        """Test recording trade exit"""
        # First record entry
        signal_id = await performance_tracker.record_trade_entry(
            strategy_id="test_strategy",
            signal=sample_signal,
            entry_price=50000.0,
            size_usd=1000.0,
            size_native=0.02
        )

        # Then record exit
        trade = await performance_tracker.record_trade_exit(
            signal_id=signal_id,
            exit_price=52000.0,
            commission_usd=1.0
        )

        assert trade is not None
        assert trade.exit_price == 52000.0
        assert trade.status == TradeStatus.FILLED
        assert trade.pnl_usd == 39.0  # (52000-50000)*0.02 - 1.0 commission

        # Check that trade moved from open to completed
        assert len(performance_tracker.open_positions["test_strategy"]) == 0
        assert len(performance_tracker.trade_records["test_strategy"]) == 1

    @pytest.mark.asyncio
    async def test_update_unrealized_pnl(self, performance_tracker, sample_signal):
        """Test updating unrealized P&L"""
        # Record a trade entry
        await performance_tracker.record_trade_entry(
            strategy_id="test_strategy",
            signal=sample_signal,
            entry_price=50000.0,
            size_usd=1000.0,
            size_native=0.02
        )

        # Update unrealized P&L
        await performance_tracker.update_unrealized_pnl("BTCUSDT", 51000.0)

        # Check that MAE/MFE are updated
        trade = performance_tracker.open_positions["test_strategy"][0]
        assert trade.max_favorable_excursion > 0  # Price moved in favor

    @pytest.mark.asyncio
    async def test_strategy_stats_calculation(self, performance_tracker, sample_signal):
        """Test strategy statistics calculation"""
        strategy_id = "test_strategy"

        # Create multiple completed trades
        for i in range(5):
            signal_id = await performance_tracker.record_trade_entry(
                strategy_id=strategy_id,
                signal=sample_signal,
                entry_price=50000.0,
                size_usd=1000.0,
                size_native=0.02
            )

            # Exit with varying profits/losses
            exit_price = 50000.0 + (i - 2) * 500  # Some wins, some losses
            await performance_tracker.record_trade_exit(
                signal_id=signal_id,
                exit_price=exit_price
            )

        # Get calculated stats
        stats = performance_tracker.get_strategy_stats(strategy_id)

        assert stats is not None
        assert stats.strategy_id == strategy_id
        assert stats.total_trades == 5
        assert stats.winning_trades > 0
        assert stats.losing_trades > 0
        assert stats.win_rate > 0
        assert stats.profit_factor >= 0

    def test_get_performance_summary(self, performance_tracker):
        """Test getting overall performance summary"""
        # Add mock stats
        performance_tracker.strategy_stats["strategy1"] = StrategyStats(
            strategy_id="strategy1",
            total_trades=10,
            total_pnl_usd=100.0,
            total_volume_usd=5000.0,
            win_rate=60.0
        )

        performance_tracker.strategy_stats["strategy2"] = StrategyStats(
            strategy_id="strategy2",
            total_trades=15,
            total_pnl_usd=200.0,
            total_volume_usd=7500.0,
            win_rate=70.0
        )

        summary = performance_tracker.get_performance_summary()

        assert summary['total_strategies'] == 2
        assert summary['total_trades'] == 25
        assert summary['total_pnl_usd'] == 300.0
        assert summary['total_volume_usd'] == 12500.0
        assert summary['average_win_rate'] == 65.0
        assert summary['best_strategy'] == "strategy2"

    def test_get_open_positions(self, performance_tracker):
        """Test getting open positions"""
        # Add mock open position
        trade = TradeRecord(
            strategy_id="test_strategy",
            signal_id="signal_123",
            symbol="BTCUSDT",
            side="buy",
            entry_time=datetime.now(),
            entry_price=50000.0,
            size_usd=1000.0,
            size_native=0.02,
            confidence=0.75
        )

        performance_tracker.open_positions["test_strategy"].append(trade)

        # Test getting all open positions
        all_positions = performance_tracker.get_open_positions()
        assert "test_strategy" in all_positions
        assert len(all_positions["test_strategy"]) == 1

        # Test getting positions for specific strategy
        strategy_positions = performance_tracker.get_open_positions("test_strategy")
        assert "test_strategy" in strategy_positions
        assert len(strategy_positions["test_strategy"]) == 1

    def test_get_recent_trades(self, performance_tracker):
        """Test getting recent trades"""
        strategy_id = "test_strategy"

        # Add mock completed trades
        for i in range(10):
            trade = TradeRecord(
                strategy_id=strategy_id,
                signal_id=f"signal_{i}",
                symbol="BTCUSDT",
                side="buy",
                entry_time=datetime.now() - timedelta(hours=i),
                entry_price=50000.0,
                size_usd=1000.0,
                size_native=0.02,
                confidence=0.75,
                status=TradeStatus.FILLED
            )
            performance_tracker.trade_records[strategy_id].append(trade)

        # Get recent trades
        recent_trades = performance_tracker.get_recent_trades(strategy_id, limit=5)

        assert len(recent_trades) == 5
        # Should be sorted by entry time, most recent first
        assert recent_trades[0].entry_time > recent_trades[1].entry_time


class TestStrategyStats:
    """Test StrategyStats functionality"""

    def test_strategy_stats_creation(self):
        """Test creating strategy stats"""
        stats = StrategyStats(strategy_id="test_strategy")

        assert stats.strategy_id == "test_strategy"
        assert stats.total_trades == 0
        assert stats.winning_trades == 0
        assert stats.total_pnl_usd == 0.0
        assert stats.win_rate == 0.0

    def test_strategy_stats_to_dict(self):
        """Test converting stats to dictionary"""
        stats = StrategyStats(
            strategy_id="test_strategy",
            total_trades=10,
            total_pnl_usd=100.0,
            win_rate=60.0,
            first_trade_time=datetime.now(),
            last_trade_time=datetime.now()
        )

        stats_dict = stats.to_dict()

        assert stats_dict['strategy_id'] == "test_strategy"
        assert stats_dict['total_trades'] == 10
        assert stats_dict['total_pnl_usd'] == 100.0
        assert 'first_trade_time' in stats_dict
        assert 'last_trade_time' in stats_dict


@pytest.mark.asyncio
async def test_concurrent_trade_operations(performance_tracker, sample_signal):
    """Test concurrent trade operations"""
    strategy_id = "test_strategy"

    # Create multiple concurrent trade operations
    tasks = []
    for i in range(5):
        task = performance_tracker.record_trade_entry(
            strategy_id=strategy_id,
            signal=sample_signal,
            entry_price=50000.0 + i * 10,
            size_usd=1000.0,
            size_native=0.02
        )
        tasks.append(task)

    # Execute all trades concurrently
    signal_ids = await asyncio.gather(*tasks)

    assert len(signal_ids) == 5
    assert len(performance_tracker.open_positions[strategy_id]) == 5

    # Exit all trades concurrently
    exit_tasks = []
    for i, signal_id in enumerate(signal_ids):
        task = performance_tracker.record_trade_exit(
            signal_id=signal_id,
            exit_price=51000.0 + i * 10
        )
        exit_tasks.append(task)

    trades = await asyncio.gather(*exit_tasks)

    assert len(trades) == 5
    assert all(trade is not None for trade in trades)
    assert len(performance_tracker.open_positions[strategy_id]) == 0
    assert len(performance_tracker.trade_records[strategy_id]) == 5


if __name__ == "__main__":
    pytest.main([__file__])
