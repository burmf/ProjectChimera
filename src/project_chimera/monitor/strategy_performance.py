"""
Strategy Performance Tracking System
Comprehensive performance measurement for individual trading strategies
"""

import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import asyncio
from collections import defaultdict, deque
import sqlite3
import pandas as pd
import numpy as np

from ..domains.market import Signal, MarketFrame
from ..settings import get_settings

logger = logging.getLogger(__name__)


class TradeStatus(Enum):
    """Trade execution status"""
    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class TradeRecord:
    """Individual trade record for performance tracking"""
    strategy_id: str
    signal_id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    entry_time: datetime
    entry_price: float
    size_usd: float
    size_native: float
    confidence: float
    
    # Exit information
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    status: TradeStatus = TradeStatus.PENDING
    
    # Performance metrics
    pnl_usd: float = 0.0
    pnl_pct: float = 0.0
    holding_time_seconds: float = 0.0
    
    # Risk metrics
    max_adverse_excursion: float = 0.0  # MAE
    max_favorable_excursion: float = 0.0  # MFE
    
    # Execution metrics
    slippage_bps: float = 0.0
    commission_usd: float = 0.0
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['entry_time'] = self.entry_time.isoformat()
        data['exit_time'] = self.exit_time.isoformat() if self.exit_time else None
        data['status'] = self.status.value
        return data
    
    def calculate_pnl(self) -> None:
        """Calculate P&L metrics"""
        if self.exit_price is None:
            return
        
        if self.side == 'buy':
            self.pnl_usd = (self.exit_price - self.entry_price) * self.size_native
            self.pnl_pct = (self.exit_price / self.entry_price - 1) * 100
        else:  # sell/short
            self.pnl_usd = (self.entry_price - self.exit_price) * self.size_native
            self.pnl_pct = (self.entry_price / self.exit_price - 1) * 100
        
        # Subtract commission
        self.pnl_usd -= self.commission_usd
        
        # Calculate holding time
        if self.exit_time:
            self.holding_time_seconds = (self.exit_time - self.entry_time).total_seconds()


@dataclass
class StrategyStats:
    """Comprehensive strategy performance statistics"""
    strategy_id: str
    
    # Basic metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # P&L metrics
    total_pnl_usd: float = 0.0
    total_pnl_pct: float = 0.0
    avg_win_usd: float = 0.0
    avg_loss_usd: float = 0.0
    largest_win_usd: float = 0.0
    largest_loss_usd: float = 0.0
    
    # Risk metrics
    win_rate: float = 0.0
    profit_factor: float = 0.0  # Gross profit / Gross loss
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar_ratio: float = 0.0
    
    # Execution metrics
    avg_holding_time_hours: float = 0.0
    avg_slippage_bps: float = 0.0
    total_commission_usd: float = 0.0
    
    # Volume metrics
    total_volume_usd: float = 0.0
    avg_trade_size_usd: float = 0.0
    
    # Time-based metrics
    daily_pnl_std: float = 0.0
    monthly_return_pct: float = 0.0
    
    # Risk-adjusted metrics
    information_ratio: float = 0.0
    recovery_factor: float = 0.0
    
    # Recent performance
    last_30d_pnl_usd: float = 0.0
    last_7d_pnl_usd: float = 0.0
    last_24h_pnl_usd: float = 0.0
    
    # Timestamps
    first_trade_time: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None
    last_updated: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        data = asdict(self)
        data['first_trade_time'] = self.first_trade_time.isoformat() if self.first_trade_time else None
        data['last_trade_time'] = self.last_trade_time.isoformat() if self.last_trade_time else None
        data['last_updated'] = self.last_updated.isoformat()
        return data


class StrategyPerformanceTracker:
    """
    Comprehensive strategy performance tracking system
    
    Features:
    - Real-time trade tracking per strategy
    - Comprehensive performance metrics calculation
    - SQLite persistence with time-series data
    - Rolling window statistics
    - Risk-adjusted performance metrics
    - Real-time P&L tracking
    """
    
    def __init__(self, db_path: str = "data/strategy_performance.db"):
        self.db_path = db_path
        self.trade_records: Dict[str, List[TradeRecord]] = defaultdict(list)
        self.strategy_stats: Dict[str, StrategyStats] = {}
        self.open_positions: Dict[str, List[TradeRecord]] = defaultdict(list)
        
        # Real-time tracking
        self.real_time_pnl: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.price_cache: Dict[str, float] = {}
        
        # Performance calculation settings
        self.settings = get_settings()
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
        # Initialize database
        self._init_database()
        
        # Load existing data
        asyncio.create_task(self._load_historical_data())
    
    def _init_database(self) -> None:
        """Initialize SQLite database for persistence"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Trade records table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    signal_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_time TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    size_usd REAL NOT NULL,
                    size_native REAL NOT NULL,
                    confidence REAL NOT NULL,
                    exit_time TEXT,
                    exit_price REAL,
                    status TEXT NOT NULL,
                    pnl_usd REAL DEFAULT 0.0,
                    pnl_pct REAL DEFAULT 0.0,
                    holding_time_seconds REAL DEFAULT 0.0,
                    max_adverse_excursion REAL DEFAULT 0.0,
                    max_favorable_excursion REAL DEFAULT 0.0,
                    slippage_bps REAL DEFAULT 0.0,
                    commission_usd REAL DEFAULT 0.0,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Strategy performance snapshots
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    snapshot_time TEXT NOT NULL,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    total_pnl_usd REAL,
                    win_rate REAL,
                    sharpe_ratio REAL,
                    max_drawdown_pct REAL,
                    total_volume_usd REAL,
                    performance_data TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Real-time P&L tracking
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS realtime_pnl (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    unrealized_pnl_usd REAL,
                    realized_pnl_usd REAL,
                    total_pnl_usd REAL,
                    open_positions INTEGER,
                    market_value_usd REAL
                )
            ''')
            
            # Create indexes for better query performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_strategy ON trade_records(strategy_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_time ON trade_records(entry_time)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_perf_strategy ON strategy_performance(strategy_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pnl_strategy ON realtime_pnl(strategy_id)')
            
            conn.commit()
    
    async def _load_historical_data(self) -> None:
        """Load historical trade data from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM trade_records
                    WHERE status = 'filled'
                    ORDER BY entry_time DESC
                    LIMIT 10000
                ''')
                
                for row in cursor.fetchall():
                    trade = self._row_to_trade_record(row)
                    self.trade_records[trade.strategy_id].append(trade)
                
                # Recalculate statistics for all strategies
                for strategy_id in self.trade_records.keys():
                    await self._calculate_strategy_stats(strategy_id)
                
                logger.info(f"Loaded {sum(len(trades) for trades in self.trade_records.values())} historical trades")
                
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
    
    def _row_to_trade_record(self, row: tuple) -> TradeRecord:
        """Convert database row to TradeRecord object"""
        return TradeRecord(
            strategy_id=row[1],
            signal_id=row[2],
            symbol=row[3],
            side=row[4],
            entry_time=datetime.fromisoformat(row[5]),
            entry_price=row[6],
            size_usd=row[7],
            size_native=row[8],
            confidence=row[9],
            exit_time=datetime.fromisoformat(row[10]) if row[10] else None,
            exit_price=row[11],
            status=TradeStatus(row[12]),
            pnl_usd=row[13],
            pnl_pct=row[14],
            holding_time_seconds=row[15],
            max_adverse_excursion=row[16],
            max_favorable_excursion=row[17],
            slippage_bps=row[18],
            commission_usd=row[19],
            metadata=json.loads(row[20]) if row[20] else {}
        )
    
    async def record_signal_generated(self, signal: Signal, market_frame: MarketFrame) -> str:
        """Record when a signal is generated (not yet executed)"""
        strategy_id = getattr(signal, 'strategy_id', None) or getattr(signal, 'strategy_name', 'unknown')
        signal_id = f"{strategy_id}_{signal.symbol}_{int(time.time())}"
        
        # Store in metadata for later trade recording  
        metadata_field = getattr(signal, 'metadata', None) or getattr(signal, 'indicators_used', {})
        if isinstance(metadata_field, dict):
            metadata_field['signal_id'] = signal_id
            metadata_field['generation_time'] = signal.timestamp.isoformat()
            if hasattr(market_frame, 'current_price') and market_frame.current_price:
                metadata_field['market_price'] = float(market_frame.current_price)
        
        logger.debug(f"Signal generated: {signal_id} for strategy {strategy_id}")
        return signal_id
    
    async def record_trade_entry(
        self,
        strategy_id: str,
        signal: Signal,
        entry_price: float,
        size_usd: float,
        size_native: float,
        slippage_bps: float = 0.0,
        commission_usd: float = 0.0
    ) -> str:
        """Record trade entry execution"""
        
        metadata_field = getattr(signal, 'metadata', None) or getattr(signal, 'indicators_used', {})
        signal_id = metadata_field.get('signal_id', f"{strategy_id}_{int(time.time())}")
        
        trade = TradeRecord(
            strategy_id=strategy_id,
            signal_id=signal_id,
            symbol=signal.symbol,
            side=str(signal.signal_type.value).lower() if hasattr(signal, 'signal_type') else getattr(signal, 'action', 'buy'),
            entry_time=datetime.now(),
            entry_price=entry_price,
            size_usd=size_usd,
            size_native=size_native,
            confidence=signal.confidence,
            slippage_bps=slippage_bps,
            commission_usd=commission_usd,
            metadata=dict(metadata_field)
        )
        
        # Add to open positions
        self.open_positions[strategy_id].append(trade)
        
        # Persist to database
        await self._save_trade_to_db(trade)
        
        side = str(signal.signal_type.value).lower() if hasattr(signal, 'signal_type') else getattr(signal, 'action', 'buy')
        logger.info(f"Trade entry recorded: {signal_id} - {strategy_id} {side} {size_usd:.2f} USD")
        return signal_id
    
    async def record_trade_exit(
        self,
        signal_id: str,
        exit_price: float,
        exit_time: Optional[datetime] = None,
        commission_usd: float = 0.0
    ) -> Optional[TradeRecord]:
        """Record trade exit and calculate performance"""
        
        if exit_time is None:
            exit_time = datetime.now()
        
        # Find the trade in open positions
        trade = None
        strategy_id = None
        
        for strat_id, positions in self.open_positions.items():
            for pos in positions:
                if pos.signal_id == signal_id:
                    trade = pos
                    strategy_id = strat_id
                    break
            if trade:
                break
        
        if not trade:
            logger.warning(f"Trade not found for signal_id: {signal_id}")
            return None
        
        # Update trade record
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.commission_usd += commission_usd
        trade.status = TradeStatus.FILLED
        
        # Calculate performance metrics
        trade.calculate_pnl()
        
        # Remove from open positions and add to completed trades
        self.open_positions[strategy_id].remove(trade)
        self.trade_records[strategy_id].append(trade)
        
        # Update database
        await self._update_trade_in_db(trade)
        
        # Recalculate strategy statistics
        await self._calculate_strategy_stats(strategy_id)
        
        logger.info(f"Trade exit recorded: {signal_id} - P&L: ${trade.pnl_usd:.2f} ({trade.pnl_pct:.2f}%)")
        return trade
    
    async def update_unrealized_pnl(self, symbol: str, current_price: float) -> None:
        """Update unrealized P&L for open positions"""
        self.price_cache[symbol] = current_price
        
        for strategy_id, positions in self.open_positions.items():
            total_unrealized = 0.0
            
            for trade in positions:
                if trade.symbol == symbol and trade.status == TradeStatus.PENDING:
                    # Calculate unrealized P&L
                    if trade.side == 'buy':
                        unrealized_pnl = (current_price - trade.entry_price) * trade.size_native
                    else:
                        unrealized_pnl = (trade.entry_price - current_price) * trade.size_native
                    
                    total_unrealized += unrealized_pnl
                    
                    # Update MAE/MFE
                    if trade.side == 'buy':
                        adverse_move = trade.entry_price - current_price
                        favorable_move = current_price - trade.entry_price
                    else:
                        adverse_move = current_price - trade.entry_price
                        favorable_move = trade.entry_price - current_price
                    
                    trade.max_adverse_excursion = max(trade.max_adverse_excursion, adverse_move)
                    trade.max_favorable_excursion = max(trade.max_favorable_excursion, favorable_move)
            
            # Record real-time P&L
            if total_unrealized != 0:
                await self._record_realtime_pnl(strategy_id, total_unrealized)
    
    async def _calculate_strategy_stats(self, strategy_id: str) -> None:
        """Calculate comprehensive strategy statistics"""
        trades = self.trade_records[strategy_id]
        
        if not trades:
            self.strategy_stats[strategy_id] = StrategyStats(strategy_id=strategy_id)
            return
        
        stats = StrategyStats(strategy_id=strategy_id)
        
        # Basic metrics
        stats.total_trades = len(trades)
        stats.winning_trades = sum(1 for t in trades if t.pnl_usd > 0)
        stats.losing_trades = sum(1 for t in trades if t.pnl_usd < 0)
        
        # P&L metrics
        stats.total_pnl_usd = sum(t.pnl_usd for t in trades)
        stats.total_pnl_pct = sum(t.pnl_pct for t in trades)
        
        wins = [t.pnl_usd for t in trades if t.pnl_usd > 0]
        losses = [t.pnl_usd for t in trades if t.pnl_usd < 0]
        
        stats.avg_win_usd = np.mean(wins) if wins else 0.0
        stats.avg_loss_usd = np.mean(losses) if losses else 0.0
        stats.largest_win_usd = max(wins) if wins else 0.0
        stats.largest_loss_usd = min(losses) if losses else 0.0
        
        # Risk metrics
        stats.win_rate = (stats.winning_trades / stats.total_trades * 100) if stats.total_trades > 0 else 0.0
        
        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        stats.profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf') if gross_profit > 0 else 0.0
        
        # Sharpe ratio calculation
        returns = [t.pnl_pct for t in trades]
        if len(returns) > 1:
            avg_return = np.mean(returns)
            std_return = np.std(returns, ddof=1)
            stats.sharpe_ratio = (avg_return - self.risk_free_rate / 252) / std_return if std_return > 0 else 0.0
        
        # Sortino ratio (downside deviation)
        negative_returns = [r for r in returns if r < 0]
        if negative_returns:
            downside_std = np.std(negative_returns, ddof=1)
            stats.sortino_ratio = (np.mean(returns) - self.risk_free_rate / 252) / downside_std if downside_std > 0 else 0.0
        
        # Maximum drawdown calculation
        cumulative_pnl = 0.0
        peak = 0.0
        max_dd = 0.0
        
        for trade in sorted(trades, key=lambda x: x.entry_time):
            cumulative_pnl += trade.pnl_usd
            peak = max(peak, cumulative_pnl)
            drawdown = (peak - cumulative_pnl) / peak * 100 if peak > 0 else 0.0
            max_dd = max(max_dd, drawdown)
        
        stats.max_drawdown_pct = max_dd
        
        # Calmar ratio
        annualized_return = stats.total_pnl_pct * (252 / len(trades)) if len(trades) > 0 else 0.0
        stats.calmar_ratio = annualized_return / stats.max_drawdown_pct if stats.max_drawdown_pct > 0 else 0.0
        
        # Execution metrics
        stats.avg_holding_time_hours = np.mean([t.holding_time_seconds / 3600 for t in trades if t.holding_time_seconds > 0])
        stats.avg_slippage_bps = np.mean([t.slippage_bps for t in trades])
        stats.total_commission_usd = sum(t.commission_usd for t in trades)
        
        # Volume metrics
        stats.total_volume_usd = sum(t.size_usd for t in trades)
        stats.avg_trade_size_usd = stats.total_volume_usd / stats.total_trades if stats.total_trades > 0 else 0.0
        
        # Time-based metrics
        daily_pnl = defaultdict(float)
        for trade in trades:
            date_key = trade.entry_time.date()
            daily_pnl[date_key] += trade.pnl_usd
        
        daily_values = list(daily_pnl.values())
        stats.daily_pnl_std = np.std(daily_values) if len(daily_values) > 1 else 0.0
        
        # Recent performance
        now = datetime.now()
        stats.last_24h_pnl_usd = sum(t.pnl_usd for t in trades if (now - t.entry_time).days < 1)
        stats.last_7d_pnl_usd = sum(t.pnl_usd for t in trades if (now - t.entry_time).days < 7)
        stats.last_30d_pnl_usd = sum(t.pnl_usd for t in trades if (now - t.entry_time).days < 30)
        
        # Timestamps
        stats.first_trade_time = min(t.entry_time for t in trades)
        stats.last_trade_time = max(t.entry_time for t in trades)
        stats.last_updated = datetime.now()
        
        self.strategy_stats[strategy_id] = stats
        
        # Save snapshot to database
        await self._save_performance_snapshot(stats)
    
    async def _save_trade_to_db(self, trade: TradeRecord) -> None:
        """Save trade record to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO trade_records (
                        strategy_id, signal_id, symbol, side, entry_time, entry_price,
                        size_usd, size_native, confidence, exit_time, exit_price, status,
                        pnl_usd, pnl_pct, holding_time_seconds, max_adverse_excursion,
                        max_favorable_excursion, slippage_bps, commission_usd, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade.strategy_id, trade.signal_id, trade.symbol, trade.side,
                    trade.entry_time.isoformat(), trade.entry_price, trade.size_usd,
                    trade.size_native, trade.confidence,
                    trade.exit_time.isoformat() if trade.exit_time else None,
                    trade.exit_price, trade.status.value, trade.pnl_usd, trade.pnl_pct,
                    trade.holding_time_seconds, trade.max_adverse_excursion,
                    trade.max_favorable_excursion, trade.slippage_bps, trade.commission_usd,
                    json.dumps(trade.metadata)
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving trade to database: {e}")
    
    async def _update_trade_in_db(self, trade: TradeRecord) -> None:
        """Update existing trade record in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE trade_records SET
                        exit_time = ?, exit_price = ?, status = ?, pnl_usd = ?,
                        pnl_pct = ?, holding_time_seconds = ?, max_adverse_excursion = ?,
                        max_favorable_excursion = ?, commission_usd = ?
                    WHERE signal_id = ?
                ''', (
                    trade.exit_time.isoformat() if trade.exit_time else None,
                    trade.exit_price, trade.status.value, trade.pnl_usd, trade.pnl_pct,
                    trade.holding_time_seconds, trade.max_adverse_excursion,
                    trade.max_favorable_excursion, trade.commission_usd, trade.signal_id
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating trade in database: {e}")
    
    async def _save_performance_snapshot(self, stats: StrategyStats) -> None:
        """Save performance snapshot to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO strategy_performance (
                        strategy_id, snapshot_time, total_trades, winning_trades,
                        total_pnl_usd, win_rate, sharpe_ratio, max_drawdown_pct,
                        total_volume_usd, performance_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    stats.strategy_id, stats.last_updated.isoformat(),
                    stats.total_trades, stats.winning_trades, stats.total_pnl_usd,
                    stats.win_rate, stats.sharpe_ratio, stats.max_drawdown_pct,
                    stats.total_volume_usd, json.dumps(stats.to_dict())
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving performance snapshot: {e}")
    
    async def _record_realtime_pnl(self, strategy_id: str, unrealized_pnl: float) -> None:
        """Record real-time P&L data"""
        try:
            realized_pnl = sum(t.pnl_usd for t in self.trade_records[strategy_id])
            total_pnl = realized_pnl + unrealized_pnl
            open_positions = len(self.open_positions[strategy_id])
            
            # Add to rolling buffer
            self.real_time_pnl[strategy_id].append({
                'timestamp': datetime.now(),
                'unrealized_pnl': unrealized_pnl,
                'realized_pnl': realized_pnl,
                'total_pnl': total_pnl,
                'open_positions': open_positions
            })
            
            # Persist to database every 10th update
            if len(self.real_time_pnl[strategy_id]) % 10 == 0:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    
                    cursor.execute('''
                        INSERT INTO realtime_pnl (
                            strategy_id, timestamp, unrealized_pnl_usd,
                            realized_pnl_usd, total_pnl_usd, open_positions
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        strategy_id, datetime.now().isoformat(),
                        unrealized_pnl, realized_pnl, total_pnl, open_positions
                    ))
                    
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Error recording real-time P&L: {e}")
    
    def get_strategy_stats(self, strategy_id: str) -> Optional[StrategyStats]:
        """Get current strategy statistics"""
        return self.strategy_stats.get(strategy_id)
    
    def get_all_strategy_stats(self) -> Dict[str, StrategyStats]:
        """Get statistics for all strategies"""
        return self.strategy_stats.copy()
    
    def get_open_positions(self, strategy_id: Optional[str] = None) -> Dict[str, List[TradeRecord]]:
        """Get current open positions"""
        if strategy_id:
            return {strategy_id: self.open_positions[strategy_id]}
        return dict(self.open_positions)
    
    def get_recent_trades(self, strategy_id: str, limit: int = 50) -> List[TradeRecord]:
        """Get recent completed trades for a strategy"""
        trades = self.trade_records[strategy_id]
        return sorted(trades, key=lambda x: x.entry_time, reverse=True)[:limit]
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary across all strategies"""
        total_trades = sum(stats.total_trades for stats in self.strategy_stats.values())
        total_pnl = sum(stats.total_pnl_usd for stats in self.strategy_stats.values())
        total_volume = sum(stats.total_volume_usd for stats in self.strategy_stats.values())
        
        strategy_count = len(self.strategy_stats)
        avg_win_rate = np.mean([stats.win_rate for stats in self.strategy_stats.values()]) if strategy_count > 0 else 0.0
        
        return {
            'total_strategies': strategy_count,
            'total_trades': total_trades,
            'total_pnl_usd': total_pnl,
            'total_volume_usd': total_volume,
            'average_win_rate': avg_win_rate,
            'best_strategy': max(self.strategy_stats.items(), key=lambda x: x[1].total_pnl_usd)[0] if self.strategy_stats else None,
            'worst_strategy': min(self.strategy_stats.items(), key=lambda x: x[1].total_pnl_usd)[0] if self.strategy_stats else None,
            'last_updated': datetime.now().isoformat()
        }
    
    async def export_performance_data(self, strategy_id: Optional[str] = None) -> pd.DataFrame:
        """Export performance data to pandas DataFrame"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = '''
                    SELECT strategy_id, symbol, side, entry_time, exit_time,
                           entry_price, exit_price, size_usd, pnl_usd, pnl_pct,
                           holding_time_seconds, confidence, slippage_bps, commission_usd
                    FROM trade_records 
                    WHERE status = 'filled'
                '''
                
                if strategy_id:
                    query += ' AND strategy_id = ?'
                    df = pd.read_sql_query(query, conn, params=[strategy_id])
                else:
                    df = pd.read_sql_query(query, conn)
                
                # Convert timestamps
                df['entry_time'] = pd.to_datetime(df['entry_time'])
                df['exit_time'] = pd.to_datetime(df['exit_time'])
                
                return df
                
        except Exception as e:
            logger.error(f"Error exporting performance data: {e}")
            return pd.DataFrame()


# Global instance for easy access
_performance_tracker: Optional[StrategyPerformanceTracker] = None

def get_performance_tracker() -> StrategyPerformanceTracker:
    """Get global performance tracker instance"""
    global _performance_tracker
    if _performance_tracker is None:
        _performance_tracker = StrategyPerformanceTracker()
    return _performance_tracker