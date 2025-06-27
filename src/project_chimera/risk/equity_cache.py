"""
Equity curve cache for drawdown calculations
Maintains running equity history with efficient peak tracking
"""

import bisect
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


@dataclass
class EquityPoint:
    """Single equity curve point"""

    timestamp: datetime
    equity: float
    pnl: float  # Trade P&L that led to this equity
    trade_id: str | None = None
    notes: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "equity": self.equity,
            "pnl": self.pnl,
            "trade_id": self.trade_id,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EquityPoint":
        """Create from dictionary"""
        return cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            equity=data["equity"],
            pnl=data["pnl"],
            trade_id=data.get("trade_id"),
            notes=data.get("notes"),
        )


@dataclass
class EquityStats:
    """Equity curve statistics"""

    current_equity: float
    peak_equity: float
    peak_timestamp: datetime
    current_drawdown: float
    max_drawdown: float
    max_drawdown_timestamp: datetime
    total_return: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    time_in_drawdown_pct: float


class EquityCache:
    """
    Efficient equity curve cache with peak tracking and persistence

    Features:
    - Rolling window storage with configurable retention
    - Fast peak/drawdown calculations
    - Automatic persistence to disk
    - Statistics calculation
    - Memory-efficient binary search for lookups
    - Backfill capability for historical data
    """

    def __init__(
        self,
        initial_equity: float = 1.0,
        max_history_days: int = 365,
        persistence_file: str | None = None,
        auto_save_interval: int = 100,  # Save every N updates
        precision: int = 6,  # Decimal places for equity
    ):
        self.initial_equity = initial_equity
        self.max_history = timedelta(days=max_history_days)
        self.persistence_file = persistence_file
        self.auto_save_interval = auto_save_interval
        self.precision = precision

        # Equity history (maintained in chronological order)
        self.equity_points: list[EquityPoint] = []

        # Peak tracking for efficient drawdown calculations
        self.peak_equity = initial_equity
        self.peak_timestamp = datetime.now()

        # Statistics cache (invalidated on updates)
        self._stats_cache: EquityStats | None = None
        self._cache_dirty = True

        # Counters
        self.update_count = 0

        # Load from persistence if available
        if persistence_file:
            self.load_from_file()

    def add_equity_point(
        self,
        equity: float,
        pnl: float = 0.0,
        timestamp: datetime | None = None,
        trade_id: str | None = None,
        notes: str | None = None,
    ) -> None:
        """Add new equity point to the cache"""

        if timestamp is None:
            timestamp = datetime.now()

        # Round equity for consistency
        equity = round(equity, self.precision)
        pnl = round(pnl, self.precision)

        # Create equity point
        point = EquityPoint(
            timestamp=timestamp, equity=equity, pnl=pnl, trade_id=trade_id, notes=notes
        )

        # Insert in chronological order (binary search)
        self._insert_point(point)

        # Update peak tracking
        if equity > self.peak_equity:
            self.peak_equity = equity
            self.peak_timestamp = timestamp

        # Maintain history window
        self._cleanup_old_data(timestamp)

        # Invalidate stats cache
        self._cache_dirty = True

        # Auto-save
        self.update_count += 1
        if self.persistence_file and self.update_count % self.auto_save_interval == 0:
            self.save_to_file()

    def _insert_point(self, point: EquityPoint) -> None:
        """Insert point in chronological order using binary search"""

        # Find insertion position
        timestamps = [p.timestamp for p in self.equity_points]
        pos = bisect.bisect_left(timestamps, point.timestamp)

        # Check for duplicate timestamp
        if (
            pos < len(self.equity_points)
            and self.equity_points[pos].timestamp == point.timestamp
        ):
            # Update existing point
            self.equity_points[pos] = point
        else:
            # Insert new point
            self.equity_points.insert(pos, point)

    def _cleanup_old_data(self, current_time: datetime) -> None:
        """Remove data older than max_history"""
        cutoff_time = current_time - self.max_history

        # Find first point to keep
        timestamps = [p.timestamp for p in self.equity_points]
        cutoff_pos = bisect.bisect_left(timestamps, cutoff_time)

        # Remove old points
        if cutoff_pos > 0:
            self.equity_points = self.equity_points[cutoff_pos:]

    def get_current_equity(self) -> float:
        """Get most recent equity value"""
        if not self.equity_points:
            return self.initial_equity
        return self.equity_points[-1].equity

    def get_drawdown(self, timestamp: datetime | None = None) -> float:
        """
        Calculate drawdown at specific timestamp

        Returns drawdown as positive percentage (0.0 to 1.0)
        """
        if timestamp is None:
            current_equity = self.get_current_equity()
        else:
            current_equity = self.get_equity_at_time(timestamp)

        if current_equity is None:
            return 0.0

        # Find peak equity up to this timestamp
        peak = self._get_peak_equity_before(timestamp)

        if peak <= 0:
            return 0.0

        drawdown = max(0.0, (peak - current_equity) / peak)
        return drawdown

    def get_equity_at_time(self, timestamp: datetime) -> float | None:
        """Get equity at specific timestamp (interpolated if needed)"""
        if not self.equity_points:
            return self.initial_equity

        # Find points around timestamp
        timestamps = [p.timestamp for p in self.equity_points]
        pos = bisect.bisect_left(timestamps, timestamp)

        if pos == 0:
            # Before first point
            return (
                self.equity_points[0].equity
                if self.equity_points
                else self.initial_equity
            )
        elif pos >= len(self.equity_points):
            # After last point
            return self.equity_points[-1].equity
        else:
            # Exact match or interpolation
            if self.equity_points[pos].timestamp == timestamp:
                return self.equity_points[pos].equity
            else:
                # Linear interpolation between points
                p1 = self.equity_points[pos - 1]
                p2 = self.equity_points[pos]

                time_diff = (p2.timestamp - p1.timestamp).total_seconds()
                if time_diff <= 0:
                    return p1.equity

                weight = (timestamp - p1.timestamp).total_seconds() / time_diff
                equity = p1.equity + weight * (p2.equity - p1.equity)
                return equity

    def _get_peak_equity_before(self, timestamp: datetime | None = None) -> float:
        """Get peak equity before given timestamp"""
        if timestamp is None:
            return self.peak_equity

        peak = self.initial_equity
        for point in self.equity_points:
            if point.timestamp > timestamp:
                break
            if point.equity > peak:
                peak = point.equity

        return peak

    def backfill_equity_curve(
        self, historical_points: list[tuple[datetime, float, float]]
    ) -> None:
        """
        Backfill historical equity data

        Args:
            historical_points: List of (timestamp, equity, pnl) tuples
        """

        # Add all points
        for timestamp, equity, pnl in historical_points:
            self.add_equity_point(equity, pnl, timestamp)

        # Recalculate peak
        self._recalculate_peak()

        # Force stats recalculation
        self._cache_dirty = True

    def _recalculate_peak(self) -> None:
        """Recalculate peak equity from all data"""
        if not self.equity_points:
            self.peak_equity = self.initial_equity
            self.peak_timestamp = datetime.now()
            return

        self.peak_equity = max(
            self.initial_equity, max(p.equity for p in self.equity_points)
        )

        # Find timestamp of peak
        for point in self.equity_points:
            if point.equity == self.peak_equity:
                self.peak_timestamp = point.timestamp
                break

    def calculate_statistics(self, force_refresh: bool = False) -> EquityStats:
        """Calculate comprehensive equity curve statistics"""

        if not force_refresh and not self._cache_dirty and self._stats_cache:
            return self._stats_cache

        if not self.equity_points:
            # Return default stats
            now = datetime.now()
            return EquityStats(
                current_equity=self.initial_equity,
                peak_equity=self.initial_equity,
                peak_timestamp=now,
                current_drawdown=0.0,
                max_drawdown=0.0,
                max_drawdown_timestamp=now,
                total_return=0.0,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                average_win=0.0,
                average_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                calmar_ratio=0.0,
                time_in_drawdown_pct=0.0,
            )

        current_equity = self.equity_points[-1].equity
        current_drawdown = self.get_drawdown()

        # Calculate max drawdown
        max_dd = 0.0
        max_dd_timestamp = self.equity_points[0].timestamp
        peak_so_far = self.initial_equity

        for point in self.equity_points:
            if point.equity > peak_so_far:
                peak_so_far = point.equity

            if peak_so_far > 0:
                dd = (peak_so_far - point.equity) / peak_so_far
                if dd > max_dd:
                    max_dd = dd
                    max_dd_timestamp = point.timestamp

        # Trade statistics
        trades = [p for p in self.equity_points if p.pnl != 0]
        total_trades = len(trades)

        if total_trades > 0:
            winning_trades = len([t for t in trades if t.pnl > 0])
            losing_trades = len([t for t in trades if t.pnl < 0])
            win_rate = winning_trades / total_trades

            wins = [t.pnl for t in trades if t.pnl > 0]
            losses = [abs(t.pnl) for t in trades if t.pnl < 0]

            average_win = sum(wins) / len(wins) if wins else 0.0
            average_loss = sum(losses) / len(losses) if losses else 0.0
            largest_win = max(wins) if wins else 0.0
            largest_loss = max(losses) if losses else 0.0

            total_wins = sum(wins)
            total_losses = sum(losses)
            profit_factor = (
                total_wins / total_losses if total_losses > 0 else float("inf")
            )
        else:
            winning_trades = losing_trades = 0
            win_rate = average_win = average_loss = 0.0
            largest_win = largest_loss = profit_factor = 0.0

        # Return calculations
        total_return = (current_equity - self.initial_equity) / self.initial_equity

        # Risk-adjusted metrics
        sharpe = self._calculate_sharpe_ratio()
        sortino = self._calculate_sortino_ratio()
        calmar = abs(total_return / max_dd) if max_dd > 0 else 0.0

        # Time in drawdown
        time_in_dd_pct = self._calculate_time_in_drawdown()

        # Cache results
        self._stats_cache = EquityStats(
            current_equity=current_equity,
            peak_equity=self.peak_equity,
            peak_timestamp=self.peak_timestamp,
            current_drawdown=current_drawdown,
            max_drawdown=max_dd,
            max_drawdown_timestamp=max_dd_timestamp,
            total_return=total_return,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            average_win=average_win,
            average_loss=average_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            time_in_drawdown_pct=time_in_dd_pct,
        )

        self._cache_dirty = False
        return self._stats_cache

    def _calculate_sharpe_ratio(self, periods_per_year: int = 365) -> float:
        """Calculate annualized Sharpe ratio"""
        if len(self.equity_points) < 2:
            return 0.0

        # Calculate daily returns
        returns = []
        for i in range(1, len(self.equity_points)):
            prev_equity = self.equity_points[i - 1].equity
            curr_equity = self.equity_points[i].equity
            if prev_equity > 0:
                daily_return = (curr_equity - prev_equity) / prev_equity
                returns.append(daily_return)

        if len(returns) < 2:
            return 0.0

        import statistics

        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)

        if std_return <= 0:
            return 0.0

        # Annualize
        annualized_return = mean_return * periods_per_year
        annualized_std = std_return * (periods_per_year**0.5)

        return annualized_return / annualized_std

    def _calculate_sortino_ratio(self, periods_per_year: int = 365) -> float:
        """Calculate annualized Sortino ratio (only downside deviation)"""
        if len(self.equity_points) < 2:
            return 0.0

        # Calculate daily returns
        returns = []
        for i in range(1, len(self.equity_points)):
            prev_equity = self.equity_points[i - 1].equity
            curr_equity = self.equity_points[i].equity
            if prev_equity > 0:
                daily_return = (curr_equity - prev_equity) / prev_equity
                returns.append(daily_return)

        if len(returns) < 2:
            return 0.0

        import statistics

        mean_return = statistics.mean(returns)

        # Downside deviation (only negative returns)
        negative_returns = [r for r in returns if r < 0]
        if len(negative_returns) < 2:
            return float("inf") if mean_return > 0 else 0.0

        downside_std = statistics.stdev(negative_returns)
        if downside_std <= 0:
            return float("inf") if mean_return > 0 else 0.0

        # Annualize
        annualized_return = mean_return * periods_per_year
        annualized_downside_std = downside_std * (periods_per_year**0.5)

        return annualized_return / annualized_downside_std

    def _calculate_time_in_drawdown(self) -> float:
        """Calculate percentage of time spent in drawdown"""
        if len(self.equity_points) < 2:
            return 0.0

        total_time = self.equity_points[-1].timestamp - self.equity_points[0].timestamp
        if total_time.total_seconds() <= 0:
            return 0.0

        dd_time = timedelta()
        peak_so_far = self.initial_equity
        in_drawdown = False
        dd_start = None

        for point in self.equity_points:
            if point.equity > peak_so_far:
                # New peak - end any drawdown
                if in_drawdown and dd_start:
                    dd_time += point.timestamp - dd_start
                    in_drawdown = False
                peak_so_far = point.equity
            elif not in_drawdown and point.equity < peak_so_far:
                # Start of drawdown
                in_drawdown = True
                dd_start = point.timestamp

        # Handle ongoing drawdown
        if in_drawdown and dd_start:
            dd_time += self.equity_points[-1].timestamp - dd_start

        return dd_time.total_seconds() / total_time.total_seconds()

    def save_to_file(self, filepath: str | None = None) -> None:
        """Save equity cache to JSON file"""
        if filepath is None:
            filepath = self.persistence_file

        if not filepath:
            return

        data = {
            "initial_equity": self.initial_equity,
            "peak_equity": self.peak_equity,
            "peak_timestamp": self.peak_timestamp.isoformat(),
            "equity_points": [point.to_dict() for point in self.equity_points],
        }

        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filepath: str | None = None) -> bool:
        """Load equity cache from JSON file"""
        if filepath is None:
            filepath = self.persistence_file

        if not filepath or not Path(filepath).exists():
            return False

        try:
            with open(filepath) as f:
                data = json.load(f)

            self.initial_equity = data["initial_equity"]
            self.peak_equity = data["peak_equity"]
            self.peak_timestamp = datetime.fromisoformat(data["peak_timestamp"])

            self.equity_points = [
                EquityPoint.from_dict(point_data)
                for point_data in data["equity_points"]
            ]

            # Invalidate cache
            self._cache_dirty = True

            return True

        except Exception as e:
            # Log error but don't raise
            print(f"Failed to load equity cache: {e}")
            return False

    def clear_cache(self) -> None:
        """Clear all equity data"""
        self.equity_points.clear()
        self.peak_equity = self.initial_equity
        self.peak_timestamp = datetime.now()
        self._cache_dirty = True
        self._stats_cache = None
        self.update_count = 0

    def get_equity_curve_data(
        self, start_time: datetime | None = None, end_time: datetime | None = None
    ) -> list[tuple[datetime, float]]:
        """Get equity curve data for plotting"""

        points = self.equity_points

        if start_time:
            points = [p for p in points if p.timestamp >= start_time]

        if end_time:
            points = [p for p in points if p.timestamp <= end_time]

        return [(p.timestamp, p.equity) for p in points]
