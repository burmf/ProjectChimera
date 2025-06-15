"""
Performance Tracking System
Real-time performance metrics and analytics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger


class TradeStatus(Enum):
    """Trade status enumeration"""
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


@dataclass
class Trade:
    """Individual trade record"""
    trade_id: str
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: Optional[float] = None
    size: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    exit_time: Optional[datetime] = None
    pnl: Optional[float] = None
    commission: float = 0.0
    status: TradeStatus = TradeStatus.OPEN
    strategy: str = "default"
    confidence: float = 0.0


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    recovery_factor: float = 0.0
    expectancy: float = 0.0


class PerformanceTracker:
    """
    Professional performance tracking and analytics system
    """
    
    def __init__(self):
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict] = []
        self.daily_returns: List[float] = []
        self.peak_equity = 0.0
        self.current_equity = 0.0
        
        # Strategy-specific tracking
        self.strategy_performance: Dict[str, PerformanceMetrics] = {}
        
        # Real-time metrics
        self.current_metrics = PerformanceMetrics()
        
        logger.info("PerformanceTracker initialized")
    
    def add_trade(self, trade: Trade) -> None:
        """Add a new trade to tracking"""
        self.trades.append(trade)
        
        if trade.status == TradeStatus.CLOSED and trade.pnl is not None:
            self._update_metrics()
            self._update_equity_curve()
        
        logger.debug(f"Trade added: {trade.symbol} {trade.side} P&L: {trade.pnl}")
    
    def close_trade(self, trade_id: str, exit_price: float, exit_time: Optional[datetime] = None) -> bool:
        """Close an open trade"""
        for trade in self.trades:
            if trade.trade_id == trade_id and trade.status == TradeStatus.OPEN:
                trade.exit_price = exit_price
                trade.exit_time = exit_time or datetime.now()
                trade.status = TradeStatus.CLOSED
                
                # Calculate P&L
                if trade.side.lower() == 'long':
                    trade.pnl = (exit_price - trade.entry_price) * trade.size
                else:  # short
                    trade.pnl = (trade.entry_price - exit_price) * trade.size
                
                # Subtract commission
                trade.pnl -= trade.commission
                
                self._update_metrics()
                self._update_equity_curve()
                
                logger.info(f"Trade closed: {trade.symbol} P&L: ${trade.pnl:.2f}")
                return True
        
        return False
    
    def _update_metrics(self) -> None:
        """Update all performance metrics"""
        closed_trades = [t for t in self.trades if t.status == TradeStatus.CLOSED]
        
        if not closed_trades:
            return
        
        # Basic metrics
        self.current_metrics.total_trades = len(closed_trades)
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl < 0]
        
        self.current_metrics.winning_trades = len(winning_trades)
        self.current_metrics.losing_trades = len(losing_trades)
        self.current_metrics.win_rate = len(winning_trades) / len(closed_trades) if closed_trades else 0
        
        # P&L metrics
        pnls = [t.pnl for t in closed_trades]
        self.current_metrics.total_pnl = sum(pnls)
        
        profits = [t.pnl for t in winning_trades]
        losses = [abs(t.pnl) for t in losing_trades]
        
        self.current_metrics.gross_profit = sum(profits) if profits else 0
        self.current_metrics.gross_loss = sum(losses) if losses else 0
        
        if self.current_metrics.gross_loss > 0:
            self.current_metrics.profit_factor = self.current_metrics.gross_profit / self.current_metrics.gross_loss
        
        # Average metrics
        self.current_metrics.average_win = np.mean(profits) if profits else 0
        self.current_metrics.average_loss = np.mean(losses) if losses else 0
        
        # Extremes
        self.current_metrics.largest_win = max(profits) if profits else 0
        self.current_metrics.largest_loss = max(losses) if losses else 0
        
        # Consecutive metrics
        self.current_metrics.max_consecutive_wins = self._calculate_max_consecutive('win')
        self.current_metrics.max_consecutive_losses = self._calculate_max_consecutive('loss')
        
        # Risk-adjusted metrics
        self._calculate_risk_metrics()
        
        # Expectancy
        if self.current_metrics.total_trades > 0:
            self.current_metrics.expectancy = (
                self.current_metrics.win_rate * self.current_metrics.average_win -
                (1 - self.current_metrics.win_rate) * self.current_metrics.average_loss
            )
    
    def _calculate_max_consecutive(self, trade_type: str) -> int:
        """Calculate maximum consecutive wins or losses"""
        closed_trades = [t for t in self.trades if t.status == TradeStatus.CLOSED]
        
        if not closed_trades:
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in closed_trades:
            if trade_type == 'win' and trade.pnl > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            elif trade_type == 'loss' and trade.pnl <= 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_risk_metrics(self) -> None:
        """Calculate risk-adjusted performance metrics"""
        if len(self.daily_returns) < 30:
            return
        
        returns = np.array(self.daily_returns)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        if np.std(returns) > 0:
            self.current_metrics.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_deviation = np.std(downside_returns)
            if downside_deviation > 0:
                self.current_metrics.sortino_ratio = np.mean(returns) / downside_deviation * np.sqrt(252)
        
        # Drawdown analysis
        equity_values = [point['equity'] for point in self.equity_curve]
        if equity_values:
            self._calculate_drawdowns(equity_values)
    
    def _calculate_drawdowns(self, equity_values: List[float]) -> None:
        """Calculate drawdown metrics"""
        equity_series = np.array(equity_values)
        running_max = np.maximum.accumulate(equity_series)
        drawdowns = (equity_series - running_max) / running_max
        
        self.current_metrics.max_drawdown = np.min(drawdowns)
        self.current_metrics.current_drawdown = drawdowns[-1]
        
        # Calmar ratio
        if self.current_metrics.max_drawdown != 0:
            annual_return = (equity_values[-1] / equity_values[0]) ** (252 / len(equity_values)) - 1
            self.current_metrics.calmar_ratio = annual_return / abs(self.current_metrics.max_drawdown)
        
        # Recovery factor
        if self.current_metrics.max_drawdown != 0:
            self.current_metrics.recovery_factor = self.current_metrics.total_pnl / abs(self.current_metrics.max_drawdown)
    
    def _update_equity_curve(self) -> None:
        """Update equity curve with latest trade"""
        current_time = datetime.now()
        
        # Calculate current equity
        closed_pnl = sum(t.pnl for t in self.trades if t.status == TradeStatus.CLOSED)
        self.current_equity = 100000 + closed_pnl  # Starting with $100k
        
        # Update peak equity
        self.peak_equity = max(self.peak_equity, self.current_equity)
        
        # Add to equity curve
        self.equity_curve.append({
            'timestamp': current_time,
            'equity': self.current_equity,
            'drawdown': (self.current_equity - self.peak_equity) / self.peak_equity if self.peak_equity > 0 else 0
        })
        
        # Calculate daily return if new day
        if len(self.equity_curve) > 1:
            last_equity = self.equity_curve[-2]['equity']
            daily_return = (self.current_equity - last_equity) / last_equity
            
            # Only add if it's a new day
            last_date = self.equity_curve[-2]['timestamp'].date()
            current_date = current_time.date()
            
            if last_date != current_date:
                self.daily_returns.append(daily_return)
                
                # Limit history
                if len(self.daily_returns) > 252:  # Keep 1 year
                    self.daily_returns = self.daily_returns[-252:]
    
    def get_win_rate(self) -> float:
        """Get current win rate"""
        return self.current_metrics.win_rate
    
    def get_sharpe_ratio(self) -> float:
        """Get current Sharpe ratio"""
        return self.current_metrics.sharpe_ratio
    
    def get_current_drawdown(self) -> float:
        """Get current drawdown percentage"""
        return self.current_metrics.current_drawdown
    
    def get_strategy_performance(self, strategy: str) -> Optional[PerformanceMetrics]:
        """Get performance metrics for specific strategy"""
        strategy_trades = [t for t in self.trades if t.strategy == strategy and t.status == TradeStatus.CLOSED]
        
        if not strategy_trades:
            return None
        
        # Calculate strategy-specific metrics (similar to main metrics)
        # This would be a simplified version focusing on key metrics
        winning_trades = [t for t in strategy_trades if t.pnl > 0]
        total_pnl = sum(t.pnl for t in strategy_trades)
        win_rate = len(winning_trades) / len(strategy_trades)
        
        return PerformanceMetrics(
            total_trades=len(strategy_trades),
            winning_trades=len(winning_trades),
            win_rate=win_rate,
            total_pnl=total_pnl
        )
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        metrics = self.current_metrics
        
        report = f"""
📊 PERFORMANCE ANALYSIS REPORT
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📈 TRADING STATISTICS:
  Total Trades: {metrics.total_trades}
  Winning Trades: {metrics.winning_trades}
  Losing Trades: {metrics.losing_trades}
  Win Rate: {metrics.win_rate:.2%}
  
💰 PROFIT & LOSS:
  Total P&L: ${metrics.total_pnl:,.2f}
  Gross Profit: ${metrics.gross_profit:,.2f}
  Gross Loss: ${metrics.gross_loss:,.2f}
  Profit Factor: {metrics.profit_factor:.2f}
  
📊 TRADE ANALYSIS:
  Average Win: ${metrics.average_win:,.2f}
  Average Loss: ${metrics.average_loss:,.2f}
  Largest Win: ${metrics.largest_win:,.2f}
  Largest Loss: ${metrics.largest_loss:,.2f}
  Expectancy: ${metrics.expectancy:.2f}
  
🎯 RISK METRICS:
  Sharpe Ratio: {metrics.sharpe_ratio:.3f}
  Sortino Ratio: {metrics.sortino_ratio:.3f}
  Calmar Ratio: {metrics.calmar_ratio:.3f}
  Max Drawdown: {metrics.max_drawdown:.2%}
  Current Drawdown: {metrics.current_drawdown:.2%}
  Recovery Factor: {metrics.recovery_factor:.2f}
  
🔥 CONSECUTIVE PERFORMANCE:
  Max Consecutive Wins: {metrics.max_consecutive_wins}
  Max Consecutive Losses: {metrics.max_consecutive_losses}
  
💡 CURRENT STATUS:
  Current Equity: ${self.current_equity:,.2f}
  Peak Equity: ${self.peak_equity:,.2f}
  Total Return: {(self.current_equity / 100000 - 1):.2%}
{'='*60}
        """
        
        return report
    
    def get_equity_curve_data(self) -> List[Dict]:
        """Get equity curve data for plotting"""
        return self.equity_curve.copy()
    
    def get_trade_history(self, limit: Optional[int] = None) -> List[Trade]:
        """Get trade history (optionally limited)"""
        trades = self.trades.copy()
        if limit:
            trades = trades[-limit:]
        return trades
    
    def reset_statistics(self) -> None:
        """Reset all statistics (keep trades for reference)"""
        self.current_metrics = PerformanceMetrics()
        self.daily_returns = []
        self.peak_equity = self.current_equity
        logger.info("Performance statistics reset")


if __name__ == "__main__":
    # Test the performance tracker
    tracker = PerformanceTracker()
    
    # Add some sample trades
    from uuid import uuid4
    
    # Winning trade
    trade1 = Trade(
        trade_id=str(uuid4()),
        symbol="BTCUSDT",
        side="long",
        entry_price=50000,
        size=0.1,
        strategy="test"
    )
    tracker.add_trade(trade1)
    tracker.close_trade(trade1.trade_id, 51000)  # +$100 profit
    
    # Losing trade
    trade2 = Trade(
        trade_id=str(uuid4()),
        symbol="ETHUSDT",
        side="short",
        entry_price=3000,
        size=1.0,
        strategy="test"
    )
    tracker.add_trade(trade2)
    tracker.close_trade(trade2.trade_id, 3050)  # -$50 loss
    
    print(tracker.generate_performance_report())