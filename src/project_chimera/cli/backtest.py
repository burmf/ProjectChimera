"""
Vectorised NumPy Backtesting CLI - Phase E Implementation
High-performance backtesting engine with multi-strategy support
Target: 100k-row CSV completes <2 min / laptop
"""

import time
import argparse
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Available strategy types"""
    VOL_BREAKOUT = "vol_breakout"
    MINI_MOMENTUM = "mini_momo"
    LOB_REVERT = "lob_revert"
    FUNDING_ALPHA = "funding_alpha"
    BASIS_ARB = "basis_arb"
    CME_GAP = "cme_gap"
    CASCADE_PRED = "cascade_pred"
    ROUND_REV = "round_rev"
    NETFLOW_FLT = "netflow_flt"
    VOL_TERM_FLIP = "vol_term_flip"
    SOCIAL_JUMP = "social_jump"
    HALVING_DRIFT = "halving_drift"
    ALT_BETA_ROT = "alt_beta_rot"
    WEEKEND_EFFECT = "wknd_eff"
    STOP_REV = "stop_rev"
    SESSION_BRK = "session_brk"
    LAT_ARB = "lat_arb"


@dataclass
class MarketData:
    """Vectorized market data container"""
    timestamps: np.ndarray    # Unix timestamps
    opens: np.ndarray        # Open prices
    highs: np.ndarray        # High prices
    lows: np.ndarray         # Low prices
    closes: np.ndarray       # Close prices
    volumes: np.ndarray      # Volumes
    
    @property
    def length(self) -> int:
        return len(self.closes)
    
    def validate(self) -> bool:
        """Validate data integrity"""
        arrays = [self.timestamps, self.opens, self.highs, self.lows, self.closes, self.volumes]
        lengths = [len(arr) for arr in arrays]
        
        # All arrays must have same length
        if len(set(lengths)) != 1:
            return False
        
        # Basic price validation
        if np.any(self.highs < self.lows) or np.any(self.closes <= 0):
            return False
            
        return True


@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    initial_capital: float = 100_000.0
    commission_rate: float = 0.001      # 0.1% per trade
    spread_cost: float = 0.0005         # 0.05% spread
    max_position_size: float = 0.2      # 20% max position
    risk_free_rate: float = 0.02        # 2% annual risk-free rate
    slippage: float = 0.0001            # 0.01% slippage


@dataclass
class BacktestResult:
    """Comprehensive backtest results"""
    # Performance metrics
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Trade statistics
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Risk metrics
    volatility: float
    var_95: float
    expected_shortfall: float
    
    # Timing
    start_date: datetime
    end_date: datetime
    duration_days: int
    
    # Data
    equity_curve: np.ndarray
    trade_log: List[Dict]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'performance': {
                'total_return': self.total_return,
                'annual_return': self.annual_return,
                'sharpe_ratio': self.sharpe_ratio,
                'sortino_ratio': self.sortino_ratio,
                'max_drawdown': self.max_drawdown,
                'calmar_ratio': self.calmar_ratio,
            },
            'trades': {
                'total_trades': self.total_trades,
                'win_rate': self.win_rate,
                'avg_win': self.avg_win,
                'avg_loss': self.avg_loss,
                'profit_factor': self.profit_factor,
            },
            'risk': {
                'volatility': self.volatility,
                'var_95': self.var_95,
                'expected_shortfall': self.expected_shortfall,
            },
            'period': {
                'start_date': self.start_date.isoformat(),
                'end_date': self.end_date.isoformat(),
                'duration_days': self.duration_days,
            }
        }


class VectorizedBacktester:
    """
    High-performance vectorized backtesting engine
    
    Features:
    - Pure NumPy operations for maximum speed
    - Vectorized signal generation and execution
    - Memory-efficient processing
    - Comprehensive performance metrics
    """
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self.strategies = self._initialize_strategies()
    
    def _initialize_strategies(self) -> Dict[StrategyType, callable]:
        """Initialize strategy implementations"""
        return {
            StrategyType.VOL_BREAKOUT: self._vol_breakout_strategy,
            StrategyType.MINI_MOMENTUM: self._mini_momentum_strategy,
            StrategyType.LOB_REVERT: self._lob_revert_strategy,
            StrategyType.FUNDING_ALPHA: self._funding_alpha_strategy,
            StrategyType.BASIS_ARB: self._basis_arb_strategy,
            StrategyType.CME_GAP: self._cme_gap_strategy,
            StrategyType.CASCADE_PRED: self._cascade_pred_strategy,
            StrategyType.ROUND_REV: self._round_rev_strategy,
            StrategyType.NETFLOW_FLT: self._netflow_flt_strategy,
            StrategyType.VOL_TERM_FLIP: self._vol_term_flip_strategy,
            StrategyType.SOCIAL_JUMP: self._social_jump_strategy,
            StrategyType.HALVING_DRIFT: self._halving_drift_strategy,
            StrategyType.ALT_BETA_ROT: self._alt_beta_rot_strategy,
            StrategyType.WEEKEND_EFFECT: self._weekend_effect_strategy,
            StrategyType.STOP_REV: self._stop_rev_strategy,
            StrategyType.SESSION_BRK: self._session_brk_strategy,
            StrategyType.LAT_ARB: self._lat_arb_strategy,
        }
    
    def load_csv_data(self, csv_path: str) -> MarketData:
        """Load market data from CSV file efficiently"""
        logger.info(f"Loading data from {csv_path}...")
        start_time = time.time()
        
        # Read CSV efficiently
        data = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        
        # Convert to numpy arrays
        n_rows = len(data)
        
        timestamps = np.zeros(n_rows, dtype=np.float64)
        opens = np.zeros(n_rows, dtype=np.float64)
        highs = np.zeros(n_rows, dtype=np.float64)
        lows = np.zeros(n_rows, dtype=np.float64)
        closes = np.zeros(n_rows, dtype=np.float64)
        volumes = np.zeros(n_rows, dtype=np.float64)
        
        for i, row in enumerate(data):
            # Handle different timestamp formats
            timestamp_str = row.get('timestamp', row.get('time', row.get('date', '')))
            if timestamp_str:
                try:
                    dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    timestamps[i] = dt.timestamp()
                except:
                    timestamps[i] = float(timestamp_str) if timestamp_str.isdigit() else i
            else:
                timestamps[i] = i
            
            opens[i] = float(row.get('open', row.get('Open', 0)))
            highs[i] = float(row.get('high', row.get('High', 0)))
            lows[i] = float(row.get('low', row.get('Low', 0)))
            closes[i] = float(row.get('close', row.get('Close', 0)))
            volumes[i] = float(row.get('volume', row.get('Volume', 1000)))
        
        market_data = MarketData(timestamps, opens, highs, lows, closes, volumes)
        
        if not market_data.validate():
            raise ValueError("Invalid market data")
        
        load_time = time.time() - start_time
        logger.info(f"Loaded {n_rows:,} rows in {load_time:.2f}s")
        
        return market_data
    
    def backtest_strategy(self, data: MarketData, strategy: StrategyType) -> BacktestResult:
        """Backtest a single strategy"""
        logger.info(f"Backtesting {strategy.value}...")
        start_time = time.time()
        
        # Generate signals
        signals = self.strategies[strategy](data)
        
        # Execute backtest
        result = self._execute_vectorized_backtest(data, signals)
        
        backtest_time = time.time() - start_time
        logger.info(f"Completed {strategy.value} in {backtest_time:.2f}s")
        
        return result
    
    def backtest_all_strategies(self, data: MarketData) -> Dict[StrategyType, BacktestResult]:
        """Backtest all available strategies"""
        logger.info("Running multi-strategy backtest...")
        start_time = time.time()
        
        results = {}
        for strategy_type in self.strategies.keys():
            try:
                results[strategy_type] = self.backtest_strategy(data, strategy_type)
            except Exception as e:
                logger.error(f"Strategy {strategy_type.value} failed: {e}")
                continue
        
        total_time = time.time() - start_time
        logger.info(f"Completed {len(results)} strategies in {total_time:.2f}s")
        
        return results
    
    def _execute_vectorized_backtest(self, data: MarketData, signals: np.ndarray) -> BacktestResult:
        """Execute vectorized backtest with signals"""
        n_points = data.length
        
        # Initialize arrays
        position = np.zeros(n_points)  # Position size at each point
        equity = np.full(n_points, self.config.initial_capital)
        cash = np.full(n_points, self.config.initial_capital)
        
        # Trade tracking
        trades = []
        in_position = False
        entry_idx = 0
        entry_price = 0.0
        
        # Process signals
        for i in range(1, n_points):
            signal = signals[i]
            price = data.closes[i]
            
            # Entry logic
            if signal != 0 and not in_position:
                # Enter position
                position_size = min(
                    self.config.max_position_size * equity[i-1],
                    equity[i-1] * 0.95  # Max 95% of equity
                )
                
                # Apply transaction costs
                transaction_cost = position_size * (self.config.commission_rate + self.config.spread_cost + self.config.slippage)
                
                if position_size > transaction_cost:
                    position[i] = signal * (position_size - transaction_cost) / price
                    cash[i] = cash[i-1] - position_size
                    equity[i] = cash[i] + position[i] * price
                    
                    in_position = True
                    entry_idx = i
                    entry_price = price
                else:
                    # Insufficient funds
                    position[i] = position[i-1]
                    cash[i] = cash[i-1]
                    equity[i] = equity[i-1]
            
            # Exit logic (simplified: exit after 10 periods or opposite signal)
            elif in_position and (i - entry_idx >= 10 or signal * position[i-1] < 0):
                # Exit position
                exit_value = abs(position[i-1]) * price
                exit_cost = exit_value * (self.config.commission_rate + self.config.spread_cost + self.config.slippage)
                
                pnl = (price - entry_price) * position[i-1] - exit_cost
                
                # Record trade
                trades.append({
                    'entry_time': data.timestamps[entry_idx],
                    'exit_time': data.timestamps[i],
                    'entry_price': entry_price,
                    'exit_price': price,
                    'position_size': position[i-1],
                    'pnl': pnl,
                    'return_pct': pnl / (abs(position[i-1]) * entry_price) if position[i-1] != 0 else 0
                })
                
                cash[i] = cash[i-1] + exit_value - exit_cost
                position[i] = 0
                equity[i] = cash[i]
                
                in_position = False
            
            else:
                # Hold position
                position[i] = position[i-1]
                cash[i] = cash[i-1]
                if position[i] != 0:
                    equity[i] = cash[i] + position[i] * price
                else:
                    equity[i] = cash[i]
        
        # Calculate metrics
        return self._calculate_metrics(data, equity, trades)
    
    def _calculate_metrics(self, data: MarketData, equity: np.ndarray, trades: List[Dict]) -> BacktestResult:
        """Calculate comprehensive performance metrics"""
        if len(trades) == 0:
            # No trades case
            return BacktestResult(
                total_return=0.0, annual_return=0.0, sharpe_ratio=0.0, sortino_ratio=0.0,
                max_drawdown=0.0, calmar_ratio=0.0, total_trades=0, win_rate=0.0,
                avg_win=0.0, avg_loss=0.0, profit_factor=0.0, volatility=0.0,
                var_95=0.0, expected_shortfall=0.0,
                start_date=datetime.fromtimestamp(data.timestamps[0]),
                end_date=datetime.fromtimestamp(data.timestamps[-1]),
                duration_days=int((data.timestamps[-1] - data.timestamps[0]) / 86400),
                equity_curve=equity, trade_log=trades
            )
        
        # Basic returns
        initial_equity = equity[0]
        final_equity = equity[-1]
        total_return = (final_equity - initial_equity) / initial_equity
        
        # Time-based metrics
        start_date = datetime.fromtimestamp(data.timestamps[0])
        end_date = datetime.fromtimestamp(data.timestamps[-1])
        duration_days = (data.timestamps[-1] - data.timestamps[0]) / 86400
        duration_years = duration_days / 365.25
        
        annual_return = (final_equity / initial_equity) ** (1 / max(duration_years, 1/365)) - 1 if duration_years > 0 else 0
        
        # Equity curve analysis
        equity_returns = np.diff(equity) / equity[:-1]
        equity_returns = equity_returns[np.isfinite(equity_returns)]  # Remove inf/nan
        
        # Risk metrics
        volatility = np.std(equity_returns) * np.sqrt(252)  # Annualized
        
        if len(equity_returns) > 0 and volatility > 0:
            sharpe_ratio = (np.mean(equity_returns) * 252 - self.config.risk_free_rate) / volatility
            
            # Sortino ratio (downside deviation)
            negative_returns = equity_returns[equity_returns < 0]
            if len(negative_returns) > 0:
                downside_vol = np.std(negative_returns) * np.sqrt(252)
                sortino_ratio = (annual_return - self.config.risk_free_rate) / downside_vol
            else:
                sortino_ratio = sharpe_ratio
        else:
            sharpe_ratio = sortino_ratio = 0.0
        
        # Drawdown analysis
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Trade statistics
        trade_returns = [trade['return_pct'] for trade in trades]
        winning_trades = [r for r in trade_returns if r > 0]
        losing_trades = [r for r in trade_returns if r < 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if avg_loss != 0 and losing_trades else float('inf')
        
        # Risk metrics
        if len(equity_returns) > 0:
            var_95 = np.percentile(equity_returns, 5)  # 5th percentile
            expected_shortfall = np.mean(equity_returns[equity_returns <= var_95]) if np.any(equity_returns <= var_95) else var_95
        else:
            var_95 = expected_shortfall = 0.0
        
        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            total_trades=len(trades),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            volatility=volatility,
            var_95=var_95,
            expected_shortfall=expected_shortfall,
            start_date=start_date,
            end_date=end_date,
            duration_days=int(duration_days),
            equity_curve=equity,
            trade_log=trades
        )
    
    # Strategy Implementations (Vectorized)
    
    def _vol_breakout_strategy(self, data: MarketData) -> np.ndarray:
        """BB-squeeze width<threshold then ±2% break"""
        # Bollinger Bands
        window = 20
        closes = data.closes
        
        sma = np.convolve(closes, np.ones(window)/window, mode='valid')
        std = np.array([np.std(closes[i:i+window]) for i in range(len(closes)-window+1)])
        
        # Pad to match original length
        sma = np.concatenate([np.full(window-1, sma[0]), sma])
        std = np.concatenate([np.full(window-1, std[0]), std])
        
        bb_width = 2 * std / sma
        threshold = np.percentile(bb_width[window:], 20)  # 20th percentile
        
        signals = np.zeros(len(closes))
        
        # Generate signals
        for i in range(window, len(closes)):
            if bb_width[i] < threshold:  # Squeeze
                price_change = (closes[i] - closes[i-1]) / closes[i-1]
                if price_change > 0.02:  # +2% break
                    signals[i] = 1
                elif price_change < -0.02:  # -2% break
                    signals[i] = -1
        
        return signals
    
    def _mini_momentum_strategy(self, data: MarketData) -> np.ndarray:
        """7-bar z-score > ±1 → trend follow"""
        window = 7
        closes = data.closes
        
        signals = np.zeros(len(closes))
        
        for i in range(window, len(closes)):
            recent_prices = closes[i-window:i]
            z_score = (closes[i] - np.mean(recent_prices)) / np.std(recent_prices)
            
            if z_score > 1:
                signals[i] = 1
            elif z_score < -1:
                signals[i] = -1
        
        return signals
    
    def _lob_revert_strategy(self, data: MarketData) -> np.ndarray:
        """Order-flow RSI 30/70 mean-reversion (simplified as price RSI)"""
        window = 14
        closes = data.closes
        
        # Calculate RSI
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        signals = np.zeros(len(closes))
        
        for i in range(window, len(closes)):
            avg_gain = np.mean(gains[i-window:i])
            avg_loss = np.mean(losses[i-window:i])
            
            if avg_loss == 0:
                rsi = 100
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            if rsi < 30:  # Oversold
                signals[i] = 1
            elif rsi > 70:  # Overbought
                signals[i] = -1
        
        return signals
    
    def _funding_alpha_strategy(self, data: MarketData) -> np.ndarray:
        """Mock funding contrarian (simplified momentum reversal)"""
        signals = np.zeros(len(data.closes))
        
        # Simple momentum reversal
        for i in range(10, len(data.closes)):
            momentum = (data.closes[i] - data.closes[i-10]) / data.closes[i-10]
            if momentum > 0.05:  # Strong up move, contrarian short
                signals[i] = -1
            elif momentum < -0.05:  # Strong down move, contrarian long
                signals[i] = 1
        
        return signals
    
    def _basis_arb_strategy(self, data: MarketData) -> np.ndarray:
        """Mock basis arbitrage (simplified volatility trading)"""
        signals = np.zeros(len(data.closes))
        
        # Trade high volatility periods
        for i in range(20, len(data.closes)):
            vol = np.std(data.closes[i-20:i])
            high_vol_threshold = np.percentile(data.closes[20:i], 80)
            
            if vol > high_vol_threshold:
                signals[i] = np.random.choice([-1, 1])  # Random direction in high vol
        
        return signals
    
    def _cme_gap_strategy(self, data: MarketData) -> np.ndarray:
        """CME gap fill (simplified gap detection)"""
        signals = np.zeros(len(data.closes))
        
        # Detect gaps and fade them
        for i in range(1, len(data.closes)):
            gap = (data.opens[i] - data.closes[i-1]) / data.closes[i-1]
            if abs(gap) > 0.01:  # 1% gap
                signals[i] = -np.sign(gap)  # Fade the gap
        
        return signals
    
    def _cascade_pred_strategy(self, data: MarketData) -> np.ndarray:
        """Mock liquidation cascade prediction"""
        signals = np.zeros(len(data.closes))
        
        # Predict cascades from rapid moves
        for i in range(5, len(data.closes)):
            rapid_move = (data.closes[i] - data.closes[i-5]) / data.closes[i-5]
            if rapid_move < -0.03:  # -3% in 5 periods
                signals[i] = 1  # Buy the dip
        
        return signals
    
    def _round_rev_strategy(self, data: MarketData) -> np.ndarray:
        """Round price level mean reversion"""
        signals = np.zeros(len(data.closes))
        
        for i in range(len(data.closes)):
            price = data.closes[i]
            # Find nearest round number (1000s, 10000s)
            round_levels = [round(price, -3), round(price, -4)]  # 1k, 10k levels
            
            for level in round_levels:
                if abs(price - level) / level < 0.005:  # Within 0.5% of round level
                    if price > level:
                        signals[i] = -1  # Fade rally at resistance
                    else:
                        signals[i] = 1   # Buy support
                    break
        
        return signals
    
    def _netflow_flt_strategy(self, data: MarketData) -> np.ndarray:
        """Mock stablecoin netflow filter"""
        signals = np.zeros(len(data.closes))
        
        # Mock netflow based on volume patterns
        for i in range(10, len(data.closes)):
            vol_ratio = data.volumes[i] / np.mean(data.volumes[i-10:i])
            if vol_ratio > 2:  # High volume = netflow
                price_change = (data.closes[i] - data.closes[i-1]) / data.closes[i-1]
                signals[i] = np.sign(price_change)
        
        return signals
    
    def _vol_term_flip_strategy(self, data: MarketData) -> np.ndarray:
        """Mock volatility term structure (reduce size in high vol)"""
        signals = np.zeros(len(data.closes))
        
        # Reduce exposure in high volatility
        for i in range(20, len(data.closes)):
            vol = np.std(data.closes[i-20:i])
            vol_threshold = np.percentile([np.std(data.closes[j-20:j]) for j in range(20, i)], 70)
            
            if vol < vol_threshold:  # Low vol regime
                momentum = (data.closes[i] - data.closes[i-5]) / data.closes[i-5]
                signals[i] = np.sign(momentum) * 0.5  # Reduced size
        
        return signals
    
    def _social_jump_strategy(self, data: MarketData) -> np.ndarray:
        """Mock social media spike (volume-based)"""
        signals = np.zeros(len(data.closes))
        
        # High volume spikes as proxy for social attention
        for i in range(10, len(data.closes)):
            vol_z = (data.volumes[i] - np.mean(data.volumes[i-10:i])) / np.std(data.volumes[i-10:i])
            if vol_z > 2:  # Volume spike
                price_move = (data.closes[i] - data.closes[i-1]) / data.closes[i-1]
                signals[i] = np.sign(price_move)
        
        return signals
    
    def _halving_drift_strategy(self, data: MarketData) -> np.ndarray:
        """Mock halving event drift (long bias)"""
        signals = np.zeros(len(data.closes))
        
        # Simple long bias strategy
        for i in range(50, len(data.closes)):
            sma_short = np.mean(data.closes[i-10:i])
            sma_long = np.mean(data.closes[i-50:i])
            if sma_short > sma_long:
                signals[i] = 1  # Long bias
        
        return signals
    
    def _alt_beta_rot_strategy(self, data: MarketData) -> np.ndarray:
        """Mock beta rotation (momentum with delay)"""
        signals = np.zeros(len(data.closes))
        
        # Delayed momentum
        for i in range(20, len(data.closes)):
            momentum = (data.closes[i-5] - data.closes[i-20]) / data.closes[i-20]  # Lagged momentum
            if momentum > 0.02:
                signals[i] = 1
            elif momentum < -0.02:
                signals[i] = -1
        
        return signals
    
    def _weekend_effect_strategy(self, data: MarketData) -> np.ndarray:
        """Mock weekend effect (time-based)"""
        signals = np.zeros(len(data.closes))
        
        # Simple time-based pattern (every 7th bar as proxy for weekly)
        for i in range(0, len(data.closes), 7):
            if i < len(signals):
                signals[i] = 1  # Buy at "week start"
        
        return signals
    
    def _stop_rev_strategy(self, data: MarketData) -> np.ndarray:
        """Stop hunt reversal (-3% in 5m + vol spike)"""
        signals = np.zeros(len(data.closes))
        
        for i in range(5, len(data.closes)):
            price_drop = (data.closes[i] - data.closes[i-5]) / data.closes[i-5]
            vol_spike = data.volumes[i] > np.mean(data.volumes[max(0, i-10):i]) * 2
            
            if price_drop < -0.03 and vol_spike:
                signals[i] = 1  # Long rebound
        
        return signals
    
    def _session_brk_strategy(self, data: MarketData) -> np.ndarray:
        """Mock session transition breakout"""
        signals = np.zeros(len(data.closes))
        
        # Breakout from recent range
        for i in range(20, len(data.closes)):
            recent_high = np.max(data.highs[i-20:i])
            recent_low = np.min(data.lows[i-20:i])
            
            if data.closes[i] > recent_high:
                signals[i] = 1
            elif data.closes[i] < recent_low:
                signals[i] = -1
        
        return signals
    
    def _lat_arb_strategy(self, data: MarketData) -> np.ndarray:
        """Mock latency arbitrage (very short-term)"""
        signals = np.zeros(len(data.closes))
        
        # Ultra-short momentum
        for i in range(2, len(data.closes)):
            micro_momentum = (data.closes[i] - data.closes[i-2]) / data.closes[i-2]
            if abs(micro_momentum) > 0.001:  # 0.1% move
                signals[i] = np.sign(micro_momentum)
        
        return signals


def print_performance_summary(results: Dict[StrategyType, BacktestResult]) -> None:
    """Print formatted performance summary"""
    print("\n" + "="*100)
    print("MULTI-STRATEGY BACKTEST RESULTS")
    print("="*100)
    
    # Sort by Sharpe ratio
    sorted_results = sorted(results.items(), key=lambda x: x[1].sharpe_ratio, reverse=True)
    
    print(f"{'Strategy':<20} {'Return':<10} {'Sharpe':<8} {'MaxDD':<8} {'Trades':<8} {'Win%':<8} {'Calmar':<8}")
    print("-" * 100)
    
    for strategy_type, result in sorted_results:
        print(f"{strategy_type.value:<20} {result.total_return:>8.1%} {result.sharpe_ratio:>7.2f} "
              f"{result.max_drawdown:>7.1%} {result.total_trades:>7d} {result.win_rate:>7.1%} {result.calmar_ratio:>7.2f}")
    
    print("-" * 100)
    
    # Summary statistics
    total_returns = [r.total_return for r in results.values()]
    sharpe_ratios = [r.sharpe_ratio for r in results.values() if not np.isnan(r.sharpe_ratio)]
    
    print(f"Summary: {len(results)} strategies | Avg Return: {np.mean(total_returns):.1%} | "
          f"Avg Sharpe: {np.mean(sharpe_ratios):.2f} | Best: {max(total_returns):.1%}")


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description="Vectorised NumPy Backtesting Engine")
    parser.add_argument('--csv', required=True, help='Path to CSV data file')
    parser.add_argument('--strats', default='all', help='Strategies to test (all, vol_breakout, mini_momo, etc.)')
    parser.add_argument('--initial-capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--output', help='Output JSON file for results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize backtester
        config = BacktestConfig(initial_capital=args.initial_capital)
        backtester = VectorizedBacktester(config)
        
        # Load data
        start_time = time.time()
        data = backtester.load_csv_data(args.csv)
        load_time = time.time() - start_time
        
        print(f"📊 Loaded {data.length:,} data points in {load_time:.2f}s")
        
        # Run backtests
        if args.strats == 'all':
            results = backtester.backtest_all_strategies(data)
        else:
            # Single strategy
            try:
                strategy = StrategyType(args.strats)
                result = backtester.backtest_strategy(data, strategy)
                results = {strategy: result}
            except ValueError:
                print(f"Unknown strategy: {args.strats}")
                print(f"Available: {[s.value for s in StrategyType]}")
                return 1
        
        total_time = time.time() - start_time
        
        # Display results
        print_performance_summary(results)
        print(f"\n⚡ Total execution time: {total_time:.2f}s")
        print(f"📈 Processing rate: {data.length / total_time:,.0f} rows/second")
        
        # Save results
        if args.output:
            output_data = {
                'config': {
                    'csv_file': args.csv,
                    'strategies': args.strats,
                    'initial_capital': args.initial_capital,
                    'total_time': total_time,
                    'data_points': data.length
                },
                'results': {strategy.value: result.to_dict() for strategy, result in results.items()}
            }
            
            with open(args.output, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            print(f"💾 Results saved to {args.output}")
        
        # Performance validation
        if data.length >= 100000:
            minutes = total_time / 60
            if minutes <= 2.0:
                print(f"✅ Performance target met: {minutes:.1f} min < 2.0 min for 100k+ rows")
            else:
                print(f"⚠️  Performance target missed: {minutes:.1f} min > 2.0 min for 100k+ rows")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())