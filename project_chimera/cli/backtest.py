"""
Professional Backtesting CLI Tool
Command-line interface for comprehensive strategy backtesting with metrics
"""

import sys
import asyncio
import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

import click
from loguru import logger
from tabulate import tabulate

from ..config import get_settings, Settings
from ..core.container import get_container, DIContainer
from ..core.risk_manager import ProfessionalRiskManager
from ..utils.logging import get_logger, StructuredLogger, EventType


class BacktestEngine:
    """
    Professional backtesting engine with comprehensive metrics
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.risk_manager = ProfessionalRiskManager(self.settings)
        self.structured_logger = get_logger()
        
        # Backtest state
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.portfolio_value = 100000.0  # Starting capital
        self.current_positions: Dict[str, Dict] = {}
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load price data from CSV file"""
        try:
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path)
            else:
                raise ValueError("Unsupported file format. Use CSV or JSON.")
            
            # Standardize column names
            column_mapping = {
                'timestamp': 'timestamp',
                'time': 'timestamp',
                'date': 'timestamp',
                'open': 'open',
                'high': 'high', 
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }
            
            # Rename columns to standard format
            df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
            
            # Ensure required columns exist
            required_columns = ['timestamp', 'open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Parse timestamp
            if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Add volume if missing
            if 'volume' not in df.columns:
                df['volume'] = 1000000.0  # Default volume
            
            logger.info(f"Loaded {len(df)} data points from {data_path}")
            logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {e}")
            raise
    
    def simple_ma_crossover_strategy(self, df: pd.DataFrame, fast_ma: int = 10, slow_ma: int = 30) -> List[Dict]:
        """
        Simple moving average crossover strategy
        Returns list of signals with entry/exit points
        """
        # Calculate moving averages
        df['fast_ma'] = df['close'].rolling(window=fast_ma).mean()
        df['slow_ma'] = df['close'].rolling(window=slow_ma).mean()
        
        # Generate signals
        df['signal'] = 0
        df.loc[df['fast_ma'] > df['slow_ma'], 'signal'] = 1  # Long signal
        df.loc[df['fast_ma'] < df['slow_ma'], 'signal'] = -1  # Short signal
        
        # Find signal changes
        df['signal_change'] = df['signal'].diff()
        
        signals = []
        position = 0
        entry_price = 0
        entry_time = None
        
        for idx, row in df.iterrows():
            if row['signal_change'] == 1 and position != 1:  # Enter long
                if position == -1:  # Close short first
                    pnl = (entry_price - row['close']) / entry_price
                    signals.append({
                        'type': 'exit',
                        'side': 'short',
                        'timestamp': row['timestamp'],
                        'price': row['close'],
                        'pnl': pnl,
                        'entry_price': entry_price,
                        'entry_time': entry_time
                    })
                
                # Enter long
                position = 1
                entry_price = row['close']
                entry_time = row['timestamp']
                signals.append({
                    'type': 'entry',
                    'side': 'long',
                    'timestamp': row['timestamp'],
                    'price': row['close'],
                    'entry_price': entry_price,
                    'entry_time': entry_time
                })
                
            elif row['signal_change'] == -1 and position != -1:  # Enter short
                if position == 1:  # Close long first
                    pnl = (row['close'] - entry_price) / entry_price
                    signals.append({
                        'type': 'exit',
                        'side': 'long',
                        'timestamp': row['timestamp'],
                        'price': row['close'],
                        'pnl': pnl,
                        'entry_price': entry_price,
                        'entry_time': entry_time
                    })
                
                # Enter short
                position = -1
                entry_price = row['close']
                entry_time = row['timestamp']
                signals.append({
                    'type': 'entry',
                    'side': 'short',
                    'timestamp': row['timestamp'],
                    'price': row['close'],
                    'entry_price': entry_price,
                    'entry_time': entry_time
                })
        
        logger.info(f"Generated {len(signals)} signals for MA({fast_ma}, {slow_ma}) strategy")
        return signals
    
    def kelly_criterion_strategy(self, df: pd.DataFrame) -> List[Dict]:
        """
        Enhanced strategy using Kelly criterion for position sizing
        """
        # Simple momentum + mean reversion signals
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['momentum'] = df['close'] / df['close'].shift(10) - 1
        df['mean_reversion'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        
        signals = []
        lookback_window = 50
        
        for idx in range(lookback_window, len(df)):
            current_row = df.iloc[idx]
            historical_data = df.iloc[idx-lookback_window:idx]
            
            # Calculate historical win rate and avg returns
            momentum_signals = historical_data['momentum'] > 0.02
            returns_when_signal = historical_data.loc[momentum_signals, 'returns'].shift(-1).dropna()
            
            if len(returns_when_signal) > 10:
                win_rate = (returns_when_signal > 0).mean()
                avg_win = returns_when_signal[returns_when_signal > 0].mean()
                avg_loss = returns_when_signal[returns_when_signal < 0].mean()
                
                # Generate signal if Kelly criterion suggests positive position
                if current_row['momentum'] > 0.02 and not pd.isna(avg_loss) and avg_loss != 0:
                    kelly_size = self.risk_manager.calculate_kelly_position_size(
                        'BACKTEST', 
                        expected_return=avg_win if not pd.isna(avg_win) else 0.01,
                        win_probability=win_rate,
                        loss_probability=1-win_rate,
                        avg_win=avg_win if not pd.isna(avg_win) else 0.01,
                        avg_loss=avg_loss
                    )
                    
                    if kelly_size > 1000:  # Minimum position size
                        signals.append({
                            'type': 'entry',
                            'side': 'long',
                            'timestamp': current_row['timestamp'],
                            'price': current_row['close'],
                            'size': kelly_size,
                            'kelly_fraction': kelly_size / 100000,  # Assuming $100k portfolio
                            'win_rate': win_rate,
                            'avg_win': avg_win,
                            'avg_loss': avg_loss
                        })
        
        logger.info(f"Generated {len(signals)} Kelly-based signals")
        return signals
    
    def run_backtest(self, df: pd.DataFrame, signals: List[Dict], initial_capital: float = 100000.0) -> Dict[str, Any]:
        """
        Execute backtest with given signals and return comprehensive metrics
        """
        self.portfolio_value = initial_capital
        self.trades = []
        self.equity_curve = []
        
        # Track positions
        open_positions = []
        daily_returns = []
        
        # Transaction costs
        commission_rate = 0.001  # 0.1% per trade
        spread_cost = 0.0005    # 0.05% spread cost
        
        # Process each signal
        for signal in signals:
            timestamp = signal['timestamp']
            price = signal['price']
            signal_type = signal['type']
            
            if signal_type == 'entry':
                # Calculate position size
                position_size = signal.get('size', self.portfolio_value * 0.1)  # Default 10%
                position_size = min(position_size, self.portfolio_value * 0.9)  # Max 90% of portfolio
                
                # Calculate shares/contracts
                shares = position_size / price
                
                # Apply transaction costs
                total_cost = position_size * (1 + commission_rate + spread_cost)
                
                if total_cost <= self.portfolio_value:
                    # Enter position
                    position = {
                        'entry_time': timestamp,
                        'entry_price': price,
                        'side': signal['side'],
                        'shares': shares,
                        'size': position_size,
                        'cost': total_cost
                    }
                    
                    open_positions.append(position)
                    self.portfolio_value -= total_cost
                    
                    logger.debug(f"Entered {signal['side']} position: {shares:.4f} shares at ${price:.2f}")
            
            elif signal_type == 'exit':
                # Close matching positions
                side = signal['side']
                exit_price = price
                
                positions_to_close = [p for p in open_positions if p['side'] == side]
                
                for position in positions_to_close:
                    # Calculate P&L
                    if side == 'long':
                        pnl_per_share = exit_price - position['entry_price']
                    else:  # short
                        pnl_per_share = position['entry_price'] - exit_price
                    
                    gross_pnl = pnl_per_share * position['shares']
                    
                    # Apply exit costs
                    exit_value = position['shares'] * exit_price
                    exit_cost = exit_value * (commission_rate + spread_cost)
                    net_pnl = gross_pnl - exit_cost
                    
                    # Update portfolio
                    self.portfolio_value += exit_value - exit_cost
                    
                    # Record trade
                    trade_duration = (timestamp - position['entry_time']).total_seconds() / 3600  # hours
                    
                    trade = {
                        'entry_time': position['entry_time'],
                        'exit_time': timestamp,
                        'side': side,
                        'entry_price': position['entry_price'],
                        'exit_price': exit_price,
                        'shares': position['shares'],
                        'gross_pnl': gross_pnl,
                        'net_pnl': net_pnl,
                        'return_pct': net_pnl / position['size'],
                        'duration_hours': trade_duration,
                        'cost': position['cost'] + exit_cost
                    }
                    
                    self.trades.append(trade)
                    
                    # Update statistics
                    self.total_trades += 1
                    if net_pnl > 0:
                        self.winning_trades += 1
                    else:
                        self.losing_trades += 1
                    
                    self.total_pnl += net_pnl
                    
                    logger.debug(f"Closed {side} position: P&L ${net_pnl:.2f}")
                
                # Remove closed positions
                open_positions = [p for p in open_positions if p['side'] != side]
            
            # Record equity point
            current_equity = self.portfolio_value + sum(
                pos['shares'] * price for pos in open_positions
            )
            
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': current_equity,
                'cash': self.portfolio_value,
                'positions_value': current_equity - self.portfolio_value
            })
        
        # Close any remaining positions at final price
        if open_positions and len(df) > 0:
            final_price = df['close'].iloc[-1]
            final_time = df['timestamp'].iloc[-1]
            
            for position in open_positions:
                if position['side'] == 'long':
                    pnl_per_share = final_price - position['entry_price']
                else:
                    pnl_per_share = position['entry_price'] - final_price
                
                gross_pnl = pnl_per_share * position['shares']
                exit_value = position['shares'] * final_price
                exit_cost = exit_value * (commission_rate + spread_cost)
                net_pnl = gross_pnl - exit_cost
                
                self.portfolio_value += exit_value - exit_cost
                
                trade = {
                    'entry_time': position['entry_time'],
                    'exit_time': final_time,
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'exit_price': final_price,
                    'shares': position['shares'],
                    'gross_pnl': gross_pnl,
                    'net_pnl': net_pnl,
                    'return_pct': net_pnl / position['size'],
                    'duration_hours': (final_time - position['entry_time']).total_seconds() / 3600,
                    'cost': position['cost'] + exit_cost
                }
                
                self.trades.append(trade)
                self.total_trades += 1
                if net_pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                self.total_pnl += net_pnl
        
        # Calculate final metrics
        return self.calculate_performance_metrics(initial_capital)
    
    def calculate_performance_metrics(self, initial_capital: float) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {'error': 'No trades executed'}
        
        # Convert to DataFrame for easier analysis
        trades_df = pd.DataFrame(self.trades)
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Basic metrics
        final_equity = self.portfolio_value
        total_return = (final_equity - initial_capital) / initial_capital
        
        # Trade statistics
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean() if self.winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['net_pnl'] < 0]['net_pnl'].mean() if self.losing_trades > 0 else 0
        
        # Risk metrics
        returns = trades_df['return_pct'].values
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            sortino_ratio = np.mean(returns) / np.std(returns[returns < 0]) * np.sqrt(252) if len(returns[returns < 0]) > 0 else 0
        else:
            sharpe_ratio = sortino_ratio = 0
        
        # Drawdown analysis
        equity_series = equity_df['equity'].values
        peak = np.maximum.accumulate(equity_series)
        drawdown = (equity_series - peak) / peak
        max_drawdown = np.min(drawdown)
        
        # Additional metrics
        profit_factor = abs(avg_win * self.winning_trades / (avg_loss * self.losing_trades)) if avg_loss != 0 and self.losing_trades > 0 else float('inf')
        avg_trade_duration = trades_df['duration_hours'].mean()
        
        # CAGR calculation
        if len(equity_df) > 1:
            start_date = equity_df['timestamp'].iloc[0]
            end_date = equity_df['timestamp'].iloc[-1]
            years = (end_date - start_date).days / 365.25
            cagr = (final_equity / initial_capital) ** (1/years) - 1 if years > 0 else 0
        else:
            cagr = 0
        
        return {
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'cagr': cagr,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'avg_trade_duration_hours': avg_trade_duration,
            'total_pnl': self.total_pnl
        }
    
    def generate_report(self, metrics: Dict[str, Any]) -> str:
        """Generate formatted backtest report"""
        report = f"""
🚀 BACKTEST PERFORMANCE REPORT
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

💰 RETURNS:
  Initial Capital: ${metrics['initial_capital']:,.2f}
  Final Equity: ${metrics['final_equity']:,.2f}
  Total Return: {metrics['total_return']:.2%}
  CAGR: {metrics['cagr']:.2%}
  Total P&L: ${metrics['total_pnl']:,.2f}

📊 TRADE STATISTICS:
  Total Trades: {metrics['total_trades']}
  Winning Trades: {metrics['winning_trades']}
  Losing Trades: {metrics['losing_trades']}
  Win Rate: {metrics['win_rate']:.2%}
  Average Win: ${metrics['avg_win']:,.2f}
  Average Loss: ${metrics['avg_loss']:,.2f}
  Profit Factor: {metrics['profit_factor']:.2f}

⚡ PERFORMANCE METRICS:
  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}
  Sortino Ratio: {metrics['sortino_ratio']:.3f}
  Max Drawdown: {metrics['max_drawdown']:.2%}
  Avg Trade Duration: {metrics['avg_trade_duration_hours']:.1f} hours

{'='*60}
"""
        return report


@click.command()
@click.option('--csv', 'csv_file', required=True, help='Path to CSV data file')
@click.option('--strategy', default='ma_crossover', 
              type=click.Choice(['ma_crossover', 'kelly']),
              help='Strategy to backtest')
@click.option('--initial-capital', default=100000.0, help='Initial capital amount')
@click.option('--fast-ma', default=10, help='Fast MA period (for MA strategy)')
@click.option('--slow-ma', default=30, help='Slow MA period (for MA strategy)')
@click.option('--output', help='Output file for results (JSON)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def backtest_command(csv_file: str, strategy: str, initial_capital: float, 
                    fast_ma: int, slow_ma: int, output: Optional[str], verbose: bool):
    """
    Professional backtesting CLI tool
    
    Example usage:
    chimera backtest --csv data/btcusdt_1m.csv --strategy ma_crossover
    chimera backtest --csv data/ethusdt_1m.csv --strategy kelly --initial-capital 50000
    """
    
    # Configure logging
    if verbose:
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stdout, level="INFO")
    
    try:
        # Initialize backtest engine
        engine = BacktestEngine()
        
        # Load data
        click.echo(f"📈 Loading data from {csv_file}...")
        df = engine.load_data(csv_file)
        
        # Generate signals based on strategy
        click.echo(f"🎯 Running {strategy} strategy...")
        
        if strategy == 'ma_crossover':
            signals = engine.simple_ma_crossover_strategy(df, fast_ma, slow_ma)
        elif strategy == 'kelly':
            signals = engine.kelly_criterion_strategy(df)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        if not signals:
            click.echo("❌ No signals generated. Check your data and strategy parameters.")
            return
        
        # Run backtest
        click.echo(f"⚡ Executing backtest with {len(signals)} signals...")
        metrics = engine.run_backtest(df, signals, initial_capital)
        
        if 'error' in metrics:
            click.echo(f"❌ Backtest failed: {metrics['error']}")
            return
        
        # Generate and display report
        report = engine.generate_report(metrics)
        click.echo(report)
        
        # Save results if output specified
        if output:
            results = {
                'strategy': strategy,
                'parameters': {
                    'fast_ma': fast_ma,
                    'slow_ma': slow_ma,
                    'initial_capital': initial_capital
                },
                'metrics': metrics,
                'trades': engine.trades,
                'equity_curve': engine.equity_curve,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            click.echo(f"💾 Results saved to {output}")
        
        # Summary metrics for quick viewing
        click.echo(f"\n📋 SUMMARY:")
        click.echo(f"Sharpe: {metrics['sharpe_ratio']:.3f} | MaxDD: {metrics['max_drawdown']:.1%} | CAGR: {metrics['cagr']:.1%}")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}")
        if verbose:
            logger.exception("Backtest failed")
        return 1
    
    return 0


def main():
    """Main CLI entry point"""
    backtest_command()


if __name__ == '__main__':
    main()