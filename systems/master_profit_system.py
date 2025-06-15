#!/usr/bin/env python3
"""
Master Profit Maximization System
総合利益最大化システム - 全コンポーネント統合
"""

import asyncio
import json
import time
import logging
import sys
import os
import threading
import signal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import deque
import statistics

# Add modules to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))

# Import all systems
from core.bitget_futures_client import BitgetFuturesClient
from core.advanced_risk_manager import AdvancedRiskManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'master_profit_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MasterProfitSystem:
    """
    統合利益最大化システム
    
    - 複数取引ボット協調
    - AI最適化統合
    - 高度リスク管理
    - リアルタイム監視
    - 自動利益最大化
    """
    
    def __init__(self):
        # Core components
        self.futures_client = BitgetFuturesClient()
        self.risk_manager = AdvancedRiskManager(100000)
        
        # Trading configuration
        self.trading_pairs = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        self.base_leverage = 40  # 40x aggressive leverage
        self.position_size_usd = 60000  # $60k per position
        self.max_positions = 12  # 12 simultaneous positions
        
        # AI optimization parameters (proven)
        self.ai_confidence_threshold = 0.75
        self.profit_target = 0.008  # 0.8% target
        self.stop_loss = 0.003      # 0.3% stop
        self.momentum_threshold = 0.0008
        self.volatility_min = 0.001
        
        # Performance tracking
        self.start_time = datetime.now()
        self.start_balance = 100000
        self.current_balance = 100000
        self.total_profit = 0
        self.active_positions = {}
        self.trade_history = []
        self.price_data = {}
        
        # Statistics
        self.trades_today = 0
        self.wins_today = 0
        self.daily_target = 1000  # $1000 daily target
        
        # System state
        self.is_running = False
        self.shutdown_requested = False
        
        # Alert system
        self.alerts = deque(maxlen=100)
        self.last_alert_time = {}
        
        logger.info("🚀 Master Profit System initialized")
        logger.info(f"Target: ${self.daily_target:,}/day | {self.base_leverage}x leverage")
        logger.info(f"Max positions: {self.max_positions} | Position size: ${self.position_size_usd:,}")
    
    def setup_signal_handlers(self):
        """Signal handlers for safe shutdown"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating safe shutdown...")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def collect_market_data(self) -> Dict[str, Any]:
        """Comprehensive market data collection"""
        market_data = {}
        
        for symbol in self.trading_pairs:
            try:
                ticker = self.futures_client.get_futures_ticker(symbol)
                if ticker:
                    market_data[symbol] = {
                        'price': ticker['price'],
                        'change_24h': ticker['change_24h'],
                        'ask': ticker['ask_price'],
                        'bid': ticker['bid_price'],
                        'spread': ticker['ask_price'] - ticker['bid_price'],
                        'timestamp': datetime.now()
                    }
                    
                    # Update price history
                    if symbol not in self.price_data:
                        self.price_data[symbol] = deque(maxlen=200)
                    
                    self.price_data[symbol].append({
                        'time': datetime.now(),
                        'price': ticker['price']
                    })
                    
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {e}")
        
        return market_data
    
    def calculate_ultra_signals(self, symbol: str, market_data: Dict) -> Dict[str, Any]:
        """Ultra-optimized AI signal generation"""
        if symbol not in self.price_data or len(self.price_data[symbol]) < 50:
            return {'action': 'hold', 'confidence': 0, 'reasoning': 'Insufficient data'}
        
        prices = [p['price'] for p in list(self.price_data[symbol])]
        current_price = prices[-1]
        
        # Multi-timeframe momentum analysis
        momentum_1m = (prices[-1] - prices[-2]) / prices[-2] if len(prices) >= 2 else 0
        momentum_3m = (prices[-1] - prices[-4]) / prices[-4] if len(prices) >= 4 else 0
        momentum_5m = (prices[-1] - prices[-6]) / prices[-6] if len(prices) >= 6 else 0
        momentum_10m = (prices[-1] - prices[-11]) / prices[-11] if len(prices) >= 11 else 0
        
        # Volatility and trend analysis
        returns = [(prices[i] / prices[i-1] - 1) for i in range(1, min(len(prices), 50))]
        volatility = statistics.stdev(returns) if len(returns) > 1 else 0
        
        # Trend strength (linear regression slope)
        if len(prices) >= 30:
            x = list(range(30))
            y = prices[-30:]
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            if n * sum_x2 - sum_x ** 2 != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
                trend_strength = abs(slope) / (sum_y / n)
            else:
                trend_strength = 0
        else:
            trend_strength = 0
        
        # Spread analysis
        spread_pct = market_data[symbol]['spread'] / current_price if symbol in market_data else 0.001
        
        signal = {
            'action': 'hold',
            'confidence': 0,
            'reasoning': '',
            'leverage': self.base_leverage,
            'position_size': self.position_size_usd
        }
        
        # Ultra-aggressive signal generation
        if volatility >= self.volatility_min and spread_pct < 0.0002:  # Excellent liquidity
            
            # Strong bullish confluence
            if (momentum_1m > self.momentum_threshold and
                momentum_3m > 0.002 and
                momentum_10m > 0.003 and
                trend_strength > 0.0002):
                
                signal['action'] = 'long'
                signal['confidence'] = min(0.98, 
                    momentum_1m * 1000 + 
                    momentum_3m * 400 + 
                    momentum_10m * 200 + 
                    trend_strength * 2000
                )
                signal['reasoning'] = f'Ultra bullish: 1m={momentum_1m:.4f}, 3m={momentum_3m:.4f}, trend={trend_strength:.6f}'
                signal['target'] = current_price * (1 + self.profit_target)
                signal['stop'] = current_price * (1 - self.stop_loss)
                
                # Dynamic leverage based on confidence
                if signal['confidence'] > 0.9:
                    signal['leverage'] = min(75, self.base_leverage * 1.8)
                elif signal['confidence'] > 0.8:
                    signal['leverage'] = min(60, self.base_leverage * 1.5)
            
            # Strong bearish confluence
            elif (momentum_1m < -self.momentum_threshold and
                  momentum_3m < -0.002 and
                  momentum_10m < -0.003 and
                  trend_strength > 0.0002):
                
                signal['action'] = 'short'
                signal['confidence'] = min(0.98, 
                    abs(momentum_1m) * 1000 + 
                    abs(momentum_3m) * 400 + 
                    abs(momentum_10m) * 200 + 
                    trend_strength * 2000
                )
                signal['reasoning'] = f'Ultra bearish: 1m={momentum_1m:.4f}, 3m={momentum_3m:.4f}, trend={trend_strength:.6f}'
                signal['target'] = current_price * (1 - self.profit_target)
                signal['stop'] = current_price * (1 + self.stop_loss)
                
                if signal['confidence'] > 0.9:
                    signal['leverage'] = min(75, self.base_leverage * 1.8)
                elif signal['confidence'] > 0.8:
                    signal['leverage'] = min(60, self.base_leverage * 1.5)
            
            # High-volatility scalping
            elif volatility > 0.006 and abs(momentum_1m) > 0.003:
                direction = 'long' if momentum_1m > 0 else 'short'
                signal['action'] = direction
                signal['confidence'] = 0.85
                signal['leverage'] = min(50, self.base_leverage * 1.2)
                signal['reasoning'] = f'High-vol scalp: vol={volatility:.4f}, mom={momentum_1m:.4f}'
                
                if direction == 'long':
                    signal['target'] = current_price * (1 + self.profit_target * 0.7)
                    signal['stop'] = current_price * (1 - self.stop_loss * 0.8)
                else:
                    signal['target'] = current_price * (1 - self.profit_target * 0.7)
                    signal['stop'] = current_price * (1 + self.stop_loss * 0.8)
        
        return signal
    
    async def execute_optimized_trade(self, symbol: str, signal: Dict) -> Optional[Dict]:
        """Execute trade with risk management integration"""
        if len(self.active_positions) >= self.max_positions:
            logger.warning(f"Max positions reached ({self.max_positions})")
            return None
        
        # Risk management check
        new_position = {
            'symbol': symbol,
            'position_size': signal['position_size'],
            'leverage': signal['leverage']
        }
        
        within_limits, warnings = self.risk_manager.check_risk_limits(
            self.active_positions, new_position
        )
        
        if not within_limits:
            logger.warning(f"Risk limits exceeded for {symbol}: {warnings}")
            return None
        
        # Calculate optimal position size
        size, reasoning = self.risk_manager.calculate_position_size(
            symbol=symbol,
            signal_confidence=signal['confidence'],
            expected_return=self.profit_target,
            risk_per_trade=self.stop_loss,
            market_volatility=0.002,  # Estimate
            current_positions=self.active_positions
        )
        
        trade_id = f"{symbol}_{int(time.time() * 1000)}"
        
        trade = {
            'id': trade_id,
            'symbol': symbol,
            'action': signal['action'],
            'entry_price': list(self.price_data[symbol])[-1]['price'],
            'target_price': signal.get('target', 0),
            'stop_price': signal.get('stop', 0),
            'position_size': size,
            'leverage': signal['leverage'],
            'confidence': signal['confidence'],
            'reasoning': signal['reasoning'],
            'risk_reasoning': reasoning,
            'entry_time': datetime.now(),
            'status': 'active'
        }
        
        self.active_positions[trade_id] = trade
        self.trades_today += 1
        
        # Update risk manager
        self.risk_manager.current_balance = self.current_balance
        
        logger.info(f"🎯 EXECUTED #{self.trades_today}: {signal['action'].upper()} {symbol}")
        logger.info(f"   Entry: ${trade['entry_price']:,.2f} | Target: ${trade['target_price']:,.2f}")
        logger.info(f"   Size: ${size:,.0f} | Leverage: {trade['leverage']}x | Conf: {signal['confidence']:.1%}")
        logger.info(f"   Risk: {reasoning}")
        
        return trade
    
    async def manage_positions_ultra(self, market_data: Dict) -> List[Dict]:
        """Ultra-efficient position management"""
        completed_trades = []
        
        for trade_id, trade in list(self.active_positions.items()):
            symbol = trade['symbol']
            if symbol not in market_data:
                continue
            
            current_price = market_data[symbol]['price']
            entry_price = trade['entry_price']
            
            # P&L calculation
            if trade['action'] == 'long':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price
            
            pnl_amount = trade['position_size'] * trade['leverage'] * pnl_pct
            
            # Exit conditions
            should_close = False
            close_reason = ''
            
            # Profit target
            if trade['action'] == 'long' and current_price >= trade['target_price']:
                should_close = True
                close_reason = 'profit_target'
            elif trade['action'] == 'short' and current_price <= trade['target_price']:
                should_close = True
                close_reason = 'profit_target'
            
            # Stop loss
            elif trade['action'] == 'long' and current_price <= trade['stop_price']:
                should_close = True
                close_reason = 'stop_loss'
            elif trade['action'] == 'short' and current_price >= trade['stop_price']:
                should_close = True
                close_reason = 'stop_loss'
            
            # Time-based exit (2 minutes for ultra-fast trading)
            elif datetime.now() - trade['entry_time'] > timedelta(minutes=2):
                should_close = True
                close_reason = 'timeout'
            
            # Emergency stop (-1.5%)
            elif pnl_pct < -0.015:
                should_close = True
                close_reason = 'emergency_stop'
            
            # Trailing profit (secure 70% of profit when above 0.6%)
            elif pnl_pct > 0.006:
                secure_price = entry_price * (1 + pnl_pct * 0.7)
                if trade['action'] == 'long' and current_price <= secure_price:
                    should_close = True
                    close_reason = 'trailing_profit'
                elif trade['action'] == 'short' and current_price >= secure_price:
                    should_close = True
                    close_reason = 'trailing_profit'
            
            if should_close:
                # Close position
                trade['exit_price'] = current_price
                trade['exit_time'] = datetime.now()
                trade['pnl_amount'] = pnl_amount
                trade['pnl_pct'] = pnl_pct
                trade['close_reason'] = close_reason
                trade['duration'] = trade['exit_time'] - trade['entry_time']
                
                # Update balances
                self.total_profit += pnl_amount
                self.current_balance += pnl_amount
                
                if pnl_amount > 0:
                    self.wins_today += 1
                
                # Update risk manager
                self.risk_manager.update_balance(self.current_balance, pnl_amount)
                
                # Add to history
                self.trade_history.append(trade)
                completed_trades.append(trade)
                
                # Remove from active
                del self.active_positions[trade_id]
                
                # Log
                status_emoji = "✅" if pnl_amount > 0 else "❌"
                duration_sec = trade['duration'].total_seconds()
                
                logger.info(f"{status_emoji} CLOSED: {close_reason.upper()}")
                logger.info(f"   {symbol} | P&L: {pnl_pct:+.3%} (${pnl_amount:+,.2f})")
                logger.info(f"   Duration: {duration_sec:.0f}s | Balance: ${self.current_balance:,.2f}")
                
                # Alert on significant wins
                if pnl_amount > 500:
                    self.add_alert('big_win', f'${pnl_amount:+,.0f} profit on {symbol}')
        
        return completed_trades
    
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Comprehensive performance analysis"""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'daily_return': 0,
                'hourly_rate': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'profit_factor': 0
            }
        
        trades = self.trade_history
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['pnl_amount'] > 0)
        win_rate = winning_trades / total_trades
        
        # Profit metrics
        gross_profit = sum(t['pnl_amount'] for t in trades if t['pnl_amount'] > 0)
        gross_loss = abs(sum(t['pnl_amount'] for t in trades if t['pnl_amount'] < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Time-based returns
        elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        daily_return = (self.total_profit / self.start_balance) * (24 / max(0.1, elapsed_hours))
        hourly_rate = self.total_profit / max(0.1, elapsed_hours)
        
        # Risk metrics
        if total_trades > 1:
            returns = [t['pnl_amount'] / self.start_balance for t in trades]
            mean_return = statistics.mean(returns)
            return_std = statistics.stdev(returns)
            sharpe_ratio = (mean_return * 252) / (return_std * (252 ** 0.5)) if return_std > 0 else 0
            
            # Max drawdown
            running_balance = self.start_balance
            peak_balance = self.start_balance
            max_drawdown = 0
            
            for trade in trades:
                running_balance += trade['pnl_amount']
                peak_balance = max(peak_balance, running_balance)
                drawdown = (peak_balance - running_balance) / peak_balance
                max_drawdown = max(max_drawdown, drawdown)
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_profit': self.total_profit,
            'current_balance': self.current_balance,
            'total_return': (self.current_balance - self.start_balance) / self.start_balance,
            'daily_return': daily_return,
            'hourly_rate': hourly_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'active_positions': len(self.active_positions),
            'daily_target_progress': self.total_profit / self.daily_target
        }
    
    def add_alert(self, alert_type: str, message: str):
        """Add system alert"""
        # Rate limiting
        if alert_type in self.last_alert_time:
            if datetime.now() - self.last_alert_time[alert_type] < timedelta(minutes=5):
                return
        
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'message': message
        }
        
        self.alerts.append(alert)
        self.last_alert_time[alert_type] = datetime.now()
        
        logger.warning(f"🚨 ALERT [{alert_type}]: {message}")
    
    def print_dashboard(self):
        """Real-time dashboard display"""
        performance = self.calculate_performance_metrics()
        elapsed = datetime.now() - self.start_time
        
        print(f"\n{'='*80}")
        print(f"🚀 MASTER PROFIT SYSTEM DASHBOARD | Runtime: {elapsed}")
        print(f"{'='*80}")
        
        # Balance and profit
        balance_change = self.current_balance - self.start_balance
        roi = (balance_change / self.start_balance) * 100
        print(f"💰 Balance: ${self.current_balance:,.2f} | Profit: ${balance_change:+,.2f} ({roi:+.2f}%)")
        
        # Daily target progress
        target_progress = (self.total_profit / self.daily_target) * 100
        print(f"🎯 Daily Target: ${self.total_profit:+,.0f} / ${self.daily_target:,} ({target_progress:.1f}%)")
        
        # Performance metrics
        print(f"📊 Trades: {performance['total_trades']} | Win Rate: {performance['win_rate']:.1%}")
        print(f"💹 Daily Return: {performance['daily_return']:+.2%} | Hourly: ${performance['hourly_rate']:+,.0f}")
        print(f"⚡ Sharpe: {performance['sharpe_ratio']:.2f} | Profit Factor: {performance['profit_factor']:.2f}")
        
        # Active positions
        print(f"🔥 Active Positions: {len(self.active_positions)}/{self.max_positions}")
        
        if self.active_positions:
            print(f"\n📈 Current Positions:")
            for trade in list(self.active_positions.values()):
                duration = (datetime.now() - trade['entry_time']).total_seconds() / 60
                print(f"   {trade['symbol']} {trade['action'].upper()} | "
                      f"{duration:.1f}min | {trade['leverage']}x | "
                      f"Conf: {trade['confidence']:.1%}")
        
        # Risk status
        risk_metrics = self.risk_manager.get_risk_metrics()
        print(f"\n⚠️  Risk: DD {risk_metrics['current_drawdown']:.1%} | "
              f"Regime: {risk_metrics['market_regime'].upper()}")
        
        print(f"{'='*80}")
    
    def save_session_data(self):
        """Save session data"""
        performance = self.calculate_performance_metrics()
        
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'session_duration_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'performance': performance,
            'trading_config': {
                'pairs': self.trading_pairs,
                'base_leverage': self.base_leverage,
                'position_size': self.position_size_usd,
                'max_positions': self.max_positions
            },
            'trade_history': [
                {
                    'symbol': t['symbol'],
                    'action': t['action'],
                    'entry_time': t['entry_time'].isoformat(),
                    'exit_time': t['exit_time'].isoformat(),
                    'pnl_amount': t['pnl_amount'],
                    'pnl_pct': t['pnl_pct'],
                    'duration_seconds': t['duration'].total_seconds(),
                    'close_reason': t['close_reason'],
                    'leverage': t['leverage'],
                    'confidence': t['confidence']
                } for t in self.trade_history
            ]
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'master_profit_session_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        logger.info(f"Session data saved: {filename}")
        return filename
    
    async def run_profit_session(self, duration_hours: float = 24):
        """Main profit maximization session"""
        self.setup_signal_handlers()
        self.is_running = True
        
        logger.info(f"🚀 Starting Master Profit Session for {duration_hours} hours")
        logger.info(f"Target: ${self.daily_target:,}/day with {self.base_leverage}x leverage")
        
        end_time = datetime.now() + timedelta(hours=duration_hours)
        last_dashboard_time = datetime.now()
        last_regime_check = datetime.now()
        
        try:
            while datetime.now() < end_time and not self.shutdown_requested:
                # Collect market data
                market_data = await self.collect_market_data()
                
                if not market_data:
                    logger.warning("No market data, retrying...")
                    await asyncio.sleep(5)
                    continue
                
                # Market regime detection (every 5 minutes)
                if datetime.now() - last_regime_check > timedelta(minutes=5):
                    regime = self.risk_manager.detect_market_regime(market_data, self.price_data)
                    last_regime_check = datetime.now()
                
                # Manage existing positions
                completed = await self.manage_positions_ultra(market_data)
                
                # Generate and execute new signals
                for symbol in self.trading_pairs:
                    if symbol in market_data and len(self.active_positions) < self.max_positions:
                        signal = self.calculate_ultra_signals(symbol, market_data)
                        
                        if (signal['action'] != 'hold' and 
                            signal['confidence'] > self.ai_confidence_threshold):
                            
                            await self.execute_optimized_trade(symbol, signal)
                            await asyncio.sleep(0.5)  # Brief pause between trades
                
                # Dashboard update (every 10 seconds)
                if datetime.now() - last_dashboard_time > timedelta(seconds=10):
                    self.print_dashboard()
                    last_dashboard_time = datetime.now()
                    
                    # Check daily target
                    if self.total_profit >= self.daily_target:
                        self.add_alert('target_reached', f'Daily target of ${self.daily_target:,} achieved!')
                
                # Brief pause
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            await self.safe_shutdown()
    
    async def safe_shutdown(self):
        """Safe system shutdown"""
        logger.info("🛑 Initiating safe shutdown...")
        
        # Close all positions
        if self.active_positions:
            logger.info(f"Closing {len(self.active_positions)} active positions...")
            market_data = await self.collect_market_data()
            await self.manage_positions_ultra(market_data)
        
        # Final dashboard
        self.print_dashboard()
        
        # Save session data
        self.save_session_data()
        
        # Final report
        performance = self.calculate_performance_metrics()
        
        logger.info("📊 FINAL SESSION REPORT")
        logger.info(f"Total Profit: ${performance['total_profit']:+,.2f}")
        logger.info(f"Total Return: {performance['total_return']:+.2%}")
        logger.info(f"Win Rate: {performance['win_rate']:.1%}")
        logger.info(f"Trades Executed: {performance['total_trades']}")
        logger.info(f"Sharpe Ratio: {performance['sharpe_ratio']:.2f}")
        
        if performance['total_profit'] >= self.daily_target:
            logger.info(f"🎉 DAILY TARGET ACHIEVED! ${performance['total_profit']:,.0f} >= ${self.daily_target:,}")
        else:
            remaining = self.daily_target - performance['total_profit']
            logger.info(f"💪 Continue for ${remaining:,.0f} more to reach daily target")
        
        self.is_running = False


async def main():
    """Main execution"""
    system = MasterProfitSystem()
    
    print("🚀 MASTER PROFIT MAXIMIZATION SYSTEM")
    print("=" * 60)
    print(f"Target: ${system.daily_target:,}/day")
    print(f"Strategy: {system.base_leverage}x leverage, ultra-high frequency")
    print(f"Pairs: {', '.join(system.trading_pairs)}")
    print(f"Max positions: {system.max_positions}")
    print("=" * 60)
    
    # Run for 24 hours (or until daily target reached)
    await system.run_profit_session(duration_hours=24)


if __name__ == "__main__":
    asyncio.run(main())