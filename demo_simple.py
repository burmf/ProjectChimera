#!/usr/bin/env python3
"""
Simple ProjectChimera Performance Tracker Demo
Demonstrates strategy performance tracking with simulated trades
"""

import asyncio
import random
import sys
import os
from datetime import datetime, timedelta
from decimal import Decimal

# Add project root to Python path
sys.path.insert(0, '/home/ec2-user/ProjectChimera')

from src.project_chimera.monitor.strategy_performance import get_performance_tracker
from src.project_chimera.domains.market import Signal, MarketFrame, SignalType, SignalStrength, Ticker


class SimplePerformanceDemo:
    """Simple demo class for testing performance tracking"""
    
    def __init__(self):
        self.performance_tracker = get_performance_tracker()
        self.strategies = [
            'weekend_effect',
            'stop_reversion', 
            'funding_contrarian',
            'lob_reversion',
            'volatility_breakout',
            'cme_gap',
            'basis_arbitrage'
        ]
        self.symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']
        self.current_prices = {
            'BTCUSDT': 50000.0,
            'ETHUSDT': 3000.0,
            'BNBUSDT': 300.0
        }
    
    async def simulate_simple_trade(self, strategy_id: str, symbol: str):
        """Simulate a simple trade lifecycle"""
        try:
            # Get entry price with realistic movement
            base_price = self.current_prices[symbol]
            price_change = random.gauss(0, 0.02)  # 2% volatility
            entry_price = base_price * (1 + price_change)
            self.current_prices[symbol] = entry_price
            
            # Create a simple signal (bypass complex domain objects for demo)
            signal = Signal(
                symbol=symbol,
                signal_type=random.choice([SignalType.BUY, SignalType.SELL]),
                strength=SignalStrength.MEDIUM,
                price=Decimal(str(entry_price)),
                timestamp=datetime.now(),
                strategy_name=strategy_id,
                confidence=random.uniform(0.6, 0.9),
                indicators_used={
                    'demo': True,
                    'market_condition': random.choice(['trending', 'ranging', 'volatile'])
                }
            )
            
            # Create market frame
            ticker = Ticker(
                symbol=symbol,
                price=Decimal(str(entry_price)),
                volume_24h=Decimal(str(random.uniform(10000, 100000))),
                change_24h=Decimal(str(price_change * 100)),
                timestamp=datetime.now()
            )
            
            market_frame = MarketFrame(
                symbol=symbol,
                timestamp=datetime.now(),
                ticker=ticker
            )
            
            # Record signal generation
            signal_id = await self.performance_tracker.record_signal_generated(signal, market_frame)
            
            # Record trade entry
            side = "buy" if signal.signal_type == SignalType.BUY else "sell"
            size_usd = random.uniform(500, 2000)
            size_native = size_usd / entry_price
            
            trade_signal_id = await self.performance_tracker.record_trade_entry(
                strategy_id=strategy_id,
                signal=signal,
                entry_price=entry_price,
                size_usd=size_usd,
                size_native=size_native,
                slippage_bps=random.uniform(1, 5),
                commission_usd=size_usd * 0.001
            )
            
            print(f"📈 Trade Entry: {strategy_id} {side} {symbol} ${size_usd:.2f} @ ${entry_price:.2f}")
            
            # Simulate holding period (speed up for demo)
            holding_time = random.uniform(1, 10)  # 1-10 seconds for demo
            await asyncio.sleep(holding_time)
            
            # Simulate exit price
            exit_change = random.gauss(0, 0.015)  # 1.5% exit volatility
            exit_price = entry_price * (1 + exit_change)
            
            # Record trade exit
            completed_trade = await self.performance_tracker.record_trade_exit(
                signal_id=trade_signal_id,
                exit_price=exit_price,
                commission_usd=size_usd * 0.001
            )
            
            if completed_trade:
                pnl_color = "🟢" if completed_trade.pnl_usd > 0 else "🔴"
                print(f"📉 Trade Exit: {strategy_id} {symbol} P&L: {pnl_color} ${completed_trade.pnl_usd:.2f} ({completed_trade.pnl_pct:.2f}%)")
            
            return completed_trade
            
        except Exception as e:
            print(f"❌ Error in trade simulation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    async def run_demo(self, num_trades: int = 20):
        """Run performance tracking demo"""
        print("🚀 Starting Simple ProjectChimera Performance Demo")
        print(f"📊 Simulating {num_trades} trades across {len(self.strategies)} strategies")
        print("=" * 60)
        
        # Run trades sequentially for clearer output
        completed_trades = []
        for i in range(num_trades):
            strategy_id = random.choice(self.strategies)
            symbol = random.choice(self.symbols)
            
            print(f"\n[{i+1}/{num_trades}] Executing trade...")
            trade = await self.simulate_simple_trade(strategy_id, symbol)
            if trade:
                completed_trades.append(trade)
            
            # Small delay between trades
            await asyncio.sleep(0.5)
        
        # Display results
        await self.display_results()
        
        return completed_trades
    
    async def display_results(self):
        """Display performance results"""
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE RESULTS")
        print("=" * 60)
        
        try:
            # Overall summary
            summary = self.performance_tracker.get_performance_summary()
            print(f"📈 Total Strategies: {summary['total_strategies']}")
            print(f"📈 Total Trades: {summary['total_trades']}")
            print(f"💰 Total P&L: ${summary['total_pnl_usd']:.2f}")
            print(f"📊 Average Win Rate: {summary['average_win_rate']:.1f}%")
            
            if summary['best_strategy'] and summary['worst_strategy']:
                print(f"🏆 Best Strategy: {summary['best_strategy']}")
                print(f"📉 Worst Strategy: {summary['worst_strategy']}")
            
            print("\n" + "-" * 60)
            print("STRATEGY PERFORMANCE BREAKDOWN")
            print("-" * 60)
            
            # Strategy-by-strategy breakdown
            all_stats = self.performance_tracker.get_all_strategy_stats()
            
            for strategy_id, stats in all_stats.items():
                win_rate_emoji = "🟢" if stats.win_rate >= 60 else "🟡" if stats.win_rate >= 50 else "🔴"
                pnl_emoji = "💰" if stats.total_pnl_usd > 0 else "📉"
                
                print(f"\n📊 {strategy_id.upper()}")
                print(f"   Trades: {stats.total_trades} | Win Rate: {win_rate_emoji} {stats.win_rate:.1f}%")
                print(f"   P&L: {pnl_emoji} ${stats.total_pnl_usd:.2f} | Sharpe: {stats.sharpe_ratio:.2f}")
                print(f"   Profit Factor: {stats.profit_factor:.2f} | Max DD: {stats.max_drawdown_pct:.1f}%")
            
            # Open positions check
            open_positions = self.performance_tracker.get_open_positions()
            total_open = sum(len(positions) for positions in open_positions.values())
            print(f"\n📋 Open Positions: {total_open}")
            
            print("\n✅ Demo completed successfully!")
            print("🌐 View detailed results in the Streamlit dashboard:")
            print("   Main Dashboard: http://localhost:8501")
            print("   Strategy Dashboard: http://localhost:8502")
            
        except Exception as e:
            print(f"❌ Error displaying results: {e}")
            import traceback
            traceback.print_exc()


async def main():
    """Main demo function"""
    demo = SimplePerformanceDemo()
    
    try:
        trades = await demo.run_demo(num_trades=15)
        print(f"\n📊 Successfully completed {len(trades)} trades")
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🎯 ProjectChimera Simple Performance Demo")
    print("📊 This demo simulates trading activity and tracks performance")
    print("⚡ Simplified version with direct trade recording")
    
    # Run the demo
    asyncio.run(main())