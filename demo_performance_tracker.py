#!/usr/bin/env python3
"""
ProjectChimera Performance Tracker Demo
Demonstrates strategy performance tracking with simulated trades
"""

import asyncio
import random
import sys
import os
from datetime import datetime, timedelta
from typing import List

# Add project root to Python path
sys.path.insert(0, '/home/ec2-user/ProjectChimera')

from src.project_chimera.monitor.strategy_performance import get_performance_tracker
from src.project_chimera.domains.market import Signal, MarketFrame


class PerformanceDemo:
    """Demo class for testing performance tracking"""
    
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
    
    def create_market_frame(self, symbol: str) -> MarketFrame:
        """Create a market frame with realistic price movement"""
        from decimal import Decimal
        from src.project_chimera.domains.market import Ticker
        
        base_price = self.current_prices[symbol]
        
        # Simulate price movement (±2% max)
        price_change = random.gauss(0, 0.02)
        new_price = base_price * (1 + price_change)
        self.current_prices[symbol] = new_price
        
        # Create ticker data
        ticker = Ticker(
            symbol=symbol,
            price=Decimal(str(new_price)),
            volume_24h=Decimal(str(random.uniform(10000, 100000))),
            change_24h=Decimal(str(price_change * 100)),
            timestamp=datetime.now()
        )
        
        return MarketFrame(
            symbol=symbol,
            timestamp=datetime.now(),
            ticker=ticker
        )
    
    def create_signal(self, strategy_id: str, symbol: str, entry_price: float) -> Signal:
        """Create a trading signal"""
        from decimal import Decimal
        from src.project_chimera.domains.market import SignalType, SignalStrength
        
        return Signal(
            symbol=symbol,
            signal_type=random.choice([SignalType.BUY, SignalType.SELL]),
            strength=random.choice([SignalStrength.WEAK, SignalStrength.MEDIUM, SignalStrength.STRONG]),
            price=Decimal(str(entry_price)),
            timestamp=datetime.now(),
            strategy_name=strategy_id,
            confidence=random.uniform(0.5, 0.95),
            indicators_used={
                'demo': True,
                'market_condition': random.choice(['trending', 'ranging', 'volatile'])
            }
        )
    
    async def simulate_trade_lifecycle(self, strategy_id: str, symbol: str):
        """Simulate a complete trade lifecycle"""
        try:
            # Create market frame 
            market_frame = self.create_market_frame(symbol)
            entry_price = float(market_frame.current_price) if market_frame.current_price else 50000.0
            
            # Create signal with proper price
            signal = self.create_signal(strategy_id, symbol, entry_price)
            
            # Record signal generation
            signal_id = await self.performance_tracker.record_signal_generated(signal, market_frame)
            
            # Record trade entry
            size_usd = random.uniform(500, 2000)  # $500-$2000 per trade
            size_native = size_usd / entry_price
            slippage_bps = random.uniform(1, 5)
            commission_usd = size_usd * 0.001  # 0.1% commission
            
            trade_signal_id = await self.performance_tracker.record_trade_entry(
                strategy_id=strategy_id,
                signal=signal,
                entry_price=entry_price,
                size_usd=size_usd,
                size_native=size_native,
                slippage_bps=slippage_bps,
                commission_usd=commission_usd
            )
            
            print(f"📈 Trade Entry: {strategy_id} {signal.action} {symbol} ${size_usd:.2f} @ ${entry_price:.2f}")
            
            # Simulate holding period
            holding_time = random.uniform(300, 7200)  # 5 minutes to 2 hours
            await asyncio.sleep(min(holding_time / 100, 2))  # Speed up for demo
            
            # Update unrealized P&L during holding
            for _ in range(random.randint(1, 5)):
                market_frame = self.create_market_frame(symbol)
                current_price = float(market_frame.current_price) if market_frame.current_price else entry_price
                await self.performance_tracker.update_unrealized_pnl(symbol, current_price)
                await asyncio.sleep(0.1)
            
            # Record trade exit
            exit_market_frame = self.create_market_frame(symbol)
            exit_price = float(exit_market_frame.current_price) if exit_market_frame.current_price else entry_price
            exit_commission = size_usd * 0.001
            
            completed_trade = await self.performance_tracker.record_trade_exit(
                signal_id=trade_signal_id,
                exit_price=exit_price,
                commission_usd=exit_commission
            )
            
            if completed_trade:
                pnl_color = "🟢" if completed_trade.pnl_usd > 0 else "🔴"
                print(f"📉 Trade Exit: {strategy_id} {symbol} P&L: {pnl_color} ${completed_trade.pnl_usd:.2f} ({completed_trade.pnl_pct:.2f}%)")
            
        except Exception as e:
            print(f"❌ Error in trade simulation: {e}")
    
    async def run_demo(self, num_trades: int = 50):
        """Run performance tracking demo"""
        print("🚀 Starting ProjectChimera Performance Tracker Demo")
        print(f"📊 Simulating {num_trades} trades across {len(self.strategies)} strategies")
        print("=" * 60)
        
        # Simulate multiple trades concurrently
        tasks = []
        for i in range(num_trades):
            strategy_id = random.choice(self.strategies)
            symbol = random.choice(self.symbols)
            
            task = self.simulate_trade_lifecycle(strategy_id, symbol)
            tasks.append(task)
            
            # Add small delay between trade starts
            if i % 5 == 0:
                await asyncio.sleep(0.2)
        
        # Execute all trades
        print("⏳ Executing trades...")
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Display results
        await self.display_results()
    
    async def display_results(self):
        """Display performance results"""
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE RESULTS")
        print("=" * 60)
        
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
        
        # Open positions
        open_positions = self.performance_tracker.get_open_positions()
        total_open = sum(len(positions) for positions in open_positions.values())
        
        print(f"\n📋 Open Positions: {total_open}")
        for strategy_id, positions in open_positions.items():
            if positions:
                print(f"   {strategy_id}: {len(positions)} positions")
        
        print("\n✅ Demo completed successfully!")
        print("🌐 View detailed results in the Streamlit dashboard at http://localhost:8502")


async def main():
    """Main demo function"""
    demo = PerformanceDemo()
    
    try:
        await demo.run_demo(num_trades=30)
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🎯 ProjectChimera Performance Tracker Demo")
    print("📊 This demo will simulate trading activity and track performance")
    
    # Run the demo
    asyncio.run(main())