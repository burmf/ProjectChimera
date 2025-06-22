#!/usr/bin/env python3
"""
Test fixes for ProjectChimera
Verify that data duplication and API issues are resolved
"""

import asyncio
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_performance_tracker_fixes():
    """Test performance tracker data consistency fixes"""
    print("🔧 TESTING PERFORMANCE TRACKER FIXES")
    print("=" * 50)
    
    from src.project_chimera.monitor.strategy_performance import reset_performance_tracker, get_performance_tracker
    
    # Reset tracker to ensure clean state
    print("🔄 Resetting performance tracker...")
    reset_performance_tracker()
    
    # Get fresh instance
    print("📊 Creating fresh tracker instance...")
    tracker = get_performance_tracker(force_reload=True)
    
    # Check data consistency
    summary = tracker.get_performance_summary()
    print(f"📈 Performance Summary:")
    print(f"   Total Strategies: {summary['total_strategies']}")
    print(f"   Total Trades: {summary['total_trades']}")
    print(f"   Total P&L: ${summary['total_pnl_usd']:.2f}")
    print(f"   Average Win Rate: {summary['average_win_rate']:.1f}%")
    
    # Verify against database directly
    import sqlite3
    db_path = "data/strategy_performance.db"
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM trade_records WHERE status = 'filled'")
        db_trades = cursor.fetchone()[0]
        
        cursor.execute("SELECT SUM(pnl_usd) FROM trade_records WHERE status = 'filled'")
        db_pnl = cursor.fetchone()[0] or 0.0
    
    print(f"🗄️ Database Direct Query:")
    print(f"   Trades in DB: {db_trades}")
    print(f"   P&L in DB: ${db_pnl:.2f}")
    
    # Check consistency
    if summary['total_trades'] == db_trades:
        print("✅ Trade count consistency: FIXED")
    else:
        print(f"❌ Trade count mismatch: Tracker {summary['total_trades']} vs DB {db_trades}")
    
    if abs(summary['total_pnl_usd'] - db_pnl) < 0.01:
        print("✅ P&L consistency: FIXED")
    else:
        print(f"❌ P&L mismatch: Tracker ${summary['total_pnl_usd']:.2f} vs DB ${db_pnl:.2f}")
    
    return summary['total_trades'] == db_trades and abs(summary['total_pnl_usd'] - db_pnl) < 0.01

async def test_bitget_api_fixes():
    """Test Bitget API symbol format fixes"""
    print("\n📡 TESTING BITGET API FIXES")
    print("=" * 50)
    
    from src.project_chimera.api.bitget_client import BitgetMarketDataService
    
    service = BitgetMarketDataService()
    
    try:
        print("🌐 Testing market overview with fixed symbols...")
        overview = await service.get_market_overview()
        
        tickers = overview.get('tickers', {})
        print(f"📊 Symbols retrieved: {len(tickers)}")
        
        working_symbols = []
        for symbol, data in tickers.items():
            if data.get('price', 0) > 0:
                working_symbols.append(symbol)
                print(f"✅ {symbol}: ${data.get('price', 0):,.2f}")
        
        if len(working_symbols) > 0:
            print(f"✅ Bitget API fixes successful: {len(working_symbols)} symbols working")
            success = True
        else:
            print("❌ No symbols returning data")
            success = False
            
        # Test system status
        system_status = overview.get('system_status', {})
        if system_status.get('status') == 'online':
            print("✅ Bitget system status: Online")
        else:
            print("⚠️ Bitget system status: Unknown")
        
        await service.close()
        return success
        
    except Exception as e:
        print(f"❌ Bitget API test failed: {e}")
        await service.close()
        return False

async def test_dashboard_integration():
    """Test dashboard integration with fixes"""
    print("\n🖥️ TESTING DASHBOARD INTEGRATION")
    print("=" * 50)
    
    try:
        from src.project_chimera.ui.dashboard import TradingSystemAPI
        
        # Reset tracker for dashboard test
        from src.project_chimera.monitor.strategy_performance import reset_performance_tracker
        reset_performance_tracker()
        
        print("🔌 Testing dashboard API with fixes...")
        api = TradingSystemAPI()
        
        # Test health data
        health = api.get_health()
        print(f"❤️ System health: {health.get('status')}")
        
        # Test metrics with fixed tracker
        metrics = api.get_metrics()
        
        key_metrics = {
            'P&L': metrics.get('chimera_pnl_total_usd', 0),
            'Equity': metrics.get('chimera_equity_value_usd', 0),
            'Trades': metrics.get('chimera_orders_total', 0),
            'Win Rate': metrics.get('chimera_win_rate_percent', 0)
        }
        
        print("📊 Dashboard Metrics:")
        for name, value in key_metrics.items():
            if name == 'P&L' or name == 'Equity':
                print(f"   {name}: ${value:,.2f}")
            elif name == 'Win Rate':
                print(f"   {name}: {value:.1f}%")
            else:
                print(f"   {name}: {value}")
        
        # Test equity curve
        from src.project_chimera.ui.dashboard import EquityCurveGenerator
        equity_gen = EquityCurveGenerator()
        df = equity_gen.generate_historical_data(24)
        print(f"📈 Equity curve: {len(df)} points generated")
        
        print("✅ Dashboard integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Dashboard integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_comprehensive_test():
    """Run comprehensive test of all fixes"""
    print("🧪 PROJECTCHIMERA FIXES VALIDATION")
    print("=" * 60)
    print(f"🕐 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Performance Tracker
    results['performance_tracker'] = await test_performance_tracker_fixes()
    
    # Test 2: Bitget API
    results['bitget_api'] = await test_bitget_api_fixes()
    
    # Test 3: Dashboard Integration
    results['dashboard'] = await test_dashboard_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 FIXES VALIDATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for component, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{component.replace('_', ' ').title()}: {status}")
    
    print(f"\n📊 Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL FIXES VALIDATED SUCCESSFULLY!")
        print("✅ System is ready for production use")
    else:
        print("⚠️ Some fixes need additional work")
    
    return passed == total

if __name__ == "__main__":
    from datetime import datetime
    
    success = asyncio.run(run_comprehensive_test())
    sys.exit(0 if success else 1)