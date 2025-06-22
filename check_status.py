#!/usr/bin/env python3
"""
ProjectChimera System Status Checker
Quick health check for all system components
"""

import asyncio
import sys
import requests
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


async def check_system_status():
    """Comprehensive system status check"""
    print("🔍 ProjectChimera System Status Check")
    print("=" * 50)
    print(f"🕐 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1. Check Performance Tracker
    try:
        from src.project_chimera.monitor.strategy_performance import get_performance_tracker
        tracker = get_performance_tracker()
        summary = tracker.get_performance_summary()
        
        print("📊 PERFORMANCE TRACKER:")
        print(f"   ✅ Status: Running")
        print(f"   📈 Strategies: {summary['total_strategies']}")
        print(f"   📊 Total Trades: {summary['total_trades']}")
        print(f"   💰 Total P&L: ${summary['total_pnl_usd']:.2f}")
        print(f"   📊 Avg Win Rate: {summary['average_win_rate']:.1f}%")
        
        if summary['best_strategy']:
            print(f"   🏆 Best Strategy: {summary['best_strategy']}")
        
    except Exception as e:
        print("📊 PERFORMANCE TRACKER:")
        print(f"   ❌ Error: {str(e)[:60]}")
    
    print()
    
    # 2. Check Bitget API
    try:
        from src.project_chimera.api.bitget_client import BitgetMarketDataService
        service = BitgetMarketDataService()
        
        # Test basic connection
        overview = await service.get_market_overview(['BTCUSDT'])
        
        print("📡 BITGET API:")
        print(f"   ✅ Status: Connected (Demo Mode)")
        print(f"   📈 Symbols Available: {len(overview.get('tickers', {}))}")
        print(f"   🌐 System Status: {overview.get('system_status', {}).get('status', 'Unknown')}")
        
        await service.close()
        
    except Exception as e:
        print("📡 BITGET API:")
        print(f"   ⚠️ Demo Mode: {str(e)[:60]}")
    
    print()
    
    # 3. Check Web Dashboards
    dashboards = [
        ("Main Dashboard", "http://localhost:8501"),
        ("Strategy Dashboard", "http://localhost:8502")
    ]
    
    print("🌐 WEB DASHBOARDS:")
    for name, url in dashboards:
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                print(f"   ✅ {name}: Running ({url})")
            else:
                print(f"   ⚠️ {name}: HTTP {response.status_code} ({url})")
        except requests.exceptions.ConnectionError:
            print(f"   ❌ {name}: Not running ({url})")
        except Exception as e:
            print(f"   ❌ {name}: Error - {str(e)[:30]}")
    
    print()
    
    # 4. Check Database
    try:
        import sqlite3
        db_path = "data/strategy_performance.db"
        
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Count trades
            cursor.execute("SELECT COUNT(*) FROM trade_records")
            trade_count = cursor.fetchone()[0]
            
            # Count strategies  
            cursor.execute("SELECT COUNT(DISTINCT strategy_id) FROM trade_records")
            strategy_count = cursor.fetchone()[0]
            
            print("🗄️ DATABASE:")
            print(f"   ✅ Status: Connected")
            print(f"   📊 Database: {db_path}")
            print(f"   📈 Trade Records: {trade_count}")
            print(f"   🎯 Unique Strategies: {strategy_count}")
            
    except Exception as e:
        print("🗄️ DATABASE:")
        print(f"   ❌ Error: {str(e)[:60]}")
    
    print()
    
    # 5. Overall Status Summary
    print("🎯 OVERALL SYSTEM STATUS:")
    try:
        # Quick health scoring
        health_score = 0
        
        # Check if we have strategies and trades
        if summary['total_strategies'] > 0:
            health_score += 25
        if summary['total_trades'] > 0:
            health_score += 25
            
        # Check dashboards
        for name, url in dashboards:
            try:
                response = requests.get(url, timeout=2)
                if response.status_code == 200:
                    health_score += 25
            except:
                pass
        
        if health_score >= 75:
            print(f"   ✅ HEALTHY ({health_score}%)")
            print("   🚀 System is ready for trading")
        elif health_score >= 50:
            print(f"   ⚠️ DEGRADED ({health_score}%)")
            print("   🔧 Some components need attention")
        else:
            print(f"   ❌ CRITICAL ({health_score}%)")
            print("   🚨 System requires immediate attention")
            
    except Exception:
        print("   ❓ UNKNOWN - Unable to determine status")
    
    print()
    print("💡 QUICK ACTIONS:")
    print("   🚀 Start System: python start_system.py")
    print("   📊 Generate Data: python demo_simple.py")
    print("   🌐 Access UI: http://localhost:8501")
    print("   📈 Strategy UI: http://localhost:8502")


if __name__ == "__main__":
    asyncio.run(check_system_status())