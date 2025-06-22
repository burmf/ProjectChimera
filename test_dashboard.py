#!/usr/bin/env python3
"""
Simple Dashboard Test Script
Tests dashboard components without Streamlit
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test all imports work correctly"""
    print("🔍 Testing imports...")
    
    try:
        from src.project_chimera.monitor.strategy_performance import get_performance_tracker
        print("✅ Performance tracker import successful")
    except Exception as e:
        print(f"❌ Performance tracker import failed: {e}")
        return False
    
    try:
        from src.project_chimera.api.bitget_client import get_bitget_service
        print("✅ Bitget API client import successful")
    except Exception as e:
        print(f"❌ Bitget API client import failed: {e}")
        return False
    
    return True

def test_performance_tracker():
    """Test performance tracker functionality"""
    print("\n📊 Testing performance tracker...")
    
    try:
        from src.project_chimera.monitor.strategy_performance import get_performance_tracker
        tracker = get_performance_tracker()
        
        summary = tracker.get_performance_summary()
        print(f"✅ Performance summary: {summary['total_strategies']} strategies, {summary['total_trades']} trades")
        
        all_stats = tracker.get_all_strategy_stats()
        print(f"✅ Strategy stats loaded: {len(all_stats)} strategies")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance tracker test failed: {e}")
        return False

def test_dashboard_components():
    """Test dashboard components without Streamlit"""
    print("\n🖥️ Testing dashboard components...")
    
    try:
        # Import dashboard classes directly
        sys.path.insert(0, str(project_root / "src" / "project_chimera" / "ui"))
        
        # Test dashboard with mock Streamlit
        from src.project_chimera.ui.dashboard import TradingSystemAPI, EquityCurveGenerator
        
        api = TradingSystemAPI()
        health = api.get_health()
        metrics = api.get_metrics()
        
        print(f"✅ API health check: {health.get('status', 'unknown')}")
        print(f"✅ Metrics loaded: {len(metrics)} metrics")
        
        equity_gen = EquityCurveGenerator()
        df = equity_gen.generate_historical_data(24)
        print(f"✅ Equity curve generated: {len(df)} data points")
        
        return True
        
    except Exception as e:
        print(f"❌ Dashboard component test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 ProjectChimera Dashboard Test Suite")
    print("=" * 50)
    
    all_passed = True
    
    # Test 1: Imports
    if not test_imports():
        all_passed = False
    
    # Test 2: Performance Tracker
    if not test_performance_tracker():
        all_passed = False
    
    # Test 3: Dashboard Components
    if not test_dashboard_components():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ ALL TESTS PASSED - Dashboard components are working!")
        print("\n💡 You can now run:")
        print("   python start_system.py")
        print("   - or -")
        print("   streamlit run run_dashboard.py main --server.port 8501")
        print("   streamlit run run_dashboard.py strategy --server.port 8502")
    else:
        print("❌ SOME TESTS FAILED - Check errors above")
        print("\n🔧 Try fixing import issues first")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)