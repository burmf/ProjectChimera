#!/usr/bin/env python3
"""
ProjectChimera System Startup Script
Starts the complete trading system with real data integration
"""

import asyncio
import subprocess
import sys
import os
import time
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.project_chimera.api.bitget_client import demo_bitget_data


def start_streamlit_dashboard(port: int, dashboard_type: str, name: str):
    """Start a Streamlit dashboard in background"""
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            "run_dashboard.py", 
            "--server.port", str(port),
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false",
            "--server.runOnSave", "false",
            "--", dashboard_type  # Pass dashboard type as argument
        ]
        
        print(f"🚀 Starting {name} on port {port}...")
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            cwd=str(project_root)
        )
        
        # Give it a moment to start
        time.sleep(3)
        
        if process.poll() is None:
            print(f"✅ {name} started successfully on http://localhost:{port}")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Failed to start {name}:")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting {name}: {e}")
        return None


async def test_system_components():
    """Test all system components"""
    print("🔍 Testing system components...")
    
    try:
        # Test Bitget API
        print("📡 Testing Bitget API connection...")
        bitget_ok = await demo_bitget_data()
        
        if bitget_ok:
            print("✅ Bitget API connection successful")
        else:
            print("⚠️ Bitget API in demo mode (no real API keys)")
        
        # Test Performance Tracker
        print("📊 Testing Performance Tracker...")
        from src.project_chimera.monitor.strategy_performance import get_performance_tracker
        tracker = get_performance_tracker()
        summary = tracker.get_performance_summary()
        print(f"✅ Performance Tracker: {summary['total_strategies']} strategies, {summary['total_trades']} trades")
        
        return True
        
    except Exception as e:
        print(f"❌ System component test failed: {e}")
        return False


def main():
    """Main startup function"""
    print("=" * 60)
    print("🚀 PROJECT CHIMERA SYSTEM STARTUP")
    print("=" * 60)
    
    # Test system components first
    print("\n1️⃣ Testing System Components...")
    component_test = asyncio.run(test_system_components())
    
    if not component_test:
        print("❌ System component tests failed. Check configuration.")
        return
    
    print("\n2️⃣ Starting Web Dashboards...")
    
    # Start main dashboard
    main_dashboard = start_streamlit_dashboard(
        8501, 
        "main",
        "Main Control Center"
    )
    
    # Start strategy dashboard  
    strategy_dashboard = start_streamlit_dashboard(
        8502,
        "strategy", 
        "Strategy Performance Dashboard"
    )
    
    if main_dashboard and strategy_dashboard:
        print("\n" + "=" * 60)
        print("✅ SYSTEM STARTUP COMPLETE")
        print("=" * 60)
        print("🌐 Access URLs:")
        print("   📊 Main Dashboard: http://localhost:8501")
        print("   📈 Strategy Dashboard: http://localhost:8502")
        print("   🌍 Global IP: http://13.239.98.253:8501 & http://13.239.98.253:8502")
        print("\n💡 System Features:")
        print("   ✅ Real-time performance tracking")
        print("   ✅ Strategy-level P&L monitoring")
        print("   ✅ Bitget market data integration (demo mode)")
        print("   ✅ Comprehensive trade lifecycle tracking")
        print("\n🔄 To generate sample data, run:")
        print("   python demo_simple.py")
        print("\n⏹️ Press Ctrl+C to stop all services")
        
        try:
            # Keep the script running
            while True:
                time.sleep(30)
                
                # Health check
                if main_dashboard.poll() is not None:
                    print("⚠️ Main dashboard stopped")
                if strategy_dashboard.poll() is not None:
                    print("⚠️ Strategy dashboard stopped")
                    
        except KeyboardInterrupt:
            print("\n🛑 Shutting down system...")
            
            if main_dashboard:
                main_dashboard.terminate()
                print("✅ Main dashboard stopped")
                
            if strategy_dashboard:
                strategy_dashboard.terminate()
                print("✅ Strategy dashboard stopped")
                
            print("👋 ProjectChimera shutdown complete")
    
    else:
        print("\n❌ Failed to start one or more dashboards")
        print("💡 Try running individual dashboards manually:")
        print("   streamlit run src/project_chimera/ui/dashboard.py --server.port 8501")
        print("   streamlit run src/project_chimera/ui/strategy_dashboard.py --server.port 8502")


if __name__ == "__main__":
    main()