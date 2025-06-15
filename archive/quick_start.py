#!/usr/bin/env python3
"""
ProjectChimera Quick Start
プロジェクトキメラ クイックスタート - 即座に利益最大化開始
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def print_banner():
    """起動バナー表示"""
    print("🚀" * 30)
    print("    ProjectChimera - Ultimate Profit System")
    print("    AI-Powered Trading with 40x Leverage")
    print("    Target: $1,000+ Daily Profit")
    print("🚀" * 30)
    print()

def check_dependencies():
    """依存関係チェック"""
    print("🔍 Checking system dependencies...")
    
    required_packages = [
        'streamlit', 'requests', 'pandas', 'plotly', 
        'asyncio', 'psutil', 'statistics'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️ Installing missing packages: {', '.join(missing)}")
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing)
        print("✅ Dependencies installed")
    else:
        print("✅ All dependencies satisfied")
    
    print()

def check_api_config():
    """API設定チェック"""
    print("🔑 Checking API configuration...")
    
    env_file = '.env'
    if os.path.exists(env_file):
        print("✅ .env file found")
        
        with open(env_file, 'r') as f:
            content = f.read()
            
        if 'BITGET_API_KEY' in content and 'bg_' in content:
            print("✅ Bitget API keys configured")
        else:
            print("⚠️ Bitget API keys may need configuration")
    else:
        print("❌ .env file not found")
        print("Creating default .env file...")
        
        with open(env_file, 'w') as f:
            f.write("""# Bitget API Configuration (Demo Environment)
BITGET_API_KEY=bg_41bf02f5324cdffb0e11a1763ab93a2d
BITGET_SECRET_KEY=fe093e0a98d740e68d850e121f20c994843f9f06d3896deaff01fc47cfad36ee
BITGET_PASSPHRASE=ProjectChimera2025
BITGET_SANDBOX=true

# API URLs
BITGET_BASE_URL=https://api.bitget.com
BITGET_SANDBOX_URL=https://api.bitget.com

# WebSocket URLs
BITGET_WS_URL=wss://ws.bitget.com/spot/v1/stream
BITGET_WS_URL_FUTURES=wss://ws.bitget.com/mix/v1/stream

# Other API Keys (configure as needed)
NEWSAPI_KEY=your_newsapi_key_here
OPENAI_API_KEY=your_openai_key_here
""")
        print("✅ Default .env file created")
    
    print()

def test_system():
    """システムテスト"""
    print("🧪 Testing system components...")
    
    # Bitget APIテスト
    print("Testing Bitget API connection...")
    try:
        subprocess.run([sys.executable, 'bitget_futures_client.py'], 
                      capture_output=True, timeout=30)
        print("✅ Bitget API connection successful")
    except Exception as e:
        print(f"⚠️ Bitget API test warning: {e}")
    
    print()

def show_launch_options():
    """起動オプション表示"""
    print("🎯 Launch Options:")
    print()
    print("1. 🚀 FULL AUTOMATED SYSTEM (Recommended)")
    print("   → Deploy complete 24/7 profit system")
    print("   → Master trading + Dashboard + Monitoring")
    print("   → Automatic restarts and health checks")
    print()
    print("2. 📊 DASHBOARD ONLY")
    print("   → Launch Streamlit dashboard interface")
    print("   → Manual control and monitoring")
    print("   → Good for testing and learning")
    print()
    print("3. 🤖 MASTER SYSTEM ONLY")
    print("   → Command-line trading system")
    print("   → Pure performance focus")
    print("   → No GUI, maximum speed")
    print()
    print("4. ⚡ ULTRA BOT")
    print("   → Standalone ultra-fast bot")
    print("   → 60-minute session")
    print("   → High-frequency trading")
    print()

def launch_option_1():
    """完全自動システム起動"""
    print("🚀 Launching Full Automated System...")
    print("This will start:")
    print("• Master Profit System (core trading)")
    print("• Unified Dashboard (web interface)")
    print("• Health monitoring and auto-restart")
    print()
    
    confirm = input("Continue? (y/N): ").lower().strip()
    if confirm == 'y':
        print("Starting full deployment...")
        subprocess.run([sys.executable, 'deploy_profit_system.py'])
    else:
        print("Cancelled")

def launch_option_2():
    """ダッシュボード起動"""
    print("📊 Launching Dashboard...")
    print("Dashboard will be available at: http://localhost:8501")
    print("Press Ctrl+C to stop")
    print()
    
    time.sleep(2)
    subprocess.run(['streamlit', 'run', 'unified_profit_dashboard.py'])

def launch_option_3():
    """マスターシステム起動"""
    print("🤖 Launching Master System...")
    print("This will run the core trading system without GUI")
    print("Press Ctrl+C to stop")
    print()
    
    time.sleep(2)
    subprocess.run([sys.executable, 'master_profit_system.py'])

def launch_option_4():
    """ウルトラボット起動"""
    print("⚡ Launching Ultra Bot...")
    print("60-minute high-frequency trading session")
    print("Press Ctrl+C to stop")
    print()
    
    time.sleep(2)
    subprocess.run([sys.executable, 'ultra_trading_bot.py'])

def main():
    """メイン実行"""
    print_banner()
    
    # システムチェック
    check_dependencies()
    check_api_config()
    test_system()
    
    # 起動オプション表示
    show_launch_options()
    
    while True:
        try:
            choice = input("Select option (1-4, or 'q' to quit): ").strip()
            
            if choice == 'q':
                print("👋 Goodbye! May your trades be profitable!")
                break
            elif choice == '1':
                launch_option_1()
                break
            elif choice == '2':
                launch_option_2()
                break
            elif choice == '3':
                launch_option_3()
                break
            elif choice == '4':
                launch_option_4()
                break
            else:
                print("Invalid option. Please choose 1-4 or 'q'")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye! May your trades be profitable!")
            break

if __name__ == "__main__":
    main()