#!/usr/bin/env python3
"""
ProjectChimera Unified Launcher
統合ランチャー - 整理されたシステム起動
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# Add core modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'systems'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ui'))

def print_banner():
    """起動バナー"""
    print("🚀" * 25)
    print("  ProjectChimera - Organized Profit System")
    print("  Clean Architecture | Maximum Performance")
    print("  Target: $1,000+ Daily Profit")
    print("🚀" * 25)
    print()

def show_file_structure():
    """ファイル構造表示"""
    print("📁 Organized File Structure:")
    print("├── 🏠 ProjectChimera/")
    print("│   ├── 🔧 core/")
    print("│   │   ├── bitget_futures_client.py     # API Client")
    print("│   │   └── advanced_risk_manager.py     # Risk Management")
    print("│   ├── ⚡ systems/")
    print("│   │   ├── master_profit_system.py      # Main Trading System")
    print("│   │   ├── ultra_trading_bot.py         # High-Frequency Bot")
    print("│   │   └── deploy_profit_system.py      # 24/7 Deployment")
    print("│   ├── 📊 ui/")
    print("│   │   ├── unified_profit_dashboard.py  # Main Dashboard")
    print("│   │   ├── profit_maximizer.py          # Profit Maximizer UI")
    print("│   │   └── scalping_dashboard.py        # Scalping UI")
    print("│   ├── 📖 docs/")
    print("│   │   └── PROFIT_MAXIMIZER_README.md   # Documentation")
    print("│   ├── 📋 logs/")
    print("│   │   └── *.log                        # System Logs")
    print("│   ├── 💾 data/")
    print("│   │   └── *.json                       # Performance Data")
    print("│   ├── 🚀 launch.py                     # This launcher")
    print("│   └── ⚙️ .env                          # Configuration")
    print()

def check_system():
    """システムチェック"""
    print("🔍 System Check:")
    
    # Core modules check
    try:
        from core.bitget_futures_client import BitgetFuturesClient
        print("✅ Core API Client ready")
    except ImportError as e:
        print(f"❌ Core API Client: {e}")
    
    try:
        from core.advanced_risk_manager import AdvancedRiskManager
        print("✅ Risk Manager ready")
    except ImportError as e:
        print(f"❌ Risk Manager: {e}")
    
    # Systems check
    if os.path.exists('systems/master_profit_system.py'):
        print("✅ Master Profit System ready")
    else:
        print("❌ Master Profit System missing")
    
    if os.path.exists('systems/ultra_trading_bot.py'):
        print("✅ Ultra Trading Bot ready")
    else:
        print("❌ Ultra Trading Bot missing")
    
    # UI check
    if os.path.exists('ui/unified_profit_dashboard.py'):
        print("✅ Unified Dashboard ready")
    else:
        print("❌ Unified Dashboard missing")
    
    print()

def show_launch_options():
    """起動オプション表示"""
    print("🎯 Clean Launch Options:")
    print()
    print("1. 🚀 MASTER PROFIT SYSTEM")
    print("   → Core 40x leverage trading system")
    print("   → Target: $1,000+ daily profit")
    print("   → Location: systems/master_profit_system.py")
    print()
    print("2. 📊 UNIFIED DASHBOARD")
    print("   → Web interface at http://localhost:8501")
    print("   → Real-time monitoring and control")
    print("   → Location: ui/unified_profit_dashboard.py")
    print()
    print("3. ⚡ ULTRA TRADING BOT")
    print("   → High-frequency standalone bot")
    print("   → 30x leverage, 60-minute sessions")
    print("   → Location: systems/ultra_trading_bot.py")
    print()
    print("4. 🔧 24/7 DEPLOYMENT")
    print("   → Full automated deployment")
    print("   → Process monitoring and auto-restart")
    print("   → Location: systems/deploy_profit_system.py")
    print()
    print("5. 📁 SHOW FILE STRUCTURE")
    print("   → Display organized directory structure")
    print()

def launch_master_system():
    """Master System起動"""
    print("🚀 Launching Master Profit System...")
    print("Target: $1,000+ daily profit | 40x leverage")
    print("Press Ctrl+C to stop")
    print()
    
    os.chdir('systems')
    subprocess.run([sys.executable, 'master_profit_system.py'])

def launch_dashboard():
    """Dashboard起動"""
    print("📊 Launching Unified Dashboard...")
    print("Access at: http://localhost:8501")
    print("Press Ctrl+C to stop")
    print()
    
    os.chdir('ui')
    subprocess.run(['streamlit', 'run', 'unified_profit_dashboard.py'])

def launch_ultra_bot():
    """Ultra Bot起動"""
    print("⚡ Launching Ultra Trading Bot...")
    print("30x leverage | 60-minute session")
    print("Press Ctrl+C to stop")
    print()
    
    os.chdir('systems')
    subprocess.run([sys.executable, 'ultra_trading_bot.py'])

def launch_deployment():
    """Deployment起動"""
    print("🔧 Launching 24/7 Deployment System...")
    print("Full automated management")
    print("Press Ctrl+C to stop")
    print()
    
    os.chdir('systems')
    subprocess.run([sys.executable, 'deploy_profit_system.py'])

def main():
    """メイン実行"""
    print_banner()
    show_file_structure()
    check_system()
    show_launch_options()
    
    while True:
        try:
            choice = input("Select option (1-5, or 'q' to quit): ").strip()
            
            if choice == 'q':
                print("👋 Clean exit! Happy trading!")
                break
            elif choice == '1':
                launch_master_system()
                break
            elif choice == '2':
                launch_dashboard()
                break
            elif choice == '3':
                launch_ultra_bot()
                break
            elif choice == '4':
                launch_deployment()
                break
            elif choice == '5':
                show_file_structure()
                continue
            else:
                print("Invalid option. Please choose 1-5 or 'q'")
                
        except KeyboardInterrupt:
            print("\n👋 Clean exit! Happy trading!")
            break

if __name__ == "__main__":
    main()