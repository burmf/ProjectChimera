#!/usr/bin/env python3
"""
Streamlit Dashboard Runner for ProjectChimera
Handles import path issues and runs dashboards correctly
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_dashboard.py <dashboard_type>")
        print("dashboard_type: main | strategy")
        sys.exit(1)
    
    dashboard_type = sys.argv[1]
    
    if dashboard_type == "main":
        from src.project_chimera.ui.dashboard import main_dashboard
        print("🚀 Starting Main Dashboard...")
        try:
            main_dashboard()
        except KeyboardInterrupt:
            print("\n⏹️ Main Dashboard stopped")
    
    elif dashboard_type == "strategy":
        from src.project_chimera.ui.strategy_dashboard import strategy_dashboard
        print("📊 Starting Strategy Dashboard...")
        try:
            strategy_dashboard()
        except KeyboardInterrupt:
            print("\n⏹️ Strategy Dashboard stopped")
    
    else:
        print(f"Unknown dashboard type: {dashboard_type}")
        print("Use 'main' or 'strategy'")
        sys.exit(1)

if __name__ == "__main__":
    main()