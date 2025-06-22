#!/usr/bin/env python3
"""
パフォーマンスダッシュボード起動スクリプト
"""

import sys
import os
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

# Streamlitアプリケーション起動
if __name__ == "__main__":
    import streamlit.web.cli as stcli
    
    # ダッシュボードファイルのパス
    dashboard_path = project_root / "src" / "project_chimera" / "ui" / "performance_dashboard.py"
    
    # Streamlit引数設定
    sys.argv = [
        "streamlit",
        "run",
        str(dashboard_path),
        "--server.port=8502",
        "--server.address=0.0.0.0",
        "--server.headless=true",
        "--browser.gatherUsageStats=false"
    ]
    
    print("🚀 Starting ProjectChimera Performance Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8502")
    print("🔄 Auto-refresh enabled for real-time monitoring")
    
    stcli.main()