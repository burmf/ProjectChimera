#!/usr/bin/env python3
"""
Start Control Server for ProjectChimera Trading System
Provides HTTP API for start/stop/health operations
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.project_chimera.api.control_server import run_control_server

def main():
    print("🚀 Starting ProjectChimera Control Server...")
    print("📡 Available endpoints:")
    print("  - http://localhost:8080/health")
    print("  - http://localhost:8080/metrics") 
    print("  - http://localhost:8080/start (POST)")
    print("  - http://localhost:8080/stop (POST)")
    print("  - http://localhost:8080/status")
    print()
    
    try:
        run_control_server(host="0.0.0.0", port=8080)
    except KeyboardInterrupt:
        print("\n⏹️ Control server stopped")
    except Exception as e:
        print(f"❌ Error starting control server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()