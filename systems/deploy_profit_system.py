#!/usr/bin/env python3
"""
Automated 24/7 Profit System Deployment
自動24時間利益システム展開
"""

import subprocess
import time
import json
import logging
import os
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
import psutil

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'deployment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ProfitSystemDeployment:
    """
    24/7利益システム自動展開
    
    - マルチプロセス管理
    - 自動再起動
    - パフォーマンス監視
    - 障害復旧
    """
    
    def __init__(self):
        self.processes = {}
        self.is_running = False
        self.shutdown_requested = False
        
        # 展開設定
        self.components = {
            'master_system': {
                'script': 'master_profit_system.py',
                'description': 'Master Profit Trading System',
                'critical': True,
                'restart_delay': 30,
                'max_restarts': 5
            },
            'dashboard': {
                'script': 'unified_profit_dashboard.py',
                'description': 'Unified Dashboard UI',
                'critical': False,
                'restart_delay': 15,
                'max_restarts': 10
            },
            'ultra_bot': {
                'script': 'ultra_trading_bot.py',
                'description': 'Ultra Trading Bot',
                'critical': True,
                'restart_delay': 20,
                'max_restarts': 3
            }
        }
        
        # 監視設定
        self.health_check_interval = 30  # 30秒ごと
        self.performance_log_interval = 300  # 5分ごと
        
        # 統計
        self.start_time = datetime.now()
        self.restart_counts = {name: 0 for name in self.components.keys()}
        self.health_history = []
        
        logger.info("🚀 Profit System Deployment initialized")
        logger.info(f"Components: {list(self.components.keys())}")
    
    def setup_signal_handlers(self):
        """シグナルハンドラー設定"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown...")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def start_component(self, component_name: str) -> bool:
        """コンポーネント開始"""
        if component_name not in self.components:
            logger.error(f"Unknown component: {component_name}")
            return False
        
        component = self.components[component_name]
        script_path = component['script']
        
        if not os.path.exists(script_path):
            logger.error(f"Script not found: {script_path}")
            return False
        
        try:
            # Streamlitダッシュボードの場合
            if 'dashboard' in component_name:
                cmd = ['streamlit', 'run', script_path, '--server.port=8501', '--server.headless=true']
            else:
                cmd = ['python3', script_path]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            self.processes[component_name] = {
                'process': process,
                'start_time': datetime.now(),
                'restart_count': self.restart_counts[component_name],
                'component': component
            }
            
            logger.info(f"✅ Started {component['description']} (PID: {process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start {component_name}: {e}")
            return False
    
    def stop_component(self, component_name: str) -> bool:
        """コンポーネント停止"""
        if component_name not in self.processes:
            return True
        
        process_info = self.processes[component_name]
        process = process_info['process']
        
        try:
            # 優雅な終了を試行
            process.terminate()
            
            # 5秒待機
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # 強制終了
                process.kill()
                process.wait()
            
            logger.info(f"⏹️ Stopped {component_name}")
            del self.processes[component_name]
            return True
            
        except Exception as e:
            logger.error(f"Error stopping {component_name}: {e}")
            return False
    
    def restart_component(self, component_name: str) -> bool:
        """コンポーネント再起動"""
        component = self.components[component_name]
        
        # 再起動回数チェック
        if self.restart_counts[component_name] >= component['max_restarts']:
            logger.error(f"Max restarts reached for {component_name}")
            return False
        
        logger.info(f"🔄 Restarting {component_name}...")
        
        # 停止
        self.stop_component(component_name)
        
        # 待機
        time.sleep(component['restart_delay'])
        
        # 開始
        success = self.start_component(component_name)
        
        if success:
            self.restart_counts[component_name] += 1
            logger.info(f"✅ {component_name} restarted successfully")
        else:
            logger.error(f"❌ Failed to restart {component_name}")
        
        return success
    
    def check_component_health(self, component_name: str) -> Dict[str, Any]:
        """コンポーネント健康状態チェック"""
        if component_name not in self.processes:
            return {'status': 'stopped', 'healthy': False}
        
        process_info = self.processes[component_name]
        process = process_info['process']
        
        # プロセス状態チェック
        if process.poll() is not None:
            return {'status': 'dead', 'healthy': False, 'exit_code': process.returncode}
        
        try:
            # CPU・メモリ使用率取得
            ps_process = psutil.Process(process.pid)
            cpu_percent = ps_process.cpu_percent()
            memory_info = ps_process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            # 健康状態判定
            healthy = True
            issues = []
            
            # CPU使用率チェック（90%以上で警告）
            if cpu_percent > 90:
                healthy = False
                issues.append(f"High CPU: {cpu_percent:.1f}%")
            
            # メモリ使用率チェック（1GB以上で警告）
            if memory_mb > 1024:
                healthy = False
                issues.append(f"High memory: {memory_mb:.0f}MB")
            
            return {
                'status': 'running',
                'healthy': healthy,
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb,
                'uptime': datetime.now() - process_info['start_time'],
                'issues': issues
            }
            
        except psutil.NoSuchProcess:
            return {'status': 'dead', 'healthy': False}
        except Exception as e:
            return {'status': 'unknown', 'healthy': False, 'error': str(e)}
    
    def health_check_cycle(self):
        """健康状態チェックサイクル"""
        logger.info("🔍 Starting health check cycle...")
        
        while not self.shutdown_requested:
            health_status = {}
            
            for component_name in self.components.keys():
                health = self.check_component_health(component_name)
                health_status[component_name] = health
                
                # 不健康なコンポーネントの処理
                if not health['healthy']:
                    component = self.components[component_name]
                    
                    if health['status'] == 'dead':
                        logger.warning(f"💀 {component_name} is dead, attempting restart...")
                        self.restart_component(component_name)
                    elif health['status'] == 'running' and component['critical']:
                        logger.warning(f"⚠️ {component_name} unhealthy: {health.get('issues', [])}")
                        # クリティカルコンポーネントは再起動
                        self.restart_component(component_name)
            
            # 健康状態ログ
            self.health_history.append({
                'timestamp': datetime.now(),
                'status': health_status
            })
            
            # 履歴サイズ制限
            if len(self.health_history) > 100:
                self.health_history.pop(0)
            
            time.sleep(self.health_check_interval)
    
    def performance_monitor(self):
        """パフォーマンス監視"""
        logger.info("📊 Starting performance monitor...")
        
        while not self.shutdown_requested:
            try:
                # システム全体の統計
                system_stats = {
                    'timestamp': datetime.now().isoformat(),
                    'uptime': (datetime.now() - self.start_time).total_seconds(),
                    'components': {}
                }
                
                for component_name in self.components.keys():
                    health = self.check_component_health(component_name)
                    
                    system_stats['components'][component_name] = {
                        'status': health.get('status', 'unknown'),
                        'healthy': health.get('healthy', False),
                        'cpu_percent': health.get('cpu_percent', 0),
                        'memory_mb': health.get('memory_mb', 0),
                        'restart_count': self.restart_counts[component_name]
                    }
                
                # ログファイルに保存
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                stats_file = f'system_stats_{timestamp}.json'
                
                with open(stats_file, 'w') as f:
                    json.dump(system_stats, f, indent=2)
                
                logger.info(f"📊 Performance stats saved: {stats_file}")
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
            
            time.sleep(self.performance_log_interval)
    
    def deploy_all_systems(self):
        """全システム展開"""
        logger.info("🚀 Starting full system deployment...")
        
        # コンポーネントを順番に開始
        for component_name, component in self.components.items():
            logger.info(f"Starting {component['description']}...")
            
            success = self.start_component(component_name)
            
            if success:
                time.sleep(5)  # 起動待機
            else:
                if component['critical']:
                    logger.error(f"Critical component {component_name} failed to start!")
                    return False
                else:
                    logger.warning(f"Non-critical component {component_name} failed to start")
        
        logger.info("✅ All systems deployed successfully!")
        return True
    
    def shutdown_all_systems(self):
        """全システム停止"""
        logger.info("🛑 Shutting down all systems...")
        
        for component_name in list(self.processes.keys()):
            self.stop_component(component_name)
        
        logger.info("✅ All systems stopped")
    
    def print_status(self):
        """ステータス表示"""
        uptime = datetime.now() - self.start_time
        
        print(f"\n{'='*80}")
        print(f"🚀 PROFIT SYSTEM DEPLOYMENT STATUS | Uptime: {uptime}")
        print(f"{'='*80}")
        
        for component_name, component in self.components.items():
            health = self.check_component_health(component_name)
            status_emoji = "🟢" if health['healthy'] else "🔴"
            
            print(f"{status_emoji} {component['description']}")
            print(f"   Status: {health.get('status', 'unknown').upper()}")
            
            if 'cpu_percent' in health:
                print(f"   CPU: {health['cpu_percent']:.1f}% | Memory: {health['memory_mb']:.0f}MB")
            
            if 'uptime' in health:
                print(f"   Uptime: {health['uptime']}")
            
            print(f"   Restarts: {self.restart_counts[component_name]}")
            
            if health.get('issues'):
                print(f"   Issues: {', '.join(health['issues'])}")
        
        print(f"{'='*80}")
    
    def run_deployment(self):
        """メイン展開実行"""
        self.setup_signal_handlers()
        self.is_running = True
        
        logger.info("🚀 Starting 24/7 Profit System Deployment")
        
        # システム展開
        if not self.deploy_all_systems():
            logger.error("System deployment failed!")
            return
        
        # 監視スレッド開始
        health_thread = threading.Thread(target=self.health_check_cycle)
        performance_thread = threading.Thread(target=self.performance_monitor)
        
        health_thread.daemon = True
        performance_thread.daemon = True
        
        health_thread.start()
        performance_thread.start()
        
        logger.info("✅ All monitoring systems active")
        
        # メインループ
        try:
            last_status_time = datetime.now()
            
            while not self.shutdown_requested:
                # 定期ステータス表示（5分ごと）
                if datetime.now() - last_status_time > timedelta(minutes=5):
                    self.print_status()
                    last_status_time = datetime.now()
                
                time.sleep(10)
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            self.shutdown_all_systems()
            self.is_running = False
            logger.info("🏁 Deployment shutdown complete")


def main():
    """メイン実行"""
    deployment = ProfitSystemDeployment()
    
    print("🚀 ProjectChimera - 24/7 Automated Profit System")
    print("=" * 60)
    print("This will deploy the complete profit maximization system:")
    print("• Master Trading System (40x leverage)")
    print("• Unified Dashboard (Streamlit UI)")
    print("• Ultra Trading Bot (backup system)")
    print("• Continuous health monitoring")
    print("• Automatic restart on failures")
    print("=" * 60)
    
    # 確認
    confirm = input("Deploy 24/7 profit system? (y/N): ").lower().strip()
    
    if confirm == 'y':
        deployment.run_deployment()
    else:
        print("Deployment cancelled")


if __name__ == "__main__":
    main()