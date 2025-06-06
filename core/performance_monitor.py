# core/performance_monitor.py
import time
import psutil
import asyncio
import logging
import datetime
from typing import Dict, List, Any, Optional
import sys
import os

# Add core modules to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.redis_manager import redis_manager
from core.database_adapter import db_adapter
from core.logging_config import get_trading_logger

logger = logging.getLogger(__name__)
trading_logger = get_trading_logger("performance")

class PerformanceMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.monitoring_interval = 30  # Monitor every 30 seconds
        
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            net_io = psutil.net_io_counters()
            
            # Process info
            process = psutil.Process()
            process_memory = process.memory_info()
            
            return {
                'timestamp': datetime.datetime.now().isoformat(),
                'cpu': {
                    'percent': cpu_percent,
                    'count': psutil.cpu_count()
                },
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent,
                    'used': memory.used
                },
                'disk': {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': (disk.used / disk.total) * 100
                },
                'network': {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv
                },
                'process': {
                    'memory_rss': process_memory.rss,
                    'memory_vms': process_memory.vms,
                    'cpu_percent': process.cpu_percent(),
                    'num_threads': process.num_threads(),
                    'uptime': time.time() - self.start_time
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {}
    
    def get_redis_metrics(self) -> Dict[str, Any]:
        """Get Redis performance metrics"""
        try:
            if not redis_manager.is_connected():
                return {'connected': False}
            
            redis_info = redis_manager.get_stats()
            
            # Get stream lengths
            stream_stats = {}
            for stream_name in ['prices', 'news', 'ai_decisions', 'trade_signals', 'system_logs']:
                try:
                    info = redis_manager.get_stream_info(stream_name)
                    stream_stats[stream_name] = info.get('length', 0)
                except:
                    stream_stats[stream_name] = 0
            
            return {
                'connected': True,
                'used_memory': redis_info.get('used_memory_human', 'N/A'),
                'connected_clients': redis_info.get('connected_clients', 0),
                'commands_processed': redis_info.get('total_commands_processed', 0),
                'uptime_seconds': redis_info.get('uptime_in_seconds', 0),
                'stream_lengths': stream_stats,
                'keyspace_hits': redis_info.get('keyspace_hits', 0),
                'keyspace_misses': redis_info.get('keyspace_misses', 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get Redis metrics: {e}")
            return {'connected': False, 'error': str(e)}
    
    def get_database_metrics(self) -> Dict[str, Any]:
        """Get database performance metrics"""
        try:
            if not db_adapter.is_connected():
                return {'connected': False}
            
            metrics = {
                'connected': True,
                'database_type': db_adapter.db_type
            }
            
            # Get table sizes
            table_prefix = db_adapter.get_table_prefix()
            tables = ['price_data', 'news_articles', 'ai_trade_decisions', 'openai_api_usage']
            
            for table in tables:
                try:
                    if db_adapter.db_type == 'postgresql':
                        query = f"SELECT COUNT(*) as count FROM {table_prefix}{table}"
                    else:
                        query = f"SELECT COUNT(*) as count FROM {table}"
                    
                    result = db_adapter.execute_query(query)
                    if result is not None and not result.empty:
                        metrics[f'{table}_count'] = int(result['count'].iloc[0])
                    else:
                        metrics[f'{table}_count'] = 0
                except:
                    metrics[f'{table}_count'] = 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get database metrics: {e}")
            return {'connected': False, 'error': str(e)}
    
    def get_trading_metrics(self) -> Dict[str, Any]:
        """Get trading performance metrics"""
        try:
            metrics = {}
            
            # Get recent trades
            trades_df = db_adapter.execute_query(
                f"SELECT * FROM {db_adapter.get_table_prefix()}trade_history "
                f"WHERE entry_time >= date('now', '-7 days') "
                f"ORDER BY entry_time DESC"
            )
            
            if trades_df is not None and not trades_df.empty:
                completed_trades = trades_df[trades_df['exit_time'].notna()]
                
                if not completed_trades.empty:
                    metrics.update({
                        'total_trades_7d': len(completed_trades),
                        'winning_trades': len(completed_trades[completed_trades['pnl_currency'] > 0]),
                        'losing_trades': len(completed_trades[completed_trades['pnl_currency'] < 0]),
                        'total_pnl_7d': float(completed_trades['pnl_currency'].sum()),
                        'avg_pnl_per_trade': float(completed_trades['pnl_currency'].mean()),
                        'max_win': float(completed_trades['pnl_currency'].max()),
                        'max_loss': float(completed_trades['pnl_currency'].min()),
                        'win_rate': len(completed_trades[completed_trades['pnl_currency'] > 0]) / len(completed_trades) * 100
                    })
                else:
                    metrics = {
                        'total_trades_7d': 0,
                        'winning_trades': 0,
                        'losing_trades': 0,
                        'total_pnl_7d': 0.0,
                        'win_rate': 0.0
                    }
            else:
                metrics = {
                    'total_trades_7d': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'total_pnl_7d': 0.0,
                    'win_rate': 0.0
                }
            
            # Get AI API usage
            usage_df = db_adapter.execute_query(
                f"SELECT SUM(estimated_cost_usd) as total_cost, COUNT(*) as requests "
                f"FROM {db_adapter.get_table_prefix()}openai_api_usage "
                f"WHERE timestamp >= date('now', '-7 days')"
            )
            
            if usage_df is not None and not usage_df.empty:
                metrics['ai_cost_7d'] = float(usage_df['total_cost'].iloc[0] or 0)
                metrics['ai_requests_7d'] = int(usage_df['requests'].iloc[0] or 0)
            else:
                metrics['ai_cost_7d'] = 0.0
                metrics['ai_requests_7d'] = 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get trading metrics: {e}")
            return {}
    
    def get_application_metrics(self) -> Dict[str, Any]:
        """Get application-specific metrics"""
        try:
            metrics = {
                'timestamp': datetime.datetime.now().isoformat(),
                'uptime_seconds': time.time() - self.start_time
            }
            
            # Get error rates from logs
            try:
                error_logs = redis_manager.read_stream('error_tracking', count=100)
                recent_errors = []
                cutoff = datetime.datetime.now() - datetime.timedelta(hours=1)
                
                for log in error_logs:
                    log_time = datetime.datetime.fromisoformat(log['data']['timestamp'])
                    if log_time >= cutoff:
                        recent_errors.append(log['data'])
                
                metrics['errors_last_hour'] = len(recent_errors)
                
                if recent_errors:
                    error_types = {}
                    for error in recent_errors:
                        error_type = error.get('error_type', 'unknown')
                        error_types[error_type] = error_types.get(error_type, 0) + 1
                    metrics['most_common_error'] = max(error_types.items(), key=lambda x: x[1])[0]
                else:
                    metrics['most_common_error'] = None
                    
            except:
                metrics['errors_last_hour'] = 0
                metrics['most_common_error'] = None
            
            # Get signal processing metrics
            try:
                signals = redis_manager.read_stream('trade_signals', count=50)
                approved = redis_manager.read_stream('approved_trades', count=50)
                rejected = redis_manager.read_stream('rejected_trades', count=50)
                
                metrics['signals_generated'] = len(signals)
                metrics['signals_approved'] = len(approved)
                metrics['signals_rejected'] = len(rejected)
                metrics['approval_rate'] = (len(approved) / len(signals) * 100) if signals else 0
                
            except:
                metrics['signals_generated'] = 0
                metrics['signals_approved'] = 0
                metrics['signals_rejected'] = 0
                metrics['approval_rate'] = 0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get application metrics: {e}")
            return {}
    
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect comprehensive performance metrics"""
        return {
            'collection_timestamp': datetime.datetime.now().isoformat(),
            'system': self.get_system_metrics(),
            'redis': self.get_redis_metrics(),
            'database': self.get_database_metrics(),
            'trading': self.get_trading_metrics(),
            'application': self.get_application_metrics()
        }
    
    def store_metrics(self, metrics: Dict[str, Any]):
        """Store metrics in Redis for monitoring"""
        try:
            # Store current metrics
            redis_manager.set_cache('current_metrics', metrics, ttl=300)
            
            # Add to metrics stream for historical tracking
            redis_manager.add_to_stream('performance_metrics', metrics, maxlen=1000)
            
            # Log key metrics
            system_metrics = metrics.get('system', {})
            if system_metrics:
                trading_logger.log_performance_metric({
                    'name': 'cpu_percent',
                    'value': system_metrics.get('cpu', {}).get('percent', 0),
                    'unit': 'percentage'
                })
                
                trading_logger.log_performance_metric({
                    'name': 'memory_percent',
                    'value': system_metrics.get('memory', {}).get('percent', 0),
                    'unit': 'percentage'
                })
            
            # Check for performance alerts
            self.check_performance_alerts(metrics)
            
        except Exception as e:
            logger.error(f"Failed to store metrics: {e}")
    
    def check_performance_alerts(self, metrics: Dict[str, Any]):
        """Check metrics for performance alerts"""
        try:
            alerts = []
            
            # System alerts
            system = metrics.get('system', {})
            if system:
                cpu_percent = system.get('cpu', {}).get('percent', 0)
                memory_percent = system.get('memory', {}).get('percent', 0)
                disk_percent = system.get('disk', {}).get('percent', 0)
                
                if cpu_percent > 80:
                    alerts.append({'type': 'high_cpu', 'value': cpu_percent, 'threshold': 80})
                
                if memory_percent > 85:
                    alerts.append({'type': 'high_memory', 'value': memory_percent, 'threshold': 85})
                
                if disk_percent > 90:
                    alerts.append({'type': 'high_disk', 'value': disk_percent, 'threshold': 90})
            
            # Application alerts
            app_metrics = metrics.get('application', {})
            if app_metrics:
                errors = app_metrics.get('errors_last_hour', 0)
                if errors > 10:
                    alerts.append({'type': 'high_error_rate', 'value': errors, 'threshold': 10})
            
            # Trading alerts
            trading = metrics.get('trading', {})
            if trading:
                win_rate = trading.get('win_rate', 0)
                if win_rate < 30 and trading.get('total_trades_7d', 0) > 10:
                    alerts.append({'type': 'low_win_rate', 'value': win_rate, 'threshold': 30})
            
            # Send alerts
            for alert in alerts:
                alert_data = {
                    'alert_type': alert['type'],
                    'current_value': alert['value'],
                    'threshold': alert['threshold'],
                    'timestamp': datetime.datetime.now().isoformat(),
                    'severity': 'warning'
                }
                
                redis_manager.add_to_stream('performance_alerts', alert_data)
                logger.warning(f"Performance alert: {alert['type']} = {alert['value']} (threshold: {alert['threshold']})")
                
        except Exception as e:
            logger.error(f"Failed to check performance alerts: {e}")
    
    async def start_monitoring(self):
        """Start continuous performance monitoring"""
        logger.info("Starting performance monitoring...")
        
        while True:
            try:
                metrics = self.collect_all_metrics()
                self.store_metrics(metrics)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the last N hours"""
        try:
            # Get recent metrics from stream
            metrics_stream = redis_manager.read_stream('performance_metrics', count=100)
            
            if not metrics_stream:
                return {'error': 'No metrics available'}
            
            # Filter by time window
            cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
            recent_metrics = []
            
            for metric in metrics_stream:
                metric_time = datetime.datetime.fromisoformat(metric['data']['collection_timestamp'])
                if metric_time >= cutoff_time:
                    recent_metrics.append(metric['data'])
            
            if not recent_metrics:
                return {'error': 'No recent metrics available'}
            
            # Calculate averages
            cpu_values = []
            memory_values = []
            error_counts = []
            
            for metric in recent_metrics:
                system = metric.get('system', {})
                if system:
                    cpu_values.append(system.get('cpu', {}).get('percent', 0))
                    memory_values.append(system.get('memory', {}).get('percent', 0))
                
                app = metric.get('application', {})
                if app:
                    error_counts.append(app.get('errors_last_hour', 0))
            
            return {
                'time_window_hours': hours,
                'metrics_count': len(recent_metrics),
                'system_performance': {
                    'avg_cpu_percent': sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                    'max_cpu_percent': max(cpu_values) if cpu_values else 0,
                    'avg_memory_percent': sum(memory_values) / len(memory_values) if memory_values else 0,
                    'max_memory_percent': max(memory_values) if memory_values else 0
                },
                'application_health': {
                    'total_errors': sum(error_counts),
                    'avg_errors_per_hour': sum(error_counts) / len(error_counts) if error_counts else 0,
                    'max_errors_per_hour': max(error_counts) if error_counts else 0
                },
                'latest_metrics': recent_metrics[-1] if recent_metrics else None
            }
            
        except Exception as e:
            logger.error(f"Failed to generate performance summary: {e}")
            return {'error': str(e)}

# Global performance monitor instance
performance_monitor = PerformanceMonitor()