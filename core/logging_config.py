# core/logging_config.py
import logging
import logging.handlers
import os
import sys
import datetime
import json
from typing import Dict, Any
import traceback

from core.redis_manager import redis_manager

class RedisHandler(logging.Handler):
    """Custom logging handler that sends logs to Redis"""
    
    def __init__(self, redis_manager, stream_name='system_logs'):
        super().__init__()
        self.redis_manager = redis_manager
        self.stream_name = stream_name
    
    def emit(self, record):
        try:
            log_entry = {
                'timestamp': datetime.datetime.fromtimestamp(record.created).isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
                'thread': record.thread,
                'process': record.process
            }
            
            # Add exception info if present
            if record.exc_info:
                log_entry['exception'] = {
                    'type': record.exc_info[0].__name__,
                    'message': str(record.exc_info[1]),
                    'traceback': ''.join(traceback.format_exception(*record.exc_info))
                }
            
            # Add extra fields if present
            if hasattr(record, 'extra_fields'):
                log_entry.update(record.extra_fields)
            
            self.redis_manager.add_to_stream(self.stream_name, log_entry, maxlen=10000)
            
        except Exception:
            # Fail silently to avoid recursive logging errors
            pass

class StructuredFormatter(logging.Formatter):
    """Structured JSON formatter for logs"""
    
    def format(self, record):
        log_obj = {
            'timestamp': datetime.datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_obj['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': self.formatException(record.exc_info)
            }
        
        return json.dumps(log_obj, ensure_ascii=False)

def setup_logging(log_level: str = None, log_to_file: bool = True, log_to_redis: bool = True):
    """Setup comprehensive logging configuration"""
    
    # Get log level from environment or default to INFO
    log_level = log_level or os.getenv('LOG_LEVEL', 'INFO')
    log_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logs directory
    logs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler with colored output
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # File handlers
    if log_to_file:
        # General application log
        app_log_file = os.path.join(logs_dir, 'app.log')
        file_handler = logging.handlers.RotatingFileHandler(
            app_log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setFormatter(StructuredFormatter())
        file_handler.setLevel(log_level)
        root_logger.addHandler(file_handler)
        
        # Error log
        error_log_file = os.path.join(logs_dir, 'errors.log')
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file, maxBytes=10*1024*1024, backupCount=5
        )
        error_handler.setFormatter(StructuredFormatter())
        error_handler.setLevel(logging.ERROR)
        root_logger.addHandler(error_handler)
        
        # Trading-specific log
        trading_log_file = os.path.join(logs_dir, 'trading.log')
        trading_handler = logging.handlers.RotatingFileHandler(
            trading_log_file, maxBytes=10*1024*1024, backupCount=10
        )
        trading_handler.setFormatter(StructuredFormatter())
        trading_handler.setLevel(logging.INFO)
        
        # Add filter for trading-related logs
        trading_handler.addFilter(lambda record: any(name in record.name for name in [
            'trading', 'risk', 'signal', 'portfolio', 'realtime'
        ]))
        root_logger.addHandler(trading_handler)
    
    # Redis handler for real-time log monitoring
    if log_to_redis and redis_manager.is_connected():
        redis_handler = RedisHandler(redis_manager)
        redis_handler.setLevel(logging.WARNING)  # Only send warnings and errors to Redis
        root_logger.addHandler(redis_handler)
    
    # Set specific logger levels
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('ccxt').setLevel(logging.WARNING)
    
    logging.info("Logging system initialized")

class TradingLogger:
    """Specialized logger for trading operations"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(f"trading.{name}")
        self.redis_manager = redis_manager
    
    def log_trade_signal(self, signal_data: Dict[str, Any]):
        """Log trade signal generation"""
        try:
            log_data = {
                'event_type': 'trade_signal_generated',
                'pair': signal_data.get('pair'),
                'direction': signal_data.get('direction'),
                'confidence': signal_data.get('confidence'),
                'source': signal_data.get('source', 'unknown'),
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            self.logger.info(f"Trade signal: {signal_data.get('pair')} {signal_data.get('direction')}", 
                           extra={'extra_fields': log_data})
            
            # Store in Redis for monitoring
            self.redis_manager.add_to_stream('trading_events', log_data)
            
        except Exception as e:
            self.logger.error(f"Failed to log trade signal: {e}")
    
    def log_trade_execution(self, execution_data: Dict[str, Any]):
        """Log trade execution"""
        try:
            log_data = {
                'event_type': 'trade_executed',
                'trade_id': execution_data.get('trade_id'),
                'pair': execution_data.get('pair'),
                'direction': execution_data.get('direction'),
                'size': execution_data.get('size'),
                'price': execution_data.get('price'),
                'status': execution_data.get('status'),
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            self.logger.info(f"Trade executed: {execution_data.get('trade_id')}", 
                           extra={'extra_fields': log_data})
            
            self.redis_manager.add_to_stream('trading_events', log_data)
            
        except Exception as e:
            self.logger.error(f"Failed to log trade execution: {e}")
    
    def log_risk_event(self, risk_data: Dict[str, Any]):
        """Log risk management events"""
        try:
            log_data = {
                'event_type': 'risk_event',
                'risk_type': risk_data.get('type'),
                'severity': risk_data.get('severity', 'info'),
                'message': risk_data.get('message'),
                'portfolio_state': risk_data.get('portfolio_state'),
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            if risk_data.get('severity') == 'critical':
                self.logger.critical(f"Risk event: {risk_data.get('message')}", 
                                   extra={'extra_fields': log_data})
            else:
                self.logger.warning(f"Risk event: {risk_data.get('message')}", 
                                  extra={'extra_fields': log_data})
            
            self.redis_manager.add_to_stream('risk_events', log_data)
            
        except Exception as e:
            self.logger.error(f"Failed to log risk event: {e}")
    
    def log_performance_metric(self, metric_data: Dict[str, Any]):
        """Log performance metrics"""
        try:
            log_data = {
                'event_type': 'performance_metric',
                'metric_name': metric_data.get('name'),
                'value': metric_data.get('value'),
                'unit': metric_data.get('unit'),
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            self.logger.info(f"Performance: {metric_data.get('name')} = {metric_data.get('value')}", 
                           extra={'extra_fields': log_data})
            
            self.redis_manager.add_to_stream('performance_metrics', log_data)
            
        except Exception as e:
            self.logger.error(f"Failed to log performance metric: {e}")

def get_trading_logger(name: str) -> TradingLogger:
    """Get a trading-specific logger"""
    return TradingLogger(name)

class ErrorTracker:
    """Track and analyze errors across the system"""
    
    def __init__(self):
        self.redis_manager = redis_manager
        self.logger = logging.getLogger("error_tracker")
    
    def track_error(self, error: Exception, context: Dict[str, Any] = None):
        """Track an error with context"""
        try:
            error_data = {
                'error_type': type(error).__name__,
                'error_message': str(error),
                'traceback': traceback.format_exc(),
                'context': context or {},
                'timestamp': datetime.datetime.now().isoformat(),
                'severity': self._classify_error_severity(error)
            }
            
            self.redis_manager.add_to_stream('error_tracking', error_data)
            
            # Log based on severity
            if error_data['severity'] == 'critical':
                self.logger.critical(f"Critical error: {str(error)}", exc_info=error)
            else:
                self.logger.error(f"Error tracked: {str(error)}", exc_info=error)
            
        except Exception as e:
            # Fail silently to avoid recursive errors
            pass
    
    def _classify_error_severity(self, error: Exception) -> str:
        """Classify error severity"""
        if isinstance(error, (ConnectionError, TimeoutError)):
            return 'high'
        elif isinstance(error, (ValueError, KeyError, TypeError)):
            return 'medium'
        elif isinstance(error, (ImportError, ModuleNotFoundError)):
            return 'critical'
        else:
            return 'low'
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the last N hours"""
        try:
            # Get recent errors from Redis
            errors = self.redis_manager.read_stream('error_tracking', count=1000)
            
            if not errors:
                return {'total_errors': 0, 'error_types': {}, 'severity_breakdown': {}}
            
            # Filter by time window
            cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
            recent_errors = []
            
            for error in errors:
                error_time = datetime.datetime.fromisoformat(error['data']['timestamp'])
                if error_time >= cutoff_time:
                    recent_errors.append(error['data'])
            
            # Analyze errors
            error_types = {}
            severity_breakdown = {}
            
            for error in recent_errors:
                error_type = error.get('error_type', 'unknown')
                severity = error.get('severity', 'unknown')
                
                error_types[error_type] = error_types.get(error_type, 0) + 1
                severity_breakdown[severity] = severity_breakdown.get(severity, 0) + 1
            
            return {
                'total_errors': len(recent_errors),
                'error_types': error_types,
                'severity_breakdown': severity_breakdown,
                'most_common_error': max(error_types.items(), key=lambda x: x[1])[0] if error_types else None,
                'time_window_hours': hours
            }
            
        except Exception as e:
            self.logger.error(f"Failed to generate error summary: {e}")
            return {'error': 'Failed to generate summary'}

# Global instances
error_tracker = ErrorTracker()

# Initialize logging when module is imported
setup_logging()