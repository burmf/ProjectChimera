"""
Structured JSON Logging for ProjectChimera
Professional logging with security, performance monitoring, and observability
"""

import json
import sys
import traceback
from typing import Any, Dict, Optional, Union, List
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import threading
from contextvars import ContextVar
import asyncio
import inspect

from loguru import logger
from ..config import Settings, get_settings


# Context variables for request/session tracking
request_id_var: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
session_id_var: ContextVar[Optional[str]] = ContextVar('session_id', default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar('user_id', default=None)


class LogLevel(Enum):
    """Log levels"""
    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EventType(Enum):
    """Event types for structured logging"""
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    API_REQUEST = "api_request"
    API_RESPONSE = "api_response"
    TRADE_PLACED = "trade_placed"
    TRADE_FILLED = "trade_filled"
    RISK_CHECK = "risk_check"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_METRIC = "performance_metric"
    SECURITY_EVENT = "security_event"
    USER_ACTION = "user_action"
    DATA_COLLECTION = "data_collection"


@dataclass
class LogContext:
    """Log context information"""
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    correlation_id: Optional[str] = None


@dataclass 
class PerformanceMetrics:
    """Performance metrics for logging"""
    duration_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_pct: Optional[float] = None
    response_size_bytes: Optional[int] = None


@dataclass
class SecurityContext:
    """Security context for sensitive operations"""
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    api_key_prefix: Optional[str] = None  # Only first 4 chars
    operation_type: Optional[str] = None


class StructuredLogger:
    """
    Structured JSON logger with security-aware field sanitization
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.log_config = self.settings.logging
        self._setup_logger()
        
        # Sensitive field patterns to sanitize
        self.sensitive_patterns = set(self.log_config.sanitize_keys)
        self.sensitive_patterns.update([
            'password', 'secret', 'key', 'token', 'auth', 'credential',
            'api_key', 'secret_key', 'passphrase', 'private_key'
        ])
    
    def _setup_logger(self):
        """Setup loguru logger with JSON formatting"""
        # Remove default handler
        logger.remove()
        
        # Add JSON formatter for structured logging
        def json_formatter(record):
            """Custom JSON formatter"""
            
            # Base log entry
            log_entry = {
                "timestamp": record["time"].isoformat(),
                "level": record["level"].name,
                "logger": record["name"],
                "message": record["message"],
                "module": record["module"],
                "function": record["function"],
                "line": record["line"]
            }
            
            # Add context information
            context = self._get_context()
            if context:
                log_entry["context"] = asdict(context)
            
            # Add extra fields from record
            if record.get("extra"):
                extra = record["extra"].copy()
                
                # Sanitize sensitive data
                extra = self._sanitize_data(extra)
                
                # Merge extra fields
                log_entry.update(extra)
            
            # Add exception information if present
            if record.get("exception"):
                exc_type, exc_value, exc_traceback = record["exception"]
                log_entry["exception"] = {
                    "type": exc_type.__name__ if exc_type else None,
                    "message": str(exc_value) if exc_value else None,
                    "traceback": traceback.format_exception(
                        exc_type, exc_value, exc_traceback
                    ) if exc_traceback else None
                }
            
            return json.dumps(log_entry, default=str, ensure_ascii=False)
        
        # Add stdout handler for development
        if self.settings.debug:
            logger.add(
                sys.stdout,
                format=json_formatter,
                level=self.log_config.level,
                colorize=False,
                serialize=False
            )
        
        # Add file handler for production
        log_file = Path(self.settings.logs_dir) / "application.log"
        log_file.parent.mkdir(exist_ok=True)
        
        logger.add(
            str(log_file),
            format=json_formatter,
            level=self.log_config.level,
            rotation=self.log_config.rotation,
            retention=self.log_config.retention,
            compression="gz",
            serialize=False
        )
        
        # Add separate error log
        error_log_file = Path(self.settings.logs_dir) / "errors.log"
        logger.add(
            str(error_log_file),
            format=json_formatter,
            level="ERROR",
            rotation=self.log_config.rotation,
            retention=self.log_config.retention,
            compression="gz",
            serialize=False
        )
    
    def _get_context(self) -> Optional[LogContext]:
        """Get current logging context"""
        request_id = request_id_var.get()
        session_id = session_id_var.get()
        user_id = user_id_var.get()
        
        if any([request_id, session_id, user_id]):
            return LogContext(
                request_id=request_id,
                session_id=session_id,
                user_id=user_id
            )
        return None
    
    def _sanitize_data(self, data: Any) -> Any:
        """Recursively sanitize sensitive data"""
        if isinstance(data, dict):
            sanitized = {}
            for key, value in data.items():
                if any(pattern in key.lower() for pattern in self.sensitive_patterns):
                    # Show only first 4 characters for keys
                    if isinstance(value, str) and len(value) > 4:
                        sanitized[key] = f"{value[:4]}***"
                    else:
                        sanitized[key] = "***"
                else:
                    sanitized[key] = self._sanitize_data(value)
            return sanitized
        
        elif isinstance(data, list):
            return [self._sanitize_data(item) for item in data]
        
        else:
            return data
    
    def _add_caller_info(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        """Add caller information to log entry"""
        frame = inspect.currentframe()
        try:
            # Go up the stack to find the actual caller
            for _ in range(4):  # Skip internal logger frames
                frame = frame.f_back
                if frame is None:
                    break
            
            if frame:
                extra["caller"] = {
                    "file": frame.f_code.co_filename,
                    "function": frame.f_code.co_name,
                    "line": frame.f_lineno
                }
        finally:
            del frame
        
        return extra
    
    def log_event(
        self,
        event_type: EventType,
        message: str,
        level: LogLevel = LogLevel.INFO,
        extra: Optional[Dict[str, Any]] = None,
        performance: Optional[PerformanceMetrics] = None,
        security: Optional[SecurityContext] = None,
        **kwargs
    ):
        """Log structured event"""
        log_data = {
            "event_type": event_type.value,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
        
        if performance:
            log_data["performance"] = asdict(performance)
        
        if security:
            log_data["security"] = asdict(security)
        
        if extra:
            log_data.update(extra)
        
        # Add caller info in debug mode
        if self.settings.debug:
            log_data = self._add_caller_info(log_data)
        
        # Log with appropriate level
        getattr(logger, level.value.lower())(message, **log_data)
    
    def log_api_request(
        self,
        method: str,
        url: str,
        headers: Optional[Dict] = None,
        params: Optional[Dict] = None,
        request_id: Optional[str] = None
    ):
        """Log API request"""
        self.log_event(
            EventType.API_REQUEST,
            f"{method} {url}",
            extra={
                "http_method": method,
                "url": url,
                "headers": self._sanitize_data(headers) if headers else None,
                "params": params,
                "request_id": request_id
            }
        )
    
    def log_api_response(
        self,
        method: str,
        url: str,
        status_code: int,
        response_time_ms: float,
        response_size: Optional[int] = None,
        request_id: Optional[str] = None
    ):
        """Log API response"""
        self.log_event(
            EventType.API_RESPONSE,
            f"{method} {url} -> {status_code}",
            extra={
                "http_method": method,
                "url": url,
                "status_code": status_code,
                "request_id": request_id
            },
            performance=PerformanceMetrics(
                duration_ms=response_time_ms,
                response_size_bytes=response_size
            )
        )
    
    def log_trade_event(
        self,
        event_type: EventType,
        symbol: str,
        side: str,
        size: float,
        price: Optional[float] = None,
        order_id: Optional[str] = None,
        **kwargs
    ):
        """Log trading event"""
        self.log_event(
            event_type,
            f"Trade {event_type.value}: {side} {size} {symbol}",
            extra={
                "symbol": symbol,
                "side": side,
                "size": size,
                "price": price,
                "order_id": order_id,
                **kwargs
            }
        )
    
    def log_risk_event(
        self,
        risk_type: str,
        symbol: str,
        risk_value: float,
        threshold: float,
        action_taken: str,
        **kwargs
    ):
        """Log risk management event"""
        level = LogLevel.WARNING if risk_value > threshold else LogLevel.INFO
        
        self.log_event(
            EventType.RISK_CHECK,
            f"Risk check: {risk_type} for {symbol}",
            level=level,
            extra={
                "risk_type": risk_type,
                "symbol": symbol,
                "risk_value": risk_value,
                "threshold": threshold,
                "action_taken": action_taken,
                **kwargs
            }
        )
    
    def log_performance_metric(
        self,
        metric_name: str,
        metric_value: float,
        unit: str,
        component: Optional[str] = None,
        **kwargs
    ):
        """Log performance metric"""
        self.log_event(
            EventType.PERFORMANCE_METRIC,
            f"Performance: {metric_name} = {metric_value} {unit}",
            extra={
                "metric_name": metric_name,
                "metric_value": metric_value,
                "unit": unit,
                "component": component,
                **kwargs
            }
        )
    
    def log_security_event(
        self,
        event_description: str,
        severity: str,
        ip_address: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs
    ):
        """Log security event"""
        level = LogLevel.WARNING if severity == "medium" else LogLevel.ERROR
        
        self.log_event(
            EventType.SECURITY_EVENT,
            event_description,
            level=level,
            security=SecurityContext(
                ip_address=ip_address,
                operation_type="security"
            ),
            extra={
                "severity": severity,
                "user_id": user_id,
                **kwargs
            }
        )
    
    def log_error(
        self,
        error: Exception,
        message: Optional[str] = None,
        **kwargs
    ):
        """Log error with full context"""
        error_message = message or f"Error occurred: {type(error).__name__}: {error}"
        
        self.log_event(
            EventType.ERROR_OCCURRED,
            error_message,
            level=LogLevel.ERROR,
            extra={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
                **kwargs
            }
        )


# Context managers for request/session tracking
class LoggingContext:
    """Context manager for logging context"""
    
    def __init__(
        self,
        request_id: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None
    ):
        self.request_id = request_id
        self.session_id = session_id
        self.user_id = user_id
        
        self.request_id_token = None
        self.session_id_token = None
        self.user_id_token = None
    
    def __enter__(self):
        if self.request_id:
            self.request_id_token = request_id_var.set(self.request_id)
        if self.session_id:
            self.session_id_token = session_id_var.set(self.session_id)
        if self.user_id:
            self.user_id_token = user_id_var.set(self.user_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.request_id_token:
            request_id_var.reset(self.request_id_token)
        if self.session_id_token:
            session_id_var.reset(self.session_id_token)
        if self.user_id_token:
            user_id_var.reset(self.user_id_token)


# Global logger instance
_structured_logger: Optional[StructuredLogger] = None
_logger_lock = threading.Lock()


def get_logger() -> StructuredLogger:
    """Get global structured logger instance"""
    global _structured_logger
    
    if _structured_logger is None:
        with _logger_lock:
            if _structured_logger is None:
                _structured_logger = StructuredLogger()
    
    return _structured_logger


# Convenience functions
def log_info(message: str, **kwargs):
    """Log info message"""
    get_logger().log_event(EventType.USER_ACTION, message, LogLevel.INFO, **kwargs)


def log_warning(message: str, **kwargs):
    """Log warning message"""
    get_logger().log_event(EventType.USER_ACTION, message, LogLevel.WARNING, **kwargs)


def log_error(error: Exception, message: Optional[str] = None, **kwargs):
    """Log error message"""
    get_logger().log_error(error, message, **kwargs)


def log_api_call(method: str, url: str, status_code: int, response_time: float, **kwargs):
    """Log API call"""
    structured_logger = get_logger()
    structured_logger.log_api_request(method, url, **kwargs)
    structured_logger.log_api_response(method, url, status_code, response_time, **kwargs)


def log_trade(event_type: EventType, symbol: str, side: str, size: float, **kwargs):
    """Log trade event"""
    get_logger().log_trade_event(event_type, symbol, side, size, **kwargs)


# Example usage and testing
async def test_logging():
    """Test structured logging"""
    structured_logger = get_logger()
    
    # Test basic logging
    structured_logger.log_event(
        EventType.SYSTEM_START,
        "ProjectChimera started",
        extra={"version": "2.1.0"}
    )
    
    # Test with context
    with LoggingContext(request_id="req_123", session_id="sess_456"):
        structured_logger.log_api_request(
            "GET", "https://api.bitget.com/ticker",
            headers={"Authorization": "Bearer secret_token_12345"}
        )
        
        structured_logger.log_trade_event(
            EventType.TRADE_PLACED,
            "BTCUSDT", "long", 0.1,
            price=50000.0,
            order_id="order_789"
        )
    
    # Test error logging
    try:
        raise ValueError("Test error")
    except Exception as e:
        structured_logger.log_error(e, "Test error occurred")


if __name__ == "__main__":
    asyncio.run(test_logging())