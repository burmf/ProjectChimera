"""
Advanced Retry Logic and Circuit Breaker Patterns
Professional resilience patterns for high-frequency trading
"""

import asyncio
import time
import random
from abc import ABC, abstractmethod
from typing import (
    Any, Callable, Optional, Type, Union, Dict, List,
    Awaitable, TypeVar, Generic
)
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import threading
from functools import wraps

from loguru import logger
from tenacity import (
    retry, stop_after_attempt, wait_exponential, 
    retry_if_exception_type, before_sleep_log,
    RetryError
)


T = TypeVar('T')


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Failing, requests blocked
    HALF_OPEN = "half_open" # Testing if service recovered


class RetryStrategy(Enum):
    """Retry strategy types"""
    EXPONENTIAL = "exponential"
    LINEAR = "linear"
    FIXED = "fixed"
    JITTERED = "jittered"


@dataclass
class RetryConfig:
    """Retry configuration"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    multiplier: float = 2.0
    jitter: bool = True
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL
    
    # Exception handling
    retryable_exceptions: tuple = (ConnectionError, TimeoutError)
    non_retryable_exceptions: tuple = (ValueError, TypeError)


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration"""
    failure_threshold: int = 5       # Failures before opening
    recovery_timeout: float = 60.0   # Seconds before half-open
    success_threshold: int = 3       # Successes to close from half-open
    timeout: float = 30.0           # Request timeout
    
    # Failure counting window
    window_size: int = 100          # Number of recent calls to track
    minimum_throughput: int = 10    # Minimum calls before breaking


@dataclass
class CallMetrics:
    """Call execution metrics"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    last_success_time: Optional[datetime] = None
    last_failure_time: Optional[datetime] = None
    average_response_time: float = 0.0
    recent_failures: List[datetime] = field(default_factory=list)


class RetryableException(Exception):
    """Exception that should trigger retry"""
    pass


class NonRetryableException(Exception):
    """Exception that should not trigger retry"""
    pass


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class ICircuitBreaker(ABC, Generic[T]):
    """Circuit breaker interface"""
    
    @abstractmethod
    async def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection"""
        pass
    
    @abstractmethod
    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> CallMetrics:
        """Get call metrics"""
        pass


class AsyncCircuitBreaker(ICircuitBreaker[T]):
    """
    Async circuit breaker implementation
    Prevents cascading failures by temporarily blocking calls to failing services
    """
    
    def __init__(self, config: CircuitBreakerConfig, name: str = "default"):
        self.config = config
        self.name = name
        self.state = CircuitBreakerState.CLOSED
        self.metrics = CallMetrics()
        self.last_failure_time: Optional[datetime] = None
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        self._lock = asyncio.Lock()
        
        logger.info(f"Circuit breaker '{name}' initialized")
    
    async def call(self, func: Callable[..., Awaitable[T]], *args, **kwargs) -> T:
        """Execute function with circuit breaker protection"""
        async with self._lock:
            # Check if circuit is open
            if self.state == CircuitBreakerState.OPEN:
                if not self._should_attempt_reset():
                    raise CircuitBreakerOpenException(
                        f"Circuit breaker '{self.name}' is OPEN"
                    )
                else:
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker '{self.name}' moved to HALF_OPEN")
        
        # Execute the function
        start_time = time.time()
        try:
            result = await asyncio.wait_for(
                func(*args, **kwargs), 
                timeout=self.config.timeout
            )
            
            # Record success
            await self._record_success()
            return result
            
        except Exception as e:
            # Record failure
            await self._record_failure(e)
            raise
        
        finally:
            # Update response time
            response_time = time.time() - start_time
            self._update_response_time(response_time)
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if not self.last_failure_time:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout
    
    async def _record_success(self):
        """Record successful call"""
        async with self._lock:
            self.metrics.successful_calls += 1
            self.metrics.total_calls += 1
            self.metrics.last_success_time = datetime.now()
            self.consecutive_failures = 0
            self.consecutive_successes += 1
            
            # Transition from HALF_OPEN to CLOSED
            if (self.state == CircuitBreakerState.HALF_OPEN and 
                self.consecutive_successes >= self.config.success_threshold):
                self.state = CircuitBreakerState.CLOSED
                logger.info(f"Circuit breaker '{self.name}' moved to CLOSED")
    
    async def _record_failure(self, exception: Exception):
        """Record failed call"""
        async with self._lock:
            self.metrics.failed_calls += 1
            self.metrics.total_calls += 1
            self.metrics.last_failure_time = datetime.now()
            self.last_failure_time = datetime.now()
            self.consecutive_successes = 0
            self.consecutive_failures += 1
            
            # Track recent failures
            self.metrics.recent_failures.append(datetime.now())
            
            # Clean old failures outside window
            cutoff = datetime.now() - timedelta(minutes=5)
            self.metrics.recent_failures = [
                f for f in self.metrics.recent_failures if f > cutoff
            ]
            
            # Check if should open circuit
            if self._should_open_circuit():
                self.state = CircuitBreakerState.OPEN
                logger.warning(f"Circuit breaker '{self.name}' OPENED due to failures")
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should be opened"""
        # Not enough throughput to make decision
        if self.metrics.total_calls < self.config.minimum_throughput:
            return False
        
        # Too many consecutive failures
        if self.consecutive_failures >= self.config.failure_threshold:
            return True
        
        # High recent failure rate
        recent_failure_count = len(self.metrics.recent_failures)
        if recent_failure_count >= self.config.failure_threshold:
            return True
        
        return False
    
    def _update_response_time(self, response_time: float):
        """Update average response time"""
        if self.metrics.total_calls == 1:
            self.metrics.average_response_time = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.average_response_time = (
                alpha * response_time + 
                (1 - alpha) * self.metrics.average_response_time
            )
    
    def get_state(self) -> CircuitBreakerState:
        """Get current state"""
        return self.state
    
    def get_metrics(self) -> CallMetrics:
        """Get metrics"""
        return self.metrics
    
    def reset(self):
        """Manual reset of circuit breaker"""
        self.state = CircuitBreakerState.CLOSED
        self.consecutive_failures = 0
        self.consecutive_successes = 0
        logger.info(f"Circuit breaker '{self.name}' manually reset")


class RetryManager:
    """
    Advanced retry manager with multiple strategies
    """
    
    def __init__(self, config: RetryConfig):
        self.config = config
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        if self.config.strategy == RetryStrategy.FIXED:
            delay = self.config.base_delay
        elif self.config.strategy == RetryStrategy.LINEAR:
            delay = self.config.base_delay * attempt
        elif self.config.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.config.base_delay * (self.config.multiplier ** (attempt - 1))
        elif self.config.strategy == RetryStrategy.JITTERED:
            base_delay = self.config.base_delay * (self.config.multiplier ** (attempt - 1))
            jitter = random.uniform(0.1, 0.3) * base_delay
            delay = base_delay + jitter
        else:
            delay = self.config.base_delay
        
        return min(delay, self.config.max_delay)
    
    def _should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if should retry based on exception and attempt"""
        if attempt >= self.config.max_attempts:
            return False
        
        # Check non-retryable exceptions
        if isinstance(exception, self.config.non_retryable_exceptions):
            return False
        
        # Check retryable exceptions
        if isinstance(exception, self.config.retryable_exceptions):
            return True
        
        # Default to not retry
        return False
    
    async def execute_with_retry(
        self, 
        func: Callable[..., Awaitable[T]], 
        *args, 
        **kwargs
    ) -> T:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                return await func(*args, **kwargs)
            
            except Exception as e:
                last_exception = e
                
                if not self._should_retry(e, attempt):
                    raise e
                
                if attempt < self.config.max_attempts:
                    delay = self._calculate_delay(attempt)
                    logger.warning(
                        f"Attempt {attempt} failed: {e}. Retrying in {delay:.2f}s"
                    )
                    await asyncio.sleep(delay)
        
        # All attempts failed
        raise last_exception


# Decorators for easy usage
def circuit_breaker(
    config: Optional[CircuitBreakerConfig] = None,
    name: str = "default"
):
    """Circuit breaker decorator"""
    if config is None:
        config = CircuitBreakerConfig()
    
    breaker = AsyncCircuitBreaker(config, name)
    
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await breaker.call(func, *args, **kwargs)
        return wrapper
    return decorator


def retry_with_config(config: Optional[RetryConfig] = None):
    """Retry decorator with configuration"""
    if config is None:
        config = RetryConfig()
    
    retry_manager = RetryManager(config)
    
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            return await retry_manager.execute_with_retry(func, *args, **kwargs)
        return wrapper
    return decorator


# Combined resilience decorator
def resilient(
    retry_config: Optional[RetryConfig] = None,
    circuit_config: Optional[CircuitBreakerConfig] = None,
    name: str = "default"
):
    """Combined retry + circuit breaker decorator"""
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        # Apply circuit breaker first, then retry
        if circuit_config:
            func = circuit_breaker(circuit_config, name)(func)
        
        if retry_config:
            func = retry_with_config(retry_config)(func)
        
        return func
    return decorator


# Predefined configurations for common scenarios
class ResiliencePresets:
    """Predefined resilience configurations"""
    
    # Quick operations (API calls)
    API_RETRY = RetryConfig(
        max_attempts=3,
        base_delay=0.5,
        max_delay=5.0,
        strategy=RetryStrategy.JITTERED,
        retryable_exceptions=(ConnectionError, TimeoutError, RetryableException)
    )
    
    API_CIRCUIT_BREAKER = CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout=30.0,
        success_threshold=2,
        timeout=10.0
    )
    
    # Critical operations (order placement)
    CRITICAL_RETRY = RetryConfig(
        max_attempts=5,
        base_delay=0.1,
        max_delay=2.0,
        strategy=RetryStrategy.EXPONENTIAL,
        retryable_exceptions=(ConnectionError, TimeoutError)
    )
    
    CRITICAL_CIRCUIT_BREAKER = CircuitBreakerConfig(
        failure_threshold=3,
        recovery_timeout=60.0,
        success_threshold=3,
        timeout=5.0
    )
    
    # Background operations (data collection)
    BACKGROUND_RETRY = RetryConfig(
        max_attempts=10,
        base_delay=2.0,
        max_delay=300.0,
        strategy=RetryStrategy.EXPONENTIAL,
        retryable_exceptions=(ConnectionError, TimeoutError, RetryableException)
    )


# Example usage
@resilient(
    retry_config=ResiliencePresets.API_RETRY,
    circuit_config=ResiliencePresets.API_CIRCUIT_BREAKER,
    name="bitget_api"
)
async def example_api_call():
    """Example API call with resilience"""
    # Simulate API call that might fail
    if random.random() < 0.3:
        raise ConnectionError("Network error")
    return {"status": "success"}


async def test_resilience():
    """Test resilience patterns"""
    try:
        result = await example_api_call()
        logger.info(f"API call succeeded: {result}")
    except Exception as e:
        logger.error(f"API call failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_resilience())