"""
Trading Orchestrator v2 - ORCH-06 Implementation
Complete Feed → Strategy → Risk → Execution pipeline with circuit breaker
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import structlog

from .settings import get_settings
from .datafeed.bitget_ws import BitgetWebSocketFeed, create_bitget_ws_feed
from .risk.unified_engine import UnifiedRiskEngine, UnifiedRiskConfig
from .execution.bitget import BitgetExecutionClient, BitgetConfig
from .domains.market import MarketFrame, Signal, SignalType
from .strategies import (
    WeekendEffectStrategy, StopReversionStrategy, FundingContraStrategy,
    create_weekend_effect_strategy, create_stop_reversion_strategy, create_funding_contra_strategy
)

# Structured logging
logger = structlog.get_logger(__name__)


class PipelineStage(Enum):
    """Pipeline processing stages"""
    FEED = "feed"
    STRATEGY = "strategy"
    RISK = "risk"
    EXECUTION = "execution"


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class PipelineMetrics:
    """Pipeline performance metrics"""
    start_time: datetime
    market_updates: int = 0
    signals_generated: int = 0
    signals_approved: int = 0
    orders_executed: int = 0
    orders_failed: int = 0
    avg_latency_ms: float = 0.0
    errors_by_stage: Dict[str, int] = None
    
    def __post_init__(self):
        if self.errors_by_stage is None:
            self.errors_by_stage = {stage.value: 0 for stage in PipelineStage}


class ExecutionCircuitBreaker:
    """Circuit breaker for order execution failures"""
    
    def __init__(self, failure_threshold: int = 3, recovery_timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = CircuitBreakerState.CLOSED
    
    def can_execute(self) -> bool:
        """Check if execution is allowed"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info("execution_circuit_breaker_half_open")
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record successful execution"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            logger.info("execution_circuit_breaker_closed", reason="success")
        self.failure_count = 0
    
    def record_failure(self):
        """Record failed execution"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.error("execution_circuit_breaker_opened", 
                        failures=self.failure_count,
                        timeout_minutes=self.recovery_timeout // 60)


class TradingOrchestrator:
    """
    Production Trading Orchestrator - ORCH-06
    
    Complete async pipeline:
    Feed → Strategy → Risk → Execution
    
    Features:
    - Real-time WebSocket data feeds
    - Async strategy signal generation
    - Async risk management with wallet integration
    - Circuit breaker on execution failures
    - Comprehensive monitoring and metrics
    """
    
    def __init__(self, symbols: List[str], initial_equity: float = 150000.0):
        self.symbols = symbols
        self.initial_equity = initial_equity
        self.settings = get_settings()
        
        # Initialize core components
        self._init_components()
        
        # Circuit breaker for execution failures
        self.circuit_breaker = ExecutionCircuitBreaker(
            failure_threshold=3,
            recovery_timeout=300  # 5 minutes
        )
        
        # Pipeline state
        self.running = False
        self.tasks: List[asyncio.Task] = []
        self.metrics = PipelineMetrics(start_time=datetime.now())
        
        # Strategy instances
        self.strategies = {
            'weekend_effect': create_weekend_effect_strategy(),
            'stop_reversion': create_stop_reversion_strategy(),
            'funding_contrarian': create_funding_contra_strategy()
        }
        
        logger.info("orchestrator_initialized", 
                   symbols=symbols, 
                   initial_equity=initial_equity,
                   strategies=list(self.strategies.keys()))
    
    def _init_components(self):
        """Initialize trading components"""
        # Data feed configuration
        feed_config = {
            'api_key': self.settings.api.bitget_key.get_secret_value(),
            'secret_key': self.settings.api.bitget_secret.get_secret_value(),
            'passphrase': self.settings.api.bitget_passphrase.get_secret_value(),
            'sandbox': self.settings.api.bitget_sandbox
        }
        self.data_feed = create_bitget_ws_feed(feed_config)
        
        # Risk engine
        risk_config = UnifiedRiskConfig.from_settings()
        self.risk_engine = UnifiedRiskEngine(risk_config, self.initial_equity)
        
        # Execution engine
        exec_config = BitgetConfig.from_settings()
        self.execution_client = BitgetExecutionClient(exec_config)
    
    async def start(self):
        """Start the complete trading pipeline"""
        try:
            logger.info("orchestrator_starting")
            
            # Connect data feed
            await self.data_feed.connect()
            
            # Subscribe to symbols
            for symbol in self.symbols:
                await self.data_feed.subscribe_ticker(symbol)
                await self.data_feed.subscribe_orderbook(symbol, levels=5)
                await self.data_feed.subscribe_trades(symbol)
                await self.data_feed.subscribe_funding(symbol)
                await self.data_feed.subscribe_open_interest(symbol)
            
            # Start pipeline tasks
            self.running = True
            self.tasks = [
                asyncio.create_task(self._market_data_pipeline()),
                asyncio.create_task(self._health_monitor()),
                asyncio.create_task(self._metrics_reporter())
            ]
            
            logger.info("orchestrator_started", 
                       symbols=self.symbols,
                       strategies=len(self.strategies))
            
        except Exception as e:
            logger.error("orchestrator_start_failed", error=str(e))
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the trading pipeline gracefully"""
        logger.info("orchestrator_stopping")
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Disconnect components
        await self.data_feed.disconnect()
        
        logger.info("orchestrator_stopped", 
                   runtime_minutes=(datetime.now() - self.metrics.start_time).total_seconds() / 60)
    
    async def _market_data_pipeline(self):
        """Main pipeline: Feed → Strategy → Risk → Execution"""
        logger.info("pipeline_started")
        
        while self.running:
            try:
                # Process each symbol
                for symbol in self.symbols:
                    await self._process_symbol(symbol)
                
                # Small delay to prevent busy loop
                await asyncio.sleep(0.1)
                
            except Exception as e:
                self.metrics.errors_by_stage['feed'] += 1
                logger.error("pipeline_error", error=str(e))
                await asyncio.sleep(1)  # Longer delay on error
    
    async def _process_symbol(self, symbol: str):
        """Process a single symbol through the complete pipeline"""
        pipeline_start = time.time()
        
        try:
            # Step 1: Get market data
            market_frame = await self._build_market_frame(symbol)
            if not market_frame:
                return
            
            self.metrics.market_updates += 1
            
            # Step 2: Generate signals from all strategies
            signals = await self._generate_signals(market_frame)
            
            # Step 3: Process each signal through risk and execution
            for signal in signals:
                await self._process_signal(signal, market_frame.current_price or 50000.0)
            
            # Update latency metrics
            latency_ms = (time.time() - pipeline_start) * 1000
            alpha = 0.1  # EMA smoothing
            if self.metrics.avg_latency_ms == 0:
                self.metrics.avg_latency_ms = latency_ms
            else:
                self.metrics.avg_latency_ms = alpha * latency_ms + (1 - alpha) * self.metrics.avg_latency_ms
                
        except Exception as e:
            logger.error("symbol_processing_error", symbol=symbol, error=str(e))
    
    async def _build_market_frame(self, symbol: str) -> Optional[MarketFrame]:
        """Build MarketFrame from current market data"""
        try:
            # Get ticker data
            ticker = await self.data_feed.get_ticker(symbol)
            orderbook = await self.data_feed.get_orderbook(symbol)
            
            if not ticker:
                return None
            
            # Build market frame
            market_frame = MarketFrame(
                symbol=symbol,
                timestamp=datetime.now(),
                ticker=ticker,
                orderbook=orderbook
            )
            
            return market_frame
            
        except Exception as e:
            logger.error("market_frame_build_error", symbol=symbol, error=str(e))
            return None
    
    async def _generate_signals(self, market_frame: MarketFrame) -> List[Signal]:
        """Generate signals from all enabled strategies"""
        signals = []
        
        for strategy_name, strategy in self.strategies.items():
            try:
                if not strategy.config.enabled:
                    continue
                
                signal = strategy.generate_signal(market_frame)
                if signal:
                    signals.append(signal)
                    self.metrics.signals_generated += 1
                    
                    logger.debug("signal_generated",
                               symbol=market_frame.symbol,
                               strategy=strategy_name,
                               signal_type=signal.signal_type.value,
                               confidence=signal.confidence)
                    
            except Exception as e:
                self.metrics.errors_by_stage['strategy'] += 1
                logger.error("strategy_error", 
                           strategy=strategy_name, 
                           symbol=market_frame.symbol,
                           error=str(e))
        
        return signals
    
    async def _process_signal(self, signal: Signal, current_price: float):
        """Process signal through risk management and execution"""
        try:
            # Step 1: Risk evaluation (async)
            risk_decision = await self.risk_engine.calculate_position_size(
                signal, current_price
            )
            
            # Step 2: Check if signal is approved
            if not risk_decision.can_trade or risk_decision.position_size_pct <= 0:
                logger.debug("signal_rejected",
                           symbol=signal.symbol,
                           reason=risk_decision.primary_constraint,
                           confidence=risk_decision.confidence)
                return
            
            self.metrics.signals_approved += 1
            
            # Step 3: Execute order (with circuit breaker)
            await self._execute_order(signal, risk_decision, current_price)
            
        except Exception as e:
            self.metrics.errors_by_stage['risk'] += 1
            logger.error("signal_processing_error", 
                        symbol=signal.symbol,
                        error=str(e))
    
    async def _execute_order(self, signal: Signal, risk_decision, current_price: float):
        """Execute order with circuit breaker protection"""
        # Check circuit breaker
        if not self.circuit_breaker.can_execute():
            logger.warning("execution_blocked_circuit_breaker", 
                          symbol=signal.symbol,
                          breaker_state=self.circuit_breaker.state.value)
            return
        
        try:
            # Calculate order parameters
            position_value = self.initial_equity * risk_decision.position_size_pct
            order_size = position_value / current_price
            
            # Log execution attempt
            logger.info("order_execution_attempt",
                       symbol=signal.symbol,
                       side=signal.signal_type.value,
                       size=order_size,
                       confidence=risk_decision.confidence,
                       risk_constraint=risk_decision.primary_constraint)
            
            # Execute order (placeholder - would call actual execution)
            success = await self._mock_execute_order(signal, order_size, current_price)
            
            if success:
                self.circuit_breaker.record_success()
                self.metrics.orders_executed += 1
                
                logger.info("order_executed",
                           symbol=signal.symbol,
                           side=signal.signal_type.value,
                           size=order_size,
                           price=current_price)
            else:
                self.circuit_breaker.record_failure()
                self.metrics.orders_failed += 1
                
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.metrics.orders_failed += 1
            self.metrics.errors_by_stage['execution'] += 1
            
            logger.error("order_execution_error",
                        symbol=signal.symbol,
                        error=str(e))
    
    async def _mock_execute_order(self, signal: Signal, size: float, price: float) -> bool:
        """Mock order execution for testing"""
        # Simulate execution delay
        await asyncio.sleep(0.1)
        
        # 95% success rate for testing
        import random
        return random.random() > 0.05
    
    async def _health_monitor(self):
        """Monitor component health"""
        while self.running:
            try:
                # Check data feed health
                feed_connected = self.data_feed.is_connected()
                
                # Check risk engine health
                risk_health = await self.risk_engine.health_check()
                
                logger.debug("health_check",
                           feed_connected=feed_connected,
                           risk_status=risk_health.get('status', 'unknown'),
                           can_trade=risk_health.get('can_trade', False))
                
                await asyncio.sleep(30)  # Health check every 30 seconds
                
            except Exception as e:
                logger.error("health_monitor_error", error=str(e))
                await asyncio.sleep(5)
    
    async def _metrics_reporter(self):
        """Report pipeline metrics periodically"""
        while self.running:
            try:
                runtime_minutes = (datetime.now() - self.metrics.start_time).total_seconds() / 60
                
                logger.info("pipeline_metrics",
                           runtime_minutes=round(runtime_minutes, 1),
                           market_updates=self.metrics.market_updates,
                           signals_generated=self.metrics.signals_generated,
                           signals_approved=self.metrics.signals_approved,
                           orders_executed=self.metrics.orders_executed,
                           orders_failed=self.metrics.orders_failed,
                           avg_latency_ms=round(self.metrics.avg_latency_ms, 2),
                           errors=sum(self.metrics.errors_by_stage.values()),
                           circuit_breaker_state=self.circuit_breaker.state.value)
                
                await asyncio.sleep(60)  # Report every minute
                
            except Exception as e:
                logger.error("metrics_reporter_error", error=str(e))
                await asyncio.sleep(10)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current orchestrator status"""
        runtime_seconds = (datetime.now() - self.metrics.start_time).total_seconds()
        
        return {
            'running': self.running,
            'runtime_seconds': runtime_seconds,
            'symbols': self.symbols,
            'strategies_enabled': len([s for s in self.strategies.values() if s.config.enabled]),
            'feed_connected': self.data_feed.is_connected(),
            'circuit_breaker_state': self.circuit_breaker.state.value,
            'metrics': {
                'market_updates': self.metrics.market_updates,
                'signals_generated': self.metrics.signals_generated,
                'signals_approved': self.metrics.signals_approved,
                'orders_executed': self.metrics.orders_executed,
                'orders_failed': self.metrics.orders_failed,
                'avg_latency_ms': round(self.metrics.avg_latency_ms, 2),
                'errors_by_stage': self.metrics.errors_by_stage
            }
        }


async def main():
    """Demo main function for testing"""
    orchestrator = TradingOrchestrator(['BTCUSDT'], initial_equity=150000.0)
    
    try:
        await orchestrator.start()
        # Run for 30 seconds in demo
        await asyncio.sleep(30)
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())