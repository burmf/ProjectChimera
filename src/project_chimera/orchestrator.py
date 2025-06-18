"""
Live Trading Orchestrator - Phase F Implementation
Async pipeline: Feed → StrategyHub → RiskEngine → Execution
"""

import asyncio
import logging
import signal
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import traceback
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ComponentState(Enum):
    """Component states"""
    STOPPED = "stopped"
    STARTING = "starting" 
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"


class PipelineStage(Enum):
    """Pipeline processing stages"""
    FEED = "feed"
    STRATEGY = "strategy"
    RISK = "risk"
    EXECUTION = "execution"


@dataclass
class MarketFrame:
    """Real-time market data frame"""
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: float
    source: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SignalFrame:
    """Trading signal from strategy"""
    symbol: str
    strategy_id: str
    signal_type: str  # 'buy', 'sell', 'hold'
    confidence: float  # 0.0 to 1.0
    target_size: float  # Position size
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RiskFrame:
    """Risk-adjusted signal"""
    symbol: str
    strategy_id: str
    original_signal: SignalFrame
    risk_adjusted_size: float
    risk_multiplier: float
    risk_warnings: List[str]
    approval_status: str  # 'approved', 'rejected', 'modified'
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionFrame:
    """Execution instruction"""
    symbol: str
    side: str  # 'buy', 'sell'
    size: float
    order_type: str  # 'market', 'limit'
    price: Optional[float]
    strategy_id: str
    risk_frame_id: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CircuitBreaker:
    """Circuit breaker for error handling"""
    
    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 300.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def can_execute(self) -> bool:
        """Check if circuit breaker allows execution"""
        current_time = time.time()
        
        if self.state == "closed":
            return True
        elif self.state == "open":
            if (self.last_failure_time and 
                current_time - self.last_failure_time > self.recovery_timeout):
                self.state = "half_open"
                logger.info("Circuit breaker entering half-open state")
                return True
            return False
        elif self.state == "half_open":
            return True
        
        return False
    
    def record_success(self):
        """Record successful operation"""
        if self.state == "half_open":
            self.state = "closed"
            self.failure_count = 0
            logger.info("Circuit breaker closed - service recovered")
    
    def record_failure(self):
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time,
            'can_execute': self.can_execute()
        }


class AsyncPipelineComponent:
    """Base class for pipeline components"""
    
    def __init__(self, name: str):
        self.name = name
        self.state = ComponentState.STOPPED
        self.circuit_breaker = CircuitBreaker()
        self.metrics = {
            'processed_count': 0,
            'error_count': 0,
            'last_processed': None,
            'avg_processing_time': 0.0
        }
        self._running = False
    
    async def start(self):
        """Start the component"""
        self.state = ComponentState.STARTING
        logger.info(f"Starting {self.name}")
        try:
            await self._start_impl()
            self.state = ComponentState.RUNNING
            self._running = True
            logger.info(f"{self.name} started successfully")
        except Exception as e:
            self.state = ComponentState.ERROR
            logger.error(f"Failed to start {self.name}: {e}")
            raise
    
    async def stop(self):
        """Stop the component"""
        self.state = ComponentState.STOPPING
        logger.info(f"Stopping {self.name}")
        self._running = False
        try:
            await self._stop_impl()
            self.state = ComponentState.STOPPED
            logger.info(f"{self.name} stopped successfully")
        except Exception as e:
            self.state = ComponentState.ERROR
            logger.error(f"Error stopping {self.name}: {e}")
            raise
    
    async def process(self, data: Any) -> Any:
        """Process data through the component"""
        if not self.circuit_breaker.can_execute():
            raise Exception(f"{self.name} circuit breaker is open")
        
        start_time = time.time()
        try:
            result = await self._process_impl(data)
            
            # Update metrics
            processing_time = time.time() - start_time
            self.metrics['processed_count'] += 1
            self.metrics['last_processed'] = datetime.now()
            
            # Update rolling average
            alpha = 0.1  # Exponential decay factor
            if self.metrics['avg_processing_time'] == 0:
                self.metrics['avg_processing_time'] = processing_time
            else:
                self.metrics['avg_processing_time'] = (
                    alpha * processing_time + 
                    (1 - alpha) * self.metrics['avg_processing_time']
                )
            
            self.circuit_breaker.record_success()
            return result
            
        except Exception as e:
            self.metrics['error_count'] += 1
            self.circuit_breaker.record_failure()
            logger.error(f"Error in {self.name}: {e}")
            raise
    
    async def _start_impl(self):
        """Override in subclasses"""
        pass
    
    async def _stop_impl(self):
        """Override in subclasses"""
        pass
    
    async def _process_impl(self, data: Any) -> Any:
        """Override in subclasses"""
        raise NotImplementedError
    
    def get_health(self) -> Dict[str, Any]:
        """Get component health status"""
        return {
            'name': self.name,
            'state': self.state.value,
            'circuit_breaker': self.circuit_breaker.get_status(),
            'metrics': self.metrics,
            'is_healthy': self.state == ComponentState.RUNNING and self.circuit_breaker.can_execute()
        }


class DataFeed(AsyncPipelineComponent):
    """Market data feed component"""
    
    def __init__(self, symbols: List[str]):
        super().__init__("DataFeed")
        self.symbols = symbols
        self.subscribers = []
        self.feed_task = None
    
    async def _start_impl(self):
        """Start the data feed"""
        self.feed_task = asyncio.create_task(self._feed_loop())
    
    async def _stop_impl(self):
        """Stop the data feed"""
        if self.feed_task:
            self.feed_task.cancel()
            try:
                await self.feed_task
            except asyncio.CancelledError:
                pass
    
    async def _feed_loop(self):
        """Main feed loop"""
        while self._running:
            try:
                # Simulate market data generation
                for symbol in self.symbols:
                    frame = await self._generate_market_frame(symbol)
                    
                    # Send to subscribers
                    for subscriber in self.subscribers:
                        try:
                            await subscriber(frame)
                        except Exception as e:
                            logger.error(f"Error sending to subscriber: {e}")
                
                await asyncio.sleep(1.0)  # 1 second interval
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in feed loop: {e}")
                await asyncio.sleep(5.0)  # Retry delay
    
    async def _generate_market_frame(self, symbol: str) -> MarketFrame:
        """Generate simulated market data"""
        import random
        
        # Simple price simulation
        base_price = 50000.0 if symbol == "BTCUSDT" else 3000.0
        spread = base_price * 0.0001  # 0.01% spread
        
        mid_price = base_price * (1 + random.gauss(0, 0.001))  # 0.1% volatility
        
        return MarketFrame(
            symbol=symbol,
            timestamp=datetime.now(),
            bid=mid_price - spread/2,
            ask=mid_price + spread/2,
            last=mid_price,
            volume=random.uniform(1000, 10000),
            source="simulation"
        )
    
    def subscribe(self, callback: Callable):
        """Subscribe to market data"""
        self.subscribers.append(callback)
    
    async def _process_impl(self, data: Any) -> Any:
        """Process method for pipeline compatibility"""
        return data


class StrategyHub(AsyncPipelineComponent):
    """Strategy signal generation hub"""
    
    def __init__(self):
        super().__init__("StrategyHub")
        self.strategies = {}
        self.signal_queue = asyncio.Queue()
    
    async def _start_impl(self):
        """Start strategy hub"""
        # Initialize mock strategies
        self.strategies = {
            'momentum': {'enabled': True, 'last_signal': None},
            'mean_reversion': {'enabled': True, 'last_signal': None},
            'volatility_breakout': {'enabled': True, 'last_signal': None}
        }
    
    async def _process_impl(self, market_frame: MarketFrame) -> List[SignalFrame]:
        """Process market data and generate signals"""
        signals = []
        
        for strategy_id, config in self.strategies.items():
            if not config['enabled']:
                continue
            
            try:
                signal = await self._generate_signal(strategy_id, market_frame)
                if signal:
                    signals.append(signal)
                    await self.signal_queue.put(signal)
                    
            except Exception as e:
                logger.error(f"Error in strategy {strategy_id}: {e}")
        
        return signals
    
    async def _generate_signal(self, strategy_id: str, frame: MarketFrame) -> Optional[SignalFrame]:
        """Generate signal from specific strategy"""
        import random
        
        # Simple signal generation logic
        if strategy_id == 'momentum':
            # Random momentum signal
            if random.random() < 0.1:  # 10% chance
                return SignalFrame(
                    symbol=frame.symbol,
                    strategy_id=strategy_id,
                    signal_type=random.choice(['buy', 'sell']),
                    confidence=random.uniform(0.6, 0.9),
                    target_size=random.uniform(0.01, 0.05),  # 1-5% of portfolio
                    timestamp=frame.timestamp,
                    metadata={'price': frame.last, 'spread': frame.ask - frame.bid}
                )
        
        elif strategy_id == 'mean_reversion':
            # Random mean reversion signal
            if random.random() < 0.05:  # 5% chance
                return SignalFrame(
                    symbol=frame.symbol,
                    strategy_id=strategy_id,
                    signal_type=random.choice(['buy', 'sell']),
                    confidence=random.uniform(0.5, 0.8),
                    target_size=random.uniform(0.005, 0.02),  # 0.5-2% of portfolio
                    timestamp=frame.timestamp,
                    metadata={'price': frame.last, 'strategy': 'mean_reversion'}
                )
        
        return None


class RiskEngine(AsyncPipelineComponent):
    """Risk management engine"""
    
    def __init__(self):
        super().__init__("RiskEngine")
        self.risk_config = {
            'max_position_size': 0.10,  # 10% max position
            'max_daily_loss': 0.05,     # 5% max daily loss
            'max_correlation': 0.7,     # Max correlation between positions
            'leverage_limit': 3.0       # Max leverage
        }
        self.current_exposure = {}
        self.daily_pnl = 0.0
    
    async def _process_impl(self, signal_frame: SignalFrame) -> Optional[RiskFrame]:
        """Process signal through risk engine"""
        
        # Risk checks
        warnings = []
        risk_multiplier = 1.0
        approval_status = "approved"
        
        # Position size check
        if signal_frame.target_size > self.risk_config['max_position_size']:
            risk_multiplier *= 0.5
            warnings.append(f"Position size reduced: {signal_frame.target_size:.3f} -> {signal_frame.target_size * risk_multiplier:.3f}")
        
        # Daily loss check
        if abs(self.daily_pnl) > self.risk_config['max_daily_loss']:
            risk_multiplier *= 0.3
            warnings.append("Daily loss limit exceeded - reducing size")
        
        # Exposure check
        current_exposure = self.current_exposure.get(signal_frame.symbol, 0.0)
        if abs(current_exposure + signal_frame.target_size) > self.risk_config['max_position_size']:
            risk_multiplier *= 0.7
            warnings.append("Exposure limit exceeded - reducing size")
        
        # Final size calculation
        risk_adjusted_size = signal_frame.target_size * risk_multiplier
        
        # Minimum size filter
        if risk_adjusted_size < 0.001:  # 0.1% minimum
            approval_status = "rejected"
            warnings.append("Position too small after risk adjustment")
        
        return RiskFrame(
            symbol=signal_frame.symbol,
            strategy_id=signal_frame.strategy_id,
            original_signal=signal_frame,
            risk_adjusted_size=risk_adjusted_size,
            risk_multiplier=risk_multiplier,
            risk_warnings=warnings,
            approval_status=approval_status,
            timestamp=datetime.now()
        )


class ExecutionEngine(AsyncPipelineComponent):
    """Order execution engine"""
    
    def __init__(self):
        super().__init__("ExecutionEngine")
        self.pending_orders = {}
        self.execution_history = []
        self.connection_healthy = True
    
    async def _start_impl(self):
        """Start execution engine"""
        # Initialize mock connection
        self.connection_healthy = True
        logger.info("Execution engine connection established")
    
    async def _process_impl(self, risk_frame: RiskFrame) -> Optional[ExecutionFrame]:
        """Execute risk-approved signals"""
        
        if risk_frame.approval_status != "approved":
            logger.info(f"Signal rejected by risk: {risk_frame.approval_status}")
            return None
        
        # Simulate order execution
        signal = risk_frame.original_signal
        
        execution_frame = ExecutionFrame(
            symbol=signal.symbol,
            side=signal.signal_type,
            size=risk_frame.risk_adjusted_size,
            order_type="market",
            price=None,  # Market order
            strategy_id=signal.strategy_id,
            risk_frame_id=id(risk_frame),
            timestamp=datetime.now()
        )
        
        # Simulate execution
        success = await self._execute_order(execution_frame)
        
        if success:
            self.execution_history.append(execution_frame)
            logger.info(f"Executed: {signal.signal_type} {risk_frame.risk_adjusted_size:.4f} {signal.symbol}")
            return execution_frame
        else:
            logger.error(f"Execution failed: {execution_frame.to_dict()}")
            return None
    
    async def _execute_order(self, execution_frame: ExecutionFrame) -> bool:
        """Simulate order execution"""
        import random
        
        # Simulate execution delay
        await asyncio.sleep(random.uniform(0.1, 0.5))
        
        # Simulate occasional failures
        if random.random() < 0.05:  # 5% failure rate
            return False
        
        return True


class TradingOrchestrator:
    """
    Main orchestrator for live trading pipeline
    Manages async components and data flow
    """
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        
        # Initialize components
        self.data_feed = DataFeed(symbols)
        self.strategy_hub = StrategyHub()
        self.risk_engine = RiskEngine()
        self.execution_engine = ExecutionEngine()
        
        self.components = [
            self.data_feed,
            self.strategy_hub, 
            self.risk_engine,
            self.execution_engine
        ]
        
        # Pipeline state
        self._running = False
        self._main_task = None
        self.stats = {
            'start_time': None,
            'market_frames_processed': 0,
            'signals_generated': 0,
            'signals_executed': 0,
            'errors': 0
        }
    
    async def start(self):
        """Start the trading orchestrator"""
        logger.info("Starting Trading Orchestrator")
        self.stats['start_time'] = datetime.now()
        
        try:
            # Start all components
            for component in self.components:
                await component.start()
            
            # Set up data flow pipeline
            self.data_feed.subscribe(self._process_market_data)
            
            self._running = True
            logger.info("Trading Orchestrator started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start orchestrator: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the trading orchestrator"""
        logger.info("Stopping Trading Orchestrator")
        self._running = False
        
        # Stop all components
        for component in reversed(self.components):
            try:
                await component.stop()
            except Exception as e:
                logger.error(f"Error stopping {component.name}: {e}")
        
        logger.info("Trading Orchestrator stopped")
    
    async def _process_market_data(self, market_frame: MarketFrame):
        """Process incoming market data through the pipeline"""
        try:
            self.stats['market_frames_processed'] += 1
            
            # Stage 1: Generate signals
            signals = await self.strategy_hub.process(market_frame)
            
            for signal in signals:
                self.stats['signals_generated'] += 1
                
                # Stage 2: Risk management
                risk_frame = await self.risk_engine.process(signal)
                
                if risk_frame and risk_frame.approval_status == "approved":
                    # Stage 3: Execution
                    execution_frame = await self.execution_engine.process(risk_frame)
                    
                    if execution_frame:
                        self.stats['signals_executed'] += 1
                        
        except Exception as e:
            self.stats['errors'] += 1
            logger.error(f"Error in pipeline: {e}")
            traceback.print_exc()
    
    def get_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        component_health = {comp.name: comp.get_health() for comp in self.components}
        
        all_healthy = all(health['is_healthy'] for health in component_health.values())
        
        return {
            'status': 'healthy' if all_healthy else 'unhealthy',
            'uptime_seconds': (datetime.now() - self.stats['start_time']).total_seconds() if self.stats['start_time'] else 0,
            'components': component_health,
            'stats': self.stats,
            'timestamp': datetime.now().isoformat()
        }
    
    async def run_forever(self):
        """Run the orchestrator indefinitely"""
        try:
            await self.start()
            
            # Set up signal handlers for graceful shutdown
            loop = asyncio.get_event_loop()
            for sig in [signal.SIGTERM, signal.SIGINT]:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
            
            # Keep running until stopped
            while self._running:
                await asyncio.sleep(1.0)
                
                # Periodic health check
                health = self.get_health()
                if health['status'] != 'healthy':
                    logger.warning("System health degraded")
        
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            traceback.print_exc()
        finally:
            await self.stop()


# CLI interface
async def main():
    """Main entry point for live trading"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Live Trading Orchestrator")
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT'], 
                       help='Trading symbols')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run orchestrator
    orchestrator = TradingOrchestrator(args.symbols)
    
    try:
        await orchestrator.run_forever()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(asyncio.run(main()))