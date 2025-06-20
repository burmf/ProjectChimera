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

from .settings import get_settings
from .datafeed.bitget_ws import BitgetWebSocketFeed, create_bitget_ws_feed
from .risk.unified_engine import UnifiedRiskEngine, UnifiedRiskConfig
from .execution.bitget import BitgetExecutionClient, BitgetConfig
from .domains.market import Ticker, OrderBook, Signal, MarketFrame, SignalType
from .strategies import (
    WeekendEffectStrategy, StopReversionStrategy, FundingContraStrategy,
    create_weekend_effect_strategy, create_stop_reversion_strategy, create_funding_contra_strategy
)

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


class CircuitBreakerState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Circuit breaker triggered
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for order execution failures"""
    failure_threshold: int = 3
    recovery_timeout: int = 300  # 5 minutes
    
    failure_count: int = 0
    last_failure_time: float = 0.0
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    
    def record_success(self) -> None:
        """Record successful operation"""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
        
    def record_failure(self) -> None:
        """Record failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")
    
    def can_attempt(self) -> bool:
        """Check if operation can be attempted"""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        elif self.state == CircuitBreakerState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit breaker entering HALF_OPEN state")
                return True
            return False
        else:  # HALF_OPEN
            return True


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


class BitgetDataFeed(AsyncPipelineComponent):
    """Real Bitget market data feed component"""
    
    def __init__(self, symbols: List[str]):
        super().__init__("BitgetDataFeed")
        self.symbols = symbols
        self.subscribers = []
        self.settings = get_settings()
        
        # Initialize Bitget adapter
        bitget_config = {
            'api_key': self.settings.api.bitget_key.get_secret_value(),
            'secret_key': self.settings.api.bitget_secret.get_secret_value(),
            'passphrase': self.settings.api.bitget_passphrase.get_secret_value(),
            'sandbox': self.settings.api.bitget_sandbox,
            'timeout_seconds': self.settings.api.timeout_seconds
        }
        
        self.adapter = BitgetEnhancedAdapter("bitget_main", bitget_config)
        self.last_data = {}
    
    async def _start_impl(self):
        """Start the Bitget data feed"""
        try:
            # Connect to Bitget
            await self.adapter.connect()
            
            # Subscribe to all symbols
            for symbol in self.symbols:
                await self.adapter.subscribe_ticker(symbol)
                await self.adapter.subscribe_orderbook(symbol, levels=20)
                await self.adapter.subscribe_trades(symbol)
                
                # For futures, also subscribe to funding rates
                if "USDT" in symbol or "USDC" in symbol:
                    await self.adapter.subscribe_funding(symbol)
                    await self.adapter.subscribe_open_interest(symbol)
            
            # Start data processing task
            self.feed_task = asyncio.create_task(self._process_market_data())
            
            logger.info(f"Bitget data feed started for symbols: {self.symbols}")
            
        except Exception as e:
            logger.error(f"Failed to start Bitget data feed: {e}")
            raise
    
    async def _stop_impl(self):
        """Stop the Bitget data feed"""
        if hasattr(self, 'feed_task') and self.feed_task:
            self.feed_task.cancel()
            try:
                await self.feed_task
            except asyncio.CancelledError:
                pass
        
        if self.adapter:
            await self.adapter.disconnect()
    
    async def _process_market_data(self):
        """Process incoming market data from Bitget"""
        while self._running:
            try:
                # Check for new data from each symbol
                for symbol in self.symbols:
                    if symbol in self.adapter.market_data:
                        market_data = self.adapter.market_data[symbol]
                        
                        # Create market frame from latest data
                        if 'ticker' in market_data:
                            frame = await self._create_market_frame(symbol, market_data)
                            
                            # Send to subscribers
                            for subscriber in self.subscribers:
                                try:
                                    await subscriber(frame)
                                except Exception as e:
                                    logger.error(f"Error sending to subscriber: {e}")
                
                await asyncio.sleep(0.1)  # 100ms polling interval
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in market data processing: {e}")
                await asyncio.sleep(1.0)
    
    async def _create_market_frame(self, symbol: str, market_data: Dict) -> MarketFrame:
        """Create MarketFrame from Bitget market data"""
        ticker = market_data.get('ticker')
        orderbook = market_data.get('orderbook')
        
        if not ticker:
            # Fallback to last known data
            if symbol in self.last_data:
                return self.last_data[symbol]
            raise ValueError(f"No ticker data available for {symbol}")
        
        # Extract price data
        last_price = float(ticker.price)
        volume = float(ticker.volume_24h)
        
        # Get bid/ask from orderbook if available
        if orderbook and orderbook.bids and orderbook.asks:
            bid = float(orderbook.bids[0][0])
            ask = float(orderbook.asks[0][0])
        else:
            # Estimate spread
            spread = last_price * 0.0001  # 0.01% default spread
            bid = last_price - spread/2
            ask = last_price + spread/2
        
        frame = MarketFrame(
            symbol=symbol,
            timestamp=ticker.timestamp,
            bid=bid,
            ask=ask,
            last=last_price,
            volume=volume,
            source="bitget"
        )
        
        # Cache for fallback
        self.last_data[symbol] = frame
        
        return frame
    
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


class UnifiedRiskManager(AsyncPipelineComponent):
    """Unified risk management engine with Dynamic Kelly, ATR, and DD Guard"""
    
    def __init__(self, initial_portfolio_value: float = 150000.0):
        super().__init__("UnifiedRiskManager")
        self.settings = get_settings()
        
        # Create unified risk config from settings
        risk_config = UnifiedRiskConfig(
            kelly_base_fraction=self.settings.risk.kelly_fraction,
            kelly_ewma_alpha=0.1,
            kelly_min_trades=20,
            atr_target_daily_vol=self.settings.risk.target_vol_daily,
            atr_periods=self.settings.risk.atr_lookback,
            atr_min_position=0.01,
            atr_max_position=self.settings.trading.max_position_pct,
            dd_caution_threshold=0.05,
            dd_warning_threshold=self.settings.risk.max_drawdown,
            dd_critical_threshold=0.20,
            dd_warning_cooldown_hours=4.0,
            dd_critical_cooldown_hours=self.settings.risk.dd_pause_duration_hours,
            max_leverage=self.settings.trading.leverage_max,
            min_confidence=self.settings.trading.confidence_threshold,
            max_portfolio_vol=0.02
        )
        
        self.risk_engine = UnifiedRiskEngine(risk_config, initial_portfolio_value)
        self.portfolio_value = initial_portfolio_value
        self.current_exposure = {}
    
    async def _process_impl(self, signal_frame: SignalFrame) -> Optional[RiskFrame]:
        """Process signal through unified risk engine"""
        
        # Convert SignalFrame to Signal domain object
        signal = Signal(
            symbol=signal_frame.symbol,
            action=signal_frame.signal_type,
            confidence=signal_frame.confidence,
            timestamp=signal_frame.timestamp,
            strategy_id=signal_frame.strategy_id,
            metadata=signal_frame.metadata
        )
        
        # Get current price (use last price from metadata)
        current_price = signal_frame.metadata.get('price', 50000.0)
        
        try:
            # Calculate position size using unified risk engine
            risk_decision = self.risk_engine.calculate_position_size(
                signal=signal,
                current_price=current_price,
                portfolio_value=self.portfolio_value,
                timestamp=signal_frame.timestamp
            )
            
            # Convert to RiskFrame
            if risk_decision.is_valid() and risk_decision.can_trade:
                approval_status = "approved"
                risk_warnings = []
                
                # Add any risk warnings based on constraints
                if risk_decision.primary_constraint != "normal_sizing":
                    risk_warnings.append(f"Limited by: {risk_decision.primary_constraint}")
                
                return RiskFrame(
                    symbol=signal_frame.symbol,
                    strategy_id=signal_frame.strategy_id,
                    original_signal=signal_frame,
                    risk_adjusted_size=risk_decision.position_size_pct,
                    risk_multiplier=risk_decision.position_size_pct / signal_frame.target_size if signal_frame.target_size > 0 else 1.0,
                    risk_warnings=risk_warnings,
                    approval_status=approval_status,
                    timestamp=datetime.now()
                )
            else:
                # Signal rejected by risk engine
                return RiskFrame(
                    symbol=signal_frame.symbol,
                    strategy_id=signal_frame.strategy_id,
                    original_signal=signal_frame,
                    risk_adjusted_size=0.0,
                    risk_multiplier=0.0,
                    risk_warnings=[risk_decision.reasoning],
                    approval_status="rejected",
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            logger.error(f"Risk engine error: {e}")
            
            # Fallback to rejection
            return RiskFrame(
                symbol=signal_frame.symbol,
                strategy_id=signal_frame.strategy_id,
                original_signal=signal_frame,
                risk_adjusted_size=0.0,
                risk_multiplier=0.0,
                risk_warnings=[f"Risk engine error: {str(e)}"],
                approval_status="rejected",
                timestamp=datetime.now()
            )
    
    def update_portfolio_value(self, new_value: float) -> None:
        """Update portfolio value for risk calculations"""
        self.portfolio_value = new_value
    
    def record_trade_result(self, return_pct: float, new_portfolio_value: float) -> None:
        """Record trade result for Kelly calculations"""
        self.portfolio_value = new_portfolio_value
        self.risk_engine.update_trade_result(return_pct, new_portfolio_value)
    
    def get_risk_statistics(self) -> Dict[str, Any]:
        """Get comprehensive risk statistics"""
        return self.risk_engine.get_statistics()


class BitgetExecutionEngine(AsyncPipelineComponent):
    """Real Bitget order execution engine"""
    
    def __init__(self):
        super().__init__("BitgetExecutionEngine")
        self.settings = get_settings()
        self.pending_orders = {}
        self.execution_history = []
        
        # Initialize Bitget execution client
        self.execution_client = None
    
    async def _start_impl(self):
        """Start execution engine"""
        try:
            # Initialize Bitget execution client
            execution_config = {
                'api_key': self.settings.api.bitget_key.get_secret_value(),
                'secret_key': self.settings.api.bitget_secret.get_secret_value(),
                'passphrase': self.settings.api.bitget_passphrase.get_secret_value(),
                'sandbox': self.settings.api.bitget_sandbox,
                'timeout': self.settings.api.timeout_seconds
            }
            
            self.execution_client = BitgetExecutionClient(execution_config)
            
            # Test connection
            await self.execution_client.test_connection()
            
            logger.info("Bitget execution engine connected")
            
        except Exception as e:
            logger.error(f"Failed to start execution engine: {e}")
            raise
    
    async def _stop_impl(self):
        """Stop execution engine"""
        if self.execution_client:
            await self.execution_client.close()
    
    async def _process_impl(self, risk_frame: RiskFrame) -> Optional[ExecutionFrame]:
        """Execute risk-approved signals"""
        
        if risk_frame.approval_status != "approved":
            logger.info(f"Signal rejected by risk: {risk_frame.approval_status}")
            return None
        
        if not self.execution_client:
            logger.error("Execution client not initialized")
            return None
        
        signal = risk_frame.original_signal
        
        # Convert percentage to actual USD amount
        portfolio_value = 150000.0  # TODO: Get from portfolio manager
        usd_size = risk_frame.risk_adjusted_size * portfolio_value
        
        execution_frame = ExecutionFrame(
            symbol=signal.symbol,
            side=signal.signal_type,
            size=usd_size,  # USD amount
            order_type="market",
            price=None,  # Market order
            strategy_id=signal.strategy_id,
            risk_frame_id=id(risk_frame),
            timestamp=datetime.now()
        )
        
        # Execute order
        success = await self._execute_order(execution_frame)
        
        if success:
            self.execution_history.append(execution_frame)
            logger.info(f"Executed: {signal.signal_type} ${usd_size:.2f} {signal.symbol}")
            return execution_frame
        else:
            logger.error(f"Execution failed: {execution_frame.to_dict()}")
            return None
    
    async def _execute_order(self, execution_frame: ExecutionFrame) -> bool:
        """Execute order via Bitget API"""
        try:
            # Convert side to Bitget format
            side = "buy" if execution_frame.side in ["buy", "long"] else "sell"
            
            # Execute market order
            result = await self.execution_client.place_market_order(
                symbol=execution_frame.symbol,
                side=side,
                size_usd=execution_frame.size
            )
            
            if result and result.get('success', False):
                logger.info(f"Order executed successfully: {result.get('order_id')}")
                return True
            else:
                logger.error(f"Order execution failed: {result}")
                return False
                
        except Exception as e:
            logger.error(f"Execution error: {e}")
            return False


class TradingOrchestrator:
    """
    Main orchestrator for live trading pipeline
    Manages async components and data flow
    """
    
    def __init__(self, symbols: List[str], initial_portfolio_value: float = 150000.0):
        self.symbols = symbols
        self.portfolio_value = initial_portfolio_value
        
        # Initialize components with real implementations
        self.data_feed = BitgetDataFeed(symbols)
        self.strategy_hub = StrategyHub()
        self.risk_engine = UnifiedRiskManager(initial_portfolio_value)
        self.execution_engine = BitgetExecutionEngine()
        
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