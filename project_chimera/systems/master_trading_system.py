"""
Master Trading System - Professional Architecture Integration
Unified system orchestrating all trading components with clean architecture
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from loguru import logger

from ..config import Settings, get_settings
from ..core.api_client import AsyncBitgetClient, TickerData, OrderSide, OrderType
from ..core.risk_manager import ProfessionalRiskManager, MarketRegime
from ..utils.performance_tracker import PerformanceTracker
from ..utils.signal_generator import SignalGenerator, TradingSignal
from ..utils.position_manager import PositionManager


class SystemState(Enum):
    """System operational states"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class SystemMetrics:
    """Real-time system metrics"""
    total_pnl: float = 0.0
    daily_pnl: float = 0.0
    win_rate: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    active_positions: int = 0
    portfolio_value: float = 100000.0
    last_update: datetime = field(default_factory=datetime.now)


@dataclass
class TradingDecision:
    """Trading decision with full context"""
    symbol: str
    action: str  # 'BUY', 'SELL', 'HOLD'
    size: float
    confidence: float
    reasoning: List[str]
    risk_score: float
    expected_return: float
    max_loss: float
    timestamp: datetime = field(default_factory=datetime.now)


class MasterTradingSystem:
    """
    Professional Master Trading System
    Orchestrates all components with fault tolerance and monitoring
    """
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.state = SystemState.INITIALIZING
        
        # Core components
        self.api_client: Optional[AsyncBitgetClient] = None
        self.risk_manager: Optional[ProfessionalRiskManager] = None
        self.performance_tracker: Optional[PerformanceTracker] = None
        self.signal_generator: Optional[SignalGenerator] = None
        self.position_manager: Optional[PositionManager] = None
        
        # System state
        self.metrics = SystemMetrics()
        self.active_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        self.trading_enabled = True
        self.last_health_check = datetime.now()
        
        # Performance tracking
        self.trade_history: List[Dict] = []
        self.error_count = 0
        self.max_errors = 10
        
        logger.info("MasterTradingSystem initialized")
    
    async def initialize(self) -> bool:
        """Initialize all system components"""
        try:
            logger.info("🚀 Initializing Master Trading System...")
            
            # Initialize core components
            self.api_client = AsyncBitgetClient(self.settings)
            self.risk_manager = ProfessionalRiskManager(self.settings)
            self.performance_tracker = PerformanceTracker()
            self.signal_generator = SignalGenerator(self.settings)
            self.position_manager = PositionManager(self.settings)
            
            # Test connectivity
            test_ticker = await self.api_client.get_futures_ticker('BTCUSDT')
            if not test_ticker:
                raise Exception("Failed to connect to exchange")
            
            # Initialize market data
            await self._initialize_market_data()
            
            self.state = SystemState.ACTIVE
            logger.success("✅ Master Trading System initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            self.state = SystemState.ERROR
            return False
    
    async def _initialize_market_data(self) -> None:
        """Initialize historical market data for all symbols"""
        logger.info("📊 Initializing market data...")
        
        for symbol in self.active_symbols:
            try:
                # Get recent klines for technical analysis
                klines = await self.api_client.get_klines(symbol, '1m', 100)
                
                # Update risk manager with price history
                for kline in klines:
                    self.risk_manager.update_price_data(
                        symbol, kline['close'], kline['timestamp']
                    )
                
                logger.debug(f"Loaded {len(klines)} data points for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to load data for {symbol}: {e}")
    
    async def run_trading_cycle(self) -> None:
        """Execute one complete trading cycle"""
        if self.state != SystemState.ACTIVE:
            return
        
        try:
            logger.debug("🔄 Starting trading cycle...")
            
            # 1. Update market data
            market_data = await self._collect_market_data()
            
            # 2. Generate trading signals
            signals = await self._generate_signals(market_data)
            
            # 3. Process each signal
            for signal in signals:
                await self._process_trading_signal(signal, market_data)
            
            # 4. Update system metrics
            await self._update_system_metrics()
            
            # 5. Perform health checks
            await self._health_check()
            
            logger.debug("✅ Trading cycle completed")
            
        except Exception as e:
            logger.error(f"❌ Trading cycle error: {e}")
            self.error_count += 1
            
            if self.error_count > self.max_errors:
                logger.critical("🚨 Too many errors, pausing system")
                self.state = SystemState.PAUSED
    
    async def _collect_market_data(self) -> Dict[str, Any]:
        """Collect real-time market data"""
        tickers = await self.api_client.get_multiple_tickers(self.active_symbols)
        account_balance = await self.api_client.get_account_balance()
        
        # Update risk manager with latest prices
        for symbol, ticker in tickers.items():
            if ticker:
                self.risk_manager.update_price_data(symbol, ticker.price, ticker.timestamp)
        
        return {
            'tickers': tickers,
            'balance': account_balance,
            'timestamp': datetime.now()
        }
    
    async def _generate_signals(self, market_data: Dict[str, Any]) -> List[TradingSignal]:
        """Generate trading signals from market data"""
        signals = []
        
        for symbol in self.active_symbols:
            ticker = market_data['tickers'].get(symbol)
            if not ticker:
                continue
            
            try:
                # Get recent price history for signal generation
                klines = await self.api_client.get_klines(symbol, '1m', 50)
                
                # Generate signal using technical analysis
                signal = await self.signal_generator.generate_signal(symbol, klines, ticker)
                
                if signal and signal.action != 'HOLD':
                    signals.append(signal)
                    logger.info(f"📈 Generated {signal.action} signal for {symbol} (confidence: {signal.confidence:.2f})")
                
            except Exception as e:
                logger.error(f"Signal generation error for {symbol}: {e}")
        
        return signals
    
    async def _process_trading_signal(self, signal: TradingSignal, market_data: Dict[str, Any]) -> None:
        """Process a trading signal with risk management"""
        try:
            # 1. Risk validation
            is_valid, warnings = self.risk_manager.validate_new_position(
                signal.symbol, signal.size, signal.leverage or 1
            )
            
            if not is_valid:
                logger.warning(f"❌ Signal rejected for {signal.symbol}: {warnings}")
                return
            
            # 2. Position sizing optimization
            kelly_size = self.risk_manager.calculate_kelly_position_size(
                symbol=signal.symbol,
                expected_return=signal.expected_return,
                win_probability=signal.win_probability,
                loss_probability=1 - signal.win_probability,
                avg_win=signal.avg_win,
                avg_loss=signal.avg_loss
            )
            
            # Use smaller of signal size and Kelly optimal
            optimal_size = min(signal.size, kelly_size)
            
            if optimal_size < 10:  # Minimum position size
                logger.debug(f"Position size too small for {signal.symbol}: ${optimal_size:.2f}")
                return
            
            # 3. Execute trade
            if self.trading_enabled:
                await self._execute_trade(signal, optimal_size)
            else:
                logger.info(f"📝 Paper trade: {signal.action} {optimal_size:.0f} {signal.symbol}")
            
        except Exception as e:
            logger.error(f"Error processing signal for {signal.symbol}: {e}")
    
    async def _execute_trade(self, signal: TradingSignal, size: float) -> None:
        """Execute actual trade through API"""
        try:
            side = OrderSide.LONG if signal.action == 'BUY' else OrderSide.SHORT
            
            # Place market order
            result = await self.api_client.place_order(
                symbol=signal.symbol,
                side=side,
                size=size / 50000,  # Convert to contract size
                order_type=OrderType.MARKET
            )
            
            if result:
                logger.success(f"✅ Trade executed: {signal.action} {size:.0f} {signal.symbol} (Order: {result.order_id})")
                
                # Record trade
                trade_record = {
                    'timestamp': datetime.now(),
                    'symbol': signal.symbol,
                    'action': signal.action,
                    'size': size,
                    'order_id': result.order_id,
                    'confidence': signal.confidence,
                    'reasoning': signal.reasoning
                }
                self.trade_history.append(trade_record)
                
                # Update position manager
                await self.position_manager.update_position(signal.symbol, size, side.value)
                
            else:
                logger.error(f"❌ Failed to execute trade for {signal.symbol}")
                
        except Exception as e:
            logger.error(f"Trade execution error: {e}")
    
    async def _update_system_metrics(self) -> None:
        """Update real-time system metrics"""
        try:
            # Get current portfolio state
            balance = await self.api_client.get_account_balance()
            
            if balance and 'USDT' in balance:
                self.metrics.portfolio_value = balance['USDT']['equity']
                self.metrics.daily_pnl = balance['USDT']['unrealized_pnl']
            
            # Calculate performance metrics
            portfolio_metrics = self.risk_manager.calculate_portfolio_metrics()
            
            self.metrics.sharpe_ratio = portfolio_metrics.sharpe_ratio
            self.metrics.max_drawdown = portfolio_metrics.max_drawdown
            self.metrics.win_rate = self.performance_tracker.get_win_rate()
            self.metrics.total_trades = len(self.trade_history)
            self.metrics.active_positions = len(self.risk_manager.current_positions)
            self.metrics.last_update = datetime.now()
            
        except Exception as e:
            logger.error(f"Metrics update error: {e}")
    
    async def _health_check(self) -> None:
        """Perform system health checks"""
        try:
            current_time = datetime.now()
            
            # Check if we're in crisis regime
            regime = self.risk_manager.detect_market_regime()
            if regime == MarketRegime.CRISIS:
                logger.warning("⚠️ Market crisis detected - reducing risk")
                self.trading_enabled = False
            else:
                self.trading_enabled = True
            
            # Check drawdown limits
            if self.metrics.max_drawdown < -0.10:  # -10% max drawdown
                logger.critical("🚨 Maximum drawdown exceeded - pausing trading")
                self.state = SystemState.PAUSED
            
            # Reset error count if system is stable
            if (current_time - self.last_health_check).seconds > 300:  # 5 minutes
                self.error_count = max(0, self.error_count - 1)
            
            self.last_health_check = current_time
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
    
    async def run_continuous(self, cycle_interval: int = 60) -> None:
        """Run system continuously with specified cycle interval"""
        logger.info(f"🚀 Starting continuous trading (cycle: {cycle_interval}s)")
        
        if not await self.initialize():
            return
        
        try:
            while self.state in [SystemState.ACTIVE, SystemState.PAUSED]:
                if self.state == SystemState.ACTIVE:
                    await self.run_trading_cycle()
                
                # Log system status
                if self.metrics.total_trades % 10 == 0:  # Every 10 trades
                    await self._log_system_status()
                
                await asyncio.sleep(cycle_interval)
                
        except KeyboardInterrupt:
            logger.info("🛑 System shutdown requested")
        except Exception as e:
            logger.critical(f"💥 System crashed: {e}")
        finally:
            await self.shutdown()
    
    async def _log_system_status(self) -> None:
        """Log current system status"""
        logger.info(f"""
📊 SYSTEM STATUS REPORT
{'='*50}
State: {self.state.value.upper()}
Portfolio Value: ${self.metrics.portfolio_value:,.2f}
Daily P&L: ${self.metrics.daily_pnl:,.2f}
Total Trades: {self.metrics.total_trades}
Win Rate: {self.metrics.win_rate:.1%}
Sharpe Ratio: {self.metrics.sharpe_ratio:.2f}
Active Positions: {self.metrics.active_positions}
Max Drawdown: {self.metrics.max_drawdown:.2%}
Market Regime: {self.risk_manager.current_regime.value.upper()}
{'='*50}
        """)
    
    async def get_system_report(self) -> Dict[str, Any]:
        """Generate comprehensive system report"""
        portfolio_metrics = self.risk_manager.calculate_portfolio_metrics()
        risk_report = self.risk_manager.generate_risk_report()
        
        return {
            'system_metrics': {
                'state': self.state.value,
                'portfolio_value': self.metrics.portfolio_value,
                'daily_pnl': self.metrics.daily_pnl,
                'total_pnl': self.metrics.total_pnl,
                'win_rate': self.metrics.win_rate,
                'total_trades': self.metrics.total_trades,
                'active_positions': self.metrics.active_positions,
                'error_count': self.error_count
            },
            'risk_metrics': {
                'var_95': portfolio_metrics.var_95,
                'sharpe_ratio': portfolio_metrics.sharpe_ratio,
                'max_drawdown': portfolio_metrics.max_drawdown,
                'correlation_risk': portfolio_metrics.correlation_risk,
                'market_regime': self.risk_manager.current_regime.value
            },
            'recent_trades': self.trade_history[-10:],  # Last 10 trades
            'risk_report': risk_report,
            'timestamp': datetime.now().isoformat()
        }
    
    async def shutdown(self) -> None:
        """Graceful system shutdown"""
        logger.info("🛑 Initiating system shutdown...")
        
        self.state = SystemState.SHUTDOWN
        
        # Close all connections
        if self.api_client:
            await self.api_client.close()
        
        # Save final state
        final_report = await self.get_system_report()
        logger.info("💾 Final system state saved")
        
        logger.success("✅ System shutdown complete")


# Example usage and testing
async def main():
    """Run the master trading system"""
    settings = get_settings()
    system = MasterTradingSystem(settings)
    
    # Run for 1 hour (for testing)
    try:
        await system.run_continuous(cycle_interval=30)  # 30-second cycles
    except KeyboardInterrupt:
        logger.info("System stopped by user")


if __name__ == "__main__":
    asyncio.run(main())