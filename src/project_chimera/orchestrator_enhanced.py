"""
Enhanced Trading Orchestrator with Strategy Performance Tracking
Integrates performance monitoring and strategy management
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

from .settings import get_settings
from .datafeed.bitget_ws import BitgetWebSocketFeed
from .risk.unified_engine import UnifiedRiskEngine, UnifiedRiskConfig
from .execution.bitget import BitgetExecutionEngine
from .domains.market import Signal, MarketFrame
from .monitor.strategy_performance import get_performance_tracker, StrategyPerformanceTracker
from .strategies import (
    create_weekend_effect_strategy, create_stop_reversion_strategy, create_funding_contra_strategy,
    create_lob_reversion_strategy, create_volatility_breakout_strategy, 
    create_cme_gap_strategy, create_basis_arbitrage_strategy
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedTradingOrchestrator:
    """
    Enhanced Trading Orchestrator with Performance Tracking
    
    Features:
    - Strategy performance monitoring per trade
    - Real-time P&L tracking
    - Strategy enable/disable controls
    - Comprehensive trade history
    - Risk-adjusted performance metrics
    """
    
    def __init__(self, symbols: List[str], initial_portfolio_value: float = 150000.0):
        self.symbols = symbols
        self.portfolio_value = initial_portfolio_value
        self.settings = get_settings()
        
        # Performance tracking
        self.performance_tracker = get_performance_tracker()
        
        # Strategy registry with individual controls
        self.strategies = {}
        self.strategy_configs = {}
        self._init_strategies()
        
        # Core components
        self.data_feed = None
        self.risk_engine = None
        self.execution_engine = None
        
        # State management
        self._running = False
        self.stats = {
            'start_time': None,
            'market_frames_processed': 0,
            'signals_generated': 0,
            'signals_executed': 0,
            'trades_completed': 0,
            'errors': 0
        }
        
        # Real-time metrics
        self.real_time_metrics = {
            'total_pnl_usd': 0.0,
            'open_positions': 0,
            'daily_trades': 0,
            'last_trade_time': None
        }
    
    def _init_strategies(self):
        """Initialize all available strategies with individual controls"""
        strategy_factories = {
            'weekend_effect': create_weekend_effect_strategy,
            'stop_reversion': create_stop_reversion_strategy,
            'funding_contrarian': create_funding_contra_strategy,
            'lob_reversion': create_lob_reversion_strategy,
            'volatility_breakout': create_volatility_breakout_strategy,
            'cme_gap': create_cme_gap_strategy,
            'basis_arbitrage': create_basis_arbitrage_strategy
        }
        
        for strategy_id, factory in strategy_factories.items():
            try:
                strategy = factory()
                self.strategies[strategy_id] = strategy
                self.strategy_configs[strategy_id] = {
                    'enabled': True,
                    'max_position_pct': 2.0,
                    'min_confidence': 0.6,
                    'cooldown_hours': 1.0,
                    'max_daily_trades': 10,
                    'last_trade_time': None,
                    'daily_trade_count': 0
                }
                logger.info(f"Initialized strategy: {strategy_id}")
            except Exception as e:
                logger.error(f"Failed to initialize strategy {strategy_id}: {e}")
    
    async def start(self):
        """Start the enhanced orchestrator"""
        logger.info("Starting Enhanced Trading Orchestrator")
        self.stats['start_time'] = datetime.now()
        
        try:
            # Initialize core components
            await self._init_components()
            
            # Set up data flow
            self.data_feed.subscribe(self._process_market_data)
            
            self._running = True
            logger.info("Enhanced Trading Orchestrator started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start orchestrator: {e}")
            await self.stop()
            raise
    
    async def _init_components(self):
        """Initialize core trading components"""
        # Initialize data feed
        self.data_feed = BitgetWebSocketFeed("bitget_feed", {
            'api_key': self.settings.api.bitget_key.get_secret_value(),
            'secret_key': self.settings.api.bitget_secret.get_secret_value(),
            'passphrase': self.settings.api.bitget_passphrase.get_secret_value(),
            'sandbox': self.settings.api.bitget_sandbox
        })
        await self.data_feed.connect()
        
        # Initialize risk engine
        risk_config = UnifiedRiskConfig(
            kelly_base_fraction=self.settings.risk.kelly_fraction,
            atr_target_daily_vol=self.settings.risk.target_vol_daily,
            dd_warning_threshold=self.settings.risk.max_drawdown,
            max_leverage=self.settings.trading.leverage_max,
            min_confidence=self.settings.trading.confidence_threshold
        )
        self.risk_engine = UnifiedRiskEngine(risk_config, self.portfolio_value)
        
        # Initialize execution engine
        self.execution_engine = BitgetExecutionEngine({
            'api_key': self.settings.api.bitget_key.get_secret_value(),
            'secret_key': self.settings.api.bitget_secret.get_secret_value(),
            'passphrase': self.settings.api.bitget_passphrase.get_secret_value(),
            'sandbox': self.settings.api.bitget_sandbox
        })
    
    async def stop(self):
        """Stop the orchestrator"""
        logger.info("Stopping Enhanced Trading Orchestrator")
        self._running = False
        
        # Stop components
        if self.data_feed:
            await self.data_feed.disconnect()
        if self.execution_engine:
            await self.execution_engine.close()
        
        logger.info("Enhanced Trading Orchestrator stopped")
    
    async def _process_market_data(self, market_frame: MarketFrame):
        """Process market data through enhanced pipeline"""
        try:
            self.stats['market_frames_processed'] += 1
            
            # Update unrealized P&L for all strategies
            await self.performance_tracker.update_unrealized_pnl(
                market_frame.symbol, 
                market_frame.last
            )
            
            # Process each enabled strategy
            for strategy_id, strategy in self.strategies.items():
                if not self._is_strategy_ready(strategy_id):
                    continue
                
                try:
                    # Generate signal
                    signal = await self._generate_strategy_signal(strategy_id, strategy, market_frame)
                    
                    if signal:
                        await self._process_signal(strategy_id, signal, market_frame)
                        
                except Exception as e:
                    logger.error(f"Error processing strategy {strategy_id}: {e}")
                    self.stats['errors'] += 1
            
            # Update real-time metrics
            await self._update_real_time_metrics()
            
        except Exception as e:
            logger.error(f"Error in market data processing: {e}")
            self.stats['errors'] += 1
    
    def _is_strategy_ready(self, strategy_id: str) -> bool:
        """Check if strategy is ready to trade"""
        config = self.strategy_configs[strategy_id]
        
        # Check if enabled
        if not config['enabled']:
            return False
        
        # Check daily trade limit
        if config['daily_trade_count'] >= config['max_daily_trades']:
            return False
        
        # Check cooldown period
        if config['last_trade_time']:
            time_since_last = datetime.now() - config['last_trade_time']
            if time_since_last < timedelta(hours=config['cooldown_hours']):
                return False
        
        return True
    
    async def _generate_strategy_signal(self, strategy_id: str, strategy, market_frame: MarketFrame) -> Optional[Signal]:
        """Generate signal from strategy with performance tracking"""
        
        # Record signal generation attempt
        signal_id = await self.performance_tracker.record_signal_generated(
            Signal(
                symbol=market_frame.symbol,
                action="hold",  # Default
                confidence=0.0,
                timestamp=market_frame.timestamp,
                strategy_id=strategy_id,
                metadata={}
            ),
            market_frame
        )
        
        try:
            # Generate actual signal from strategy
            signal = strategy.generate_signal(market_frame)
            
            if signal and signal.confidence >= self.strategy_configs[strategy_id]['min_confidence']:
                signal.strategy_id = strategy_id
                signal.metadata['signal_id'] = signal_id
                self.stats['signals_generated'] += 1
                return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {strategy_id}: {e}")
        
        return None
    
    async def _process_signal(self, strategy_id: str, signal: Signal, market_frame: MarketFrame):
        """Process trading signal through risk management and execution"""
        
        # Risk management
        risk_decision = self.risk_engine.calculate_position_size(
            signal=signal,
            current_price=market_frame.last,
            portfolio_value=self.portfolio_value,
            timestamp=signal.timestamp
        )
        
        if not risk_decision.is_valid() or not risk_decision.can_trade:
            logger.info(f"Signal rejected by risk engine: {risk_decision.reasoning}")
            return
        
        # Calculate position size in USD
        position_size_usd = risk_decision.position_size_pct * self.portfolio_value
        position_size_native = position_size_usd / market_frame.last
        
        # Execute trade
        try:
            # Record trade entry
            signal_id = await self.performance_tracker.record_trade_entry(
                strategy_id=strategy_id,
                signal=signal,
                entry_price=market_frame.last,
                size_usd=position_size_usd,
                size_native=position_size_native,
                slippage_bps=2.0,  # Estimated slippage
                commission_usd=position_size_usd * 0.001  # 0.1% commission
            )
            
            # Update strategy config
            self.strategy_configs[strategy_id]['last_trade_time'] = datetime.now()
            self.strategy_configs[strategy_id]['daily_trade_count'] += 1
            
            self.stats['signals_executed'] += 1
            self.real_time_metrics['daily_trades'] += 1
            self.real_time_metrics['last_trade_time'] = datetime.now()
            
            logger.info(f"Trade executed: {strategy_id} {signal.action} ${position_size_usd:.2f}")
            
            # Simulate trade exit after some time (for demo purposes)
            asyncio.create_task(self._simulate_trade_exit(signal_id, market_frame.last))
            
        except Exception as e:
            logger.error(f"Failed to execute trade: {e}")
            self.stats['errors'] += 1
    
    async def _simulate_trade_exit(self, signal_id: str, entry_price: float):
        """Simulate trade exit for demonstration (replace with real exit logic)"""
        try:
            # Wait for random time (1-60 minutes)
            import random
            wait_time = random.uniform(60, 3600)  # 1-60 minutes
            await asyncio.sleep(wait_time)
            
            # Simulate price movement
            price_change = random.gauss(0, 0.02)  # 2% volatility
            exit_price = entry_price * (1 + price_change)
            
            # Record trade exit
            trade = await self.performance_tracker.record_trade_exit(
                signal_id=signal_id,
                exit_price=exit_price,
                commission_usd=1.0  # Exit commission
            )
            
            if trade:
                self.stats['trades_completed'] += 1
                logger.info(f"Trade completed: {signal_id} P&L: ${trade.pnl_usd:.2f}")
            
        except Exception as e:
            logger.error(f"Error in simulated trade exit: {e}")
    
    async def _update_real_time_metrics(self):
        """Update real-time performance metrics"""
        try:
            # Get performance summary
            summary = self.performance_tracker.get_performance_summary()
            
            self.real_time_metrics.update({
                'total_pnl_usd': summary.get('total_pnl_usd', 0.0),
                'open_positions': sum(
                    len(positions) for positions in 
                    self.performance_tracker.get_open_positions().values()
                )
            })
            
        except Exception as e:
            logger.error(f"Error updating real-time metrics: {e}")
    
    # Strategy Management API
    
    def enable_strategy(self, strategy_id: str) -> bool:
        """Enable a specific strategy"""
        if strategy_id in self.strategy_configs:
            self.strategy_configs[strategy_id]['enabled'] = True
            logger.info(f"Strategy {strategy_id} enabled")
            return True
        return False
    
    def disable_strategy(self, strategy_id: str) -> bool:
        """Disable a specific strategy"""
        if strategy_id in self.strategy_configs:
            self.strategy_configs[strategy_id]['enabled'] = False
            logger.info(f"Strategy {strategy_id} disabled")
            return True
        return False
    
    def update_strategy_config(self, strategy_id: str, config_updates: Dict[str, Any]) -> bool:
        """Update strategy configuration"""
        if strategy_id in self.strategy_configs:
            self.strategy_configs[strategy_id].update(config_updates)
            logger.info(f"Strategy {strategy_id} configuration updated: {config_updates}")
            return True
        return False
    
    def get_strategy_stats(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """Get strategy performance statistics"""
        if strategy_id:
            stats = self.performance_tracker.get_strategy_stats(strategy_id)
            return stats.to_dict() if stats else {}
        else:
            return {
                strategy_id: stats.to_dict() 
                for strategy_id, stats in self.performance_tracker.get_all_strategy_stats().items()
            }
    
    def get_open_positions(self, strategy_id: Optional[str] = None) -> Dict[str, List[Dict]]:
        """Get current open positions"""
        positions = self.performance_tracker.get_open_positions(strategy_id)
        return {
            strat_id: [trade.to_dict() for trade in trades]
            for strat_id, trades in positions.items()
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        
        # Component health
        components = {
            'data_feed': {'status': 'running' if self.data_feed else 'stopped'},
            'risk_engine': {'status': 'running' if self.risk_engine else 'stopped'},
            'execution_engine': {'status': 'running' if self.execution_engine else 'stopped'},
            'performance_tracker': {'status': 'running'}
        }
        
        # Strategy status
        strategy_status = {}
        for strategy_id, config in self.strategy_configs.items():
            strategy_status[strategy_id] = {
                'enabled': config['enabled'],
                'daily_trades': config['daily_trade_count'],
                'last_trade': config['last_trade_time'].isoformat() if config['last_trade_time'] else None
            }
        
        # Overall metrics
        uptime = (datetime.now() - self.stats['start_time']).total_seconds() if self.stats['start_time'] else 0
        
        return {
            'status': 'healthy' if self._running else 'stopped',
            'uptime_seconds': uptime,
            'components': components,
            'strategies': strategy_status,
            'stats': self.stats,
            'real_time_metrics': self.real_time_metrics,
            'timestamp': datetime.now().isoformat()
        }
    
    async def run_forever(self):
        """Run the orchestrator indefinitely with graceful shutdown"""
        try:
            await self.start()
            
            # Set up signal handlers
            loop = asyncio.get_event_loop()
            for sig in [signal.SIGTERM, signal.SIGINT]:
                loop.add_signal_handler(
                    sig, 
                    lambda: asyncio.create_task(self.stop())
                )
            
            # Main run loop
            while self._running:
                await asyncio.sleep(1.0)
                
                # Reset daily trade counts at midnight
                current_time = datetime.now()
                if current_time.hour == 0 and current_time.minute == 0:
                    for config in self.strategy_configs.values():
                        config['daily_trade_count'] = 0
                    self.real_time_metrics['daily_trades'] = 0
                
                # Health check logging
                if self.stats['market_frames_processed'] % 1000 == 0:
                    health = self.get_system_health()
                    logger.info(f"System health: {health['status']}, "
                              f"Processed: {self.stats['market_frames_processed']}, "
                              f"P&L: ${self.real_time_metrics['total_pnl_usd']:.2f}")
        
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
        except Exception as e:
            logger.error(f"Orchestrator error: {e}")
            traceback.print_exc()
        finally:
            await self.stop()


# Factory function for easy instantiation
def create_enhanced_orchestrator(symbols: List[str] = None, portfolio_value: float = 150000.0) -> EnhancedTradingOrchestrator:
    """Create enhanced orchestrator with default configuration"""
    if symbols is None:
        symbols = ['BTCUSDT', 'ETHUSDT']
    
    return EnhancedTradingOrchestrator(symbols, portfolio_value)


# CLI interface
async def main():
    """Main entry point for enhanced live trading"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Live Trading Orchestrator")
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT'], 
                       help='Trading symbols')
    parser.add_argument('--portfolio', type=float, default=150000.0,
                       help='Initial portfolio value')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create and run enhanced orchestrator
    orchestrator = create_enhanced_orchestrator(args.symbols, args.portfolio)
    
    try:
        await orchestrator.run_forever()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(asyncio.run(main()))