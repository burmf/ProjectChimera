# workers/risk_monitor.py
import asyncio
import signal
import sys
import os
import logging
import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logging_config import setup_logging, get_trading_logger
from core.risk_manager import risk_manager
from core.redis_manager import redis_manager

# Setup logging
setup_logging()
logger = logging.getLogger("risk_monitor")
trading_logger = get_trading_logger("risk_monitor")

class RiskMonitor:
    def __init__(self):
        self.running = False
        self.check_interval = 10  # Check every 10 seconds
        
    async def start(self):
        """Start the risk monitoring worker"""
        logger.info("Starting Risk Monitor...")
        
        if not redis_manager.is_connected():
            logger.error("Redis not connected. Cannot start risk monitor.")
            return
        
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            while self.running:
                await self.monitor_cycle()
                await asyncio.sleep(self.check_interval)
                
        except Exception as e:
            logger.error(f"Risk monitor error: {e}")
        finally:
            await self.stop()
    
    async def monitor_cycle(self):
        """Perform one risk monitoring cycle"""
        try:
            # Update portfolio metrics
            risk_manager.update_portfolio_metrics()
            
            # Get current risk summary
            risk_summary = risk_manager.get_risk_summary()
            
            # Check for risk violations
            await self.check_risk_violations(risk_summary)
            
            # Check for trade signals that need risk evaluation
            await self.process_pending_signals()
            
            # Log metrics
            trading_logger.log_performance_metric({
                'name': 'daily_pnl',
                'value': risk_summary['current_status']['daily_pnl'],
                'unit': 'currency'
            })
            
            trading_logger.log_performance_metric({
                'name': 'current_drawdown',
                'value': risk_summary['current_status']['current_drawdown'],
                'unit': 'percentage'
            })
            
        except Exception as e:
            logger.error(f"Risk monitoring cycle error: {e}")
    
    async def check_risk_violations(self, risk_summary: dict):
        """Check for risk violations and take action"""
        try:
            current_status = risk_summary.get('current_status', {})
            
            # Check daily loss limit
            daily_loss_pct = current_status.get('daily_loss_pct', 0)
            if daily_loss_pct >= risk_manager.max_daily_loss_pct:
                risk_manager.emergency_stop(f"Daily loss limit exceeded: {daily_loss_pct:.2f}%")
                return
            
            # Check drawdown limit
            drawdown = current_status.get('current_drawdown', 0)
            if drawdown >= risk_manager.drawdown_limit_pct:
                risk_manager.emergency_stop(f"Maximum drawdown exceeded: {drawdown:.2f}%")
                return
            
            # Check position limits
            open_positions = current_status.get('open_positions_count', 0)
            if open_positions >= risk_manager.max_open_positions:
                trading_logger.log_risk_event({
                    'type': 'position_limit_warning',
                    'severity': 'warning',
                    'message': f"Approaching position limit: {open_positions}/{risk_manager.max_open_positions}"
                })
            
            # Warning thresholds (80% of limits)
            if daily_loss_pct >= risk_manager.max_daily_loss_pct * 0.8:
                trading_logger.log_risk_event({
                    'type': 'daily_loss_warning',
                    'severity': 'warning',
                    'message': f"Daily loss approaching limit: {daily_loss_pct:.2f}%"
                })
            
            if drawdown >= risk_manager.drawdown_limit_pct * 0.8:
                trading_logger.log_risk_event({
                    'type': 'drawdown_warning',
                    'severity': 'warning',
                    'message': f"Drawdown approaching limit: {drawdown:.2f}%"
                })
            
        except Exception as e:
            logger.error(f"Risk violation check error: {e}")
    
    async def process_pending_signals(self):
        """Process trade signals waiting for risk evaluation"""
        try:
            # Read pending trade signals
            messages = redis_manager.read_stream('trade_signals', count=10, start_id='>')
            
            for message in messages:
                signal_data = message['data']
                
                # Evaluate trade risk
                approved, risk_assessment = risk_manager.evaluate_trade_risk(signal_data)
                
                if approved:
                    # Update signal with risk-adjusted parameters
                    risk_adjusted_signal = {
                        **signal_data,
                        'risk_approved': True,
                        'adjusted_lot_size': risk_assessment['adjusted_lot_size'],
                        'risk_score': risk_assessment['risk_score'],
                        'risk_assessment': risk_assessment,
                        'processed_by_risk_manager': datetime.datetime.now().isoformat()
                    }
                    
                    # Send to execution queue
                    redis_manager.add_to_stream('approved_trades', risk_adjusted_signal)
                    
                    trading_logger.log_trade_signal({
                        'pair': signal_data.get('pair'),
                        'direction': signal_data.get('direction'),
                        'confidence': signal_data.get('confidence'),
                        'source': 'risk_approved',
                        'risk_score': risk_assessment['risk_score']
                    })
                    
                else:
                    # Log rejected signal
                    trading_logger.log_risk_event({
                        'type': 'trade_rejected',
                        'severity': 'info',
                        'message': f"Trade rejected for {signal_data.get('pair')}: {', '.join(risk_assessment.get('reasons', []))}",
                        'signal_data': signal_data
                    })
                    
                    # Add to rejected trades stream
                    rejected_signal = {
                        **signal_data,
                        'risk_approved': False,
                        'rejection_reasons': risk_assessment.get('reasons', []),
                        'processed_by_risk_manager': datetime.datetime.now().isoformat()
                    }
                    
                    redis_manager.add_to_stream('rejected_trades', rejected_signal)
                
        except Exception as e:
            logger.error(f"Signal processing error: {e}")
    
    async def stop(self):
        """Stop the risk monitor"""
        if not self.running:
            return
            
        logger.info("Stopping Risk Monitor...")
        self.running = False
        logger.info("Risk Monitor stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Risk monitor received signal {signum}, shutting down...")
        asyncio.create_task(self.stop())

async def main():
    """Main risk monitor entry point"""
    monitor = RiskMonitor()
    
    try:
        await monitor.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Risk monitor startup failed: {e}")
    finally:
        await monitor.stop()

if __name__ == "__main__":
    asyncio.run(main())