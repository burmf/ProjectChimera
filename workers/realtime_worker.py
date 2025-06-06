# workers/realtime_worker.py
import asyncio
import signal
import sys
import os
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logging_config import setup_logging, get_trading_logger
from core.realtime_processor import realtime_processor
from core.redis_manager import redis_manager
from core.risk_manager import risk_manager

# Setup logging
setup_logging()
logger = logging.getLogger("realtime_worker")
trading_logger = get_trading_logger("worker")

class RealtimeWorker:
    def __init__(self):
        self.running = False
        self.tasks = []
        
    async def start(self):
        """Start the realtime worker"""
        logger.info("Starting Realtime Worker...")
        
        # Check dependencies
        if not redis_manager.is_connected():
            logger.error("Redis not connected. Cannot start worker.")
            return
        
        self.running = True
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            # Start the realtime processor
            await realtime_processor.start_processing()
            
        except Exception as e:
            logger.error(f"Worker error: {e}")
            trading_logger.log_risk_event({
                'type': 'worker_error',
                'severity': 'critical',
                'message': f"Realtime worker failed: {e}"
            })
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the worker gracefully"""
        if not self.running:
            return
            
        logger.info("Stopping Realtime Worker...")
        self.running = False
        
        # Stop realtime processor
        await realtime_processor.stop_processing()
        
        # Cancel any running tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        logger.info("Realtime Worker stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self.stop())

async def main():
    """Main worker entry point"""
    worker = RealtimeWorker()
    
    try:
        await worker.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Worker startup failed: {e}")
    finally:
        await worker.stop()

if __name__ == "__main__":
    asyncio.run(main())