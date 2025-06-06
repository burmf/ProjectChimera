# workers/openai_worker.py
import asyncio
import signal
import sys
import os
import logging

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logging_config import setup_logging, get_trading_logger
from core.openai_manager import openai_manager
from core.redis_manager import redis_manager

# Setup logging
setup_logging()
logger = logging.getLogger("openai_worker")
trading_logger = get_trading_logger("openai_worker")

class OpenAIWorker:
    def __init__(self):
        self.running = False
        
    async def start(self):
        """Start the OpenAI worker"""
        logger.info("Starting OpenAI Worker...")
        
        # Check dependencies
        if not redis_manager.is_connected():
            logger.error("Redis not connected. Cannot start worker.")
            return
        
        if not openai_manager.async_client:
            logger.error("OpenAI client not initialized. Cannot start worker.")
            return
        
        self.running = True
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            # Start OpenAI workers (3 workers for processing requests)
            await openai_manager.start_workers(num_workers=3)
            
            # Keep the worker running
            while self.running:
                await asyncio.sleep(10)
                
                # Log queue status periodically
                stats = openai_manager.get_usage_statistics(hours=1)
                if 'queue_status' in stats:
                    queue_status = stats['queue_status']
                    total_queued = sum(queue_status.values())
                    
                    if total_queued > 0:
                        logger.info(f"OpenAI queue status: {queue_status}")
                    
                    # Alert if queues are getting too long
                    if total_queued > 50:
                        trading_logger.log_risk_event({
                            'type': 'openai_queue_backlog',
                            'severity': 'warning',
                            'message': f"OpenAI request queue backlog: {total_queued} requests"
                        })
            
        except Exception as e:
            logger.error(f"OpenAI worker error: {e}")
            trading_logger.log_risk_event({
                'type': 'worker_error',
                'severity': 'critical',
                'message': f"OpenAI worker failed: {e}"
            })
        finally:
            await self.stop()
    
    async def stop(self):
        """Stop the worker gracefully"""
        if not self.running:
            return
            
        logger.info("Stopping OpenAI Worker...")
        self.running = False
        
        # Stop OpenAI workers
        await openai_manager.stop_workers()
        
        logger.info("OpenAI Worker stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        asyncio.create_task(self.stop())

async def main():
    """Main worker entry point"""
    worker = OpenAIWorker()
    
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