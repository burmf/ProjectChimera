# workers/ai_result_processor.py
import asyncio
import signal
import sys
import os
import logging
import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logging_config import setup_logging, get_trading_logger
from core.openai_manager import openai_manager
from core.redis_manager import redis_manager
from core.news_stream import news_stream

# Setup logging
setup_logging()
logger = logging.getLogger("ai_result_processor")
trading_logger = get_trading_logger("ai_result_processor")

class AIResultProcessor:
    def __init__(self):
        self.running = False
        self.check_interval = 5  # Check every 5 seconds
        
    async def start(self):
        """Start the AI result processor"""
        logger.info("Starting AI Result Processor...")
        
        if not redis_manager.is_connected():
            logger.error("Redis not connected. Cannot start processor.")
            return
        
        self.running = True
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        try:
            while self.running:
                await self.process_completed_requests()
                await asyncio.sleep(self.check_interval)
                
        except Exception as e:
            logger.error(f"AI result processor error: {e}")
        finally:
            await self.stop()
    
    async def process_completed_requests(self):
        """Process completed OpenAI requests and create AI decisions"""
        try:
            # Get keys matching the pattern for news analysis mappings
            pattern = "news_analysis_mapping:*"
            
            # Use Redis SCAN to find matching keys
            cursor = 0
            while True:
                cursor, keys = redis_manager.client.scan(
                    cursor=cursor, 
                    match=pattern, 
                    count=100
                )
                
                for key in keys:
                    await self.process_mapping_key(key)
                
                if cursor == 0:
                    break
                    
        except Exception as e:
            logger.error(f"Failed to process completed requests: {e}")
    
    async def process_mapping_key(self, mapping_key: str):
        """Process a single news analysis mapping"""
        try:
            # Parse the key: news_analysis_mapping:{article_id}:{model}
            parts = mapping_key.split(':')
            if len(parts) != 4:
                return
                
            article_id = parts[2]
            model = parts[3]
            
            # Get the request ID
            request_id = redis_manager.get_cache(mapping_key)
            if not request_id:
                return
            
            # Check if result is available
            result = redis_manager.get_cache(f"openai_result:{request_id}")
            if not result:
                return  # Still processing
            
            # Process the result
            await self.process_ai_result(article_id, model, result)
            
            # Clean up the mapping
            redis_manager.client.delete(mapping_key)
            
        except Exception as e:
            logger.error(f"Failed to process mapping key {mapping_key}: {e}")
    
    async def process_ai_result(self, article_id: str, model: str, result: Dict[str, Any]):
        """Process an AI analysis result and create decision"""
        try:
            if result.get('status') != 'success':
                logger.warning(f"AI request failed for article {article_id}, model {model}: {result.get('error', 'Unknown error')}")
                return
            
            # Extract the analysis data
            analysis_data = result.get('data', {})
            if not analysis_data:
                logger.warning(f"No analysis data for article {article_id}, model {model}")
                return
            
            # Validate required fields
            required_fields = ['trade_warranted', 'pair', 'direction', 'confidence']
            if not all(field in analysis_data for field in required_fields):
                logger.warning(f"Missing required fields in analysis for article {article_id}, model {model}")
                return
            
            # Create AI decision
            decision_data = {
                'model_name': model,
                'trade_warranted': analysis_data.get('trade_warranted', False),
                'pair': analysis_data.get('pair', 'N/A'),
                'direction': analysis_data.get('direction', 'N/A'),
                'confidence': float(analysis_data.get('confidence', 0.0)),
                'reasoning': analysis_data.get('reasoning', ''),
                'stop_loss_pips': int(analysis_data.get('stop_loss_pips', 0)),
                'take_profit_pips': int(analysis_data.get('take_profit_pips', 0)),
                'suggested_lot_size_factor': float(analysis_data.get('suggested_lot_size_factor', 0.0)),
                'processing_time': result.get('processing_time', 0),
                'cost_usd': result.get('usage', {}).get('cost_usd', 0)
            }
            
            # Add to news stream
            success = news_stream.add_ai_decision(article_id, decision_data)
            
            if success:
                # Mark article as processed
                news_stream.mark_article_processed(article_id)
                
                # Log the decision
                trading_logger.log_trade_signal({
                    'pair': decision_data['pair'],
                    'direction': decision_data['direction'],
                    'confidence': decision_data['confidence'],
                    'source': f'ai_analysis_{model}',
                    'article_id': article_id
                })
                
                logger.info(f"AI decision created for article {article_id} with model {model}")
                
                # If trade is warranted, trigger immediate signal processing
                if decision_data['trade_warranted'] and decision_data['confidence'] > 0.6:
                    await self.trigger_immediate_signal_processing(article_id, decision_data)
            
        except Exception as e:
            logger.error(f"Failed to process AI result for article {article_id}, model {model}: {e}")
    
    async def trigger_immediate_signal_processing(self, article_id: str, decision_data: Dict[str, Any]):
        """Trigger immediate processing for high-confidence trading decisions"""
        try:
            # Create an immediate processing signal
            immediate_signal = {
                'type': 'immediate_ai_decision',
                'article_id': article_id,
                'model_name': decision_data['model_name'],
                'confidence': decision_data['confidence'],
                'pair': decision_data['pair'],
                'direction': decision_data['direction'],
                'timestamp': datetime.datetime.now().isoformat(),
                'priority': 'high' if decision_data['confidence'] > 0.8 else 'normal'
            }
            
            # Publish for immediate processing
            redis_manager.publish('immediate_ai_signals', immediate_signal)
            
            logger.info(f"Triggered immediate signal processing for article {article_id} (confidence: {decision_data['confidence']:.2f})")
            
        except Exception as e:
            logger.error(f"Failed to trigger immediate signal processing: {e}")
    
    async def cleanup_old_mappings(self):
        """Clean up old mapping keys that may have been abandoned"""
        try:
            pattern = "news_analysis_mapping:*"
            cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=2)
            
            cursor = 0
            while True:
                cursor, keys = redis_manager.client.scan(
                    cursor=cursor, 
                    match=pattern, 
                    count=100
                )
                
                for key in keys:
                    # Check if the mapping is old
                    ttl = redis_manager.client.ttl(key)
                    if ttl < 1800:  # Less than 30 minutes remaining
                        redis_manager.client.delete(key)
                        logger.debug(f"Cleaned up old mapping: {key}")
                
                if cursor == 0:
                    break
                    
        except Exception as e:
            logger.error(f"Failed to cleanup old mappings: {e}")
    
    async def stop(self):
        """Stop the processor"""
        if not self.running:
            return
            
        logger.info("Stopping AI Result Processor...")
        self.running = False
        logger.info("AI Result Processor stopped")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"AI result processor received signal {signum}, shutting down...")
        asyncio.create_task(self.stop())

async def main():
    """Main processor entry point"""
    processor = AIResultProcessor()
    
    try:
        await processor.start()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Processor startup failed: {e}")
    finally:
        await processor.stop()

if __name__ == "__main__":
    asyncio.run(main())