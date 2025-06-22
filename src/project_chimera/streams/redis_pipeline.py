"""
Redis Streams pipeline for real-time data flow
Handles message publishing, consumption, and routing for the 4-layer trading system
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, AsyncGenerator
from dataclasses import dataclass
import redis.asyncio as redis
from redis.asyncio import Redis
import json

from .message_types import StreamMessage, MessageEncoder

logger = logging.getLogger(__name__)


@dataclass
class StreamConfig:
    """Configuration for Redis stream"""
    name: str
    max_length: int = 10000
    consumer_group: str = "default"
    consumer_name: str = "worker"
    block_ms: int = 1000
    count: int = 10


class StreamConsumer:
    """Base class for stream consumers"""
    
    def __init__(self, stream_config: StreamConfig, message_handler: Callable):
        self.config = stream_config
        self.message_handler = message_handler
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self, redis_client: Redis):
        """Start consuming messages"""
        self._running = True
        self._task = asyncio.create_task(self._consume_loop(redis_client))
        logger.info(f"Started consumer for stream: {self.config.name}")
    
    async def stop(self):
        """Stop consuming messages"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info(f"Stopped consumer for stream: {self.config.name}")
    
    async def _consume_loop(self, redis_client: Redis):
        """Main consumption loop"""
        while self._running:
            try:
                # Create consumer group if it doesn't exist
                try:
                    await redis_client.xgroup_create(
                        self.config.name,
                        self.config.consumer_group,
                        id="0",
                        mkstream=True
                    )
                except redis.exceptions.ResponseError:
                    # Group already exists
                    pass
                
                # Read messages
                messages = await redis_client.xreadgroup(
                    self.config.consumer_group,
                    self.config.consumer_name,
                    {self.config.name: ">"},
                    count=self.config.count,
                    block=self.config.block_ms
                )
                
                # Process messages
                for stream_name, stream_messages in messages:
                    for message_id, fields in stream_messages:
                        try:
                            await self._process_message(redis_client, message_id, fields)
                        except Exception as e:
                            logger.error(f"Error processing message {message_id}: {e}")
                            # Acknowledge message even if processing failed to avoid infinite retry
                            await redis_client.xack(self.config.name, self.config.consumer_group, message_id)
                            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in consumer loop: {e}")
                await asyncio.sleep(1)
    
    async def _process_message(self, redis_client: Redis, message_id: str, fields: Dict):
        """Process individual message"""
        try:
            # Decode message
            message_type = fields.get("message_type", "unknown")
            message = MessageEncoder.decode_message(message_type, fields)
            
            # Handle message
            await self.message_handler(message)
            
            # Acknowledge successful processing
            await redis_client.xack(self.config.name, self.config.consumer_group, message_id)
            
        except Exception as e:
            logger.error(f"Failed to process message {message_id}: {e}")
            raise


class RedisStreamPipeline:
    """
    Main Redis Streams pipeline for ProjectChimera
    Coordinates data flow between collectors, AI engine, and execution
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[Redis] = None
        self.consumers: Dict[str, StreamConsumer] = {}
        self.stream_configs = {
            "market_data": StreamConfig("market_data", max_length=50000),
            "news": StreamConfig("news", max_length=10000),
            "x_posts": StreamConfig("x_posts", max_length=10000),
            "ai_decisions": StreamConfig("ai_decisions", max_length=10000),
            "executions": StreamConfig("executions", max_length=10000),
            "risk_decisions": StreamConfig("risk_decisions", max_length=10000)
        }
        self._running = False
    
    async def start(self):
        """Start the pipeline"""
        logger.info("Starting Redis Streams pipeline")
        
        # Connect to Redis
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        
        # Test connection
        await self.redis_client.ping()
        
        self._running = True
        logger.info("Redis Streams pipeline started")
    
    async def stop(self):
        """Stop the pipeline"""
        logger.info("Stopping Redis Streams pipeline")
        
        self._running = False
        
        # Stop all consumers
        for consumer in self.consumers.values():
            await consumer.stop()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Redis Streams pipeline stopped")
    
    async def publish_message(self, stream_name: str, message: StreamMessage) -> str:
        """Publish message to stream"""
        if not self.redis_client:
            raise RuntimeError("Pipeline not started")
        
        # Encode message
        encoded_data = MessageEncoder.encode_message(message)
        
        # Add to stream
        message_id = await self.redis_client.xadd(
            stream_name,
            encoded_data,
            maxlen=self.stream_configs[stream_name].max_length
        )
        
        logger.debug(f"Published message to {stream_name}: {message_id}")
        return message_id
    
    async def publish_market_data(self, message) -> str:
        """Publish market data message"""
        return await self.publish_message("market_data", message)
    
    async def publish_news(self, message) -> str:
        """Publish news message"""
        return await self.publish_message("news", message)
    
    async def publish_x_post(self, message) -> str:
        """Publish X post message"""
        return await self.publish_message("x_posts", message)
    
    async def publish_ai_decision(self, message) -> str:
        """Publish AI decision message"""
        return await self.publish_message("ai_decisions", message)
    
    async def publish_execution(self, message) -> str:
        """Publish execution message"""
        return await self.publish_message("executions", message)
    
    async def publish_risk_decision(self, message) -> str:
        """Publish risk decision message"""
        return await self.publish_message("risk_decisions", message)
    
    def register_consumer(self, stream_name: str, consumer_group: str, 
                         consumer_name: str, message_handler: Callable) -> StreamConsumer:
        """Register a message consumer"""
        if stream_name not in self.stream_configs:
            raise ValueError(f"Unknown stream: {stream_name}")
        
        config = StreamConfig(
            name=stream_name,
            max_length=self.stream_configs[stream_name].max_length,
            consumer_group=consumer_group,
            consumer_name=consumer_name
        )
        
        consumer = StreamConsumer(config, message_handler)
        consumer_key = f"{stream_name}:{consumer_group}:{consumer_name}"
        self.consumers[consumer_key] = consumer
        
        return consumer
    
    async def start_consumer(self, consumer: StreamConsumer):
        """Start a specific consumer"""
        if not self.redis_client:
            raise RuntimeError("Pipeline not started")
        
        await consumer.start(self.redis_client)
    
    async def get_stream_info(self, stream_name: str) -> Dict[str, Any]:
        """Get information about a stream"""
        if not self.redis_client:
            raise RuntimeError("Pipeline not started")
        
        try:
            info = await self.redis_client.xinfo_stream(stream_name)
            return info
        except redis.exceptions.ResponseError:
            return {"exists": False}
    
    async def get_consumer_group_info(self, stream_name: str) -> List[Dict[str, Any]]:
        """Get consumer group information"""
        if not self.redis_client:
            raise RuntimeError("Pipeline not started")
        
        try:
            groups = await self.redis_client.xinfo_groups(stream_name)
            return groups
        except redis.exceptions.ResponseError:
            return []
    
    async def trim_stream(self, stream_name: str, max_length: int):
        """Trim stream to maximum length"""
        if not self.redis_client:
            raise RuntimeError("Pipeline not started")
        
        await self.redis_client.xtrim(stream_name, maxlen=max_length)
        logger.info(f"Trimmed stream {stream_name} to {max_length} messages")
    
    async def get_pending_messages(self, stream_name: str, consumer_group: str) -> List[Dict]:
        """Get pending messages for consumer group"""
        if not self.redis_client:
            raise RuntimeError("Pipeline not started")
        
        try:
            pending = await self.redis_client.xpending(stream_name, consumer_group)
            return pending
        except redis.exceptions.ResponseError:
            return []
    
    def is_running(self) -> bool:
        """Check if pipeline is running"""
        return self._running and self.redis_client is not None


# Utility functions for creating specific consumers
async def create_market_data_consumer(pipeline: RedisStreamPipeline, 
                                    handler: Callable) -> StreamConsumer:
    """Create consumer for market data stream"""
    return pipeline.register_consumer(
        "market_data", "processors", "market_processor", handler
    )


async def create_news_consumer(pipeline: RedisStreamPipeline, 
                             handler: Callable) -> StreamConsumer:
    """Create consumer for news stream"""
    return pipeline.register_consumer(
        "news", "ai_processors", "news_processor", handler
    )


async def create_ai_decision_consumer(pipeline: RedisStreamPipeline, 
                                    handler: Callable) -> StreamConsumer:
    """Create consumer for AI decisions stream"""
    return pipeline.register_consumer(
        "ai_decisions", "execution", "executor", handler
    )