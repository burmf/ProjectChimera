# core/redis_manager.py
import redis
import json
import datetime
import os
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class RedisManager:
    def __init__(self, redis_url: str = None):
        """Initialize Redis connection"""
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        try:
            self.client = redis.from_url(self.redis_url, decode_responses=True)
            # Test connection
            self.client.ping()
            logger.info(f"Redis connected: {self.redis_url}")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            self.client = None

    def is_connected(self) -> bool:
        """Check if Redis is connected"""
        try:
            return self.client is not None and self.client.ping()
        except:
            return False

    def add_to_stream(self, stream_name: str, data: Dict[str, Any], maxlen: int = 10000) -> Optional[str]:
        """Add data to Redis stream with automatic trimming"""
        if not self.is_connected():
            return None
        
        try:
            # Add timestamp if not present
            if 'timestamp' not in data:
                data['timestamp'] = datetime.datetime.now().isoformat()
            
            # Convert all values to strings for Redis
            stream_data = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) 
                          for k, v in data.items()}
            
            # Add to stream with automatic trimming
            message_id = self.client.xadd(stream_name, stream_data, maxlen=maxlen, approximate=True)
            logger.debug(f"Added to stream {stream_name}: {message_id}")
            return message_id
        except Exception as e:
            logger.error(f"Failed to add to stream {stream_name}: {e}")
            return None

    def read_stream(self, stream_name: str, count: int = 10, start_id: str = '-') -> List[Dict]:
        """Read messages from Redis stream"""
        if not self.is_connected():
            return []
        
        try:
            # Read from stream
            messages = self.client.xread({stream_name: start_id}, count=count, block=0)
            
            result = []
            for stream, msgs in messages:
                for msg_id, fields in msgs:
                    # Parse JSON fields back to original types
                    parsed_fields = {}
                    for k, v in fields.items():
                        try:
                            # Try to parse as JSON first
                            parsed_fields[k] = json.loads(v)
                        except:
                            # If JSON parsing fails, keep as string
                            parsed_fields[k] = v
                    
                    result.append({
                        'id': msg_id,
                        'stream': stream,
                        'data': parsed_fields
                    })
            
            return result
        except Exception as e:
            logger.error(f"Failed to read from stream {stream_name}: {e}")
            return []

    def get_stream_info(self, stream_name: str) -> Dict[str, Any]:
        """Get information about a Redis stream"""
        if not self.is_connected():
            return {}
        
        try:
            info = self.client.xinfo_stream(stream_name)
            return {
                'length': info.get('length', 0),
                'first_entry': info.get('first-entry'),
                'last_entry': info.get('last-entry'),
                'groups': info.get('groups', 0)
            }
        except Exception as e:
            logger.debug(f"Stream {stream_name} info not available: {e}")
            return {'length': 0}

    def trim_stream(self, stream_name: str, maxlen: int = 1000):
        """Manually trim a Redis stream"""
        if not self.is_connected():
            return False
        
        try:
            self.client.xtrim(stream_name, maxlen=maxlen, approximate=True)
            logger.info(f"Trimmed stream {stream_name} to {maxlen} messages")
            return True
        except Exception as e:
            logger.error(f"Failed to trim stream {stream_name}: {e}")
            return False

    def set_cache(self, key: str, value: Any, ttl: int = 3600):
        """Set a cache value with TTL"""
        if not self.is_connected():
            return False
        
        try:
            serialized_value = json.dumps(value) if not isinstance(value, str) else value
            self.client.setex(key, ttl, serialized_value)
            return True
        except Exception as e:
            logger.error(f"Failed to set cache {key}: {e}")
            return False

    def get_cache(self, key: str) -> Any:
        """Get a cache value"""
        if not self.is_connected():
            return None
        
        try:
            value = self.client.get(key)
            if value is None:
                return None
            
            # Try to parse as JSON
            try:
                return json.loads(value)
            except:
                return value
        except Exception as e:
            logger.error(f"Failed to get cache {key}: {e}")
            return None

    def publish(self, channel: str, message: Any):
        """Publish message to Redis pub/sub channel"""
        if not self.is_connected():
            return False
        
        try:
            serialized_message = json.dumps(message) if not isinstance(message, str) else message
            self.client.publish(channel, serialized_message)
            return True
        except Exception as e:
            logger.error(f"Failed to publish to {channel}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get Redis usage statistics"""
        if not self.is_connected():
            return {}
        
        try:
            info = self.client.info()
            return {
                'used_memory': info.get('used_memory_human', 'N/A'),
                'connected_clients': info.get('connected_clients', 0),
                'total_commands_processed': info.get('total_commands_processed', 0),
                'uptime_in_seconds': info.get('uptime_in_seconds', 0)
            }
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {}

# Global Redis manager instance
redis_manager = RedisManager()

# Stream names constants
PRICE_STREAM = "prices"
NEWS_STREAM = "news" 
AI_DECISIONS_STREAM = "ai_decisions"
TRADE_SIGNALS_STREAM = "trade_signals"