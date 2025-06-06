# core/price_stream.py
import pandas as pd
import datetime
from typing import Dict, List, Optional
import logging
from .redis_manager import redis_manager, PRICE_STREAM

logger = logging.getLogger(__name__)

class PriceStreamManager:
    def __init__(self):
        self.stream_name = PRICE_STREAM
        
    def add_price_data(self, pair: str, ohlcv_data: Dict) -> bool:
        """Add price data to Redis stream"""
        try:
            price_message = {
                'pair': pair,
                'timestamp': ohlcv_data.get('timestamp', datetime.datetime.now().isoformat()),
                'open': float(ohlcv_data['open']),
                'high': float(ohlcv_data['high']),
                'low': float(ohlcv_data['low']),
                'close': float(ohlcv_data['close']),
                'volume': int(ohlcv_data.get('volume', 0)),
                'source': 'yfinance'
            }
            
            message_id = redis_manager.add_to_stream(self.stream_name, price_message)
            if message_id:
                logger.debug(f"Added price data for {pair}: {message_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to add price data for {pair}: {e}")
            return False
    
    def add_price_dataframe(self, pair: str, df: pd.DataFrame) -> int:
        """Add multiple price data points from DataFrame"""
        if df.empty:
            return 0
        
        added_count = 0
        for timestamp, row in df.iterrows():
            price_data = {
                'timestamp': timestamp.isoformat() if isinstance(timestamp, pd.Timestamp) else str(timestamp),
                'open': row['open'],
                'high': row['high'], 
                'low': row['low'],
                'close': row['close'],
                'volume': row.get('volume', 0)
            }
            
            if self.add_price_data(pair, price_data):
                added_count += 1
        
        logger.info(f"Added {added_count}/{len(df)} price data points for {pair}")
        return added_count
    
    def get_latest_prices(self, count: int = 10) -> List[Dict]:
        """Get latest price data from stream"""
        messages = redis_manager.read_stream(self.stream_name, count=count, start_id='-')
        
        prices = []
        for msg in messages:
            data = msg['data']
            prices.append({
                'id': msg['id'],
                'pair': data.get('pair'),
                'timestamp': data.get('timestamp'),
                'open': float(data.get('open', 0)),
                'high': float(data.get('high', 0)),
                'low': float(data.get('low', 0)),
                'close': float(data.get('close', 0)),
                'volume': int(data.get('volume', 0)),
                'source': data.get('source', 'unknown')
            })
        
        return prices
    
    def get_latest_price_for_pair(self, pair: str) -> Optional[Dict]:
        """Get the most recent price for a specific pair"""
        recent_prices = self.get_latest_prices(count=50)  # Get more to find the pair
        
        for price in recent_prices:
            if price['pair'] == pair:
                return price
        
        return None
    
    def trigger_technical_analysis(self, pair: str, price_data: Dict):
        """Trigger technical analysis calculation via Redis pub/sub"""
        try:
            analysis_message = {
                'action': 'calculate_technical_indicators',
                'pair': pair,
                'timestamp': datetime.datetime.now().isoformat(),
                'price_data': price_data
            }
            
            redis_manager.publish('technical_analysis', analysis_message)
            logger.debug(f"Triggered technical analysis for {pair}")
            
        except Exception as e:
            logger.error(f"Failed to trigger technical analysis for {pair}: {e}")
    
    def get_stream_stats(self) -> Dict:
        """Get price stream statistics"""
        info = redis_manager.get_stream_info(self.stream_name)
        
        # Get pair distribution from recent messages
        recent_prices = self.get_latest_prices(count=100)
        pair_counts = {}
        
        for price in recent_prices:
            pair = price.get('pair', 'unknown')
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        
        return {
            'stream_length': info.get('length', 0),
            'pair_distribution': pair_counts,
            'latest_update': recent_prices[0]['timestamp'] if recent_prices else None
        }
    
    def cleanup_old_data(self, keep_messages: int = 10000):
        """Cleanup old price data to manage memory"""
        return redis_manager.trim_stream(self.stream_name, maxlen=keep_messages)

# Global price stream manager
price_stream = PriceStreamManager()