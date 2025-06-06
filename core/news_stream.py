# core/news_stream.py
import datetime
from typing import Dict, List, Optional
import logging
from .redis_manager import redis_manager, NEWS_STREAM, AI_DECISIONS_STREAM

logger = logging.getLogger(__name__)

class NewsStreamManager:
    def __init__(self):
        self.news_stream = NEWS_STREAM
        self.ai_stream = AI_DECISIONS_STREAM
        
    def add_news_article(self, article_data: Dict) -> bool:
        """Add news article to Redis stream"""
        try:
            news_message = {
                'id': article_data['id'],
                'title': article_data['title'],
                'content': article_data.get('content', ''),
                'source': article_data.get('source', 'unknown'),
                'published_at': article_data['published_at'],
                'url': article_data.get('url', ''),
                'processed_for_ai': False,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            message_id = redis_manager.add_to_stream(self.news_stream, news_message)
            if message_id:
                logger.debug(f"Added news article: {article_data['id']}")
                # Trigger AI processing
                self.trigger_ai_analysis(article_data['id'], news_message)
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to add news article {article_data.get('id', 'unknown')}: {e}")
            return False
    
    def add_news_batch(self, articles: List[Dict]) -> int:
        """Add multiple news articles"""
        added_count = 0
        for article in articles:
            if self.add_news_article(article):
                added_count += 1
        
        logger.info(f"Added {added_count}/{len(articles)} news articles")
        return added_count
    
    def get_latest_news(self, count: int = 10) -> List[Dict]:
        """Get latest news from stream"""
        messages = redis_manager.read_stream(self.news_stream, count=count, start_id='-')
        
        news = []
        for msg in messages:
            data = msg['data']
            news.append({
                'id': msg['id'],
                'article_id': data.get('id'),
                'title': data.get('title'),
                'content': data.get('content'),
                'source': data.get('source'),
                'published_at': data.get('published_at'),
                'url': data.get('url'),
                'processed_for_ai': data.get('processed_for_ai', False),
                'timestamp': data.get('timestamp')
            })
        
        return news
    
    def get_unprocessed_news(self, count: int = 20) -> List[Dict]:
        """Get news articles that haven't been processed by AI"""
        recent_news = self.get_latest_news(count=count * 2)  # Get more to filter
        
        unprocessed = []
        for news in recent_news:
            if not news.get('processed_for_ai', False):
                unprocessed.append(news)
                if len(unprocessed) >= count:
                    break
        
        return unprocessed
    
    def trigger_ai_analysis(self, article_id: str, news_data: Dict):
        """Trigger AI analysis for news article via Redis pub/sub"""
        try:
            ai_message = {
                'action': 'analyze_news',
                'article_id': article_id,
                'timestamp': datetime.datetime.now().isoformat(),
                'news_data': news_data
            }
            
            redis_manager.publish('ai_analysis', ai_message)
            logger.debug(f"Triggered AI analysis for article: {article_id}")
            
        except Exception as e:
            logger.error(f"Failed to trigger AI analysis for {article_id}: {e}")
    
    def add_ai_decision(self, article_id: str, decision_data: Dict) -> bool:
        """Add AI trading decision to stream"""
        try:
            ai_message = {
                'article_id': article_id,
                'model_name': decision_data.get('model_name'),
                'trade_warranted': decision_data.get('trade_warranted', False),
                'pair': decision_data.get('pair', 'N/A'),
                'direction': decision_data.get('direction', 'N/A'),
                'confidence': float(decision_data.get('confidence', 0.0)),
                'reasoning': decision_data.get('reasoning', ''),
                'stop_loss_pips': int(decision_data.get('stop_loss_pips', 0)),
                'take_profit_pips': int(decision_data.get('take_profit_pips', 0)),
                'suggested_lot_size_factor': float(decision_data.get('suggested_lot_size_factor', 0.0)),
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            message_id = redis_manager.add_to_stream(self.ai_stream, ai_message)
            if message_id:
                logger.debug(f"Added AI decision for article: {article_id}")
                # Trigger trade signal generation if trade is warranted
                if decision_data.get('trade_warranted', False):
                    self.trigger_trade_signal_generation(article_id, ai_message)
                return True
            return False
            
        except Exception as e:
            logger.error(f"Failed to add AI decision for {article_id}: {e}")
            return False
    
    def get_latest_ai_decisions(self, count: int = 10) -> List[Dict]:
        """Get latest AI trading decisions"""
        messages = redis_manager.read_stream(self.ai_stream, count=count, start_id='-')
        
        decisions = []
        for msg in messages:
            data = msg['data']
            decisions.append({
                'id': msg['id'],
                'article_id': data.get('article_id'),
                'model_name': data.get('model_name'),
                'trade_warranted': data.get('trade_warranted', False),
                'pair': data.get('pair'),
                'direction': data.get('direction'),
                'confidence': float(data.get('confidence', 0)),
                'reasoning': data.get('reasoning'),
                'stop_loss_pips': int(data.get('stop_loss_pips', 0)),
                'take_profit_pips': int(data.get('take_profit_pips', 0)),
                'suggested_lot_size_factor': float(data.get('suggested_lot_size_factor', 0)),
                'timestamp': data.get('timestamp')
            })
        
        return decisions
    
    def trigger_trade_signal_generation(self, article_id: str, ai_decision: Dict):
        """Trigger trade signal generation for warranted trades"""
        try:
            signal_message = {
                'action': 'generate_trade_signal',
                'article_id': article_id,
                'ai_decision': ai_decision,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
            redis_manager.publish('trade_signals', signal_message)
            logger.debug(f"Triggered trade signal generation for article: {article_id}")
            
        except Exception as e:
            logger.error(f"Failed to trigger trade signal for {article_id}: {e}")
    
    def mark_article_processed(self, article_id: str):
        """Mark a news article as processed (via cache)"""
        cache_key = f"processed_article:{article_id}"
        redis_manager.set_cache(cache_key, True, ttl=86400)  # 24 hours
    
    def is_article_processed(self, article_id: str) -> bool:
        """Check if article has been processed"""
        cache_key = f"processed_article:{article_id}"
        return redis_manager.get_cache(cache_key) is not None
    
    def get_stream_stats(self) -> Dict:
        """Get news stream statistics"""
        news_info = redis_manager.get_stream_info(self.news_stream)
        ai_info = redis_manager.get_stream_info(self.ai_stream)
        
        # Get source distribution from recent news
        recent_news = self.get_latest_news(count=100)
        source_counts = {}
        
        for news in recent_news:
            source = news.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        
        # Get AI decision stats
        recent_decisions = self.get_latest_ai_decisions(count=100)
        trade_count = sum(1 for d in recent_decisions if d.get('trade_warranted', False))
        
        return {
            'news_stream_length': news_info.get('length', 0),
            'ai_decisions_length': ai_info.get('length', 0),
            'source_distribution': source_counts,
            'recent_trade_decisions': trade_count,
            'latest_news': recent_news[0]['timestamp'] if recent_news else None,
            'latest_ai_decision': recent_decisions[0]['timestamp'] if recent_decisions else None
        }
    
    def cleanup_old_data(self, keep_messages: int = 5000):
        """Cleanup old news and AI decision data"""
        news_trimmed = redis_manager.trim_stream(self.news_stream, maxlen=keep_messages)
        ai_trimmed = redis_manager.trim_stream(self.ai_stream, maxlen=keep_messages)
        return news_trimmed and ai_trimmed

# Global news stream manager
news_stream = NewsStreamManager()