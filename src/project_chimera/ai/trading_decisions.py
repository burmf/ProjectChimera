"""
Trading decisions processor for AI-driven trading
Handles market context aggregation and decision processing
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json

from ..streams.message_types import (
    MarketDataMessage, NewsMessage, XPostMessage, AIDecisionMessage
)
from ..domains.market import MarketFrame

logger = logging.getLogger(__name__)


@dataclass
class MarketContext:
    """
    Aggregated market context for AI decision making
    Contains all relevant data from the last period
    """
    
    # Core market data
    symbol: str
    timestamp: datetime
    current_price: float
    bid: float
    ask: float
    volume: float
    
    # Price history
    price_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Order book data
    orderbook_imbalance: Optional[float] = None
    orderbook_bids: List[tuple[float, float]] = field(default_factory=list)
    orderbook_asks: List[tuple[float, float]] = field(default_factory=list)
    
    # Funding and derivatives
    funding_rate: Optional[float] = None
    open_interest: Optional[float] = None
    
    # News and sentiment
    recent_news: List[NewsMessage] = field(default_factory=list)
    recent_x_posts: List[XPostMessage] = field(default_factory=list)
    
    # Technical indicators (if available)
    technical_indicators: Dict[str, float] = field(default_factory=dict)
    
    # Portfolio context
    current_positions: List[Dict[str, Any]] = field(default_factory=list)
    portfolio_value: float = 0.0
    portfolio_pnl: float = 0.0
    
    def get_market_summary(self) -> Dict[str, Any]:
        """Get summary of market conditions"""
        return {
            "symbol": self.symbol,
            "price": self.current_price,
            "spread": self.ask - self.bid,
            "volume": self.volume,
            "imbalance": self.orderbook_imbalance,
            "funding_rate": self.funding_rate,
            "news_count": len(self.recent_news),
            "posts_count": len(self.recent_x_posts),
            "price_history_count": len(self.price_history)
        }
    
    def get_sentiment_summary(self) -> Dict[str, Any]:
        """Get sentiment analysis summary"""
        if not self.recent_x_posts:
            return {"sentiment": 0.0, "engagement": 0.0, "posts": 0}
        
        sentiments = [p.sentiment_score for p in self.recent_x_posts if p.sentiment_score is not None]
        engagements = [p.engagement_score for p in self.recent_x_posts]
        
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0
        avg_engagement = sum(engagements) / len(engagements) if engagements else 0.0
        
        return {
            "sentiment": avg_sentiment,
            "engagement": avg_engagement, 
            "posts": len(self.recent_x_posts),
            "sentiment_range": [min(sentiments), max(sentiments)] if sentiments else [0, 0]
        }
    
    def get_news_summary(self) -> Dict[str, Any]:
        """Get news relevance summary"""
        if not self.recent_news:
            return {"relevance": 0.0, "count": 0}
        
        relevances = [n.relevance_score for n in self.recent_news if n.relevance_score is not None]
        avg_relevance = sum(relevances) / len(relevances) if relevances else 0.0
        
        return {
            "relevance": avg_relevance,
            "count": len(self.recent_news),
            "high_relevance_count": len([r for r in relevances if r > 0.5])
        }


class TradingDecisionProcessor:
    """
    Processes market data and generates trading decisions
    Manages context aggregation and decision caching
    """
    
    def __init__(self):
        self.market_contexts: Dict[str, MarketContext] = {}
        self.decision_cache: Dict[str, AIDecisionMessage] = {}
        self.context_history: Dict[str, List[MarketContext]] = {}
        
        # Configuration
        self.max_context_history = 100
        self.context_timeout_minutes = 10
        self.price_history_limit = 50
        self.news_history_hours = 2
        self.posts_history_hours = 1
    
    async def update_market_data(self, market_msg: MarketDataMessage):
        """Update market context with new market data"""
        symbol = market_msg.symbol
        
        if symbol not in self.market_contexts:
            self.market_contexts[symbol] = MarketContext(
                symbol=symbol,
                timestamp=market_msg.timestamp,
                current_price=market_msg.last,
                bid=market_msg.bid,
                ask=market_msg.ask,
                volume=market_msg.volume
            )
        
        context = self.market_contexts[symbol]
        
        # Update current market data
        context.timestamp = market_msg.timestamp
        context.current_price = market_msg.last
        context.bid = market_msg.bid
        context.ask = market_msg.ask
        context.volume = market_msg.volume
        context.orderbook_imbalance = market_msg.orderbook_imbalance
        context.funding_rate = market_msg.funding_rate
        
        # Add to price history
        price_point = {
            "timestamp": market_msg.timestamp,
            "open": market_msg.last,  # Approximation for streaming data
            "high": market_msg.last,
            "low": market_msg.last,
            "close": market_msg.last,
            "volume": market_msg.volume
        }
        
        context.price_history.append(price_point)
        
        # Limit history size
        if len(context.price_history) > self.price_history_limit:
            context.price_history = context.price_history[-self.price_history_limit:]
        
        logger.debug(f"Updated market context for {symbol}")
    
    async def update_news_data(self, news_msg: NewsMessage):
        """Update all contexts with news data"""
        # News affects all symbols, so update all contexts
        cutoff_time = datetime.now() - timedelta(hours=self.news_history_hours)
        
        for context in self.market_contexts.values():
            # Add news to context
            context.recent_news.append(news_msg)
            
            # Remove old news
            context.recent_news = [
                n for n in context.recent_news 
                if n.timestamp > cutoff_time
            ]
        
        logger.debug(f"Updated news data: {news_msg.title[:50]}...")
    
    async def update_x_posts_data(self, post_msg: XPostMessage):
        """Update all contexts with X posts data"""
        # X posts affect all symbols
        cutoff_time = datetime.now() - timedelta(hours=self.posts_history_hours)
        
        for context in self.market_contexts.values():
            # Add post to context
            context.recent_x_posts.append(post_msg)
            
            # Remove old posts
            context.recent_x_posts = [
                p for p in context.recent_x_posts
                if p.timestamp > cutoff_time
            ]
        
        logger.debug(f"Updated X posts data: {post_msg.text[:30]}...")
    
    async def update_portfolio_context(self, symbol: str, positions: List[Dict], 
                                     portfolio_value: float, pnl: float):
        """Update portfolio context for decision making"""
        if symbol in self.market_contexts:
            context = self.market_contexts[symbol]
            context.current_positions = positions
            context.portfolio_value = portfolio_value
            context.portfolio_pnl = pnl
    
    def get_market_context(self, symbol: str) -> Optional[MarketContext]:
        """Get current market context for symbol"""
        context = self.market_contexts.get(symbol)
        
        if context:
            # Check if context is too old
            age_minutes = (datetime.now() - context.timestamp).total_seconds() / 60
            if age_minutes > self.context_timeout_minutes:
                logger.warning(f"Market context for {symbol} is {age_minutes:.1f} minutes old")
        
        return context
    
    def is_context_ready_for_decision(self, symbol: str) -> bool:
        """Check if context has enough data for trading decision"""
        context = self.get_market_context(symbol)
        
        if not context:
            return False
        
        # Must have recent price data
        if not context.price_history:
            return False
        
        # Context should not be too old
        age_minutes = (datetime.now() - context.timestamp).total_seconds() / 60
        if age_minutes > 5:  # Max 5 minutes old for 1-min decisions
            return False
        
        # Should have basic market data
        if context.current_price <= 0:
            return False
        
        return True
    
    def prepare_1min_decision_context(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Prepare context data for 1-minute trading decision"""
        context = self.get_market_context(symbol)
        
        if not context or not self.is_context_ready_for_decision(symbol):
            return None
        
        # Format data for AI prompt
        decision_context = {
            "market_context": {
                "symbol": context.symbol,
                "price": context.current_price,
                "bid": context.bid,
                "ask": context.ask,
                "spread": context.ask - context.bid,
                "volume": context.volume,
                "imbalance": context.orderbook_imbalance,
                "funding_rate": context.funding_rate,
                "timestamp": context.timestamp.isoformat()
            },
            "price_data": context.price_history[-20:],  # Last 20 data points
            "orderbook_data": {
                "bids": context.orderbook_bids[:10],  # Top 10 levels
                "asks": context.orderbook_asks[:10],
                "imbalance": context.orderbook_imbalance
            },
            "sentiment_data": {
                "news": [
                    {
                        "title": n.title,
                        "relevance": n.relevance_score,
                        "timestamp": n.timestamp.isoformat()
                    }
                    for n in context.recent_news[-5:]  # Last 5 news items
                ],
                "x_posts": [
                    {
                        "text": p.text[:100],  # First 100 chars
                        "sentiment": p.sentiment_score,
                        "engagement": p.engagement_score,
                        "timestamp": p.timestamp.isoformat()
                    }
                    for p in context.recent_x_posts[-5:]  # Last 5 posts
                ]
            },
            "position_data": {
                "current_positions": context.current_positions,
                "portfolio_value": context.portfolio_value,
                "portfolio_pnl": context.portfolio_pnl
            }
        }
        
        return decision_context
    
    def prepare_1hour_strategy_context(self, symbols: List[str]) -> Optional[Dict[str, Any]]:
        """Prepare context data for 1-hour strategy planning"""
        if not symbols:
            return None
        
        # Aggregate data from multiple symbols
        all_contexts = [self.get_market_context(symbol) for symbol in symbols]
        valid_contexts = [c for c in all_contexts if c is not None]
        
        if not valid_contexts:
            return None
        
        # Aggregate news and sentiment across all symbols
        all_news = []
        all_posts = []
        
        for context in valid_contexts:
            all_news.extend(context.recent_news)
            all_posts.extend(context.recent_x_posts)
        
        # Remove duplicates and sort by timestamp
        unique_news = {n.url: n for n in all_news}.values()
        unique_posts = {p.post_id: p for p in all_posts}.values()
        
        sorted_news = sorted(unique_news, key=lambda x: x.timestamp, reverse=True)
        sorted_posts = sorted(unique_posts, key=lambda x: x.timestamp, reverse=True)
        
        strategy_context = {
            "market_overview": {
                "symbols": symbols,
                "contexts_available": len(valid_contexts),
                "timestamp": datetime.now().isoformat()
            },
            "technical_analysis": {
                symbol: {
                    "price": context.current_price,
                    "volume": context.volume,
                    "price_history": context.price_history[-50:],  # More history for strategy
                    "technical_indicators": context.technical_indicators
                }
                for symbol, context in zip(symbols, valid_contexts)
            },
            "news_sentiment": {
                "news_count": len(sorted_news),
                "posts_count": len(sorted_posts),
                "recent_news": [
                    {
                        "title": n.title,
                        "content": n.content[:200],  # First 200 chars
                        "relevance": n.relevance_score,
                        "source": n.source,
                        "timestamp": n.timestamp.isoformat()
                    }
                    for n in sorted_news[:10]  # Top 10 news items
                ],
                "sentiment_summary": self._aggregate_sentiment(sorted_posts)
            },
            "portfolio_state": {
                "total_positions": sum(len(c.current_positions) for c in valid_contexts),
                "total_portfolio_value": sum(c.portfolio_value for c in valid_contexts),
                "total_pnl": sum(c.portfolio_pnl for c in valid_contexts)
            }
        }
        
        return strategy_context
    
    def _aggregate_sentiment(self, posts: List[XPostMessage]) -> Dict[str, Any]:
        """Aggregate sentiment from X posts"""
        if not posts:
            return {"avg_sentiment": 0.0, "avg_engagement": 0.0, "total_posts": 0}
        
        sentiments = [p.sentiment_score for p in posts if p.sentiment_score is not None]
        engagements = [p.engagement_score for p in posts]
        
        return {
            "avg_sentiment": sum(sentiments) / len(sentiments) if sentiments else 0.0,
            "avg_engagement": sum(engagements) / len(engagements) if engagements else 0.0,
            "total_posts": len(posts),
            "sentiment_distribution": {
                "positive": len([s for s in sentiments if s > 0.1]),
                "neutral": len([s for s in sentiments if -0.1 <= s <= 0.1]),
                "negative": len([s for s in sentiments if s < -0.1])
            }
        }
    
    def cleanup_old_contexts(self):
        """Clean up old market contexts to prevent memory growth"""
        cutoff_time = datetime.now() - timedelta(hours=6)  # Keep 6 hours of context
        
        symbols_to_remove = []
        for symbol, context in self.market_contexts.items():
            if context.timestamp < cutoff_time:
                symbols_to_remove.append(symbol)
        
        for symbol in symbols_to_remove:
            del self.market_contexts[symbol]
            if symbol in self.context_history:
                del self.context_history[symbol]
        
        if symbols_to_remove:
            logger.info(f"Cleaned up {len(symbols_to_remove)} old market contexts")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processor statistics"""
        return {
            "active_contexts": len(self.market_contexts),
            "symbols": list(self.market_contexts.keys()),
            "oldest_context": min(
                (c.timestamp for c in self.market_contexts.values()),
                default=datetime.now()
            ).isoformat(),
            "total_news_items": sum(
                len(c.recent_news) for c in self.market_contexts.values()
            ),
            "total_x_posts": sum(
                len(c.recent_x_posts) for c in self.market_contexts.values()
            )
        }