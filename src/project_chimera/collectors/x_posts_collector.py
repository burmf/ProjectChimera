"""
X/Twitter Posts Collector for cryptocurrency and trading sentiment
Collects and processes X posts every hour for AI sentiment analysis
"""

import asyncio
import logging
import hashlib
import subprocess
import json
import re
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
from dataclasses import dataclass

from ..streams.redis_pipeline import RedisStreamPipeline
from ..streams.message_types import XPostMessage

logger = logging.getLogger(__name__)


@dataclass
class XSearchQuery:
    """Configuration for X search query"""
    query: str
    name: str
    priority: int = 1
    tags: List[str] = None
    max_results: int = 100
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class XPostsCollector:
    """
    X/Twitter posts collector for cryptocurrency sentiment analysis
    Uses snscrape to collect posts and publishes to Redis Streams
    """
    
    def __init__(self, redis_pipeline: RedisStreamPipeline):
        self.redis_pipeline = redis_pipeline
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._seen_posts: Set[str] = set()
        
        # X search queries for crypto/trading content
        self.search_queries = [
            # Bitcoin queries
            XSearchQuery(
                query="bitcoin OR btc (price OR trading OR bullish OR bearish) -filter:retweets lang:en",
                name="bitcoin_sentiment",
                priority=1,
                tags=["bitcoin", "crypto", "sentiment"],
                max_results=50
            ),
            XSearchQuery(
                query="$BTC (pump OR dump OR moon OR crash OR breakout) -filter:retweets lang:en",
                name="bitcoin_signals",
                priority=1,
                tags=["bitcoin", "trading", "signals"],
                max_results=30
            ),
            
            # Ethereum queries
            XSearchQuery(
                query="ethereum OR eth (price OR defi OR gas) -filter:retweets lang:en",
                name="ethereum_sentiment",
                priority=1,
                tags=["ethereum", "crypto", "defi"],
                max_results=30
            ),
            XSearchQuery(
                query="$ETH (bullish OR bearish OR support OR resistance) -filter:retweets lang:en",
                name="ethereum_trading",
                priority=1,
                tags=["ethereum", "trading"],
                max_results=30
            ),
            
            # General crypto trading
            XSearchQuery(
                query="crypto (trading OR scalping OR futures OR leverage) -filter:retweets lang:en",
                name="crypto_trading",
                priority=2,
                tags=["crypto", "trading"],
                max_results=40
            ),
            XSearchQuery(
                query="altcoin (season OR rally OR rotation) -filter:retweets lang:en",
                name="altcoin_sentiment",
                priority=2,
                tags=["altcoins", "crypto"],
                max_results=25
            ),
            
            # Forex trading
            XSearchQuery(
                query="forex OR fx (usd OR eur OR jpy) (bullish OR bearish) -filter:retweets lang:en",
                name="forex_sentiment",
                priority=2,
                tags=["forex", "trading"],
                max_results=20
            ),
            
            # Market sentiment
            XSearchQuery(
                query="(crypto OR bitcoin) (fear OR greed OR fomo OR capitulation) -filter:retweets lang:en",
                name="market_sentiment",
                priority=1,
                tags=["sentiment", "psychology"],
                max_results=30
            ),
            
            # Whale activity
            XSearchQuery(
                query="whale (alert OR movement OR transfer) (bitcoin OR crypto) -filter:retweets lang:en",
                name="whale_activity",
                priority=1,
                tags=["whales", "onchain"],
                max_results=20
            )
        ]
        
        # Influential crypto accounts to monitor
        self.influential_accounts = [
            "@cz_binance", "@elonmusk", "@saylor", "@APompliano",
            "@VitalikButerin", "@justinsuntron", "@novogratz",
            "@CryptoCobain", "@CryptoWendyO", "@AltcoinDailyio",
            "@BlockchainCap", "@CoinBureau", "@intocryptoverse"
        ]
        
        # Keywords for engagement scoring
        self.high_impact_keywords = {
            "breaking", "urgent", "alert", "massive", "huge", "pump", "dump",
            "moon", "crash", "ath", "breakout", "squeeze", "liquidation"
        }
        
        self.sentiment_keywords = {
            "positive": {"bullish", "moon", "pump", "rally", "breakout", "buy", "long", "hodl"},
            "negative": {"bearish", "dump", "crash", "sell", "short", "capitulation", "bear"},
            "neutral": {"consolidation", "sideways", "range", "wait", "watch"}
        }
    
    async def start(self, collection_interval_hours: int = 1):
        """Start the X posts collector"""
        if self._running:
            logger.warning("X posts collector already running")
            return
        
        logger.info(f"Starting X posts collector (interval: {collection_interval_hours}h)")
        
        # Check if snscrape is available
        if not await self._check_snscrape():
            logger.error("snscrape not available - X posts collection disabled")
            return
        
        self._running = True
        self._task = asyncio.create_task(
            self._collection_loop(collection_interval_hours)
        )
    
    async def stop(self):
        """Stop the X posts collector"""
        logger.info("Stopping X posts collector")
        
        self._running = False
        
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def _check_snscrape(self) -> bool:
        """Check if snscrape is available"""
        try:
            result = await asyncio.create_subprocess_exec(
                "snscrape", "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await result.communicate()
            return result.returncode == 0
        except FileNotFoundError:
            return False
        except Exception as e:
            logger.error(f"Error checking snscrape: {e}")
            return False
    
    async def _collection_loop(self, interval_hours: int):
        """Main collection loop"""
        while self._running:
            try:
                start_time = datetime.now()
                
                # Collect from search queries
                total_posts = 0
                for query in self.search_queries:
                    try:
                        posts = await self._collect_from_query(query)
                        total_posts += posts
                        
                        # Small delay between queries
                        await asyncio.sleep(2)
                        
                    except Exception as e:
                        logger.error(f"Error collecting query '{query.name}': {e}")
                
                # Collect from influential accounts
                try:
                    account_posts = await self._collect_from_accounts()
                    total_posts += account_posts
                except Exception as e:
                    logger.error(f"Error collecting from accounts: {e}")
                
                collection_time = (datetime.now() - start_time).total_seconds()
                logger.info(f"Collected {total_posts} X posts in {collection_time:.1f}s")
                
                # Wait for next collection
                await asyncio.sleep(interval_hours * 3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in X collection loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    async def _collect_from_query(self, query: XSearchQuery) -> int:
        """Collect posts from a search query"""
        try:
            logger.debug(f"Collecting from query: {query.name}")
            
            # Calculate time range (last 2 hours to ensure overlap)
            since_time = datetime.now() - timedelta(hours=2)
            since_str = since_time.strftime("%Y-%m-%d_%H:%M:%S")
            
            # Build snscrape command
            cmd = [
                "snscrape",
                "--jsonl",
                "--max-results", str(query.max_results),
                "twitter-search",
                f"{query.query} since:{since_str}"
            ]
            
            # Execute snscrape
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                logger.warning(f"snscrape failed for {query.name}: {stderr.decode()}")
                return 0
            
            # Process results
            posts_processed = 0
            for line in stdout.decode().strip().split('\n'):
                if not line:
                    continue
                
                try:
                    post_data = json.loads(line)
                    post_message = await self._process_post(post_data, query)
                    
                    if post_message and self._is_relevant_post(post_message):
                        # Check for duplicates
                        post_hash = self._create_post_hash(post_data)
                        if post_hash not in self._seen_posts:
                            # Publish to Redis
                            await self.redis_pipeline.publish_x_post(post_message)
                            
                            # Track as seen
                            self._seen_posts.add(post_hash)
                            posts_processed += 1
                            
                            logger.debug(f"Published X post: {post_message.text[:50]}...")
                
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"Error processing post: {e}")
            
            return posts_processed
            
        except Exception as e:
            logger.error(f"Error collecting from query {query.name}: {e}")
            return 0
    
    async def _collect_from_accounts(self) -> int:
        """Collect recent posts from influential accounts"""
        total_posts = 0
        
        # Limit to a few key accounts per collection to avoid rate limits
        selected_accounts = self.influential_accounts[:3]  # Rotate in production
        
        for account in selected_accounts:
            try:
                posts = await self._collect_account_posts(account)
                total_posts += posts
                await asyncio.sleep(3)  # Delay between accounts
            except Exception as e:
                logger.error(f"Error collecting from {account}: {e}")
        
        return total_posts
    
    async def _collect_account_posts(self, account: str) -> int:
        """Collect posts from a specific account"""
        try:
            # Calculate time range
            since_time = datetime.now() - timedelta(hours=2)
            since_str = since_time.strftime("%Y-%m-%d_%H:%M:%S")
            
            # Build snscrape command for user timeline
            cmd = [
                "snscrape",
                "--jsonl",
                "--max-results", "10",
                "twitter-user",
                f"{account.replace('@', '')}",
                "--since", since_str
            ]
            
            # Execute snscrape
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                logger.warning(f"snscrape failed for {account}: {stderr.decode()}")
                return 0
            
            # Process results
            posts_processed = 0
            for line in stdout.decode().strip().split('\n'):
                if not line:
                    continue
                
                try:
                    post_data = json.loads(line)
                    
                    # Create query-like object for processing
                    dummy_query = XSearchQuery(
                        query="",
                        name=f"account_{account}",
                        tags=["influential", "account"]
                    )
                    
                    post_message = await self._process_post(post_data, dummy_query)
                    
                    if post_message and self._is_crypto_related(post_message.text):
                        post_hash = self._create_post_hash(post_data)
                        if post_hash not in self._seen_posts:
                            await self.redis_pipeline.publish_x_post(post_message)
                            self._seen_posts.add(post_hash)
                            posts_processed += 1
                
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"Error processing account post: {e}")
            
            return posts_processed
            
        except Exception as e:
            logger.error(f"Error collecting from account {account}: {e}")
            return 0
    
    async def _process_post(self, post_data: Dict, query: XSearchQuery) -> Optional[XPostMessage]:
        """Process raw post data into XPostMessage"""
        try:
            # Extract basic info
            post_id = post_data.get('id', '')
            text = post_data.get('content', '').strip()
            user = post_data.get('user', {})
            author = user.get('username', 'unknown')
            
            if not post_id or not text:
                return None
            
            # Get post date
            date_str = post_data.get('date', '')
            if date_str:
                # Parse ISO format
                timestamp = datetime.fromisoformat(date_str.replace('Z', '+00:00')).replace(tzinfo=None)
            else:
                timestamp = datetime.now()
            
            # Calculate engagement score
            engagement_score = self._calculate_engagement_score(post_data)
            
            # Calculate sentiment score
            sentiment_score = self._calculate_sentiment_score(text)
            
            # Create message
            message = XPostMessage(
                timestamp=timestamp,
                source="x_twitter",
                post_id=post_id,
                text=text,
                author=author,
                engagement_score=engagement_score,
                sentiment_score=sentiment_score,
                tags=query.tags.copy()
            )
            
            return message
            
        except Exception as e:
            logger.error(f"Error processing post: {e}")
            return None
    
    def _calculate_engagement_score(self, post_data: Dict) -> float:
        """Calculate engagement score for post"""
        try:
            retweets = post_data.get('retweetCount', 0)
            likes = post_data.get('likeCount', 0)
            replies = post_data.get('replyCount', 0)
            quotes = post_data.get('quoteCount', 0)
            
            # Weighted engagement score
            score = (
                retweets * 3.0 +  # Retweets are most valuable
                likes * 1.0 +
                replies * 2.0 +  # Replies indicate engagement
                quotes * 2.5
            )
            
            # Normalize to 0-1 range (logarithmic scale)
            import math
            normalized = math.log(score + 1) / math.log(10000)  # Max expected ~10k
            
            return min(normalized, 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_sentiment_score(self, text: str) -> float:
        """Calculate sentiment score (-1 to +1)"""
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.sentiment_keywords["positive"] if word in text_lower)
        negative_count = sum(1 for word in self.sentiment_keywords["negative"] if word in text_lower)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return 0.0  # Neutral
        
        # Calculate score
        score = (positive_count - negative_count) / total_sentiment_words
        return max(-1.0, min(1.0, score))
    
    def _is_relevant_post(self, post: XPostMessage) -> bool:
        """Check if post is relevant for trading analysis"""
        text = post.text.lower()
        
        # Must have minimum engagement or be from influential source
        if post.engagement_score < 0.1 and "influential" not in post.tags:
            return False
        
        # Check for crypto/trading relevance
        return self._is_crypto_related(text)
    
    def _is_crypto_related(self, text: str) -> bool:
        """Check if text is crypto/trading related"""
        crypto_terms = [
            "bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency",
            "altcoin", "defi", "nft", "blockchain", "trading", "forex", "fx",
            "pump", "dump", "moon", "bullish", "bearish", "hodl", "whale",
            "support", "resistance", "breakout", "ath", "dip", "rally"
        ]
        
        text_lower = text.lower()
        return any(term in text_lower for term in crypto_terms)
    
    def _create_post_hash(self, post_data: Dict) -> str:
        """Create unique hash for post deduplication"""
        post_id = post_data.get('id', '')
        content = post_data.get('content', '')
        hash_content = post_id + content
        return hashlib.md5(hash_content.encode()).hexdigest()
    
    def cleanup_seen_posts(self, max_age_hours: int = 24):
        """Clean up old post hashes"""
        if len(self._seen_posts) > 50000:
            self._seen_posts.clear()
            logger.info("Cleared seen posts cache")
    
    def get_statistics(self) -> Dict[str, any]:
        """Get collector statistics"""
        return {
            "running": self._running,
            "total_queries": len(self.search_queries),
            "influential_accounts": len(self.influential_accounts),
            "seen_posts": len(self._seen_posts),
            "query_names": [q.name for q in self.search_queries]
        }