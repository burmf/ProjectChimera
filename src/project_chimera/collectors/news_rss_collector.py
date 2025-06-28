"""
RSS News Feed Collector for cryptocurrency and financial news
Collects and processes news articles every hour for AI analysis
"""

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime

import aiohttp
import feedparser
from bs4 import BeautifulSoup

from ..streams.message_types import NewsMessage
from ..streams.redis_pipeline import RedisStreamPipeline

logger = logging.getLogger(__name__)


@dataclass
class NewsSource:
    """Configuration for a news source"""

    name: str
    url: str
    enabled: bool = True
    priority: int = 1  # 1=high, 2=medium, 3=low
    tags: list[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class NewsRSSCollector:
    """
    RSS news collector for cryptocurrency and trading news
    Collects from multiple sources and publishes to Redis Streams
    """

    def __init__(self, redis_pipeline: RedisStreamPipeline):
        self.redis_pipeline = redis_pipeline
        self._running = False
        self._task: asyncio.Task | None = None
        self._seen_articles: set[str] = set()
        self.session: aiohttp.ClientSession | None = None

        # Financial news RSS sources
        self.news_sources = [
            # Cryptocurrency News
            NewsSource(
                name="CoinDesk",
                url="https://www.coindesk.com/arc/outboundfeeds/rss/",
                priority=1,
                tags=["crypto", "bitcoin", "ethereum", "defi"],
            ),
            NewsSource(
                name="CoinTelegraph",
                url="https://cointelegraph.com/rss",
                priority=1,
                tags=["crypto", "blockchain", "altcoins"],
            ),
            NewsSource(
                name="CryptoNews",
                url="https://cryptonews.com/news/feed/",
                priority=2,
                tags=["crypto", "trading"],
            ),
            NewsSource(
                name="Decrypt",
                url="https://decrypt.co/feed",
                priority=2,
                tags=["crypto", "web3", "nft"],
            ),
            # Financial News
            NewsSource(
                name="Reuters Business",
                url="https://feeds.reuters.com/reuters/businessNews",
                priority=1,
                tags=["finance", "markets", "economy"],
            ),
            NewsSource(
                name="MarketWatch",
                url="https://feeds.marketwatch.com/marketwatch/latest",
                priority=2,
                tags=["stocks", "markets", "finance"],
            ),
            NewsSource(
                name="Financial Times",
                url="https://www.ft.com/markets?format=rss",
                priority=1,
                tags=["finance", "markets", "trading"],
            ),
            # Forex/Trading News
            NewsSource(
                name="ForexLive",
                url="https://www.forexlive.com/feed/",
                priority=1,
                tags=["forex", "central-banks", "trading"],
            ),
            NewsSource(
                name="FXStreet",
                url="https://www.fxstreet.com/rss/news",
                priority=2,
                tags=["forex", "analysis", "signals"],
            ),
        ]

        # Keywords for relevance filtering
        self.crypto_keywords = {
            "bitcoin",
            "btc",
            "ethereum",
            "eth",
            "cryptocurrency",
            "crypto",
            "blockchain",
            "defi",
            "nft",
            "altcoin",
            "trading",
            "exchange",
            "binance",
            "coinbase",
            "ftx",
            "kraken",
            "whale",
            "dex",
            "yield",
            "staking",
            "mining",
            "halving",
            "fork",
            "usdt",
            "usdc",
            "stablecoin",
        }

        self.forex_keywords = {
            "forex",
            "fx",
            "usd",
            "eur",
            "jpy",
            "gbp",
            "aud",
            "cad",
            "chf",
            "central bank",
            "fed",
            "boj",
            "ecb",
            "interest rate",
            "monetary policy",
            "inflation",
            "gdp",
            "employment",
            "nonfarm",
            "cpi",
            "ppi",
            "retail sales",
        }

        # All relevant keywords
        self.relevant_keywords = self.crypto_keywords | self.forex_keywords

    async def start(self, collection_interval_hours: int = 1):
        """Start the RSS collector"""
        if self._running:
            logger.warning("RSS collector already running")
            return

        logger.info(
            f"Starting RSS news collector (interval: {collection_interval_hours}h)"
        )

        # Create HTTP session
        timeout = aiohttp.ClientTimeout(total=30)
        self.session = aiohttp.ClientSession(timeout=timeout)

        self._running = True
        self._task = asyncio.create_task(
            self._collection_loop(collection_interval_hours)
        )

    async def stop(self):
        """Stop the RSS collector"""
        logger.info("Stopping RSS news collector")

        self._running = False

        from contextlib import suppress

        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task

        if self.session:
            await self.session.close()

    async def _collection_loop(self, interval_hours: int):
        """Main collection loop"""
        while self._running:
            try:
                start_time = datetime.now()

                # Collect from all sources
                total_articles = 0
                for source in self.news_sources:
                    if not source.enabled:
                        continue

                    try:
                        articles = await self._collect_from_source(source)
                        total_articles += articles

                        # Small delay between sources to avoid rate limiting
                        await asyncio.sleep(1)

                    except Exception as e:
                        logger.error(f"Error collecting from {source.name}: {e}")

                collection_time = (datetime.now() - start_time).total_seconds()
                logger.info(
                    f"Collected {total_articles} articles in {collection_time:.1f}s"
                )

                # Wait for next collection
                await asyncio.sleep(interval_hours * 3600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in collection loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute on error

    async def _collect_from_source(self, source: NewsSource) -> int:
        """Collect articles from a single RSS source"""
        try:
            logger.debug(f"Collecting from {source.name}")

            # Fetch RSS feed
            async with self.session.get(source.url) as response:
                if response.status != 200:
                    logger.warning(f"HTTP {response.status} for {source.name}")
                    return 0

                content = await response.text()

            # Parse RSS feed
            feed = feedparser.parse(content)

            if not feed.entries:
                logger.warning(f"No entries in feed for {source.name}")
                return 0

            articles_processed = 0

            for entry in feed.entries:
                try:
                    # Create article hash for deduplication
                    article_hash = self._create_article_hash(entry)

                    if article_hash in self._seen_articles:
                        continue

                    # Process article
                    article = await self._process_article(entry, source)

                    if article and self._is_relevant_article(article):
                        # Publish to Redis
                        await self.redis_pipeline.publish_news(article)

                        # Track as seen
                        self._seen_articles.add(article_hash)
                        articles_processed += 1

                        logger.debug(f"Published article: {article.title[:60]}...")

                except Exception as e:
                    logger.error(f"Error processing article from {source.name}: {e}")

            return articles_processed

        except Exception as e:
            logger.error(f"Error collecting from {source.name}: {e}")
            return 0

    async def _process_article(self, entry, source: NewsSource) -> NewsMessage | None:
        """Process RSS entry into NewsMessage"""
        try:
            # Extract basic info
            title = entry.get("title", "").strip()
            link = entry.get("link", "").strip()

            if not title or not link:
                return None

            # Get publication date
            published = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                import time

                published = datetime.fromtimestamp(time.mktime(entry.published_parsed))
            else:
                published = datetime.now()

            # Extract content
            content = ""
            if hasattr(entry, "summary"):
                content = entry.summary
            elif hasattr(entry, "description"):
                content = entry.description
            elif hasattr(entry, "content"):
                if isinstance(entry.content, list) and entry.content:
                    content = entry.content[0].get("value", "")
                else:
                    content = str(entry.content)

            # Clean content
            content = self._clean_html(content)

            # Calculate relevance score
            relevance_score = self._calculate_relevance(title, content)

            # Create message
            message = NewsMessage(
                timestamp=published,
                message_type="news",
                source=source.name,
                title=title,
                content=content,
                url=link,
                tags=source.tags.copy(),
                relevance_score=relevance_score,
            )

            return message

        except Exception as e:
            logger.error(f"Error processing article: {e}")
            return None

    def _clean_html(self, text: str) -> str:
        """Clean HTML from text content"""
        if not text:
            return ""

        # Parse with BeautifulSoup
        soup = BeautifulSoup(text, "html.parser")

        # Extract text
        clean_text = soup.get_text()

        # Clean whitespace
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        return clean_text

    def _calculate_relevance(self, title: str, content: str) -> float:
        """Calculate relevance score for trading/crypto content"""
        text = (title + " " + content).lower()

        # Count keyword matches
        keyword_matches = 0
        total_keywords = len(self.relevant_keywords)

        for keyword in self.relevant_keywords:
            if keyword in text:
                keyword_matches += 1

        # Calculate base relevance
        relevance = keyword_matches / total_keywords if total_keywords > 0 else 0

        # Boost for crypto keywords
        crypto_matches = sum(1 for kw in self.crypto_keywords if kw in text)
        if crypto_matches > 0:
            relevance += 0.3 * min(crypto_matches / len(self.crypto_keywords), 1.0)

        # Boost for trading terms
        trading_terms = [
            "trading",
            "price",
            "volume",
            "breakout",
            "support",
            "resistance",
            "bull",
            "bear",
        ]
        trading_matches = sum(1 for term in trading_terms if term in text)
        if trading_matches > 0:
            relevance += 0.2 * min(trading_matches / len(trading_terms), 1.0)

        return min(relevance, 1.0)

    def _is_relevant_article(self, article: NewsMessage) -> bool:
        """Check if article is relevant for trading"""
        # Minimum relevance threshold
        if article.relevance_score < 0.1:
            return False

        # Check if title/content contains relevant keywords
        text = (article.title + " " + article.content).lower()

        # Must contain at least one relevant keyword
        return any(keyword in text for keyword in self.relevant_keywords)

    def _create_article_hash(self, entry) -> str:
        """Create unique hash for article deduplication"""
        title = entry.get("title", "")
        link = entry.get("link", "")
        content = title + link
        return hashlib.md5(content.encode()).hexdigest()

    def cleanup_seen_articles(self, max_age_hours: int = 24):
        """Clean up old article hashes to prevent memory growth"""
        # Note: This is a simple implementation
        # In production, you might want to use Redis or a database for persistence
        if len(self._seen_articles) > 10000:
            # Keep only recent hashes (naive approach)
            # Could be improved by storing timestamps
            self._seen_articles.clear()
            logger.info("Cleared seen articles cache")

    def get_statistics(self) -> dict[str, any]:
        """Get collector statistics"""
        enabled_sources = [s for s in self.news_sources if s.enabled]

        return {
            "running": self._running,
            "total_sources": len(self.news_sources),
            "enabled_sources": len(enabled_sources),
            "seen_articles": len(self._seen_articles),
            "source_names": [s.name for s in enabled_sources],
        }
