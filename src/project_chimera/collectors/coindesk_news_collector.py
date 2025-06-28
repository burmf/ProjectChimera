"""
CoinDesk News Collector
Collects real-time news articles from the CoinDesk Data API.
"""

import asyncio
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import httpx
import structlog

logger = structlog.get_logger(__name__)


class CoinDeskNewsCollector:
    """
    Collects news articles from the CoinDesk Data API.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://data-api.coindesk.com",
        timeout_seconds: float = 10.0,
        min_request_interval: float = 1.0,  # To respect API rate limits
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            base_url=self.base_url, timeout=timeout_seconds
        )
        self.last_request_time = 0.0
        self.min_request_interval = min_request_interval

    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with API key if available."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def _make_request(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make HTTP GET request with rate limiting and error handling."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)

        self.last_request_time = time.time()

        try:
            response = await self.client.get(
                endpoint, params=params, headers=self._get_headers()
            )
            response.raise_for_status()  # Raise an exception for 4xx/5xx responses
            data = response.json()
            return data
        except httpx.HTTPStatusError as e:
            logger.error(
                "coindesk_api_http_error",
                endpoint=endpoint,
                status_code=e.response.status_code,
                response_text=e.response.text,
                error=str(e),
            )
            return {}
        except httpx.RequestError as e:
            logger.error(
                "coindesk_api_request_error", endpoint=endpoint, error=str(e)
            )
            return {}
        except Exception as e:
            logger.error(
                "coindesk_api_unknown_error", endpoint=endpoint, error=str(e)
            )
            return {}

    async def get_latest_articles(
        self, limit: int = 10, language: str = "en"
    ) -> list[dict[str, Any]]:
        """
        Fetches the latest news articles from CoinDesk.
        Args:
            limit: The maximum number of articles to retrieve (default: 10).
            language: The language of the articles (default: 'en').
        Returns:
            A list of dictionaries, each representing a news article.
        """
        endpoint = "/news/v1/articles"
        params = {"limit": limit, "language": language}

        data = await self._make_request(endpoint, params)

        articles = []
        if data and isinstance(data.get("data"), list):
            for article_data in data["data"]:
                articles.append(
                    {
                        "id": article_data.get("id"),
                        "title": article_data.get("title"),
                        "url": article_data.get("url"),
                        "published_at": datetime.fromisoformat(
                            article_data["published_at"].replace("Z", "+00:00")
                        )
                        if "published_at" in article_data
                        else None,
                        "sentiment": article_data.get("sentiment"),  # positive, negative, neutral
                        "source": article_data.get("source", {}).get("name"),
                        "categories": [
                            cat.get("name")
                            for cat in article_data.get("categories", [])
                        ],
                        "tags": article_data.get("tags", []),
                    }
                )
        logger.info("coindesk_articles_fetched", count=len(articles))
        return articles

    async def get_news_sources(self) -> List[Dict[str, Any]]:
        """
        Fetches the list of available news sources from CoinDesk.
        """
        endpoint = "/news/v1/sources"
        data = await self._make_request(endpoint)
        if data and isinstance(data.get("data"), list):
            logger.info("coindesk_sources_fetched", count=len(data["data"]))
            return data["data"]
        return []

    async def get_news_categories(self) -> list[dict[str, Any]]:
        """
        Fetches the list of available news categories from CoinDesk.
        """
        endpoint = "/news/v1/categories"
        data = await self._make_request(endpoint)
        if data and isinstance(data.get("data"), list):
            logger.info("coindesk_categories_fetched", count=len(data["data"]))
            return data["data"]
        return []


# Global instance for easy access
_coindesk_collector: CoinDeskNewsCollector | None = None


def get_coindesk_collector() -> CoinDeskNewsCollector:
    """Get global CoinDesk news collector instance"""
    global _coindesk_collector
    if _coindesk_collector is None:
        api_key = os.getenv("COINDESK_API_KEY", "")
        _coindesk_collector = CoinDeskNewsCollector(api_key=api_key)
    return _coindesk_collector


async def demo_coindesk_data():
    """Demo function to test CoinDesk news fetching"""
    global _coindesk_collector
    _coindesk_collector = None  # Reset for demo

    collector = get_coindesk_collector()

    try:
        print("🔍 Testing CoinDesk News API connection...")
        print(f"🔑 API Key configured: {bool(collector.api_key)}")

        # Test fetching latest articles
        articles = await collector.get_latest_articles(limit=3)
        if articles:
            print(f"✅ Fetched {len(articles)} latest articles:")
            for article in articles:
                print(
                    f"  - [{article.get('published_at').strftime('%Y-%m-%d %H:%M')}] "
                    f"[{article.get('sentiment', 'N/A')}] {article.get('title')} "
                    f"({article.get('source', 'N/A')})"
                )
        else:
            print("❌ No articles fetched or error occurred.")

        # Test fetching sources
        sources = await collector.get_news_sources()
        print(f"✅ Fetched {len(sources)} news sources.")

        return True

    except Exception as e:
        print(f"❌ Error testing CoinDesk API: {e}")
        return False
    finally:
        await collector.close()


if __name__ == "__main__":
    # Load environment variables for testing
    from dotenv import load_dotenv

    load_dotenv()

    asyncio.run(demo_coindesk_data())
