#!/usr/bin/env python3
"""
Data Collector for Trading Bot
Supports both price and news data collection with AI processing
"""

import sys
import argparse
import logging
import asyncio
import time
import os
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import yfinance as yf
import requests
import pandas as pd
from core.logging_config import setup_logging
from core.database_adapter import db_adapter

class PriceCollector:
    """Collects price data from Yahoo Finance."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.symbols = {
            'USD/JPY': 'USDJPY=X',
            'EUR/USD': 'EURUSD=X',
            'GBP/USD': 'GBPUSD=X'
        }

    def collect_price_data(self, symbol: str = 'USD/JPY', hours_back: int = 2) -> bool:
        """Collect price data for specified symbol."""
        try:
            ticker_symbol = self.symbols.get(symbol, 'USDJPY=X')
            ticker = yf.Ticker(ticker_symbol)
            
            # Get data for last few hours
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)
            
            hist = ticker.history(start=start_time, end=end_time, interval='1h')
            
            if hist.empty:
                self.logger.warning(f"No price data retrieved for {symbol}")
                return False
            
            # Convert to database format
            price_records = []
            for timestamp, row in hist.iterrows():
                price_records.append({
                    'timestamp': timestamp.to_pydatetime(),
                    'symbol': symbol,
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': float(row['Volume']) if 'Volume' in row else 0,
                    'interval_minutes': 60
                })
            
            # Insert into database (convert list to individual inserts)
            success_count = 0
            for record in price_records:
                # Fix field name mismatch: 'symbol' -> 'pair'
                record_fixed = record.copy()
                record_fixed['pair'] = record_fixed.pop('symbol')
                
                if db_adapter.insert_price_data(record_fixed):
                    success_count += 1
            
            if success_count > 0:
                self.logger.info(f"Inserted {success_count} price records for {symbol}")
            
            return success_count == len(price_records)
            
        except Exception as e:
            self.logger.error(f"Price collection failed for {symbol}: {e}")
            return False
    
    def run_collection(self):
        """Run price collection for all symbols."""
        for symbol in self.symbols.keys():
            self.collect_price_data(symbol)


class NewsCollector:
    """Collects news data from NewsAPI and processes with AI."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_key = os.getenv('NEWSAPI_KEY')
        
    def collect_news_data(self, hours_back: int = 6) -> bool:
        """Collect news data and process with AI."""
        if not self.api_key:
            self.logger.error("NewsAPI key not found")
            return False
            
        try:
            # Calculate time range
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours_back)
            
            # NewsAPI parameters
            params = {
                'apiKey': self.api_key,
                'q': 'forex OR currency OR "central bank" OR inflation OR GDP OR unemployment',
                'language': 'en',
                'sortBy': 'publishedAt',
                'from': start_time.strftime('%Y-%m-%dT%H:%M:%S'),
                'to': end_time.strftime('%Y-%m-%dT%H:%M:%S'),
                'pageSize': 50
            }
            
            response = requests.get('https://newsapi.org/v2/everything', params=params)
            response.raise_for_status()
            
            news_data = response.json()
            articles = news_data.get('articles', [])
            
            if not articles:
                self.logger.info("No new articles found")
                return True
            
            # Process articles
            processed_count = 0
            for article in articles:
                if self.process_article(article):
                    processed_count += 1
            
            self.logger.info(f"Processed {processed_count} news articles")
            return True
            
        except Exception as e:
            self.logger.error(f"News collection failed: {e}")
            return False
    
    def process_article(self, article: Dict) -> bool:
        """Process a single news article."""
        try:
            # Clean and prepare article data
            # Generate unique ID from URL
            article_id = hashlib.md5(article['url'].encode()).hexdigest()
            
            article_data = {
                'id': article_id,
                'url': article['url'],
                'title': article['title'],
                'content': article.get('description', ''),
                'published_at': datetime.fromisoformat(
                    article['publishedAt'].replace('Z', '+00:00')
                ),
                'source': article['source']['name']
            }
            
            # Insert article into database
            return db_adapter.insert_news_article(article_data)
            
        except Exception as e:
            self.logger.error(f"Article processing failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description='Trading Bot Data Collector')
    parser.add_argument('--type', choices=['price', 'news', 'all'], default='all',
                       help='Type of data to collect')
    parser.add_argument('--interval', type=int, default=300,
                       help='Collection interval in seconds')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting data collector for: {args.type}")
    
    if not db_adapter.is_connected():
        logger.error("Database connection failed. Exiting.")
        sys.exit(1)
    
    # Initialize collectors
    price_collector = PriceCollector()
    news_collector = NewsCollector()
    
    # Run collection loop
    try:
        while True:
            start_time = time.time()
            
            if args.type in ['price', 'all']:
                logger.info("Running price collection...")
                price_collector.run_collection()
            
            if args.type in ['news', 'all']:
                logger.info("Running news collection...")
                news_collector.collect_news_data()
            
            # Sleep until next interval
            elapsed = time.time() - start_time
            sleep_time = max(0, args.interval - elapsed)
            
            if sleep_time > 0:
                logger.info(f"Sleeping for {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        logger.info("Data collector stopped by user")
    except Exception as e:
        logger.error(f"Data collector failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()