#!/usr/bin/env python3
"""
Bitget REST API Client for Historical Data Collection
スキャルピング用高頻度データ取得システム
"""

import requests
import hmac
import hashlib
import base64
import time
import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import os
from dotenv import load_dotenv

load_dotenv()

class BitgetRestClient:
    def __init__(self):
        self.api_key = os.getenv('BITGET_API_KEY')
        self.secret_key = os.getenv('BITGET_SECRET_KEY')
        self.passphrase = os.getenv('BITGET_PASSPHRASE')
        self.base_url = 'https://api.bitget.com'
        
        self.session = requests.Session()
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting (Bitget allows 20 requests per second for public endpoints)
        self.request_delay = 0.05  # 50ms between requests
        self.last_request_time = 0
        
    def _generate_signature(self, timestamp: str, method: str, request_path: str, body: str = '') -> str:
        """Bitget API署名を生成"""
        message = timestamp + method + request_path + body
        signature = base64.b64encode(
            hmac.new(
                self.secret_key.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).digest()
        ).decode('utf-8')
        return signature
    
    def _get_headers(self, method: str, request_path: str, body: str = '') -> Dict[str, str]:
        """API リクエストヘッダーを生成"""
        timestamp = str(int(time.time() * 1000))
        signature = self._generate_signature(timestamp, method, request_path, body)
        
        return {
            'ACCESS-KEY': self.api_key,
            'ACCESS-SIGN': signature,
            'ACCESS-TIMESTAMP': timestamp,
            'ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type': 'application/json'
        }
    
    def _rate_limit(self):
        """レート制限を適用"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            time.sleep(self.request_delay - time_since_last)
        self.last_request_time = time.time()
    
    def get_kline_data(self, symbol: str, granularity: str, 
                       start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """
        ローソク足データを取得
        
        Args:
            symbol: 通貨ペア (例: BTCUSDT)
            granularity: 時間間隔 (1m, 5m, 15m, 30m, 1H, 4H, 1D)
            start_time: 開始時間
            end_time: 終了時間
        
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume, quote_volume
        """
        self._rate_limit()
        
        # Convert datetime to milliseconds
        start_ms = int(start_time.timestamp() * 1000)
        end_ms = int(end_time.timestamp() * 1000)
        
        endpoint = '/api/spot/v1/market/candles'
        params = {
            'symbol': symbol,
            'granularity': granularity,
            'startTime': start_ms,
            'endTime': end_ms,
            'limit': 1000  # Bitget max limit
        }
        
        try:
            response = self.session.get(
                self.base_url + endpoint,
                params=params,
                timeout=30
            )
            
            if response.status_code != 200:
                self.logger.error(f"API Error {response.status_code}: {response.text}")
                return pd.DataFrame()
            
            data = response.json()
            
            if data.get('code') != '00000':
                self.logger.error(f"Bitget API Error: {data}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            candles = data.get('data', [])
            if not candles:
                return pd.DataFrame()
            
            df = pd.DataFrame(candles, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume', 'quote_volume'
            ])
            
            # Convert data types
            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(int), unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
                df[col] = pd.to_numeric(df[col])
            
            df.sort_values('timestamp', inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            self.logger.info(f"Retrieved {len(df)} candles for {symbol} ({granularity})")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching kline data: {e}")
            return pd.DataFrame()
    
    def get_historical_data_batch(self, symbol: str, granularity: str, 
                                 days_back: int = 30) -> pd.DataFrame:
        """
        過去N日分のデータをバッチ取得
        Bitgetの1000件制限を考慮して複数回リクエスト
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        # Calculate interval duration in minutes
        interval_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1H': 60, '4H': 240, '1D': 1440
        }.get(granularity, 1)
        
        # Calculate how many requests needed (1000 records per request max)
        total_minutes = int((end_time - start_time).total_seconds() / 60)
        records_needed = total_minutes // interval_minutes
        batches_needed = (records_needed // 1000) + 1
        
        self.logger.info(f"Fetching {records_needed} records in {batches_needed} batches for {symbol}")
        
        all_data = []
        current_end = end_time
        
        for batch in range(batches_needed):
            batch_start = current_end - timedelta(minutes=1000 * interval_minutes)
            if batch_start < start_time:
                batch_start = start_time
            
            batch_data = self.get_kline_data(symbol, granularity, batch_start, current_end)
            
            if not batch_data.empty:
                all_data.append(batch_data)
            
            current_end = batch_start
            
            if current_end <= start_time:
                break
            
            # Progress logging
            self.logger.info(f"Batch {batch + 1}/{batches_needed} completed")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df.drop_duplicates(subset=['timestamp'], inplace=True)
            combined_df.sort_values('timestamp', inplace=True)
            combined_df.reset_index(drop=True, inplace=True)
            
            self.logger.info(f"Total records collected: {len(combined_df)}")
            return combined_df
        
        return pd.DataFrame()
    
    def get_orderbook_snapshot(self, symbol: str, limit: int = 50) -> Dict:
        """
        オーダーブックスナップショットを取得
        """
        self._rate_limit()
        
        endpoint = '/api/spot/v1/market/depth'
        params = {
            'symbol': symbol,
            'limit': limit,
            'type': 'step0'  # 最高精度
        }
        
        try:
            response = self.session.get(
                self.base_url + endpoint,
                params=params,
                timeout=30
            )
            
            if response.status_code != 200:
                self.logger.error(f"Orderbook API Error {response.status_code}: {response.text}")
                return {}
            
            data = response.json()
            
            if data.get('code') != '00000':
                self.logger.error(f"Bitget Orderbook API Error: {data}")
                return {}
            
            return data.get('data', {})
            
        except Exception as e:
            self.logger.error(f"Error fetching orderbook: {e}")
            return {}
    
    def get_ticker_24hr(self, symbol: str) -> Dict:
        """24時間統計情報を取得"""
        self._rate_limit()
        
        endpoint = '/api/spot/v1/market/ticker'
        params = {'symbol': symbol}
        
        try:
            response = self.session.get(
                self.base_url + endpoint,
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    return data.get('data', {})
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Error fetching ticker: {e}")
            return {}


class ScalpingDataCollector:
    """スキャルピング特化データ収集システム"""
    
    def __init__(self, database_adapter=None):
        self.client = BitgetRestClient()
        self.db = database_adapter
        self.logger = logging.getLogger(__name__)
        
        # スキャルピング対象の主要ペア
        self.scalping_pairs = [
            'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'XRPUSDT', 
            'SOLUSDT', 'DOGEUSDT', 'MATICUSDT', 'AVAXUSDT'
        ]
        
        # スキャルピング用時間間隔
        self.timeframes = ['1m', '5m', '15m']
        
    def collect_scalping_dataset(self, days_back: int = 30) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        スキャルピング用包括的データセットを収集
        
        Returns:
            {
                'BTCUSDT': {
                    '1m': DataFrame,
                    '5m': DataFrame,
                    '15m': DataFrame
                },
                ...
            }
        """
        self.logger.info(f"Starting scalping dataset collection for {days_back} days")
        
        dataset = {}
        
        for pair in self.scalping_pairs:
            self.logger.info(f"Collecting data for {pair}")
            pair_data = {}
            
            for timeframe in self.timeframes:
                self.logger.info(f"  - Fetching {timeframe} data...")
                
                df = self.client.get_historical_data_batch(
                    symbol=pair,
                    granularity=timeframe,
                    days_back=days_back
                )
                
                if not df.empty:
                    # Add technical indicators for scalping
                    df = self._add_scalping_indicators(df)
                    pair_data[timeframe] = df
                    
                    # Save to database if available
                    if self.db:
                        self._save_to_database(pair, timeframe, df)
                
                # Rate limiting
                time.sleep(0.1)
            
            dataset[pair] = pair_data
            self.logger.info(f"Completed {pair}: {sum(len(data) for data in pair_data.values())} total records")
        
        return dataset
    
    def _add_scalping_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """スキャルピング用テクニカル指標を追加"""
        try:
            # Price-based indicators
            df['price_change'] = df['close'].pct_change()
            df['price_change_abs'] = df['price_change'].abs()
            
            # Volume indicators
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Volatility indicators
            df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
            df['body_size'] = abs(df['close'] - df['open']) / df['close']
            df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
            df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
            
            # Short-term moving averages for scalping
            for window in [5, 10, 20]:
                df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
                df[f'price_vs_sma_{window}'] = (df['close'] - df[f'sma_{window}']) / df[f'sma_{window}']
            
            # Momentum indicators
            df['rsi_14'] = self._calculate_rsi(df['close'], 14)
            df['rsi_7'] = self._calculate_rsi(df['close'], 7)  # Faster for scalping
            
            # Bollinger Bands (short period for scalping)
            bb_window = 10
            bb_std = 1.5
            df['bb_middle'] = df['close'].rolling(window=bb_window).mean()
            bb_std_calc = df['close'].rolling(window=bb_window).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std_calc * bb_std)
            df['bb_lower'] = df['bb_middle'] - (bb_std_calc * bb_std)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # MACD (fast settings for scalping)
            df = self._add_macd(df, fast=8, slow=17, signal=9)
            
            self.logger.debug(f"Added scalping indicators to {len(df)} records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error adding scalping indicators: {e}")
            return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _add_macd(self, df: pd.DataFrame, fast: int = 8, slow: int = 17, signal: int = 9) -> pd.DataFrame:
        """MACD計算（スキャルピング用高速設定）"""
        exp_fast = df['close'].ewm(span=fast).mean()
        exp_slow = df['close'].ewm(span=slow).mean()
        df['macd'] = exp_fast - exp_slow
        df['macd_signal'] = df['macd'].ewm(span=signal).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        return df
    
    def _save_to_database(self, pair: str, timeframe: str, df: pd.DataFrame):
        """データベースに保存"""
        if not self.db:
            return
        
        try:
            # Create table if not exists
            table_name = f"scalping_data_{pair.lower()}_{timeframe}"
            
            # Save DataFrame to database
            # This would need implementation based on your database adapter
            self.logger.info(f"Saved {len(df)} records to {table_name}")
            
        except Exception as e:
            self.logger.error(f"Error saving to database: {e}")
    
    def analyze_scalping_patterns(self, dataset: Dict) -> Dict:
        """スキャルピングパターン分析"""
        analysis = {}
        
        for pair, timeframes in dataset.items():
            pair_analysis = {}
            
            for timeframe, df in timeframes.items():
                if df.empty:
                    continue
                
                # Volatility analysis
                avg_volatility = df['high_low_ratio'].mean()
                volatility_std = df['high_low_ratio'].std()
                
                # Volume patterns
                avg_volume_ratio = df['volume_ratio'].mean()
                high_volume_threshold = df['volume_ratio'].quantile(0.8)
                
                # Price movement analysis
                significant_moves = df[df['price_change_abs'] > df['price_change_abs'].quantile(0.9)]
                
                pair_analysis[timeframe] = {
                    'avg_volatility': avg_volatility,
                    'volatility_std': volatility_std,
                    'avg_volume_ratio': avg_volume_ratio,
                    'high_volume_threshold': high_volume_threshold,
                    'significant_moves_count': len(significant_moves),
                    'best_scalping_hours': self._find_best_hours(df)
                }
            
            analysis[pair] = pair_analysis
        
        return analysis
    
    def _find_best_hours(self, df: pd.DataFrame) -> List[int]:
        """スキャルピングに最適な時間帯を特定"""
        if 'timestamp' not in df.columns:
            return []
        
        df['hour'] = df['timestamp'].dt.hour
        hourly_volatility = df.groupby('hour')['high_low_ratio'].mean()
        hourly_volume = df.groupby('hour')['volume_ratio'].mean()
        
        # Combine volatility and volume scores
        combined_score = (hourly_volatility.rank() + hourly_volume.rank()) / 2
        best_hours = combined_score.nlargest(6).index.tolist()
        
        return sorted(best_hours)