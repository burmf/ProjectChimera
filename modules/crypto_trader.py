# modules/crypto_trader.py
import ccxt
import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, List
import datetime
import sys
import os

from core.redis_manager import redis_manager
from core.price_stream import price_stream

logger = logging.getLogger(__name__)

class CryptoTrader:
    def __init__(self, exchange_name: str = 'bitget', demo_mode: bool = True):
        self.exchange_name = exchange_name
        self.demo_mode = demo_mode
        self.exchange = None
        self.supported_symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        self.initialize_exchange()
    
    def initialize_exchange(self):
        """Initialize crypto exchange connection"""
        try:
            if self.exchange_name == 'bitget':
                self.exchange = ccxt.bitget({
                    'apiKey': os.getenv('BITGET_API_KEY', ''),
                    'secret': os.getenv('BITGET_SECRET', ''),
                    'password': os.getenv('BITGET_PASSPHRASE', ''),
                    'sandbox': self.demo_mode,  # Use sandbox for demo
                    'enableRateLimit': True,
                })
            else:
                # Add other exchanges as needed
                logger.warning(f"Exchange {self.exchange_name} not implemented")
                return
            
            # Test connection
            if self.exchange.apiKey:
                balance = self.exchange.fetch_balance()
                logger.info(f"Connected to {self.exchange_name} - Balance: {balance.get('total', {})}")
            else:
                logger.info(f"Connected to {self.exchange_name} in read-only mode")
                
        except Exception as e:
            logger.error(f"Failed to initialize {self.exchange_name}: {e}")
            self.exchange = None
    
    def is_connected(self) -> bool:
        """Check if exchange is connected"""
        return self.exchange is not None
    
    def fetch_ticker(self, symbol: str) -> Optional[Dict]:
        """Fetch current ticker data"""
        if not self.is_connected():
            return None
        
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return {
                'symbol': symbol,
                'timestamp': ticker['timestamp'],
                'bid': ticker['bid'],
                'ask': ticker['ask'],
                'last': ticker['last'],
                'volume': ticker['baseVolume'],
                'change_24h': ticker['percentage']
            }
        except Exception as e:
            logger.error(f"Failed to fetch ticker for {symbol}: {e}")
            return None
    
    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data"""
        if not self.is_connected():
            return None
        
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df['pair'] = symbol
            return df
        except Exception as e:
            logger.error(f"Failed to fetch OHLCV for {symbol}: {e}")
            return None
    
    def stream_to_redis(self, symbols: List[str] = None):
        """Stream crypto price data to Redis"""
        if not symbols:
            symbols = self.supported_symbols
        
        for symbol in symbols:
            try:
                # Fetch ticker data
                ticker = self.fetch_ticker(symbol)
                if ticker:
                    # Convert to price stream format
                    price_data = {
                        'timestamp': datetime.datetime.now().isoformat(),
                        'open': ticker['last'],  # Using last price as proxy
                        'high': ticker['last'],
                        'low': ticker['last'],
                        'close': ticker['last'],
                        'volume': ticker['volume'] or 0
                    }
                    
                    # Add to Redis stream
                    price_stream.add_price_data(symbol, price_data)
                    logger.debug(f"Streamed {symbol} ticker to Redis")
                
                # Fetch recent OHLCV
                ohlcv_df = self.fetch_ohlcv(symbol, '1h', limit=10)
                if ohlcv_df is not None and not ohlcv_df.empty:
                    # Add recent data to stream
                    price_stream.add_price_dataframe(symbol, ohlcv_df.tail(5))
                    
            except Exception as e:
                logger.error(f"Failed to stream {symbol}: {e}")
    
    def calculate_position_size(self, symbol: str, account_balance: float, 
                              risk_percent: float = 0.5, atr_multiplier: float = 2.0) -> float:
        """Calculate position size based on volatility (ATR)"""
        try:
            # Fetch recent data for ATR calculation
            ohlcv_df = self.fetch_ohlcv(symbol, '1h', limit=50)
            if ohlcv_df is None or len(ohlcv_df) < 20:
                return 0.001  # Minimum position size
            
            # Simple ATR calculation (True Range average)
            high = ohlcv_df['high']
            low = ohlcv_df['low']
            close = ohlcv_df['close']
            
            prev_close = close.shift(1)
            tr1 = high - low
            tr2 = abs(high - prev_close)
            tr3 = abs(low - prev_close)
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=14).mean().iloc[-1]
            
            # Position sizing based on ATR
            risk_amount = account_balance * (risk_percent / 100)
            price = close.iloc[-1]
            stop_distance = atr * atr_multiplier
            
            position_size = risk_amount / stop_distance
            
            # Convert to appropriate units for crypto
            if 'BTC' in symbol:
                position_size = max(0.001, min(position_size / price, 0.01))  # Min 0.001, max 0.01 BTC
            else:
                position_size = max(0.01, min(position_size / price, 1.0))    # Min 0.01, max 1.0 for others
            
            logger.debug(f"Calculated position size for {symbol}: {position_size:.6f}")
            return position_size
            
        except Exception as e:
            logger.error(f"Position size calculation failed for {symbol}: {e}")
            return 0.001
    
    def place_market_order(self, symbol: str, side: str, amount: float) -> Optional[Dict]:
        """Place a market order"""
        if not self.is_connected() or not self.exchange.apiKey:
            logger.warning("Cannot place order: exchange not connected or no API key")
            return None
        
        try:
            order = self.exchange.create_market_order(symbol, side, amount)
            logger.info(f"Placed {side} order: {amount} {symbol} - Order ID: {order['id']}")
            
            # Store order info in Redis
            order_info = {
                'id': order['id'],
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'timestamp': datetime.datetime.now().isoformat(),
                'status': order.get('status', 'unknown')
            }
            
            redis_manager.add_to_stream('crypto_orders', order_info)
            return order
            
        except Exception as e:
            logger.error(f"Failed to place {side} order for {symbol}: {e}")
            return None
    
    def get_account_balance(self) -> Dict[str, float]:
        """Get account balance"""
        if not self.is_connected() or not self.exchange.apiKey:
            return {}
        
        try:
            balance = self.exchange.fetch_balance()
            
            # Extract relevant balances
            relevant_balance = {}
            for currency in ['USDT', 'BTC', 'ETH', 'SOL']:
                if currency in balance['total']:
                    relevant_balance[currency] = balance['total'][currency]
            
            # Cache balance in Redis
            redis_manager.set_cache('crypto_balance', relevant_balance, ttl=300)
            
            return relevant_balance
            
        except Exception as e:
            logger.error(f"Failed to get account balance: {e}")
            return {}
    
    def get_open_positions(self) -> List[Dict]:
        """Get open positions (for derivatives)"""
        if not self.is_connected() or not self.exchange.apiKey:
            return []
        
        try:
            # For spot trading, check non-zero balances
            balance = self.get_account_balance()
            positions = []
            
            for currency, amount in balance.items():
                if amount > 0 and currency != 'USDT':  # Exclude USDT (base currency)
                    positions.append({
                        'symbol': f"{currency}/USDT",
                        'side': 'long',  # Spot holdings are always long
                        'amount': amount,
                        'value_usdt': amount * self.get_current_price(f"{currency}/USDT")
                    })
            
            return positions
            
        except Exception as e:
            logger.error(f"Failed to get open positions: {e}")
            return []
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol"""
        ticker = self.fetch_ticker(symbol)
        return ticker['last'] if ticker else 0.0
    
    def generate_crypto_signals(self, symbol: str) -> Dict[str, Any]:
        """Generate trading signals for crypto symbols"""
        try:
            # Fetch recent data
            ohlcv_df = self.fetch_ohlcv(symbol, '1h', limit=100)
            if ohlcv_df is None or len(ohlcv_df) < 50:
                return {}
            
            close = ohlcv_df['close']
            
            # Simple signals
            signals = {}
            
            # Moving average signals
            sma_20 = close.rolling(20).mean()
            sma_50 = close.rolling(50).mean()
            
            current_price = close.iloc[-1]
            current_sma_20 = sma_20.iloc[-1]
            current_sma_50 = sma_50.iloc[-1]
            
            if current_price > current_sma_20 > current_sma_50:
                signals['trend'] = 'bullish'
                signals['strength'] = min((current_price - current_sma_50) / current_sma_50 * 100, 10)
            elif current_price < current_sma_20 < current_sma_50:
                signals['trend'] = 'bearish'
                signals['strength'] = min((current_sma_50 - current_price) / current_sma_50 * 100, 10)
            else:
                signals['trend'] = 'neutral'
                signals['strength'] = 0
            
            # Volatility check
            returns = close.pct_change()
            volatility = returns.rolling(20).std() * np.sqrt(24)  # 24h volatility
            signals['volatility'] = volatility.iloc[-1] if not pd.isna(volatility.iloc[-1]) else 0
            
            # Volume analysis
            volume = ohlcv_df['volume']
            avg_volume = volume.rolling(20).mean()
            signals['volume_ratio'] = volume.iloc[-1] / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1
            
            # Store signals in Redis
            signal_key = f"crypto_signals:{symbol}"
            redis_manager.set_cache(signal_key, signals, ttl=300)
            
            return signals
            
        except Exception as e:
            logger.error(f"Failed to generate signals for {symbol}: {e}")
            return {}
    
    def get_market_overview(self) -> Dict[str, Any]:
        """Get overview of crypto markets"""
        overview = {
            'symbols': self.supported_symbols,
            'prices': {},
            'changes': {},
            'volumes': {}
        }
        
        for symbol in self.supported_symbols:
            ticker = self.fetch_ticker(symbol)
            if ticker:
                overview['prices'][symbol] = ticker['last']
                overview['changes'][symbol] = ticker['change_24h']
                overview['volumes'][symbol] = ticker['volume']
        
        return overview

# Global crypto trader instance
crypto_trader = CryptoTrader(demo_mode=True)  # Start in demo mode