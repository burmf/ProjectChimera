# core/realtime_processor.py
import asyncio
import json
import logging
import datetime
from typing import Dict, List, Optional, Any
import sys
import os


from core.redis_manager import redis_manager, PRICE_STREAM, NEWS_STREAM, AI_DECISIONS_STREAM, TRADE_SIGNALS_STREAM
from core.news_stream import news_stream
from core.price_stream import price_stream
from core.openai_manager import openai_manager, Priority
from core.bitget_websocket import BitgetWebSocketClient, BitgetDataProcessor
from modules.feature_builder import feature_builder
from modules.signal_fusion import signal_fusion
from modules.crypto_trader import crypto_trader

logger = logging.getLogger(__name__)

class RealtimeProcessor:
    def __init__(self):
        self.running = False
        self.processors = {}
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.news_api_key = os.getenv('NEWS_API_KEY')
        self.bitget_client = None
        self.bitget_processor = None
        
    async def start_processing(self):
        """Start all real-time processing pipelines"""
        if self.running:
            logger.warning("Realtime processor already running")
            return
        
        self.running = True
        logger.info("Starting realtime processing pipelines...")
        
        # Start concurrent processors
        tasks = [
            asyncio.create_task(self.price_processor()),
            asyncio.create_task(self.news_processor()),
            asyncio.create_task(self.ai_decision_processor()),
            asyncio.create_task(self.trade_signal_processor()),
            asyncio.create_task(self.crypto_stream_processor()),
            asyncio.create_task(self.bitget_websocket_processor())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            logger.error(f"Realtime processor error: {e}")
        finally:
            self.running = False
    
    async def stop_processing(self):
        """Stop all processing"""
        self.running = False
        logger.info("Stopping realtime processing...")
    
    async def price_processor(self):
        """Process incoming price data and trigger technical analysis"""
        logger.info("Price processor started")
        
        while self.running:
            try:
                # Read new price data from stream
                messages = redis_manager.read_stream(PRICE_STREAM, count=10, start_id='>')
                
                for message in messages:
                    await self.process_price_message(message)
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Price processor error: {e}")
                await asyncio.sleep(5)
    
    async def process_price_message(self, message: Dict):
        """Process a single price message"""
        try:
            data = message['data']
            pair = data.get('pair')
            
            if not pair:
                return
            
            # Get recent price data for technical analysis
            recent_prices = price_stream.get_latest_prices(count=100)
            pair_prices = [p for p in recent_prices if p['pair'] == pair]
            
            if len(pair_prices) < 50:  # Need enough data for indicators
                return
            
            # Convert to DataFrame for analysis
            import pandas as pd
            df = pd.DataFrame(pair_prices)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Build technical features
            features = feature_builder.build_features_for_pair(pair, df)
            
            if features:
                # Generate trading signals
                signals = feature_builder.generate_trading_signals(pair, features)
                
                if signals:
                    # Publish technical analysis results
                    analysis_result = {
                        'pair': pair,
                        'timestamp': datetime.datetime.now().isoformat(),
                        'features': {k: float(v.iloc[-1]) if hasattr(v, 'iloc') else v 
                                   for k, v in features.items() if hasattr(v, 'iloc') and len(v) > 0},
                        'signals': signals
                    }
                    
                    redis_manager.publish('technical_analysis_results', analysis_result)
                    logger.debug(f"Technical analysis completed for {pair}")
            
        except Exception as e:
            logger.error(f"Price message processing error: {e}")
    
    async def news_processor(self):
        """Process incoming news and trigger AI analysis"""
        logger.info("News processor started")
        
        while self.running:
            try:
                # Read new news from stream
                messages = redis_manager.read_stream(NEWS_STREAM, count=5, start_id='>')
                
                for message in messages:
                    await self.process_news_message(message)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"News processor error: {e}")
                await asyncio.sleep(10)
    
    async def process_news_message(self, message: Dict):
        """Process a single news message and trigger AI analysis"""
        try:
            data = message['data']
            article_id = data.get('id') or data.get('article_id')
            
            if not article_id or news_stream.is_article_processed(article_id):
                return
            
            # Check if we have API keys
            if not self.openai_api_key:
                logger.warning("No OpenAI API key available for news processing")
                return
            
            # Prepare news content for AI analysis
            title = data.get('title', '')
            content = data.get('content', '')
            news_text = f"{title}\n\n{content}"
            
            if len(news_text.strip()) < 50:  # Skip very short articles
                return
            
            # Queue AI analysis requests with priority
            models_to_use = ['gpt-4o', 'gpt-3.5-turbo']
            
            # Determine priority based on content urgency
            priority = Priority.HIGH if any(word in news_text.lower() for word in [
                'breaking', 'urgent', 'federal reserve', 'bank of japan', 'interest rate'
            ]) else Priority.NORMAL
            
            # Create messages for OpenAI
            messages = [
                {
                    "role": "system", 
                    "content": """You are an expert financial analyst. Analyze the provided news article and respond ONLY in JSON format.
The JSON should contain:
- "trade_warranted": boolean (true if a trade is recommended based SOLELY on this news, false otherwise)
- "pair": string (e.g., "USD/JPY", "EUR/USD", or "N/A" if no trade)
- "direction": string ("long", "short", or "N/A" if no trade)
- "confidence": float (0.0 to 1.0, your confidence in this trade recommendation, 0.0 if no trade)
- "reasoning": string (brief justification, max 100 words)
- "stop_loss_pips": integer (0 if no trade)
- "take_profit_pips": integer (0 if no trade)
- "suggested_lot_size_factor": float (0.0 to 1.0)
If no trade, set trade_warranted to false and other fields to "N/A" or 0."""
                },
                {
                    "role": "user",
                    "content": f"Analyze the following news article and provide a trade plan in JSON format based on its content:\n\n---\n{news_text}\n---"
                }
            ]
            
            # Queue requests for each model
            request_ids = []
            for model in models_to_use:
                request_id = await openai_manager.queue_request(
                    model=model,
                    messages=messages,
                    priority=priority,
                    max_tokens=300,
                    temperature=0.3,
                    news_id=article_id,
                    purpose="realtime_news_analysis"
                )
                request_ids.append((model, request_id))
            
            # Wait for results (non-blocking, results will be processed by worker)
            for model, request_id in request_ids:
                # Store mapping for later result processing
                redis_manager.set_cache(
                    f"news_analysis_mapping:{article_id}:{model}", 
                    request_id, 
                    ttl=3600
                )
            
            logger.debug(f"Queued AI analysis for article {article_id} with {len(models_to_use)} models")
            
        except Exception as e:
            logger.error(f"News message processing error: {e}")
    
    async def ai_decision_processor(self):
        """Process AI decisions and generate trade signals"""
        logger.info("AI decision processor started")
        
        while self.running:
            try:
                # Read new AI decisions
                messages = redis_manager.read_stream(AI_DECISIONS_STREAM, count=5, start_id='>')
                
                for message in messages:
                    await self.process_ai_decision(message)
                
                await asyncio.sleep(3)  # Check every 3 seconds
                
            except Exception as e:
                logger.error(f"AI decision processor error: {e}")
                await asyncio.sleep(10)
    
    async def process_ai_decision(self, message: Dict):
        """Process AI trading decision and generate signals"""
        try:
            data = message['data']
            
            trade_warranted = data.get('trade_warranted', False)
            if not trade_warranted:
                return
            
            pair = data.get('pair')
            direction = data.get('direction')
            confidence = float(data.get('confidence', 0))
            
            if not pair or not direction or confidence < 0.6:  # Only high-confidence trades
                return
            
            # Get current technical signals for signal fusion
            cached_features = feature_builder.get_cached_features(pair)
            cached_signals = redis_manager.get_cache(f"signals:{pair}")
            
            if cached_features and cached_signals:
                # Generate fused signal
                sentiment_score = confidence if direction == 'long' else -confidence
                
                fused_signal = signal_fusion.predict_signal(
                    cached_signals,
                    sentiment_score,
                    cached_features,
                    confidence
                )
                
                # Create trade signal if conditions are met
                if abs(fused_signal['signal']) > 0 and fused_signal['confidence'] > 0.7:
                    trade_signal = {
                        'pair': pair,
                        'direction': 'long' if fused_signal['signal'] > 0 else 'short',
                        'confidence': fused_signal['confidence'],
                        'ai_confidence': confidence,
                        'technical_signal': fused_signal.get('technical_signal', 0),
                        'sentiment_signal': fused_signal.get('sentiment_signal', 0),
                        'stop_loss_pips': data.get('stop_loss_pips', 50),
                        'take_profit_pips': data.get('take_profit_pips', 150),
                        'lot_size_factor': data.get('suggested_lot_size_factor', 0.1),
                        'source_article': data.get('article_id'),
                        'timestamp': datetime.datetime.now().isoformat()
                    }
                    
                    # Add to trade signals stream
                    redis_manager.add_to_stream(TRADE_SIGNALS_STREAM, trade_signal)
                    
                    logger.info(f"Trade signal generated: {pair} {direction} (confidence: {fused_signal['confidence']:.2f})")
            
        except Exception as e:
            logger.error(f"AI decision processing error: {e}")
    
    async def trade_signal_processor(self):
        """Process trade signals and execute trades (or log for demo)"""
        logger.info("Trade signal processor started")
        
        while self.running:
            try:
                # Read new trade signals
                messages = redis_manager.read_stream(TRADE_SIGNALS_STREAM, count=3, start_id='>')
                
                for message in messages:
                    await self.process_trade_signal(message)
                
                await asyncio.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                logger.error(f"Trade signal processor error: {e}")
                await asyncio.sleep(10)
    
    async def process_trade_signal(self, message: Dict):
        """Process trade signal and execute trade (demo mode for now)"""
        try:
            data = message['data']
            
            pair = data.get('pair')
            direction = data.get('direction')
            confidence = data.get('confidence', 0)
            
            # For now, just log the trade signal (demo mode)
            trade_log = {
                'signal_id': message['id'],
                'pair': pair,
                'direction': direction,
                'confidence': confidence,
                'stop_loss_pips': data.get('stop_loss_pips'),
                'take_profit_pips': data.get('take_profit_pips'),
                'lot_size_factor': data.get('lot_size_factor'),
                'timestamp': datetime.datetime.now().isoformat(),
                'status': 'demo_logged'
            }
            
            # Log to Redis for tracking
            redis_manager.add_to_stream('trade_executions', trade_log)
            
            logger.info(f"DEMO TRADE: {direction} {pair} (confidence: {confidence:.2f})")
            
            # TODO: Implement actual trade execution based on exchange configuration
            # if crypto_trader.is_connected():
            #     result = crypto_trader.place_market_order(pair, direction, amount)
            
        except Exception as e:
            logger.error(f"Trade signal processing error: {e}")
    
    async def crypto_stream_processor(self):
        """Stream crypto price data from exchanges"""
        logger.info("Crypto stream processor started")
        
        while self.running:
            try:
                if crypto_trader.is_connected():
                    # Stream crypto data to Redis
                    crypto_trader.stream_to_redis()
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Crypto stream processor error: {e}")
                await asyncio.sleep(60)
    
    async def bitget_websocket_processor(self):
        """Process Bitget WebSocket real-time data"""
        logger.info("Bitget WebSocket processor started")
        
        while self.running:
            try:
                # Initialize Bitget client if not exists
                if not self.bitget_client:
                    self.bitget_client = BitgetWebSocketClient()
                    self.bitget_processor = BitgetDataProcessor()
                
                # Connect to Bitget WebSocket
                success = await self.bitget_client.connect()
                if not success:
                    logger.error("Failed to connect to Bitget WebSocket")
                    await asyncio.sleep(30)
                    continue
                
                # Subscribe to major cryptocurrency pairs
                major_pairs = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "XRPUSDT", "SOLUSDT"]
                
                for symbol in major_pairs:
                    await self.bitget_client.subscribe_ticker(symbol, self.on_bitget_ticker)
                    await asyncio.sleep(0.5)  # Avoid rate limiting
                
                # Subscribe to orderbook for BTC (for spread analysis)
                await self.bitget_client.subscribe_orderbook("BTCUSDT", callback=self.on_bitget_orderbook)
                
                logger.info(f"Subscribed to {len(major_pairs)} Bitget symbols")
                
                # Listen for messages
                await self.bitget_client.listen()
                
            except Exception as e:
                logger.error(f"Bitget WebSocket processor error: {e}")
                
                # Cleanup on error
                if self.bitget_client:
                    try:
                        await self.bitget_client.disconnect()
                    except:
                        pass
                    self.bitget_client = None
                
                await asyncio.sleep(30)  # Wait before retry
    
    async def on_bitget_ticker(self, data):
        """Handle Bitget ticker updates"""
        try:
            processed = await self.bitget_processor.process_ticker_data(data)
            if not processed:
                return
            
            # Convert to our price data format
            price_data = {
                'pair': processed['symbol'],
                'timestamp': processed['timestamp'].isoformat(),
                'price': processed['price'],
                'bid': processed['bid'],
                'ask': processed['ask'],
                'volume': processed['volume'],
                'change_24h': processed['change_24h'],
                'source': 'bitget'
            }
            
            # Publish to Redis price stream
            redis_manager.add_to_stream(PRICE_STREAM, price_data)
            
            # Log significant price movements (>1%)
            if abs(processed['change_24h']) > 1.0:
                logger.info(f"Significant movement: {processed['symbol']} {processed['change_24h']:+.2f}% at ${processed['price']:.4f}")
            
        except Exception as e:
            logger.error(f"Bitget ticker processing error: {e}")
    
    async def on_bitget_orderbook(self, data):
        """Handle Bitget orderbook updates"""
        try:
            processed = await self.bitget_processor.process_orderbook_data(data)
            if not processed:
                return
            
            # Calculate spread analytics
            if processed['bids'] and processed['asks']:
                best_bid = processed['bids'][0][0]
                best_ask = processed['asks'][0][0]
                spread = best_ask - best_bid
                spread_pct = (spread / best_bid) * 100 if best_bid > 0 else 0
                
                # Publish spread data for trading decisions
                spread_data = {
                    'pair': processed['symbol'],
                    'timestamp': processed['timestamp'].isoformat(),
                    'best_bid': best_bid,
                    'best_ask': best_ask,
                    'spread': spread,
                    'spread_pct': spread_pct,
                    'source': 'bitget'
                }
                
                redis_manager.add_to_stream('spread_analysis', spread_data)
                
                # Alert on wide spreads (>0.1%)
                if spread_pct > 0.1:
                    logger.warning(f"Wide spread detected: {processed['symbol']} {spread_pct:.3f}%")
            
        except Exception as e:
            logger.error(f"Bitget orderbook processing error: {e}")
    
    def get_processor_status(self) -> Dict[str, Any]:
        """Get status of all processors"""
        return {
            'running': self.running,
            'stream_stats': {
                'prices': redis_manager.get_stream_info(PRICE_STREAM),
                'news': redis_manager.get_stream_info(NEWS_STREAM),
                'ai_decisions': redis_manager.get_stream_info(AI_DECISIONS_STREAM),
                'trade_signals': redis_manager.get_stream_info(TRADE_SIGNALS_STREAM)
            },
            'api_keys_configured': {
                'openai': bool(self.openai_api_key),
                'news': bool(self.news_api_key)
            }
        }

# Global realtime processor instance
realtime_processor = RealtimeProcessor()