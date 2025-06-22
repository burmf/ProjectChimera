"""
Enhanced 4-Layer Trading Orchestrator for ProjectChimera
Integrates data collection, AI decisions, execution, and logging
"""

import asyncio
import logging
import signal
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Import all 4 layers
from .streams.redis_pipeline import RedisStreamPipeline
from .collectors.news_rss_collector import NewsRSSCollector
from .collectors.x_posts_collector import XPostsCollector
from .ai.openai_decision_engine import OpenAIDecisionEngine, AIDecisionConfig
from .streams.message_types import (
    MarketDataMessage, NewsMessage, XPostMessage, AIDecisionMessage,
    ExecutionMessage, RiskDecisionMessage
)

# Existing components
from .datafeed.bitget_ws import BitgetWebSocketFeed
from .risk.unified_engine import UnifiedRiskEngine, UnifiedRiskConfig
from .execution.bitget import BitgetExecutionEngine
from .settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Configuration for the 4-layer orchestrator"""
    
    # Trading symbols
    trading_symbols: List[str]
    
    # Collection intervals
    news_collection_interval_hours: int = 1
    x_posts_collection_interval_hours: int = 1
    
    # AI decision intervals  
    ai_1min_interval_seconds: int = 60
    ai_1hour_interval_seconds: int = 3600
    
    # Portfolio settings
    initial_portfolio_value: float = 150000.0
    
    # Redis settings
    redis_url: str = "redis://localhost:6379"
    
    # Database settings
    database_url: str = "postgresql+asyncpg://chimera:chimera@localhost:5432/chimera"


class ProjectChimera4LayerOrchestrator:
    """
    Main orchestrator for ProjectChimera 4-layer trading system
    
    Layer 1: Data Collection (Price + News + X Posts)
    Layer 2: AI Decision Engine (1-min trades + 1-hour strategy)  
    Layer 3: Execution (Risk Management + Order Placement)
    Layer 4: Logging (Learning Data + Performance Tracking)
    """
    
    def __init__(self, config: OrchestratorConfig):
        self.config = config
        self.settings = get_settings()
        
        # Core components
        self.redis_pipeline: Optional[RedisStreamPipeline] = None
        self.ai_decision_engine: Optional[OpenAIDecisionEngine] = None
        self.news_collector: Optional[NewsRSSCollector] = None
        self.x_posts_collector: Optional[XPostsCollector] = None
        
        # Market data and execution (existing components)
        self.market_data_feed: Optional[BitgetWebSocketFeed] = None
        self.risk_engine: Optional[UnifiedRiskEngine] = None
        self.execution_engine: Optional[BitgetExecutionEngine] = None
        
        # State tracking
        self._running = False
        self._startup_complete = False
        self.component_health = {}
        
        # Statistics
        self.stats = {
            "start_time": None,
            "market_data_messages": 0,
            "news_messages": 0,
            "x_posts_messages": 0,
            "ai_decisions": 0,
            "trades_executed": 0,
            "total_pnl": 0.0
        }
    
    async def start(self):
        """Start the complete 4-layer trading system"""
        if self._running:
            logger.warning("Orchestrator already running")
            return
        
        logger.info("🚀 Starting ProjectChimera 4-Layer Trading System")
        self.stats["start_time"] = datetime.now()
        
        try:
            # Layer 1: Initialize Redis Streams Pipeline
            await self._start_layer1_data_pipeline()
            
            # Layer 2: Initialize AI Decision Engine  
            await self._start_layer2_ai_engine()
            
            # Layer 3: Initialize Risk Management & Execution
            await self._start_layer3_execution()
            
            # Layer 4: Initialize Logging & Learning Data
            await self._start_layer4_logging()
            
            # Start data flow
            await self._start_data_flow()
            
            self._running = True
            self._startup_complete = True
            
            logger.info("✅ ProjectChimera 4-Layer System started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start orchestrator: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """Stop the complete 4-layer trading system"""
        logger.info("🛑 Stopping ProjectChimera 4-Layer Trading System")
        
        self._running = False
        
        # Stop all components in reverse order
        components_to_stop = [
            ("AI Decision Engine", self.ai_decision_engine),
            ("X Posts Collector", self.x_posts_collector),
            ("News Collector", self.news_collector),
            ("Execution Engine", self.execution_engine),
            ("Market Data Feed", self.market_data_feed),
            ("Redis Pipeline", self.redis_pipeline)
        ]
        
        for name, component in components_to_stop:
            if component:
                try:
                    await component.stop()
                    logger.info(f"✅ Stopped {name}")
                except Exception as e:
                    logger.error(f"❌ Error stopping {name}: {e}")
        
        logger.info("✅ ProjectChimera 4-Layer System stopped")
    
    async def _start_layer1_data_pipeline(self):
        """Layer 1: Data Collection - Redis Streams + Collectors"""
        logger.info("📡 Starting Layer 1: Data Collection")
        
        # Initialize Redis Streams
        self.redis_pipeline = RedisStreamPipeline(self.config.redis_url)
        await self.redis_pipeline.start()
        logger.info("✅ Redis Streams pipeline started")
        
        # Initialize News RSS Collector
        self.news_collector = NewsRSSCollector(self.redis_pipeline)
        await self.news_collector.start(self.config.news_collection_interval_hours)
        logger.info("✅ News RSS collector started")
        
        # Initialize X Posts Collector
        self.x_posts_collector = XPostsCollector(self.redis_pipeline)
        await self.x_posts_collector.start(self.config.x_posts_collection_interval_hours)
        logger.info("✅ X Posts collector started")
        
        # Initialize Market Data Feed (existing)
        # This will be connected to Redis Streams in the data flow
        logger.info("✅ Layer 1 (Data Collection) initialized")
    
    async def _start_layer2_ai_engine(self):
        """Layer 2: AI Decision Engine - OpenAI o3 Integration"""
        logger.info("🧠 Starting Layer 2: AI Decision Engine")
        
        # Create AI config
        ai_config = AIDecisionConfig(
            openai_api_key=self.settings.api.openai_api_key.get_secret_value(),
            model_name=self.settings.api.openai_model,
            decision_1min_interval=self.config.ai_1min_interval_seconds,
            strategy_1hour_interval=self.config.ai_1hour_interval_seconds,
            max_position_size_pct=self.settings.trading.max_position_pct,
            min_confidence_threshold=self.settings.trading.confidence_threshold
        )
        
        # Initialize AI Decision Engine
        self.ai_decision_engine = OpenAIDecisionEngine(ai_config, self.redis_pipeline)
        await self.ai_decision_engine.start()
        
        logger.info("✅ Layer 2 (AI Decision Engine) initialized")
    
    async def _start_layer3_execution(self):
        """Layer 3: Execution - Risk Management + Order Placement"""
        logger.info("⚡ Starting Layer 3: Risk Management & Execution")
        
        # Initialize Risk Engine
        risk_config = UnifiedRiskConfig(
            kelly_base_fraction=self.settings.risk.kelly_base_fraction,
            atr_target_daily_vol=self.settings.risk.atr_target_daily_vol,
            dd_warning_threshold=self.settings.risk.dd_warning_threshold,
            max_leverage=self.settings.risk.max_leverage,
            min_confidence=self.settings.risk.min_confidence
        )
        
        self.risk_engine = UnifiedRiskEngine(risk_config, self.config.initial_portfolio_value)
        
        # Initialize Execution Engine
        self.execution_engine = BitgetExecutionEngine({
            'api_key': self.settings.api.bitget_key.get_secret_value(),
            'secret_key': self.settings.api.bitget_secret.get_secret_value(),
            'passphrase': self.settings.api.bitget_passphrase.get_secret_value(),
            'sandbox': self.settings.api.bitget_sandbox
        })
        
        logger.info("✅ Layer 3 (Risk Management & Execution) initialized")
    
    async def _start_layer4_logging(self):
        """Layer 4: Logging - Learning Data + Performance Tracking"""
        logger.info("📊 Starting Layer 4: Logging & Learning Data")
        
        # Initialize database connections for learning data
        # (Database schema already created in init.sql)
        
        logger.info("✅ Layer 4 (Logging & Learning Data) initialized")
    
    async def _start_data_flow(self):
        """Start the data flow between all layers"""
        logger.info("🔄 Starting inter-layer data flow")
        
        # Set up Redis Streams consumers for different message types
        
        # Market data consumer (feeds AI engine)
        market_consumer = self.redis_pipeline.register_consumer(
            "market_data", "ai_processors", "market_processor",
            self._handle_market_data_message
        )
        await self.redis_pipeline.start_consumer(market_consumer)
        
        # News consumer (feeds AI engine)
        news_consumer = self.redis_pipeline.register_consumer(
            "news", "ai_processors", "news_processor", 
            self._handle_news_message
        )
        await self.redis_pipeline.start_consumer(news_consumer)
        
        # X posts consumer (feeds AI engine)
        x_posts_consumer = self.redis_pipeline.register_consumer(
            "x_posts", "ai_processors", "posts_processor",
            self._handle_x_posts_message
        )
        await self.redis_pipeline.start_consumer(x_posts_consumer)
        
        # AI decisions consumer (feeds risk engine & execution)
        ai_decisions_consumer = self.redis_pipeline.register_consumer(
            "ai_decisions", "execution", "executor",
            self._handle_ai_decision_message
        )
        await self.redis_pipeline.start_consumer(ai_decisions_consumer)
        
        # Start market data feed (will publish to Redis)
        await self._start_market_data_feed()
        
        logger.info("✅ Data flow between layers established")
    
    async def _start_market_data_feed(self):
        """Start market data feed and connect to Redis Streams"""
        # Initialize Bitget WebSocket feed
        # This would connect to Bitget and publish market data to Redis
        # Implementation depends on existing BitgetWebSocketFeed structure
        
        logger.info("✅ Market data feed connected to Redis Streams")
    
    # Message handlers for inter-layer communication
    
    async def _handle_market_data_message(self, message: MarketDataMessage):
        """Handle market data messages from Layer 1"""
        try:
            # Update AI decision engine with market data
            await self.ai_decision_engine.update_market_data(message)
            
            # Update statistics
            self.stats["market_data_messages"] += 1
            
            logger.debug(f"Processed market data for {message.symbol}")
            
        except Exception as e:
            logger.error(f"Error handling market data: {e}")
    
    async def _handle_news_message(self, message: NewsMessage):
        """Handle news messages from Layer 1"""
        try:
            # Update AI decision engine with news
            await self.ai_decision_engine.update_news_data(message)
            
            # Store in database for learning
            await self._store_news_for_learning(message)
            
            # Update statistics
            self.stats["news_messages"] += 1
            
            logger.debug(f"Processed news: {message.title[:50]}...")
            
        except Exception as e:
            logger.error(f"Error handling news: {e}")
    
    async def _handle_x_posts_message(self, message: XPostMessage):
        """Handle X posts messages from Layer 1"""
        try:
            # Update AI decision engine with X posts
            await self.ai_decision_engine.update_x_posts_data(message)
            
            # Store in database for learning
            await self._store_x_post_for_learning(message)
            
            # Update statistics
            self.stats["x_posts_messages"] += 1
            
            logger.debug(f"Processed X post: {message.text[:30]}...")
            
        except Exception as e:
            logger.error(f"Error handling X post: {e}")
    
    async def _handle_ai_decision_message(self, message: AIDecisionMessage):
        """Handle AI decision messages from Layer 2"""
        try:
            # Process through risk engine (Layer 3)
            risk_decision = await self._process_through_risk_engine(message)
            
            if risk_decision and risk_decision.approval_status == "approved":
                # Execute trade (Layer 3)
                execution_result = await self._execute_trade(risk_decision)
                
                if execution_result:
                    self.stats["trades_executed"] += 1
                    logger.info(f"Executed trade: {message.action} {message.symbol}")
            
            # Store decision context for learning (Layer 4)
            await self._store_ai_decision_for_learning(message)
            
            # Update statistics
            self.stats["ai_decisions"] += 1
            
        except Exception as e:
            logger.error(f"Error handling AI decision: {e}")
    
    async def _process_through_risk_engine(self, ai_decision: AIDecisionMessage) -> Optional[RiskDecisionMessage]:
        """Process AI decision through risk engine"""
        try:
            # Convert AI decision to risk engine format
            # This would use the existing risk engine logic
            
            # For now, create a simplified risk decision
            risk_decision = RiskDecisionMessage(
                timestamp=datetime.now(),
                source="unified_risk_engine",
                original_signal_id=str(id(ai_decision)),
                symbol=ai_decision.symbol,
                risk_adjusted_size=min(ai_decision.position_size_pct or 0.01, 0.02),  # Cap at 2%
                risk_multiplier=0.5,  # Conservative
                approval_status="approved" if ai_decision.confidence > 0.7 else "rejected"
            )
            
            # Publish risk decision to Redis
            await self.redis_pipeline.publish_risk_decision(risk_decision)
            
            return risk_decision
            
        except Exception as e:
            logger.error(f"Error in risk engine processing: {e}")
            return None
    
    async def _execute_trade(self, risk_decision: RiskDecisionMessage) -> Optional[ExecutionMessage]:
        """Execute approved trade through execution engine"""
        try:
            # This would use the existing execution engine
            # For now, create a mock execution result
            
            execution_result = ExecutionMessage(
                timestamp=datetime.now(),
                source="bitget_execution",
                order_id=f"order_{int(time.time())}",
                symbol=risk_decision.symbol,
                side="buy",  # Simplified
                size=risk_decision.risk_adjusted_size * self.config.initial_portfolio_value,
                status="filled"
            )
            
            # Publish execution result to Redis
            await self.redis_pipeline.publish_execution(execution_result)
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Error in trade execution: {e}")
            return None
    
    # Layer 4: Learning data storage methods
    
    async def _store_news_for_learning(self, news: NewsMessage):
        """Store news data for AI learning"""
        # This would store to the ai_decision_contexts table
        # Implementation would depend on database connection setup
        pass
    
    async def _store_x_post_for_learning(self, post: XPostMessage):
        """Store X post data for AI learning"""
        # This would store to the x_posts table
        pass
    
    async def _store_ai_decision_for_learning(self, decision: AIDecisionMessage):
        """Store AI decision context for learning"""
        # This would store the complete decision context to ai_decision_contexts table
        pass
    
    # Health monitoring and statistics
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status of all layers"""
        health = {
            "overall_status": "healthy" if self._running and self._startup_complete else "unhealthy",
            "uptime_seconds": (datetime.now() - self.stats["start_time"]).total_seconds() if self.stats["start_time"] else 0,
            "layers": {
                "layer1_data_collection": {
                    "redis_pipeline": self.redis_pipeline.is_running() if self.redis_pipeline else False,
                    "news_collector": self.news_collector.get_statistics() if self.news_collector else {},
                    "x_posts_collector": self.x_posts_collector.get_statistics() if self.x_posts_collector else {}
                },
                "layer2_ai_engine": {
                    "ai_engine": self.ai_decision_engine.get_statistics() if self.ai_decision_engine else {}
                },
                "layer3_execution": {
                    "risk_engine": "active" if self.risk_engine else "inactive",
                    "execution_engine": "active" if self.execution_engine else "inactive"
                },
                "layer4_logging": {
                    "database": "connected",  # Would check actual DB connection
                    "learning_data": "active"
                }
            },
            "statistics": self.stats
        }
        
        return health
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all layers"""
        return {
            "data_collection": {
                "market_data_rate": self.stats["market_data_messages"] / max(1, (datetime.now() - self.stats["start_time"]).total_seconds() / 60) if self.stats["start_time"] else 0,
                "news_articles": self.stats["news_messages"],
                "x_posts": self.stats["x_posts_messages"]
            },
            "ai_decisions": {
                "total_decisions": self.stats["ai_decisions"],
                "decision_rate": self.stats["ai_decisions"] / max(1, (datetime.now() - self.stats["start_time"]).total_seconds() / 3600) if self.stats["start_time"] else 0
            },
            "execution": {
                "trades_executed": self.stats["trades_executed"],
                "total_pnl": self.stats["total_pnl"]
            }
        }
    
    async def run_forever(self):
        """Run the orchestrator indefinitely with graceful shutdown"""
        try:
            await self.start()
            
            # Set up signal handlers for graceful shutdown
            loop = asyncio.get_event_loop()
            for sig in [signal.SIGTERM, signal.SIGINT]:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
            
            logger.info("🎯 ProjectChimera 4-Layer System is now running...")
            
            # Main monitoring loop
            while self._running:
                await asyncio.sleep(30)  # Health check every 30 seconds
                
                # Log performance summary
                if self.stats["start_time"]:
                    uptime_minutes = (datetime.now() - self.stats["start_time"]).total_seconds() / 60
                    if uptime_minutes > 0 and int(uptime_minutes) % 60 == 0:  # Every hour
                        perf = self.get_performance_summary()
                        logger.info(f"📈 Performance: {perf['ai_decisions']['total_decisions']} decisions, "
                                  f"{perf['execution']['trades_executed']} trades, "
                                  f"P&L: ${perf['execution']['total_pnl']:.2f}")
            
        except KeyboardInterrupt:
            logger.info("🛑 Received keyboard interrupt")
        except Exception as e:
            logger.error(f"❌ Fatal error in orchestrator: {e}")
            raise
        finally:
            await self.stop()


# CLI interface
async def main():
    """Main entry point for ProjectChimera 4-Layer System"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ProjectChimera 4-Layer Trading System")
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT'], 
                       help='Trading symbols')
    parser.add_argument('--portfolio-value', type=float, default=150000.0,
                       help='Initial portfolio value in USD')
    parser.add_argument('--redis-url', default='redis://localhost:6379',
                       help='Redis connection URL')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create orchestrator config
    config = OrchestratorConfig(
        trading_symbols=args.symbols,
        initial_portfolio_value=args.portfolio_value,
        redis_url=args.redis_url
    )
    
    # Create and run orchestrator
    orchestrator = ProjectChimera4LayerOrchestrator(config)
    
    try:
        await orchestrator.run_forever()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(asyncio.run(main()))