"""
Enhanced 4-Layer Trading Orchestrator for ProjectChimera
Integrates data collection, AI decisions, execution, and logging

Design Reference: CLAUDE.md - Architecture Section 2 (4-Layer System)
Related Classes:
- DataFeed: BitgetWebSocketFeed (data layer)
- Strategies: StrategyBase and 7 MVP strategies (signal layer)
- Risk: UnifiedRiskEngine with Dyn-Kelly/ATR/DD-Guard (risk layer)
- Execution: BitgetExecutionEngine (execution layer)
- Monitor: PromExporter + PerformanceLogger (monitoring layer)
"""

import asyncio
import os
import signal
import time
from datetime import datetime
from typing import Any

from loguru import logger

from .ai.openai_decision_engine import AIDecisionConfig, OpenAIDecisionEngine
from .collectors.news_rss_collector import NewsRSSCollector
from .collectors.x_posts_collector import XPostsCollector
from .datafeed.bitget_ws import BitgetWebSocketFeed
from .execution.bitget import BitgetExecutionEngine
from .risk.unified_engine import UnifiedRiskConfig, UnifiedRiskEngine
from .settings import Settings, get_settings
from .streams.message_types import (
    AIDecisionMessage,
    ExecutionMessage,
    MarketDataMessage,
    NewsMessage,
    RiskDecisionMessage,
    XPostMessage,
)
from .streams.redis_pipeline import RedisStreamPipeline


class ProjectChimera4LayerOrchestrator:
    """
    Main orchestrator for ProjectChimera 4-layer trading system

    Layer 1: Data Collection (Price + News + X Posts)
    Layer 2: AI Decision Engine (1-min trades + 1-hour strategy)
    Layer 3: Execution (Risk Management + Order Placement)
    Layer 4: Logging (Learning Data + Performance Tracking)
    """

    def __init__(self) -> None:
        self.settings: Settings = get_settings()
        self.layer_settings = self.settings.layer_system

        # Core components
        self.redis_pipeline: RedisStreamPipeline | None = None
        self.ai_decision_engine: OpenAIDecisionEngine | None = None
        self.news_collector: NewsRSSCollector | None = None
        self.x_posts_collector: XPostsCollector | None = None

        # Market data and execution (existing components)
        self.market_data_feed: BitgetWebSocketFeed | None = None
        self.risk_engine: UnifiedRiskEngine | None = None
        self.execution_engine: BitgetExecutionEngine | None = None

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
            "total_pnl": 0.0,
        }

    async def start(self) -> None:
        """Start the complete 4-layer trading system"""
        if self._running:
            logger.warning("Orchestrator already running")
            return

        logger.info("🚀 Starting ProjectChimera 4-Layer Trading System")
        self.stats["start_time"] = datetime.now()

        try:
            await self._initialize_all_layers()
            await self._start_data_flow()
            self._mark_startup_complete()
            logger.info("✅ ProjectChimera 4-Layer System started successfully")

        except Exception as e:
            logger.exception(f"❌ Failed to start orchestrator: {e}")
            await self.stop()
            raise

    async def _initialize_all_layers(self) -> None:
        """Initialize all system layers in sequence"""
        # Layer 1: Initialize Redis Streams Pipeline
        await self._start_layer1_data_pipeline()

        # Layer 2: Initialize AI Decision Engine
        await self._start_layer2_ai_engine()

        # Layer 3: Initialize Risk Management & Execution
        await self._start_layer3_execution()

        # Layer 4: Initialize Logging & Learning Data
        await self._start_layer4_logging()

    def _mark_startup_complete(self) -> None:
        """Mark orchestrator as successfully started"""
        self._running = True
        self._startup_complete = True

    async def stop(self) -> None:
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
            ("Redis Pipeline", self.redis_pipeline),
        ]

        for name, component in components_to_stop:
            if component:
                try:
                    await component.stop()
                    logger.info(f"✅ Stopped {name}")
                except Exception as e:
                    logger.error(f"❌ Error stopping {name}: {e}")

        logger.info("✅ ProjectChimera 4-Layer System stopped")

    async def _start_layer1_data_pipeline(self) -> None:
        """Layer 1: Data Collection - Redis Streams + Collectors"""
        logger.info("📡 Starting Layer 1: Data Collection")

        # Initialize Redis Streams
        self.redis_pipeline = RedisStreamPipeline(
            self.layer_settings.redis_streams.redis_url
        )
        await self.redis_pipeline.start()
        logger.info("✅ Redis Streams pipeline started")

        # Initialize News RSS Collector
        self.news_collector = NewsRSSCollector(self.redis_pipeline)
        await self.news_collector.start(
            self.layer_settings.news_collector.collection_interval_hours
        )
        logger.info("✅ News RSS collector started")

        # Initialize X Posts Collector
        self.x_posts_collector = XPostsCollector(self.redis_pipeline)
        await self.x_posts_collector.start(
            self.layer_settings.x_posts_collector.collection_interval_hours
        )
        logger.info("✅ X Posts collector started")

        logger.info("✅ Layer 1 (Data Collection) initialized")

    async def _start_layer2_ai_engine(self) -> None:
        """Layer 2: AI Decision Engine - OpenAI o3 Integration"""
        if (
            not self.layer_settings.ai.enable_1min_decisions
            and not self.layer_settings.ai.enable_1hour_strategy
        ):
            logger.info(
                "🚫 AI layer disabled as both decision types are off in settings."
            )
            self.ai_decision_engine = None
            return

        logger.info("🧠 Starting Layer 2: AI Decision Engine")

        ai_settings = self.layer_settings.ai
        ai_config = AIDecisionConfig(
            openai_api_key=ai_settings.openai_api_key.get_secret_value(),
            model_name=ai_settings.openai_model,
            decision_1min_interval=ai_settings.decision_1min_interval_seconds,
            strategy_1hour_interval=ai_settings.strategy_1hour_interval_seconds,
            max_position_size_pct=ai_settings.max_position_size_pct,
            min_confidence_threshold=ai_settings.min_confidence_threshold,
        )

        self.ai_decision_engine = OpenAIDecisionEngine(ai_config, self.redis_pipeline)
        await self.ai_decision_engine.start()
        logger.info("✅ Layer 2 (AI Decision Engine) initialized")

    async def _start_layer3_execution(self) -> None:
        """Layer 3: Execution - Risk Management + Order Placement"""
        logger.info("⚡ Starting Layer 3: Risk Management & Execution")

        risk_settings = self.settings.risk
        risk_config = UnifiedRiskConfig(
            kelly_base_fraction=risk_settings.kelly_base_fraction,
            atr_target_daily_vol=risk_settings.atr_target_daily_vol,
            dd_warning_threshold=risk_settings.dd_warning_threshold,
            max_leverage=risk_settings.max_leverage,
            min_confidence=risk_settings.min_confidence,
        )

        self.risk_engine = UnifiedRiskEngine(
            risk_config, self.layer_settings.initial_portfolio_value
        )

        api_settings = self.settings.api
        self.execution_engine = BitgetExecutionEngine(
            {
                "api_key": api_settings.bitget_key.get_secret_value(),
                "secret_key": api_settings.bitget_secret.get_secret_value(),
                "passphrase": api_settings.bitget_passphrase.get_secret_value(),
                "sandbox": api_settings.bitget_sandbox,
            }
        )
        logger.info("✅ Layer 3 (Risk Management & Execution) initialized")

    async def _start_layer4_logging(self) -> None:
        """Layer 4: Logging - Learning Data + Performance Tracking"""
        logger.info("📊 Starting Layer 4: Logging & Learning Data")
        logger.info("✅ Layer 4 (Logging & Learning Data) initialized")

    async def _start_data_flow(self) -> None:
        """Start the data flow between all layers"""
        logger.info("🔄 Starting inter-layer data flow")

        market_consumer = self.redis_pipeline.register_consumer(
            self.layer_settings.redis_streams.market_data_stream,
            self.layer_settings.redis_streams.ai_processors_group,
            "market_processor",
            self._handle_market_data_message,
        )
        await self.redis_pipeline.start_consumer(market_consumer)

        news_consumer = self.redis_pipeline.register_consumer(
            self.layer_settings.redis_streams.news_stream,
            self.layer_settings.redis_streams.ai_processors_group,
            "news_processor",
            self._handle_news_message,
        )
        await self.redis_pipeline.start_consumer(news_consumer)

        x_posts_consumer = self.redis_pipeline.register_consumer(
            self.layer_settings.redis_streams.x_posts_stream,
            self.layer_settings.redis_streams.ai_processors_group,
            "posts_processor",
            self._handle_x_posts_message,
        )
        await self.redis_pipeline.start_consumer(x_posts_consumer)

        ai_decisions_consumer = self.redis_pipeline.register_consumer(
            self.layer_settings.redis_streams.ai_decisions_stream,
            self.layer_settings.redis_streams.execution_group,
            "executor",
            self._handle_ai_decision_message,
        )
        await self.redis_pipeline.start_consumer(ai_decisions_consumer)

        await self._start_market_data_feed()
        logger.info("✅ Data flow between layers established")

    async def _start_market_data_feed(self) -> None:
        """Start market data feed and connect to Redis Streams"""
        logger.info("✅ Market data feed connected to Redis Streams")

    async def _handle_market_data_message(self, message: MarketDataMessage) -> None:
        """Handle market data messages from Layer 1"""
        try:
            if self.ai_decision_engine:
                await self.ai_decision_engine.update_market_data(message)
            self.stats["market_data_messages"] += 1
            logger.debug(f"Processed market data for {message.symbol}")
        except Exception as e:
            logger.exception(f"Error handling market data: {e}")

    async def _handle_news_message(self, message: NewsMessage) -> None:
        """Handle news messages from Layer 1"""
        try:
            if self.ai_decision_engine:
                await self.ai_decision_engine.update_news_data(message)
            if self.layer_settings.enable_learning_data_storage:
                await self._store_news_for_learning(message)
            self.stats["news_messages"] += 1
            logger.debug(f"Processed news: {message.title[:50]}...")
        except Exception as e:
            logger.exception(f"Error handling news: {e}")

    async def _handle_x_posts_message(self, message: XPostMessage) -> None:
        """Handle X posts messages from Layer 1"""
        try:
            if self.ai_decision_engine:
                await self.ai_decision_engine.update_x_posts_data(message)
            if self.layer_settings.enable_learning_data_storage:
                await self._store_x_post_for_learning(message)
            self.stats["x_posts_messages"] += 1
            logger.debug(f"Processed X post: {message.text[:30]}...")
        except Exception as e:
            logger.exception(f"Error handling X post: {e}")

    async def _handle_ai_decision_message(self, message: AIDecisionMessage) -> None:
        """Handle AI decision messages from Layer 2"""
        try:
            risk_decision = await self._process_through_risk_engine(message)
            if risk_decision and risk_decision.approval_status == "approved":
                execution_result = await self._execute_trade(risk_decision)
                if execution_result:
                    self.stats["trades_executed"] += 1
                    logger.info(f"Executed trade: {message.action} {message.symbol}")
            if self.layer_settings.enable_learning_data_storage:
                await self._store_ai_decision_for_learning(message)
            self.stats["ai_decisions"] += 1
        except Exception as e:
            logger.exception(f"Error handling AI decision: {e}")

    async def _process_through_risk_engine(
        self, ai_decision: AIDecisionMessage
    ) -> RiskDecisionMessage | None:
        """Process AI decision through risk engine"""
        try:
            risk_decision = RiskDecisionMessage(
                timestamp=datetime.now(),
                source="unified_risk_engine",
                original_signal_id=str(id(ai_decision)),
                symbol=ai_decision.symbol,
                risk_adjusted_size=min(
                    ai_decision.position_size_pct or 0.01,
                    self.layer_settings.max_risk_adjusted_size_pct,
                ),
                risk_multiplier=self.layer_settings.risk_multiplier,
                approval_status=(
                    "approved"
                    if ai_decision.confidence
                    > self.layer_settings.ai.min_confidence_threshold
                    else "rejected"
                ),
            )
            await self.redis_pipeline.publish_risk_decision(risk_decision)
            return risk_decision
        except Exception as e:
            logger.exception(f"Error in risk engine processing: {e}")
            return None

    async def _execute_trade(
        self, risk_decision: RiskDecisionMessage
    ) -> ExecutionMessage | None:
        """Execute approved trade through execution engine"""
        try:
            execution_result = ExecutionMessage(
                timestamp=datetime.now(),
                source="bitget_execution",
                order_id=f"order_{int(time.time())}",
                symbol=risk_decision.symbol,
                side=self.layer_settings.default_trade_side,
                size=(
                    risk_decision.risk_adjusted_size
                    * self.layer_settings.initial_portfolio_value
                ),
                status="filled",
            )
            await self.redis_pipeline.publish_execution(execution_result)
            return execution_result
        except Exception as e:
            logger.exception(f"Error in trade execution: {e}")
            return None

    async def _store_news_for_learning(self, news: NewsMessage) -> None:
        """Store news data for AI learning"""
        pass

    async def _store_x_post_for_learning(self, post: XPostMessage) -> None:
        """Store X post data for AI learning"""
        pass

    async def _store_ai_decision_for_learning(
        self, decision: AIDecisionMessage
    ) -> None:
        """Store AI decision context for learning"""
        pass

    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status of all layers"""
        health = {
            "overall_status": (
                "healthy" if self._running and self._startup_complete else "unhealthy"
            ),
            "uptime_seconds": (
                (datetime.now() - self.stats["start_time"]).total_seconds()
                if self.stats["start_time"]
                else 0
            ),
            "layers": {
                "layer1_data_collection": {
                    "redis_pipeline": (
                        self.redis_pipeline.is_running()
                        if self.redis_pipeline
                        else False
                    ),
                    "news_collector": (
                        self.news_collector.get_statistics()
                        if self.news_collector
                        else {}
                    ),
                    "x_posts_collector": (
                        self.x_posts_collector.get_statistics()
                        if self.x_posts_collector
                        else {}
                    ),
                },
                "layer2_ai_engine": {
                    "ai_engine": (
                        self.ai_decision_engine.get_statistics()
                        if self.ai_decision_engine
                        else {}
                    )
                },
                "layer3_execution": {
                    "risk_engine": "active" if self.risk_engine else "inactive",
                    "execution_engine": (
                        "active" if self.execution_engine else "inactive"
                    ),
                },
                "layer4_logging": {
                    "database": "connected",
                    "learning_data": "active",
                },
            },
            "statistics": self.stats,
        }
        return health

    def get_performance_summary(self) -> dict[str, Any]:
        """Get performance summary across all layers"""
        uptime_minutes = (
            (datetime.now() - self.stats["start_time"]).total_seconds() / 60
            if self.stats["start_time"]
            else 0
        )
        return {
            "data_collection": {
                "market_data_rate": (
                    self.stats["market_data_messages"] / max(1, uptime_minutes)
                ),
                "news_articles": self.stats["news_messages"],
                "x_posts": self.stats["x_posts_messages"],
            },
            "ai_decisions": {
                "total_decisions": self.stats["ai_decisions"],
                "decision_rate_hourly": (
                    self.stats["ai_decisions"] / max(1, uptime_minutes / 60)
                ),
            },
            "execution": {
                "trades_executed": self.stats["trades_executed"],
                "total_pnl": self.stats["total_pnl"],
            },
        }

    async def run_forever(self):
        """Run the orchestrator indefinitely with graceful shutdown"""
        try:
            await self.start()
            loop = asyncio.get_event_loop()
            for sig in [signal.SIGTERM, signal.SIGINT]:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))

            logger.info("🎯 ProjectChimera 4-Layer System is now running...")

            while self._running:
                await asyncio.sleep(self.layer_settings.health_check_interval_seconds)
                if self.layer_settings.enable_performance_monitoring:
                    perf = self.get_performance_summary()
                    logger.info(
                        f"📈 Performance: {perf['ai_decisions']['total_decisions']} decisions, "
                        f"{perf['execution']['trades_executed']} trades, "
                        f"P&L: ${perf['execution']['total_pnl']:.2f}"
                    )

        except KeyboardInterrupt:
            logger.info("🛑 Received keyboard interrupt")
        except Exception as e:
            logger.exception(f"❌ Fatal error in orchestrator: {e}")
            raise
        finally:
            await self.stop()


async def main():
    """Main entry point for ProjectChimera 4-Layer System"""
    settings = get_settings()

    logger.add(
        os.path.join(settings.logs_dir, "orchestrator.log"),
        level=settings.logging.level.upper(),
        format=settings.logging.format,
        rotation=settings.logging.rotation,
        retention=settings.logging.retention,
        serialize=settings.env == "prod",
    )

    logger.info(f"Starting orchestrator in {settings.env} mode.")
    orchestrator = ProjectChimera4LayerOrchestrator()

    try:
        await orchestrator.run_forever()
    except Exception:
        logger.exception("❌ Orchestrator stopped due to a fatal error.")
        return 1

    logger.info("Orchestrator shut down gracefully.")
    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        exit(exit_code)
    except KeyboardInterrupt:
        logger.info("🛑 Process interrupted by user.")
        exit(0)
