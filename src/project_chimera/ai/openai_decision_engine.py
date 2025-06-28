"""
OpenAI o3 Decision Engine for ProjectChimera
Handles AI-driven trading decisions using OpenAI's o3 model

Design Reference: CLAUDE.md - AI Department System (部門別AIシステム)
Related Classes:
- TradingPrompts: Specialized prompts for trading decisions
- TradingDecisionProcessor: JSON response parsing
- RedisStreamPipeline: Message streaming for AI decisions
- AIDecisionMessage: Structured AI output format
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from openai import AsyncOpenAI

from ..streams.message_types import AIDecisionMessage
from ..streams.redis_pipeline import RedisStreamPipeline
from .prompts import TradingPrompts
from .trading_decisions import TradingDecisionProcessor

logger = logging.getLogger(__name__)


@dataclass
class AIDecisionConfig:
    """Configuration for AI decision engine"""

    openai_api_key: str
    model_name: str = "o3-mini"  # Use o3-mini for advanced reasoning
    max_tokens: int = 1000
    temperature: float = 0.1  # Low temperature for consistent trading decisions

    # Decision intervals
    decision_1min_interval: int = 60  # 1 minute
    strategy_1hour_interval: int = 3600  # 1 hour

    # Risk limits
    max_position_size_pct: float = 0.05  # 5% max position
    min_confidence_threshold: float = 0.6  # Minimum confidence to trade

    # Cost management
    max_daily_api_cost: float = 50.0  # $50 daily limit
    cost_tracking_window_hours: int = 24


@dataclass
class AIDecisionResult:
    """Result from AI decision engine"""

    success: bool
    decision: AIDecisionMessage | None
    error_message: str | None
    api_cost: float
    response_time: float
    tokens_used: int


class OpenAIDecisionEngine:
    """
    OpenAI o3 integration for trading decisions
    Handles both 1-minute trading decisions and 1-hour strategy planning
    """

    def __init__(self, config: AIDecisionConfig, redis_pipeline: RedisStreamPipeline):
        self.config = config
        self.redis_pipeline = redis_pipeline
        self.decision_processor = TradingDecisionProcessor()

        # OpenAI client
        self.openai_client = AsyncOpenAI(api_key=config.openai_api_key)

        # Cost tracking
        self.daily_api_cost = 0.0
        self.api_call_history = []  # [(timestamp, cost), ...]

        # Decision tracking
        self.last_1min_decisions = {}  # symbol -> (timestamp, decision)
        self.last_1hour_strategy = None
        self.decision_performance = []  # Track decision outcomes

        # Running state
        self._running = False
        self._decision_1min_task = None
        self._strategy_1hour_task = None

        # Model pricing (estimated for o3, adjust based on actual pricing)
        self.model_pricing = {
            "o3-mini": {"input": 0.0001, "output": 0.0004},  # per token
            "o3": {"input": 0.001, "output": 0.004},  # per token
        }

    async def start(self):
        """Start the AI decision engine"""
        if self._running:
            logger.warning("AI decision engine already running")
            return

        logger.info("Starting OpenAI decision engine")

        # Test OpenAI connection
        try:
            await self._test_openai_connection()
        except Exception as e:
            logger.warning(f"OpenAI connection test failed, continuing without AI: {e}")
            # Don't raise exception, allow system to continue without AI

        self._running = True

        # Start decision loops
        self._decision_1min_task = asyncio.create_task(self._decision_1min_loop())
        self._strategy_1hour_task = asyncio.create_task(self._strategy_1hour_loop())

        logger.info("AI decision engine started successfully")

    async def stop(self):
        """Stop the AI decision engine"""
        logger.info("Stopping AI decision engine")

        self._running = False

        # Cancel running tasks
        if self._decision_1min_task:
            self._decision_1min_task.cancel()
        if self._strategy_1hour_task:
            self._strategy_1hour_task.cancel()

        from contextlib import suppress

        # Wait for tasks to complete
        for task in [self._decision_1min_task, self._strategy_1hour_task]:
            if task:
                with suppress(asyncio.CancelledError):
                    await task

        logger.info("AI decision engine stopped")

    async def _test_openai_connection(self):
        """Test OpenAI API connection"""
        try:
            logger.info(
                f"Testing OpenAI connection with model: {self.config.model_name}"
            )
            logger.info(f"API key configured: {self.config.openai_api_key[:10]}...")

            # Test connection (use max_completion_tokens for o3 models)
            if self.config.model_name.startswith("o3"):
                response = await self.openai_client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": "Test connection"}],
                    max_completion_tokens=10,
                    timeout=60.0,  # o3 models may take longer
                )
            else:
                response = await self.openai_client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": "Test connection"}],
                    max_tokens=10,
                    timeout=30.0,
                )
            logger.info("OpenAI connection test successful")
            logger.info(f"Response: {response.choices[0].message.content}")
        except Exception as e:
            logger.error(f"OpenAI connection test failed: {e}")
            logger.error(f"Exception type: {type(e)}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def _decision_1min_loop(self):
        """Main loop for 1-minute trading decisions"""
        logger.info("Starting 1-minute decision loop")

        while self._running:
            try:
                start_time = time.time()

                # Get active trading symbols
                symbols = list(self.decision_processor.market_contexts.keys())

                decisions_made = 0
                for symbol in symbols:
                    try:
                        # Check if we need a new decision for this symbol
                        if self._should_make_1min_decision(symbol):
                            result = await self._make_1min_decision(symbol)

                            if result.success and result.decision:
                                await self.redis_pipeline.publish_ai_decision(
                                    result.decision
                                )
                                self._track_decision(symbol, result.decision)
                                decisions_made += 1

                                # Update cost tracking
                                self._update_cost_tracking(result.api_cost)

                                logger.info(
                                    f"1-min decision for {symbol}: {result.decision.action} "
                                    f"(confidence: {result.decision.confidence:.2f})"
                                )
                            else:
                                logger.warning(
                                    f"Failed 1-min decision for {symbol}: {result.error_message}"
                                )

                        # Small delay between symbols to avoid rate limits
                        await asyncio.sleep(0.5)

                    except Exception as e:
                        logger.error(f"Error in 1-min decision for {symbol}: {e}")

                loop_time = time.time() - start_time
                logger.debug(
                    f"1-min decision loop completed: {decisions_made} decisions in {loop_time:.1f}s"
                )

                # Wait for next interval
                sleep_time = max(0, self.config.decision_1min_interval - loop_time)
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in 1-min decision loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying

    async def _strategy_1hour_loop(self):
        """Main loop for 1-hour strategy planning"""
        logger.info("Starting 1-hour strategy loop")

        while self._running:
            try:
                start_time = time.time()

                # Get all trading symbols for strategy planning
                symbols = list(self.decision_processor.market_contexts.keys())

                if symbols:
                    result = await self._make_1hour_strategy(symbols)

                    if result.success and result.decision:
                        await self.redis_pipeline.publish_ai_decision(result.decision)
                        self.last_1hour_strategy = result.decision
                        self._update_cost_tracking(result.api_cost)

                        logger.info(
                            f"1-hour strategy update: {result.decision.action} "
                            f"(confidence: {result.decision.confidence:.2f})"
                        )
                    else:
                        logger.warning(
                            f"Failed 1-hour strategy: {result.error_message}"
                        )

                loop_time = time.time() - start_time
                logger.debug(f"1-hour strategy loop completed in {loop_time:.1f}s")

                # Wait for next interval
                sleep_time = max(0, self.config.strategy_1hour_interval - loop_time)
                await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in 1-hour strategy loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    def _should_make_1min_decision(self, symbol: str) -> bool:
        """Check if we should make a new 1-minute decision for symbol"""
        # Check daily cost limit
        if self.daily_api_cost >= self.config.max_daily_api_cost:
            return False

        # Check if context is ready
        if not self.decision_processor.is_context_ready_for_decision(symbol):
            return False

        # Check if we made a recent decision
        if symbol in self.last_1min_decisions:
            last_time, _ = self.last_1min_decisions[symbol]
            time_since_last = (datetime.now() - last_time).total_seconds()
            if time_since_last < self.config.decision_1min_interval:
                return False

        return True

    async def _make_1min_decision(self, symbol: str) -> AIDecisionResult:
        """Make a 1-minute trading decision for symbol"""
        start_time = time.time()

        try:
            # Prepare context
            context_data = self.decision_processor.prepare_1min_decision_context(symbol)
            if not context_data:
                return AIDecisionResult(
                    success=False,
                    decision=None,
                    error_message="No context data available",
                    api_cost=0.0,
                    response_time=0.0,
                    tokens_used=0,
                )

            # Build prompt
            prompt = self._build_1min_prompt(context_data)

            # Call OpenAI (use max_completion_tokens for o3 models)
            if self.config.model_name.startswith("o3"):
                response = await self.openai_client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert cryptocurrency trader.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_completion_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )
            else:
                response = await self.openai_client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert cryptocurrency trader.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                )

            # Parse response
            decision = self._parse_ai_response(
                response.choices[0].message.content, "1min_trade", symbol
            )

            if not decision:
                return AIDecisionResult(
                    success=False,
                    decision=None,
                    error_message="Failed to parse AI response",
                    api_cost=self._calculate_api_cost(response.usage),
                    response_time=time.time() - start_time,
                    tokens_used=response.usage.total_tokens,
                )

            return AIDecisionResult(
                success=True,
                decision=decision,
                error_message=None,
                api_cost=self._calculate_api_cost(response.usage),
                response_time=time.time() - start_time,
                tokens_used=response.usage.total_tokens,
            )

        except Exception as e:
            return AIDecisionResult(
                success=False,
                decision=None,
                error_message=str(e),
                api_cost=0.0,
                response_time=time.time() - start_time,
                tokens_used=0,
            )

    async def _make_1hour_strategy(self, symbols: list[str]) -> AIDecisionResult:
        """Make a 1-hour strategy planning decision"""
        start_time = time.time()

        try:
            # Prepare context
            context_data = self.decision_processor.prepare_1hour_strategy_context(
                symbols
            )
            if not context_data:
                return AIDecisionResult(
                    success=False,
                    decision=None,
                    error_message="No strategy context data available",
                    api_cost=0.0,
                    response_time=0.0,
                    tokens_used=0,
                )

            # Build prompt
            prompt = self._build_1hour_prompt(context_data)

            # Call OpenAI (use max_completion_tokens for o3 models)
            if self.config.model_name.startswith("o3"):
                response = await self.openai_client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a senior cryptocurrency fund manager.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_completion_tokens=self.config.max_tokens
                    * 2,  # More tokens for strategy
                    temperature=self.config.temperature,
                )
            else:
                response = await self.openai_client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a senior cryptocurrency fund manager.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=self.config.max_tokens * 2,  # More tokens for strategy
                    temperature=self.config.temperature,
                )

            # Parse response
            decision = self._parse_ai_response(
                response.choices[0].message.content, "1hour_strategy", "PORTFOLIO"
            )

            if not decision:
                return AIDecisionResult(
                    success=False,
                    decision=None,
                    error_message="Failed to parse strategy response",
                    api_cost=self._calculate_api_cost(response.usage),
                    response_time=time.time() - start_time,
                    tokens_used=response.usage.total_tokens,
                )

            return AIDecisionResult(
                success=True,
                decision=decision,
                error_message=None,
                api_cost=self._calculate_api_cost(response.usage),
                response_time=time.time() - start_time,
                tokens_used=response.usage.total_tokens,
            )

        except Exception as e:
            return AIDecisionResult(
                success=False,
                decision=None,
                error_message=str(e),
                api_cost=0.0,
                response_time=time.time() - start_time,
                tokens_used=0,
            )

    def _build_1min_prompt(self, context_data: dict[str, Any]) -> str:
        """Build prompt for 1-minute trading decision"""
        prompt = TradingPrompts.get_1min_trading_prompt()

        # Format context data
        market_context = TradingPrompts.format_market_context(
            context_data["market_context"]
        )
        price_data = TradingPrompts.format_price_data(context_data["price_data"])
        orderbook_data = TradingPrompts.format_orderbook_data(
            context_data["orderbook_data"]
        )
        sentiment_data = TradingPrompts.format_sentiment_data(
            context_data["sentiment_data"]["news"],
            context_data["sentiment_data"]["x_posts"],
        )

        position_data = f"""
Current Positions: {len(context_data["position_data"]["current_positions"])}
Portfolio Value: ${context_data["position_data"]["portfolio_value"]:,.2f}
Portfolio P&L: ${context_data["position_data"]["portfolio_pnl"]:,.2f}
"""

        # Fill in template
        formatted_prompt = prompt.format(
            market_context=market_context,
            price_data=price_data,
            orderbook_data=orderbook_data,
            sentiment_data=sentiment_data,
            position_data=position_data,
        )

        return formatted_prompt

    def _build_1hour_prompt(self, context_data: dict[str, Any]) -> str:
        """Build prompt for 1-hour strategy planning"""
        prompt = TradingPrompts.get_1hour_strategy_prompt()

        # Format context data for strategy prompt
        market_overview = json.dumps(context_data["market_overview"], indent=2)
        technical_analysis = json.dumps(
            context_data["technical_analysis"], indent=2, default=str
        )
        news_sentiment = json.dumps(
            context_data["news_sentiment"], indent=2, default=str
        )
        portfolio_state = json.dumps(context_data["portfolio_state"], indent=2)

        # Fill in template
        formatted_prompt = prompt.format(
            market_overview=market_overview,
            technical_analysis=technical_analysis,
            fundamental_data="No fundamental data integration yet",  # Future enhancement
            news_sentiment=news_sentiment,
            market_structure="Normal trading conditions",  # Future enhancement
            portfolio_state=portfolio_state,
        )

        return formatted_prompt

    def _parse_ai_response(
        self, response_text: str, decision_type: str, symbol: str
    ) -> AIDecisionMessage | None:
        """Parse AI response into decision message"""
        try:
            # Extract JSON from response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                logger.error("No JSON found in AI response")
                return None

            json_text = response_text[json_start:json_end]
            decision_data = json.loads(json_text)

            # Validate required fields
            required_fields = ["action", "confidence", "reasoning"]
            if not all(field in decision_data for field in required_fields):
                logger.error(f"Missing required fields in AI response: {decision_data}")
                return None

            # Validate confidence
            confidence = float(decision_data["confidence"])
            if confidence < self.config.min_confidence_threshold:
                logger.info(
                    f"AI confidence {confidence:.2f} below threshold {self.config.min_confidence_threshold}"
                )
                # Force to hold if confidence too low
                decision_data["action"] = "hold"

            # Create decision message
            decision = AIDecisionMessage(
                timestamp=datetime.now(),
                source="openai_o3",
                decision_type=decision_type,
                symbol=symbol,
                action=decision_data["action"],
                confidence=confidence,
                reasoning=decision_data["reasoning"],
                target_price=decision_data.get("entry_price"),
                stop_loss=decision_data.get("stop_loss"),
                take_profit=decision_data.get("take_profit"),
                position_size_pct=min(
                    decision_data.get("position_size_pct", 0.01),
                    self.config.max_position_size_pct,
                ),
                metadata={
                    "key_signals": decision_data.get("key_signals", []),
                    "risk_factors": decision_data.get("risk_factors", []),
                    "timeframe_minutes": decision_data.get("timeframe_minutes", 1),
                    "model_name": self.config.model_name,
                },
            )

            return decision

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from AI response: {e}")
            return None
        except Exception as e:
            logger.error(f"Error parsing AI response: {e}")
            return None

    def _calculate_api_cost(self, usage) -> float:
        """Calculate API cost based on token usage"""
        if not usage:
            return 0.0

        model_pricing = self.model_pricing.get(
            self.config.model_name, {"input": 0.001, "output": 0.004}
        )

        input_cost = usage.prompt_tokens * model_pricing["input"]
        output_cost = usage.completion_tokens * model_pricing["output"]

        return input_cost + output_cost

    def _update_cost_tracking(self, cost: float):
        """Update daily cost tracking"""
        now = datetime.now()
        self.api_call_history.append((now, cost))

        # Remove old entries (older than 24 hours)
        cutoff_time = now - timedelta(hours=self.config.cost_tracking_window_hours)
        self.api_call_history = [
            (timestamp, cost)
            for timestamp, cost in self.api_call_history
            if timestamp > cutoff_time
        ]

        # Update daily total
        self.daily_api_cost = sum(cost for _, cost in self.api_call_history)

    def _track_decision(self, symbol: str, decision: AIDecisionMessage):
        """Track decision for performance analysis"""
        self.last_1min_decisions[symbol] = (decision.timestamp, decision)

        # TODO: Implement decision performance tracking
        # This would track how profitable each decision was

    # Public interface methods
    async def update_market_data(self, market_msg):
        """Update market context with new data"""
        await self.decision_processor.update_market_data(market_msg)

    async def update_news_data(self, news_msg):
        """Update context with news data"""
        await self.decision_processor.update_news_data(news_msg)

    async def update_x_posts_data(self, post_msg):
        """Update context with X posts data"""
        await self.decision_processor.update_x_posts_data(post_msg)

    def get_statistics(self) -> dict[str, Any]:
        """Get decision engine statistics"""
        return {
            "running": self._running,
            "daily_api_cost": self.daily_api_cost,
            "cost_limit": self.config.max_daily_api_cost,
            "api_calls_24h": len(self.api_call_history),
            "active_symbols": len(self.last_1min_decisions),
            "last_strategy_time": (
                self.last_1hour_strategy.timestamp.isoformat()
                if self.last_1hour_strategy
                else None
            ),
            "processor_stats": self.decision_processor.get_statistics(),
        }
