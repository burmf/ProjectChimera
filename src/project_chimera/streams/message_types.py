"""
Message types for Redis Streams pipeline
Defines structured data formats for different message types in the trading system
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class StreamMessage:
    """Base message for Redis Streams"""

    timestamp: datetime
    message_type: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for Redis"""
        data = asdict(self)
        # Convert datetime to ISO string
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StreamMessage":
        """Create from dictionary"""
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)


@dataclass
class MarketDataMessage(StreamMessage):
    """Market data stream message"""

    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    orderbook_imbalance: float | None = None
    funding_rate: float | None = None
    message_type: str = field(default="market_data", init=False)

    def __post_init__(self):
        self.message_type = "market_data"


@dataclass
class NewsMessage(StreamMessage):
    """News/sentiment stream message"""

    title: str
    content: str
    url: str
    sentiment_score: float | None = None
    tags: list[str] = field(default_factory=list)
    relevance_score: float | None = None
    message_type: str = field(default="news", init=False)

    def __post_init__(self):
        self.message_type = "news"


@dataclass
class XPostMessage(StreamMessage):
    """X/Twitter post stream message"""

    post_id: str
    text: str
    author: str
    engagement_score: float
    sentiment_score: float | None = None
    tags: list[str] = field(default_factory=list)
    message_type: str = field(default="x_post", init=False)

    def __post_init__(self):
        self.message_type = "x_post"


@dataclass
class AIDecisionMessage(StreamMessage):
    """AI decision stream message"""

    decision_type: str  # "1min_trade" or "1hour_strategy"
    symbol: str
    action: str  # "buy", "sell", "hold"
    confidence: float
    reasoning: str
    target_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    position_size_pct: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    message_type: str = field(default="ai_decision", init=False)

    def __post_init__(self):
        self.message_type = "ai_decision"


@dataclass
class ExecutionMessage(StreamMessage):
    """Execution result stream message"""

    order_id: str
    symbol: str
    side: str
    size: float
    status: str  # "placed", "filled", "failed"
    price: float | None = None
    error_message: str | None = None
    message_type: str = field(default="execution", init=False)

    def __post_init__(self):
        self.message_type = "execution"


@dataclass
class RiskDecisionMessage(StreamMessage):
    """Risk management decision message"""

    original_signal_id: str
    symbol: str
    risk_adjusted_size: float
    risk_multiplier: float
    approval_status: str  # "approved", "rejected", "modified"
    risk_warnings: list[str] = field(default_factory=list)
    message_type: str = field(default="risk_decision", init=False)

    def __post_init__(self):
        self.message_type = "risk_decision"


class MessageEncoder:
    """Helper class to encode/decode messages for Redis"""

    @staticmethod
    def encode_message(message: StreamMessage) -> dict[str, str]:
        """Encode message for Redis (all values must be strings)"""
        data = message.to_dict()
        return {
            k: json.dumps(v) if not isinstance(v, str) else v for k, v in data.items()
        }

    @staticmethod
    def decode_message(message_type: str, data: dict[str, str]) -> StreamMessage:
        """Decode message from Redis"""
        # Parse JSON values back
        parsed_data = {}
        for k, v in data.items():
            try:
                parsed_data[k] = json.loads(v)
            except (json.JSONDecodeError, TypeError):
                parsed_data[k] = v

        # Create appropriate message type
        message_classes = {
            "market_data": MarketDataMessage,
            "news": NewsMessage,
            "x_post": XPostMessage,
            "ai_decision": AIDecisionMessage,
            "execution": ExecutionMessage,
            "risk_decision": RiskDecisionMessage,
        }

        message_class = message_classes.get(message_type, StreamMessage)
        return message_class.from_dict(parsed_data)
