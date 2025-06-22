"""
Message types for Redis Streams pipeline
Defines structured data formats for different message types in the trading system
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Dict, Any, Optional, List
from decimal import Decimal
import json


@dataclass
class StreamMessage:
    """Base message for Redis Streams"""
    
    timestamp: datetime
    message_type: str
    source: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis"""
        data = asdict(self)
        # Convert datetime to ISO string
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StreamMessage':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class MarketDataMessage(StreamMessage):
    """Market data stream message"""
    
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    orderbook_imbalance: Optional[float] = None
    funding_rate: Optional[float] = None
    
    def __post_init__(self):
        self.message_type = "market_data"


@dataclass
class NewsMessage(StreamMessage):
    """News/sentiment stream message"""
    
    title: str
    content: str
    url: str
    sentiment_score: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    relevance_score: Optional[float] = None
    
    def __post_init__(self):
        self.message_type = "news"


@dataclass
class XPostMessage(StreamMessage):
    """X/Twitter post stream message"""
    
    post_id: str
    text: str
    author: str
    engagement_score: float
    sentiment_score: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    
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
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    position_size_pct: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.message_type = "ai_decision"


@dataclass
class ExecutionMessage(StreamMessage):
    """Execution result stream message"""
    
    order_id: str
    symbol: str
    side: str
    size: float
    price: Optional[float] = None
    status: str  # "placed", "filled", "failed"
    error_message: Optional[str] = None
    
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
    risk_warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.message_type = "risk_decision"


class MessageEncoder:
    """Helper class to encode/decode messages for Redis"""
    
    @staticmethod
    def encode_message(message: StreamMessage) -> Dict[str, str]:
        """Encode message for Redis (all values must be strings)"""
        data = message.to_dict()
        return {k: json.dumps(v) if not isinstance(v, str) else v for k, v in data.items()}
    
    @staticmethod
    def decode_message(message_type: str, data: Dict[str, str]) -> StreamMessage:
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
            "risk_decision": RiskDecisionMessage
        }
        
        message_class = message_classes.get(message_type, StreamMessage)
        return message_class.from_dict(parsed_data)