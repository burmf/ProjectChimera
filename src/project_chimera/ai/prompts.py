"""
Trading prompts for OpenAI o3 model
Specialized prompts for 1-minute trading decisions and 1-hour strategy planning
"""

from typing import Dict, Any
from datetime import datetime


class TradingPrompts:
    """
    Collection of prompts for AI trading decisions
    Optimized for OpenAI o3 model capabilities
    """
    
    @staticmethod
    def get_1min_trading_prompt() -> str:
        """
        Prompt for 1-minute trading decisions
        Focus on immediate market action based on price/orderbook/sentiment
        """
        return """You are an expert cryptocurrency scalping trader with 10+ years of experience. 
Your task is to make ultra-short-term trading decisions (1-3 minute holds) based on real-time market data.

## Your Expertise:
- Scalping and momentum trading
- Order book analysis and flow interpretation  
- Sentiment-driven price movements
- Risk management and position sizing
- Multi-timeframe analysis

## Current Market Context:
{market_context}

## Recent Price Action:
{price_data}

## Order Book Analysis:
{orderbook_data}

## Recent News/Sentiment:
{sentiment_data}

## Current Positions:
{position_data}

## Decision Framework:
Analyze the data and provide a trading decision in this EXACT JSON format:

{{
    "action": "buy|sell|hold",
    "confidence": 0.0-1.0,
    "reasoning": "2-3 sentence explanation focusing on key signals",
    "entry_price": number or null,
    "stop_loss": number or null,
    "take_profit": number or null,
    "position_size_pct": 0.0-0.05,
    "timeframe_minutes": 1-5,
    "key_signals": ["signal1", "signal2", "signal3"],
    "risk_factors": ["risk1", "risk2"]
}}

## Decision Criteria:
1. **BUY signals**: Strong bid support, positive sentiment spike, breakout above resistance, momentum acceleration
2. **SELL signals**: Weak bid support, negative sentiment, breakdown below support, momentum reversal
3. **HOLD signals**: Unclear signals, high volatility, conflicting data, insufficient confidence

## Risk Management:
- Max position size: 5% of portfolio
- Always set stop losses (1-2% from entry)
- Take profits at 0.5-1.5% for scalps
- Never risk more than 0.1% of portfolio per trade

Focus on HIGH PROBABILITY setups only. When in doubt, choose HOLD.

Provide your analysis:"""

    @staticmethod
    def get_1hour_strategy_prompt() -> str:
        """
        Prompt for 1-hour strategy planning
        Focus on medium-term market direction and positioning
        """
        return """You are a senior cryptocurrency fund manager developing hourly trading strategies. 
Your task is to analyze market conditions and set strategic direction for the next 1-4 hours.

## Your Expertise:
- Macro market analysis and trend identification
- Multi-asset correlation analysis
- News impact assessment and market psychology
- Strategic positioning and portfolio management
- Risk regime identification

## Market Overview:
{market_overview}

## Technical Analysis (1H/4H):
{technical_analysis}

## Fundamental Factors:
{fundamental_data}

## News & Sentiment Analysis:
{news_sentiment}

## Market Structure:
{market_structure}

## Current Portfolio State:
{portfolio_state}

## Strategy Framework:
Provide strategic guidance in this EXACT JSON format:

{{
    "market_bias": "bullish|bearish|neutral",
    "confidence": 0.0-1.0,
    "time_horizon_hours": 1-4,
    "key_themes": ["theme1", "theme2", "theme3"],
    "target_pairs": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    "position_sizing": {{
        "aggressive": 0.0-0.15,
        "moderate": 0.0-0.10, 
        "conservative": 0.0-0.05
    }},
    "risk_level": "low|medium|high",
    "stop_loss_strategy": "tight|normal|wide",
    "profit_targets": {{
        "short_term": 0.005-0.02,
        "medium_term": 0.01-0.05
    }},
    "market_regime": "trending|ranging|volatile|uncertain",
    "key_levels": {{
        "support": [number, number],
        "resistance": [number, number]
    }},
    "execution_notes": "Strategic guidance for trade execution",
    "risk_warnings": ["warning1", "warning2"]
}}

## Strategy Focus Areas:
1. **Market Regime**: Identify current market phase (trending/ranging/breaking out)
2. **Macro Context**: Consider broader crypto market sentiment and correlations
3. **News Impact**: Assess how recent news may affect price action
4. **Technical Setup**: Identify key support/resistance and momentum signals
5. **Risk Management**: Adapt position sizing to current volatility regime

## Risk Assessment:
- High volatility = reduce position sizes
- Unclear market structure = defensive positioning  
- Strong trend = moderate aggression acceptable
- News-driven markets = extra caution

Provide comprehensive strategic analysis:"""

    @staticmethod
    def get_risk_assessment_prompt() -> str:
        """
        Prompt for risk assessment and position sizing
        """
        return """You are a risk management specialist for a cryptocurrency trading fund.
Analyze current market conditions and provide risk-adjusted position sizing recommendations.

## Current Portfolio:
{portfolio_data}

## Market Volatility:
{volatility_data}

## Correlation Analysis:
{correlation_data}

## Recent Performance:
{performance_data}

## Risk Assessment Framework:
Provide risk analysis in this EXACT JSON format:

{{
    "overall_risk_level": "low|medium|high|extreme",
    "portfolio_heat": 0.0-1.0,
    "max_position_size": 0.0-0.20,
    "recommended_leverage": 1.0-10.0,
    "volatility_regime": "low|normal|high|extreme",
    "correlation_risk": 0.0-1.0,
    "liquidity_conditions": "excellent|good|poor|concerning",
    "market_stress_indicators": ["indicator1", "indicator2"],
    "position_adjustments": {{
        "reduce_by": 0.0-0.50,
        "affected_pairs": ["pair1", "pair2"]
    }},
    "risk_warnings": ["warning1", "warning2"],
    "recommended_stops": {{
        "tight": 0.005-0.01,
        "normal": 0.01-0.02,
        "wide": 0.02-0.05
    }}
}}

Focus on portfolio preservation while maintaining profit potential."""

    @staticmethod
    def format_market_context(market_data: Dict[str, Any]) -> str:
        """Format market context for prompt"""
        return f"""
Symbol: {market_data.get('symbol', 'Unknown')}
Current Price: ${market_data.get('price', 0):.2f}
24h Change: {market_data.get('change_24h', 0):.2f}%
24h Volume: ${market_data.get('volume', 0):,.0f}
Bid-Ask Spread: {market_data.get('spread', 0):.4f}
Order Book Imbalance: {market_data.get('imbalance', 0):.3f}
Funding Rate: {market_data.get('funding_rate', 0):.6f}%
Timestamp: {market_data.get('timestamp', datetime.now())}
"""

    @staticmethod
    def format_price_data(price_history: list) -> str:
        """Format recent price data for prompt"""
        if not price_history:
            return "No recent price data available"
        
        output = "Recent 1-minute candles:\n"
        for i, candle in enumerate(price_history[-10:]):  # Last 10 candles
            output += f"  {i+1}. {candle.get('timestamp', '')}: O:{candle.get('open', 0):.2f} "
            output += f"H:{candle.get('high', 0):.2f} L:{candle.get('low', 0):.2f} "
            output += f"C:{candle.get('close', 0):.2f} V:{candle.get('volume', 0):.0f}\n"
        
        return output

    @staticmethod
    def format_orderbook_data(orderbook: Dict[str, Any]) -> str:
        """Format order book data for prompt"""
        if not orderbook:
            return "No order book data available"
        
        output = "Order Book Snapshot:\n"
        
        # Top 5 asks
        asks = orderbook.get('asks', [])[:5]
        output += "Asks (sell orders):\n"
        for price, qty in asks:
            output += f"  ${price:.2f} - {qty:.4f}\n"
        
        # Current spread
        best_bid = orderbook.get('best_bid', 0)
        best_ask = orderbook.get('best_ask', 0)
        spread = best_ask - best_bid if best_ask and best_bid else 0
        output += f"Spread: ${spread:.4f}\n"
        
        # Top 5 bids  
        bids = orderbook.get('bids', [])[:5]
        output += "Bids (buy orders):\n"
        for price, qty in bids:
            output += f"  ${price:.2f} - {qty:.4f}\n"
        
        return output

    @staticmethod  
    def format_sentiment_data(news_items: list, x_posts: list) -> str:
        """Format recent news and X posts for prompt"""
        output = "Recent News & Sentiment:\n\n"
        
        # Recent news
        output += "News Headlines:\n"
        for item in news_items[-5:]:  # Last 5 news items
            title = item.get('title', '')[:100]
            relevance = item.get('relevance_score', 0)
            output += f"  • {title}... (relevance: {relevance:.2f})\n"
        
        output += "\nSocial Sentiment (X/Twitter):\n"
        # Recent X posts
        for post in x_posts[-5:]:  # Last 5 posts
            text = post.get('text', '')[:80]
            sentiment = post.get('sentiment_score', 0)
            engagement = post.get('engagement_score', 0)
            output += f"  • {text}... (sentiment: {sentiment:.2f}, engagement: {engagement:.2f})\n"
        
        return output