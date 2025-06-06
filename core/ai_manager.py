#!/usr/bin/env python3
"""
AI Manager for Trading Bot
Handles OpenAI API calls with cost tracking and parallel execution
"""

import asyncio
import json
import datetime
import os
import logging
from typing import Dict, List, Optional, Any, Union
from openai import OpenAI

DEFAULT_AI_SYSTEM_PROMPT_FOR_UI = """You are an expert financial analyst. Analyze the provided news article and respond ONLY in JSON format.
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

# o3 optimized prompt for enhanced reasoning
O3_SYSTEM_PROMPT = """You are an expert forex analyst with deep understanding of market psychology and fundamental analysis. 

TASK: Analyze the provided news article through systematic reasoning to determine trade opportunities.

REASONING FRAMEWORK:
1. NEWS IMPACT ASSESSMENT: Evaluate the news significance, credibility, and market impact potential
2. MARKET CONTEXT: Consider current economic environment, central bank policies, and technical levels  
3. RISK-REWARD ANALYSIS: Calculate probability-weighted outcomes and position sizing
4. TRADE EXECUTION: Define precise entry, stop-loss, and take-profit levels

OUTPUT FORMAT (JSON only):
{
  "trade_warranted": boolean,
  "pair": string,
  "direction": string,
  "confidence": float (0.0-1.0),
  "reasoning": string (detailed step-by-step analysis, max 200 words),
  "stop_loss_pips": integer,
  "take_profit_pips": integer,
  "suggested_lot_size_factor": float (0.0-1.0),
  "market_impact_score": float (0.0-1.0),
  "time_horizon_hours": integer
}

Think step-by-step before concluding."""

MODEL_PRICING_USD_PER_1M_TOKENS = {
    "o3": {"input": 10.00, "output": 40.00},
    "o3-mini": {"input": 1.10, "output": 4.40},
    "o4-mini": {"input": 4.00, "output": 8.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-4o": {"input": 5.00, "output": 15.00},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-4-32k": {"input": 60.00, "output": 120.00}
}

def estimate_openai_cost(model_name, prompt_tokens, completion_tokens):
  pricing = MODEL_PRICING_USD_PER_1M_TOKENS.get(model_name)
  if not pricing:
    return 0.0
  return (prompt_tokens / 1_000_000) * pricing["input"] + (completion_tokens / 1_000_000) * pricing["output"]

async def _query_single_openai_model_async(client, model, system_prompt, user_prompt, news_id=None, purpose="manual_analysis"):
  try:
    response = await asyncio.to_thread(
      client.chat.completions.create,
      model=model,
      messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt}
      ]
    )
    content = response.choices[0].message.content
    data = json.loads(content)
    usage = response.usage
    log = None
    if usage:
      pt, ct, tt = usage.prompt_tokens, usage.completion_tokens, usage.total_tokens
      cost = estimate_openai_cost(model, pt, ct)
      log = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model_name": model,
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "total_tokens": tt,
        "estimated_cost_usd": cost,
        "related_news_id": news_id,
        "purpose": purpose
      }
    return model, data, log
  except Exception as e:
    log = {
      "timestamp": datetime.datetime.now().isoformat(),
      "model_name": model,
      "error": str(e),
      "related_news_id": news_id,
      "purpose": purpose
    }
    return model, {"error": str(e)}, log

def get_optimal_prompt_for_model(model_name):
  """Select optimal prompt based on model capabilities"""
  if model_name in ["o3", "o3-mini"]:
    return O3_SYSTEM_PROMPT
  return DEFAULT_AI_SYSTEM_PROMPT_FOR_UI

class AIManager:
    """
    Manages OpenAI API calls with cost tracking, parallel execution, and structured responses.
    Supports o3 and other OpenAI models.
    """
    
    def __init__(self, api_key: str = None):
        self.logger = logging.getLogger(__name__)
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        
        if not self.api_key:
            self.logger.warning("No OpenAI API key provided")
        
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.default_model = os.getenv('OPENAI_MODEL', 'gpt-4o')
        
    def get_trading_decision(self, news_context: str, price_context: str = None, model: str = None) -> Optional[Dict]:
        """
        Get AI trading decision with structured JSON response.
        """
        if not self.client:
            self.logger.error("OpenAI client not initialized")
            return None
            
        if model is None:
            model = self.default_model
            
        # Use model-specific prompt
        system_prompt = get_optimal_prompt_for_model(model)
        
        user_prompt = f"Analyze the following news for trading opportunities:\n\n{news_context}"
        if price_context:
            user_prompt += f"\n\nPrice/Technical Context:\n{price_context}"
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            usage = response.usage
            
            # Parse JSON response
            try:
                decision = json.loads(content)
                
                # Add metadata
                if usage:
                    decision['tokens_used'] = usage.total_tokens
                    decision['cost_usd'] = estimate_openai_cost(
                        model, usage.prompt_tokens, usage.completion_tokens
                    )
                
                decision['model_used'] = model
                decision['timestamp'] = datetime.datetime.now().isoformat()
                
                return decision
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse JSON from AI response: {e}")
                self.logger.debug(f"Raw response: {content}")
                
        except Exception as e:
            self.logger.error(f"AI request failed: {e}")
            
        return None
    
    def analyze_news_sentiment(self, news_title: str, news_content: str = None, model: str = None) -> Optional[Dict]:
        """
        Analyze news sentiment for forex impact.
        """
        if not self.client:
            return None
            
        if model is None:
            model = 'gpt-4o-mini'  # Use cheaper model for sentiment
            
        system_prompt = """
        Analyze the news for forex market sentiment, specifically for USD/JPY.
        
        Respond ONLY with valid JSON:
        {
            "sentiment": -1.0 to 1.0 (negative to positive for USD/JPY),
            "impact_level": "low|medium|high",
            "key_factors": ["factor1", "factor2"],
            "confidence": 0.0-1.0
        }
        """
        
        content_text = f"Title: {news_title}"
        if news_content:
            content_text += f"\nContent: {news_content[:500]}"
        
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content_text}
                ],
                temperature=0.3
            )
            
            content = response.choices[0].message.content
            sentiment_data = json.loads(content)
            
            # Add metadata
            usage = response.usage
            if usage:
                sentiment_data['tokens_used'] = usage.total_tokens
                sentiment_data['cost_usd'] = estimate_openai_cost(
                    model, usage.prompt_tokens, usage.completion_tokens
                )
            
            sentiment_data['model_used'] = model
            sentiment_data['timestamp'] = datetime.datetime.now().isoformat()
            
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            
        return None
    
    def health_check(self) -> Dict[str, Any]:
        """Perform a health check of the AI service."""
        if not self.client:
            return {
                'status': 'error',
                'message': 'No API key configured',
                'available': False
            }
        
        try:
            # Test with a simple request
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=[{"role": "user", "content": "Say 'OK' if you can respond."}],
                max_tokens=10
            )
            
            if response.choices[0].message.content:
                return {
                    'status': 'healthy',
                    'message': 'AI service is operational',
                    'available': True,
                    'default_model': self.default_model
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Empty response from API',
                    'available': False
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Health check failed: {str(e)}",
                'available': False
            }


async def run_openai_parallel_inference_async(api_key, model_list, news_text, system_prompt_override=None, news_id=None, purpose="manual_analysis"):
  """Legacy function for backward compatibility."""
  if not api_key or api_key.startswith("YOUR_"):
    return {}, []
  if not model_list:
    return {}, []
  client = OpenAI(api_key=api_key)
  tasks = []
  
  for model in model_list:
    # Use model-specific prompt if no override provided
    if system_prompt_override:
      system_prompt = system_prompt_override.strip()
    else:
      system_prompt = get_optimal_prompt_for_model(model)
    
    user_prompt = f"Analyze the following news article and provide a trade plan in JSON format based on its content:\n\n---\n{news_text}\n---"
    
    tasks.append(
      _query_single_openai_model_async(client, model, system_prompt, user_prompt, news_id, purpose)
    )
  results = await asyncio.gather(*tasks, return_exceptions=True)
  final, logs = {}, []
  for i, res in enumerate(results):
    model_name = model_list[i]
    if isinstance(res, Exception):
      final[model_name] = {"error": str(res)}
    else:
      m, data, log = res
      final[m] = data
      if log:
        logs.append(log)
  return final, logs