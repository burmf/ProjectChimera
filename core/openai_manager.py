# core/openai_manager.py
import asyncio
import json
import datetime
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import uuid
import sys
import os
from openai import OpenAI, AsyncOpenAI
from openai.types import CompletionUsage

# Add core modules to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.redis_manager import redis_manager
from core.database_adapter import db_adapter
from core.logging_config import get_trading_logger

logger = logging.getLogger(__name__)
trading_logger = get_trading_logger("openai_manager")

class Priority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class RequestStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    BUDGET_EXCEEDED = "budget_exceeded"

@dataclass
class OpenAIRequest:
    id: str
    model: str
    messages: List[Dict[str, str]]
    priority: Priority
    max_tokens: Optional[int]
    temperature: float
    timeout: int
    created_at: datetime.datetime
    news_id: Optional[str] = None
    purpose: str = "analysis"
    retry_count: int = 0
    max_retries: int = 3
    status: RequestStatus = RequestStatus.PENDING

@dataclass
class RateLimitConfig:
    requests_per_minute: int
    tokens_per_minute: int
    requests_per_day: int
    tokens_per_day: int
    concurrent_requests: int

@dataclass
class BudgetConfig:
    daily_budget_usd: float
    monthly_budget_usd: float
    per_request_limit_usd: float
    alert_threshold_pct: float = 80.0

class OpenAIManager:
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.client = None
        self.async_client = None
        
        # Rate limiting configuration per model tier
        self.rate_limits = {
            'o3': RateLimitConfig(
                requests_per_minute=100,
                tokens_per_minute=20000,
                requests_per_day=5000,
                tokens_per_day=500000,
                concurrent_requests=5
            ),
            'o3-mini': RateLimitConfig(
                requests_per_minute=500,
                tokens_per_minute=40000,
                requests_per_day=15000,
                tokens_per_day=1500000,
                concurrent_requests=10
            ),
            'gpt-4o': RateLimitConfig(
                requests_per_minute=500,
                tokens_per_minute=30000,
                requests_per_day=10000,
                tokens_per_day=1000000,
                concurrent_requests=10
            ),
            'gpt-4-turbo': RateLimitConfig(
                requests_per_minute=500,
                tokens_per_minute=30000,
                requests_per_day=10000,
                tokens_per_day=1000000,
                concurrent_requests=8
            ),
            'gpt-3.5-turbo': RateLimitConfig(
                requests_per_minute=3500,
                tokens_per_minute=90000,
                requests_per_day=10000,
                tokens_per_day=2000000,
                concurrent_requests=15
            ),
            'o4-mini': RateLimitConfig(
                requests_per_minute=1000,
                tokens_per_minute=50000,
                requests_per_day=20000,
                tokens_per_day=3000000,
                concurrent_requests=20
            )
        }
        
        # Budget configuration
        self.budget_config = BudgetConfig(
            daily_budget_usd=20.0,
            monthly_budget_usd=500.0,
            per_request_limit_usd=5.0,
            alert_threshold_pct=80.0
        )
        
        # Pricing per 1M tokens (input/output)
        self.pricing = {
            "o3": {"input": 10.00, "output": 40.00},
            "o3-mini": {"input": 1.10, "output": 4.40},
            "o4-mini": {"input": 4.00, "output": 8.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
            "gpt-4o": {"input": 5.00, "output": 15.00},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00},
            "gpt-4": {"input": 30.00, "output": 60.00},
            "gpt-4-32k": {"input": 60.00, "output": 120.00}
        }
        
        # Request queues by priority
        self.request_queues = {
            Priority.CRITICAL: asyncio.Queue(),
            Priority.HIGH: asyncio.Queue(),
            Priority.NORMAL: asyncio.Queue(),
            Priority.LOW: asyncio.Queue()
        }
        
        # Rate limiting state
        self.usage_tracking = {}
        self.semaphores = {}
        
        # Active workers
        self.workers = []
        self.running = False
        
        self.initialize_clients()
        
    def initialize_clients(self):
        """Initialize OpenAI clients"""
        if not self.api_key or self.api_key.startswith('YOUR_'):
            logger.warning("OpenAI API key not configured")
            return
            
        try:
            self.client = OpenAI(api_key=self.api_key)
            self.async_client = AsyncOpenAI(api_key=self.api_key)
            
            # Initialize semaphores for concurrent request limiting
            for model, config in self.rate_limits.items():
                self.semaphores[model] = asyncio.Semaphore(config.concurrent_requests)
                
            logger.info("OpenAI clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI clients: {e}")
    
    def estimate_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost for API usage"""
        pricing = self.pricing.get(model)
        if not pricing:
            return 0.0
        
        input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars = 1 token)"""
        return len(text) // 4
    
    async def check_rate_limits(self, model: str, estimated_tokens: int) -> Tuple[bool, str]:
        """Check if request would exceed rate limits"""
        try:
            config = self.rate_limits.get(model)
            if not config:
                return True, "Model not configured"
            
            current_time = datetime.datetime.now()
            minute_key = f"rate_limit:{model}:{current_time.strftime('%Y-%m-%d:%H:%M')}"
            day_key = f"rate_limit:{model}:{current_time.strftime('%Y-%m-%d')}"
            
            # Get current usage
            minute_requests = int(redis_manager.get_cache(f"{minute_key}:requests") or 0)
            minute_tokens = int(redis_manager.get_cache(f"{minute_key}:tokens") or 0)
            day_requests = int(redis_manager.get_cache(f"{day_key}:requests") or 0)
            day_tokens = int(redis_manager.get_cache(f"{day_key}:tokens") or 0)
            
            # Check limits
            if minute_requests >= config.requests_per_minute:
                return False, "Minute request limit exceeded"
            
            if minute_tokens + estimated_tokens > config.tokens_per_minute:
                return False, "Minute token limit exceeded"
            
            if day_requests >= config.requests_per_day:
                return False, "Daily request limit exceeded"
            
            if day_tokens + estimated_tokens > config.tokens_per_day:
                return False, "Daily token limit exceeded"
            
            return True, "OK"
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            return True, "Rate limit check failed"
    
    async def update_usage_tracking(self, model: str, prompt_tokens: int, completion_tokens: int):
        """Update usage tracking in Redis"""
        try:
            current_time = datetime.datetime.now()
            minute_key = f"rate_limit:{model}:{current_time.strftime('%Y-%m-%d:%H:%M')}"
            day_key = f"rate_limit:{model}:{current_time.strftime('%Y-%m-%d')}"
            
            # Update minute counters
            redis_manager.client.incr(f"{minute_key}:requests")
            redis_manager.client.incrby(f"{minute_key}:tokens", prompt_tokens + completion_tokens)
            redis_manager.client.expire(f"{minute_key}:requests", 120)  # 2 minutes TTL
            redis_manager.client.expire(f"{minute_key}:tokens", 120)
            
            # Update daily counters
            redis_manager.client.incr(f"{day_key}:requests")
            redis_manager.client.incrby(f"{day_key}:tokens", prompt_tokens + completion_tokens)
            redis_manager.client.expire(f"{day_key}:requests", 86400)  # 24 hours TTL
            redis_manager.client.expire(f"{day_key}:tokens", 86400)
            
        except Exception as e:
            logger.error(f"Failed to update usage tracking: {e}")
    
    async def check_budget_limits(self, estimated_cost: float) -> Tuple[bool, str]:
        """Check if request would exceed budget limits"""
        try:
            if estimated_cost > self.budget_config.per_request_limit_usd:
                return False, f"Request cost ${estimated_cost:.4f} exceeds per-request limit"
            
            current_date = datetime.date.today()
            daily_key = f"budget:daily:{current_date}"
            monthly_key = f"budget:monthly:{current_date.strftime('%Y-%m')}"
            
            daily_spent = float(redis_manager.get_cache(daily_key) or 0)
            monthly_spent = float(redis_manager.get_cache(monthly_key) or 0)
            
            if daily_spent + estimated_cost > self.budget_config.daily_budget_usd:
                return False, f"Daily budget exceeded: ${daily_spent:.4f} + ${estimated_cost:.4f} > ${self.budget_config.daily_budget_usd}"
            
            if monthly_spent + estimated_cost > self.budget_config.monthly_budget_usd:
                return False, f"Monthly budget exceeded: ${monthly_spent:.4f} + ${estimated_cost:.4f} > ${self.budget_config.monthly_budget_usd}"
            
            # Check alert thresholds
            daily_pct = (daily_spent + estimated_cost) / self.budget_config.daily_budget_usd * 100
            if daily_pct > self.budget_config.alert_threshold_pct:
                trading_logger.log_risk_event({
                    'type': 'budget_warning',
                    'severity': 'warning',
                    'message': f"Daily budget at {daily_pct:.1f}% of limit"
                })
            
            return True, "OK"
            
        except Exception as e:
            logger.error(f"Budget check failed: {e}")
            return True, "Budget check failed"
    
    async def update_budget_tracking(self, actual_cost: float):
        """Update budget tracking"""
        try:
            current_date = datetime.date.today()
            daily_key = f"budget:daily:{current_date}"
            monthly_key = f"budget:monthly:{current_date.strftime('%Y-%m')}"
            
            # Update counters
            current_daily = float(redis_manager.get_cache(daily_key) or 0)
            current_monthly = float(redis_manager.get_cache(monthly_key) or 0)
            
            redis_manager.set_cache(daily_key, current_daily + actual_cost, ttl=86400)
            redis_manager.set_cache(monthly_key, current_monthly + actual_cost, ttl=2592000)  # 30 days
            
        except Exception as e:
            logger.error(f"Failed to update budget tracking: {e}")
    
    async def queue_request(self, 
                          model: str,
                          messages: List[Dict[str, str]],
                          priority: Priority = Priority.NORMAL,
                          max_tokens: Optional[int] = None,
                          temperature: float = 0.7,
                          timeout: int = 60,
                          news_id: Optional[str] = None,
                          purpose: str = "analysis") -> str:
        """Queue an OpenAI API request"""
        
        request = OpenAIRequest(
            id=str(uuid.uuid4()),
            model=model,
            messages=messages,
            priority=priority,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            created_at=datetime.datetime.now(),
            news_id=news_id,
            purpose=purpose
        )
        
        # Estimate tokens and cost
        total_text = " ".join([msg['content'] for msg in messages])
        estimated_tokens = self.estimate_tokens(total_text)
        estimated_cost = self.estimate_cost(model, estimated_tokens, max_tokens or 150)
        
        # Check rate limits and budget
        rate_ok, rate_msg = await self.check_rate_limits(model, estimated_tokens)
        if not rate_ok:
            request.status = RequestStatus.RATE_LIMITED
            logger.warning(f"Request {request.id} rate limited: {rate_msg}")
            
        budget_ok, budget_msg = await self.check_budget_limits(estimated_cost)
        if not budget_ok:
            request.status = RequestStatus.BUDGET_EXCEEDED
            logger.warning(f"Request {request.id} budget exceeded: {budget_msg}")
        
        # Queue the request
        await self.request_queues[priority].put(request)
        
        # Store request in Redis for tracking
        request_data = {
            'id': request.id,
            'model': model,
            'priority': priority.name,
            'status': request.status.value,
            'created_at': request.created_at.isoformat(),
            'estimated_cost': estimated_cost,
            'estimated_tokens': estimated_tokens,
            'news_id': news_id,
            'purpose': purpose
        }
        
        redis_manager.set_cache(f"openai_request:{request.id}", request_data, ttl=3600)
        
        logger.info(f"Queued OpenAI request {request.id} for model {model} (priority: {priority.name})")
        
        return request.id
    
    async def process_request(self, request: OpenAIRequest) -> Dict[str, Any]:
        """Process a single OpenAI request with comprehensive error handling"""
        
        request.status = RequestStatus.PROCESSING
        
        # Update request status in Redis
        redis_manager.set_cache(f"openai_request:{request.id}", {
            'status': request.status.value,
            'processing_started': datetime.datetime.now().isoformat()
        }, ttl=3600)
        
        try:
            # Acquire semaphore for concurrent request limiting
            async with self.semaphores.get(request.model, asyncio.Semaphore(5)):
                
                # Make API call with timeout
                start_time = time.time()
                
                response = await asyncio.wait_for(
                    self.async_client.chat.completions.create(
                        model=request.model,
                        messages=request.messages,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature
                    ),
                    timeout=request.timeout
                )
                
                end_time = time.time()
                
                # Extract response data
                content = response.choices[0].message.content
                usage = response.usage
                
                # Parse JSON response
                try:
                    parsed_data = json.loads(content)
                except json.JSONDecodeError as e:
                    # Attempt to fix common JSON issues
                    content_fixed = self.fix_json_response(content)
                    try:
                        parsed_data = json.loads(content_fixed)
                    except:
                        logger.error(f"Failed to parse JSON response: {content}")
                        raise ValueError(f"Invalid JSON response: {str(e)}")
                
                # Calculate actual cost
                actual_cost = self.estimate_cost(
                    request.model, 
                    usage.prompt_tokens, 
                    usage.completion_tokens
                )
                
                # Update tracking
                await self.update_usage_tracking(
                    request.model,
                    usage.prompt_tokens,
                    usage.completion_tokens
                )
                await self.update_budget_tracking(actual_cost)
                
                # Store usage in database
                await self.store_usage_record(request, usage, actual_cost, end_time - start_time)
                
                request.status = RequestStatus.COMPLETED
                
                result = {
                    'request_id': request.id,
                    'model': request.model,
                    'data': parsed_data,
                    'usage': {
                        'prompt_tokens': usage.prompt_tokens,
                        'completion_tokens': usage.completion_tokens,
                        'total_tokens': usage.total_tokens,
                        'cost_usd': actual_cost
                    },
                    'processing_time': end_time - start_time,
                    'status': 'success'
                }
                
                logger.info(f"Request {request.id} completed successfully (cost: ${actual_cost:.4f})")
                
                return result
                
        except asyncio.TimeoutError:
            request.status = RequestStatus.FAILED
            error_msg = f"Request timeout after {request.timeout} seconds"
            logger.error(f"Request {request.id} timed out")
            
        except Exception as e:
            request.status = RequestStatus.FAILED
            error_msg = str(e)
            logger.error(f"Request {request.id} failed: {e}")
            
            # Check if we should retry
            if request.retry_count < request.max_retries and self.should_retry_error(e):
                request.retry_count += 1
                request.status = RequestStatus.PENDING
                
                # Re-queue with exponential backoff
                delay = 2 ** request.retry_count
                await asyncio.sleep(delay)
                await self.request_queues[request.priority].put(request)
                
                logger.info(f"Retrying request {request.id} (attempt {request.retry_count})")
                return await self.process_request(request)
        
        # Return error result
        return {
            'request_id': request.id,
            'model': request.model,
            'error': error_msg,
            'status': 'failed',
            'retry_count': request.retry_count
        }
    
    def fix_json_response(self, content: str) -> str:
        """Attempt to fix common JSON formatting issues"""
        try:
            # Remove leading/trailing whitespace
            content = content.strip()
            
            # Remove markdown code blocks
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            
            # Fix common quote issues
            content = content.replace("'", '"')
            content = content.replace('True', 'true').replace('False', 'false')
            
            return content.strip()
            
        except Exception:
            return content
    
    def should_retry_error(self, error: Exception) -> bool:
        """Determine if an error should trigger a retry"""
        error_str = str(error).lower()
        
        # Retry on rate limits and temporary server errors
        if any(phrase in error_str for phrase in [
            'rate limit', 'timeout', 'server error', 'service unavailable',
            'internal error', 'bad gateway', 'connection error'
        ]):
            return True
        
        # Don't retry on authentication or quota errors
        if any(phrase in error_str for phrase in [
            'authentication', 'unauthorized', 'quota exceeded', 
            'invalid api key', 'billing'
        ]):
            return False
        
        return False
    
    async def store_usage_record(self, request: OpenAIRequest, usage: CompletionUsage, 
                               cost: float, processing_time: float):
        """Store usage record in database"""
        try:
            usage_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'model_name': request.model,
                'prompt_tokens': usage.prompt_tokens,
                'completion_tokens': usage.completion_tokens,
                'total_tokens': usage.total_tokens,
                'estimated_cost_usd': cost,
                'related_news_id': request.news_id,
                'purpose': request.purpose,
                'processing_time_seconds': processing_time,
                'request_id': request.id
            }
            
            db_adapter.insert_api_usage(usage_data)
            
        except Exception as e:
            logger.error(f"Failed to store usage record: {e}")
    
    async def start_workers(self, num_workers: int = 3):
        """Start worker processes to handle queued requests"""
        if self.running:
            logger.warning("Workers already running")
            return
        
        if not self.async_client:
            logger.error("AsyncOpenAI client not initialized")
            return
        
        self.running = True
        
        for i in range(num_workers):
            worker = asyncio.create_task(self.worker_loop(f"worker-{i}"))
            self.workers.append(worker)
        
        logger.info(f"Started {num_workers} OpenAI workers")
    
    async def worker_loop(self, worker_name: str):
        """Main worker loop to process requests"""
        logger.info(f"OpenAI worker {worker_name} started")
        
        while self.running:
            try:
                request = None
                
                # Check queues in priority order
                for priority in [Priority.CRITICAL, Priority.HIGH, Priority.NORMAL, Priority.LOW]:
                    try:
                        request = self.request_queues[priority].get_nowait()
                        break
                    except asyncio.QueueEmpty:
                        continue
                
                if request is None:
                    await asyncio.sleep(1)  # No requests, wait a bit
                    continue
                
                # Skip if request has failed conditions
                if request.status in [RequestStatus.RATE_LIMITED, RequestStatus.BUDGET_EXCEEDED]:
                    continue
                
                # Process the request
                result = await self.process_request(request)
                
                # Mark queue task as done
                self.request_queues[request.priority].task_done()
                
                # Store result in Redis
                redis_manager.set_cache(f"openai_result:{request.id}", result, ttl=3600)
                
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(5)  # Back off on error
        
        logger.info(f"OpenAI worker {worker_name} stopped")
    
    async def stop_workers(self):
        """Stop all worker processes"""
        if not self.running:
            return
        
        self.running = False
        
        for worker in self.workers:
            worker.cancel()
        
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
        logger.info("All OpenAI workers stopped")
    
    async def wait_for_result(self, request_id: str, timeout: int = 300) -> Optional[Dict[str, Any]]:
        """Wait for a request result"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            result = redis_manager.get_cache(f"openai_result:{request_id}")
            if result:
                return result
            
            await asyncio.sleep(1)
        
        logger.warning(f"Request {request_id} timed out waiting for result")
        return None
    
    def get_usage_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get OpenAI usage statistics"""
        try:
            # Get usage from database
            cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
            
            usage_df = db_adapter.execute_query(
                f"SELECT model_name, SUM(total_tokens) as total_tokens, "
                f"SUM(estimated_cost_usd) as total_cost, COUNT(*) as requests "
                f"FROM {db_adapter.get_table_prefix()}openai_api_usage "
                f"WHERE timestamp >= :cutoff "
                f"GROUP BY model_name",
                {'cutoff': cutoff_time.isoformat()}
            )
            
            if usage_df is None or usage_df.empty:
                return {'hours': hours, 'models': {}, 'total_cost': 0, 'total_requests': 0}
            
            models_stats = {}
            total_cost = 0
            total_requests = 0
            
            for _, row in usage_df.iterrows():
                model = row['model_name']
                models_stats[model] = {
                    'requests': int(row['requests']),
                    'total_tokens': int(row['total_tokens']),
                    'total_cost_usd': float(row['total_cost'])
                }
                total_cost += float(row['total_cost'])
                total_requests += int(row['requests'])
            
            # Get current budget status
            current_date = datetime.date.today()
            daily_spent = float(redis_manager.get_cache(f"budget:daily:{current_date}") or 0)
            monthly_spent = float(redis_manager.get_cache(f"budget:monthly:{current_date.strftime('%Y-%m')}") or 0)
            
            return {
                'time_window_hours': hours,
                'models': models_stats,
                'total_cost_usd': total_cost,
                'total_requests': total_requests,
                'budget_status': {
                    'daily_spent': daily_spent,
                    'daily_limit': self.budget_config.daily_budget_usd,
                    'daily_remaining': self.budget_config.daily_budget_usd - daily_spent,
                    'monthly_spent': monthly_spent,
                    'monthly_limit': self.budget_config.monthly_budget_usd,
                    'monthly_remaining': self.budget_config.monthly_budget_usd - monthly_spent
                },
                'queue_status': {
                    'critical': self.request_queues[Priority.CRITICAL].qsize(),
                    'high': self.request_queues[Priority.HIGH].qsize(),
                    'normal': self.request_queues[Priority.NORMAL].qsize(),
                    'low': self.request_queues[Priority.LOW].qsize()
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get usage statistics: {e}")
            return {'error': str(e)}

# Global OpenAI manager instance
openai_manager = OpenAIManager()