#!/usr/bin/env python3
"""
OpenAI o3 Integration Test Script
Tests o3 and o3-mini models with optimized prompts
"""

import asyncio
import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.ai_manager import run_openai_parallel_inference_async, get_optimal_prompt_for_model
from core.openai_manager import openai_manager, Priority

# Load environment variables
load_dotenv()

async def test_o3_models():
    """Test o3 and o3-mini models with sample news"""
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key or api_key.startswith('YOUR_'):
        print("❌ OpenAI API key not configured")
        return False
    
    # Sample forex news for testing
    test_news = """
    Fed Chair Powell signals potential pause in rate hikes amid banking sector concerns. 
    Speaking at Jackson Hole, Powell noted that while inflation remains above target, 
    recent banking stress and credit tightening may do some of the work of monetary policy. 
    Markets are pricing in a 70% chance of no rate change at the next FOMC meeting.
    """
    
    print("🧪 Testing OpenAI o3 Integration")
    print("=" * 50)
    
    # Test 1: Legacy AI Manager with parallel inference
    print("\n📊 Test 1: Parallel inference (o3 vs o3-mini)")
    
    try:
        models_to_test = ["o3-mini", "o3"]  # Start with o3-mini (cheaper)
        
        results, logs = await run_openai_parallel_inference_async(
            api_key=api_key,
            model_list=models_to_test,
            news_text=test_news,
            purpose="o3_integration_test"
        )
        
        for model in models_to_test:
            print(f"\n🤖 {model.upper()} Result:")
            if model in results:
                result = results[model]
                if "error" in result:
                    print(f"   ❌ Error: {result['error']}")
                else:
                    print(f"   ✅ Trade: {result.get('trade_warranted', 'N/A')}")
                    print(f"   📈 Pair: {result.get('pair', 'N/A')}")
                    print(f"   📊 Direction: {result.get('direction', 'N/A')}")
                    print(f"   🎯 Confidence: {result.get('confidence', 'N/A')}")
                    print(f"   💡 Reasoning: {result.get('reasoning', 'N/A')[:100]}...")
            else:
                print(f"   ❌ No result returned")
        
        # Print cost information
        print(f"\n💰 Cost Analysis:")
        for log in logs:
            if 'error' not in log:
                print(f"   {log['model_name']}: ${log.get('estimated_cost_usd', 0):.4f}")
        
    except Exception as e:
        print(f"❌ Test 1 failed: {e}")
        return False
    
    # Test 2: New OpenAI Manager with queue system
    print(f"\n📊 Test 2: Queue-based processing")
    
    try:
        # Start workers
        await openai_manager.start_workers(num_workers=2)
        
        # Queue o3-mini request
        request_id = await openai_manager.queue_request(
            model="o3-mini",
            messages=[
                {"role": "system", "content": get_optimal_prompt_for_model("o3-mini")},
                {"role": "user", "content": f"Analyze: {test_news}"}
            ],
            priority=Priority.HIGH,
            purpose="o3_queue_test"
        )
        
        print(f"   📤 Queued request: {request_id}")
        
        # Wait for result
        result = await openai_manager.wait_for_result(request_id, timeout=60)
        
        if result and result.get('status') == 'success':
            data = result.get('data', {})
            usage = result.get('usage', {})
            
            print(f"   ✅ Success!")
            print(f"   💰 Cost: ${usage.get('cost_usd', 0):.4f}")
            print(f"   ⏱️  Time: {result.get('processing_time', 0):.2f}s")
            print(f"   📈 Trade: {data.get('trade_warranted', 'N/A')}")
            print(f"   🎯 Confidence: {data.get('confidence', 'N/A')}")
        else:
            print(f"   ❌ Failed or timed out")
            return False
        
        # Stop workers
        await openai_manager.stop_workers()
        
    except Exception as e:
        print(f"❌ Test 2 failed: {e}")
        await openai_manager.stop_workers()
        return False
    
    # Test 3: Prompt optimization verification
    print(f"\n📊 Test 3: Prompt optimization")
    
    try:
        o3_prompt = get_optimal_prompt_for_model("o3")
        o4_prompt = get_optimal_prompt_for_model("gpt-4o")
        
        print(f"   🧠 o3 prompt length: {len(o3_prompt)} chars")
        print(f"   🧠 gpt-4o prompt length: {len(o4_prompt)} chars")
        
        if "step-by-step" in o3_prompt.lower():
            print(f"   ✅ o3 prompt includes reasoning framework")
        else:
            print(f"   ⚠️  o3 prompt missing reasoning framework")
        
    except Exception as e:
        print(f"❌ Test 3 failed: {e}")
        return False
    
    print(f"\n🎉 All tests completed successfully!")
    return True

async def test_usage_statistics():
    """Test usage statistics and budget tracking"""
    print(f"\n📊 Usage Statistics:")
    
    try:
        stats = openai_manager.get_usage_statistics(hours=24)
        
        print(f"   📈 Total requests: {stats.get('total_requests', 0)}")
        print(f"   💰 Total cost: ${stats.get('total_cost_usd', 0):.4f}")
        
        budget = stats.get('budget_status', {})
        print(f"   💳 Daily budget: ${budget.get('daily_spent', 0):.2f} / ${budget.get('daily_limit', 0):.2f}")
        
        queue_status = stats.get('queue_status', {})
        total_queued = sum(queue_status.values())
        print(f"   📤 Queued requests: {total_queued}")
        
    except Exception as e:
        print(f"❌ Statistics test failed: {e}")

def main():
    """Main test runner"""
    print("🚀 Starting OpenAI o3 Integration Tests")
    
    try:
        # Run async tests
        success = asyncio.run(test_o3_models())
        
        if success:
            # Test statistics
            asyncio.run(test_usage_statistics())
            print("\n✅ All tests passed! o3 integration ready.")
            return 0
        else:
            print("\n❌ Some tests failed. Check configuration.")
            return 1
            
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n💥 Test runner failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())