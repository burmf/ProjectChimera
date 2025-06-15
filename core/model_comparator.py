#!/usr/bin/env python3
"""
AI Model Comparator
複数AIモデル比較システム
"""

import asyncio
import datetime
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sys
import os


from core.ab_test_manager import ab_test_manager, ModelType, TestResult
from core.ai_manager import run_openai_parallel_inference_async


@dataclass
class ModelComparisonRequest:
    """モデル比較リクエスト"""
    request_id: str
    models_to_compare: List[ModelType]
    input_data: Dict[str, Any]
    test_id: Optional[str] = None
    priority: int = 2
    timeout_seconds: int = 30


@dataclass
class ModelResponse:
    """個別モデルレスポンス"""
    model: ModelType
    response: Dict[str, Any]
    processing_time_ms: float
    cost_usd: float
    confidence_score: float
    error: Optional[str] = None
    success: bool = True


@dataclass
class ComparisonResult:
    """比較結果"""
    request_id: str
    responses: List[ModelResponse]
    best_model: Optional[ModelType]
    consensus_response: Dict[str, Any]
    total_cost: float
    total_time_ms: float
    timestamp: datetime.datetime
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['responses'] = [
            {**resp.__dict__, 'model': resp.model.value}
            for resp in self.responses
        ]
        if self.best_model:
            result['best_model'] = self.best_model.value
        return result


class ModelComparator:
    """AIモデル比較システム"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # モデル設定
        self.model_configs = {
            ModelType.GPT4: {
                'model_name': 'gpt-4',
                'max_tokens': 1000,
                'temperature': 0.3,
                'cost_per_1k_input': 0.03,
                'cost_per_1k_output': 0.06
            },
            ModelType.GPT4_TURBO: {
                'model_name': 'gpt-4-turbo',
                'max_tokens': 1000,
                'temperature': 0.3,
                'cost_per_1k_input': 0.01,
                'cost_per_1k_output': 0.03
            },
            ModelType.O3_MINI: {
                'model_name': 'o3-mini',
                'max_tokens': 1000,
                'temperature': 0.3,
                'cost_per_1k_input': 1.10,
                'cost_per_1k_output': 4.40
            },
            ModelType.O3: {
                'model_name': 'o3',
                'max_tokens': 1000,
                'temperature': 0.3,
                'cost_per_1k_input': 5.00,  # 推定
                'cost_per_1k_output': 20.00  # 推定
            }
        }
        
        self.logger.info("Model Comparator initialized")
    
    async def compare_models(self, request: ModelComparisonRequest) -> ComparisonResult:
        """複数モデルを並行実行して比較"""
        start_time = time.time()
        
        try:
            # 並行してモデル実行
            tasks = []
            for model in request.models_to_compare:
                task = self._execute_single_model(model, request.input_data)
                tasks.append((model, task))
            
            # 結果収集
            responses = []
            total_cost = 0.0
            
            for model, task in tasks:
                try:
                    response = await asyncio.wait_for(task, timeout=request.timeout_seconds)
                    responses.append(response)
                    total_cost += response.cost_usd
                    
                    # A/Bテストへの記録
                    if request.test_id:
                        await ab_test_manager.record_test_result(
                            test_id=request.test_id,
                            model_used=model,
                            request_data=request.input_data,
                            ai_response=response.response,
                            processing_time_ms=response.processing_time_ms,
                            cost_usd=response.cost_usd,
                            confidence_score=response.confidence_score
                        )
                    
                except asyncio.TimeoutError:
                    self.logger.warning(f"Model {model.value} timed out")
                    responses.append(ModelResponse(
                        model=model,
                        response={},
                        processing_time_ms=request.timeout_seconds * 1000,
                        cost_usd=0.0,
                        confidence_score=0.0,
                        error="Timeout",
                        success=False
                    ))
                except Exception as e:
                    self.logger.error(f"Model {model.value} failed: {e}")
                    responses.append(ModelResponse(
                        model=model,
                        response={},
                        processing_time_ms=0.0,
                        cost_usd=0.0,
                        confidence_score=0.0,
                        error=str(e),
                        success=False
                    ))
            
            # 最良モデル決定
            best_model = self._determine_best_model(responses)
            
            # コンセンサスレスポンス生成
            consensus_response = self._generate_consensus_response(responses)
            
            total_time_ms = (time.time() - start_time) * 1000
            
            result = ComparisonResult(
                request_id=request.request_id,
                responses=responses,
                best_model=best_model,
                consensus_response=consensus_response,
                total_cost=total_cost,
                total_time_ms=total_time_ms,
                timestamp=datetime.datetime.now()
            )
            
            self.logger.info(
                f"Model comparison completed: {len(responses)} models, "
                f"best: {best_model.value if best_model else 'None'}, "
                f"cost: ${total_cost:.4f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Model comparison failed: {e}")
            return ComparisonResult(
                request_id=request.request_id,
                responses=[],
                best_model=None,
                consensus_response={'error': str(e)},
                total_cost=0.0,
                total_time_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.datetime.now()
            )
    
    async def _execute_single_model(self, model: ModelType, input_data: Dict[str, Any]) -> ModelResponse:
        """単一モデルの実行"""
        start_time = time.time()
        
        try:
            config = self.model_configs.get(model)
            if not config:
                raise ValueError(f"Unsupported model: {model.value}")
            
            # ニュース分析の場合
            if 'news' in input_data or 'title' in input_data:
                news_content = input_data.get('news', input_data.get('title', ''))
                
                # OpenAI API呼び出し
                if model in [ModelType.GPT4, ModelType.GPT4_TURBO, ModelType.O3_MINI, ModelType.O3]:
                    response = await self._call_openai_model(model, news_content, config)
                else:
                    # 他のモデル（将来実装）
                    response = await self._call_other_model(model, news_content, config)
            
            # 市場データ分析の場合
            elif 'price_data' in input_data:
                response = await self._analyze_market_data(model, input_data, config)
            
            else:
                # 一般的な分析
                response = await self._general_analysis(model, input_data, config)
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # 信頼度スコア抽出
            confidence_score = response.get('confidence', 0.5)
            if isinstance(confidence_score, str):
                try:
                    confidence_score = float(confidence_score)
                except ValueError:
                    confidence_score = 0.5
            
            # コスト計算（簡易版）
            estimated_tokens = len(str(input_data)) + len(str(response))
            cost_usd = self._estimate_cost(model, estimated_tokens, config)
            
            return ModelResponse(
                model=model,
                response=response,
                processing_time_ms=processing_time_ms,
                cost_usd=cost_usd,
                confidence_score=confidence_score,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Model {model.value} execution failed: {e}")
            return ModelResponse(
                model=model,
                response={},
                processing_time_ms=(time.time() - start_time) * 1000,
                cost_usd=0.0,
                confidence_score=0.0,
                error=str(e),
                success=False
            )
    
    async def _call_openai_model(self, model: ModelType, content: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """OpenAIモデル呼び出し"""
        try:
            # システムプロンプト
            system_prompt = self._get_system_prompt_for_model(model)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Analyze this news: {content}"}
            ]
            
            # 既存のAI管理システムを使用
            results = await run_openai_parallel_inference_async(
                news_list=[content],
                models=[config['model_name']],
                return_all_results=True
            )
            
            if results and len(results) > 0:
                result = results[0]
                if 'analysis' in result:
                    return result['analysis']
                else:
                    return result
            else:
                return {'error': 'No response from OpenAI'}
                
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}")
            return {'error': str(e)}
    
    async def _call_other_model(self, model: ModelType, content: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """他のモデル呼び出し（将来実装用）"""
        # Claude, Gemini等の実装予定
        return {
            'trade_warranted': False,
            'pair': 'N/A',
            'direction': 'N/A',
            'confidence': 0.5,
            'reasoning': f'Model {model.value} not yet implemented',
            'error': 'Model not implemented'
        }
    
    async def _analyze_market_data(self, model: ModelType, input_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """市場データ分析"""
        # 市場データ用の分析ロジック
        price_data = input_data.get('price_data', {})
        
        # 簡易的な分析（実際はより詳細な分析を実装）
        close_price = price_data.get('close', 0)
        open_price = price_data.get('open', 0)
        
        if close_price > open_price:
            direction = 'long'
            confidence = 0.6
        elif close_price < open_price:
            direction = 'short'
            confidence = 0.6
        else:
            direction = 'hold'
            confidence = 0.4
        
        return {
            'trade_warranted': direction != 'hold',
            'pair': input_data.get('pair', 'USD/JPY'),
            'direction': direction,
            'confidence': confidence,
            'reasoning': f'{model.value} market data analysis'
        }
    
    async def _general_analysis(self, model: ModelType, input_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """一般的な分析"""
        return {
            'trade_warranted': False,
            'pair': 'N/A',
            'direction': 'hold',
            'confidence': 0.5,
            'reasoning': f'{model.value} general analysis'
        }
    
    def _get_system_prompt_for_model(self, model: ModelType) -> str:
        """モデル別システムプロンプト"""
        base_prompt = """You are an expert financial analyst. Analyze the provided news and respond ONLY in JSON format.
The JSON should contain:
- "trade_warranted": boolean
- "pair": string (e.g., "USD/JPY")
- "direction": string ("long", "short", or "hold")
- "confidence": float (0.0 to 1.0)
- "reasoning": string (brief justification, max 100 words)"""
        
        if model == ModelType.O3 or model == ModelType.O3_MINI:
            return base_prompt + "\n\nUse systematic reasoning and consider market context carefully."
        else:
            return base_prompt
    
    def _estimate_cost(self, model: ModelType, estimated_tokens: int, config: Dict[str, Any]) -> float:
        """コスト推定"""
        try:
            # 入力・出力トークンを大まかに分割
            input_tokens = estimated_tokens * 0.7
            output_tokens = estimated_tokens * 0.3
            
            input_cost = (input_tokens / 1000) * config.get('cost_per_1k_input', 0.01)
            output_cost = (output_tokens / 1000) * config.get('cost_per_1k_output', 0.03)
            
            return input_cost + output_cost
            
        except Exception:
            return 0.01  # デフォルトコスト
    
    def _determine_best_model(self, responses: List[ModelResponse]) -> Optional[ModelType]:
        """最良モデルの決定"""
        successful_responses = [r for r in responses if r.success and r.confidence_score > 0]
        
        if not successful_responses:
            return None
        
        # 総合スコア計算（信頼度、コスト効率、応答時間を考慮）
        best_model = None
        best_score = -1.0
        
        for response in successful_responses:
            # コスト効率
            cost_efficiency = response.confidence_score / max(response.cost_usd, 0.001)
            
            # 時間効率
            time_efficiency = response.confidence_score / max(response.processing_time_ms / 1000.0, 0.1)
            
            # 総合スコア
            score = (
                response.confidence_score * 0.4 +  # 信頼度
                min(cost_efficiency / 10.0, 1.0) * 0.3 +  # コスト効率
                min(time_efficiency, 1.0) * 0.3  # 時間効率
            )
            
            if score > best_score:
                best_score = score
                best_model = response.model
        
        return best_model
    
    def _generate_consensus_response(self, responses: List[ModelResponse]) -> Dict[str, Any]:
        """コンセンサスレスポンス生成"""
        successful_responses = [r for r in responses if r.success]
        
        if not successful_responses:
            return {'error': 'No successful responses'}
        
        # 投票による決定
        trade_votes = []
        directions = []
        confidences = []
        pairs = []
        
        for response in successful_responses:
            resp_data = response.response
            
            if isinstance(resp_data, dict):
                trade_votes.append(resp_data.get('trade_warranted', False))
                directions.append(resp_data.get('direction', 'hold'))
                confidences.append(resp_data.get('confidence', 0.5))
                pairs.append(resp_data.get('pair', 'USD/JPY'))
        
        if not trade_votes:
            return {'error': 'No valid responses to analyze'}
        
        # 多数決
        trade_warranted = sum(trade_votes) > len(trade_votes) / 2
        
        # 最頻値
        most_common_direction = max(set(directions), key=directions.count) if directions else 'hold'
        most_common_pair = max(set(pairs), key=pairs.count) if pairs else 'USD/JPY'
        
        # 平均信頼度
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        return {
            'trade_warranted': trade_warranted,
            'pair': most_common_pair,
            'direction': most_common_direction,
            'confidence': avg_confidence,
            'reasoning': f'Consensus from {len(successful_responses)} models',
            'model_count': len(successful_responses),
            'agreement_rate': max(directions.count(most_common_direction) / len(directions), 0.5) if directions else 0.5
        }
    
    async def run_batch_comparison(
        self, 
        test_cases: List[Dict[str, Any]], 
        models: List[ModelType],
        test_id: Optional[str] = None
    ) -> List[ComparisonResult]:
        """バッチ比較実行"""
        results = []
        
        for i, test_case in enumerate(test_cases):
            request = ModelComparisonRequest(
                request_id=f"batch_{i}_{int(time.time())}",
                models_to_compare=models,
                input_data=test_case,
                test_id=test_id
            )
            
            result = await self.compare_models(request)
            results.append(result)
            
            # 少し待機（レート制限対策）
            await asyncio.sleep(0.5)
        
        return results
    
    def analyze_batch_results(self, results: List[ComparisonResult]) -> Dict[str, Any]:
        """バッチ結果の分析"""
        if not results:
            return {'error': 'No results to analyze'}
        
        model_stats = {}
        total_cost = 0.0
        total_time = 0.0
        
        for result in results:
            total_cost += result.total_cost
            total_time += result.total_time_ms
            
            for response in result.responses:
                if not response.success:
                    continue
                
                model_name = response.model.value
                if model_name not in model_stats:
                    model_stats[model_name] = {
                        'count': 0,
                        'total_confidence': 0.0,
                        'total_cost': 0.0,
                        'total_time': 0.0,
                        'best_count': 0
                    }
                
                stats = model_stats[model_name]
                stats['count'] += 1
                stats['total_confidence'] += response.confidence_score
                stats['total_cost'] += response.cost_usd
                stats['total_time'] += response.processing_time_ms
                
                # 最良モデル回数カウント
                if result.best_model == response.model:
                    stats['best_count'] += 1
        
        # 平均値計算
        analysis = {
            'total_comparisons': len(results),
            'total_cost': total_cost,
            'avg_cost_per_comparison': total_cost / len(results),
            'total_time_ms': total_time,
            'avg_time_per_comparison': total_time / len(results),
            'model_performance': {}
        }
        
        for model_name, stats in model_stats.items():
            if stats['count'] > 0:
                analysis['model_performance'][model_name] = {
                    'requests': stats['count'],
                    'avg_confidence': stats['total_confidence'] / stats['count'],
                    'avg_cost': stats['total_cost'] / stats['count'],
                    'avg_time': stats['total_time'] / stats['count'],
                    'best_model_rate': stats['best_count'] / len(results),
                    'total_cost': stats['total_cost'],
                    'cost_efficiency': stats['total_confidence'] / max(stats['total_cost'], 0.001)
                }
        
        return analysis


# シングルトンインスタンス
model_comparator = ModelComparator()