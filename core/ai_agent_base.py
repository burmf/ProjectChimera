#!/usr/bin/env python3
"""
AI Agent Base Class for Department-Based AI System
部門別AIエージェントシステムの基底クラス
"""

import asyncio
import datetime
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import os
from openai import OpenAI


class DepartmentType(Enum):
    """AI部門タイプ定義"""
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental" 
    SENTIMENT = "sentiment"
    RISK = "risk"
    EXECUTION = "execution"
    PORTFOLIO = "portfolio"


class AnalysisPriority(Enum):
    """分析優先度"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AnalysisRequest:
    """分析リクエストデータ構造"""
    request_id: str
    department: DepartmentType
    priority: AnalysisPriority
    data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime.datetime] = None
    expires_at: Optional[datetime.datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.datetime.now()
        if self.expires_at is None:
            self.expires_at = self.timestamp + datetime.timedelta(minutes=5)


@dataclass 
class AnalysisResult:
    """分析結果データ構造"""
    request_id: str
    department: DepartmentType
    confidence: float  # 0.0-1.0
    decision: Dict[str, Any]
    reasoning: str
    metadata: Dict[str, Any]
    timestamp: datetime.datetime
    processing_time_ms: float
    cost_usd: float
    model_used: str
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['department'] = self.department.value
        return result


class AIAgentBase(ABC):
    """
    AI部門エージェントの基底クラス
    各部門AIはこのクラスを継承して専門機能を実装
    """
    
    def __init__(self, department: DepartmentType, model_config: Dict[str, Any] = None):
        self.department = department
        self.logger = logging.getLogger(f"{__name__}.{department.value}")
        
        # OpenAI設定
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # モデル設定（部門別最適化）
        self.model_config = model_config or self._get_default_model_config()
        
        # パフォーマンス追跡
        self.request_count = 0
        self.total_cost = 0.0
        self.avg_processing_time = 0.0
        
        # 部門別制限設定
        self.max_requests_per_hour = self._get_rate_limit()
        self.max_daily_cost = self._get_cost_limit()
        
        self.logger.info(f"Initialized {department.value} AI Agent")
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """部門専用のシステムプロンプトを返す"""
        pass
    
    @abstractmethod
    def _validate_request_data(self, request: AnalysisRequest) -> bool:
        """リクエストデータの妥当性チェック"""
        pass
    
    @abstractmethod
    async def _analyze_data(self, request: AnalysisRequest) -> Dict[str, Any]:
        """部門固有の分析ロジック"""
        pass
    
    def _get_default_model_config(self) -> Dict[str, Any]:
        """部門別デフォルトモデル設定"""
        configs = {
            DepartmentType.TECHNICAL: {
                "model": "gpt-4o",
                "temperature": 0.3,
                "max_tokens": 1500,
                "top_p": 0.9
            },
            DepartmentType.FUNDAMENTAL: {
                "model": "o3",
                "temperature": 0.2,
                "max_tokens": 2000,
                "top_p": 0.85
            },
            DepartmentType.SENTIMENT: {
                "model": "gpt-4o-mini",
                "temperature": 0.4,
                "max_tokens": 1000,
                "top_p": 0.9
            },
            DepartmentType.RISK: {
                "model": "o3-mini",
                "temperature": 0.1,
                "max_tokens": 1200,
                "top_p": 0.8
            },
            DepartmentType.EXECUTION: {
                "model": "o3-mini",
                "temperature": 0.2,
                "max_tokens": 1000,
                "top_p": 0.85
            },
            DepartmentType.PORTFOLIO: {
                "model": "o3",
                "temperature": 0.15,
                "max_tokens": 1800,
                "top_p": 0.8
            }
        }
        return configs.get(self.department, configs[DepartmentType.TECHNICAL])
    
    def _get_rate_limit(self) -> int:
        """部門別レート制限"""
        limits = {
            DepartmentType.TECHNICAL: 60,      # 1分毎の分析
            DepartmentType.FUNDAMENTAL: 12,    # 5分毎の分析
            DepartmentType.SENTIMENT: 120,     # 30秒毎の分析
            DepartmentType.RISK: 240,         # 15秒毎の分析
            DepartmentType.EXECUTION: 180,     # 20秒毎の分析
            DepartmentType.PORTFOLIO: 24      # 2.5分毎の分析
        }
        return limits.get(self.department, 60)
    
    def _get_cost_limit(self) -> float:
        """部門別コスト制限（USD/日）"""
        limits = {
            DepartmentType.TECHNICAL: 15.0,
            DepartmentType.FUNDAMENTAL: 25.0,
            DepartmentType.SENTIMENT: 10.0,
            DepartmentType.RISK: 20.0,
            DepartmentType.EXECUTION: 12.0,
            DepartmentType.PORTFOLIO: 20.0
        }
        return limits.get(self.department, 15.0)
    
    async def process_request(self, request: AnalysisRequest) -> AnalysisResult:
        """
        分析リクエストを処理してレスポンスを返す
        """
        start_time = datetime.datetime.now()
        
        try:
            # リクエスト妥当性チェック
            if not self._validate_request_data(request):
                raise ValueError(f"Invalid request data for {self.department.value}")
            
            # レート制限チェック
            if not self._check_rate_limit():
                raise RuntimeError(f"Rate limit exceeded for {self.department.value}")
            
            # コスト制限チェック  
            if not self._check_cost_limit():
                raise RuntimeError(f"Daily cost limit exceeded for {self.department.value}")
            
            # 分析実行
            decision = await self._analyze_data(request)
            
            # OpenAI API呼び出し
            system_prompt = self._get_system_prompt()
            user_prompt = self._format_user_prompt(request)
            
            response = await self._call_openai_api(system_prompt, user_prompt)
            
            # 結果解析
            ai_result = self._parse_ai_response(response)
            
            # 分析結果をdecisionにマージ
            final_decision = {**decision, **ai_result.get('decision', {})}
            
            # 処理時間計算
            processing_time = (datetime.datetime.now() - start_time).total_seconds() * 1000
            
            # 結果作成
            result = AnalysisResult(
                request_id=request.request_id,
                department=self.department,
                confidence=ai_result.get('confidence', 0.5),
                decision=final_decision,
                reasoning=ai_result.get('reasoning', ''),
                metadata=ai_result.get('metadata', {}),
                timestamp=datetime.datetime.now(),
                processing_time_ms=processing_time,
                cost_usd=response.get('cost', 0.0),
                model_used=self.model_config['model']
            )
            
            # 統計更新
            self._update_statistics(result)
            
            self.logger.info(
                f"Processed {request.request_id} in {processing_time:.2f}ms "
                f"with confidence {result.confidence:.3f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process request {request.request_id}: {e}")
            
            # エラー時のデフォルト結果
            processing_time = (datetime.datetime.now() - start_time).total_seconds() * 1000
            return AnalysisResult(
                request_id=request.request_id,
                department=self.department,
                confidence=0.0,
                decision={'error': str(e)},
                reasoning=f"Processing failed: {str(e)}",
                metadata={'error': True},
                timestamp=datetime.datetime.now(),
                processing_time_ms=processing_time,
                cost_usd=0.0,
                model_used=self.model_config['model']
            )
    
    def _format_user_prompt(self, request: AnalysisRequest) -> str:
        """ユーザープロンプトのフォーマット"""
        prompt = f"分析リクエスト ID: {request.request_id}\n"
        prompt += f"優先度: {request.priority.name}\n"
        prompt += f"データ: {json.dumps(request.data, indent=2, ensure_ascii=False)}\n"
        
        if request.context:
            prompt += f"コンテキスト: {json.dumps(request.context, indent=2, ensure_ascii=False)}\n"
        
        return prompt
    
    async def _call_openai_api(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        """OpenAI API呼び出し"""
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=self.model_config['model'],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.model_config['temperature'],
                max_tokens=self.model_config['max_tokens'],
                top_p=self.model_config['top_p']
            )
            
            content = response.choices[0].message.content
            usage = response.usage
            
            # コスト計算（簡略化）
            cost = 0.0
            if usage:
                model_name = self.model_config['model']
                if 'o3' in model_name:
                    cost = (usage.prompt_tokens * 10 + usage.completion_tokens * 40) / 1_000_000
                elif 'gpt-4o' in model_name:
                    cost = (usage.prompt_tokens * 5 + usage.completion_tokens * 15) / 1_000_000
                else:
                    cost = (usage.prompt_tokens * 0.5 + usage.completion_tokens * 1.5) / 1_000_000
            
            return {
                'content': content,
                'usage': usage,
                'cost': cost
            }
            
        except Exception as e:
            self.logger.error(f"OpenAI API call failed: {e}")
            raise
    
    def _parse_ai_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """AI応答の解析"""
        try:
            content = response['content']
            
            # JSON形式での応答を期待
            if content.strip().startswith('{'):
                return json.loads(content)
            else:
                # JSON形式でない場合の処理
                return {
                    'confidence': 0.5,
                    'decision': {},
                    'reasoning': content,
                    'metadata': {}
                }
                
        except json.JSONDecodeError:
            # JSON解析失敗時のフォールバック
            return {
                'confidence': 0.3,
                'decision': {},
                'reasoning': response.get('content', 'Failed to parse response'),
                'metadata': {'parse_error': True}
            }
    
    def _check_rate_limit(self) -> bool:
        """レート制限チェック（簡略化）"""
        # 実装では Redis でリクエスト数を追跡
        return True
    
    def _check_cost_limit(self) -> bool:
        """コスト制限チェック（簡略化）"""
        # 実装では日次コストを追跡
        return self.total_cost < self.max_daily_cost
    
    def _update_statistics(self, result: AnalysisResult):
        """統計情報更新"""
        self.request_count += 1
        self.total_cost += result.cost_usd
        
        # 移動平均で処理時間更新
        alpha = 0.1
        self.avg_processing_time = (
            alpha * result.processing_time_ms + 
            (1 - alpha) * self.avg_processing_time
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """統計情報取得"""
        return {
            'department': self.department.value,
            'request_count': self.request_count,
            'total_cost_usd': self.total_cost,
            'avg_processing_time_ms': self.avg_processing_time,
            'rate_limit': self.max_requests_per_hour,
            'daily_cost_limit': self.max_daily_cost,
            'cost_utilization': self.total_cost / self.max_daily_cost
        }