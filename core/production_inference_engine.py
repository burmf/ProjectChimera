"""
Production Inference Engine
学習済み推論エンジンを用いた実運用システム

Purpose:
1. 学習済みパターンを用いた高精度推論
2. リアルタイム市場データでの投資判断
3. 継続的なフィードバック学習
4. リスク管理とポジション最適化
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

from core.backtest_learning_engine import backtest_learning_engine, InferenceResult
from core.ai_manager import ai_manager
from core.alpha_integration import integration_engine
from core.database_adapter import database_adapter

@dataclass
class ProductionDecision:
    """実運用投資判断"""
    timestamp: datetime
    market_data: Dict[str, Any]
    ai_inference: Dict[str, Any]
    alpha_signal: Dict[str, Any]
    
    # 投資判断
    action: str  # 'buy', 'sell', 'hold'
    position_size: float  # 0.0-1.0
    confidence: float  # 0.0-1.0
    
    # 実行パラメータ
    entry_price: float
    stop_loss: float
    take_profit: float
    holding_period: int  # minutes
    
    # リスク指標
    expected_return: float
    max_risk: float
    sharpe_estimate: float
    
    # メタデータ
    decision_id: str
    learning_confidence: float
    model_version: str

class ProductionInferenceEngine:
    """実運用推論エンジン"""
    
    def __init__(self):
        """初期化"""
        self.logger = logging.getLogger(__name__)
        
        # 学習済みエンジン参照
        self.learning_engine = backtest_learning_engine
        
        # 実運用設定
        self.production_config = {
            'min_confidence_threshold': 0.70,    # 最小実行信頼度
            'max_position_size': 0.05,           # 最大ポジションサイズ 5%
            'max_daily_trades': 10,              # 最大日次取引数
            'risk_budget_per_trade': 0.01,      # 取引あたりリスク予算 1%
            'learning_update_frequency': 100,    # 学習更新頻度（取引数）
            'emergency_stop_drawdown': 0.05     # 緊急停止ドローダウン 5%
        }
        
        # 実行履歴
        self.production_history: List[ProductionDecision] = []
        self.daily_trade_count = 0
        self.last_trade_date = datetime.now().date()
        
        # パフォーマンス追跡
        self.performance_metrics = {
            'total_trades': 0,
            'profitable_trades': 0,
            'cumulative_return': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'last_update': datetime.now()
        }

    async def execute_production_inference(self, current_market_data: Dict) -> Optional[ProductionDecision]:
        """
        実運用推論実行
        
        Args:
            current_market_data: 現在の市場データ
            
        Returns:
            投資判断決定 or None（条件未満足時）
        """
        
        try:
            # 1. 前提条件チェック
            if not await self._validate_execution_conditions():
                return None
            
            # 2. 学習済みエンジンの準備状態確認
            if not self._is_learning_ready():
                self.logger.warning("Learning engine not ready for production")
                return None
            
            # 3. 市場データの前処理
            processed_data = await self._preprocess_market_data(current_market_data)
            
            # 4. AI推論実行（学習済みパターンを活用）
            ai_inference = await self._execute_learned_inference(processed_data)
            
            # 5. Alpha統合エンジンでの検証
            alpha_signal = await self._get_alpha_validation(processed_data)
            
            # 6. 統合判断とリスク調整
            production_decision = await self._make_production_decision(
                processed_data, ai_inference, alpha_signal
            )
            
            if production_decision:
                # 7. 決定記録と実行準備
                await self._record_decision(production_decision)
                
                # 8. パフォーマンス更新
                self._update_performance_tracking()
                
                self.logger.info(f"Production decision made: {production_decision.action} "
                               f"size: {production_decision.position_size:.2%} "
                               f"confidence: {production_decision.confidence:.2%}")
            
            return production_decision
            
        except Exception as e:
            self.logger.error(f"Production inference failed: {e}")
            return None

    async def _validate_execution_conditions(self) -> bool:
        """実行条件検証"""
        
        # 日付チェック・取引回数リセット
        today = datetime.now().date()
        if today != self.last_trade_date:
            self.daily_trade_count = 0
            self.last_trade_date = today
        
        # 取引回数制限チェック
        if self.daily_trade_count >= self.production_config['max_daily_trades']:
            self.logger.warning("Daily trade limit reached")
            return False
        
        # ドローダウン緊急停止チェック
        if self.performance_metrics['current_drawdown'] > self.production_config['emergency_stop_drawdown']:
            self.logger.error("Emergency stop triggered due to excessive drawdown")
            return False
        
        # 市場時間チェック（簡易版）
        current_hour = datetime.now().hour
        if not (8 <= current_hour <= 20):  # 8:00-20:00のみ取引
            return False
        
        return True

    def _is_learning_ready(self) -> bool:
        """学習準備状態確認"""
        
        summary = self.learning_engine.get_learning_summary()
        return summary['ready_for_production']

    async def _preprocess_market_data(self, market_data: Dict) -> Dict:
        """市場データ前処理"""
        
        try:
            # 基本データ検証
            required_fields = ['current_price', 'timestamp']
            for field in required_fields:
                if field not in market_data:
                    raise ValueError(f"Missing required field: {field}")
            
            # 追加コンテキスト情報取得
            enhanced_data = market_data.copy()
            
            # 過去データ補完（データベースから）
            if 'price_history' not in enhanced_data:
                enhanced_data['price_history'] = await self._get_recent_price_history()
            
            # テクニカル指標補完
            if 'technical_indicators' not in enhanced_data:
                enhanced_data['technical_indicators'] = self._calculate_technical_indicators(
                    enhanced_data.get('price_history', [])
                )
            
            # 市場コンテキスト強化
            enhanced_data['market_context'] = await self._enhance_market_context(enhanced_data)
            
            return enhanced_data
            
        except Exception as e:
            self.logger.error(f"Market data preprocessing failed: {e}")
            return market_data

    async def _execute_learned_inference(self, market_data: Dict) -> Dict:
        """学習済みパターンを活用した推論実行"""
        
        try:
            # 学習済みエンジンから成功パターンを取得
            successful_patterns = self.learning_engine._get_successful_patterns()
            recent_metrics = self.learning_engine.learning_metrics
            
            # 学習結果を組み込んだカスタムプロンプト生成
            enhanced_prompt = f"""
高精度投資判断システム - 学習済みパターン活用

学習結果サマリー:
- 予測精度: {recent_metrics.accuracy_rate:.2%}
- 累計収益: {recent_metrics.cumulative_profit:.2%}
- 信頼度: {recent_metrics.learning_confidence:.2%}
- 成功パターン: {successful_patterns}

現在の市場状況:
{json.dumps(market_data, indent=2)}

過去の学習から特に有効だった投資パターン:
{self._get_top_performing_patterns()}

タスク:
上記の学習結果を活用し、現在の市場状況で最適な投資判断を行ってください。

判断基準:
1. 学習済み成功パターンとの類似度
2. 現在の市場環境での適用可能性
3. リスク・リターンの最適化
4. 信頼度の定量評価

必須回答形式（JSON）:
{{
    "action": "buy/sell/hold",
    "confidence": 0-100の信頼度,
    "position_size": 0-100のポジションサイズ％,
    "reasoning": "判断理由",
    "pattern_match": "適用した学習パターン",
    "expected_return": 期待リターン％,
    "max_risk": 最大リスク％,
    "holding_period": 推奨保有期間（分）,
    "stop_loss_distance": ストップロス距離％,
    "take_profit_distance": テイクプロフィット距離％
}}
"""
            
            # AI推論実行
            ai_response = await ai_manager.get_trading_decision_async(
                news_content=json.dumps(market_data),
                custom_prompt=enhanced_prompt
            )
            
            # レスポンス解析
            parsed_response = self._parse_ai_response(ai_response)
            
            return parsed_response
            
        except Exception as e:
            self.logger.error(f"Learned inference execution failed: {e}")
            return {}

    async def _get_alpha_validation(self, market_data: Dict) -> Dict:
        """Alpha統合エンジンでの検証"""
        
        try:
            # Alpha統合エンジンで独立検証
            current_price = market_data.get('current_price', 0)
            alpha_signal = integration_engine.generate_integrated_alpha_signal(
                market_data, current_price
            )
            
            if alpha_signal:
                return {
                    'signal_strength': alpha_signal.signal_strength,
                    'confidence': alpha_signal.confidence,
                    'expected_edge': alpha_signal.expected_edge_bps,
                    'risk_adjusted_return': alpha_signal.risk_adjusted_return,
                    'optimal_position_size': alpha_signal.optimal_position_size
                }
            else:
                return {'signal_strength': 0, 'confidence': 0}
                
        except Exception as e:
            self.logger.error(f"Alpha validation failed: {e}")
            return {}

    async def _make_production_decision(self, market_data: Dict, 
                                      ai_inference: Dict, alpha_signal: Dict) -> Optional[ProductionDecision]:
        """統合判断とリスク調整"""
        
        try:
            # 信頼度チェック
            ai_confidence = ai_inference.get('confidence', 0) / 100.0
            alpha_confidence = alpha_signal.get('confidence', 0)
            
            # 統合信頼度計算
            integrated_confidence = (ai_confidence * 0.6 + alpha_confidence * 0.4)
            
            if integrated_confidence < self.production_config['min_confidence_threshold']:
                self.logger.info(f"Confidence too low: {integrated_confidence:.2%}")
                return None
            
            # アクション決定
            ai_action = ai_inference.get('action', 'hold')
            alpha_action = 'buy' if alpha_signal.get('signal_strength', 0) > 0.1 else (
                'sell' if alpha_signal.get('signal_strength', 0) < -0.1 else 'hold'
            )
            
            # アクション一致性確認
            if ai_action != alpha_action and ai_action != 'hold' and alpha_action != 'hold':
                self.logger.warning(f"Action mismatch: AI={ai_action}, Alpha={alpha_action}")
                return None
            
            final_action = ai_action if ai_action != 'hold' else alpha_action
            
            if final_action == 'hold':
                return None
            
            # ポジションサイズ決定
            ai_size = ai_inference.get('position_size', 0) / 100.0
            alpha_size = alpha_signal.get('optimal_position_size', 0)
            
            # 保守的にサイズ決定
            position_size = min(
                ai_size * 0.7 + alpha_size * 0.3,  # 加重平均
                self.production_config['max_position_size'],  # 最大制限
                integrated_confidence * 0.1  # 信頼度比例
            )
            
            # リスク調整
            position_size = self._apply_risk_adjustments(position_size, market_data)
            
            # 実行価格とストップ計算
            current_price = market_data.get('current_price', 0)
            stop_loss_distance = ai_inference.get('stop_loss_distance', 2.0) / 100.0
            take_profit_distance = ai_inference.get('take_profit_distance', 4.0) / 100.0
            
            if final_action == 'buy':
                entry_price = current_price * 1.0005  # 0.05% slippage
                stop_loss = entry_price * (1 - stop_loss_distance)
                take_profit = entry_price * (1 + take_profit_distance)
            else:  # sell
                entry_price = current_price * 0.9995
                stop_loss = entry_price * (1 + stop_loss_distance)
                take_profit = entry_price * (1 - take_profit_distance)
            
            # 期待リターンとリスク計算
            expected_return = ai_inference.get('expected_return', 0) / 100.0
            max_risk = stop_loss_distance * position_size
            
            # シャープ推定
            sharpe_estimate = expected_return / max_risk if max_risk > 0 else 0
            
            # 保有期間
            holding_period = min(
                ai_inference.get('holding_period', 60),
                240  # 最大4時間
            )
            
            # 決定ID生成
            decision_id = f"prod_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.production_history)}"
            
            decision = ProductionDecision(
                timestamp=datetime.now(),
                market_data=market_data,
                ai_inference=ai_inference,
                alpha_signal=alpha_signal,
                
                action=final_action,
                position_size=position_size,
                confidence=integrated_confidence,
                
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                holding_period=holding_period,
                
                expected_return=expected_return,
                max_risk=max_risk,
                sharpe_estimate=sharpe_estimate,
                
                decision_id=decision_id,
                learning_confidence=self.learning_engine.learning_metrics.learning_confidence,
                model_version="v1.0"
            )
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Production decision making failed: {e}")
            return None

    def _apply_risk_adjustments(self, position_size: float, market_data: Dict) -> float:
        """リスク調整適用"""
        
        try:
            # ボラティリティ調整
            volatility = market_data.get('volatility', 0.01)
            if volatility > 0.03:  # 3%以上の高ボラティリティ
                position_size *= 0.5
            elif volatility > 0.02:  # 2%以上の中ボラティリティ
                position_size *= 0.75
            
            # 時間帯調整
            current_hour = datetime.now().hour
            if current_hour < 9 or current_hour > 17:  # 非アクティブ時間
                position_size *= 0.6
            
            # 直近パフォーマンス調整
            if self.performance_metrics['current_drawdown'] > 0.02:  # 2%以上のドローダウン
                position_size *= 0.5
            
            # リスク予算制限
            risk_budget_limit = self.production_config['risk_budget_per_trade']
            if position_size > risk_budget_limit:
                position_size = risk_budget_limit
            
            return max(position_size, 0.001)  # 最小0.1%
            
        except Exception as e:
            self.logger.error(f"Risk adjustment failed: {e}")
            return position_size * 0.5  # 保守的にサイズ削減

    async def _record_decision(self, decision: ProductionDecision):
        """決定記録"""
        
        try:
            # 履歴に追加
            self.production_history.append(decision)
            
            # 日次取引カウント更新
            self.daily_trade_count += 1
            
            # データベース保存
            decision_data = asdict(decision)
            decision_data['timestamp'] = decision.timestamp.isoformat()
            
            # 簡易保存（実装では適切なDB使用）
            with open('/home/ec2-user/BOT/data/production_decisions.jsonl', 'a') as f:
                f.write(json.dumps(decision_data) + '\n')
            
            self.logger.info(f"Decision recorded: {decision.decision_id}")
            
        except Exception as e:
            self.logger.error(f"Decision recording failed: {e}")

    def _parse_ai_response(self, ai_response: Dict) -> Dict:
        """AI レスポンス解析"""
        
        try:
            # 構造化されたレスポンスから値抽出
            if isinstance(ai_response, dict):
                return ai_response
            
            # テキストレスポンスの場合のJSONパース
            if isinstance(ai_response, str):
                # JSON部分を抽出
                import re
                json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    return json.loads(json_str)
            
            # デフォルト値
            return {
                'action': 'hold',
                'confidence': 0,
                'position_size': 0,
                'reasoning': 'Response parsing failed'
            }
            
        except Exception as e:
            self.logger.error(f"AI response parsing failed: {e}")
            return {'action': 'hold', 'confidence': 0}

    def _get_top_performing_patterns(self) -> str:
        """トップパフォーマンスパターン取得"""
        
        # 学習履歴から高利益パターンを抽出
        if not self.learning_engine.inference_history:
            return "学習データ不足"
        
        profitable_inferences = [
            r for r in self.learning_engine.inference_history 
            if r.profit_impact and r.profit_impact > 0.005  # 0.5%以上の利益
        ]
        
        if not profitable_inferences:
            return "高利益パターンなし"
        
        # 簡易パターン分析
        patterns = []
        for inference in profitable_inferences[-5:]:  # 直近5件
            market_ctx = inference.market_context
            if market_ctx.get('rsi', 50) < 30:
                patterns.append("RSI過売れ状態での買い")
            elif market_ctx.get('trend_direction') == 'up':
                patterns.append("上昇トレンド継続での順張り")
        
        return "; ".join(patterns) if patterns else "パターン分析中"

    async def _get_recent_price_history(self) -> List[float]:
        """直近価格履歴取得"""
        
        try:
            query = """
            SELECT close FROM price_data 
            ORDER BY timestamp DESC 
            LIMIT 100
            """
            
            conn = database_adapter.get_connection()
            cursor = conn.cursor()
            cursor.execute(query)
            results = cursor.fetchall()
            conn.close()
            
            return [float(row[0]) for row in results]
            
        except Exception as e:
            self.logger.error(f"Price history retrieval failed: {e}")
            return []

    def _calculate_technical_indicators(self, price_history: List[float]) -> Dict:
        """テクニカル指標計算"""
        
        if len(price_history) < 20:
            return {}
        
        try:
            prices = np.array(price_history)
            
            # RSI
            delta = np.diff(prices)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = np.mean(gain[-14:])
            avg_loss = np.mean(loss[-14:])
            rs = avg_gain / avg_loss if avg_loss > 0 else 100
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = pd.Series(prices).ewm(span=12).mean().iloc[-1]
            ema_26 = pd.Series(prices).ewm(span=26).mean().iloc[-1]
            macd = ema_12 - ema_26
            
            return {
                'rsi': float(rsi),
                'macd': float(macd),
                'current_price': float(prices[-1]),
                'sma_20': float(np.mean(prices[-20:]))
            }
            
        except Exception as e:
            self.logger.error(f"Technical indicator calculation failed: {e}")
            return {}

    async def _enhance_market_context(self, market_data: Dict) -> Dict:
        """市場コンテキスト強化"""
        
        context = {
            'timestamp': datetime.now(),
            'trading_session': self._get_trading_session(),
            'volatility_regime': self._assess_volatility_regime(market_data),
            'liquidity_assessment': self._assess_liquidity(market_data)
        }
        
        return context

    def _get_trading_session(self) -> str:
        """取引セッション判定"""
        hour = datetime.now().hour
        if 8 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        else:
            return "evening"

    def _assess_volatility_regime(self, market_data: Dict) -> str:
        """ボラティリティレジーム評価"""
        volatility = market_data.get('volatility', 0.01)
        if volatility > 0.03:
            return "high"
        elif volatility > 0.015:
            return "medium"
        else:
            return "low"

    def _assess_liquidity(self, market_data: Dict) -> str:
        """流動性評価"""
        volume = market_data.get('volume', 1000)
        avg_volume = market_data.get('avg_volume', 1000)
        
        if volume > avg_volume * 1.5:
            return "high"
        elif volume > avg_volume * 0.8:
            return "normal"
        else:
            return "low"

    def _update_performance_tracking(self):
        """パフォーマンス追跡更新"""
        
        self.performance_metrics['total_trades'] = len(self.production_history)
        self.performance_metrics['last_update'] = datetime.now()
        
        # 詳細パフォーマンス計算は実際の取引結果が必要

    def get_production_status(self) -> Dict[str, Any]:
        """実運用ステータス取得"""
        
        return {
            'learning_status': self.learning_engine.get_learning_summary(),
            'production_config': self.production_config,
            'daily_trade_count': self.daily_trade_count,
            'total_decisions': len(self.production_history),
            'performance_metrics': self.performance_metrics,
            'last_decision': self.production_history[-1] if self.production_history else None,
            'ready_for_trading': self._is_learning_ready() and self.daily_trade_count < self.production_config['max_daily_trades']
        }

# グローバルインスタンス
production_inference_engine = ProductionInferenceEngine()