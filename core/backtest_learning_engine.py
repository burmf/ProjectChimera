"""
Backtest Learning Engine
バックテストデータからの学習・推論・フィードバックシステム

Purpose:
1. 過去のバックテストデータから推論の妥当性を学習
2. プロンプト送信 → AI推論受信 → 実績検証のループ実行
3. 学習した推論パターンを次回投資判断に活用
4. 利益目標達成まで学習を継続してから実運用開始
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import sqlite3
from pathlib import Path

# AI/ML imports
from core.ai_manager import ai_manager
from core.database_adapter import database_adapter
from core.alpha_integration import integration_engine

@dataclass
class InferenceResult:
    """推論結果データクラス"""
    timestamp: datetime
    market_context: Dict[str, Any]
    ai_prediction: Dict[str, Any]
    actual_outcome: Optional[float] = None
    prediction_accuracy: Optional[float] = None
    profit_impact: Optional[float] = None
    inference_id: str = ""
    
@dataclass
class LearningMetrics:
    """学習指標データクラス"""
    total_inferences: int
    correct_predictions: int
    accuracy_rate: float
    cumulative_profit: float
    sharpe_ratio: float
    max_drawdown: float
    profitable_periods: int
    learning_confidence: float

class BacktestLearningEngine:
    """バックテストデータ学習エンジン"""
    
    def __init__(self):
        """初期化"""
        self.logger = logging.getLogger(__name__)
        
        # 学習設定
        self.learning_config = {
            'target_accuracy': 0.65,        # 目標予測精度 65%
            'target_profit': 0.10,          # 目標累計利益 10%
            'min_learning_samples': 100,    # 最小学習サンプル数
            'max_learning_iterations': 1000, # 最大学習反復回数
            'feedback_window_days': 30,     # フィードバック期間
            'confidence_threshold': 0.75    # 実運用開始信頼度閾値
        }
        
        # AI推論プロンプトテンプレート
        self.inference_prompts = {
            'market_analysis': self._create_market_analysis_prompt,
            'trade_decision': self._create_trade_decision_prompt,
            'risk_assessment': self._create_risk_assessment_prompt,
            'timing_optimization': self._create_timing_prompt
        }
        
        # 学習データストレージ
        self.inference_history: List[InferenceResult] = []
        self.learning_metrics = LearningMetrics(0, 0, 0.0, 0.0, 0.0, 0.0, 0, 0.0)
        
        # パフォーマンス追跡
        self.performance_tracker = {
            'daily_returns': [],
            'prediction_errors': [],
            'strategy_effectiveness': {},
            'learning_curve': []
        }

    async def start_learning_cycle(self, backtest_data: pd.DataFrame) -> bool:
        """
        学習サイクル開始
        
        Args:
            backtest_data: バックテストデータ (OHLCV + indicators)
            
        Returns:
            学習成功フラグ
        """
        try:
            self.logger.info("Starting backtest learning cycle...")
            
            # バックテストデータの前処理
            processed_data = self._preprocess_backtest_data(backtest_data)
            
            # 学習ループ実行
            learning_success = await self._execute_learning_loop(processed_data)
            
            if learning_success:
                # 学習結果保存
                await self._save_learning_results()
                self.logger.info(f"Learning completed successfully. "
                               f"Accuracy: {self.learning_metrics.accuracy_rate:.2%}, "
                               f"Profit: {self.learning_metrics.cumulative_profit:.2%}")
                return True
            else:
                self.logger.warning("Learning did not meet target criteria")
                return False
                
        except Exception as e:
            self.logger.error(f"Learning cycle failed: {e}")
            return False

    async def _execute_learning_loop(self, data: pd.DataFrame) -> bool:
        """学習ループ実行"""
        
        iteration = 0
        learning_window_size = 50  # 学習ウィンドウサイズ
        
        while iteration < self.learning_config['max_learning_iterations']:
            iteration += 1
            
            # 学習ウィンドウ選択
            start_idx = max(0, len(data) - learning_window_size - iteration)
            end_idx = len(data) - iteration if iteration < len(data) else len(data)
            
            if start_idx >= end_idx:
                break
                
            window_data = data.iloc[start_idx:end_idx]
            
            # バッチ推論実行
            batch_results = await self._execute_inference_batch(window_data)
            
            # 結果検証とフィードバック
            await self._validate_and_feedback(batch_results, window_data)
            
            # 学習進捗評価
            if await self._evaluate_learning_progress():
                self.logger.info(f"Learning target achieved at iteration {iteration}")
                return True
                
            # 進捗ログ
            if iteration % 10 == 0:
                self._log_learning_progress(iteration)
        
        # 最終評価
        return self._meets_target_criteria()

    async def _execute_inference_batch(self, data: pd.DataFrame) -> List[InferenceResult]:
        """推論バッチ実行"""
        
        batch_results = []
        
        for idx in range(len(data) - 1):  # 最後の行は予測対象として除外
            try:
                current_data = data.iloc[idx]
                future_data = data.iloc[idx + 1] if idx + 1 < len(data) else None
                
                # 市場コンテキスト準備
                market_context = self._prepare_market_context(current_data, data.iloc[max(0, idx-20):idx+1])
                
                # AI推論実行（複数プロンプト並行処理）
                inference_tasks = [
                    self._execute_single_inference(prompt_type, market_context)
                    for prompt_type in self.inference_prompts.keys()
                ]
                
                inference_results = await asyncio.gather(*inference_tasks, return_exceptions=True)
                
                # 推論結果統合
                consolidated_inference = self._consolidate_inferences(inference_results)
                
                # 実際の結果計算（将来データがある場合）
                actual_outcome = None
                if future_data is not None:
                    actual_outcome = self._calculate_actual_outcome(current_data, future_data)
                
                # 推論結果記録
                result = InferenceResult(
                    timestamp=current_data.get('timestamp', datetime.now()),
                    market_context=market_context,
                    ai_prediction=consolidated_inference,
                    actual_outcome=actual_outcome,
                    inference_id=f"inf_{idx}_{iteration}"
                )
                
                batch_results.append(result)
                
            except Exception as e:
                self.logger.error(f"Inference failed for index {idx}: {e}")
                continue
        
        return batch_results

    async def _execute_single_inference(self, prompt_type: str, market_context: Dict) -> Dict:
        """単一推論実行"""
        
        try:
            # プロンプト生成
            prompt_generator = self.inference_prompts[prompt_type]
            prompt = prompt_generator(market_context)
            
            # AI推論実行
            ai_response = await ai_manager.get_trading_decision_async(
                news_content=json.dumps(market_context),
                custom_prompt=prompt
            )
            
            return {
                'type': prompt_type,
                'response': ai_response,
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Single inference failed for {prompt_type}: {e}")
            return {
                'type': prompt_type,
                'response': None,
                'success': False,
                'error': str(e)
            }

    def _create_market_analysis_prompt(self, market_context: Dict) -> str:
        """市場分析プロンプト生成"""
        
        return f"""
市場分析タスク:

現在の市場状況:
- 価格: {market_context.get('current_price', 'N/A')}
- ボラティリティ: {market_context.get('volatility', 'N/A'):.4f}
- RSI: {market_context.get('rsi', 'N/A'):.2f}
- MACD: {market_context.get('macd', 'N/A'):.4f}
- トレンド強度: {market_context.get('trend_strength', 'N/A'):.4f}

テクニカル指標:
{json.dumps(market_context.get('technical_indicators', {}), indent=2)}

過去20期間の価格動向:
{market_context.get('price_history_summary', 'データなし')}

質問:
1. 現在の市場レジームは何ですか？（トレンド・レンジ・高ボラティリティ・低ボラティリティ）
2. 次の1-5期間で価格はどう動くと予想しますか？（上昇・下降・横ばい）
3. その予想の根拠となる主要要因を3つ挙げてください
4. 予想の信頼度を0-100%で評価してください

JSON形式で回答してください:
{{
    "market_regime": "レジーム名",
    "price_prediction": "方向性", 
    "confidence": 信頼度数値,
    "key_factors": ["要因1", "要因2", "要因3"],
    "expected_magnitude": 予想変動幅,
    "time_horizon": 時間軸
}}
"""

    def _create_trade_decision_prompt(self, market_context: Dict) -> str:
        """取引判断プロンプト生成"""
        
        return f"""
取引判断タスク:

現在の市場データ:
{json.dumps(market_context, indent=2)}

過去の学習から得られた洞察:
- 現在の精度: {self.learning_metrics.accuracy_rate:.2%}
- 累計収益: {self.learning_metrics.cumulative_profit:.2%}
- 成功パターン: {self._get_successful_patterns()}

タスク:
この市場状況で最適な取引行動を決定してください。

考慮事項:
1. エントリーすべきか、様子見すべきか
2. エントリーする場合のサイズ（0-100%）
3. ストップロス・テイクプロフィット価格
4. 予想保有期間

JSON形式で回答:
{{
    "action": "buy/sell/hold",
    "position_size": サイズ％,
    "entry_price": エントリー価格,
    "stop_loss": ストップロス価格,
    "take_profit": テイクプロフィット価格,
    "holding_period": 保有期間（分）,
    "rationale": "判断理由",
    "risk_reward": リスクリワード比率
}}
"""

    def _create_risk_assessment_prompt(self, market_context: Dict) -> str:
        """リスク評価プロンプト生成"""
        
        return f"""
リスク評価タスク:

市場コンテキスト:
{json.dumps(market_context, indent=2)}

現在のポートフォリオ状況:
- 累計収益: {self.learning_metrics.cumulative_profit:.2%}
- 最大ドローダウン: {self.learning_metrics.max_drawdown:.2%}
- 直近の勝率: {self._get_recent_win_rate():.2%}

リスク要因を評価してください:

1. 市場リスク（ボラティリティ、流動性）
2. システマティックリスク（相関、レジーム変化）  
3. 実行リスク（スリッページ、遅延）
4. モデルリスク（オーバーフィッティング、データ品質）

JSON形式で回答:
{{
    "overall_risk": "低・中・高",
    "risk_score": 1-10のスコア,
    "risk_factors": {{
        "market_risk": スコア,
        "systematic_risk": スコア,
        "execution_risk": スコア,
        "model_risk": スコア
    }},
    "recommended_actions": ["アクション1", "アクション2"],
    "position_sizing_adjustment": 調整係数
}}
"""

    def _create_timing_prompt(self, market_context: Dict) -> str:
        """タイミング最適化プロンプト"""
        
        return f"""
タイミング最適化タスク:

現在時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
市場データ: {json.dumps(market_context, indent=2)}

時間要因:
- 時間帯: {datetime.now().hour}時
- 曜日: {datetime.now().strftime('%A')}
- 月: {datetime.now().month}月

質問:
1. 現在は取引に適した時間帯ですか？
2. より良いエントリータイミングはいつですか？
3. 流動性・ボラティリティの観点から最適な実行戦略は？

JSON形式で回答:
{{
    "timing_quality": "最適・良好・普通・悪い",
    "optimal_entry_time": "即座・30分後・1時間後・翌日",
    "execution_strategy": "成行・指値・TWAP・VWAP",
    "expected_slippage": 予想スリッページbps,
    "liquidity_assessment": "高・中・低"
}}
"""

    def _prepare_market_context(self, current_data: pd.Series, historical_data: pd.DataFrame) -> Dict:
        """市場コンテキスト準備"""
        
        try:
            context = {
                'current_price': float(current_data.get('close', 0)),
                'timestamp': current_data.get('timestamp', datetime.now()),
                'volatility': float(historical_data['close'].pct_change().std()) if len(historical_data) > 1 else 0.01,
            }
            
            # テクニカル指標追加
            if 'rsi' in current_data:
                context['rsi'] = float(current_data['rsi'])
            if 'macd' in current_data:
                context['macd'] = float(current_data['macd'])
            if 'volume' in current_data:
                context['volume'] = float(current_data['volume'])
                
            # 価格履歴サマリー
            if len(historical_data) >= 5:
                price_changes = historical_data['close'].pct_change().tail(5)
                context['price_history_summary'] = {
                    'recent_returns': price_changes.tolist(),
                    'trend_direction': 'up' if price_changes.mean() > 0 else 'down',
                    'volatility_regime': 'high' if price_changes.std() > 0.02 else 'normal'
                }
            
            # トレンド強度
            if len(historical_data) >= 20:
                sma_20 = historical_data['close'].tail(20).mean()
                context['trend_strength'] = (context['current_price'] - sma_20) / sma_20
            
            return context
            
        except Exception as e:
            self.logger.error(f"Market context preparation failed: {e}")
            return {'current_price': 0, 'timestamp': datetime.now()}

    def _consolidate_inferences(self, inference_results: List[Dict]) -> Dict:
        """推論結果統合"""
        
        consolidated = {
            'market_analysis': None,
            'trade_decision': None,
            'risk_assessment': None,
            'timing_optimization': None,
            'overall_confidence': 0.0
        }
        
        confidence_scores = []
        
        for result in inference_results:
            if isinstance(result, Exception):
                continue
                
            if result.get('success', False):
                prompt_type = result['type']
                response = result['response']
                
                consolidated[prompt_type] = response
                
                # 信頼度抽出
                if response and 'confidence' in str(response):
                    try:
                        confidence = float(str(response).split('confidence')[1].split(',')[0].strip(': "'))
                        confidence_scores.append(confidence)
                    except:
                        pass
        
        # 全体信頼度計算
        if confidence_scores:
            consolidated['overall_confidence'] = np.mean(confidence_scores) / 100.0
        
        return consolidated

    def _calculate_actual_outcome(self, current_data: pd.Series, future_data: pd.Series) -> float:
        """実際の結果計算"""
        
        try:
            current_price = float(current_data.get('close', 0))
            future_price = float(future_data.get('close', 0))
            
            if current_price > 0:
                return (future_price - current_price) / current_price
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Actual outcome calculation failed: {e}")
            return 0.0

    async def _validate_and_feedback(self, batch_results: List[InferenceResult], data: pd.DataFrame):
        """推論結果検証とフィードバック"""
        
        for result in batch_results:
            if result.actual_outcome is not None:
                # 予測精度計算
                accuracy = self._calculate_prediction_accuracy(result)
                result.prediction_accuracy = accuracy
                
                # 利益インパクト計算
                profit_impact = self._calculate_profit_impact(result)
                result.profit_impact = profit_impact
                
                # 履歴に追加
                self.inference_history.append(result)
                
                # パフォーマンストラッキング更新
                self._update_performance_tracking(result)
        
        # 学習指標更新
        self._update_learning_metrics()

    def _calculate_prediction_accuracy(self, result: InferenceResult) -> float:
        """予測精度計算"""
        
        try:
            ai_prediction = result.ai_prediction.get('trade_decision', {})
            actual_outcome = result.actual_outcome
            
            if not ai_prediction or actual_outcome is None:
                return 0.0
            
            # 方向性予測の精度
            predicted_direction = ai_prediction.get('action', 'hold')
            actual_direction = 'buy' if actual_outcome > 0.001 else ('sell' if actual_outcome < -0.001 else 'hold')
            
            direction_correct = predicted_direction == actual_direction
            
            # 大きさ予測の精度
            magnitude_accuracy = 0.0
            if 'expected_magnitude' in str(ai_prediction):
                try:
                    expected_mag = float(str(ai_prediction).split('expected_magnitude')[1].split(',')[0].strip(': "'))
                    magnitude_accuracy = 1.0 - min(abs(expected_mag - abs(actual_outcome)), 1.0)
                except:
                    magnitude_accuracy = 0.5
            
            # 統合精度（方向70%、大きさ30%）
            return 0.7 * (1.0 if direction_correct else 0.0) + 0.3 * magnitude_accuracy
            
        except Exception as e:
            self.logger.error(f"Accuracy calculation failed: {e}")
            return 0.0

    def _calculate_profit_impact(self, result: InferenceResult) -> float:
        """利益インパクト計算"""
        
        try:
            trade_decision = result.ai_prediction.get('trade_decision', {})
            actual_outcome = result.actual_outcome
            
            if not trade_decision or actual_outcome is None:
                return 0.0
            
            action = trade_decision.get('action', 'hold')
            position_size = float(trade_decision.get('position_size', 0)) / 100.0
            
            if action == 'buy':
                return actual_outcome * position_size
            elif action == 'sell':
                return -actual_outcome * position_size
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Profit impact calculation failed: {e}")
            return 0.0

    def _update_learning_metrics(self):
        """学習指標更新"""
        
        if not self.inference_history:
            return
        
        # 有効な推論結果のみフィルタ
        valid_results = [r for r in self.inference_history if r.prediction_accuracy is not None]
        
        if not valid_results:
            return
        
        # 基本指標計算
        total_inferences = len(valid_results)
        correct_predictions = sum(1 for r in valid_results if r.prediction_accuracy > 0.5)
        accuracy_rate = correct_predictions / total_inferences if total_inferences > 0 else 0.0
        
        # 利益指標計算
        profit_impacts = [r.profit_impact for r in valid_results if r.profit_impact is not None]
        cumulative_profit = sum(profit_impacts) if profit_impacts else 0.0
        
        # シャープレシオ計算
        if len(profit_impacts) > 1:
            sharpe_ratio = np.mean(profit_impacts) / np.std(profit_impacts) if np.std(profit_impacts) > 0 else 0.0
        else:
            sharpe_ratio = 0.0
        
        # 最大ドローダウン計算
        if profit_impacts:
            cumsum = np.cumsum(profit_impacts)
            running_max = np.maximum.accumulate(cumsum)
            drawdowns = cumsum - running_max
            max_drawdown = abs(min(drawdowns)) if len(drawdowns) > 0 else 0.0
        else:
            max_drawdown = 0.0
        
        # 利益期間計算
        profitable_periods = sum(1 for p in profit_impacts if p > 0)
        
        # 学習信頼度計算
        learning_confidence = min(
            accuracy_rate * 1.5,  # 精度ボーナス
            (profitable_periods / len(profit_impacts)) * 1.2 if profit_impacts else 0,  # 利益率ボーナス
            1.0
        )
        
        # 指標更新
        self.learning_metrics = LearningMetrics(
            total_inferences=total_inferences,
            correct_predictions=correct_predictions,
            accuracy_rate=accuracy_rate,
            cumulative_profit=cumulative_profit,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            profitable_periods=profitable_periods,
            learning_confidence=learning_confidence
        )

    async def _evaluate_learning_progress(self) -> bool:
        """学習進捗評価"""
        
        return self._meets_target_criteria()

    def _meets_target_criteria(self) -> bool:
        """目標基準達成判定"""
        
        config = self.learning_config
        metrics = self.learning_metrics
        
        criteria_met = [
            metrics.total_inferences >= config['min_learning_samples'],
            metrics.accuracy_rate >= config['target_accuracy'],
            metrics.cumulative_profit >= config['target_profit'],
            metrics.learning_confidence >= config['confidence_threshold']
        ]
        
        return all(criteria_met)

    def _preprocess_backtest_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """バックテストデータ前処理"""
        
        try:
            # データクリーニング
            data = data.dropna()
            
            # タイムスタンプ確保
            if 'timestamp' not in data.columns:
                data['timestamp'] = pd.date_range(end=datetime.now(), periods=len(data), freq='1H')
            
            # 基本テクニカル指標計算（存在しない場合）
            if 'rsi' not in data.columns:
                data['rsi'] = self._calculate_rsi(data['close'])
            if 'macd' not in data.columns:
                data['macd'] = self._calculate_macd(data['close'])
            
            self.logger.info(f"Preprocessed {len(data)} data points for learning")
            return data
            
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {e}")
            return data

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """RSI計算"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        """MACD計算"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        return ema_fast - ema_slow

    def _get_successful_patterns(self) -> str:
        """成功パターン抽出"""
        
        if not self.inference_history:
            return "学習データ不足"
        
        successful_results = [r for r in self.inference_history if r.profit_impact and r.profit_impact > 0]
        
        if not successful_results:
            return "成功パターンなし"
        
        # 簡易パターン分析
        patterns = []
        for result in successful_results[-10:]:  # 直近10件
            market_ctx = result.market_context
            if market_ctx.get('trend_direction') == 'up' and result.profit_impact > 0.01:
                patterns.append("上昇トレンド時の買い")
        
        return ", ".join(patterns) if patterns else "パターン分析中"

    def _get_recent_win_rate(self) -> float:
        """直近勝率取得"""
        
        if not self.inference_history:
            return 0.0
        
        recent_results = self.inference_history[-20:]  # 直近20件
        profitable_trades = [r for r in recent_results if r.profit_impact and r.profit_impact > 0]
        
        return len(profitable_trades) / len(recent_results) if recent_results else 0.0

    def _update_performance_tracking(self, result: InferenceResult):
        """パフォーマンス追跡更新"""
        
        if result.profit_impact is not None:
            self.performance_tracker['daily_returns'].append(result.profit_impact)
        
        if result.prediction_accuracy is not None:
            prediction_error = 1.0 - result.prediction_accuracy
            self.performance_tracker['prediction_errors'].append(prediction_error)

    def _log_learning_progress(self, iteration: int):
        """学習進捗ログ"""
        
        metrics = self.learning_metrics
        self.logger.info(
            f"Learning Progress [{iteration}]: "
            f"Accuracy: {metrics.accuracy_rate:.2%}, "
            f"Profit: {metrics.cumulative_profit:.2%}, "
            f"Confidence: {metrics.learning_confidence:.2%}"
        )

    async def _save_learning_results(self):
        """学習結果保存"""
        
        try:
            # データベースに保存
            learning_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': asdict(self.learning_metrics),
                'inference_count': len(self.inference_history),
                'target_criteria_met': self._meets_target_criteria()
            }
            
            # 簡易JSON保存（実装では適切なDB使用）
            import json
            with open('/home/ec2-user/BOT/data/learning_results.json', 'w') as f:
                json.dump(learning_data, f, indent=2)
            
            self.logger.info("Learning results saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save learning results: {e}")

    def get_learning_summary(self) -> Dict[str, Any]:
        """学習サマリー取得"""
        
        return {
            'learning_metrics': asdict(self.learning_metrics),
            'ready_for_production': self._meets_target_criteria(),
            'total_inferences': len(self.inference_history),
            'recent_performance': {
                'win_rate': self._get_recent_win_rate(),
                'successful_patterns': self._get_successful_patterns()
            },
            'target_criteria': self.learning_config,
            'recommendation': self._get_recommendation()
        }

    def _get_recommendation(self) -> str:
        """推奨アクション取得"""
        
        if self._meets_target_criteria():
            return "学習完了 - 実運用開始可能"
        elif self.learning_metrics.total_inferences < self.learning_config['min_learning_samples']:
            return "学習データ不足 - 継続学習が必要"
        elif self.learning_metrics.accuracy_rate < self.learning_config['target_accuracy']:
            return "予測精度不足 - モデル調整が必要"
        elif self.learning_metrics.cumulative_profit < self.learning_config['target_profit']:
            return "収益性不足 - 戦略見直しが必要"
        else:
            return "学習継続中 - もう少し学習が必要"

# グローバルインスタンス
backtest_learning_engine = BacktestLearningEngine()