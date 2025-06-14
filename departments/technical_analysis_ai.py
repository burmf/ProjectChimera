#!/usr/bin/env python3
"""
Technical Analysis AI Department
テクニカル分析部門AI
"""

import numpy as np
import pandas as pd
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import sys
import os

# Add core modules to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ai_agent_base import AIAgentBase, DepartmentType, AnalysisRequest, AnalysisResult
from core.department_prompts import DepartmentPrompts, PromptFormatter


class TechnicalAnalysisAI(AIAgentBase):
    """
    テクニカル分析専門AIエージェント
    
    チャートパターン、テクニカル指標、価格アクション分析を専門とする
    """
    
    def __init__(self, model_config: Dict[str, Any] = None):
        super().__init__(DepartmentType.TECHNICAL, model_config)
        
        # テクニカル分析固有の設定
        self.lookback_periods = {
            'short': 14,    # RSI等の短期指標
            'medium': 50,   # 中期移動平均
            'long': 200     # 長期移動平均
        }
        
        # パターン認識の閾値
        self.pattern_thresholds = {
            'support_resistance_strength': 0.02,  # 2%以内でサポレジ認識
            'trend_strength_threshold': 0.7,      # トレンド強度閾値
            'breakout_volume_multiplier': 1.5,    # ブレイクアウト時の出来高倍率
            'divergence_lookback': 20             # ダイバージェンス検出期間
        }
        
        self.logger.info("Technical Analysis AI initialized")
    
    def _get_system_prompt(self) -> str:
        """テクニカル分析専用システムプロンプトを取得"""
        return DepartmentPrompts.get_system_prompt(DepartmentType.TECHNICAL)
    
    def _validate_request_data(self, request: AnalysisRequest) -> bool:
        """リクエストデータの妥当性チェック"""
        try:
            data = request.data
            
            # 必須データの存在確認
            if 'price_data' not in data:
                self.logger.error("Price data is required for technical analysis")
                return False
            
            price_data = data['price_data']
            required_fields = ['open', 'high', 'low', 'close']
            
            for field in required_fields:
                if field not in price_data:
                    self.logger.error(f"Missing required price field: {field}")
                    return False
            
            # 価格データの妥当性チェック
            if not self._validate_price_data(price_data):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation failed: {e}")
            return False
    
    def _validate_price_data(self, price_data: Dict[str, Any]) -> bool:
        """価格データの妥当性をチェック"""
        try:
            # 基本的な価格関係をチェック
            high = float(price_data['high'])
            low = float(price_data['low'])
            open_price = float(price_data['open'])
            close = float(price_data['close'])
            
            # 高値 >= 安値
            if high < low:
                self.logger.error("High price is less than low price")
                return False
            
            # 始値・終値が高値・安値の範囲内
            if not (low <= open_price <= high and low <= close <= high):
                self.logger.error("Open/Close price outside High/Low range")
                return False
            
            # 価格が正数
            if any(price <= 0 for price in [high, low, open_price, close]):
                self.logger.error("Negative or zero prices detected")
                return False
            
            return True
            
        except (ValueError, KeyError) as e:
            self.logger.error(f"Price data validation error: {e}")
            return False
    
    async def _analyze_data(self, request: AnalysisRequest) -> Dict[str, Any]:
        """テクニカル分析のメインロジック"""
        try:
            data = request.data
            price_data = data['price_data']
            technical_indicators = data.get('technical_indicators', {})
            
            # 価格データをDataFrameに変換（履歴がある場合）
            df = self._prepare_price_dataframe(price_data)
            
            # テクニカル指標の計算・補完
            indicators = self._calculate_technical_indicators(df, technical_indicators)
            
            # チャートパターンの認識
            patterns = self._recognize_chart_patterns(df, indicators)
            
            # トレンド分析
            trend_analysis = self._analyze_trend(df, indicators)
            
            # サポート・レジスタンス分析
            support_resistance = self._analyze_support_resistance(df)
            
            # モメンタム分析
            momentum_analysis = self._analyze_momentum(indicators)
            
            # 総合シグナル生成
            signal = self._generate_trading_signal(
                trend_analysis, momentum_analysis, patterns, 
                support_resistance, indicators
            )
            
            # リスクレベル評価
            risk_level = self._assess_technical_risk(df, indicators, signal)
            
            return {
                'trend': trend_analysis['direction'],
                'trend_strength': trend_analysis['strength'],
                'momentum': momentum_analysis['direction'],
                'key_levels': support_resistance,
                'patterns': patterns,
                'indicators': indicators,
                'action': signal['action'],
                'confidence': signal['confidence'],
                'reasoning': signal['reasoning'],
                'key_factors': signal['key_factors'],
                'time_horizon': signal['time_horizon'],
                'risk_level': risk_level
            }
            
        except Exception as e:
            self.logger.error(f"Technical analysis failed: {e}")
            return {
                'error': str(e),
                'action': 'hold',
                'confidence': 0.0,
                'reasoning': f'Analysis failed: {str(e)}'
            }
    
    def _prepare_price_dataframe(self, price_data: Dict[str, Any]) -> pd.DataFrame:
        """価格データをDataFrameに変換"""
        try:
            # 単一の価格データの場合
            if isinstance(price_data, dict) and 'close' in price_data:
                # シンプルなOHLCデータの場合
                return pd.DataFrame([{
                    'open': float(price_data['open']),
                    'high': float(price_data['high']),
                    'low': float(price_data['low']),
                    'close': float(price_data['close']),
                    'volume': float(price_data.get('volume', 0))
                }])
            
            # 時系列データの場合
            elif isinstance(price_data, list):
                df = pd.DataFrame(price_data)
                if 'timestamp' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index('timestamp', inplace=True)
                return df
            
            else:
                # フォールバック：辞書から直接変換を試行
                return pd.DataFrame([price_data])
                
        except Exception as e:
            self.logger.warning(f"Price data conversion failed: {e}")
            # 最小限のダミーデータ
            return pd.DataFrame([{
                'open': 100, 'high': 101, 'low': 99, 'close': 100.5, 'volume': 1000
            }])
    
    def _calculate_technical_indicators(
        self, 
        df: pd.DataFrame, 
        existing_indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """テクニカル指標の計算"""
        indicators = existing_indicators.copy()
        
        try:
            close_prices = df['close'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            
            # RSI計算
            if 'rsi' not in indicators:
                rsi_value = self._calculate_rsi(close_prices)
                indicators['rsi'] = {
                    'value': rsi_value,
                    'signal': self._classify_rsi_signal(rsi_value)
                }
            
            # MACD計算
            if 'macd' not in indicators:
                macd_signal = self._calculate_macd_signal(close_prices)
                indicators['macd'] = {
                    'signal': macd_signal
                }
            
            # 移動平均計算
            if 'moving_averages' not in indicators:
                ma_signal = self._calculate_moving_average_signal(close_prices)
                indicators['moving_averages'] = {
                    'signal': ma_signal
                }
            
            # ボリンジャーバンド
            if 'bollinger_bands' not in indicators:
                bb_signal = self._calculate_bollinger_signal(close_prices)
                indicators['bollinger_bands'] = bb_signal
            
            # ストキャスティクス
            if 'stochastic' not in indicators:
                stoch_value = self._calculate_stochastic(high_prices, low_prices, close_prices)
                indicators['stochastic'] = {
                    'value': stoch_value,
                    'signal': self._classify_stochastic_signal(stoch_value)
                }
            
        except Exception as e:
            self.logger.error(f"Technical indicator calculation failed: {e}")
        
        return indicators
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """RSI計算"""
        if len(prices) < period + 1:
            return 50.0  # 中立値
        
        try:
            deltas = np.diff(prices)
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gains[-period:])
            avg_loss = np.mean(losses[-period:])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi)
            
        except Exception:
            return 50.0
    
    def _classify_rsi_signal(self, rsi: float) -> str:
        """RSIシグナル分類"""
        if rsi > 70:
            return 'overbought'
        elif rsi < 30:
            return 'oversold'
        else:
            return 'neutral'
    
    def _calculate_macd_signal(self, prices: np.ndarray) -> str:
        """MACD信号計算（簡略版）"""
        if len(prices) < 26:
            return 'neutral'
        
        try:
            # 簡易MACD計算
            ema12 = self._calculate_ema(prices, 12)
            ema26 = self._calculate_ema(prices, 26)
            
            macd_line = ema12[-1] - ema26[-1]
            prev_macd = ema12[-2] - ema26[-2] if len(ema12) > 1 else macd_line
            
            if macd_line > 0 and prev_macd <= 0:
                return 'bullish_cross'
            elif macd_line < 0 and prev_macd >= 0:
                return 'bearish_cross'
            else:
                return 'neutral'
                
        except Exception:
            return 'neutral'
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> np.ndarray:
        """EMA計算"""
        alpha = 2.0 / (period + 1)
        ema = np.zeros(len(prices))
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema
    
    def _calculate_moving_average_signal(self, prices: np.ndarray) -> str:
        """移動平均クロス信号"""
        if len(prices) < 50:
            return 'neutral'
        
        try:
            ma20 = np.mean(prices[-20:])
            ma50 = np.mean(prices[-50:])
            
            prev_ma20 = np.mean(prices[-21:-1])
            prev_ma50 = np.mean(prices[-51:-1])
            
            # ゴールデンクロス・デッドクロス判定
            if ma20 > ma50 and prev_ma20 <= prev_ma50:
                return 'golden_cross'
            elif ma20 < ma50 and prev_ma20 >= prev_ma50:
                return 'death_cross'
            else:
                return 'neutral'
                
        except Exception:
            return 'neutral'
    
    def _calculate_bollinger_signal(self, prices: np.ndarray, period: int = 20) -> Dict[str, Any]:
        """ボリンジャーバンド信号"""
        if len(prices) < period:
            return {'signal': 'neutral', 'position': 'middle'}
        
        try:
            ma = np.mean(prices[-period:])
            std = np.std(prices[-period:])
            
            upper_band = ma + 2 * std
            lower_band = ma - 2 * std
            current_price = prices[-1]
            
            if current_price > upper_band:
                return {'signal': 'overbought', 'position': 'upper'}
            elif current_price < lower_band:
                return {'signal': 'oversold', 'position': 'lower'}
            else:
                return {'signal': 'neutral', 'position': 'middle'}
                
        except Exception:
            return {'signal': 'neutral', 'position': 'middle'}
    
    def _calculate_stochastic(
        self, 
        highs: np.ndarray, 
        lows: np.ndarray, 
        closes: np.ndarray, 
        period: int = 14
    ) -> float:
        """ストキャスティクス計算"""
        if len(closes) < period:
            return 50.0
        
        try:
            recent_high = np.max(highs[-period:])
            recent_low = np.min(lows[-period:])
            current_close = closes[-1]
            
            if recent_high == recent_low:
                return 50.0
            
            stoch = 100 * (current_close - recent_low) / (recent_high - recent_low)
            return float(stoch)
            
        except Exception:
            return 50.0
    
    def _classify_stochastic_signal(self, stoch: float) -> str:
        """ストキャスティクス信号分類"""
        if stoch > 80:
            return 'overbought'
        elif stoch < 20:
            return 'oversold'
        else:
            return 'neutral'
    
    def _recognize_chart_patterns(
        self, 
        df: pd.DataFrame, 
        indicators: Dict[str, Any]
    ) -> List[str]:
        """チャートパターン認識"""
        patterns = []
        
        if len(df) < 10:
            return patterns
        
        try:
            prices = df['close'].values
            highs = df['high'].values
            lows = df['low'].values
            
            # ダブルトップ・ダブルボトムの検出
            if self._detect_double_top(highs):
                patterns.append('double_top')
            
            if self._detect_double_bottom(lows):
                patterns.append('double_bottom')
            
            # トレンドライン突破の検出
            if self._detect_trendline_breakout(prices):
                patterns.append('trendline_breakout')
            
            # 三角保ち合いの検出
            if self._detect_triangle_pattern(highs, lows):
                patterns.append('triangle_consolidation')
            
            # フラッグ・ペナント
            if self._detect_flag_pattern(prices):
                patterns.append('flag_pattern')
            
        except Exception as e:
            self.logger.error(f"Pattern recognition failed: {e}")
        
        return patterns
    
    def _detect_double_top(self, highs: np.ndarray) -> bool:
        """ダブルトップパターンの検出"""
        if len(highs) < 20:
            return False
        
        try:
            # 直近20期間での高値を分析
            recent_highs = highs[-20:]
            
            # 局所的な高値を検出
            peaks = []
            for i in range(1, len(recent_highs) - 1):
                if recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i+1]:
                    peaks.append((i, recent_highs[i]))
            
            if len(peaks) >= 2:
                # 最後の2つの高値を比較
                peak1_price = peaks[-2][1]
                peak2_price = peaks[-1][1]
                
                # 2つの高値がほぼ同じレベル（2%以内）
                if abs(peak1_price - peak2_price) / peak1_price < 0.02:
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _detect_double_bottom(self, lows: np.ndarray) -> bool:
        """ダブルボトムパターンの検出"""
        if len(lows) < 20:
            return False
        
        try:
            recent_lows = lows[-20:]
            
            # 局所的な安値を検出
            troughs = []
            for i in range(1, len(recent_lows) - 1):
                if recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i+1]:
                    troughs.append((i, recent_lows[i]))
            
            if len(troughs) >= 2:
                trough1_price = troughs[-2][1]
                trough2_price = troughs[-1][1]
                
                if abs(trough1_price - trough2_price) / trough1_price < 0.02:
                    return True
            
            return False
            
        except Exception:
            return False
    
    def _detect_trendline_breakout(self, prices: np.ndarray) -> bool:
        """トレンドライン突破の検出"""
        if len(prices) < 10:
            return False
        
        try:
            # 簡易的なトレンドライン計算
            x = np.arange(len(prices))
            slope, intercept = np.polyfit(x, prices, 1)
            
            # トレンドラインからの乖離度
            trendline_values = slope * x + intercept
            current_deviation = (prices[-1] - trendline_values[-1]) / trendline_values[-1]
            
            # 3%以上の乖離でブレイクアウトと判定
            return abs(current_deviation) > 0.03
            
        except Exception:
            return False
    
    def _detect_triangle_pattern(self, highs: np.ndarray, lows: np.ndarray) -> bool:
        """三角保ち合いパターンの検出"""
        if len(highs) < 15:
            return False
        
        try:
            recent_highs = highs[-15:]
            recent_lows = lows[-15:]
            
            # 高値の傾向（下降トレンド）
            x = np.arange(len(recent_highs))
            high_slope, _ = np.polyfit(x, recent_highs, 1)
            
            # 安値の傾向（上昇トレンド）
            low_slope, _ = np.polyfit(x, recent_lows, 1)
            
            # 三角保ち合い：高値が下降、安値が上昇
            return high_slope < -0.001 and low_slope > 0.001
            
        except Exception:
            return False
    
    def _detect_flag_pattern(self, prices: np.ndarray) -> bool:
        """フラッグパターンの検出"""
        if len(prices) < 20:
            return False
        
        try:
            # 直近の値動きが小さく、その前に大きな動きがあった場合
            recent_range = np.max(prices[-10:]) - np.min(prices[-10:])
            previous_range = np.max(prices[-20:-10]) - np.min(prices[-20:-10])
            
            # フラッグ：直近のレンジが前のレンジの30%以下
            return recent_range < previous_range * 0.3
            
        except Exception:
            return False
    
    def _analyze_trend(self, df: pd.DataFrame, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """トレンド分析"""
        try:
            prices = df['close'].values
            
            if len(prices) < 20:
                return {'direction': 'sideways', 'strength': 0.5}
            
            # 移動平均によるトレンド判定
            ma20 = np.mean(prices[-20:])
            ma50 = np.mean(prices[-50:]) if len(prices) >= 50 else ma20
            current_price = prices[-1]
            
            # トレンドの方向性
            if current_price > ma20 > ma50:
                direction = 'uptrend'
            elif current_price < ma20 < ma50:
                direction = 'downtrend'
            else:
                direction = 'sideways'
            
            # トレンドの強度（価格と移動平均の乖離度で測定）
            if ma20 != 0:
                strength = abs(current_price - ma20) / ma20
                strength = min(strength * 10, 1.0)  # 正規化
            else:
                strength = 0.5
            
            return {
                'direction': direction,
                'strength': float(strength)
            }
            
        except Exception as e:
            self.logger.error(f"Trend analysis failed: {e}")
            return {'direction': 'sideways', 'strength': 0.5}
    
    def _analyze_support_resistance(self, df: pd.DataFrame) -> Dict[str, List[float]]:
        """サポート・レジスタンス分析"""
        try:
            if len(df) < 20:
                return {'support': [], 'resistance': []}
            
            highs = df['high'].values
            lows = df['low'].values
            
            # 局所的な高値・安値を検出
            resistance_levels = []
            support_levels = []
            
            # 高値（レジスタンス）の検出
            for i in range(2, len(highs) - 2):
                if (highs[i] > highs[i-1] and highs[i] > highs[i-2] and 
                    highs[i] > highs[i+1] and highs[i] > highs[i+2]):
                    resistance_levels.append(float(highs[i]))
            
            # 安値（サポート）の検出
            for i in range(2, len(lows) - 2):
                if (lows[i] < lows[i-1] and lows[i] < lows[i-2] and 
                    lows[i] < lows[i+1] and lows[i] < lows[i+2]):
                    support_levels.append(float(lows[i]))
            
            # 重要度でソート（直近のレベルを重視）
            resistance_levels = sorted(resistance_levels, reverse=True)[:3]
            support_levels = sorted(support_levels)[:3]
            
            return {
                'support': support_levels,
                'resistance': resistance_levels
            }
            
        except Exception as e:
            self.logger.error(f"Support/Resistance analysis failed: {e}")
            return {'support': [], 'resistance': []}
    
    def _analyze_momentum(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """モメンタム分析"""
        try:
            rsi = indicators.get('rsi', {})
            macd = indicators.get('macd', {})
            stochastic = indicators.get('stochastic', {})
            
            # モメンタム方向の総合判定
            momentum_signals = []
            
            if rsi.get('signal') == 'oversold':
                momentum_signals.append('bullish')
            elif rsi.get('signal') == 'overbought':
                momentum_signals.append('bearish')
            
            if macd.get('signal') == 'bullish_cross':
                momentum_signals.append('bullish')
            elif macd.get('signal') == 'bearish_cross':
                momentum_signals.append('bearish')
            
            if stochastic.get('signal') == 'oversold':
                momentum_signals.append('bullish')
            elif stochastic.get('signal') == 'overbought':
                momentum_signals.append('bearish')
            
            # 多数決でモメンタム方向を決定
            if momentum_signals.count('bullish') > momentum_signals.count('bearish'):
                direction = 'bullish'
            elif momentum_signals.count('bearish') > momentum_signals.count('bullish'):
                direction = 'bearish'
            else:
                direction = 'neutral'
            
            return {'direction': direction, 'signals': momentum_signals}
            
        except Exception as e:
            self.logger.error(f"Momentum analysis failed: {e}")
            return {'direction': 'neutral', 'signals': []}
    
    def _generate_trading_signal(
        self,
        trend_analysis: Dict[str, Any],
        momentum_analysis: Dict[str, Any],
        patterns: List[str],
        support_resistance: Dict[str, List[float]],
        indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """総合的なトレーディングシグナル生成"""
        
        try:
            signals = []
            confidence_factors = []
            key_factors = []
            
            # トレンド分析からのシグナル
            trend_direction = trend_analysis['direction']
            trend_strength = trend_analysis['strength']
            
            if trend_direction == 'uptrend':
                signals.append('buy')
                confidence_factors.append(trend_strength)
                key_factors.append(f'強い上昇トレンド (強度: {trend_strength:.2f})')
            elif trend_direction == 'downtrend':
                signals.append('sell')
                confidence_factors.append(trend_strength)
                key_factors.append(f'強い下降トレンド (強度: {trend_strength:.2f})')
            
            # モメンタムからのシグナル
            momentum_direction = momentum_analysis['direction']
            if momentum_direction == 'bullish':
                signals.append('buy')
                confidence_factors.append(0.6)
                key_factors.append('ブルリッシュモメンタム')
            elif momentum_direction == 'bearish':
                signals.append('sell')
                confidence_factors.append(0.6)
                key_factors.append('ベアリッシュモメンタム')
            
            # パターンからのシグナル
            for pattern in patterns:
                if pattern in ['double_bottom', 'trendline_breakout']:
                    signals.append('buy')
                    confidence_factors.append(0.7)
                    key_factors.append(f'ブルリッシュパターン: {pattern}')
                elif pattern in ['double_top']:
                    signals.append('sell')
                    confidence_factors.append(0.7)
                    key_factors.append(f'ベアリッシュパターン: {pattern}')
            
            # RSIからのシグナル
            rsi_data = indicators.get('rsi', {})
            if rsi_data.get('signal') == 'oversold':
                signals.append('buy')
                confidence_factors.append(0.5)
                key_factors.append(f'RSI買われすぎ: {rsi_data.get("value", 0):.1f}')
            elif rsi_data.get('signal') == 'overbought':
                signals.append('sell')
                confidence_factors.append(0.5)
                key_factors.append(f'RSI売られすぎ: {rsi_data.get("value", 0):.1f}')
            
            # 最終シグナル決定
            if not signals:
                final_action = 'hold'
                confidence = 0.3
                reasoning = 'テクニカル指標に明確なシグナルなし'
            else:
                # 多数決
                buy_count = signals.count('buy')
                sell_count = signals.count('sell')
                
                if buy_count > sell_count:
                    final_action = 'buy'
                    confidence = min(np.mean(confidence_factors) + 0.1 * (buy_count - sell_count), 1.0)
                elif sell_count > buy_count:
                    final_action = 'sell'
                    confidence = min(np.mean(confidence_factors) + 0.1 * (sell_count - buy_count), 1.0)
                else:
                    final_action = 'hold'
                    confidence = max(np.mean(confidence_factors) - 0.2, 0.2)
                
                reasoning = f'{len(key_factors)}個の要因に基づく総合判断'
            
            # 時間軸の決定
            if any('trend' in factor.lower() for factor in key_factors):
                time_horizon = 'medium'
            elif any('momentum' in factor.lower() for factor in key_factors):
                time_horizon = 'short'
            else:
                time_horizon = 'short'
            
            return {
                'action': final_action,
                'confidence': float(confidence),
                'reasoning': reasoning,
                'key_factors': key_factors[:5],  # 最大5個
                'time_horizon': time_horizon
            }
            
        except Exception as e:
            self.logger.error(f"Signal generation failed: {e}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'reasoning': f'シグナル生成エラー: {str(e)}',
                'key_factors': [],
                'time_horizon': 'short'
            }
    
    def _assess_technical_risk(
        self, 
        df: pd.DataFrame, 
        indicators: Dict[str, Any], 
        signal: Dict[str, Any]
    ) -> str:
        """テクニカルリスクレベルの評価"""
        try:
            risk_factors = []
            
            # ボラティリティによるリスク
            if len(df) >= 20:
                recent_volatility = np.std(df['close'].pct_change().dropna()[-20:])
                if recent_volatility > 0.02:  # 日次2%以上のボラティリティ
                    risk_factors.append('high_volatility')
            
            # RSIの極端な値
            rsi_value = indicators.get('rsi', {}).get('value', 50)
            if rsi_value > 80 or rsi_value < 20:
                risk_factors.append('extreme_rsi')
            
            # トレンドの強度
            if signal.get('confidence', 0) < 0.4:
                risk_factors.append('low_confidence')
            
            # リスクレベル決定
            if len(risk_factors) >= 3:
                return 'high'
            elif len(risk_factors) >= 1:
                return 'medium'
            else:
                return 'low'
                
        except Exception:
            return 'medium'