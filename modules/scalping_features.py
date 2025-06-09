#!/usr/bin/env python3
"""
スキャルピング特化特徴量エンジニアリングシステム
高頻度取引用テクニカル指標とマーケットマイクロストラクチャー特徴量
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

class ScalpingFeatureEngine:
    """スキャルピング用特徴量エンジニアリング"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        self.feature_selector = SelectKBest(f_regression, k=20)
        
        # スキャルピング用パラメータ設定
        self.fast_periods = [3, 5, 8, 13, 21]
        self.volume_periods = [5, 10, 20]
        self.volatility_periods = [5, 10, 15]
        
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """全特徴量を生成"""
        if df.empty or len(df) < 50:
            self.logger.warning("データが不足しています")
            return df
        
        self.logger.info(f"特徴量エンジニアリング開始: {len(df)}レコード")
        
        # 基本的な価格特徴量
        df = self._add_price_features(df)
        
        # 高速テクニカル指標
        df = self._add_fast_technical_indicators(df)
        
        # ボリューム特徴量
        df = self._add_volume_features(df)
        
        # ボラティリティ特徴量
        df = self._add_volatility_features(df)
        
        # マーケットマイクロストラクチャー特徴量
        df = self._add_microstructure_features(df)
        
        # 時間特徴量
        df = self._add_time_features(df)
        
        # ラグ特徴量
        df = self._add_lag_features(df)
        
        # 相互作用特徴量
        df = self._add_interaction_features(df)
        
        # 統計的特徴量
        df = self._add_statistical_features(df)
        
        self.logger.info(f"特徴量生成完了: {df.shape[1]}個の特徴量")
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """価格ベース特徴量"""
        try:
            # 基本価格特徴量
            df['price_change'] = df['close'].pct_change()
            df['price_change_abs'] = df['price_change'].abs()
            df['log_return'] = np.log(df['close'] / df['close'].shift(1))
            
            # OHLC関係
            df['body_size'] = abs(df['close'] - df['open']) / df['close']
            df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
            df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
            df['total_range'] = (df['high'] - df['low']) / df['close']
            
            # 価格位置
            df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            df['open_close_ratio'] = df['open'] / df['close']
            
            # 高速移動平均
            for period in self.fast_periods:
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                df[f'price_vs_sma_{period}'] = (df['close'] - df[f'sma_{period}']) / df[f'sma_{period}']
                df[f'price_vs_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']
            
            return df
            
        except Exception as e:
            self.logger.error(f"価格特徴量エラー: {e}")
            return df
    
    def _add_fast_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """高速テクニカル指標（スキャルピング用）"""
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # RSI（複数期間）
            for period in [7, 14, 21]:
                df[f'rsi_{period}'] = talib.RSI(close, timeperiod=period)
                df[f'rsi_{period}_overbought'] = (df[f'rsi_{period}'] > 70).astype(int)
                df[f'rsi_{period}_oversold'] = (df[f'rsi_{period}'] < 30).astype(int)
            
            # MACD（高速設定）
            macd_fast, macd_signal, macd_hist = talib.MACD(close, fastperiod=8, slowperiod=17, signalperiod=9)
            df['macd'] = macd_fast
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_hist
            df['macd_bullish'] = (df['macd'] > df['macd_signal']).astype(int)
            df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) & 
                                  (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
            df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) & 
                                    (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
            
            # ボリンジャーバンド（短期）
            for period in [10, 20]:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(close, timeperiod=period, nbdevup=2, nbdevdn=2)
                df[f'bb_upper_{period}'] = bb_upper
                df[f'bb_middle_{period}'] = bb_middle
                df[f'bb_lower_{period}'] = bb_lower
                df[f'bb_position_{period}'] = (close - bb_lower) / (bb_upper - bb_lower)
                df[f'bb_squeeze_{period}'] = (bb_upper - bb_lower) / bb_middle
            
            # ストキャスティクス（高速）
            stoch_k, stoch_d = talib.STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3)
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d
            df['stoch_overbought'] = (stoch_k > 80).astype(int)
            df['stoch_oversold'] = (stoch_k < 20).astype(int)
            
            # ウィリアムズ%R
            df['williams_r'] = talib.WILLR(high, low, close, timeperiod=14)
            
            # CCI（Commodity Channel Index）
            df['cci'] = talib.CCI(high, low, close, timeperiod=14)
            df['cci_overbought'] = (df['cci'] > 100).astype(int)
            df['cci_oversold'] = (df['cci'] < -100).astype(int)
            
            return df
            
        except Exception as e:
            self.logger.error(f"テクニカル指標エラー: {e}")
            return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ボリューム特徴量"""
        try:
            close = df['close'].values
            volume = df['volume'].values
            
            # ボリューム移動平均
            for period in self.volume_periods:
                df[f'volume_ma_{period}'] = df['volume'].rolling(window=period).mean()
                df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_ma_{period}']
            
            # 価格ボリューム関係
            df['pv_trend'] = df['close'].diff() * df['volume']
            df['volume_price_trend'] = talib.VPT(close, volume)
            
            # ボリューム率
            df['volume_change'] = df['volume'].pct_change()
            df['volume_acceleration'] = df['volume_change'].diff()
            
            # On Balance Volume
            df['obv'] = talib.OBV(close, volume)
            df['obv_ma'] = df['obv'].rolling(window=10).mean()
            df['obv_signal'] = (df['obv'] > df['obv_ma']).astype(int)
            
            # Accumulation/Distribution Line
            df['ad_line'] = talib.AD(df['high'].values, df['low'].values, close, volume)
            
            # Money Flow Index
            df['mfi'] = talib.MFI(df['high'].values, df['low'].values, close, volume, timeperiod=14)
            
            # Volume Weighted Average Price (近似)
            df['vwap'] = (df['close'] * df['volume']).rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
            df['price_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap']
            
            return df
            
        except Exception as e:
            self.logger.error(f"ボリューム特徴量エラー: {e}")
            return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ボラティリティ特徴量"""
        try:
            # True Range & ATR
            df['true_range'] = talib.TRANGE(df['high'].values, df['low'].values, df['close'].values)
            for period in self.volatility_periods:
                df[f'atr_{period}'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=period)
                df[f'atr_ratio_{period}'] = df[f'atr_{period}'] / df['close']
            
            # Realized Volatility
            for period in [5, 10, 20]:
                df[f'realized_vol_{period}'] = df['log_return'].rolling(window=period).std() * np.sqrt(period)
            
            # ボラティリティ変化
            df['volatility_change'] = df['atr_5'].pct_change()
            df['volatility_acceleration'] = df['volatility_change'].diff()
            
            # ボラティリティレジーム
            df['vol_regime'] = pd.qcut(df['atr_10'], q=3, labels=[0, 1, 2])
            
            # Garman-Klass Volatility
            df['gk_volatility'] = np.log(df['high']/df['low'])**2 - (2*np.log(2)-1) * np.log(df['close']/df['open'])**2
            
            return df
            
        except Exception as e:
            self.logger.error(f"ボラティリティ特徴量エラー: {e}")
            return df
    
    def _add_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """マーケットマイクロストラクチャー特徴量"""
        try:
            # スプレッド近似（high-low）
            df['hl_spread'] = (df['high'] - df['low']) / df['close']
            df['hl_spread_ma'] = df['hl_spread'].rolling(window=10).mean()
            df['hl_spread_ratio'] = df['hl_spread'] / df['hl_spread_ma']
            
            # 価格インパクト（ボリュームと価格変化の関係）
            df['price_impact'] = df['price_change'].abs() / (df['volume'] + 1e-8)
            df['price_impact_ma'] = df['price_impact'].rolling(window=10).mean()
            
            # 流動性指標
            df['liquidity_proxy'] = df['volume'] / (df['high'] - df['low'] + 1e-8)
            df['liquidity_ma'] = df['liquidity_proxy'].rolling(window=10).mean()
            
            # ティック方向（価格の動き方向）
            df['tick_direction'] = np.sign(df['close'].diff())
            df['tick_direction_ma'] = df['tick_direction'].rolling(window=5).mean()
            
            # 価格圧力指標
            df['buying_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            df['selling_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low'])
            
            # オーダーフロー近似
            df['order_flow'] = df['volume'] * df['tick_direction']
            df['order_flow_ma'] = df['order_flow'].rolling(window=10).mean()
            
            return df
            
        except Exception as e:
            self.logger.error(f"マイクロストラクチャー特徴量エラー: {e}")
            return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """時間特徴量"""
        try:
            if 'timestamp' in df.columns:
                df['hour'] = df['timestamp'].dt.hour
                df['minute'] = df['timestamp'].dt.minute
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                
                # 取引セッション（概算）
                df['asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
                df['european_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
                df['us_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
                
                # 週末効果
                df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
                
                # 時間別ボラティリティパターン
                hourly_vol = df.groupby('hour')['atr_5'].mean()
                df['hourly_vol_pattern'] = df['hour'].map(hourly_vol)
                df['vol_vs_hourly_avg'] = df['atr_5'] / df['hourly_vol_pattern']
            
            return df
            
        except Exception as e:
            self.logger.error(f"時間特徴量エラー: {e}")
            return df
    
    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ラグ特徴量（過去の値）"""
        try:
            # 重要指標のラグ
            lag_features = ['price_change', 'volume_ratio_10', 'rsi_14', 'macd_histogram']
            lag_periods = [1, 2, 3, 5]
            
            for feature in lag_features:
                if feature in df.columns:
                    for lag in lag_periods:
                        df[f'{feature}_lag_{lag}'] = df[feature].shift(lag)
            
            return df
            
        except Exception as e:
            self.logger.error(f"ラグ特徴量エラー: {e}")
            return df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """相互作用特徴量"""
        try:
            # ボリュームと価格の相互作用
            if 'volume_ratio_10' in df.columns and 'price_change' in df.columns:
                df['volume_price_interaction'] = df['volume_ratio_10'] * df['price_change']
            
            # ボラティリティとモメンタムの相互作用
            if 'atr_ratio_10' in df.columns and 'rsi_14' in df.columns:
                df['volatility_momentum_interaction'] = df['atr_ratio_10'] * (df['rsi_14'] - 50) / 50
            
            # トレンドとボリュームの相互作用
            if 'price_vs_sma_21' in df.columns and 'volume_ratio_20' in df.columns:
                df['trend_volume_interaction'] = df['price_vs_sma_21'] * df['volume_ratio_20']
            
            return df
            
        except Exception as e:
            self.logger.error(f"相互作用特徴量エラー: {e}")
            return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """統計的特徴量"""
        try:
            # 移動統計
            windows = [5, 10, 20]
            for window in windows:
                # 移動分散
                df[f'price_var_{window}'] = df['price_change'].rolling(window=window).var()
                
                # 移動歪度
                df[f'price_skew_{window}'] = df['price_change'].rolling(window=window).skew()
                
                # 移動尖度
                df[f'price_kurt_{window}'] = df['price_change'].rolling(window=window).kurt()
                
                # 移動最大・最小
                df[f'price_max_{window}'] = df['close'].rolling(window=window).max()
                df[f'price_min_{window}'] = df['close'].rolling(window=window).min()
                df[f'price_range_{window}'] = df[f'price_max_{window}'] - df[f'price_min_{window}']
            
            # Z-score
            df['price_zscore'] = (df['close'] - df['close'].rolling(window=20).mean()) / df['close'].rolling(window=20).std()
            
            return df
            
        except Exception as e:
            self.logger.error(f"統計的特徴量エラー: {e}")
            return df
    
    def create_target_variables(self, df: pd.DataFrame, 
                               horizons: List[int] = [1, 3, 5, 10]) -> pd.DataFrame:
        """スキャルピング用ターゲット変数を作成"""
        try:
            for horizon in horizons:
                # 将来価格変化
                df[f'future_return_{horizon}'] = df['close'].shift(-horizon).pct_change()
                
                # 将来最高価格（利確可能性）
                df[f'future_max_{horizon}'] = df['high'].rolling(window=horizon).max().shift(-horizon)
                df[f'future_max_return_{horizon}'] = (df[f'future_max_{horizon}'] / df['close']) - 1
                
                # 将来最低価格（損失可能性）
                df[f'future_min_{horizon}'] = df['low'].rolling(window=horizon).min().shift(-horizon)
                df[f'future_min_return_{horizon}'] = (df[f'future_min_{horizon}'] / df['close']) - 1
                
                # スキャルピング収益性（スプレッドを考慮）
                spread_cost = 0.0001  # 0.01%と仮定
                df[f'scalping_profit_{horizon}'] = np.where(
                    df[f'future_max_return_{horizon}'] > spread_cost * 2,  # スプレッド*2以上の利益
                    1,  # 利益確定可能
                    0   # 利益確定困難
                )
            
            self.logger.info(f"ターゲット変数生成完了: {len(horizons)}期間")
            return df
            
        except Exception as e:
            self.logger.error(f"ターゲット変数エラー: {e}")
            return df
    
    def select_best_features(self, df: pd.DataFrame, target_col: str, 
                           top_k: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        """最適な特徴量を選択"""
        try:
            # 特徴量カラムを特定
            feature_cols = [col for col in df.columns 
                           if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume'] 
                           and not col.startswith('future_')
                           and not col.startswith('scalping_profit')]
            
            # NaNを除去
            feature_df = df[feature_cols + [target_col]].dropna()
            
            if len(feature_df) < 100:
                self.logger.warning("特徴量選択用データが不足")
                return df, feature_cols
            
            X = feature_df[feature_cols]
            y = feature_df[target_col]
            
            # 特徴量選択
            selector = SelectKBest(f_regression, k=min(top_k, len(feature_cols)))
            X_selected = selector.fit_transform(X, y)
            
            # 選択された特徴量を取得
            selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
            
            self.logger.info(f"特徴量選択完了: {len(selected_features)}/{len(feature_cols)} 選択")
            
            return df, selected_features
            
        except Exception as e:
            self.logger.error(f"特徴量選択エラー: {e}")
            return df, []