"""
Temporal Validator - Prevents look-ahead bias in backtesting
時系列整合性チェッカー - バックテスト時の先見バイアスを防止
"""

import pandas as pd
import datetime
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class TemporalBounds:
    """時系列制約の定義"""
    backtest_start: datetime.datetime
    backtest_end: datetime.datetime
    current_time: datetime.datetime
    execution_delay_minutes: int = 10  # 現実的な執行遅延
    
    def is_valid_data_time(self, data_timestamp: datetime.datetime) -> bool:
        """データのタイムスタンプが現在のバックテスト時点で参照可能か判定"""
        if not isinstance(data_timestamp, datetime.datetime):
            return False
        return data_timestamp <= self.current_time
    
    def get_signal_execution_time(self, news_time: datetime.datetime) -> datetime.datetime:
        """ニュース発表時刻から現実的な執行時刻を計算"""
        return news_time + datetime.timedelta(minutes=self.execution_delay_minutes)

class TemporalValidator:
    """時系列データアクセスの検証と制御"""
    
    def __init__(self, bounds: TemporalBounds):
        self.bounds = bounds
        self.violation_count = 0
        self.violations_log = []
    
    def validate_data_access(self, data_timestamp: datetime.datetime, 
                           data_type: str = "unknown") -> bool:
        """データアクセスの時系列妥当性を検証"""
        if not self.bounds.is_valid_data_time(data_timestamp):
            violation = {
                'timestamp': datetime.datetime.now().isoformat(),
                'backtest_time': self.bounds.current_time.isoformat(),
                'data_time': data_timestamp.isoformat(),
                'data_type': data_type,
                'violation': 'future_data_access'
            }
            self.violations_log.append(violation)
            self.violation_count += 1
            
            logger.warning(f"Look-ahead bias detected: {data_type} data from "
                         f"{data_timestamp} accessed at backtest time {self.bounds.current_time}")
            return False
        return True
    
    def filter_dataframe_by_time(self, df: pd.DataFrame, 
                                timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """DataFrameを現在のバックテスト時点まで時系列フィルタリング"""
        if df.empty:
            return df
        
        # タイムスタンプ列の処理
        if timestamp_col in df.columns:
            time_series = pd.to_datetime(df[timestamp_col])
        elif timestamp_col in df.index.names or isinstance(df.index, pd.DatetimeIndex):
            time_series = pd.to_datetime(df.index)
        else:
            logger.error(f"Timestamp column '{timestamp_col}' not found")
            return df
        
        # 時系列フィルタリング
        valid_mask = time_series <= self.bounds.current_time
        filtered_df = df[valid_mask].copy()
        
        excluded_count = len(df) - len(filtered_df)
        if excluded_count > 0:
            logger.info(f"Filtered out {excluded_count} future data points from {timestamp_col}")
        
        return filtered_df
    
    def get_ai_decisions_before_time(self, engine, model_name: str, 
                                   limit: int = 10000) -> pd.DataFrame:
        """指定時刻以前のAI判断のみを取得（先見バイアス防止）"""
        from sqlalchemy import text
        
        query = text("""
            SELECT ad.*, na.published_at, na.title as news_title
            FROM ai_trade_decisions ad
            JOIN news_articles na ON ad.news_id = na.id
            WHERE ad.processed_at <= :max_time 
            AND ad.model_name = :model
            AND na.published_at <= :max_time
            ORDER BY ad.processed_at ASC
            LIMIT :limit
        """)
        
        try:
            with engine.connect() as conn:
                df = pd.read_sql_query(query, conn, params={
                    'max_time': self.bounds.current_time.isoformat(),
                    'model': model_name,
                    'limit': limit
                })
            
            logger.info(f"Retrieved {len(df)} AI decisions before {self.bounds.current_time}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get AI decisions: {e}")
            return pd.DataFrame()
    
    def generate_temporal_signals(self, ai_decisions_df: pd.DataFrame, 
                                price_df: pd.DataFrame) -> Dict[datetime.datetime, int]:
        """時系列制約を守ったシグナル生成"""
        signals = {}
        
        for _, row in ai_decisions_df.iterrows():
            if not row['trade_warranted']:
                continue
            
            # ニュース発表時刻
            news_time = pd.to_datetime(row['published_at'])
            
            # 時系列妥当性チェック
            if not self.validate_data_access(news_time, "news"):
                continue
            
            # 現実的な執行時刻計算
            execution_time = self.bounds.get_signal_execution_time(news_time)
            
            # 執行時刻がバックテスト範囲内かチェック
            if execution_time > self.bounds.current_time:
                continue
            
            # 執行可能な価格データを探す
            future_prices = price_df.index[price_df.index >= execution_time]
            if not future_prices.empty:
                signal_timestamp = future_prices[0]
                signal_value = 1 if row['direction'] == 'long' else -1
                signals[signal_timestamp] = signal_value
                
                logger.debug(f"Generated signal at {signal_timestamp} from news at {news_time}")
        
        return signals
    
    def update_backtest_time(self, new_time: datetime.datetime):
        """バックテスト時刻を更新"""
        if new_time < self.bounds.current_time:
            logger.warning(f"Backtest time moved backwards: {new_time} < {self.bounds.current_time}")
        
        self.bounds.current_time = new_time
        logger.debug(f"Updated backtest time to {new_time}")
    
    def get_violation_report(self) -> Dict[str, Any]:
        """時系列違反レポートを生成"""
        return {
            'total_violations': self.violation_count,
            'violations': self.violations_log,
            'backtest_period': {
                'start': self.bounds.backtest_start.isoformat(),
                'end': self.bounds.backtest_end.isoformat(),
                'current': self.bounds.current_time.isoformat()
            },
            'execution_delay_minutes': self.bounds.execution_delay_minutes
        }
    
    def reset_violations(self):
        """違反カウントをリセット"""
        self.violation_count = 0
        self.violations_log.clear()

def create_temporal_validator(start_date: str, end_date: str, 
                            current_time: Optional[datetime.datetime] = None,
                            execution_delay_minutes: int = 10) -> TemporalValidator:
    """TemporalValidatorのファクトリ関数"""
    
    backtest_start = pd.to_datetime(start_date)
    backtest_end = pd.to_datetime(end_date)
    
    if current_time is None:
        current_time = backtest_start
    
    bounds = TemporalBounds(
        backtest_start=backtest_start,
        backtest_end=backtest_end,
        current_time=current_time,
        execution_delay_minutes=execution_delay_minutes
    )
    
    return TemporalValidator(bounds)