"""
Temporal-Aware Backtester - Prevents look-ahead bias
時系列制約対応バックテスター - 先見バイアス防止版
"""

import pandas as pd
import datetime
from typing import Dict, Optional, Any
from .portfolio import Portfolio
from .temporal_validator import TemporalValidator, create_temporal_validator
import logging

logger = logging.getLogger(__name__)

def run_temporal_backtest(price_data: pd.DataFrame, 
                         initial_capital: float = 1_000_000,
                         pair_name: str = 'USD/JPY',
                         strategy_type: str = 'technical',
                         strategy_params: Optional[Dict[str, Any]] = None,
                         execution_delay_minutes: int = 10,
                         default_lot_size: float = 0.1,
                         default_stop_loss_pips: int = 0,
                         default_take_profit_pips: int = 0,
                         spread_pips: float = 0.2) -> Portfolio:
    """
    時系列制約を守るバックテスト実行
    
    Args:
        price_data: 価格データ（時系列順）
        strategy_type: 'technical' または 'ai_news'
        strategy_params: 戦略パラメータ（AIモデル名等）
        execution_delay_minutes: 現実的な執行遅延（分）
    """
    
    if price_data.empty:
        logger.error("価格データが空です")
        return Portfolio(initial_capital, pair_name, spread_pips)
    
    # バックテスト期間の設定
    backtest_start = price_data.index[0]
    backtest_end = price_data.index[-1]
    
    # 時系列バリデーター初期化
    validator = create_temporal_validator(
        start_date=backtest_start.isoformat(),
        end_date=backtest_end.isoformat(),
        current_time=backtest_start,
        execution_delay_minutes=execution_delay_minutes
    )
    
    # ポートフォリオ初期化
    portfolio = Portfolio(initial_capital, pair_name, spread_pips)
    
    # ストラテジー初期化
    if strategy_type == 'ai_news':
        strategy = AINewsStrategy(validator, strategy_params or {})
    else:
        strategy = TechnicalStrategy(validator, strategy_params or {})
    
    # プログレッシブバックテスト実行
    open_position_ids = []
    
    logger.info(f"Starting temporal backtest: {backtest_start} to {backtest_end}")
    
    for current_timestamp, current_row in price_data.iterrows():
        # バックテスト時刻を更新
        validator.update_backtest_time(current_timestamp)
        
        # 現在時刻までの履歴データを取得
        historical_data = validator.filter_dataframe_by_time(
            price_data.loc[:current_timestamp], 
            timestamp_col=price_data.index.name or 'timestamp'
        )
        
        # エクイティ記録
        ts_to_record = current_timestamp
        if isinstance(ts_to_record, pd.Timestamp) and ts_to_record.tz is not None:
            ts_to_record = ts_to_record.tz_localize(None)
        portfolio.record_equity(ts_to_record)
        
        # ストップロス・テイクプロフィット処理
        open_position_ids = _process_stop_orders(
            portfolio, open_position_ids, current_timestamp, current_row
        )
        
        # 戦略に基づくシグナル生成（過去データのみ使用）
        signal = strategy.generate_signal(historical_data, current_timestamp)
        
        # シグナル実行
        if signal != 0:
            open_position_ids = _execute_signal(
                portfolio, open_position_ids, signal, current_timestamp, 
                current_row, default_lot_size, default_stop_loss_pips, 
                default_take_profit_pips
            )
    
    # 最終ポジション決済
    if open_position_ids:
        final_price = price_data['close'].iloc[-1]
        final_ts = backtest_end
        if isinstance(final_ts, pd.Timestamp) and final_ts.tz is not None:
            final_ts = final_ts.tz_localize(None)
            
        for pos_id in list(open_position_ids):
            if pos_id in portfolio.positions and portfolio.positions[pos_id]['status'] == 'open':
                portfolio.close_position(pos_id, final_ts, final_price, 
                                       exit_reason="End of Backtest")
    
    # 最終エクイティ記録
    portfolio.record_equity(final_ts)
    
    # 時系列違反レポート
    violation_report = validator.get_violation_report()
    if violation_report['total_violations'] > 0:
        logger.warning(f"Look-ahead bias violations detected: {violation_report['total_violations']}")
    
    # レポートをポートフォリオに添付
    portfolio.temporal_report = violation_report
    
    return portfolio

class TechnicalStrategy:
    """テクニカル分析ストラテジー"""
    
    def __init__(self, validator: TemporalValidator, params: Dict[str, Any]):
        self.validator = validator
        self.short_window = params.get('short_window', 5)
        self.long_window = params.get('long_window', 20)
        self.last_signal = 0
    
    def generate_signal(self, historical_data: pd.DataFrame, current_time: datetime.datetime) -> int:
        """過去データのみを使用してテクニカルシグナル生成"""
        if len(historical_data) < self.long_window:
            return 0
        
        # 移動平均計算（過去データのみ）
        short_sma = historical_data['close'].rolling(
            window=self.short_window, min_periods=self.short_window
        ).mean().iloc[-1]
        
        long_sma = historical_data['close'].rolling(
            window=self.long_window, min_periods=self.long_window
        ).mean().iloc[-1]
        
        # クロスオーバー検出
        if pd.isna(short_sma) or pd.isna(long_sma):
            return 0
        
        current_signal = 1 if short_sma > long_sma else -1
        
        # シグナル変化時のみ発火
        if current_signal != self.last_signal:
            self.last_signal = current_signal
            return current_signal
        
        return 0

class AINewsStrategy:
    """AI ニュース分析ストラテジー"""
    
    def __init__(self, validator: TemporalValidator, params: Dict[str, Any]):
        self.validator = validator
        self.model_name = params.get('model_name', 'gpt-4o')
        self.confidence_threshold = params.get('confidence_threshold', 0.7)
        self.processed_news = set()
        
        from sqlalchemy import create_engine
        self.engine = create_engine(params.get('database_url', 'sqlite:///data/system_data.db'))
    
    def generate_signal(self, historical_data: pd.DataFrame, current_time: datetime.datetime) -> int:
        """時系列制約を守ったAIニュースシグナル生成"""
        
        # 現在時刻以前のAI判断のみ取得
        ai_decisions = self.validator.get_ai_decisions_before_time(
            self.engine, self.model_name, limit=100
        )
        
        if ai_decisions.empty:
            return 0
        
        # 未処理の高信頼度取引推奨をチェック
        for _, row in ai_decisions.iterrows():
            news_id = row['news_id']
            
            # 既に処理済みのニュースはスキップ
            if news_id in self.processed_news:
                continue
            
            # 取引推奨かつ高信頼度
            if (row['trade_warranted'] and 
                row['confidence'] >= self.confidence_threshold):
                
                news_time = pd.to_datetime(row['published_at'])
                execution_time = self.validator.bounds.get_signal_execution_time(news_time)
                
                # 執行時刻が現在時刻と一致または直前
                time_diff = abs((execution_time - current_time).total_seconds())
                
                if time_diff <= 300:  # 5分以内
                    self.processed_news.add(news_id)
                    signal = 1 if row['direction'] == 'long' else -1
                    
                    logger.info(f"AI signal generated: {signal} from news {news_id} "
                              f"(confidence: {row['confidence']:.2f})")
                    return signal
        
        return 0

def _process_stop_orders(portfolio: Portfolio, open_position_ids: list, 
                        current_timestamp: datetime.datetime, 
                        current_row: pd.Series) -> list:
    """ストップロス・テイクプロフィット処理"""
    
    current_price_high = current_row['high']
    current_price_low = current_row['low']
    
    for pos_id in list(open_position_ids):
        if pos_id not in portfolio.positions or portfolio.positions[pos_id]['status'] == 'closed':
            open_position_ids.remove(pos_id)
            continue
        
        pos = portfolio.positions[pos_id]
        
        # ロングポジション
        if pos['direction'] == 'long':
            if pos['stop_loss_price'] and current_price_low <= pos['stop_loss_price']:
                portfolio.close_position(pos_id, current_timestamp, pos['stop_loss_price'],
                                       exit_reason=f"Stop Loss ({pos['stop_loss_pips']} pips)")
                open_position_ids.remove(pos_id)
                continue
            if pos['take_profit_price'] and current_price_high >= pos['take_profit_price']:
                portfolio.close_position(pos_id, current_timestamp, pos['take_profit_price'],
                                       exit_reason=f"Take Profit ({pos['take_profit_pips']} pips)")
                open_position_ids.remove(pos_id)
                continue
        
        # ショートポジション  
        else:
            if pos['stop_loss_price'] and current_price_high >= pos['stop_loss_price']:
                portfolio.close_position(pos_id, current_timestamp, pos['stop_loss_price'],
                                       exit_reason=f"Stop Loss ({pos['stop_loss_pips']} pips)")
                open_position_ids.remove(pos_id)
                continue
            if pos['take_profit_price'] and current_price_low <= pos['take_profit_price']:
                portfolio.close_position(pos_id, current_timestamp, pos['take_profit_price'],
                                       exit_reason=f"Take Profit ({pos['take_profit_pips']} pips)")
                open_position_ids.remove(pos_id)
                continue
    
    return open_position_ids

def _execute_signal(portfolio: Portfolio, open_position_ids: list, signal: int,
                   current_timestamp: datetime.datetime, current_row: pd.Series,
                   lot_size: float, stop_loss_pips: int, take_profit_pips: int) -> list:
    """シグナル実行"""
    
    current_price_open = current_row['open']
    
    if not open_position_ids:
        # 新規ポジション
        direction = 'long' if signal == 1 else 'short'
        portfolio.add_position(direction, current_timestamp, current_price_open,
                             lot_size, stop_loss_pips, take_profit_pips)
        
        new_ids = [pid for pid in portfolio.positions
                  if portfolio.positions[pid]['status'] == 'open'
                  and pid not in open_position_ids]
        if new_ids:
            open_position_ids.append(new_ids[0])
    
    else:
        # ドテン
        cur_id = open_position_ids[0]
        if cur_id in portfolio.positions and portfolio.positions[cur_id]['status'] == 'open':
            cur_direction = portfolio.positions[cur_id]['direction']
            opposite_signal = (cur_direction == 'long' and signal == -1) or \
                            (cur_direction == 'short' and signal == 1)
            
            if opposite_signal:
                portfolio.close_position(cur_id, current_timestamp, current_price_open,
                                       exit_reason="Opposite Signal")
                open_position_ids.remove(cur_id)
                
                new_direction = 'short' if signal == -1 else 'long'
                portfolio.add_position(new_direction, current_timestamp, current_price_open,
                                     lot_size, stop_loss_pips, take_profit_pips)
                
                new_ids = [pid for pid in portfolio.positions
                          if portfolio.positions[pid]['status'] == 'open'
                          and pid not in open_position_ids]
                if new_ids:
                    open_position_ids.append(new_ids[0])
    
    return open_position_ids