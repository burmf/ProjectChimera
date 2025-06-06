# core/backtester.py
import pandas as pd
from .portfolio import Portfolio

def run_backtest(price_data, signals_dict, initial_capital=1_000_000, pair_name='USD/JPY',
                 default_lot_size=0.1, default_stop_loss_pips=0, default_take_profit_pips=0, spread_pips=0.2,
                 ai_trade_decisions_df=None):
    """
    指定されたテクニカルシグナルに基づきバックテストを実行し、
    ストップロス、テイクプロフィット、スプレッドを考慮した損益計算を行う。
    """
    if price_data.empty:
        print("バックテストエラー: 価格データが空です。")
        return Portfolio(initial_capital, pair_name, spread_pips)

    # ポートフォリオ初期化
    portfolio = Portfolio(initial_capital, pair_name, spread_pips)

    # AIニュース判断データをマップとして準備（将来の統合ロジック用）
    ai_decisions_map = {}
    if ai_trade_decisions_df is not None and not ai_trade_decisions_df.empty:
        # ニュースIDや公開日時に基づくシグナル適用ロジックをここに追加
        # 例: for _, row in ai_trade_decisions_df.iterrows(): ai_decisions_map[row['news_id']] = row
        pass
    
    # 初期エクイティ記録（データ開始時刻）
    if not price_data.index.empty:
        start_time = price_data.index[0]
        if isinstance(start_time, pd.Timestamp) and start_time.tz is not None:
            start_time = start_time.tz_localize(None)
        portfolio.equity_curve = [{'timestamp': start_time, 'equity': initial_capital}]
    
    open_position_ids = []

    for timestamp, row in price_data.iterrows():
        current_price_open = row['open']
        current_price_high = row['high']
        current_price_low = row['low']

        # エクイティ記録
        ts_to_record = timestamp
        if isinstance(ts_to_record, pd.Timestamp) and ts_to_record.tz is not None:
            ts_to_record = ts_to_record.tz_localize(None)
        portfolio.record_equity(ts_to_record)
        
        # シグナル取得（無ければ 0）
        current_signal = signals_dict.get(timestamp, 0)

        # --- ストップロス/テイクプロフィット優先の決済 ---
        for pos_id in list(open_position_ids):
            if pos_id not in portfolio.positions or portfolio.positions[pos_id]['status'] == 'closed':
                open_position_ids.remove(pos_id)
                continue

            pos = portfolio.positions[pos_id]
            # ロングポジション
            if pos['direction'] == 'long':
                if pos['stop_loss_price'] and current_price_low <= pos['stop_loss_price']:
                    portfolio.close_position(pos_id, timestamp, pos['stop_loss_price'],
                                             exit_reason=f"Stop Loss ({pos['stop_loss_pips']} pips)")
                    open_position_ids.remove(pos_id)
                    continue
                if pos['take_profit_price'] and current_price_high >= pos['take_profit_price']:
                    portfolio.close_position(pos_id, timestamp, pos['take_profit_price'],
                                             exit_reason=f"Take Profit ({pos['take_profit_pips']} pips)")
                    open_position_ids.remove(pos_id)
                    continue
            # ショートポジション
            else:
                if pos['stop_loss_price'] and current_price_high >= pos['stop_loss_price']:
                    portfolio.close_position(pos_id, timestamp, pos['stop_loss_price'],
                                             exit_reason=f"Stop Loss ({pos['stop_loss_pips']} pips)")
                    open_position_ids.remove(pos_id)
                    continue
                if pos['take_profit_price'] and current_price_low <= pos['take_profit_price']:
                    portfolio.close_position(pos_id, timestamp, pos['take_profit_price'],
                                             exit_reason=f"Take Profit ({pos['take_profit_pips']} pips)")
                    open_position_ids.remove(pos_id)
                    continue
        
        # --- シグナルによるエントリーとクローズ ---
        if not open_position_ids:
            # 新規ポジションオープン
            if current_signal == 1:
                portfolio.add_position('long', timestamp, current_price_open,
                                       default_lot_size, default_stop_loss_pips, default_take_profit_pips)
                new_ids = [pid for pid in portfolio.positions
                           if portfolio.positions[pid]['status']=='open'
                           and pid not in open_position_ids]
                if new_ids:
                    open_position_ids.append(new_ids[0])
            elif current_signal == -1:
                portfolio.add_position('short', timestamp, current_price_open,
                                       default_lot_size, default_stop_loss_pips, default_take_profit_pips)
                new_ids = [pid for pid in portfolio.positions
                           if portfolio.positions[pid]['status']=='open'
                           and pid not in open_position_ids]
                if new_ids:
                    open_position_ids.append(new_ids[0])
        else:
            # 既存ポジションのドテンシグナルチェック
            cur_id = open_position_ids[0]
            if cur_id in portfolio.positions and portfolio.positions[cur_id]['status']=='open':
                dirn = portfolio.positions[cur_id]['direction']
                if (dirn=='long' and current_signal==-1) or (dirn=='short' and current_signal==1):
                    # 決済
                    portfolio.close_position(cur_id, timestamp, current_price_open,
                                             exit_reason="Opposite Signal")
                    open_position_ids.remove(cur_id)
                    # 反対方向でエントリー
                    new_dir = 'short' if current_signal==-1 else 'long'
                    portfolio.add_position(new_dir, timestamp, current_price_open,
                                           default_lot_size, default_stop_loss_pips, default_take_profit_pips)
                    new_ids = [pid for pid in portfolio.positions
                               if portfolio.positions[pid]['status']=='open'
                               and pid not in open_position_ids]
                    if new_ids:
                        open_position_ids.append(new_ids[0])

    # バックテスト終了時のオープンポジション決済
    if open_position_ids:
        final_ts = price_data.index[-1]
        final_price = price_data['close'].iloc[-1]
        if isinstance(final_ts, pd.Timestamp) and final_ts.tz is not None:
            final_ts = final_ts.tz_localize(None)
        for pos_id in list(open_position_ids):
            if pos_id in portfolio.positions and portfolio.positions[pos_id]['status']=='open':
                portfolio.close_position(pos_id, final_ts, final_price, exit_reason="End of Backtest")
            
    # 最終エクイティ記録
    final_ts = price_data.index[-1]
    if isinstance(final_ts, pd.Timestamp) and final_ts.tz is not None:
        final_ts = final_ts.tz_localize(None)
    portfolio.record_equity(final_ts)

    return portfolio