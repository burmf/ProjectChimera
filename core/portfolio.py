import pandas as pd
import datetime

class Portfolio:
    def __init__(self, initial_capital=1_000_000, pair='USD/JPY', spread_pips=0.2): # スプレッドを追加
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.pair = pair
        self.spread_pips = spread_pips # pips単位
        self.positions = {} 
        self.trade_history = []
        self.equity_curve = [{'timestamp': datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None), 'equity': initial_capital}]
        self._position_counter = 0

    def _generate_position_id(self):
        self._position_counter += 1
        return f"pos_{self.pair.replace('/', '')}_{self._position_counter}_{int(datetime.datetime.now().timestamp())}"

    def _get_pip_value_multiplier(self):
        # USD/JPYのような通貨ペアを想定 (1pip = 0.01円)
        # 価格が XXX.YYY の形式の場合（例：USDJPY 150.123）、pipは小数第2位
        # 価格が X.YYYYY の形式の場合（例：EURUSD 1.08123）、pipは小数第4位
        if "JPY" in self.pair.upper():
            return 100 # 0.01円が1pipなので、価格差を0.01で割る＝100を掛ける
        else:
            return 10000 # 0.0001ドルが1pipなので、価格差を0.0001で割る＝10000を掛ける

    def add_position(self, direction, entry_time, entry_price_actual, lot_size=0.1, 
                     stop_loss_pips=0, take_profit_pips=0, entry_reason="Signal"):
        pos_id = self._generate_position_id()
        
        # スプレッドを考慮した実際の約定価格
        # 買いの場合、スプレッド分不利な価格で約定
        # 売りの場合、スプレッド分不利な価格で約定
        pip_unit = 1.0 / self._get_pip_value_multiplier() # 1pipの価格単位
        spread_cost_in_price = self.spread_pips * pip_unit

        if direction == 'long':
            entry_price_effective = entry_price_actual + spread_cost_in_price
        else: # short
            entry_price_effective = entry_price_actual - spread_cost_in_price

        sl_price = None
        tp_price = None

        if stop_loss_pips > 0:
            if direction == 'long':
                sl_price = entry_price_effective - (stop_loss_pips * pip_unit)
            else: # short
                sl_price = entry_price_effective + (stop_loss_pips * pip_unit)
        
        if take_profit_pips > 0:
            if direction == 'long':
                tp_price = entry_price_effective + (take_profit_pips * pip_unit)
            else: # short
                tp_price = entry_price_effective - (take_profit_pips * pip_unit)

        self.positions[pos_id] = {
            'direction': direction,
            'entry_time': entry_time,
            'entry_price_actual': entry_price_actual, # スプレッド適用前の価格
            'entry_price_effective': entry_price_effective, # スプレッド適用後の実効価格
            'lot_size': lot_size,
            'stop_loss_price': sl_price,
            'take_profit_price': tp_price,
            'stop_loss_pips': stop_loss_pips, # 参考情報
            'take_profit_pips': take_profit_pips, # 参考情報
            'status': 'open'
        }
        print(f"{entry_time}: {self.pair} {direction.upper()} @ {entry_price_actual:.3f} (Spread考慮後 {entry_price_effective:.3f}), SL:{sl_price if sl_price else 'N/A'}, TP:{tp_price if tp_price else 'N/A'} (ID: {pos_id})")

    def close_position(self, pos_id, exit_time, exit_price_actual, exit_reason="Signal"):
        if pos_id not in self.positions or self.positions[pos_id]['status'] == 'closed':
            # print(f"警告: ポジション {pos_id} は存在しないか、既に決済済みです。") # ログが煩雑になるためコメントアウト
            return

        pos = self.positions[pos_id]
        
        # スプレッドを考慮した実効エントリー価格を使用
        entry_price_to_use = pos['entry_price_effective']
        
        # 決済時にもスプレッドを考慮する場合（通常はエントリー時のみだが、ここでは対称性を保つため）
        # 買いポジションの決済（売り）はビッド価格、売りポジションの決済（買い）はアスク価格になる
        pip_unit = 1.0 / self._get_pip_value_multiplier()
        spread_cost_in_price = self.spread_pips * pip_unit
        
        if pos['direction'] == 'long': # 買いポジションを決済 = 売る
            exit_price_effective = exit_price_actual - spread_cost_in_price
            pips = (exit_price_effective - entry_price_to_use) * self._get_pip_value_multiplier()
        else: # short # 売りポジションを決済 = 買う
            exit_price_effective = exit_price_actual + spread_cost_in_price
            pips = (entry_price_to_use - exit_price_effective) * self._get_pip_value_multiplier()
        
        # 損益計算 (0.1ロット = 10,000通貨, USDJPYなら1pips=0.01円なので、0.01*10000 = 100円/pips)
        pnl_per_pip_per_lot = pip_unit * (100_000) # 1ロットあたりの1pipの価値
        pnl_currency = pips * pnl_per_pip_per_lot * pos['lot_size']

        self.cash += pnl_currency
        pos['status'] = 'closed' 

        trade_log = {
            'trade_id': pos_id,
            'entry_time': pos['entry_time'],
            'exit_time': exit_time,
            'pair': self.pair,
            'direction': pos['direction'],
            'lot_size': pos['lot_size'],
            'entry_price_actual': pos['entry_price_actual'],
            'entry_price_effective': entry_price_to_use,
            'exit_price_actual': exit_price_actual,
            'exit_price_effective': exit_price_effective,
            'pnl_pips': pips,
            'pnl_currency': pnl_currency,
            'entry_reason': "Signal", 
            'exit_reason': exit_reason,
            'sl_pips_setting': pos['stop_loss_pips'],
            'tp_pips_setting': pos['take_profit_pips'],
            'stop_loss_price': pos['stop_loss_price'],
            'take_profit_price': pos['take_profit_price']
        }
        self.trade_history.append(trade_log)
        print(f"{exit_time}: {self.pair} ポジション {pos_id} 決済 @ {exit_price_actual:.3f} (Spread考慮後 {exit_price_effective:.3f})。損益: {pnl_currency:,.0f}円 ({pips:.2f} pips, 理由: {exit_reason})")
        self.record_equity(exit_time)

    def record_equity(self, timestamp):
        current_equity = self.cash 
        if isinstance(timestamp, pd.Timestamp) and timestamp.tz is not None:
            timestamp = timestamp.tz_localize(None)
        self.equity_curve.append({'timestamp': timestamp, 'equity': current_equity})

    def get_equity_curve_df(self):
        if not self.equity_curve:
            return pd.DataFrame(columns=['timestamp', 'equity']).set_index('timestamp')
        df = pd.DataFrame(self.equity_curve)
        if df.empty:
             return pd.DataFrame(columns=['timestamp', 'equity']).set_index('timestamp')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df.set_index('timestamp')

    def get_trade_history_df(self):
        if not self.trade_history:
            return pd.DataFrame()
        return pd.DataFrame(self.trade_history)

    def get_open_positions(self):
        return {pid: pinfo for pid, pinfo in self.positions.items() if pinfo['status'] == 'open'}