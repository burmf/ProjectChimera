#!/usr/bin/env python3
"""
Aggressive Bitget Scalping Engine
アグレッシブスキャルピングエンジン - より低い閾値で高頻度取引
"""

import asyncio
import json
import time
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import deque
import requests
import statistics
import random

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AggressiveScalpingEngine:
    """
    アグレッシブスキャルピングエンジン
    
    - 0.1%以上の小さな価格変動を狙う
    - 0.05%ストップロス（タイト）
    - 30秒-2分の超短期取引
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 取引設定
        self.symbol = 'BTCUSDT_SPBL'
        self.base_url = 'https://api.bitget.com'
        
        # アグレッシブスキャルピング設定
        self.min_profit_target = 0.001   # 0.1%最小利益目標
        self.max_profit_target = 0.003   # 0.3%最大利益目標
        self.stop_loss = 0.0005          # 0.05%ストップロス
        self.position_size = 25000       # $25,000ポジション（より大きく）
        self.max_hold_time = 120         # 2分最大保有時間
        
        # データ管理
        self.price_history = deque(maxlen=50)
        self.trades = []
        self.current_position = None
        self.account_balance = 100000
        
        # パフォーマンス追跡
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0
        self.start_time = datetime.now()
        
        # 高頻度分析設定
        self.momentum_window = 10
        self.volatility_window = 8
        
        self.logger.info("Aggressive Scalping Engine initialized")
    
    async def get_realtime_price(self) -> Optional[float]:
        """リアルタイム価格取得"""
        try:
            response = requests.get(
                f'{self.base_url}/api/spot/v1/market/ticker?symbol={self.symbol}',
                timeout=1.5
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    price = float(data['data']['close'])
                    return price
            
            return None
            
        except Exception as e:
            self.logger.error(f"Price fetch error: {e}")
            return None
    
    def calculate_micro_indicators(self) -> Dict[str, float]:
        """マイクロ指標計算（高頻度取引用）"""
        if len(self.price_history) < 5:
            return {'momentum': 0, 'volatility': 0, 'trend': 0, 'noise': 0}
        
        prices = list(self.price_history)
        
        # 極短期モメンタム（直近3価格の変化）
        if len(prices) >= 3:
            momentum = (prices[-1] - prices[-3]) / prices[-3]
        else:
            momentum = 0
        
        # 瞬間ボラティリティ
        if len(prices) >= self.volatility_window:
            returns = [(prices[i] - prices[i-1]) / prices[i-1] 
                      for i in range(-self.volatility_window, 0)]
            volatility = statistics.stdev(returns) if len(returns) > 1 else 0
        else:
            volatility = 0
        
        # 短期トレンド（直近5価格の線形回帰）
        if len(prices) >= 5:
            x = list(range(5))
            y = prices[-5:]
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            if n * sum_x2 - sum_x ** 2 != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
                trend = slope / (sum_y / n)  # 正規化
            else:
                trend = 0
        else:
            trend = 0
        
        # ノイズレベル（価格の不安定性）
        if len(prices) >= 5:
            changes = [abs(prices[i] - prices[i-1]) / prices[i-1] 
                      for i in range(-4, 0)]
            noise = statistics.mean(changes)
        else:
            noise = 0
        
        return {
            'momentum': momentum,
            'volatility': volatility,
            'trend': trend,
            'noise': noise
        }
    
    def generate_aggressive_signal(self, current_price: float) -> Dict[str, Any]:
        """アグレッシブスキャルピングシグナル生成"""
        indicators = self.calculate_micro_indicators()
        
        signal = {
            'action': 'hold',
            'confidence': 0.0,
            'reasoning': '',
            'entry_price': current_price,
            'target_price': 0,
            'stop_loss_price': 0
        }
        
        momentum = indicators['momentum']
        volatility = indicators['volatility']
        trend = indicators['trend']
        noise = indicators['noise']
        
        # 低ボラティリティでは取引しない
        if volatility < 0.0005:  # 0.05%未満
            signal['reasoning'] = f'Too low volatility: {volatility:.6f}'
            return signal
        
        # ノイズが多すぎる場合は避ける
        if noise > 0.002:  # 0.2%以上のノイズ
            signal['reasoning'] = f'Too much noise: {noise:.6f}'
            return signal
        
        # より感度の高いロングシグナル
        if momentum > 0.0003 and trend > 0:  # 0.03%以上の上昇モメンタム
            signal['action'] = 'long'
            signal['confidence'] = min(0.95, momentum * 1000 + trend * 100)
            signal['target_price'] = current_price * (1 + self.min_profit_target)
            signal['stop_loss_price'] = current_price * (1 - self.stop_loss)
            signal['reasoning'] = f'Micro bullish: mom={momentum:.5f}, trend={trend:.5f}'
        
        # より感度の高いショートシグナル
        elif momentum < -0.0003 and trend < 0:  # 0.03%以上の下落モメンタム
            signal['action'] = 'short'
            signal['confidence'] = min(0.95, abs(momentum) * 1000 + abs(trend) * 100)
            signal['target_price'] = current_price * (1 - self.min_profit_target)
            signal['stop_loss_price'] = current_price * (1 + self.stop_loss)
            signal['reasoning'] = f'Micro bearish: mom={momentum:.5f}, trend={trend:.5f}'
        
        # オーバーシュート狙い（反転狙い）
        elif abs(momentum) > 0.001 and volatility > 0.001:
            if momentum > 0 and trend < 0:  # 上昇後の反転狙い
                signal['action'] = 'short'
                signal['confidence'] = 0.7
                signal['target_price'] = current_price * (1 - self.min_profit_target)
                signal['stop_loss_price'] = current_price * (1 + self.stop_loss)
                signal['reasoning'] = f'Reversal short: mom={momentum:.5f}, trend={trend:.5f}'
            elif momentum < 0 and trend > 0:  # 下落後の反転狙い
                signal['action'] = 'long'
                signal['confidence'] = 0.7
                signal['target_price'] = current_price * (1 + self.min_profit_target)
                signal['stop_loss_price'] = current_price * (1 - self.stop_loss)
                signal['reasoning'] = f'Reversal long: mom={momentum:.5f}, trend={trend:.5f}'
        
        else:
            signal['reasoning'] = f'No signal: mom={momentum:.5f}, trend={trend:.5f}, vol={volatility:.5f}'
        
        return signal
    
    def execute_virtual_trade(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """仮想取引実行"""
        trade_result = {
            'timestamp': datetime.now(),
            'action': signal['action'],
            'entry_price': signal['entry_price'],
            'position_size': self.position_size,
            'target_price': signal['target_price'],
            'stop_loss_price': signal['stop_loss_price'],
            'confidence': signal['confidence'],
            'reasoning': signal['reasoning'],
            'status': 'opened'
        }
        
        self.current_position = trade_result
        self.total_trades += 1
        
        profit_pct = ((signal['target_price'] - signal['entry_price']) / signal['entry_price']) * 100
        if signal['action'] == 'short':
            profit_pct = ((signal['entry_price'] - signal['target_price']) / signal['entry_price']) * 100
        
        self.logger.info(f"⚡ TRADE #{self.total_trades}: {signal['action'].upper()}")
        self.logger.info(f"   Entry: ${signal['entry_price']:,.2f}")
        self.logger.info(f"   Target: ${signal['target_price']:,.2f} ({profit_pct:+.2f}%)")
        self.logger.info(f"   Confidence: {signal['confidence']:.1%}")
        
        return trade_result
    
    def check_position_exit(self, current_price: float) -> Optional[Dict[str, Any]]:
        """ポジション決済チェック"""
        if not self.current_position:
            return None
        
        pos = self.current_position
        entry_price = pos['entry_price']
        target_price = pos['target_price']
        stop_loss_price = pos['stop_loss_price']
        
        # 利確チェック
        if pos['action'] == 'long' and current_price >= target_price:
            return self.close_position(current_price, 'profit_target')
        elif pos['action'] == 'short' and current_price <= target_price:
            return self.close_position(current_price, 'profit_target')
        
        # ストップロスチェック
        if pos['action'] == 'long' and current_price <= stop_loss_price:
            return self.close_position(current_price, 'stop_loss')
        elif pos['action'] == 'short' and current_price >= stop_loss_price:
            return self.close_position(current_price, 'stop_loss')
        
        # タイムアウト（2分経過）
        if (datetime.now() - pos['timestamp']).total_seconds() > self.max_hold_time:
            return self.close_position(current_price, 'timeout')
        
        return None
    
    def close_position(self, exit_price: float, reason: str) -> Dict[str, Any]:
        """ポジション決済"""
        pos = self.current_position
        entry_price = pos['entry_price']
        
        if pos['action'] == 'long':
            profit_loss = (exit_price - entry_price) / entry_price
        else:  # short
            profit_loss = (entry_price - exit_price) / entry_price
        
        profit_amount = self.position_size * profit_loss
        
        trade_result = {
            'entry_time': pos['timestamp'],
            'exit_time': datetime.now(),
            'action': pos['action'],
            'entry_price': entry_price,
            'exit_price': exit_price,
            'profit_loss_pct': profit_loss,
            'profit_loss_amount': profit_amount,
            'reason': reason,
            'duration': datetime.now() - pos['timestamp']
        }
        
        # 統計更新
        self.total_profit += profit_amount
        self.account_balance += profit_amount
        
        if profit_amount > 0:
            self.winning_trades += 1
        
        self.trades.append(trade_result)
        self.current_position = None
        
        # ログ出力
        status_emoji = '✅' if profit_amount > 0 else '❌'
        duration_sec = trade_result['duration'].total_seconds()
        
        self.logger.info(f"{status_emoji} CLOSED: {reason.upper()}")
        self.logger.info(f"   Exit: ${exit_price:,.2f}")
        self.logger.info(f"   P&L: {profit_loss:+.3%} (${profit_amount:+,.2f})")
        self.logger.info(f"   Duration: {duration_sec:.0f}s")
        
        return trade_result
    
    def print_live_stats(self):
        """ライブ統計表示"""
        if self.total_trades == 0:
            return
        
        win_rate = self.winning_trades / self.total_trades
        running_time = datetime.now() - self.start_time
        trades_per_minute = self.total_trades / (running_time.total_seconds() / 60)
        
        print(f"\\n⚡ LIVE STATS | Trades: {self.total_trades} | Win: {win_rate:.1%} | P&L: ${self.total_profit:+,.2f} | Rate: {trades_per_minute:.1f}/min")
    
    async def run_aggressive_session(self, duration_minutes: int = 10):
        """アグレッシブスキャルピングセッション実行"""
        self.logger.info(f"⚡ Starting {duration_minutes}-minute AGGRESSIVE scalping session")
        self.logger.info(f"Symbol: {self.symbol}")
        self.logger.info(f"Position Size: ${self.position_size:,}")
        self.logger.info(f"Profit Target: {self.min_profit_target:.1%}")
        self.logger.info(f"Stop Loss: {self.stop_loss:.2%}")
        self.logger.info(f"Max Hold Time: {self.max_hold_time}s")
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        last_price = None
        price_update_count = 0
        
        try:
            while datetime.now() < end_time:
                # 価格取得
                current_price = await self.get_realtime_price()
                
                if current_price is None:
                    await asyncio.sleep(0.5)
                    continue
                
                # 価格が変わった場合のみ処理
                if last_price is None or abs(current_price - last_price) / last_price > 0.00001:  # 0.001%以上の変化
                    self.price_history.append(current_price)
                    last_price = current_price
                    price_update_count += 1
                
                # 既存ポジションの管理
                if self.current_position:
                    exit_result = self.check_position_exit(current_price)
                    if exit_result:
                        # 決済後即座に次の機会を探す
                        await asyncio.sleep(1)
                
                # 新規エントリーシグナルチェック（より頻繁に）
                elif len(self.price_history) >= self.momentum_window:
                    signal = self.generate_aggressive_signal(current_price)
                    
                    # より低い閾値でエントリー
                    if signal['action'] != 'hold' and signal['confidence'] > 0.5:
                        self.execute_virtual_trade(signal)
                
                # ライブ統計表示（10秒ごと）
                if int(time.time()) % 10 == 0:
                    self.print_live_stats()
                
                # 高頻度チェック（0.5秒間隔）
                await asyncio.sleep(0.5)
                
        except KeyboardInterrupt:
            self.logger.info("Aggressive scalping session interrupted")
        
        # 最終ポジション決済
        if self.current_position:
            final_price = await self.get_realtime_price()
            if final_price:
                self.close_position(final_price, 'session_end')
        
        # 最終統計
        self.print_final_stats()
        
        # セッション結果保存
        self.save_session_results()
    
    def print_final_stats(self):
        """最終統計表示"""
        if self.total_trades == 0:
            print("\\n⚡ No trades executed during session")
            return
        
        win_rate = self.winning_trades / self.total_trades
        avg_profit = self.total_profit / self.total_trades
        running_time = datetime.now() - self.start_time
        trades_per_minute = self.total_trades / (running_time.total_seconds() / 60)
        
        print(f"\\n⚡ AGGRESSIVE SCALPING FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Session Duration: {running_time}")
        print(f"Total Trades: {self.total_trades}")
        print(f"Winning Trades: {self.winning_trades}")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Trading Frequency: {trades_per_minute:.1f} trades/minute")
        print(f"Total P&L: ${self.total_profit:+,.2f}")
        print(f"Average P&L per Trade: ${avg_profit:+,.2f}")
        print(f"Final Balance: ${self.account_balance:,.2f}")
        print(f"Session Return: {(self.account_balance - 100000) / 100000:+.2%}")
        
        if self.trades:
            profits = [t['profit_loss_amount'] for t in self.trades if t['profit_loss_amount'] > 0]
            losses = [t['profit_loss_amount'] for t in self.trades if t['profit_loss_amount'] < 0]
            
            durations = [t['duration'].total_seconds() for t in self.trades]
            avg_duration = sum(durations) / len(durations)
            
            print(f"Average Trade Duration: {avg_duration:.0f} seconds")
            
            if profits and losses:
                avg_win = sum(profits) / len(profits)
                avg_loss = sum(losses) / len(losses)
                profit_factor = abs(sum(profits) / sum(losses))
                
                print(f"Average Win: ${avg_win:+.2f}")
                print(f"Average Loss: ${avg_loss:+.2f}")
                print(f"Profit Factor: {profit_factor:.2f}")
        
        print(f"{'='*60}")
    
    def save_session_results(self):
        """セッション結果保存"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'aggressive_scalping_{timestamp}.json'
        
        session_data = {
            'timestamp': timestamp,
            'symbol': self.symbol,
            'strategy': 'aggressive_scalping',
            'duration': str(datetime.now() - self.start_time),
            'position_size': self.position_size,
            'profit_target': self.min_profit_target,
            'stop_loss': self.stop_loss,
            'max_hold_time': self.max_hold_time,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
            'total_profit': self.total_profit,
            'final_balance': self.account_balance,
            'return_pct': (self.account_balance - 100000) / 100000,
            'trades_per_minute': self.total_trades / ((datetime.now() - self.start_time).total_seconds() / 60),
            'trades': [
                {
                    'entry_time': str(trade['entry_time']),
                    'exit_time': str(trade['exit_time']),
                    'action': trade['action'],
                    'entry_price': trade['entry_price'],
                    'exit_price': trade['exit_price'],
                    'profit_loss_pct': trade['profit_loss_pct'],
                    'profit_loss_amount': trade['profit_loss_amount'],
                    'reason': trade['reason'],
                    'duration_seconds': trade['duration'].total_seconds()
                }
                for trade in self.trades
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        self.logger.info(f"Aggressive scalping results saved to {filename}")


async def main():
    """メイン実行"""
    engine = AggressiveScalpingEngine()
    
    print("⚡ Bitget AGGRESSIVE Real-Time Scalping Engine")
    print("=" * 60)
    print("⚠️  HIGH FREQUENCY TRADING MODE")
    print(f"Symbol: {engine.symbol}")
    print(f"Strategy: {engine.min_profit_target:.1%} profit, {engine.stop_loss:.2%} stop")
    print(f"Position Size: ${engine.position_size:,}")
    print(f"Max Hold Time: {engine.max_hold_time}s")
    print("=" * 60)
    
    # 10分間のアグレッシブセッション
    await engine.run_aggressive_session(duration_minutes=10)


if __name__ == "__main__":
    asyncio.run(main())