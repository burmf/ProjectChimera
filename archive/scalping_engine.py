#!/usr/bin/env python3
"""
Bitget Real-Time Scalping Engine
リアルタイムスキャルピングエンジン
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


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BitgetScalpingEngine:
    """
    Bitgetリアルタイムスキャルピングエンジン
    
    - 1分足データでのスキャルピング
    - 最小0.5%の価格変動を狙う
    - リスク管理: 0.2%ストップロス
    - 利確: 0.5-1.0%
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 取引設定
        self.symbol = 'BTCUSDT_SPBL'
        self.base_url = 'https://api.bitget.com'
        
        # スキャルピング戦略設定
        self.min_profit_target = 0.005  # 0.5%最小利益目標
        self.max_profit_target = 0.01   # 1.0%最大利益目標
        self.stop_loss = 0.002          # 0.2%ストップロス
        self.position_size = 10000      # $10,000ポジション
        
        # データ管理
        self.price_history = deque(maxlen=100)  # 最新100価格
        self.trades = []
        self.current_position = None
        self.account_balance = 100000  # $100k仮想残高
        
        # パフォーマンス追跡
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0
        self.start_time = datetime.now()
        
        # AI分析用インディケータ
        self.momentum_window = 20
        self.volatility_window = 15
        
        self.logger.info("Bitget Scalping Engine initialized")
    
    async def get_realtime_price(self) -> Optional[float]:
        """リアルタイム価格取得"""
        try:
            response = requests.get(
                f'{self.base_url}/api/spot/v1/market/ticker?symbol={self.symbol}',
                timeout=2
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
    
    def calculate_technical_indicators(self) -> Dict[str, float]:
        """テクニカル指標計算"""
        if len(self.price_history) < self.momentum_window:
            return {'momentum': 0, 'volatility': 0, 'trend': 0}
        
        prices = list(self.price_history)
        
        # モメンタム計算（価格変化率）
        if len(prices) >= 2:
            momentum = (prices[-1] - prices[-5]) / prices[-5] if len(prices) >= 5 else 0
        else:
            momentum = 0
        
        # ボラティリティ計算
        if len(prices) >= self.volatility_window:
            returns = [(prices[i] - prices[i-1]) / prices[i-1] 
                      for i in range(1, min(len(prices), self.volatility_window))]
            volatility = statistics.stdev(returns) if len(returns) > 1 else 0
        else:
            volatility = 0
        
        # トレンド計算（移動平均比較）
        if len(prices) >= 10:
            short_ma = sum(prices[-5:]) / 5
            long_ma = sum(prices[-10:]) / 10
            trend = (short_ma - long_ma) / long_ma
        else:
            trend = 0
        
        return {
            'momentum': momentum,
            'volatility': volatility,
            'trend': trend
        }
    
    def generate_scalping_signal(self, current_price: float) -> Dict[str, Any]:
        """スキャルピングシグナル生成"""
        indicators = self.calculate_technical_indicators()
        
        # シグナル生成ロジック
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
        
        # ボラティリティチェック（十分な変動があるか）
        if volatility < 0.001:  # 0.1%未満の変動
            signal['reasoning'] = 'Low volatility - no trading opportunity'
            return signal
        
        # ロングシグナル条件
        if (momentum > 0.002 and trend > 0.001 and volatility > 0.002):
            signal['action'] = 'long'
            signal['confidence'] = min(0.9, (momentum + trend) * 100)
            signal['target_price'] = current_price * (1 + self.min_profit_target)
            signal['stop_loss_price'] = current_price * (1 - self.stop_loss)
            signal['reasoning'] = f'Bullish momentum: {momentum:.4f}, trend: {trend:.4f}'
        
        # ショートシグナル条件
        elif (momentum < -0.002 and trend < -0.001 and volatility > 0.002):
            signal['action'] = 'short'
            signal['confidence'] = min(0.9, abs(momentum + trend) * 100)
            signal['target_price'] = current_price * (1 - self.min_profit_target)
            signal['stop_loss_price'] = current_price * (1 + self.stop_loss)
            signal['reasoning'] = f'Bearish momentum: {momentum:.4f}, trend: {trend:.4f}'
        
        else:
            signal['reasoning'] = f'No clear signal - momentum: {momentum:.4f}, trend: {trend:.4f}, vol: {volatility:.4f}'
        
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
        
        self.logger.info(f"🎯 TRADE #{self.total_trades}: {signal['action'].upper()}")
        self.logger.info(f"   Entry: ${signal['entry_price']:,.2f}")
        self.logger.info(f"   Target: ${signal['target_price']:,.2f}")
        self.logger.info(f"   Stop: ${signal['stop_loss_price']:,.2f}")
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
        
        # タイムアウト（5分経過）
        if datetime.now() - pos['timestamp'] > timedelta(minutes=5):
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
        self.logger.info(f"{status_emoji} CLOSED: {reason.upper()}")
        self.logger.info(f"   Exit: ${exit_price:,.2f}")
        self.logger.info(f"   P&L: {profit_loss:+.2%} (${profit_amount:+,.2f})")
        self.logger.info(f"   Duration: {trade_result['duration']}")
        
        return trade_result
    
    def print_performance_stats(self):
        """パフォーマンス統計表示"""
        if self.total_trades == 0:
            return
        
        win_rate = self.winning_trades / self.total_trades
        avg_profit = self.total_profit / self.total_trades
        running_time = datetime.now() - self.start_time
        
        print(f"\n📊 SCALPING PERFORMANCE STATS")
        print(f"{'='*50}")
        print(f"Running Time: {running_time}")
        print(f"Total Trades: {self.total_trades}")
        print(f"Winning Trades: {self.winning_trades}")
        print(f"Win Rate: {win_rate:.1%}")
        print(f"Total P&L: ${self.total_profit:+,.2f}")
        print(f"Average P&L: ${avg_profit:+,.2f}")
        print(f"Account Balance: ${self.account_balance:,.2f}")
        print(f"Return: {(self.account_balance - 100000) / 100000:+.2%}")
        
        if self.trades:
            profits = [t['profit_loss_amount'] for t in self.trades if t['profit_loss_amount'] > 0]
            losses = [t['profit_loss_amount'] for t in self.trades if t['profit_loss_amount'] < 0]
            
            if profits and losses:
                avg_win = sum(profits) / len(profits)
                avg_loss = sum(losses) / len(losses)
                profit_factor = abs(sum(profits) / sum(losses))
                
                print(f"Average Win: ${avg_win:+.2f}")
                print(f"Average Loss: ${avg_loss:+.2f}")
                print(f"Profit Factor: {profit_factor:.2f}")
        
        print(f"{'='*50}")
    
    async def run_scalping_session(self, duration_minutes: int = 30):
        """スキャルピングセッション実行"""
        self.logger.info(f"🚀 Starting {duration_minutes}-minute scalping session")
        self.logger.info(f"Symbol: {self.symbol}")
        self.logger.info(f"Position Size: ${self.position_size:,}")
        self.logger.info(f"Profit Target: {self.min_profit_target:.1%}")
        self.logger.info(f"Stop Loss: {self.stop_loss:.1%}")
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        
        try:
            while datetime.now() < end_time:
                # 価格取得
                current_price = await self.get_realtime_price()
                
                if current_price is None:
                    await asyncio.sleep(1)
                    continue
                
                self.price_history.append(current_price)
                
                # 既存ポジションの管理
                if self.current_position:
                    exit_result = self.check_position_exit(current_price)
                    if exit_result:
                        # ポジション決済後は少し待機
                        await asyncio.sleep(5)
                
                # 新規エントリーシグナルチェック
                elif len(self.price_history) >= self.momentum_window:
                    signal = self.generate_scalping_signal(current_price)
                    
                    if signal['action'] != 'hold' and signal['confidence'] > 0.6:
                        self.execute_virtual_trade(signal)
                
                # 統計表示（5秒ごと）
                if int(time.time()) % 15 == 0:  # 15秒ごと
                    self.print_performance_stats()
                
                # 1秒待機
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Scalping session interrupted by user")
        
        # 最終ポジション決済
        if self.current_position:
            final_price = await self.get_realtime_price()
            if final_price:
                self.close_position(final_price, 'session_end')
        
        # 最終統計
        self.print_performance_stats()
        
        # セッション結果保存
        self.save_session_results()
    
    def save_session_results(self):
        """セッション結果保存"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'scalping_session_{timestamp}.json'
        
        session_data = {
            'timestamp': timestamp,
            'symbol': self.symbol,
            'duration': str(datetime.now() - self.start_time),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / self.total_trades if self.total_trades > 0 else 0,
            'total_profit': self.total_profit,
            'final_balance': self.account_balance,
            'return_pct': (self.account_balance - 100000) / 100000,
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
        
        self.logger.info(f"Session results saved to {filename}")


async def main():
    """メイン実行"""
    engine = BitgetScalpingEngine()
    
    print("🎯 Bitget Real-Time Scalping Engine")
    print("=" * 50)
    print("Ready to start scalping session")
    print(f"Symbol: {engine.symbol}")
    print(f"Strategy: 0.5% profit target, 0.2% stop loss")
    print(f"Position Size: ${engine.position_size:,}")
    print("=" * 50)
    
    # 5分間のテストセッション
    await engine.run_scalping_session(duration_minutes=5)


if __name__ == "__main__":
    asyncio.run(main())