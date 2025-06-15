#!/usr/bin/env python3
"""
Scalping Demo with Simulated Market Volatility
ボラティリティのあるシミュレート市場でのスキャルピングデモ
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
import random
import statistics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ScalpingDemo:
    """
    リアルタイムスキャルピングのデモンストレーション
    
    シミュレートされた価格変動でスキャルピング戦略をテスト
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 基準価格（実際のBTC価格に近い）
        self.base_price = 104936.47
        self.current_price = self.base_price
        
        # スキャルピング設定
        self.min_profit_target = 0.002   # 0.2%
        self.stop_loss = 0.001           # 0.1%
        self.position_size = 20000       # $20,000
        self.max_hold_time = 90          # 90秒
        
        # データ管理
        self.price_history = deque(maxlen=30)
        self.trades = []
        self.current_position = None
        self.account_balance = 100000
        
        # パフォーマンス追跡
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0
        self.start_time = datetime.now()
        
        # シミュレーション設定
        self.volatility = 0.001  # 0.1%基本ボラティリティ
        self.trend_strength = 0
        self.market_regime = 'normal'  # normal, trending, volatile
        
        self.logger.info("Scalping Demo initialized")
    
    def generate_realistic_price(self) -> float:
        """リアルな価格変動をシミュレート"""
        
        # 市場レジーム変更（5%の確率）
        if random.random() < 0.05:
            regimes = ['normal', 'trending', 'volatile']
            self.market_regime = random.choice(regimes)
            
            if self.market_regime == 'trending':
                self.trend_strength = random.choice([-0.002, 0.002])  # ±0.2%のトレンド
                self.volatility = 0.0008
            elif self.market_regime == 'volatile':
                self.trend_strength = 0
                self.volatility = 0.003  # 0.3%の高ボラティリティ
            else:  # normal
                self.trend_strength = 0
                self.volatility = 0.001
        
        # 価格変動生成
        random_change = random.gauss(0, self.volatility)
        trend_change = self.trend_strength
        
        # ノイズ追加
        noise = random.gauss(0, self.volatility * 0.3)
        
        # 価格更新
        total_change = random_change + trend_change + noise
        self.current_price *= (1 + total_change)
        
        # 価格の範囲制限（±5%）
        min_price = self.base_price * 0.95
        max_price = self.base_price * 1.05
        self.current_price = max(min_price, min(max_price, self.current_price))
        
        return self.current_price
    
    def calculate_scalping_indicators(self) -> Dict[str, float]:
        """スキャルピング用指標計算"""
        if len(self.price_history) < 5:
            return {'momentum': 0, 'volatility': 0, 'trend': 0}
        
        prices = list(self.price_history)
        
        # 短期モメンタム（直近3価格の変化）
        momentum = (prices[-1] - prices[-3]) / prices[-3] if len(prices) >= 3 else 0
        
        # ボラティリティ
        if len(prices) >= 10:
            returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, min(len(prices), 10))]
            volatility = statistics.stdev(returns) if len(returns) > 1 else 0
        else:
            volatility = 0
        
        # マイクロトレンド
        if len(prices) >= 5:
            trend = (prices[-1] - prices[-5]) / prices[-5]
        else:
            trend = 0
        
        return {
            'momentum': momentum,
            'volatility': volatility,
            'trend': trend
        }
    
    def generate_scalping_signal(self, current_price: float) -> Dict[str, Any]:
        """スキャルピングシグナル生成"""
        indicators = self.calculate_scalping_indicators()
        
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
        
        # ボラティリティ不足チェック
        if volatility < 0.0008:
            signal['reasoning'] = f'Low volatility: {volatility:.5f}'
            return signal
        
        # ロングシグナル
        if momentum > 0.0005 and trend > 0.0003:
            signal['action'] = 'long'
            signal['confidence'] = min(0.9, momentum * 500)
            signal['target_price'] = current_price * (1 + self.min_profit_target)
            signal['stop_loss_price'] = current_price * (1 - self.stop_loss)
            signal['reasoning'] = f'Bullish scalp: mom={momentum:.5f}, trend={trend:.5f}'
        
        # ショートシグナル
        elif momentum < -0.0005 and trend < -0.0003:
            signal['action'] = 'short'
            signal['confidence'] = min(0.9, abs(momentum) * 500)
            signal['target_price'] = current_price * (1 - self.min_profit_target)
            signal['stop_loss_price'] = current_price * (1 + self.stop_loss)
            signal['reasoning'] = f'Bearish scalp: mom={momentum:.5f}, trend={trend:.5f}'
        
        # 平均回帰シグナル（オーバーシュート狙い）
        elif abs(momentum) > 0.0015:
            if momentum > 0:  # 急上昇後の下落狙い
                signal['action'] = 'short'
                signal['confidence'] = 0.7
                signal['target_price'] = current_price * (1 - self.min_profit_target * 0.5)
                signal['stop_loss_price'] = current_price * (1 + self.stop_loss)
                signal['reasoning'] = f'Mean reversion short: mom={momentum:.5f}'
            else:  # 急下落後の上昇狙い
                signal['action'] = 'long'
                signal['confidence'] = 0.7
                signal['target_price'] = current_price * (1 + self.min_profit_target * 0.5)
                signal['stop_loss_price'] = current_price * (1 - self.stop_loss)
                signal['reasoning'] = f'Mean reversion long: mom={momentum:.5f}'
        
        else:
            signal['reasoning'] = f'No signal: mom={momentum:.5f}, vol={volatility:.5f}'
        
        return signal
    
    def execute_trade(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """取引実行"""
        trade_result = {
            'timestamp': datetime.now(),
            'action': signal['action'],
            'entry_price': signal['entry_price'],
            'position_size': self.position_size,
            'target_price': signal['target_price'],
            'stop_loss_price': signal['stop_loss_price'],
            'confidence': signal['confidence'],
            'reasoning': signal['reasoning'],
            'status': 'opened',
            'market_regime': self.market_regime
        }
        
        self.current_position = trade_result
        self.total_trades += 1
        
        profit_pct = ((signal['target_price'] - signal['entry_price']) / signal['entry_price']) * 100
        if signal['action'] == 'short':
            profit_pct = ((signal['entry_price'] - signal['target_price']) / signal['entry_price']) * 100
        
        self.logger.info(f"🎯 TRADE #{self.total_trades}: {signal['action'].upper()} [{self.market_regime}]")
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
        
        # タイムアウト
        if (datetime.now() - pos['timestamp']).total_seconds() > self.max_hold_time:
            return self.close_position(current_price, 'timeout')
        
        return None
    
    def close_position(self, exit_price: float, reason: str) -> Dict[str, Any]:
        """ポジション決済"""
        pos = self.current_position
        entry_price = pos['entry_price']
        
        if pos['action'] == 'long':
            profit_loss = (exit_price - entry_price) / entry_price
        else:
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
            'duration': datetime.now() - pos['timestamp'],
            'market_regime': pos['market_regime']
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
        running_time = datetime.now() - self.start_time
        minutes = running_time.total_seconds() / 60
        
        if self.total_trades > 0:
            win_rate = self.winning_trades / self.total_trades
            trades_per_minute = self.total_trades / minutes if minutes > 0 else 0
            print(f"\\n📊 LIVE | Regime: {self.market_regime} | Price: ${self.current_price:,.2f} | Trades: {self.total_trades} | Win: {win_rate:.1%} | P&L: ${self.total_profit:+,.2f} | Rate: {trades_per_minute:.1f}/min")
        else:
            print(f"\\n📊 LIVE | Regime: {self.market_regime} | Price: ${self.current_price:,.2f} | No trades yet")
    
    async def run_demo_session(self, duration_minutes: int = 5):
        """デモセッション実行"""
        self.logger.info(f"🎬 Starting {duration_minutes}-minute SCALPING DEMO")
        self.logger.info(f"Base Price: ${self.base_price:,.2f}")
        self.logger.info(f"Position Size: ${self.position_size:,}")
        self.logger.info(f"Profit Target: {self.min_profit_target:.1%}")
        self.logger.info(f"Stop Loss: {self.stop_loss:.1%}")
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        iteration = 0
        
        try:
            while datetime.now() < end_time:
                iteration += 1
                
                # 新しい価格生成
                current_price = self.generate_realistic_price()
                self.price_history.append(current_price)
                
                # 既存ポジション管理
                if self.current_position:
                    exit_result = self.check_position_exit(current_price)
                    if exit_result:
                        await asyncio.sleep(2)  # 決済後の小休止
                
                # 新規エントリーチェック
                elif len(self.price_history) >= 10:
                    signal = self.generate_scalping_signal(current_price)
                    
                    if signal['action'] != 'hold' and signal['confidence'] > 0.6:
                        self.execute_trade(signal)
                
                # 統計表示（30秒ごと）
                if iteration % 30 == 0:
                    self.print_live_stats()
                
                # 1秒間隔
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Demo session interrupted")
        
        # 最終決済
        if self.current_position:
            self.close_position(self.current_price, 'session_end')
        
        # 最終結果
        self.print_final_results()
        self.save_demo_results()
    
    def print_final_results(self):
        """最終結果表示"""
        running_time = datetime.now() - self.start_time
        
        print(f"\\n🎬 SCALPING DEMO FINAL RESULTS")
        print(f"{'='*50}")
        print(f"Session Duration: {running_time}")
        print(f"Starting Price: ${self.base_price:,.2f}")
        print(f"Final Price: ${self.current_price:,.2f}")
        print(f"Price Change: {((self.current_price - self.base_price) / self.base_price):+.2%}")
        print(f"")
        print(f"Total Trades: {self.total_trades}")
        
        if self.total_trades > 0:
            win_rate = self.winning_trades / self.total_trades
            avg_profit = self.total_profit / self.total_trades
            trades_per_minute = self.total_trades / (running_time.total_seconds() / 60)
            
            print(f"Winning Trades: {self.winning_trades}")
            print(f"Win Rate: {win_rate:.1%}")
            print(f"Trading Frequency: {trades_per_minute:.1f}/minute")
            print(f"Total P&L: ${self.total_profit:+,.2f}")
            print(f"Average P&L: ${avg_profit:+,.2f}")
            print(f"Final Balance: ${self.account_balance:,.2f}")
            print(f"Session Return: {(self.account_balance - 100000) / 100000:+.2%}")
            
            # 市場レジーム別統計
            regime_stats = {}
            for trade in self.trades:
                regime = trade['market_regime']
                if regime not in regime_stats:
                    regime_stats[regime] = {'count': 0, 'profit': 0, 'wins': 0}
                regime_stats[regime]['count'] += 1
                regime_stats[regime]['profit'] += trade['profit_loss_amount']
                if trade['profit_loss_amount'] > 0:
                    regime_stats[regime]['wins'] += 1
            
            print(f"\\nMarket Regime Performance:")
            for regime, stats in regime_stats.items():
                win_rate = stats['wins'] / stats['count'] if stats['count'] > 0 else 0
                print(f"  {regime}: {stats['count']} trades, {win_rate:.1%} win rate, ${stats['profit']:+.2f} P&L")
        
        print(f"{'='*50}")
    
    def save_demo_results(self):
        """デモ結果保存"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'scalping_demo_{timestamp}.json'
        
        session_data = {
            'timestamp': timestamp,
            'strategy': 'scalping_demo',
            'base_price': self.base_price,
            'final_price': self.current_price,
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
                    'duration_seconds': trade['duration'].total_seconds(),
                    'market_regime': trade['market_regime']
                }
                for trade in self.trades
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        self.logger.info(f"Demo results saved to {filename}")


async def main():
    """メイン実行"""
    demo = ScalpingDemo()
    
    print("🎬 Bitget Scalping Strategy DEMO")
    print("=" * 50)
    print("🔄 Simulated Market with Variable Volatility")
    print(f"Strategy: {demo.min_profit_target:.1%} profit, {demo.stop_loss:.1%} stop")
    print(f"Position Size: ${demo.position_size:,}")
    print(f"Max Hold: {demo.max_hold_time}s")
    print("=" * 50)
    
    # 5分間のデモセッション
    await demo.run_demo_session(duration_minutes=5)


if __name__ == "__main__":
    asyncio.run(main())