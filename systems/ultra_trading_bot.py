#!/usr/bin/env python3
"""
Ultra High-Performance Trading Bot
超高性能自動取引ボット - 複数ペア同時運用
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
import statistics
import threading
import signal

# Add core modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.bitget_futures_client import BitgetFuturesClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UltraTradingBot:
    """
    超高性能自動取引ボット
    
    - 複数ペア同時取引
    - 高レバレッジ戦略
    - リアルタイムリスク管理
    - AI最適化アルゴリズム
    """
    
    def __init__(self):
        self.futures_client = BitgetFuturesClient()
        
        # ボット設定
        self.trading_pairs = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
        self.base_leverage = 30  # 30倍レバレッジ
        self.position_size_usd = 40000  # $40k per position
        self.max_positions = 8  # 最大8ポジション同時
        
        # 戦略パラメータ（最適化済み）
        self.profit_target = 0.006  # 0.6%利確
        self.stop_loss = 0.002      # 0.2%損切り
        self.confidence_threshold = 0.7
        self.momentum_threshold = 0.0006
        self.volatility_min = 0.0008
        
        # データ管理
        self.price_data = {}
        self.active_positions = {}
        self.trade_history = []
        self.performance_metrics = {}
        
        # パフォーマンス追跡
        self.start_balance = 100000
        self.current_balance = 100000
        self.total_profit = 0
        self.trades_today = 0
        self.wins_today = 0
        self.start_time = datetime.now()
        
        # ボット状態
        self.is_running = False
        self.shutdown_requested = False
        
        # ログ設定
        self.setup_logging()
        
        logger.info("Ultra Trading Bot initialized")
        logger.info(f"Trading pairs: {self.trading_pairs}")
        logger.info(f"Base leverage: {self.base_leverage}x")
        logger.info(f"Position size: ${self.position_size_usd:,}")
        logger.info(f"Max positions: {self.max_positions}")
    
    def setup_logging(self):
        """詳細ログ設定"""
        # ファイルハンドラー追加
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = f'ultra_bot_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        
        self.log_filename = log_filename
        logger.info(f"Detailed logging to: {log_filename}")
    
    def setup_signal_handlers(self):
        """シグナルハンドラー設定（安全な終了）"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating safe shutdown...")
            self.shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def collect_market_data(self) -> Dict[str, Any]:
        """リアルタイム市場データ収集"""
        market_data = {}
        
        for symbol in self.trading_pairs:
            try:
                ticker = self.futures_client.get_futures_ticker(symbol)
                if ticker:
                    market_data[symbol] = {
                        'price': ticker['price'],
                        'change_24h': ticker['change_24h'],
                        'ask': ticker['ask_price'],
                        'bid': ticker['bid_price'],
                        'spread': ticker['ask_price'] - ticker['bid_price'],
                        'timestamp': datetime.now()
                    }
                    
                    # 価格履歴更新
                    if symbol not in self.price_data:
                        self.price_data[symbol] = deque(maxlen=100)
                    
                    self.price_data[symbol].append({
                        'time': datetime.now(),
                        'price': ticker['price']
                    })
                    
            except Exception as e:
                logger.error(f"Error collecting data for {symbol}: {e}")
        
        return market_data
    
    def calculate_advanced_signals(self, symbol: str, market_data: Dict) -> Dict[str, Any]:
        """高度なAIシグナル生成"""
        if symbol not in self.price_data or len(self.price_data[symbol]) < 30:
            return {'action': 'hold', 'confidence': 0, 'reasoning': 'Insufficient data'}
        
        prices = [p['price'] for p in list(self.price_data[symbol])]
        current_price = prices[-1]
        
        # 複数時間軸分析
        momentum_short = (prices[-1] - prices[-3]) / prices[-3] if len(prices) >= 3 else 0
        momentum_medium = (prices[-1] - prices[-10]) / prices[-10] if len(prices) >= 10 else 0
        momentum_long = (prices[-1] - prices[-20]) / prices[-20] if len(prices) >= 20 else 0
        
        # ボラティリティ分析
        returns = [(prices[i] / prices[i-1] - 1) for i in range(1, min(len(prices), 30))]
        volatility = statistics.stdev(returns) if len(returns) > 1 else 0
        
        # トレンド強度
        if len(prices) >= 20:
            x = list(range(20))
            y = prices[-20:]
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            if n * sum_x2 - sum_x ** 2 != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
                trend_strength = abs(slope) / (sum_y / n)
            else:
                trend_strength = 0
        else:
            trend_strength = 0
        
        # スプレッド分析
        spread_pct = market_data[symbol]['spread'] / current_price if symbol in market_data else 0.001
        
        signal = {
            'action': 'hold',
            'confidence': 0,
            'reasoning': '',
            'leverage': self.base_leverage,
            'position_size': self.position_size_usd
        }
        
        # 高性能シグナル生成
        if volatility >= self.volatility_min and spread_pct < 0.0002:  # 優良流動性
            
            # 強い上昇トレンド
            if (momentum_short > self.momentum_threshold and 
                momentum_medium > 0.002 and 
                trend_strength > 0.0001):
                
                signal['action'] = 'long'
                signal['confidence'] = min(0.95, 
                    momentum_short * 800 + 
                    momentum_medium * 300 + 
                    trend_strength * 1000
                )
                signal['reasoning'] = f'Strong uptrend: short={momentum_short:.4f}, med={momentum_medium:.4f}'
                signal['target'] = current_price * (1 + self.profit_target)
                signal['stop'] = current_price * (1 - self.stop_loss)
                
                # ボラティリティに応じてレバレッジ調整
                if volatility > 0.004:
                    signal['leverage'] = min(50, self.base_leverage * 1.5)
            
            # 強い下降トレンド
            elif (momentum_short < -self.momentum_threshold and 
                  momentum_medium < -0.002 and 
                  trend_strength > 0.0001):
                
                signal['action'] = 'short'
                signal['confidence'] = min(0.95, 
                    abs(momentum_short) * 800 + 
                    abs(momentum_medium) * 300 + 
                    trend_strength * 1000
                )
                signal['reasoning'] = f'Strong downtrend: short={momentum_short:.4f}, med={momentum_medium:.4f}'
                signal['target'] = current_price * (1 - self.profit_target)
                signal['stop'] = current_price * (1 + self.stop_loss)
                
                if volatility > 0.004:
                    signal['leverage'] = min(50, self.base_leverage * 1.5)
            
            # 平均回帰戦略（高ボラティリティ環境）
            elif volatility > 0.006 and abs(momentum_short) > 0.004:
                if momentum_short > 0 and momentum_medium < 0:
                    signal['action'] = 'short'
                    signal['target'] = current_price * (1 - self.profit_target * 0.7)
                    signal['reasoning'] = 'Mean reversion: overbought correction'
                elif momentum_short < 0 and momentum_medium > 0:
                    signal['action'] = 'long'
                    signal['target'] = current_price * (1 + self.profit_target * 0.7)
                    signal['reasoning'] = 'Mean reversion: oversold bounce'
                
                if signal['action'] != 'hold':
                    signal['confidence'] = 0.8
                    signal['stop'] = current_price * (1 + self.stop_loss if signal['action'] == 'short' else 1 - self.stop_loss)
                    signal['leverage'] = min(40, self.base_leverage * 1.3)
        
        return signal
    
    async def execute_trade(self, symbol: str, signal: Dict) -> Optional[Dict]:
        """取引実行（仮想取引）"""
        if len(self.active_positions) >= self.max_positions:
            logger.warning(f"Max positions reached ({self.max_positions}), skipping {symbol}")
            return None
        
        trade_id = f"{symbol}_{int(time.time() * 1000)}"
        
        trade = {
            'id': trade_id,
            'symbol': symbol,
            'action': signal['action'],
            'entry_price': signal.get('target', 0) or list(self.price_data[symbol])[-1]['price'],
            'target_price': signal.get('target', 0),
            'stop_price': signal.get('stop', 0),
            'position_size': signal['position_size'],
            'leverage': signal['leverage'],
            'confidence': signal['confidence'],
            'reasoning': signal['reasoning'],
            'entry_time': datetime.now(),
            'status': 'active'
        }
        
        self.active_positions[trade_id] = trade
        self.trades_today += 1
        
        logger.info(f"🎯 TRADE #{self.trades_today}: {signal['action'].upper()} {symbol}")
        logger.info(f"   Entry: ${trade['entry_price']:,.2f} | Target: ${trade['target_price']:,.2f}")
        logger.info(f"   Leverage: {trade['leverage']}x | Confidence: {signal['confidence']:.1%}")
        logger.info(f"   Reasoning: {signal['reasoning']}")
        
        return trade
    
    async def manage_positions(self, market_data: Dict) -> List[Dict]:
        """ポジション管理・決済"""
        completed_trades = []
        
        for trade_id, trade in list(self.active_positions.items()):
            symbol = trade['symbol']
            if symbol not in market_data:
                continue
            
            current_price = market_data[symbol]['price']
            entry_price = trade['entry_price']
            
            # P&L計算
            if trade['action'] == 'long':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price
            
            pnl_amount = trade['position_size'] * trade['leverage'] * pnl_pct
            
            # 決済条件チェック
            should_close = False
            close_reason = ''
            
            # 利確チェック
            if trade['action'] == 'long' and current_price >= trade['target_price']:
                should_close = True
                close_reason = 'profit_target'
            elif trade['action'] == 'short' and current_price <= trade['target_price']:
                should_close = True
                close_reason = 'profit_target'
            
            # ストップロス
            elif trade['action'] == 'long' and current_price <= trade['stop_price']:
                should_close = True
                close_reason = 'stop_loss'
            elif trade['action'] == 'short' and current_price >= trade['stop_price']:
                should_close = True
                close_reason = 'stop_loss'
            
            # 時間制限（3分）
            elif datetime.now() - trade['entry_time'] > timedelta(minutes=3):
                should_close = True
                close_reason = 'timeout'
            
            # 緊急損切り（-1%以下）
            elif pnl_pct < -0.01:
                should_close = True
                close_reason = 'emergency_stop'
            
            if should_close:
                # 決済処理
                trade['exit_price'] = current_price
                trade['exit_time'] = datetime.now()
                trade['pnl_amount'] = pnl_amount
                trade['pnl_pct'] = pnl_pct
                trade['close_reason'] = close_reason
                trade['duration'] = trade['exit_time'] - trade['entry_time']
                
                # 統計更新
                self.total_profit += pnl_amount
                self.current_balance += pnl_amount
                
                if pnl_amount > 0:
                    self.wins_today += 1
                
                # 履歴追加
                self.trade_history.append(trade)
                completed_trades.append(trade)
                
                # アクティブポジションから削除
                del self.active_positions[trade_id]
                
                # ログ出力
                status_emoji = "✅" if pnl_amount > 0 else "❌"
                duration_sec = trade['duration'].total_seconds()
                
                logger.info(f"{status_emoji} CLOSED: {close_reason.upper()}")
                logger.info(f"   Symbol: {symbol} | Exit: ${current_price:,.2f}")
                logger.info(f"   P&L: {pnl_pct:+.3%} (${pnl_amount:+,.2f})")
                logger.info(f"   Duration: {duration_sec:.0f}s")
        
        return completed_trades
    
    def calculate_performance(self) -> Dict[str, Any]:
        """パフォーマンス計算"""
        if not self.trade_history:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'current_balance': self.current_balance,
                'daily_return': 0,
                'sharpe_ratio': 0
            }
        
        total_trades = len(self.trade_history)
        winning_trades = sum(1 for t in self.trade_history if t['pnl_amount'] > 0)
        win_rate = winning_trades / total_trades
        
        # 日次リターン推計
        elapsed_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        daily_return = (self.total_profit / self.start_balance) * (24 / max(0.1, elapsed_hours))
        
        # 簡易シャープレシオ
        if total_trades > 1:
            returns = [t['pnl_amount'] / self.start_balance for t in self.trade_history]
            mean_return = statistics.mean(returns)
            return_std = statistics.stdev(returns)
            sharpe_ratio = (mean_return * 252) / (return_std * (252 ** 0.5)) if return_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_profit': self.total_profit,
            'current_balance': self.current_balance,
            'daily_return': daily_return,
            'sharpe_ratio': sharpe_ratio,
            'active_positions': len(self.active_positions)
        }
    
    def print_status(self):
        """ステータス表示"""
        performance = self.calculate_performance()
        elapsed = datetime.now() - self.start_time
        
        print(f"\\n{'='*60}")
        print(f"🤖 ULTRA TRADING BOT STATUS | Runtime: {elapsed}")
        print(f"{'='*60}")
        print(f"💰 Balance: ${performance['current_balance']:,.2f} | Profit: ${performance['total_profit']:+,.2f}")
        print(f"📊 Trades: {performance['total_trades']} | Win Rate: {performance['win_rate']:.1%}")
        print(f"📈 Daily Return: {performance['daily_return']:+.2%} | Sharpe: {performance['sharpe_ratio']:.2f}")
        print(f"⚡ Active Positions: {performance['active_positions']}/{self.max_positions}")
        
        if self.active_positions:
            print(f"\\n🎯 Active Positions:")
            for trade in self.active_positions.values():
                duration = (datetime.now() - trade['entry_time']).total_seconds() / 60
                print(f"   {trade['symbol']} {trade['action'].upper()} | {duration:.1f}min | {trade['leverage']}x")
        
        print(f"{'='*60}")
    
    def save_performance_log(self):
        """パフォーマンスログ保存"""
        performance = self.calculate_performance()
        
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'runtime_minutes': (datetime.now() - self.start_time).total_seconds() / 60,
            'performance': performance,
            'trading_pairs': self.trading_pairs,
            'strategy_params': {
                'base_leverage': self.base_leverage,
                'position_size': self.position_size_usd,
                'profit_target': self.profit_target,
                'stop_loss': self.stop_loss,
                'max_positions': self.max_positions
            },
            'trade_history': [
                {
                    'symbol': t['symbol'],
                    'action': t['action'],
                    'entry_time': t['entry_time'].isoformat(),
                    'exit_time': t['exit_time'].isoformat(),
                    'pnl_amount': t['pnl_amount'],
                    'pnl_pct': t['pnl_pct'],
                    'duration_seconds': t['duration'].total_seconds(),
                    'close_reason': t['close_reason'],
                    'leverage': t['leverage']
                } for t in self.trade_history
            ]
        }
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'ultra_bot_performance_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        logger.info(f"Performance log saved: {filename}")
        return filename
    
    async def run_trading_session(self, duration_minutes: int = 60):
        """メイン取引セッション実行"""
        self.setup_signal_handlers()
        self.is_running = True
        
        logger.info(f"🚀 Starting Ultra Trading Bot session for {duration_minutes} minutes")
        logger.info(f"Strategy: {self.base_leverage}x leverage, ${self.position_size_usd:,} positions")
        
        end_time = datetime.now() + timedelta(minutes=duration_minutes)
        last_status_time = datetime.now()
        
        try:
            while datetime.now() < end_time and not self.shutdown_requested:
                # 市場データ収集
                market_data = await self.collect_market_data()
                
                if not market_data:
                    logger.warning("No market data available, retrying...")
                    await asyncio.sleep(5)
                    continue
                
                # 既存ポジション管理
                completed = await self.manage_positions(market_data)
                
                # 新規シグナル生成・実行
                for symbol in self.trading_pairs:
                    if symbol in market_data:
                        signal = self.calculate_advanced_signals(symbol, market_data)
                        
                        if (signal['action'] != 'hold' and 
                            signal['confidence'] > self.confidence_threshold and
                            len(self.active_positions) < self.max_positions):
                            
                            await self.execute_trade(symbol, signal)
                            await asyncio.sleep(1)  # 少し間隔を空ける
                
                # 定期ステータス表示（30秒ごと）
                if datetime.now() - last_status_time > timedelta(seconds=30):
                    self.print_status()
                    last_status_time = datetime.now()
                
                # 1秒待機
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            # 安全な終了処理
            await self.safe_shutdown()
    
    async def safe_shutdown(self):
        """安全な終了処理"""
        logger.info("🛑 Initiating safe shutdown...")
        
        # 全ポジション決済
        if self.active_positions:
            logger.info(f"Closing {len(self.active_positions)} active positions...")
            
            # 最新市場データ取得
            market_data = await self.collect_market_data()
            
            # 強制決済
            for trade_id, trade in list(self.active_positions.items()):
                symbol = trade['symbol']
                if symbol in market_data:
                    current_price = market_data[symbol]['price']
                    
                    # 決済処理
                    if trade['action'] == 'long':
                        pnl_pct = (current_price - trade['entry_price']) / trade['entry_price']
                    else:
                        pnl_pct = (trade['entry_price'] - current_price) / trade['entry_price']
                    
                    pnl_amount = trade['position_size'] * trade['leverage'] * pnl_pct
                    
                    trade['exit_price'] = current_price
                    trade['exit_time'] = datetime.now()
                    trade['pnl_amount'] = pnl_amount
                    trade['pnl_pct'] = pnl_pct
                    trade['close_reason'] = 'session_end'
                    trade['duration'] = trade['exit_time'] - trade['entry_time']
                    
                    self.total_profit += pnl_amount
                    self.current_balance += pnl_amount
                    
                    if pnl_amount > 0:
                        self.wins_today += 1
                    
                    self.trade_history.append(trade)
                    
                    logger.info(f"Closed {symbol} {trade['action']}: ${pnl_amount:+,.2f}")
            
            self.active_positions.clear()
        
        # 最終結果表示
        self.print_status()
        
        # パフォーマンスログ保存
        log_file = self.save_performance_log()
        
        logger.info("🏁 Ultra Trading Bot session completed successfully")
        logger.info(f"📄 Detailed logs: {self.log_filename}")
        logger.info(f"📊 Performance data: {log_file}")
        
        self.is_running = False


async def main():
    """メイン実行"""
    bot = UltraTradingBot()
    
    print("🤖 Ultra High-Performance Trading Bot")
    print("=" * 60)
    print(f"Strategy: {bot.base_leverage}x leverage, ultra-high frequency")
    print(f"Pairs: {', '.join(bot.trading_pairs)}")
    print(f"Max positions: {bot.max_positions}")
    print(f"Target profit: {bot.profit_target:.1%} per trade")
    print("=" * 60)
    
    # 60分間の自動取引セッション
    await bot.run_trading_session(duration_minutes=60)


if __name__ == "__main__":
    asyncio.run(main())