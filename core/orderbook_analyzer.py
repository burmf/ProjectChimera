#!/usr/bin/env python3
"""
Bitget オーダーブック・スプレッド分析システム
スキャルピング用マーケットマイクロストラクチャー分析
"""

import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import statistics
from concurrent.futures import ThreadPoolExecutor
import json

from core.bitget_rest_client import BitgetRestClient

class OrderbookAnalyzer:
    """オーダーブック分析・スプレッド計算システム"""
    
    def __init__(self):
        self.client = BitgetRestClient()
        self.logger = logging.getLogger(__name__)
        
        # スプレッド分析対象ペア
        self.liquid_pairs = [
            'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'XRPUSDT', 
            'SOLUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT'
        ]
        
    def collect_orderbook_snapshots(self, pairs: List[str], duration_minutes: int = 60, 
                                   interval_seconds: int = 10) -> Dict[str, List[Dict]]:
        """
        指定期間中のオーダーブックスナップショットを収集
        
        Args:
            pairs: 通貨ペアリスト
            duration_minutes: 収集期間（分）
            interval_seconds: スナップショット間隔（秒）
        
        Returns:
            {pair: [orderbook_snapshots]}
        """
        self.logger.info(f"オーダーブック収集開始: {len(pairs)}ペア, {duration_minutes}分間")
        
        snapshots = {pair: [] for pair in pairs}
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        snapshot_count = 0
        
        while time.time() < end_time:
            round_start = time.time()
            
            # 各ペアのオーダーブックを並列取得
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(self._get_enhanced_orderbook, pair): pair 
                    for pair in pairs
                }
                
                for future in futures:
                    pair = futures[future]
                    try:
                        orderbook_data = future.result(timeout=5)
                        if orderbook_data:
                            snapshots[pair].append({
                                'timestamp': datetime.now(),
                                'data': orderbook_data
                            })
                    except Exception as e:
                        self.logger.warning(f"オーダーブック取得エラー {pair}: {e}")
            
            snapshot_count += 1
            
            # 進捗ログ
            if snapshot_count % 10 == 0:
                elapsed = (time.time() - start_time) / 60
                self.logger.info(f"スナップショット収集中: {snapshot_count}回, {elapsed:.1f}分経過")
            
            # 次のスナップショットまで待機
            elapsed_round = time.time() - round_start
            sleep_time = max(0, interval_seconds - elapsed_round)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # 結果サマリー
        total_snapshots = sum(len(snaps) for snaps in snapshots.values())
        self.logger.info(f"オーダーブック収集完了: {total_snapshots}スナップショット")
        
        return snapshots
    
    def _get_enhanced_orderbook(self, pair: str) -> Optional[Dict]:
        """強化されたオーダーブック情報を取得"""
        try:
            orderbook = self.client.get_orderbook_snapshot(pair, limit=50)
            ticker = self.client.get_ticker_24hr(pair)
            
            if not orderbook or not ticker:
                return None
            
            bids = [[float(bid[0]), float(bid[1])] for bid in orderbook.get('bids', [])]
            asks = [[float(ask[0]), float(ask[1])] for ask in orderbook.get('asks', [])]
            
            if not bids or not asks:
                return None
            
            # 基本スプレッド情報
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            spread = best_ask - best_bid
            spread_pct = (spread / best_bid) * 100
            
            # 深度分析
            depth_analysis = self._analyze_order_depth(bids, asks)
            
            # 流動性分析
            liquidity_analysis = self._analyze_liquidity(bids, asks, best_bid, best_ask)
            
            return {
                'pair': pair,
                'best_bid': best_bid,
                'best_ask': best_ask,
                'spread': spread,
                'spread_pct': spread_pct,
                'mid_price': (best_bid + best_ask) / 2,
                'depth_analysis': depth_analysis,
                'liquidity_analysis': liquidity_analysis,
                '24h_volume': float(ticker.get('baseVolume', 0)),
                '24h_change': float(ticker.get('change', 0))
            }
            
        except Exception as e:
            self.logger.error(f"拡張オーダーブック取得エラー {pair}: {e}")
            return None
    
    def _analyze_order_depth(self, bids: List[List[float]], asks: List[List[float]]) -> Dict:
        """オーダー深度分析"""
        try:
            # 各レベルでの累積出来高計算
            bid_cumulative = []
            ask_cumulative = []
            
            bid_volume = 0
            for price, volume in bids[:20]:  # Top 20 levels
                bid_volume += volume
                bid_cumulative.append([price, bid_volume])
            
            ask_volume = 0
            for price, volume in asks[:20]:
                ask_volume += volume
                ask_cumulative.append([price, ask_volume])
            
            # 深度指標計算
            bid_depth_5 = sum(vol for _, vol in bids[:5])
            ask_depth_5 = sum(vol for _, vol in asks[:5])
            
            bid_depth_10 = sum(vol for _, vol in bids[:10])
            ask_depth_10 = sum(vol for _, vol in asks[:10])
            
            return {
                'bid_depth_5': bid_depth_5,
                'ask_depth_5': ask_depth_5,
                'bid_depth_10': bid_depth_10,
                'ask_depth_10': ask_depth_10,
                'depth_imbalance_5': (bid_depth_5 - ask_depth_5) / (bid_depth_5 + ask_depth_5),
                'depth_imbalance_10': (bid_depth_10 - ask_depth_10) / (bid_depth_10 + ask_depth_10),
                'bid_levels': len(bids),
                'ask_levels': len(asks)
            }
            
        except Exception as e:
            self.logger.error(f"深度分析エラー: {e}")
            return {}
    
    def _analyze_liquidity(self, bids: List[List[float]], asks: List[List[float]], 
                          best_bid: float, best_ask: float) -> Dict:
        """流動性分析"""
        try:
            # 価格レンジ内の流動性計算
            mid_price = (best_bid + best_ask) / 2
            
            # 0.1%, 0.5%, 1.0%以内の流動性
            ranges = [0.001, 0.005, 0.01]  # 0.1%, 0.5%, 1.0%
            liquidity_metrics = {}
            
            for range_pct in ranges:
                range_str = f"{range_pct*100:.1f}pct"
                
                # Bid側流動性
                bid_range_min = mid_price * (1 - range_pct)
                bid_liquidity = sum(vol for price, vol in bids if price >= bid_range_min)
                
                # Ask側流動性
                ask_range_max = mid_price * (1 + range_pct)
                ask_liquidity = sum(vol for price, vol in asks if price <= ask_range_max)
                
                liquidity_metrics[f'bid_liquidity_{range_str}'] = bid_liquidity
                liquidity_metrics[f'ask_liquidity_{range_str}'] = ask_liquidity
                liquidity_metrics[f'total_liquidity_{range_str}'] = bid_liquidity + ask_liquidity
            
            return liquidity_metrics
            
        except Exception as e:
            self.logger.error(f"流動性分析エラー: {e}")
            return {}
    
    def analyze_spread_patterns(self, snapshots: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """スプレッドパターン分析"""
        self.logger.info("スプレッドパターン分析開始")
        
        analysis_results = {}
        
        for pair, snap_list in snapshots.items():
            if not snap_list:
                continue
            
            # データフレーム化
            spread_data = []
            for snap in snap_list:
                data = snap['data']
                spread_data.append({
                    'timestamp': snap['timestamp'],
                    'spread': data['spread'],
                    'spread_pct': data['spread_pct'],
                    'depth_imbalance_5': data['depth_analysis'].get('depth_imbalance_5', 0),
                    'bid_depth_5': data['depth_analysis'].get('bid_depth_5', 0),
                    'ask_depth_5': data['depth_analysis'].get('ask_depth_5', 0),
                    'volume_24h': data['24h_volume']
                })
            
            df = pd.DataFrame(spread_data)
            
            if df.empty:
                continue
            
            # 統計分析
            spread_stats = {
                'avg_spread': df['spread'].mean(),
                'median_spread': df['spread'].median(),
                'min_spread': df['spread'].min(),
                'max_spread': df['spread'].max(),
                'std_spread': df['spread'].std(),
                'avg_spread_pct': df['spread_pct'].mean(),
                'median_spread_pct': df['spread_pct'].median()
            }
            
            # 時間別分析
            df['hour'] = df['timestamp'].dt.hour
            hourly_spread = df.groupby('hour')['spread_pct'].agg(['mean', 'std', 'count'])
            
            # 深度不均衡分析
            imbalance_correlation = df['depth_imbalance_5'].corr(df['spread_pct'])
            
            # 最適取引条件特定
            optimal_conditions = self._find_optimal_scalping_conditions(df)
            
            analysis_results[pair] = {
                'spread_statistics': spread_stats,
                'hourly_patterns': hourly_spread.to_dict(),
                'imbalance_correlation': imbalance_correlation,
                'optimal_conditions': optimal_conditions,
                'sample_count': len(df)
            }
            
            self.logger.info(f"{pair} スプレッド分析完了: {len(df)}サンプル")
        
        return analysis_results
    
    def _find_optimal_scalping_conditions(self, df: pd.DataFrame) -> Dict:
        """最適スキャルピング条件を特定"""
        try:
            # 低スプレッド条件（下位25%）
            spread_q25 = df['spread_pct'].quantile(0.25)
            low_spread_data = df[df['spread_pct'] <= spread_q25]
            
            # 高流動性条件（深度が豊富）
            high_liquidity_threshold = df['bid_depth_5'].quantile(0.75)
            high_liquidity_data = df[df['bid_depth_5'] >= high_liquidity_threshold]
            
            # 最適条件の組み合わせ
            optimal_data = df[
                (df['spread_pct'] <= spread_q25) & 
                (df['bid_depth_5'] >= high_liquidity_threshold)
            ]
            
            if len(optimal_data) > 0:
                optimal_hours = optimal_data['hour'].value_counts().head(6).index.tolist()
                avg_optimal_spread = optimal_data['spread_pct'].mean()
            else:
                optimal_hours = []
                avg_optimal_spread = df['spread_pct'].mean()
            
            return {
                'optimal_spread_threshold': spread_q25,
                'optimal_liquidity_threshold': high_liquidity_threshold,
                'optimal_hours': sorted(optimal_hours),
                'optimal_conditions_count': len(optimal_data),
                'avg_optimal_spread': avg_optimal_spread,
                'improvement_factor': df['spread_pct'].mean() / avg_optimal_spread if avg_optimal_spread > 0 else 1.0
            }
            
        except Exception as e:
            self.logger.error(f"最適条件特定エラー: {e}")
            return {}
    
    def generate_scalping_signals(self, current_orderbook: Dict, 
                                 historical_patterns: Dict) -> Dict:
        """現在のオーダーブック状況からスキャルピングシグナルを生成"""
        try:
            pair = current_orderbook.get('pair')
            if not pair or pair not in historical_patterns:
                return {'signal': 'NO_DATA', 'confidence': 0.0}
            
            patterns = historical_patterns[pair]
            current_spread = current_orderbook.get('spread_pct', float('inf'))
            current_imbalance = current_orderbook.get('depth_analysis', {}).get('depth_imbalance_5', 0)
            current_hour = datetime.now().hour
            
            # シグナル計算
            signal_score = 0.0
            confidence_factors = []
            
            # 1. スプレッド条件チェック
            optimal_spread = patterns.get('optimal_conditions', {}).get('optimal_spread_threshold')
            if optimal_spread and current_spread <= optimal_spread:
                signal_score += 3.0
                confidence_factors.append('LOW_SPREAD')
            
            # 2. 時間帯チェック
            optimal_hours = patterns.get('optimal_conditions', {}).get('optimal_hours', [])
            if current_hour in optimal_hours:
                signal_score += 2.0
                confidence_factors.append('OPTIMAL_HOUR')
            
            # 3. 深度不均衡チェック
            if abs(current_imbalance) > 0.1:  # 10%以上の不均衡
                if current_imbalance > 0:
                    signal_score += 1.5
                    confidence_factors.append('BUY_PRESSURE')
                else:
                    signal_score += 1.5
                    confidence_factors.append('SELL_PRESSURE')
            
            # 4. スプレッド安定性チェック
            avg_spread = patterns.get('spread_statistics', {}).get('avg_spread_pct', float('inf'))
            if current_spread < avg_spread * 0.8:  # 平均より20%低い
                signal_score += 1.0
                confidence_factors.append('STABLE_SPREAD')
            
            # シグナル判定
            if signal_score >= 4.0:
                signal = 'STRONG_BUY' if current_imbalance > 0.05 else 'STRONG_SELL' if current_imbalance < -0.05 else 'SCALP_READY'
            elif signal_score >= 2.5:
                signal = 'WEAK_BUY' if current_imbalance > 0.02 else 'WEAK_SELL' if current_imbalance < -0.02 else 'MONITOR'
            else:
                signal = 'WAIT'
            
            confidence = min(signal_score / 6.0, 1.0)  # 最大6点で正規化
            
            return {
                'signal': signal,
                'confidence': confidence,
                'score': signal_score,
                'factors': confidence_factors,
                'current_spread_pct': current_spread,
                'depth_imbalance': current_imbalance,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"シグナル生成エラー: {e}")
            return {'signal': 'ERROR', 'confidence': 0.0}