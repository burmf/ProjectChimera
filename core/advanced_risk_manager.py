#!/usr/bin/env python3
"""
Advanced Risk Management System
高度なリスク管理システム - 動的リスク調整・ポートフォリオ保護
"""

import json
import time
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
import statistics
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedRiskManager:
    """
    高度なリスク管理システム
    
    - 動的ポジションサイジング
    - ポートフォリオレベルVaR
    - 相関分析
    - ドローダウン保護
    - 市場レジーム検出
    """
    
    def __init__(self, initial_balance: float = 100000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        
        # リスク管理設定
        self.max_portfolio_risk = 0.15      # 15%最大ポートフォリオリスク
        self.max_daily_loss = 0.05          # 5%日次最大損失
        self.max_drawdown = 0.10            # 10%最大ドローダウン
        self.max_correlation = 0.7          # 70%最大相関
        self.kelly_fraction = 0.25          # ケリー基準の25%
        
        # VaR設定
        self.var_confidence = 0.95          # 95%信頼区間
        self.var_lookback = 50              # 50取引の履歴
        
        # データ追跡
        self.position_history = deque(maxlen=1000)
        self.daily_pnl = deque(maxlen=30)
        self.correlation_matrix = {}
        self.volatility_estimates = {}
        
        # 市場レジーム
        self.current_regime = 'normal'      # normal, volatile, trending
        self.regime_history = deque(maxlen=100)
        
        # アラート履歴
        self.alerts = deque(maxlen=100)
        
        logger.info("Advanced Risk Manager initialized")
        logger.info(f"Max portfolio risk: {self.max_portfolio_risk:.1%}")
        logger.info(f"Max daily loss: {self.max_daily_loss:.1%}")
        logger.info(f"Max drawdown: {self.max_drawdown:.1%}")
    
    def update_balance(self, new_balance: float, trade_pnl: float = 0):
        """残高更新とリスク追跡"""
        self.current_balance = new_balance
        
        if trade_pnl != 0:
            self.daily_pnl.append({
                'timestamp': datetime.now(),
                'pnl': trade_pnl,
                'balance': new_balance
            })
    
    def calculate_position_size(self, 
                              symbol: str,
                              signal_confidence: float,
                              expected_return: float,
                              risk_per_trade: float,
                              market_volatility: float,
                              current_positions: Dict) -> Tuple[float, str]:
        """
        動的ポジションサイジング計算
        
        Returns:
            (position_size, reasoning)
        """
        
        # 基本ポジションサイズ
        base_size = self.current_balance * 0.1  # 10%ベース
        
        # 1. ケリー基準調整
        if expected_return > 0 and risk_per_trade > 0:
            kelly_size = (signal_confidence * expected_return - risk_per_trade) / (risk_per_trade ** 2)
            kelly_size = max(0, min(kelly_size, 1)) * self.kelly_fraction
        else:
            kelly_size = 0.05
        
        # 2. 信頼度調整
        confidence_multiplier = signal_confidence ** 2  # 信頼度の二乗で調整
        
        # 3. ボラティリティ調整
        if market_volatility > 0:
            vol_adjustment = 0.02 / max(market_volatility, 0.005)  # 2%基準ボラティリティ
            vol_adjustment = min(vol_adjustment, 2.0)  # 最大2倍
        else:
            vol_adjustment = 1.0
        
        # 4. 相関調整
        correlation_adjustment = self._get_correlation_adjustment(symbol, current_positions)
        
        # 5. 市場レジーム調整
        regime_adjustment = self._get_regime_adjustment()
        
        # 6. ドローダウン保護
        drawdown_adjustment = self._get_drawdown_adjustment()
        
        # 最終ポジションサイズ計算
        adjusted_size = (base_size * 
                        kelly_size * 
                        confidence_multiplier * 
                        vol_adjustment * 
                        correlation_adjustment * 
                        regime_adjustment * 
                        drawdown_adjustment)
        
        # 制限適用
        max_position = self.current_balance * 0.3  # 最大30%
        adjusted_size = min(adjusted_size, max_position)
        
        reasoning = (f"Kelly: {kelly_size:.3f}, Conf: {confidence_multiplier:.3f}, "
                    f"Vol: {vol_adjustment:.3f}, Corr: {correlation_adjustment:.3f}, "
                    f"Regime: {regime_adjustment:.3f}, DD: {drawdown_adjustment:.3f}")
        
        return adjusted_size, reasoning
    
    def _get_correlation_adjustment(self, symbol: str, current_positions: Dict) -> float:
        """相関調整係数計算"""
        if not current_positions:
            return 1.0
        
        # 簡易相関マトリックス（実際の実装ではより精密な計算が必要）
        correlation_map = {
            ('BTCUSDT', 'ETHUSDT'): 0.8,
            ('BTCUSDT', 'SOLUSDT'): 0.7,
            ('ETHUSDT', 'SOLUSDT'): 0.75
        }
        
        max_correlation = 0
        for existing_symbol in current_positions:
            pair = tuple(sorted([symbol, existing_symbol]))
            correlation = correlation_map.get(pair, 0.3)  # デフォルト30%相関
            max_correlation = max(max_correlation, correlation)
        
        # 高相関ほどポジションサイズを削減
        if max_correlation > self.max_correlation:
            return 0.5  # 50%削減
        elif max_correlation > 0.5:
            return 1 - (max_correlation - 0.5) * 0.5  # 線形削減
        else:
            return 1.0
    
    def _get_regime_adjustment(self) -> float:
        """市場レジーム調整係数"""
        regime_adjustments = {
            'normal': 1.0,
            'volatile': 0.7,    # ボラタイル環境では30%削減
            'trending': 1.2,    # トレンド環境では20%増加
            'crisis': 0.3       # 危機時は70%削減
        }
        
        return regime_adjustments.get(self.current_regime, 1.0)
    
    def _get_drawdown_adjustment(self) -> float:
        """ドローダウン保護調整"""
        if not self.daily_pnl:
            return 1.0
        
        # 現在のドローダウン計算
        peak_balance = self.initial_balance
        for pnl_data in self.daily_pnl:
            peak_balance = max(peak_balance, pnl_data['balance'])
        
        current_drawdown = (peak_balance - self.current_balance) / peak_balance
        
        if current_drawdown > self.max_drawdown * 0.8:  # 80%に達したら警告
            self._add_alert('high_drawdown', f'Drawdown at {current_drawdown:.1%}')
            return 0.5  # 50%削減
        elif current_drawdown > self.max_drawdown * 0.6:  # 60%で注意
            return 0.7  # 30%削減
        elif current_drawdown > self.max_drawdown * 0.4:  # 40%で小幅削減
            return 0.85  # 15%削減
        else:
            return 1.0
    
    def calculate_portfolio_var(self, current_positions: Dict, confidence: float = 0.95) -> float:
        """ポートフォリオVaR計算"""
        if not current_positions or not self.position_history:
            return 0.0
        
        # 履歴からリターン分布を計算
        recent_returns = []
        for pos_data in list(self.position_history)[-self.var_lookback:]:
            if 'portfolio_return' in pos_data:
                recent_returns.append(pos_data['portfolio_return'])
        
        if len(recent_returns) < 10:
            return self.current_balance * 0.02  # 2%デフォルト
        
        # VaR計算（ヒストリカル法）
        returns = sorted(recent_returns)
        var_index = int((1 - confidence) * len(returns))
        var_return = returns[var_index] if var_index < len(returns) else returns[0]
        
        portfolio_var = abs(var_return * self.current_balance)
        
        return portfolio_var
    
    def check_risk_limits(self, 
                         current_positions: Dict,
                         new_position: Optional[Dict] = None) -> Tuple[bool, List[str]]:
        """リスク制限チェック"""
        warnings = []
        
        # 1. ポートフォリオVaRチェック
        portfolio_var = self.calculate_portfolio_var(current_positions)
        max_var = self.current_balance * self.max_portfolio_risk
        
        if portfolio_var > max_var:
            warnings.append(f"Portfolio VaR exceeded: {portfolio_var:.0f} > {max_var:.0f}")
        
        # 2. 日次損失チェック
        daily_loss = self._calculate_daily_loss()
        max_daily = self.current_balance * self.max_daily_loss
        
        if daily_loss > max_daily:
            warnings.append(f"Daily loss limit exceeded: {daily_loss:.0f} > {max_daily:.0f}")
        
        # 3. 最大ドローダウンチェック
        current_drawdown = self._calculate_current_drawdown()
        
        if current_drawdown > self.max_drawdown:
            warnings.append(f"Max drawdown exceeded: {current_drawdown:.1%} > {self.max_drawdown:.1%}")
        
        # 4. ポジション集中度チェック
        if new_position:
            concentration = self._check_concentration(current_positions, new_position)
            if concentration > 0.4:  # 40%以上の集中
                warnings.append(f"Position concentration too high: {concentration:.1%}")
        
        # 5. 相関チェック
        if new_position and current_positions:
            max_correlation = self._check_portfolio_correlation(current_positions, new_position)
            if max_correlation > self.max_correlation:
                warnings.append(f"Portfolio correlation too high: {max_correlation:.1%}")
        
        # アラート追加
        for warning in warnings:
            self._add_alert('risk_limit', warning)
        
        # 制限内かどうか
        within_limits = len(warnings) == 0
        
        return within_limits, warnings
    
    def _calculate_daily_loss(self) -> float:
        """日次損失計算"""
        if not self.daily_pnl:
            return 0.0
        
        today = datetime.now().date()
        daily_loss = 0.0
        
        for pnl_data in self.daily_pnl:
            if pnl_data['timestamp'].date() == today and pnl_data['pnl'] < 0:
                daily_loss += abs(pnl_data['pnl'])
        
        return daily_loss
    
    def _calculate_current_drawdown(self) -> float:
        """現在のドローダウン計算"""
        if not self.daily_pnl:
            return 0.0
        
        peak_balance = self.initial_balance
        for pnl_data in self.daily_pnl:
            peak_balance = max(peak_balance, pnl_data['balance'])
        
        return (peak_balance - self.current_balance) / peak_balance
    
    def _check_concentration(self, current_positions: Dict, new_position: Dict) -> float:
        """ポジション集中度チェック"""
        total_exposure = sum(pos.get('position_size', 0) * pos.get('leverage', 1) 
                           for pos in current_positions.values())
        new_exposure = new_position.get('position_size', 0) * new_position.get('leverage', 1)
        
        total_with_new = total_exposure + new_exposure
        return total_with_new / self.current_balance if self.current_balance > 0 else 0
    
    def _check_portfolio_correlation(self, current_positions: Dict, new_position: Dict) -> float:
        """ポートフォリオ相関チェック"""
        new_symbol = new_position.get('symbol')
        if not new_symbol:
            return 0.0
        
        max_correlation = 0.0
        for pos in current_positions.values():
            existing_symbol = pos.get('symbol')
            if existing_symbol:
                # 簡易相関計算（実際の実装ではより精密が必要）
                correlation = self._get_symbol_correlation(new_symbol, existing_symbol)
                max_correlation = max(max_correlation, correlation)
        
        return max_correlation
    
    def _get_symbol_correlation(self, symbol1: str, symbol2: str) -> float:
        """シンボル間相関取得"""
        correlation_map = {
            ('BTCUSDT', 'ETHUSDT'): 0.8,
            ('BTCUSDT', 'SOLUSDT'): 0.7,
            ('ETHUSDT', 'SOLUSDT'): 0.75
        }
        
        pair = tuple(sorted([symbol1, symbol2]))
        return correlation_map.get(pair, 0.3)  # デフォルト30%
    
    def detect_market_regime(self, market_data: Dict, price_history: Dict) -> str:
        """市場レジーム検出"""
        if not market_data or not price_history:
            return 'normal'
        
        volatilities = []
        trends = []
        
        for symbol, data in market_data.items():
            if symbol in price_history and len(price_history[symbol]) >= 20:
                prices = [p['price'] for p in list(price_history[symbol])]
                
                # ボラティリティ計算
                returns = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
                if len(returns) > 1:
                    volatility = statistics.stdev(returns)
                    volatilities.append(volatility)
                
                # トレンド強度計算
                if len(prices) >= 10:
                    trend = (prices[-1] - prices[-10]) / prices[-10]
                    trends.append(abs(trend))
        
        if not volatilities:
            return 'normal'
        
        avg_volatility = statistics.mean(volatilities)
        avg_trend = statistics.mean(trends) if trends else 0
        
        # レジーム判定
        if avg_volatility > 0.008:  # 高ボラティリティ
            regime = 'volatile'
        elif avg_trend > 0.05:  # 強いトレンド
            regime = 'trending'
        elif avg_volatility > 0.015:  # 危機レベル
            regime = 'crisis'
        else:
            regime = 'normal'
        
        # レジーム変更検出
        if regime != self.current_regime:
            logger.info(f"Market regime changed: {self.current_regime} -> {regime}")
            self._add_alert('regime_change', f'Market regime: {regime}')
            self.current_regime = regime
        
        self.regime_history.append({
            'timestamp': datetime.now(),
            'regime': regime,
            'volatility': avg_volatility,
            'trend': avg_trend
        })
        
        return regime
    
    def _add_alert(self, alert_type: str, message: str):
        """アラート追加"""
        alert = {
            'timestamp': datetime.now(),
            'type': alert_type,
            'message': message
        }
        
        self.alerts.append(alert)
        logger.warning(f"RISK ALERT [{alert_type}]: {message}")
    
    def get_risk_metrics(self) -> Dict[str, Any]:
        """リスク指標取得"""
        current_drawdown = self._calculate_current_drawdown()
        daily_loss = self._calculate_daily_loss()
        
        return {
            'current_balance': self.current_balance,
            'total_return': (self.current_balance - self.initial_balance) / self.initial_balance,
            'current_drawdown': current_drawdown,
            'daily_loss': daily_loss,
            'max_daily_loss_limit': self.current_balance * self.max_daily_loss,
            'max_drawdown_limit': self.max_drawdown,
            'market_regime': self.current_regime,
            'portfolio_var_95': self.calculate_portfolio_var({}, 0.95),
            'risk_alerts': len([a for a in self.alerts if a['timestamp'].date() == datetime.now().date()]),
            'kelly_fraction': self.kelly_fraction
        }
    
    def generate_risk_report(self) -> str:
        """リスクレポート生成"""
        metrics = self.get_risk_metrics()
        
        report = f"""
📊 ADVANCED RISK MANAGEMENT REPORT
{'='*50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

💰 PORTFOLIO OVERVIEW:
  Balance: ${metrics['current_balance']:,.2f}
  Total Return: {metrics['total_return']:+.2%}
  Current Drawdown: {metrics['current_drawdown']:.2%}
  
⚠️  RISK METRICS:
  Daily Loss: ${metrics['daily_loss']:,.2f} / ${metrics['max_daily_loss_limit']:,.2f}
  Drawdown: {metrics['current_drawdown']:.1%} / {metrics['max_drawdown_limit']:.1%}
  Portfolio VaR(95%): ${metrics['portfolio_var_95']:,.2f}
  
🌍 MARKET ENVIRONMENT:
  Current Regime: {metrics['market_regime'].upper()}
  Kelly Fraction: {metrics['kelly_fraction']:.1%}
  
🚨 ALERTS:
  Today's Risk Alerts: {metrics['risk_alerts']}
"""
        
        if self.alerts:
            report += "\n  Recent Alerts:\n"
            recent_alerts = [a for a in list(self.alerts)[-5:]]
            for alert in recent_alerts:
                report += f"    {alert['timestamp'].strftime('%H:%M')} [{alert['type']}]: {alert['message']}\n"
        
        report += f"\n{'='*50}"
        
        return report


# テスト関数
def test_risk_manager():
    """リスク管理システムテスト"""
    print("🧪 Testing Advanced Risk Manager...")
    
    rm = AdvancedRiskManager(100000)
    
    # テストデータ
    test_positions = {
        'pos1': {'symbol': 'BTCUSDT', 'position_size': 30000, 'leverage': 10},
        'pos2': {'symbol': 'ETHUSDT', 'position_size': 20000, 'leverage': 5}
    }
    
    # ポジションサイズ計算テスト
    size, reasoning = rm.calculate_position_size(
        'SOLUSDT', 0.8, 0.01, 0.002, 0.003, test_positions
    )
    
    print(f"✅ Position size calculated: ${size:.0f}")
    print(f"   Reasoning: {reasoning}")
    
    # リスク制限チェック
    within_limits, warnings = rm.check_risk_limits(test_positions)
    print(f"✅ Risk limits check: {'PASS' if within_limits else 'FAIL'}")
    
    if warnings:
        for warning in warnings:
            print(f"   ⚠️  {warning}")
    
    # レポート生成
    report = rm.generate_risk_report()
    print("✅ Risk report generated:")
    print(report)


if __name__ == "__main__":
    test_risk_manager()