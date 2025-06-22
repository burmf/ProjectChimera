"""
PDCA分析エンジン
取引ロジックの継続的改善のための自動分析システム
"""

import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import structlog

from .performance_logger import (
    PerformanceLogger, StrategyPerformanceMetrics, TradeExecution, 
    PDCALogEntry, PerformancePhase, TradeResult
)

logger = structlog.get_logger(__name__)


@dataclass
class OptimizationRecommendation:
    """最適化推奨事項"""
    strategy_name: str
    parameter_name: str
    current_value: Any
    recommended_value: Any
    expected_improvement: float
    confidence: float
    reasoning: str
    priority: str  # high, medium, low


@dataclass
class MarketRegimeAnalysis:
    """マーケットレジーム分析"""
    regime_type: str  # trending, ranging, volatile, calm
    start_time: datetime
    end_time: datetime
    performance_metrics: Dict[str, float]
    strategy_effectiveness: Dict[str, float]
    recommendations: List[str]


class PDCAAnalyzer:
    """
    PDCA分析エンジン
    
    機能:
    1. 自動パフォーマンス分析
    2. マーケットレジーム適応性評価
    3. パラメータ最適化提案
    4. リスク調整後リターン分析
    5. 実行効率分析
    """
    
    def __init__(self, performance_logger: PerformanceLogger):
        self.performance_logger = performance_logger
        self.analysis_history: List[Dict[str, Any]] = []
        self.optimization_queue: List[OptimizationRecommendation] = []
        
        # 分析設定
        self.min_trades_for_analysis = 20
        self.confidence_threshold = 0.7
        self.improvement_threshold = 0.05  # 5%以上の改善
        
        logger.info("PDCA Analyzer initialized")
    
    async def run_comprehensive_analysis(self, strategy_name: str) -> Dict[str, Any]:
        """包括的分析実行"""
        logger.info("Starting comprehensive analysis", strategy=strategy_name)
        
        # 基本統計分析
        basic_analysis = await self._analyze_basic_performance(strategy_name)
        
        # 時系列分析
        temporal_analysis = await self._analyze_temporal_patterns(strategy_name)
        
        # マーケットレジーム分析
        regime_analysis = await self._analyze_market_regimes(strategy_name)
        
        # パラメータ最適化分析
        optimization_analysis = await self._analyze_parameter_optimization(strategy_name)
        
        # リスク分析
        risk_analysis = await self._analyze_risk_metrics(strategy_name)
        
        # 実行効率分析
        execution_analysis = await self._analyze_execution_efficiency(strategy_name)
        
        # 総合スコア計算
        overall_score = await self._calculate_overall_score(strategy_name, {
            "basic": basic_analysis,
            "temporal": temporal_analysis,
            "regime": regime_analysis,
            "optimization": optimization_analysis,
            "risk": risk_analysis,
            "execution": execution_analysis
        })
        
        comprehensive_report = {
            "strategy_name": strategy_name,
            "analysis_timestamp": datetime.now().isoformat(),
            "overall_score": overall_score,
            "analyses": {
                "basic_performance": basic_analysis,
                "temporal_patterns": temporal_analysis,
                "market_regimes": regime_analysis,
                "parameter_optimization": optimization_analysis,
                "risk_metrics": risk_analysis,
                "execution_efficiency": execution_analysis
            },
            "recommendations": await self._generate_comprehensive_recommendations(strategy_name),
            "next_steps": await self._suggest_next_steps(strategy_name)
        }
        
        # 分析履歴に保存
        self.analysis_history.append(comprehensive_report)
        
        logger.info("Comprehensive analysis completed", 
                   strategy=strategy_name,
                   overall_score=overall_score)
        
        return comprehensive_report
    
    async def _analyze_basic_performance(self, strategy_name: str) -> Dict[str, Any]:
        """基本パフォーマンス分析"""
        trades = [t for t in self.performance_logger.trade_history if t.strategy_name == strategy_name]
        
        if len(trades) < self.min_trades_for_analysis:
            return {"error": "Insufficient trade data", "trade_count": len(trades)}
        
        # 基本統計
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.result == TradeResult.WIN])
        win_rate = winning_trades / total_trades
        
        # PnL分析
        pnls = [t.pnl for t in trades if t.pnl is not None]
        total_pnl = sum(pnls)
        avg_pnl = total_pnl / len(pnls) if pnls else 0
        
        # リターン分析
        returns = [t.pnl_percentage for t in trades if t.pnl_percentage is not None]
        avg_return = sum(returns) / len(returns) if returns else 0
        return_std = np.std(returns) if len(returns) > 1 else 0
        
        # Sharpe ratio
        sharpe_ratio = avg_return / return_std if return_std > 0 else 0
        
        # 最大ドローダウン計算
        cumulative_pnl = []
        running_total = 0
        for pnl in pnls:
            running_total += pnl
            cumulative_pnl.append(running_total)
        
        max_drawdown = 0
        peak = 0
        for value in cumulative_pnl:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
        
        # プロフィットファクター
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "average_pnl": avg_pnl,
            "average_return": avg_return,
            "return_volatility": return_std,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "profit_factor": profit_factor,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "confidence_level": min(total_trades / 100, 1.0),  # 100取引で最大信頼度
            "quality_score": self._calculate_analysis_quality(trades)
        }
    
    async def _analyze_temporal_patterns(self, strategy_name: str) -> Dict[str, Any]:
        """時系列パターン分析"""
        trades = [t for t in self.performance_logger.trade_history if t.strategy_name == strategy_name]
        
        if not trades:
            return {}
        
        # 時間帯別パフォーマンス
        hourly_performance = {}
        daily_performance = {}
        
        for trade in trades:
            hour = trade.timestamp.hour
            day = trade.timestamp.strftime('%Y-%m-%d')
            
            if hour not in hourly_performance:
                hourly_performance[hour] = []
            if day not in daily_performance:
                daily_performance[day] = []
            
            if trade.pnl is not None:
                hourly_performance[hour].append(trade.pnl)
                daily_performance[day].append(trade.pnl)
        
        # 時間帯統計
        hourly_stats = {}
        for hour, pnls in hourly_performance.items():
            if len(pnls) >= 3:  # 最低3取引
                hourly_stats[hour] = {
                    "trade_count": len(pnls),
                    "avg_pnl": sum(pnls) / len(pnls),
                    "total_pnl": sum(pnls),
                    "win_rate": len([p for p in pnls if p > 0]) / len(pnls)
                }
        
        # 最適時間帯特定
        best_hours = sorted(
            [(hour, stats["avg_pnl"]) for hour, stats in hourly_stats.items()],
            key=lambda x: x[1], reverse=True
        )[:3]
        
        # 曜日別パフォーマンス
        weekday_performance = {}
        for trade in trades:
            weekday = trade.timestamp.weekday()  # 0=月曜日
            if weekday not in weekday_performance:
                weekday_performance[weekday] = []
            if trade.pnl is not None:
                weekday_performance[weekday].append(trade.pnl)
        
        weekday_stats = {}
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for weekday, pnls in weekday_performance.items():
            if len(pnls) >= 2:
                weekday_stats[weekday_names[weekday]] = {
                    "trade_count": len(pnls),
                    "avg_pnl": sum(pnls) / len(pnls),
                    "win_rate": len([p for p in pnls if p > 0]) / len(pnls)
                }
        
        # トレンド分析（最近30日 vs 全期間）
        recent_cutoff = datetime.now() - timedelta(days=30)
        recent_trades = [t for t in trades if t.timestamp >= recent_cutoff]
        
        trend_analysis = {}
        if len(recent_trades) >= 10:
            recent_pnls = [t.pnl for t in recent_trades if t.pnl is not None]
            all_pnls = [t.pnl for t in trades if t.pnl is not None]
            
            recent_avg = sum(recent_pnls) / len(recent_pnls) if recent_pnls else 0
            all_avg = sum(all_pnls) / len(all_pnls) if all_pnls else 0
            
            trend_analysis = {
                "recent_30d_avg_pnl": recent_avg,
                "overall_avg_pnl": all_avg,
                "performance_trend": "improving" if recent_avg > all_avg * 1.1 else "declining" if recent_avg < all_avg * 0.9 else "stable",
                "recent_trade_count": len(recent_trades),
                "trend_strength": abs(recent_avg - all_avg) / abs(all_avg) if all_avg != 0 else 0
            }
        
        return {
            "hourly_stats": hourly_stats,
            "best_trading_hours": best_hours,
            "weekday_stats": weekday_stats,
            "trend_analysis": trend_analysis,
            "seasonal_patterns": await self._detect_seasonal_patterns(trades),
            "performance_consistency": self._calculate_consistency_score(daily_performance)
        }
    
    async def _analyze_market_regimes(self, strategy_name: str) -> Dict[str, Any]:
        """マーケットレジーム分析"""
        trades = [t for t in self.performance_logger.trade_history if t.strategy_name == strategy_name]
        
        if not trades:
            return {}
        
        # ボラティリティベースのレジーム分類
        volatility_regimes = {
            "low_vol": [],    # < 0.01
            "medium_vol": [], # 0.01 - 0.03
            "high_vol": []    # > 0.03
        }
        
        for trade in trades:
            if trade.market_volatility is not None:
                vol = trade.market_volatility
                if vol < 0.01:
                    volatility_regimes["low_vol"].append(trade)
                elif vol < 0.03:
                    volatility_regimes["medium_vol"].append(trade)
                else:
                    volatility_regimes["high_vol"].append(trade)
        
        # レジーム別パフォーマンス
        regime_performance = {}
        for regime, regime_trades in volatility_regimes.items():
            if len(regime_trades) >= 5:
                pnls = [t.pnl for t in regime_trades if t.pnl is not None]
                if pnls:
                    regime_performance[regime] = {
                        "trade_count": len(regime_trades),
                        "avg_pnl": sum(pnls) / len(pnls),
                        "win_rate": len([p for p in pnls if p > 0]) / len(pnls),
                        "total_pnl": sum(pnls),
                        "volatility_range": f"{min(t.market_volatility for t in regime_trades):.3f}-{max(t.market_volatility for t in regime_trades):.3f}"
                    }
        
        # 最適レジーム特定
        best_regime = None
        if regime_performance:
            best_regime = max(regime_performance.items(), key=lambda x: x[1]["avg_pnl"])
        
        # スプレッド環境分析
        spread_analysis = {}
        if any(t.spread_bps for t in trades):
            spread_trades = [t for t in trades if t.spread_bps is not None]
            spreads = [t.spread_bps for t in spread_trades]
            median_spread = np.median(spreads)
            
            tight_spread_trades = [t for t in spread_trades if t.spread_bps <= median_spread]
            wide_spread_trades = [t for t in spread_trades if t.spread_bps > median_spread]
            
            for label, regime_trades in [("tight_spread", tight_spread_trades), ("wide_spread", wide_spread_trades)]:
                if regime_trades:
                    pnls = [t.pnl for t in regime_trades if t.pnl is not None]
                    if pnls:
                        spread_analysis[label] = {
                            "trade_count": len(regime_trades),
                            "avg_pnl": sum(pnls) / len(pnls),
                            "win_rate": len([p for p in pnls if p > 0]) / len(pnls)
                        }
        
        return {
            "volatility_regimes": regime_performance,
            "best_regime": best_regime[0] if best_regime else None,
            "best_regime_performance": best_regime[1] if best_regime else None,
            "spread_analysis": spread_analysis,
            "regime_adaptation_score": self._calculate_regime_adaptation_score(regime_performance),
            "recommendations": self._generate_regime_recommendations(regime_performance, spread_analysis)
        }
    
    async def _analyze_parameter_optimization(self, strategy_name: str) -> Dict[str, Any]:
        """パラメータ最適化分析"""
        trades = [t for t in self.performance_logger.trade_history if t.strategy_name == strategy_name]
        
        if not trades:
            return {}
        
        # 信頼度別パフォーマンス分析
        confidence_analysis = {}
        confidence_buckets = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
        
        for min_conf, max_conf in confidence_buckets:
            bucket_trades = [t for t in trades if min_conf <= t.signal_confidence < max_conf]
            if len(bucket_trades) >= 3:
                pnls = [t.pnl for t in bucket_trades if t.pnl is not None]
                if pnls:
                    confidence_analysis[f"{min_conf}-{max_conf}"] = {
                        "trade_count": len(bucket_trades),
                        "avg_pnl": sum(pnls) / len(pnls),
                        "win_rate": len([p for p in pnls if p > 0]) / len(pnls),
                        "avg_confidence": sum(t.signal_confidence for t in bucket_trades) / len(bucket_trades)
                    }
        
        # 最適信頼度閾値
        optimal_confidence_threshold = None
        if confidence_analysis:
            positive_buckets = {k: v for k, v in confidence_analysis.items() if v["avg_pnl"] > 0}
            if positive_buckets:
                best_bucket = max(positive_buckets.items(), key=lambda x: x[1]["avg_pnl"])
                optimal_confidence_threshold = best_bucket[1]["avg_confidence"]
        
        # 保有時間最適化
        hold_duration_analysis = {}
        if any(t.hold_duration_seconds for t in trades):
            duration_trades = [t for t in trades if t.hold_duration_seconds is not None]
            durations = [t.hold_duration_seconds for t in duration_trades]
            
            # 四分位数で分析
            q25, q50, q75 = np.percentile(durations, [25, 50, 75])
            duration_buckets = [
                ("short", lambda d: d <= q25),
                ("medium", lambda d: q25 < d <= q75),
                ("long", lambda d: d > q75)
            ]
            
            for label, condition in duration_buckets:
                bucket_trades = [t for t in duration_trades if condition(t.hold_duration_seconds)]
                if bucket_trades:
                    pnls = [t.pnl for t in bucket_trades if t.pnl is not None]
                    if pnls:
                        hold_duration_analysis[label] = {
                            "trade_count": len(bucket_trades),
                            "avg_pnl": sum(pnls) / len(pnls),
                            "win_rate": len([p for p in pnls if p > 0]) / len(pnls),
                            "avg_duration_minutes": sum(t.hold_duration_seconds for t in bucket_trades) / len(bucket_trades) / 60
                        }
        
        # パラメータ最適化推奨
        optimization_recommendations = []
        
        if optimal_confidence_threshold:
            current_threshold = 0.5  # デフォルト値と仮定
            if abs(optimal_confidence_threshold - current_threshold) > 0.1:
                optimization_recommendations.append(OptimizationRecommendation(
                    strategy_name=strategy_name,
                    parameter_name="confidence_threshold",
                    current_value=current_threshold,
                    recommended_value=round(optimal_confidence_threshold, 2),
                    expected_improvement=0.15,  # 推定改善率
                    confidence=0.8,
                    reasoning=f"分析により信頼度{optimal_confidence_threshold:.2f}でのパフォーマンスが最適",
                    priority="high"
                ))
        
        return {
            "confidence_analysis": confidence_analysis,
            "optimal_confidence_threshold": optimal_confidence_threshold,
            "hold_duration_analysis": hold_duration_analysis,
            "optimization_recommendations": [asdict(rec) for rec in optimization_recommendations],
            "parameter_sensitivity": await self._analyze_parameter_sensitivity(trades)
        }
    
    async def _analyze_risk_metrics(self, strategy_name: str) -> Dict[str, Any]:
        """リスク指標分析"""
        trades = [t for t in self.performance_logger.trade_history if t.strategy_name == strategy_name]
        
        if not trades:
            return {}
        
        pnls = [t.pnl for t in trades if t.pnl is not None]
        if not pnls:
            return {}
        
        # VaR計算（複数信頼区間）
        var_calculations = {}
        for confidence in [0.95, 0.99]:
            if len(pnls) >= 20:
                var_index = int(len(pnls) * (1 - confidence))
                sorted_pnls = sorted(pnls)
                var_calculations[f"VaR_{int(confidence*100)}"] = sorted_pnls[var_index]
        
        # 条件付きVaR (CVaR/Expected Shortfall)
        cvar_calculations = {}
        for confidence in [0.95, 0.99]:
            if len(pnls) >= 20:
                var_index = int(len(pnls) * (1 - confidence))
                sorted_pnls = sorted(pnls)
                tail_losses = sorted_pnls[:var_index]
                if tail_losses:
                    cvar_calculations[f"CVaR_{int(confidence*100)}"] = sum(tail_losses) / len(tail_losses)
        
        # 最大連続損失期間
        consecutive_losses = 0
        max_consecutive_losses = 0
        consecutive_loss_amount = 0
        max_consecutive_loss_amount = 0
        
        for trade in trades:
            if trade.result == TradeResult.LOSS and trade.pnl is not None:
                consecutive_losses += 1
                consecutive_loss_amount += abs(trade.pnl)
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                max_consecutive_loss_amount = max(max_consecutive_loss_amount, consecutive_loss_amount)
            else:
                consecutive_losses = 0
                consecutive_loss_amount = 0
        
        # リスク調整後リターン指標
        returns = [t.pnl_percentage for t in trades if t.pnl_percentage is not None]
        risk_adjusted_metrics = {}
        
        if len(returns) > 1:
            avg_return = sum(returns) / len(returns)
            return_std = np.std(returns)
            
            # Sharpe ratio
            risk_adjusted_metrics["sharpe_ratio"] = avg_return / return_std if return_std > 0 else 0
            
            # Sortino ratio (下方偏差のみ)
            downside_returns = [r for r in returns if r < 0]
            if downside_returns:
                downside_std = np.std(downside_returns)
                risk_adjusted_metrics["sortino_ratio"] = avg_return / downside_std if downside_std > 0 else 0
            
            # Calmar ratio (年率リターン / 最大ドローダウン)
            cumulative_returns = np.cumsum(returns)
            max_dd = 0
            peak = 0
            for ret in cumulative_returns:
                if ret > peak:
                    peak = ret
                dd = (peak - ret) / peak if peak > 0 else 0
                max_dd = max(max_dd, dd)
            
            if max_dd > 0:
                annualized_return = avg_return * 252  # 日次リターンを年率換算
                risk_adjusted_metrics["calmar_ratio"] = annualized_return / max_dd
        
        # リスクアラート
        risk_alerts = []
        if var_calculations.get("VaR_95", 0) < -0.05:  # 5%以上の損失リスク
            risk_alerts.append("高リスク: 95%VaRが5%を超過")
        
        if max_consecutive_losses > 5:
            risk_alerts.append(f"連続損失リスク: {max_consecutive_losses}回連続")
        
        if risk_adjusted_metrics.get("sharpe_ratio", 0) < 0.5:
            risk_alerts.append("低効率: Sharpe比率が0.5未満")
        
        return {
            "var_calculations": var_calculations,
            "cvar_calculations": cvar_calculations,
            "max_consecutive_losses": max_consecutive_losses,
            "max_consecutive_loss_amount": max_consecutive_loss_amount,
            "risk_adjusted_metrics": risk_adjusted_metrics,
            "risk_alerts": risk_alerts,
            "risk_score": self._calculate_risk_score(var_calculations, max_consecutive_losses, risk_adjusted_metrics),
            "risk_recommendations": self._generate_risk_recommendations(var_calculations, risk_adjusted_metrics)
        }
    
    async def _analyze_execution_efficiency(self, strategy_name: str) -> Dict[str, Any]:
        """実行効率分析"""
        trades = [t for t in self.performance_logger.trade_history if t.strategy_name == strategy_name]
        
        if not trades:
            return {}
        
        # 実行時間分析
        signal_times = [t.signal_generation_time_ms for t in trades if t.signal_generation_time_ms > 0]
        execution_times = [t.total_execution_time_ms for t in trades if t.total_execution_time_ms > 0]
        
        timing_analysis = {}
        if signal_times:
            timing_analysis["signal_generation"] = {
                "avg_ms": sum(signal_times) / len(signal_times),
                "p95_ms": np.percentile(signal_times, 95),
                "p99_ms": np.percentile(signal_times, 99),
                "max_ms": max(signal_times)
            }
        
        if execution_times:
            timing_analysis["order_execution"] = {
                "avg_ms": sum(execution_times) / len(execution_times),
                "p95_ms": np.percentile(execution_times, 95),
                "p99_ms": np.percentile(execution_times, 99),
                "max_ms": max(execution_times)
            }
        
        # スリッページ分析
        slippage_analysis = {}
        fill_times = [t.fill_time_ms for t in trades if t.fill_time_ms > 0]
        if fill_times:
            slippage_analysis = {
                "avg_fill_time_ms": sum(fill_times) / len(fill_times),
                "fast_fills": len([t for t in fill_times if t < 100]),  # 100ms未満
                "slow_fills": len([t for t in fill_times if t > 1000])   # 1秒超過
            }
        
        # 手数料効率
        commission_analysis = {}
        commissions = [t.commission for t in trades if t.commission > 0]
        pnls = [t.pnl for t in trades if t.pnl is not None]
        
        if commissions and pnls:
            total_commission = sum(commissions)
            total_gross_pnl = sum(pnls)
            commission_analysis = {
                "total_commission": total_commission,
                "avg_commission_per_trade": total_commission / len(commissions),
                "commission_ratio": total_commission / abs(total_gross_pnl) if total_gross_pnl != 0 else 0,
                "commission_efficiency": "good" if total_commission / abs(total_gross_pnl) < 0.1 else "poor"
            }
        
        # 効率スコア計算
        efficiency_score = 100  # 満点から減点方式
        
        if timing_analysis.get("signal_generation", {}).get("avg_ms", 0) > 500:
            efficiency_score -= 20
        if timing_analysis.get("order_execution", {}).get("avg_ms", 0) > 1000:
            efficiency_score -= 25
        if slippage_analysis.get("slow_fills", 0) > len(trades) * 0.1:  # 10%以上が遅い
            efficiency_score -= 15
        if commission_analysis.get("commission_ratio", 0) > 0.1:  # 手数料比率10%以上
            efficiency_score -= 20
        
        efficiency_recommendations = []
        if timing_analysis.get("signal_generation", {}).get("avg_ms", 0) > 500:
            efficiency_recommendations.append("シグナル生成ロジックの最適化が必要")
        if timing_analysis.get("order_execution", {}).get("avg_ms", 0) > 1000:
            efficiency_recommendations.append("注文実行の高速化が必要")
        
        return {
            "timing_analysis": timing_analysis,
            "slippage_analysis": slippage_analysis,
            "commission_analysis": commission_analysis,
            "efficiency_score": max(0, efficiency_score),
            "efficiency_recommendations": efficiency_recommendations,
            "bottlenecks": self._identify_performance_bottlenecks(timing_analysis, slippage_analysis)
        }
    
    async def _calculate_overall_score(self, strategy_name: str, analyses: Dict[str, Any]) -> float:
        """総合スコア計算"""
        scores = {}
        weights = {
            "basic": 0.3,
            "risk": 0.25,
            "execution": 0.2,
            "temporal": 0.15,
            "regime": 0.1
        }
        
        # 基本パフォーマンススコア
        basic = analyses.get("basic", {})
        if basic:
            profit_factor = basic.get("profit_factor", 0)
            win_rate = basic.get("win_rate", 0)
            sharpe = basic.get("sharpe_ratio", 0)
            
            basic_score = (
                min(profit_factor / 2, 50) +  # プロフィットファクター2.0で50点
                win_rate * 30 +                # 勝率30点満点
                min(abs(sharpe) * 10, 20)      # Sharpe比率20点満点
            )
            scores["basic"] = min(basic_score, 100)
        
        # リスクスコア
        risk = analyses.get("risk", {})
        if risk:
            risk_score = risk.get("risk_score", 50)
            scores["risk"] = risk_score
        
        # 実行効率スコア
        execution = analyses.get("execution", {})
        if execution:
            scores["execution"] = execution.get("efficiency_score", 50)
        
        # 時系列スコア（一貫性）
        temporal = analyses.get("temporal", {})
        if temporal:
            consistency = temporal.get("performance_consistency", 0.5)
            scores["temporal"] = consistency * 100
        
        # レジーム適応スコア
        regime = analyses.get("regime", {})
        if regime:
            adaptation_score = regime.get("regime_adaptation_score", 0.5)
            scores["regime"] = adaptation_score * 100
        
        # 重み付き平均
        weighted_score = sum(scores.get(key, 50) * weight for key, weight in weights.items())
        
        return round(weighted_score, 2)
    
    async def _generate_comprehensive_recommendations(self, strategy_name: str) -> List[Dict[str, Any]]:
        """包括的推奨事項生成"""
        recommendations = []
        
        # 最適化キューから推奨事項を取得
        strategy_optimizations = [opt for opt in self.optimization_queue if opt.strategy_name == strategy_name]
        
        for opt in strategy_optimizations:
            recommendations.append({
                "type": "parameter_optimization",
                "priority": opt.priority,
                "title": f"{opt.parameter_name}の最適化",
                "description": opt.reasoning,
                "action": f"{opt.current_value} → {opt.recommended_value}",
                "expected_improvement": f"{opt.expected_improvement:.1%}",
                "confidence": f"{opt.confidence:.1%}"
            })
        
        # 追加の戦略的推奨事項
        metrics = self.performance_logger.strategy_metrics.get(strategy_name)
        if metrics:
            if metrics.win_rate < 0.4:
                recommendations.append({
                    "type": "strategy_adjustment",
                    "priority": "high",
                    "title": "エントリー基準の厳格化",
                    "description": f"勝率{metrics.win_rate:.1%}が低すぎます。シグナル精度の向上が必要",
                    "action": "信頼度閾値の引き上げまたはフィルター条件の追加",
                    "expected_improvement": "15-25%",
                    "confidence": "高"
                })
            
            if metrics.profit_factor < 1.5:
                recommendations.append({
                    "type": "risk_management",
                    "priority": "medium",
                    "title": "リスク・リワード比の改善",
                    "description": f"プロフィットファクター{metrics.profit_factor:.2f}が低い",
                    "action": "利確・損切りレベルの調整",
                    "expected_improvement": "10-20%",
                    "confidence": "中"
                })
        
        return recommendations
    
    async def _suggest_next_steps(self, strategy_name: str) -> List[str]:
        """次のステップ提案"""
        steps = []
        
        # 取引数チェック
        trade_count = len([t for t in self.performance_logger.trade_history if t.strategy_name == strategy_name])
        
        if trade_count < 50:
            steps.append(f"データ収集: あと{50-trade_count}取引のデータ蓄積が推奨")
        
        if trade_count >= 100:
            steps.append("A/Bテスト実行: パラメータ最適化案のテスト")
            steps.append("フォワードテスト: 最適化結果の検証")
        
        # パフォーマンス状況に応じたステップ
        metrics = self.performance_logger.strategy_metrics.get(strategy_name)
        if metrics:
            if metrics.total_pnl < 0:
                steps.append("緊急対応: 戦略一時停止・原因分析")
            elif metrics.win_rate > 0.6 and metrics.profit_factor > 2.0:
                steps.append("スケールアップ: ポジションサイズ増加検討")
            else:
                steps.append("継続改善: PDCAサイクル実行")
        
        steps.append("定期レビュー: 週次パフォーマンス評価")
        
        return steps
    
    def _calculate_analysis_quality(self, trades: List[TradeExecution]) -> float:
        """分析品質スコア計算"""
        if not trades:
            return 0.0
        
        quality_factors = []
        
        # データ完全性
        complete_data_count = sum(1 for t in trades if all([
            t.pnl is not None,
            t.hold_duration_seconds is not None,
            t.total_execution_time_ms > 0
        ]))
        quality_factors.append(complete_data_count / len(trades))
        
        # サンプルサイズ
        sample_quality = min(len(trades) / 100, 1.0)  # 100取引で満点
        quality_factors.append(sample_quality)
        
        # 時間分散
        time_span = (max(t.timestamp for t in trades) - min(t.timestamp for t in trades)).days
        time_quality = min(time_span / 30, 1.0)  # 30日間で満点
        quality_factors.append(time_quality)
        
        return sum(quality_factors) / len(quality_factors)
    
    async def _detect_seasonal_patterns(self, trades: List[TradeExecution]) -> Dict[str, Any]:
        """季節性パターン検出"""
        # 月別パフォーマンス
        monthly_performance = {}
        for trade in trades:
            month = trade.timestamp.month
            if month not in monthly_performance:
                monthly_performance[month] = []
            if trade.pnl is not None:
                monthly_performance[month].append(trade.pnl)
        
        monthly_stats = {}
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for month, pnls in monthly_performance.items():
            if len(pnls) >= 3:
                monthly_stats[month_names[month-1]] = {
                    "avg_pnl": sum(pnls) / len(pnls),
                    "trade_count": len(pnls),
                    "win_rate": len([p for p in pnls if p > 0]) / len(pnls)
                }
        
        return {
            "monthly_stats": monthly_stats,
            "seasonal_trend": "データ不足" if len(monthly_stats) < 3 else "分析中"
        }
    
    def _calculate_consistency_score(self, daily_performance: Dict[str, List[float]]) -> float:
        """パフォーマンス一貫性スコア計算"""
        if len(daily_performance) < 5:
            return 0.5  # デフォルト値
        
        daily_pnls = [sum(pnls) for pnls in daily_performance.values()]
        
        if not daily_pnls:
            return 0.5
        
        # 標準偏差ベースの一貫性
        mean_pnl = sum(daily_pnls) / len(daily_pnls)
        if mean_pnl == 0:
            return 0.5
        
        coefficient_of_variation = np.std(daily_pnls) / abs(mean_pnl)
        consistency = max(0, 1 - coefficient_of_variation / 2)  # CV 2.0で0点
        
        return min(consistency, 1.0)
    
    def _calculate_regime_adaptation_score(self, regime_performance: Dict[str, Any]) -> float:
        """レジーム適応スコア計算"""
        if not regime_performance:
            return 0.5
        
        positive_regimes = len([r for r in regime_performance.values() if r["avg_pnl"] > 0])
        total_regimes = len(regime_performance)
        
        return positive_regimes / total_regimes if total_regimes > 0 else 0.5
    
    def _generate_regime_recommendations(self, regime_performance: Dict[str, Any], 
                                       spread_analysis: Dict[str, Any]) -> List[str]:
        """レジーム対応推奨事項生成"""
        recommendations = []
        
        if regime_performance:
            best_regime = max(regime_performance.items(), key=lambda x: x[1]["avg_pnl"])
            worst_regime = min(regime_performance.items(), key=lambda x: x[1]["avg_pnl"])
            
            if best_regime[1]["avg_pnl"] > 0 and worst_regime[1]["avg_pnl"] < 0:
                recommendations.append(f"{best_regime[0]}環境での取引に特化することを検討")
                recommendations.append(f"{worst_regime[0]}環境での取引を制限することを検討")
        
        if spread_analysis:
            tight_perf = spread_analysis.get("tight_spread", {}).get("avg_pnl", 0)
            wide_perf = spread_analysis.get("wide_spread", {}).get("avg_pnl", 0)
            
            if tight_perf > wide_perf * 1.5:
                recommendations.append("スプレッドフィルターの導入でパフォーマンス向上可能")
        
        return recommendations
    
    async def _analyze_parameter_sensitivity(self, trades: List[TradeExecution]) -> Dict[str, Any]:
        """パラメータ感度分析"""
        # 簡易版: 信頼度とパフォーマンスの相関分析
        if len(trades) < 20:
            return {"error": "Insufficient data for sensitivity analysis"}
        
        confidences = [t.signal_confidence for t in trades]
        pnls = [t.pnl for t in trades if t.pnl is not None]
        
        if len(confidences) != len(pnls):
            return {"error": "Data alignment issue"}
        
        # 相関係数計算
        correlation = np.corrcoef(confidences, pnls)[0, 1] if len(confidences) > 1 else 0
        
        return {
            "confidence_pnl_correlation": correlation,
            "sensitivity_level": "high" if abs(correlation) > 0.3 else "medium" if abs(correlation) > 0.1 else "low",
            "optimization_potential": "high" if abs(correlation) > 0.2 else "medium"
        }
    
    def _calculate_risk_score(self, var_calculations: Dict[str, float], 
                            max_consecutive_losses: int,
                            risk_adjusted_metrics: Dict[str, float]) -> float:
        """リスクスコア計算（100点満点、高いほど安全）"""
        score = 100
        
        # VaRペナルティ
        var_95 = var_calculations.get("VaR_95", 0)
        if var_95 < -0.1:  # 10%以上の損失リスク
            score -= 40
        elif var_95 < -0.05:  # 5%以上の損失リスク
            score -= 20
        
        # 連続損失ペナルティ
        if max_consecutive_losses > 10:
            score -= 30
        elif max_consecutive_losses > 5:
            score -= 15
        
        # Sharpe比率ボーナス
        sharpe = risk_adjusted_metrics.get("sharpe_ratio", 0)
        if sharpe > 1.0:
            score += 10
        elif sharpe < 0:
            score -= 20
        
        return max(0, score)
    
    def _generate_risk_recommendations(self, var_calculations: Dict[str, float],
                                     risk_adjusted_metrics: Dict[str, float]) -> List[str]:
        """リスク管理推奨事項生成"""
        recommendations = []
        
        var_95 = var_calculations.get("VaR_95", 0)
        if var_95 < -0.05:
            recommendations.append("ポジションサイズの削減またはストップロス設定の強化が必要")
        
        sharpe = risk_adjusted_metrics.get("sharpe_ratio", 0)
        if sharpe < 0.5:
            recommendations.append("リスク調整後リターンの改善が必要（目標Sharpe比率 > 1.0）")
        
        sortino = risk_adjusted_metrics.get("sortino_ratio", 0)
        if sortino < sharpe:
            recommendations.append("下方リスクの管理強化が有効")
        
        return recommendations
    
    def _identify_performance_bottlenecks(self, timing_analysis: Dict[str, Any],
                                        slippage_analysis: Dict[str, Any]) -> List[str]:
        """パフォーマンスボトルネック特定"""
        bottlenecks = []
        
        signal_time = timing_analysis.get("signal_generation", {}).get("avg_ms", 0)
        if signal_time > 1000:
            bottlenecks.append("シグナル生成処理が重い（1秒超過）")
        
        exec_time = timing_analysis.get("order_execution", {}).get("avg_ms", 0)
        if exec_time > 2000:
            bottlenecks.append("注文実行処理が重い（2秒超過）")
        
        slow_fills = slippage_analysis.get("slow_fills", 0)
        if slow_fills > 0:
            bottlenecks.append(f"約定遅延が{slow_fills}件発生")
        
        return bottlenecks


# グローバルPDCA分析エンジンインスタンス
_pdca_analyzer: Optional[PDCAAnalyzer] = None


def get_pdca_analyzer(performance_logger: PerformanceLogger) -> PDCAAnalyzer:
    """グローバルPDCA分析エンジン取得"""
    global _pdca_analyzer
    if _pdca_analyzer is None:
        _pdca_analyzer = PDCAAnalyzer(performance_logger)
    return _pdca_analyzer