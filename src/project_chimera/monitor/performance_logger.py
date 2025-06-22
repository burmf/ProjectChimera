"""
取引ロジックパフォーマンス測定・ログシステム
PDCAサイクル実行のための詳細な分析データを収集
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import structlog
from pathlib import Path

logger = structlog.get_logger(__name__)


class PerformancePhase(Enum):
    """PDCA サイクルのフェーズ"""
    PLAN = "plan"       # 計画: 戦略パラメータ設定
    DO = "do"           # 実行: 実際の取引実行
    CHECK = "check"     # 評価: パフォーマンス分析
    ACT = "act"         # 改善: パラメータ調整


class TradeResult(Enum):
    """取引結果"""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"
    PENDING = "pending"


@dataclass
class StrategyPerformanceMetrics:
    """戦略パフォーマンス指標"""
    strategy_name: str
    timestamp: datetime
    
    # 基本統計
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # 収益指標
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    
    # リスク指標
    max_drawdown: float = 0.0
    max_consecutive_losses: int = 0
    sharpe_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # 実行指標
    signal_generation_time_ms: float = 0.0
    order_execution_time_ms: float = 0.0
    latency_p95_ms: float = 0.0
    
    # 戦略固有パラメータ
    strategy_parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.strategy_parameters is None:
            self.strategy_parameters = {}


@dataclass
class TradeExecution:
    """個別取引実行データ"""
    trade_id: str
    strategy_name: str
    timestamp: datetime
    symbol: str
    
    # シグナル情報
    signal_type: str  # buy/sell
    signal_confidence: float
    signal_generation_time_ms: float
    
    # 実行情報
    entry_price: float
    exit_price: Optional[float] = None
    quantity: float = 0.0
    commission: float = 0.0
    
    # パフォーマンス
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    hold_duration_seconds: Optional[float] = None
    
    # タイミング分析
    order_placement_time_ms: float = 0.0
    fill_time_ms: float = 0.0
    total_execution_time_ms: float = 0.0
    
    # マーケット条件
    market_volatility: Optional[float] = None
    spread_bps: Optional[float] = None
    volume_at_signal: Optional[float] = None
    
    # 結果分類
    result: TradeResult = TradeResult.PENDING
    
    # 戦略固有データ
    strategy_context: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.strategy_context is None:
            self.strategy_context = {}


@dataclass
class PDCALogEntry:
    """PDCAサイクル用ログエントリ"""
    session_id: str
    phase: PerformancePhase
    timestamp: datetime
    strategy_name: str
    
    # Plan フェーズ
    hypothesis: Optional[str] = None
    target_metrics: Optional[Dict[str, float]] = None
    parameter_changes: Optional[Dict[str, Any]] = None
    
    # Do フェーズ
    execution_summary: Optional[Dict[str, Any]] = None
    market_conditions: Optional[Dict[str, Any]] = None
    
    # Check フェーズ
    actual_metrics: Optional[Dict[str, float]] = None
    performance_analysis: Optional[Dict[str, Any]] = None
    
    # Act フェーズ
    insights: Optional[List[str]] = None
    next_actions: Optional[List[str]] = None
    optimization_suggestions: Optional[Dict[str, Any]] = None
    
    # メタデータ
    confidence_level: float = 0.0
    data_quality_score: float = 0.0


class PerformanceLogger:
    """
    取引ロジックパフォーマンス測定・ログシステム
    
    機能:
    1. リアルタイム取引パフォーマンス追跡
    2. 戦略別詳細分析
    3. PDCAサイクル管理
    4. 自動最適化提案
    """
    
    def __init__(self, log_directory: str = "logs/performance"):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        # パフォーマンスデータ保存
        self.strategy_metrics: Dict[str, StrategyPerformanceMetrics] = {}
        self.trade_history: List[TradeExecution] = []
        self.pdca_log: List[PDCALogEntry] = []
        
        # セッション管理
        self.current_session_id = self._generate_session_id()
        self.session_start_time = datetime.now()
        
        # リアルタイム集計
        self.real_time_stats = {}
        
        # ファイルハンドラ
        self.performance_file = self.log_directory / f"performance_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self.pdca_file = self.log_directory / f"pdca_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        logger.info("PerformanceLogger initialized", 
                   session_id=self.current_session_id,
                   log_directory=str(self.log_directory))
    
    def _generate_session_id(self) -> str:
        """セッションID生成"""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    async def log_trade_execution(self, trade: TradeExecution) -> None:
        """取引実行ログ記録"""
        try:
            # 取引履歴に追加
            self.trade_history.append(trade)
            
            # 戦略別統計更新
            await self._update_strategy_metrics(trade)
            
            # ファイル出力
            trade_data = {
                "type": "trade_execution",
                "timestamp": trade.timestamp.isoformat(),
                "session_id": self.current_session_id,
                "data": asdict(trade)
            }
            
            await self._write_to_file(self.performance_file, trade_data)
            
            # リアルタイム統計更新
            await self._update_realtime_stats(trade)
            
            logger.info("Trade execution logged",
                       trade_id=trade.trade_id,
                       strategy=trade.strategy_name,
                       pnl=trade.pnl,
                       result=trade.result.value)
                       
        except Exception as e:
            logger.error("Failed to log trade execution", error=str(e))
    
    async def _update_strategy_metrics(self, trade: TradeExecution) -> None:
        """戦略別メトリクス更新"""
        strategy_name = trade.strategy_name
        
        if strategy_name not in self.strategy_metrics:
            self.strategy_metrics[strategy_name] = StrategyPerformanceMetrics(
                strategy_name=strategy_name,
                timestamp=datetime.now(),
                strategy_parameters=trade.strategy_context.copy()
            )
        
        metrics = self.strategy_metrics[strategy_name]
        metrics.timestamp = datetime.now()
        
        # 基本統計更新
        metrics.total_trades += 1
        
        if trade.result == TradeResult.WIN:
            metrics.winning_trades += 1
            if trade.pnl:
                metrics.gross_profit += trade.pnl
        elif trade.result == TradeResult.LOSS:
            metrics.losing_trades += 1
            if trade.pnl:
                metrics.gross_loss += abs(trade.pnl)
        
        # 勝率計算
        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades
        
        # 平均損益計算
        if metrics.winning_trades > 0:
            metrics.average_win = metrics.gross_profit / metrics.winning_trades
        if metrics.losing_trades > 0:
            metrics.average_loss = metrics.gross_loss / metrics.losing_trades
        
        # プロフィットファクター計算
        if metrics.gross_loss > 0:
            metrics.profit_factor = metrics.gross_profit / metrics.gross_loss
        
        # トータルPnL更新
        if trade.pnl:
            metrics.total_pnl += trade.pnl
        
        # 実行時間統計更新
        if trade.signal_generation_time_ms > 0:
            # 移動平均で更新
            alpha = 0.1
            metrics.signal_generation_time_ms = (
                alpha * trade.signal_generation_time_ms + 
                (1 - alpha) * metrics.signal_generation_time_ms
            )
        
        if trade.total_execution_time_ms > 0:
            metrics.order_execution_time_ms = (
                alpha * trade.total_execution_time_ms + 
                (1 - alpha) * metrics.order_execution_time_ms
            )
    
    async def _update_realtime_stats(self, trade: TradeExecution) -> None:
        """リアルタイム統計更新"""
        current_time = datetime.now()
        time_window = timedelta(minutes=15)  # 15分間のリアルタイム統計
        
        # 時間枠内の取引のみ抽出
        recent_trades = [
            t for t in self.trade_history 
            if current_time - t.timestamp <= time_window
        ]
        
        # リアルタイム統計計算
        total_recent = len(recent_trades)
        winning_recent = len([t for t in recent_trades if t.result == TradeResult.WIN])
        
        self.real_time_stats = {
            "timestamp": current_time.isoformat(),
            "window_minutes": 15,
            "total_trades": total_recent,
            "win_rate": winning_recent / total_recent if total_recent > 0 else 0,
            "recent_pnl": sum(t.pnl or 0 for t in recent_trades),
            "avg_execution_time_ms": sum(t.total_execution_time_ms for t in recent_trades) / total_recent if total_recent > 0 else 0
        }
    
    async def start_pdca_cycle(self, strategy_name: str, hypothesis: str, 
                              target_metrics: Dict[str, float],
                              parameter_changes: Dict[str, Any]) -> str:
        """PDCAサイクル開始 (Plan フェーズ)"""
        pdca_entry = PDCALogEntry(
            session_id=self.current_session_id,
            phase=PerformancePhase.PLAN,
            timestamp=datetime.now(),
            strategy_name=strategy_name,
            hypothesis=hypothesis,
            target_metrics=target_metrics,
            parameter_changes=parameter_changes
        )
        
        self.pdca_log.append(pdca_entry)
        
        await self._write_to_file(self.pdca_file, {
            "type": "pdca_plan",
            "session_id": self.current_session_id,
            "data": asdict(pdca_entry)
        })
        
        logger.info("PDCA cycle started - Plan phase",
                   strategy=strategy_name,
                   hypothesis=hypothesis,
                   targets=target_metrics)
        
        return self.current_session_id
    
    async def log_execution_phase(self, strategy_name: str,
                                 execution_summary: Dict[str, Any],
                                 market_conditions: Dict[str, Any]) -> None:
        """実行フェーズログ (Do フェーズ)"""
        pdca_entry = PDCALogEntry(
            session_id=self.current_session_id,
            phase=PerformancePhase.DO,
            timestamp=datetime.now(),
            strategy_name=strategy_name,
            execution_summary=execution_summary,
            market_conditions=market_conditions
        )
        
        self.pdca_log.append(pdca_entry)
        
        await self._write_to_file(self.pdca_file, {
            "type": "pdca_do",
            "session_id": self.current_session_id,
            "data": asdict(pdca_entry)
        })
        
        logger.info("PDCA cycle - Do phase logged",
                   strategy=strategy_name,
                   trades_executed=execution_summary.get("trades_executed", 0))
    
    async def analyze_performance(self, strategy_name: str) -> Dict[str, Any]:
        """パフォーマンス分析 (Check フェーズ)"""
        if strategy_name not in self.strategy_metrics:
            return {"error": "No data available for strategy"}
        
        metrics = self.strategy_metrics[strategy_name]
        
        # 詳細分析実行
        analysis = {
            "basic_metrics": asdict(metrics),
            "advanced_analysis": await self._calculate_advanced_metrics(strategy_name),
            "optimization_signals": await self._generate_optimization_signals(strategy_name),
            "risk_assessment": await self._assess_risk_profile(strategy_name)
        }
        
        # Check フェーズログ
        pdca_entry = PDCALogEntry(
            session_id=self.current_session_id,
            phase=PerformancePhase.CHECK,
            timestamp=datetime.now(),
            strategy_name=strategy_name,
            actual_metrics={
                "win_rate": metrics.win_rate,
                "profit_factor": metrics.profit_factor,
                "total_pnl": metrics.total_pnl,
                "max_drawdown": metrics.max_drawdown
            },
            performance_analysis=analysis,
            confidence_level=analysis.get("confidence_level", 0.0),
            data_quality_score=analysis.get("data_quality_score", 0.0)
        )
        
        self.pdca_log.append(pdca_entry)
        
        await self._write_to_file(self.pdca_file, {
            "type": "pdca_check",
            "session_id": self.current_session_id,
            "data": asdict(pdca_entry)
        })
        
        logger.info("Performance analysis completed - Check phase",
                   strategy=strategy_name,
                   win_rate=metrics.win_rate,
                   profit_factor=metrics.profit_factor)
        
        return analysis
    
    async def generate_improvements(self, strategy_name: str, 
                                  analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """改善提案生成 (Act フェーズ)"""
        insights = []
        next_actions = []
        optimization_suggestions = {}
        
        # 戦略別取引データ取得
        strategy_trades = [t for t in self.trade_history if t.strategy_name == strategy_name]
        
        if not strategy_trades:
            return {"error": "No trade data available for improvement analysis"}
        
        # 勝率分析
        win_rate = len([t for t in strategy_trades if t.result == TradeResult.WIN]) / len(strategy_trades)
        if win_rate < 0.5:
            insights.append(f"勝率が低い ({win_rate:.2%}) - エントリー条件の見直しが必要")
            next_actions.append("エントリーシグナルの精度向上")
            optimization_suggestions["entry_threshold"] = "より厳格な条件設定を推奨"
        
        # 実行時間分析
        avg_execution_time = sum(t.total_execution_time_ms for t in strategy_trades) / len(strategy_trades)
        if avg_execution_time > 1000:  # 1秒以上
            insights.append(f"実行時間が長い ({avg_execution_time:.0f}ms) - 最適化が必要")
            next_actions.append("実行ロジックの最適化")
            optimization_suggestions["execution_optimization"] = "並列処理またはキャッシュ導入を検討"
        
        # ドローダウン分析
        metrics = self.strategy_metrics.get(strategy_name)
        if metrics and metrics.max_drawdown > 0.1:  # 10%以上
            insights.append(f"最大ドローダウンが大きい ({metrics.max_drawdown:.2%}) - リスク管理強化が必要")
            next_actions.append("ポジションサイズ調整")
            optimization_suggestions["risk_management"] = "Kelly基準の調整またはストップロス設定の見直し"
        
        # Act フェーズログ
        pdca_entry = PDCALogEntry(
            session_id=self.current_session_id,
            phase=PerformancePhase.ACT,
            timestamp=datetime.now(),
            strategy_name=strategy_name,
            insights=insights,
            next_actions=next_actions,
            optimization_suggestions=optimization_suggestions
        )
        
        self.pdca_log.append(pdca_entry)
        
        await self._write_to_file(self.pdca_file, {
            "type": "pdca_act",
            "session_id": self.current_session_id,
            "data": asdict(pdca_entry)
        })
        
        improvement_plan = {
            "insights": insights,
            "next_actions": next_actions,
            "optimization_suggestions": optimization_suggestions,
            "priority_score": len(insights) * 10,  # 簡易優先度スコア
            "implementation_difficulty": "medium"  # TODO: より詳細な分析
        }
        
        logger.info("Improvement plan generated - Act phase",
                   strategy=strategy_name,
                   insights_count=len(insights),
                   actions_count=len(next_actions))
        
        return improvement_plan
    
    async def _calculate_advanced_metrics(self, strategy_name: str) -> Dict[str, Any]:
        """高度なメトリクス計算"""
        strategy_trades = [t for t in self.trade_history if t.strategy_name == strategy_name]
        
        if not strategy_trades:
            return {}
        
        # Sharpe ratio計算（簡易版）
        returns = [t.pnl_percentage or 0 for t in strategy_trades if t.pnl_percentage is not None]
        if len(returns) > 1:
            avg_return = sum(returns) / len(returns)
            return_std = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
            sharpe_ratio = avg_return / return_std if return_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        # 最大連続負け数
        consecutive_losses = 0
        max_consecutive_losses = 0
        for trade in strategy_trades:
            if trade.result == TradeResult.LOSS:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0
        
        # 平均保有時間
        hold_durations = [t.hold_duration_seconds for t in strategy_trades if t.hold_duration_seconds is not None]
        avg_hold_duration = sum(hold_durations) / len(hold_durations) if hold_durations else 0
        
        return {
            "sharpe_ratio": sharpe_ratio,
            "max_consecutive_losses": max_consecutive_losses,
            "average_hold_duration_seconds": avg_hold_duration,
            "total_commission": sum(t.commission for t in strategy_trades),
            "confidence_level": min(len(strategy_trades) / 30, 1.0),  # 30取引で100%信頼度
            "data_quality_score": self._calculate_data_quality(strategy_trades)
        }
    
    async def _generate_optimization_signals(self, strategy_name: str) -> List[Dict[str, Any]]:
        """最適化シグナル生成"""
        signals = []
        strategy_trades = [t for t in self.trade_history if t.strategy_name == strategy_name]
        
        if not strategy_trades:
            return signals
        
        # 時間帯別パフォーマンス分析
        hourly_performance = {}
        for trade in strategy_trades:
            hour = trade.timestamp.hour
            if hour not in hourly_performance:
                hourly_performance[hour] = []
            if trade.pnl is not None:
                hourly_performance[hour].append(trade.pnl)
        
        # 最高パフォーマンス時間帯特定
        best_hours = []
        for hour, pnls in hourly_performance.items():
            if len(pnls) >= 3:  # 最低3取引
                avg_pnl = sum(pnls) / len(pnls)
                if avg_pnl > 0:
                    best_hours.append((hour, avg_pnl))
        
        if best_hours:
            best_hours.sort(key=lambda x: x[1], reverse=True)
            signals.append({
                "type": "time_optimization",
                "message": f"時間帯 {best_hours[0][0]}:00-{best_hours[0][0]+1}:00 のパフォーマンスが最高",
                "recommendation": "この時間帯での取引頻度を増加させることを検討",
                "priority": "high" if best_hours[0][1] > 0.01 else "medium"
            })
        
        # ボラティリティ最適化
        volatility_trades = [t for t in strategy_trades if t.market_volatility is not None]
        if len(volatility_trades) >= 5:
            high_vol_pnl = [t.pnl for t in volatility_trades if t.market_volatility > 0.02 and t.pnl is not None]
            low_vol_pnl = [t.pnl for t in volatility_trades if t.market_volatility <= 0.02 and t.pnl is not None]
            
            if high_vol_pnl and low_vol_pnl:
                high_vol_avg = sum(high_vol_pnl) / len(high_vol_pnl)
                low_vol_avg = sum(low_vol_pnl) / len(low_vol_pnl)
                
                if high_vol_avg > low_vol_avg * 1.5:
                    signals.append({
                        "type": "volatility_optimization",
                        "message": "高ボラティリティ環境でのパフォーマンスが良好",
                        "recommendation": "ボラティリティフィルターの導入を検討",
                        "priority": "medium"
                    })
        
        return signals
    
    async def _assess_risk_profile(self, strategy_name: str) -> Dict[str, Any]:
        """リスクプロファイル評価"""
        strategy_trades = [t for t in self.trade_history if t.strategy_name == strategy_name]
        
        if not strategy_trades:
            return {}
        
        # PnL分布分析
        pnls = [t.pnl for t in strategy_trades if t.pnl is not None]
        if not pnls:
            return {}
        
        # VaR計算（簡易版）
        pnls_sorted = sorted(pnls)
        var_95 = pnls_sorted[int(len(pnls_sorted) * 0.05)] if len(pnls_sorted) > 20 else min(pnls)
        
        # 最大損失
        max_loss = min(pnls)
        
        # リスク・リターン比
        positive_pnls = [p for p in pnls if p > 0]
        negative_pnls = [p for p in pnls if p < 0]
        
        avg_profit = sum(positive_pnls) / len(positive_pnls) if positive_pnls else 0
        avg_loss = sum(negative_pnls) / len(negative_pnls) if negative_pnls else 0
        risk_return_ratio = abs(avg_profit / avg_loss) if avg_loss < 0 else float('inf')
        
        return {
            "var_95": var_95,
            "max_loss": max_loss,
            "risk_return_ratio": risk_return_ratio,
            "volatility": (sum((p - sum(pnls)/len(pnls))**2 for p in pnls) / len(pnls))**0.5,
            "risk_level": "high" if var_95 < -0.02 else "medium" if var_95 < -0.01 else "low"
        }
    
    def _calculate_data_quality(self, trades: List[TradeExecution]) -> float:
        """データ品質スコア計算"""
        if not trades:
            return 0.0
        
        # 必須フィールドの完全性チェック
        complete_fields = 0
        total_fields = 0
        
        for trade in trades:
            fields_to_check = [
                trade.pnl is not None,
                trade.exit_price is not None,
                trade.hold_duration_seconds is not None,
                trade.total_execution_time_ms > 0,
                trade.market_volatility is not None,
                trade.spread_bps is not None
            ]
            complete_fields += sum(fields_to_check)
            total_fields += len(fields_to_check)
        
        return complete_fields / total_fields if total_fields > 0 else 0.0
    
    async def _write_to_file(self, file_path: Path, data: Dict[str, Any]) -> None:
        """ファイル出力"""
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, default=str)
                f.write('\n')
        except Exception as e:
            logger.error("Failed to write to log file", file=str(file_path), error=str(e))
    
    async def get_realtime_dashboard_data(self) -> Dict[str, Any]:
        """リアルタイムダッシュボード用データ取得"""
        dashboard_data = {
            "session_info": {
                "session_id": self.current_session_id,
                "start_time": self.session_start_time.isoformat(),
                "uptime_minutes": (datetime.now() - self.session_start_time).total_seconds() / 60
            },
            "overall_stats": {
                "total_trades": len(self.trade_history),
                "total_strategies": len(self.strategy_metrics),
                "active_pdca_cycles": len([p for p in self.pdca_log if p.phase != PerformancePhase.ACT])
            },
            "strategy_performance": {
                name: {
                    "win_rate": metrics.win_rate,
                    "total_pnl": metrics.total_pnl,
                    "total_trades": metrics.total_trades,
                    "profit_factor": metrics.profit_factor
                }
                for name, metrics in self.strategy_metrics.items()
            },
            "realtime_stats": self.real_time_stats,
            "recent_trades": [
                {
                    "timestamp": trade.timestamp.isoformat(),
                    "strategy": trade.strategy_name,
                    "symbol": trade.symbol,
                    "pnl": trade.pnl,
                    "result": trade.result.value
                }
                for trade in self.trade_history[-10:]  # 直近10取引
            ]
        }
        
        return dashboard_data
    
    async def export_analysis_report(self, strategy_name: Optional[str] = None) -> str:
        """分析レポート出力"""
        report_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if strategy_name:
            report_data = {
                "strategy": strategy_name,
                "metrics": asdict(self.strategy_metrics.get(strategy_name, {})),
                "trades": [asdict(t) for t in self.trade_history if t.strategy_name == strategy_name],
                "pdca_log": [asdict(p) for p in self.pdca_log if p.strategy_name == strategy_name]
            }
            filename = f"strategy_report_{strategy_name}_{report_timestamp}.json"
        else:
            report_data = {
                "session_id": self.current_session_id,
                "all_metrics": {name: asdict(metrics) for name, metrics in self.strategy_metrics.items()},
                "all_trades": [asdict(t) for t in self.trade_history],
                "pdca_log": [asdict(p) for p in self.pdca_log],
                "summary": await self.get_realtime_dashboard_data()
            }
            filename = f"full_report_{report_timestamp}.json"
        
        report_path = self.log_directory / filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info("Analysis report exported", 
                   filename=filename,
                   strategy=strategy_name or "all")
        
        return str(report_path)


# グローバルパフォーマンスロガーインスタンス
_performance_logger: Optional[PerformanceLogger] = None


def get_performance_logger() -> PerformanceLogger:
    """グローバルパフォーマンスロガー取得"""
    global _performance_logger
    if _performance_logger is None:
        _performance_logger = PerformanceLogger()
    return _performance_logger


async def initialize_performance_logging(log_directory: str = "logs/performance") -> PerformanceLogger:
    """パフォーマンスロギング初期化"""
    global _performance_logger
    _performance_logger = PerformanceLogger(log_directory)
    return _performance_logger