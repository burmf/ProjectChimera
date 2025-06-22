"""
パフォーマンス分析ダッシュボード
取引ロジックのPDCAサイクル管理用リアルタイムダッシュボード
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

from ..monitor.performance_logger import get_performance_logger, PerformancePhase, TradeResult
from ..monitor.pdca_analyzer import get_pdca_analyzer

# ページ設定
st.set_page_config(
    page_title="ProjectChimera Performance Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


class PerformanceDashboard:
    """パフォーマンス分析ダッシュボード"""
    
    def __init__(self):
        self.performance_logger = get_performance_logger()
        self.pdca_analyzer = get_pdca_analyzer(self.performance_logger)
        
        # セッション状態初期化
        if 'selected_strategy' not in st.session_state:
            st.session_state.selected_strategy = None
        if 'auto_refresh' not in st.session_state:
            st.session_state.auto_refresh = True
        if 'refresh_interval' not in st.session_state:
            st.session_state.refresh_interval = 30
    
    def run(self):
        """ダッシュボード実行"""
        st.title("📊 ProjectChimera Performance Dashboard")
        st.markdown("---")
        
        # サイドバー設定
        self._render_sidebar()
        
        # メインコンテンツ
        if st.session_state.selected_strategy:
            self._render_strategy_dashboard()
        else:
            self._render_overview_dashboard()
        
        # 自動リフレッシュ
        if st.session_state.auto_refresh:
            st.rerun()
    
    def _render_sidebar(self):
        """サイドバーレンダリング"""
        st.sidebar.header("⚙️ Dashboard Settings")
        
        # 戦略選択
        strategies = list(self.performance_logger.strategy_metrics.keys())
        if strategies:
            selected = st.sidebar.selectbox(
                "Strategy Selection",
                ["Overview"] + strategies,
                index=0 if not st.session_state.selected_strategy else strategies.index(st.session_state.selected_strategy) + 1
            )
            st.session_state.selected_strategy = None if selected == "Overview" else selected
        else:
            st.sidebar.info("No strategy data available")
        
        # リフレッシュ設定
        st.sidebar.header("🔄 Refresh Settings")
        st.session_state.auto_refresh = st.sidebar.checkbox("Auto Refresh", value=st.session_state.auto_refresh)
        if st.session_state.auto_refresh:
            st.session_state.refresh_interval = st.sidebar.slider("Refresh Interval (sec)", 10, 120, st.session_state.refresh_interval)
        
        # データエクスポート
        st.sidebar.header("📁 Data Export")
        if st.sidebar.button("Export Full Report"):
            self._export_full_report()
        
        if st.session_state.selected_strategy and st.sidebar.button("Export Strategy Report"):
            self._export_strategy_report()
        
        # システム情報
        st.sidebar.header("📈 System Status")
        dashboard_data = asyncio.run(self.performance_logger.get_realtime_dashboard_data())
        st.sidebar.metric("Total Trades", dashboard_data["overall_stats"]["total_trades"])
        st.sidebar.metric("Active Strategies", dashboard_data["overall_stats"]["total_strategies"])
        st.sidebar.metric("Uptime (min)", f"{dashboard_data['session_info']['uptime_minutes']:.1f}")
    
    def _render_overview_dashboard(self):
        """概要ダッシュボード"""
        st.header("🌐 Overall Performance Overview")
        
        # リアルタイムデータ取得
        dashboard_data = asyncio.run(self.performance_logger.get_realtime_dashboard_data())
        
        # メトリクス表示
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", dashboard_data["overall_stats"]["total_trades"])
        with col2:
            st.metric("Active Strategies", dashboard_data["overall_stats"]["total_strategies"])
        with col3:
            total_pnl = sum(perf["total_pnl"] for perf in dashboard_data["strategy_performance"].values())
            st.metric("Total P&L", f"${total_pnl:.2f}")
        with col4:
            avg_win_rate = sum(perf["win_rate"] for perf in dashboard_data["strategy_performance"].values()) / len(dashboard_data["strategy_performance"]) if dashboard_data["strategy_performance"] else 0
            st.metric("Avg Win Rate", f"{avg_win_rate:.1%}")
        
        # 戦略別パフォーマンス比較
        if dashboard_data["strategy_performance"]:
            st.subheader("📊 Strategy Performance Comparison")
            self._render_strategy_comparison(dashboard_data["strategy_performance"])
        
        # 最近の取引履歴
        st.subheader("📝 Recent Trades")
        self._render_recent_trades(dashboard_data["recent_trades"])
        
        # リアルタイム統計
        if dashboard_data["realtime_stats"]:
            st.subheader("⚡ Real-time Statistics (15min window)")
            self._render_realtime_stats(dashboard_data["realtime_stats"])
    
    def _render_strategy_dashboard(self):
        """戦略別ダッシュボード"""
        strategy_name = st.session_state.selected_strategy
        st.header(f"🎯 {strategy_name} Performance Analysis")
        
        # 戦略メトリクス取得
        metrics = self.performance_logger.strategy_metrics.get(strategy_name)
        if not metrics:
            st.error("No data available for this strategy")
            return
        
        # 基本メトリクス表示
        self._render_strategy_metrics(metrics)
        
        # 詳細分析実行
        with st.spinner("Running comprehensive analysis..."):
            analysis = asyncio.run(self.pdca_analyzer.run_comprehensive_analysis(strategy_name))
        
        # 分析結果表示
        self._render_comprehensive_analysis(analysis)
        
        # PDCA管理
        self._render_pdca_management(strategy_name)
        
        # 取引履歴詳細
        self._render_trade_history_details(strategy_name)
    
    def _render_strategy_metrics(self, metrics):
        """戦略メトリクス表示"""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Trades", metrics.total_trades)
        with col2:
            st.metric("Win Rate", f"{metrics.win_rate:.1%}")
        with col3:
            st.metric("Total P&L", f"${metrics.total_pnl:.2f}")
        with col4:
            st.metric("Profit Factor", f"{metrics.profit_factor:.2f}")
        with col5:
            st.metric("Max Drawdown", f"{metrics.max_drawdown:.1%}")
        
        # 詳細メトリクス
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Avg Win", f"${metrics.average_win:.2f}")
            st.metric("Signal Gen Time", f"{metrics.signal_generation_time_ms:.0f}ms")
        
        with col2:
            st.metric("Avg Loss", f"${metrics.average_loss:.2f}")
            st.metric("Execution Time", f"{metrics.order_execution_time_ms:.0f}ms")
        
        with col3:
            st.metric("Gross Profit", f"${metrics.gross_profit:.2f}")
            st.metric("Gross Loss", f"${metrics.gross_loss:.2f}")
    
    def _render_comprehensive_analysis(self, analysis):
        """包括的分析結果表示"""
        if "error" in analysis:
            st.error(f"Analysis error: {analysis['error']}")
            return
        
        # 総合スコア
        st.subheader(f"🎯 Overall Performance Score: {analysis['overall_score']:.1f}/100")
        
        # プログレスバーでスコア表示
        score_color = "green" if analysis['overall_score'] >= 70 else "orange" if analysis['overall_score'] >= 50 else "red"
        st.progress(analysis['overall_score']/100)
        
        # 分析結果タブ
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Basic Performance", "⏰ Temporal Patterns", "🌊 Market Regimes", 
            "🔧 Optimization", "⚠️ Risk Analysis", "⚡ Execution"
        ])
        
        with tab1:
            self._render_basic_analysis(analysis["analyses"]["basic_performance"])
        
        with tab2:
            self._render_temporal_analysis(analysis["analyses"]["temporal_patterns"])
        
        with tab3:
            self._render_regime_analysis(analysis["analyses"]["market_regimes"])
        
        with tab4:
            self._render_optimization_analysis(analysis["analyses"]["parameter_optimization"])
        
        with tab5:
            self._render_risk_analysis(analysis["analyses"]["risk_metrics"])
        
        with tab6:
            self._render_execution_analysis(analysis["analyses"]["execution_efficiency"])
        
        # 推奨事項
        st.subheader("💡 Recommendations")
        for i, rec in enumerate(analysis["recommendations"][:5]):  # 上位5つ
            priority_color = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            st.write(f"{priority_color.get(rec['priority'], '⚪')} **{rec['title']}**")
            st.write(f"   {rec['description']}")
            st.write(f"   *Action: {rec['action']}*")
            st.write(f"   *Expected: {rec['expected_improvement']} (Confidence: {rec['confidence']})*")
            st.write("---")
    
    def _render_basic_analysis(self, basic_analysis):
        """基本分析表示"""
        if "error" in basic_analysis:
            st.error(basic_analysis["error"])
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Sharpe Ratio", f"{basic_analysis['sharpe_ratio']:.2f}")
            st.metric("Return Volatility", f"{basic_analysis['return_volatility']:.2%}")
            st.metric("Average Return", f"{basic_analysis['average_return']:.2%}")
        
        with col2:
            st.metric("Max Drawdown", f"{basic_analysis['max_drawdown']:.1%}")
            st.metric("Confidence Level", f"{basic_analysis['confidence_level']:.1%}")
            st.metric("Quality Score", f"{basic_analysis['quality_score']:.1%}")
        
        # P&L分布チャート
        trades = [t for t in self.performance_logger.trade_history 
                 if t.strategy_name == st.session_state.selected_strategy and t.pnl is not None]
        
        if trades:
            pnls = [t.pnl for t in trades]
            fig = px.histogram(x=pnls, title="P&L Distribution", nbins=20)
            fig.update_layout(xaxis_title="P&L ($)", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_temporal_analysis(self, temporal_analysis):
        """時系列分析表示"""
        if not temporal_analysis:
            st.info("Insufficient data for temporal analysis")
            return
        
        # 時間帯別パフォーマンス
        if "hourly_stats" in temporal_analysis:
            hourly_data = temporal_analysis["hourly_stats"]
            if hourly_data:
                hours = list(hourly_data.keys())
                avg_pnls = [hourly_data[h]["avg_pnl"] for h in hours]
                
                fig = px.bar(x=hours, y=avg_pnls, title="Average P&L by Hour")
                fig.update_layout(xaxis_title="Hour", yaxis_title="Average P&L ($)")
                st.plotly_chart(fig, use_container_width=True)
        
        # 最適取引時間
        if "best_trading_hours" in temporal_analysis and temporal_analysis["best_trading_hours"]:
            st.write("🕐 **Best Trading Hours:**")
            for hour, pnl in temporal_analysis["best_trading_hours"][:3]:
                st.write(f"   {hour}:00-{hour+1}:00 (Avg P&L: ${pnl:.2f})")
        
        # トレンド分析
        if "trend_analysis" in temporal_analysis and temporal_analysis["trend_analysis"]:
            trend = temporal_analysis["trend_analysis"]
            trend_icon = {"improving": "📈", "declining": "📉", "stable": "➡️"}
            st.write(f"📊 **Performance Trend:** {trend_icon.get(trend['performance_trend'], '❓')} {trend['performance_trend'].title()}")
            st.write(f"   Recent 30d Avg: ${trend['recent_30d_avg_pnl']:.2f}")
            st.write(f"   Overall Avg: ${trend['overall_avg_pnl']:.2f}")
    
    def _render_regime_analysis(self, regime_analysis):
        """レジーム分析表示"""
        if not regime_analysis:
            st.info("Insufficient data for regime analysis")
            return
        
        # ボラティリティレジーム
        if "volatility_regimes" in regime_analysis:
            regimes = regime_analysis["volatility_regimes"]
            if regimes:
                regime_names = list(regimes.keys())
                regime_pnls = [regimes[r]["avg_pnl"] for r in regime_names]
                
                fig = px.bar(x=regime_names, y=regime_pnls, title="Performance by Volatility Regime")
                fig.update_layout(xaxis_title="Volatility Regime", yaxis_title="Average P&L ($)")
                st.plotly_chart(fig, use_container_width=True)
        
        # 最適レジーム
        if "best_regime" in regime_analysis and regime_analysis["best_regime"]:
            st.write(f"🌟 **Best Regime:** {regime_analysis['best_regime']}")
            best_perf = regime_analysis["best_regime_performance"]
            st.write(f"   Avg P&L: ${best_perf['avg_pnl']:.2f}")
            st.write(f"   Win Rate: {best_perf['win_rate']:.1%}")
        
        # 推奨事項
        if "recommendations" in regime_analysis:
            st.write("💡 **Regime Recommendations:**")
            for rec in regime_analysis["recommendations"]:
                st.write(f"   • {rec}")
    
    def _render_optimization_analysis(self, optimization_analysis):
        """最適化分析表示"""
        if not optimization_analysis:
            st.info("Insufficient data for optimization analysis")
            return
        
        # 信頼度分析
        if "confidence_analysis" in optimization_analysis:
            conf_data = optimization_analysis["confidence_analysis"]
            if conf_data:
                conf_ranges = list(conf_data.keys())
                avg_pnls = [conf_data[c]["avg_pnl"] for c in conf_ranges]
                
                fig = px.bar(x=conf_ranges, y=avg_pnls, title="Performance by Confidence Level")
                fig.update_layout(xaxis_title="Confidence Range", yaxis_title="Average P&L ($)")
                st.plotly_chart(fig, use_container_width=True)
        
        # 最適信頼度閾値
        if "optimal_confidence_threshold" in optimization_analysis and optimization_analysis["optimal_confidence_threshold"]:
            st.write(f"🎯 **Optimal Confidence Threshold:** {optimization_analysis['optimal_confidence_threshold']:.2f}")
        
        # 最適化推奨
        if "optimization_recommendations" in optimization_analysis:
            recs = optimization_analysis["optimization_recommendations"]
            if recs:
                st.write("🔧 **Parameter Optimization Recommendations:**")
                for rec in recs:
                    st.write(f"   • **{rec['parameter_name']}**: {rec['current_value']} → {rec['recommended_value']}")
                    st.write(f"     Reasoning: {rec['reasoning']}")
    
    def _render_risk_analysis(self, risk_analysis):
        """リスク分析表示"""
        if not risk_analysis:
            st.info("Insufficient data for risk analysis")
            return
        
        # リスクメトリクス
        col1, col2 = st.columns(2)
        
        with col1:
            if "var_calculations" in risk_analysis:
                for var_name, var_value in risk_analysis["var_calculations"].items():
                    st.metric(var_name, f"${var_value:.2f}")
        
        with col2:
            if "risk_adjusted_metrics" in risk_analysis:
                for metric_name, metric_value in risk_analysis["risk_adjusted_metrics"].items():
                    st.metric(metric_name.replace("_", " ").title(), f"{metric_value:.2f}")
        
        # リスクスコア
        if "risk_score" in risk_analysis:
            risk_score = risk_analysis["risk_score"]
            score_color = "green" if risk_score >= 70 else "orange" if risk_score >= 50 else "red"
            st.write(f"⚠️ **Risk Score:** {risk_score:.1f}/100")
            st.progress(risk_score/100)
        
        # リスクアラート
        if "risk_alerts" in risk_analysis and risk_analysis["risk_alerts"]:
            st.write("🚨 **Risk Alerts:**")
            for alert in risk_analysis["risk_alerts"]:
                st.warning(alert)
        
        # リスク推奨事項
        if "risk_recommendations" in risk_analysis:
            st.write("💡 **Risk Management Recommendations:**")
            for rec in risk_analysis["risk_recommendations"]:
                st.write(f"   • {rec}")
    
    def _render_execution_analysis(self, execution_analysis):
        """実行分析表示"""
        if not execution_analysis:
            st.info("Insufficient data for execution analysis")
            return
        
        # 効率スコア
        if "efficiency_score" in execution_analysis:
            eff_score = execution_analysis["efficiency_score"]
            st.write(f"⚡ **Execution Efficiency Score:** {eff_score:.1f}/100")
            st.progress(eff_score/100)
        
        # タイミング分析
        if "timing_analysis" in execution_analysis:
            timing = execution_analysis["timing_analysis"]
            col1, col2 = st.columns(2)
            
            with col1:
                if "signal_generation" in timing:
                    sg = timing["signal_generation"]
                    st.metric("Avg Signal Time", f"{sg['avg_ms']:.0f}ms")
                    st.metric("P95 Signal Time", f"{sg['p95_ms']:.0f}ms")
            
            with col2:
                if "order_execution" in timing:
                    oe = timing["order_execution"]
                    st.metric("Avg Execution Time", f"{oe['avg_ms']:.0f}ms")
                    st.metric("P95 Execution Time", f"{oe['p95_ms']:.0f}ms")
        
        # ボトルネック
        if "bottlenecks" in execution_analysis and execution_analysis["bottlenecks"]:
            st.write("🚧 **Performance Bottlenecks:**")
            for bottleneck in execution_analysis["bottlenecks"]:
                st.warning(bottleneck)
        
        # 改善推奨
        if "efficiency_recommendations" in execution_analysis:
            st.write("💡 **Efficiency Recommendations:**")
            for rec in execution_analysis["efficiency_recommendations"]:
                st.write(f"   • {rec}")
    
    def _render_pdca_management(self, strategy_name):
        """PDCA管理セクション"""
        st.subheader("🔄 PDCA Cycle Management")
        
        # 現在のPDCAセッション状態
        pdca_sessions = [p for p in self.performance_logger.pdca_log if p.strategy_name == strategy_name]
        current_session = pdca_sessions[-1] if pdca_sessions else None
        
        if current_session:
            st.write(f"**Current Phase:** {current_session.phase.value.title()}")
            st.write(f"**Session ID:** {current_session.session_id}")
            
            if current_session.hypothesis:
                st.write(f"**Hypothesis:** {current_session.hypothesis}")
        
        # 新しいPDCAサイクル開始
        with st.expander("Start New PDCA Cycle"):
            hypothesis = st.text_area("Hypothesis", placeholder="例: 信頼度閾値を0.7に上げることで勝率が向上する")
            
            col1, col2 = st.columns(2)
            with col1:
                target_win_rate = st.number_input("Target Win Rate", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
                target_profit_factor = st.number_input("Target Profit Factor", min_value=0.0, value=2.0, step=0.1)
            
            with col2:
                param_name = st.text_input("Parameter to Change", placeholder="confidence_threshold")
                param_value = st.text_input("New Value", placeholder="0.7")
            
            if st.button("Start PDCA Cycle") and hypothesis:
                target_metrics = {
                    "win_rate": target_win_rate,
                    "profit_factor": target_profit_factor
                }
                parameter_changes = {param_name: param_value} if param_name and param_value else {}
                
                # 非同期でPDCAサイクル開始（実際の実装では適切な非同期処理が必要）
                st.success("PDCA Cycle started! (Note: This is a demo - actual implementation needed)")
        
        # PDCAログ履歴
        if pdca_sessions:
            st.write("📋 **PDCA History:**")
            for session in pdca_sessions[-5:]:  # 最新5件
                with st.expander(f"{session.timestamp.strftime('%Y-%m-%d %H:%M')} - {session.phase.value}"):
                    st.json(session.__dict__, default=str)
    
    def _render_trade_history_details(self, strategy_name):
        """取引履歴詳細"""
        st.subheader("📊 Trade History Details")
        
        trades = [t for t in self.performance_logger.trade_history if t.strategy_name == strategy_name]
        
        if not trades:
            st.info("No trade history available")
            return
        
        # 取引データをDataFrameに変換
        trade_data = []
        for trade in trades[-50:]:  # 最新50取引
            trade_data.append({
                "Timestamp": trade.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "Symbol": trade.symbol,
                "Signal": trade.signal_type,
                "Confidence": f"{trade.signal_confidence:.2f}",
                "Entry Price": f"${trade.entry_price:.4f}",
                "Exit Price": f"${trade.exit_price:.4f}" if trade.exit_price else "N/A",
                "P&L": f"${trade.pnl:.2f}" if trade.pnl else "N/A",
                "P&L %": f"{trade.pnl_percentage:.2f}%" if trade.pnl_percentage else "N/A",
                "Result": trade.result.value,
                "Hold Time (min)": f"{trade.hold_duration_seconds/60:.1f}" if trade.hold_duration_seconds else "N/A",
                "Exec Time (ms)": f"{trade.total_execution_time_ms:.0f}" if trade.total_execution_time_ms else "N/A"
            })
        
        df = pd.DataFrame(trade_data)
        st.dataframe(df, use_container_width=True)
        
        # P&L推移チャート
        pnl_data = [(t.timestamp, t.pnl or 0) for t in trades if t.pnl is not None]
        if pnl_data:
            timestamps, pnls = zip(*pnl_data)
            cumulative_pnl = pd.Series(pnls).cumsum().tolist()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=timestamps, y=cumulative_pnl, mode='lines', name='Cumulative P&L'))
            fig.update_layout(title="Cumulative P&L Over Time", xaxis_title="Time", yaxis_title="Cumulative P&L ($)")
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_strategy_comparison(self, strategy_performance):
        """戦略比較チャート"""
        strategies = list(strategy_performance.keys())
        win_rates = [strategy_performance[s]["win_rate"] for s in strategies]
        total_pnls = [strategy_performance[s]["total_pnl"] for s in strategies]
        profit_factors = [strategy_performance[s]["profit_factor"] for s in strategies]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(x=strategies, y=win_rates, title="Win Rate by Strategy")
            fig1.update_layout(yaxis_title="Win Rate", xaxis_title="Strategy")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            fig2 = px.bar(x=strategies, y=total_pnls, title="Total P&L by Strategy")
            fig2.update_layout(yaxis_title="Total P&L ($)", xaxis_title="Strategy")
            st.plotly_chart(fig2, use_container_width=True)
        
        # プロフィットファクター比較
        fig3 = px.bar(x=strategies, y=profit_factors, title="Profit Factor by Strategy")
        fig3.update_layout(yaxis_title="Profit Factor", xaxis_title="Strategy")
        st.plotly_chart(fig3, use_container_width=True)
    
    def _render_recent_trades(self, recent_trades):
        """最近の取引表示"""
        if not recent_trades:
            st.info("No recent trades")
            return
        
        # 取引結果別色分け
        for trade in recent_trades[-10:]:  # 最新10取引
            result_color = {"win": "🟢", "loss": "🔴", "breakeven": "🟡", "pending": "⚪"}
            color = result_color.get(trade["result"], "⚪")
            
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            with col1:
                st.write(f"{color} {trade['strategy']} - {trade['symbol']}")
            with col2:
                st.write(f"${trade['pnl']:.2f}" if trade['pnl'] else "Pending")
            with col3:
                st.write(trade['result'].title())
            with col4:
                st.write(trade['timestamp'].split('T')[1][:8])  # 時刻のみ表示
    
    def _render_realtime_stats(self, realtime_stats):
        """リアルタイム統計表示"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("15min Trades", realtime_stats["total_trades"])
        with col2:
            st.metric("15min Win Rate", f"{realtime_stats['win_rate']:.1%}")
        with col3:
            st.metric("15min P&L", f"${realtime_stats['recent_pnl']:.2f}")
        with col4:
            st.metric("Avg Exec Time", f"{realtime_stats['avg_execution_time_ms']:.0f}ms")
    
    def _export_full_report(self):
        """フルレポートエクスポート"""
        try:
            report_path = asyncio.run(self.performance_logger.export_analysis_report())
            st.success(f"Full report exported to: {report_path}")
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
    
    def _export_strategy_report(self):
        """戦略別レポートエクスポート"""
        try:
            strategy_name = st.session_state.selected_strategy
            report_path = asyncio.run(self.performance_logger.export_analysis_report(strategy_name))
            st.success(f"Strategy report exported to: {report_path}")
        except Exception as e:
            st.error(f"Export failed: {str(e)}")


def main():
    """メイン関数"""
    dashboard = PerformanceDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()