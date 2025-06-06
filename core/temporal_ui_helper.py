"""
Temporal UI Helper - UI components for temporal-aware backtesting
時系列制約対応UI補助モジュール
"""

import streamlit as st
import pandas as pd
import datetime
from typing import Tuple, Optional, Dict, Any
from .temporal_backtester import run_temporal_backtest
from .temporal_validator import create_temporal_validator
import logging

logger = logging.getLogger(__name__)

def create_temporal_backtest_ui() -> Dict[str, Any]:
    """時系列制約対応バックテストUI"""
    
    st.subheader("🛡️ 時系列制約対応バックテスト")
    st.info("先見バイアス（Look-ahead bias）を防ぐバックテスト機能")
    
    # バックテスト設定
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**期間設定**")
        start_date = st.date_input(
            "開始日",
            value=datetime.date.today() - datetime.timedelta(days=30),
            key="temporal_start_date"
        )
        end_date = st.date_input(
            "終了日", 
            value=datetime.date.today(),
            key="temporal_end_date"
        )
        
        if start_date >= end_date:
            st.error("開始日は終了日より前に設定してください")
            return {}
    
    with col2:
        st.write("**実行設定**")
        execution_delay = st.slider(
            "執行遅延（分）", 
            min_value=1, max_value=60, value=10,
            help="ニュース発表から実際の注文執行までの現実的な遅延時間"
        )
        
        initial_capital = st.number_input(
            "初期資本", 
            min_value=100000, max_value=10000000, 
            value=1000000, step=100000
        )
    
    # 戦略選択
    st.write("**戦略選択**")
    strategy_type = st.selectbox(
        "戦略タイプ",
        ["technical", "ai_news"],
        format_func=lambda x: "テクニカル分析" if x == "technical" else "AI ニュース分析"
    )
    
    strategy_params = {}
    
    if strategy_type == "technical":
        col_tech1, col_tech2 = st.columns(2)
        with col_tech1:
            strategy_params['short_window'] = st.number_input(
                "短期移動平均", min_value=2, max_value=50, value=5
            )
        with col_tech2:
            strategy_params['long_window'] = st.number_input(
                "長期移動平均", min_value=5, max_value=200, value=20
            )
    
    elif strategy_type == "ai_news":
        col_ai1, col_ai2 = st.columns(2)
        with col_ai1:
            strategy_params['model_name'] = st.selectbox(
                "AIモデル",
                ["o3", "o3-mini", "gpt-4o", "gpt-4-turbo"]
            )
        with col_ai2:
            strategy_params['confidence_threshold'] = st.slider(
                "信頼度閾値", 
                min_value=0.1, max_value=1.0, value=0.7, step=0.1
            )
    
    # リスク設定
    with st.expander("リスク管理設定"):
        col_risk1, col_risk2, col_risk3 = st.columns(3)
        
        with col_risk1:
            default_lot_size = st.number_input(
                "ロットサイズ", 
                min_value=0.01, max_value=1.0, value=0.1, step=0.01
            )
        
        with col_risk2:
            default_stop_loss = st.number_input(
                "ストップロス (pips)", 
                min_value=0, max_value=500, value=50
            )
        
        with col_risk3:
            default_take_profit = st.number_input(
                "テイクプロフィット (pips)", 
                min_value=0, max_value=500, value=100
            )
    
    return {
        'start_date': start_date,
        'end_date': end_date,
        'execution_delay': execution_delay,
        'initial_capital': initial_capital,
        'strategy_type': strategy_type,
        'strategy_params': strategy_params,
        'lot_size': default_lot_size,
        'stop_loss': default_stop_loss,
        'take_profit': default_take_profit
    }

def run_temporal_backtest_ui(config: Dict[str, Any], price_data: pd.DataFrame) -> Optional[Any]:
    """UI設定に基づく時系列制約バックテスト実行"""
    
    if st.button("🚀 時系列制約バックテスト実行", type="primary"):
        
        # 期間フィルタリング
        start_datetime = pd.to_datetime(config['start_date'])
        end_datetime = pd.to_datetime(config['end_date']) + pd.Timedelta(days=1)
        
        filtered_price_data = price_data[
            (price_data.index >= start_datetime) & 
            (price_data.index < end_datetime)
        ]
        
        if filtered_price_data.empty:
            st.error("指定期間にデータがありません")
            return None
        
        st.info(f"バックテスト期間: {len(filtered_price_data)}データポイント")
        
        # プログレスバー
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("時系列制約バックテスト実行中...")
            
            # バックテスト実行
            portfolio = run_temporal_backtest(
                price_data=filtered_price_data,
                initial_capital=config['initial_capital'],
                strategy_type=config['strategy_type'],
                strategy_params=config['strategy_params'],
                execution_delay_minutes=config['execution_delay'],
                default_lot_size=config['lot_size'],
                default_stop_loss_pips=config['stop_loss'],
                default_take_profit_pips=config['take_profit']
            )
            
            progress_bar.progress(1.0)
            status_text.text("バックテスト完了!")
            
            return portfolio
            
        except Exception as e:
            st.error(f"バックテスト実行エラー: {e}")
            logger.error(f"Temporal backtest failed: {e}")
            return None
    
    return None

def display_temporal_backtest_results(portfolio: Any):
    """時系列制約バックテスト結果表示"""
    
    if not portfolio:
        return
    
    st.subheader("📊 バックテスト結果")
    
    # 基本パフォーマンス
    total_return = portfolio.calculate_total_return()
    total_trades = len([p for p in portfolio.positions.values() 
                      if p['status'] == 'closed'])
    
    col_perf1, col_perf2, col_perf3 = st.columns(3)
    
    with col_perf1:
        st.metric("総リターン", f"{total_return:.2f}%")
    
    with col_perf2:
        st.metric("総取引数", total_trades)
    
    with col_perf3:
        final_equity = portfolio.equity_curve[-1]['equity'] if portfolio.equity_curve else 0
        st.metric("最終資本", f"¥{final_equity:,.0f}")
    
    # 時系列違反レポート
    if hasattr(portfolio, 'temporal_report'):
        st.subheader("🛡️ 時系列整合性レポート")
        
        violation_count = portfolio.temporal_report.get('total_violations', 0)
        
        if violation_count == 0:
            st.success("✅ 先見バイアス検出なし - 信頼性の高いバックテスト結果")
        else:
            st.warning(f"⚠️ {violation_count}件の時系列違反を検出")
            
            if st.checkbox("違反詳細を表示"):
                violations = portfolio.temporal_report.get('violations', [])
                if violations:
                    violation_df = pd.DataFrame(violations)
                    st.dataframe(violation_df)
    
    # エクイティカーブ
    if portfolio.equity_curve:
        st.subheader("📈 エクイティカーブ")
        
        equity_df = pd.DataFrame(portfolio.equity_curve)
        equity_df['timestamp'] = pd.to_datetime(equity_df['timestamp'])
        equity_df.set_index('timestamp', inplace=True)
        
        st.line_chart(equity_df['equity'])
    
    # 取引履歴
    closed_positions = [p for p in portfolio.positions.values() 
                       if p['status'] == 'closed']
    
    if closed_positions:
        st.subheader("📋 取引履歴")
        
        trades_data = []
        for pos in closed_positions:
            trades_data.append({
                '開始時刻': pos['open_time'],
                '終了時刻': pos['close_time'],
                '方向': pos['direction'],
                'ロット': pos['lot_size'],
                '損益': pos['pnl'],
                '終了理由': pos.get('exit_reason', 'N/A')
            })
        
        trades_df = pd.DataFrame(trades_data)
        st.dataframe(trades_df)
        
        # 勝率統計
        profitable_trades = len([t for t in trades_data if t['損益'] > 0])
        win_rate = (profitable_trades / len(trades_data)) * 100 if trades_data else 0
        
        st.metric("勝率", f"{win_rate:.1f}%")

def create_comparison_ui():
    """従来版 vs 時系列制約版の比較UI"""
    
    st.subheader("⚖️ バックテスト比較")
    st.info("従来版（先見バイアスあり）と時系列制約版の結果比較")
    
    if st.button("比較バックテスト実行"):
        st.warning("実装中: 両方式でのバックテストを並行実行し、差異を分析します")
        
        # TODO: 実装
        # 1. 従来のバックテスト実行
        # 2. 時系列制約バックテスト実行  
        # 3. 結果比較表示
        # 4. パフォーマンス差異の説明