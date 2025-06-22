"""
Strategy Performance Dashboard - Enhanced Streamlit UI
Individual strategy performance tracking and management interface
"""

import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Mock streamlit for systems without it
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    print("Streamlit not available, using mock dashboard")
    STREAMLIT_AVAILABLE = False
    
    # Mock streamlit classes
    class MockStreamlit:
        @staticmethod
        def title(text): print(f"TITLE: {text}")
        @staticmethod
        def header(text): print(f"HEADER: {text}")
        @staticmethod
        def subheader(text): print(f"SUBHEADER: {text}")
        @staticmethod
        def write(text): print(f"WRITE: {text}")
        @staticmethod
        def metric(label, value, delta=None): print(f"METRIC: {label} = {value} (Δ{delta})")
        @staticmethod
        def success(text): print(f"SUCCESS: {text}")
        @staticmethod
        def error(text): print(f"ERROR: {text}")
        @staticmethod
        def warning(text): print(f"WARNING: {text}")
        @staticmethod
        def info(text): print(f"INFO: {text}")
        @staticmethod
        def button(text): return False
        @staticmethod
        def selectbox(label, options): return options[0] if options else None
        @staticmethod
        def multiselect(label, options): return options[:2] if len(options) >= 2 else options
        @staticmethod
        def slider(label, min_val, max_val, value): return value
        @staticmethod
        def checkbox(label, value=False): return value
        @staticmethod
        def radio(label, options): return options[0] if options else None
        @staticmethod
        def plotly_chart(fig, use_container_width=True): print("CHART: Plotly figure displayed")
        @staticmethod
        def json(data): print(f"JSON: {data}")
        @staticmethod
        def dataframe(df): print(f"DATAFRAME: {len(df)} rows")
        @staticmethod
        def table(df): print(f"TABLE: {len(df)} rows")
        @staticmethod
        def columns(n): return [MockStreamlit() for _ in range(n)]
        @staticmethod
        def container(): return MockStreamlit()
        @staticmethod
        def empty(): return MockStreamlit()
        @staticmethod
        def expander(label): return MockStreamlit()
        @staticmethod
        def tabs(labels): return [MockStreamlit() for _ in labels]
        @staticmethod
        def rerun(): pass
        @staticmethod
        def set_page_config(**kwargs): pass
    
    st = MockStreamlit()

# Import performance tracker and related classes
try:
    from ..monitor.strategy_performance import get_performance_tracker, StrategyStats, TradeRecord
    from .dashboard import TradingSystemAPI
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
    from src.project_chimera.monitor.strategy_performance import get_performance_tracker, StrategyStats, TradeRecord
    from src.project_chimera.ui.dashboard import TradingSystemAPI

logger = logging.getLogger(__name__)


class StrategyPerformanceAPI:
    """API client for strategy performance data"""
    
    def __init__(self):
        self.performance_tracker = get_performance_tracker()
    
    def get_all_strategies(self) -> Dict[str, StrategyStats]:
        """Get performance stats for all strategies"""
        try:
            return self.performance_tracker.get_all_strategy_stats()
        except Exception as e:
            logger.warning(f"Failed to get strategy stats: {e}")
            return self._mock_strategy_data()
    
    def get_strategy_details(self, strategy_id: str) -> Optional[StrategyStats]:
        """Get detailed stats for a specific strategy"""
        try:
            return self.performance_tracker.get_strategy_stats(strategy_id)
        except Exception as e:
            logger.warning(f"Failed to get strategy details: {e}")
            return self._mock_single_strategy(strategy_id)
    
    def get_recent_trades(self, strategy_id: str, limit: int = 50) -> List[TradeRecord]:
        """Get recent trades for a strategy"""
        try:
            return self.performance_tracker.get_recent_trades(strategy_id, limit)
        except Exception as e:
            logger.warning(f"Failed to get recent trades: {e}")
            return []
    
    def get_open_positions(self, strategy_id: Optional[str] = None) -> Dict[str, List[TradeRecord]]:
        """Get current open positions"""
        try:
            return self.performance_tracker.get_open_positions(strategy_id)
        except Exception as e:
            logger.warning(f"Failed to get open positions: {e}")
            return {}
    
    def toggle_strategy(self, strategy_id: str, enabled: bool) -> bool:
        """Enable/disable a strategy"""
        # This would integrate with the orchestrator to enable/disable strategies
        try:
            # TODO: Implement actual strategy toggle via orchestrator API
            logger.info(f"Strategy {strategy_id} {'enabled' if enabled else 'disabled'}")
            return True
        except Exception as e:
            logger.error(f"Failed to toggle strategy {strategy_id}: {e}")
            return False
    
    def _mock_strategy_data(self) -> Dict[str, StrategyStats]:
        """Generate mock strategy data"""
        import random
        
        strategies = [
            'weekend_effect', 'stop_reversion', 'funding_contrarian',
            'lob_reversion', 'volatility_breakout', 'cme_gap', 'basis_arbitrage'
        ]
        
        mock_data = {}
        for strategy in strategies:
            mock_data[strategy] = self._mock_single_strategy(strategy)
        
        return mock_data
    
    def _mock_single_strategy(self, strategy_id: str) -> StrategyStats:
        """Generate mock data for a single strategy"""
        import random
        
        return StrategyStats(
            strategy_id=strategy_id,
            total_trades=random.randint(10, 200),
            winning_trades=random.randint(5, 120),
            losing_trades=random.randint(5, 80),
            total_pnl_usd=random.uniform(-2000, 5000),
            total_pnl_pct=random.uniform(-10, 25),
            avg_win_usd=random.uniform(50, 300),
            avg_loss_usd=random.uniform(-200, -30),
            largest_win_usd=random.uniform(300, 800),
            largest_loss_usd=random.uniform(-500, -100),
            win_rate=random.uniform(40, 70),
            profit_factor=random.uniform(0.8, 2.5),
            sharpe_ratio=random.uniform(-0.5, 2.0),
            sortino_ratio=random.uniform(-0.3, 2.5),
            max_drawdown_pct=random.uniform(2, 15),
            calmar_ratio=random.uniform(0.1, 3.0),
            avg_holding_time_hours=random.uniform(0.5, 48),
            avg_slippage_bps=random.uniform(1, 10),
            total_commission_usd=random.uniform(10, 100),
            total_volume_usd=random.uniform(5000, 50000),
            avg_trade_size_usd=random.uniform(100, 1000),
            daily_pnl_std=random.uniform(50, 300),
            last_30d_pnl_usd=random.uniform(-1000, 2000),
            last_7d_pnl_usd=random.uniform(-500, 1000),
            last_24h_pnl_usd=random.uniform(-200, 400),
            first_trade_time=datetime.now() - timedelta(days=random.randint(30, 365)),
            last_trade_time=datetime.now() - timedelta(hours=random.randint(1, 24)),
            last_updated=datetime.now()
        )


def create_strategy_overview_chart(strategies: Dict[str, StrategyStats]) -> go.Figure:
    """Create strategy overview comparison chart"""
    
    if not strategies:
        # Create empty chart
        fig = go.Figure()
        fig.add_annotation(text="No strategy data available", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Extract data for chart
    strategy_names = list(strategies.keys())
    pnl_values = [stats.total_pnl_usd for stats in strategies.values()]
    win_rates = [stats.win_rate for stats in strategies.values()]
    total_trades = [stats.total_trades for stats in strategies.values()]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total P&L by Strategy', 'Win Rate Distribution', 
                       'Trade Count', 'Sharpe Ratio Comparison'),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "bar"}]]
    )
    
    # P&L bar chart
    colors = ['green' if pnl >= 0 else 'red' for pnl in pnl_values]
    fig.add_trace(go.Bar(
        x=strategy_names,
        y=pnl_values,
        name='P&L (USD)',
        marker_color=colors,
        text=[f'${pnl:.0f}' for pnl in pnl_values],
        textposition='auto'
    ), row=1, col=1)
    
    # Win rate scatter
    fig.add_trace(go.Scatter(
        x=strategy_names,
        y=win_rates,
        mode='markers+text',
        name='Win Rate (%)',
        marker=dict(
            size=[trades/10 for trades in total_trades],
            color=win_rates,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Win Rate (%)", x=0.46)
        ),
        text=[f'{wr:.1f}%' for wr in win_rates],
        textposition='top center'
    ), row=1, col=2)
    
    # Trade count
    fig.add_trace(go.Bar(
        x=strategy_names,
        y=total_trades,
        name='Total Trades',
        marker_color='lightblue',
        text=total_trades,
        textposition='auto'
    ), row=2, col=1)
    
    # Sharpe ratio
    sharpe_ratios = [stats.sharpe_ratio for stats in strategies.values()]
    sharpe_colors = ['darkgreen' if sr >= 1 else 'orange' if sr >= 0 else 'red' for sr in sharpe_ratios]
    fig.add_trace(go.Bar(
        x=strategy_names,
        y=sharpe_ratios,
        name='Sharpe Ratio',
        marker_color=sharpe_colors,
        text=[f'{sr:.2f}' for sr in sharpe_ratios],
        textposition='auto'
    ), row=2, col=2)
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=False,
        template='plotly_dark',
        title_text="Strategy Performance Overview",
        title_x=0.5
    )
    
    # Update axes
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(title_text="P&L (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Win Rate (%)", row=1, col=2)
    fig.update_yaxes(title_text="Trade Count", row=2, col=1)
    fig.update_yaxes(title_text="Sharpe Ratio", row=2, col=2)
    
    return fig


def create_strategy_performance_chart(stats: StrategyStats) -> go.Figure:
    """Create detailed performance chart for a single strategy"""
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('P&L Distribution', 'Risk Metrics', 
                       'Time-based Performance', 'Execution Quality'),
        specs=[[{"type": "bar"}, {"type": "indicator"}],
               [{"type": "scatter"}, {"type": "bar"}]]
    )
    
    # P&L distribution (wins vs losses)
    fig.add_trace(go.Bar(
        x=['Winning Trades', 'Losing Trades'],
        y=[stats.winning_trades, stats.losing_trades],
        marker_color=['green', 'red'],
        text=[stats.winning_trades, stats.losing_trades],
        textposition='auto',
        name='Trade Distribution'
    ), row=1, col=1)
    
    # Risk metrics gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=stats.sharpe_ratio,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Sharpe Ratio"},
        gauge={
            'axis': {'range': [-2, 3]},
            'bar': {'color': "darkgreen" if stats.sharpe_ratio >= 1 else "orange" if stats.sharpe_ratio >= 0 else "red"},
            'steps': [
                {'range': [-2, 0], 'color': "lightgray"},
                {'range': [0, 1], 'color': "yellow"},
                {'range': [1, 3], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 1.0
            }
        }
    ), row=1, col=2)
    
    # Time-based performance (mock time series)
    time_points = [datetime.now() - timedelta(days=x) for x in range(30, 0, -1)]
    cumulative_pnl = np.cumsum(np.random.normal(stats.total_pnl_usd/30, stats.daily_pnl_std, 30))
    
    fig.add_trace(go.Scatter(
        x=time_points,
        y=cumulative_pnl,
        mode='lines',
        name='Cumulative P&L',
        line=dict(color='blue', width=2)
    ), row=2, col=1)
    
    # Execution quality metrics
    execution_metrics = ['Avg Slippage', 'Commission Rate', 'Fill Rate']
    execution_values = [
        stats.avg_slippage_bps,
        (stats.total_commission_usd / stats.total_volume_usd * 10000) if stats.total_volume_usd > 0 else 0,
        95.0  # Mock fill rate
    ]
    
    fig.add_trace(go.Bar(
        x=execution_metrics,
        y=execution_values,
        marker_color=['orange', 'blue', 'green'],
        text=[f'{val:.1f}' for val in execution_values],
        textposition='auto',
        name='Execution Metrics'
    ), row=2, col=2)
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=False,
        template='plotly_dark',
        title_text=f"Detailed Performance: {stats.strategy_id}",
        title_x=0.5
    )
    
    return fig


def create_trades_table(trades: List[TradeRecord]) -> pd.DataFrame:
    """Create trades table for display"""
    if not trades:
        return pd.DataFrame()
    
    data = []
    for trade in trades[:50]:  # Show last 50 trades
        data.append({
            'Time': trade.entry_time.strftime('%Y-%m-%d %H:%M'),
            'Symbol': trade.symbol,
            'Side': trade.side.upper(),
            'Size (USD)': f'${trade.size_usd:.2f}',
            'Entry Price': f'${trade.entry_price:.2f}',
            'Exit Price': f'${trade.exit_price:.2f}' if trade.exit_price else 'Open',
            'P&L (USD)': f'${trade.pnl_usd:.2f}',
            'P&L (%)': f'{trade.pnl_pct:.2f}%',
            'Holding Time': f'{trade.holding_time_seconds/3600:.1f}h' if trade.holding_time_seconds > 0 else '-',
            'Confidence': f'{trade.confidence:.2f}',
            'Status': trade.status.value.upper()
        })
    
    return pd.DataFrame(data)


def strategy_dashboard():
    """Main strategy performance dashboard"""
    
    # Page configuration
    st.set_page_config(
        page_title="Strategy Performance Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize API clients
    strategy_api = StrategyPerformanceAPI()
    system_api = TradingSystemAPI()
    
    # Sidebar controls
    st.sidebar.title("🎯 Strategy Controls")
    
    # Strategy selection
    all_strategies = strategy_api.get_all_strategies()
    strategy_names = list(all_strategies.keys())
    
    if strategy_names:
        selected_strategy = st.sidebar.selectbox(
            "Select Strategy",
            strategy_names,
            index=0
        )
    else:
        selected_strategy = None
        st.sidebar.warning("No strategies found")
    
    # Strategy controls
    if selected_strategy:
        st.sidebar.subheader(f"Controls: {selected_strategy}")
        
        # Enable/Disable toggle
        current_enabled = True  # Mock current state
        new_enabled = st.sidebar.checkbox(
            "Strategy Enabled",
            value=current_enabled,
            key=f"enabled_{selected_strategy}"
        )
        
        if new_enabled != current_enabled:
            if strategy_api.toggle_strategy(selected_strategy, new_enabled):
                st.sidebar.success(f"Strategy {'enabled' if new_enabled else 'disabled'}")
            else:
                st.sidebar.error("Failed to update strategy")
        
        # Risk controls
        st.sidebar.subheader("Risk Controls")
        max_position_size = st.sidebar.slider(
            "Max Position Size (%)",
            min_value=0.1,
            max_value=10.0,
            value=2.0,
            step=0.1,
            key=f"position_{selected_strategy}"
        )
        
        confidence_threshold = st.sidebar.slider(
            "Min Confidence",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            key=f"confidence_{selected_strategy}"
        )
    
    # Main dashboard
    st.title("📊 Strategy Performance Dashboard")
    st.markdown("*Individual strategy performance tracking and management*")
    
    # Auto-refresh control
    auto_refresh = st.checkbox("Auto-refresh (5s)", value=True)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔍 Strategy Details", "📋 Trade History", "⚙️ Settings"])
    
    with tab1:
        st.subheader("Strategy Performance Overview")
        
        if all_strategies:
            # Performance summary metrics
            col1, col2, col3, col4, col5 = st.columns(5)
            
            total_pnl = sum(stats.total_pnl_usd for stats in all_strategies.values())
            total_trades = sum(stats.total_trades for stats in all_strategies.values())
            avg_win_rate = np.mean([stats.win_rate for stats in all_strategies.values()])
            best_strategy = max(all_strategies.items(), key=lambda x: x[1].total_pnl_usd)
            active_strategies = len([s for s in all_strategies.values() if s.total_trades > 0])
            
            with col1:
                st.metric("Total P&L", f"${total_pnl:.2f}", 
                         delta=f"${total_pnl*0.1:.2f}")  # Mock delta
            
            with col2:
                st.metric("Total Trades", f"{total_trades:,}")
            
            with col3:
                st.metric("Avg Win Rate", f"{avg_win_rate:.1f}%")
            
            with col4:
                st.metric("Best Strategy", best_strategy[0], 
                         delta=f"${best_strategy[1].total_pnl_usd:.2f}")
            
            with col5:
                st.metric("Active Strategies", f"{active_strategies}/{len(all_strategies)}")
            
            # Strategy overview chart
            overview_chart = create_strategy_overview_chart(all_strategies)
            st.plotly_chart(overview_chart, use_container_width=True)
            
            # Strategy comparison table
            st.subheader("Strategy Comparison")
            
            comparison_data = []
            for strategy_id, stats in all_strategies.items():
                comparison_data.append({
                    'Strategy': strategy_id,
                    'Trades': stats.total_trades,
                    'Win Rate (%)': f"{stats.win_rate:.1f}%",
                    'Total P&L (USD)': f"${stats.total_pnl_usd:.2f}",
                    'Profit Factor': f"{stats.profit_factor:.2f}",
                    'Sharpe Ratio': f"{stats.sharpe_ratio:.2f}",
                    'Max DD (%)': f"{stats.max_drawdown_pct:.1f}%",
                    'Avg Trade (USD)': f"${stats.avg_trade_size_usd:.2f}",
                    'Last 24h (USD)': f"${stats.last_24h_pnl_usd:.2f}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
        
        else:
            st.warning("No strategy performance data available")
    
    with tab2:
        st.subheader("Strategy Details")
        
        if selected_strategy and selected_strategy in all_strategies:
            stats = all_strategies[selected_strategy]
            
            # Detailed metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total P&L", f"${stats.total_pnl_usd:.2f}")
                st.metric("Win Rate", f"{stats.win_rate:.1f}%")
                st.metric("Profit Factor", f"{stats.profit_factor:.2f}")
                st.metric("Total Trades", stats.total_trades)
            
            with col2:
                st.metric("Sharpe Ratio", f"{stats.sharpe_ratio:.2f}")
                st.metric("Sortino Ratio", f"{stats.sortino_ratio:.2f}")
                st.metric("Calmar Ratio", f"{stats.calmar_ratio:.2f}")
                st.metric("Max Drawdown", f"{stats.max_drawdown_pct:.1f}%")
            
            with col3:
                st.metric("Avg Win", f"${stats.avg_win_usd:.2f}")
                st.metric("Avg Loss", f"${stats.avg_loss_usd:.2f}")
                st.metric("Largest Win", f"${stats.largest_win_usd:.2f}")
                st.metric("Largest Loss", f"${stats.largest_loss_usd:.2f}")
            
            # Detailed performance chart
            performance_chart = create_strategy_performance_chart(stats)
            st.plotly_chart(performance_chart, use_container_width=True)
            
            # Recent performance
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Recent Performance")
                recent_data = {
                    'Period': ['Last 24h', 'Last 7d', 'Last 30d'],
                    'P&L (USD)': [
                        f"${stats.last_24h_pnl_usd:.2f}",
                        f"${stats.last_7d_pnl_usd:.2f}",
                        f"${stats.last_30d_pnl_usd:.2f}"
                    ]
                }
                st.table(pd.DataFrame(recent_data))
            
            with col2:
                st.subheader("Execution Metrics")
                execution_data = {
                    'Metric': ['Avg Holding Time', 'Avg Slippage', 'Total Commission', 'Total Volume'],
                    'Value': [
                        f"{stats.avg_holding_time_hours:.1f} hours",
                        f"{stats.avg_slippage_bps:.1f} bps",
                        f"${stats.total_commission_usd:.2f}",
                        f"${stats.total_volume_usd:,.0f}"
                    ]
                }
                st.table(pd.DataFrame(execution_data))
            
            # Open positions
            open_positions = strategy_api.get_open_positions(selected_strategy)
            if selected_strategy in open_positions and open_positions[selected_strategy]:
                st.subheader("Open Positions")
                positions_data = []
                for pos in open_positions[selected_strategy]:
                    # Calculate unrealized P&L (mock)
                    current_price = pos.entry_price * (1 + np.random.uniform(-0.02, 0.02))
                    unrealized_pnl = (current_price - pos.entry_price) * pos.size_native if pos.side == 'buy' else (pos.entry_price - current_price) * pos.size_native
                    
                    positions_data.append({
                        'Symbol': pos.symbol,
                        'Side': pos.side.upper(),
                        'Size (USD)': f'${pos.size_usd:.2f}',
                        'Entry Price': f'${pos.entry_price:.2f}',
                        'Current Price': f'${current_price:.2f}',
                        'Unrealized P&L': f'${unrealized_pnl:.2f}',
                        'Duration': f'{(datetime.now() - pos.entry_time).total_seconds()/3600:.1f}h'
                    })
                
                if positions_data:
                    st.dataframe(pd.DataFrame(positions_data), use_container_width=True)
            else:
                st.info(f"No open positions for {selected_strategy}")
        
        else:
            st.warning("Please select a strategy to view details")
    
    with tab3:
        st.subheader("Trade History")
        
        if selected_strategy:
            # Trade filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                trade_limit = st.selectbox("Show Trades", [20, 50, 100, 200], index=1)
            
            with col2:
                trade_status = st.selectbox("Status Filter", ['All', 'Filled', 'Open', 'Cancelled'])
            
            with col3:
                date_range = st.selectbox("Date Range", ['All Time', 'Last 30 days', 'Last 7 days', 'Last 24 hours'])
            
            # Get trade history
            trades = strategy_api.get_recent_trades(selected_strategy, trade_limit)
            
            if trades:
                trades_df = create_trades_table(trades)
                st.dataframe(trades_df, use_container_width=True)
                
                # Trade statistics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Trade Statistics")
                    if not trades_df.empty:
                        # Extract P&L values for analysis
                        pnl_values = [float(pnl.replace('$', '').replace(',', '')) for pnl in trades_df['P&L (USD)'] if pnl != '$0.00']
                        
                        if pnl_values:
                            st.metric("Avg P&L per Trade", f"${np.mean(pnl_values):.2f}")
                            st.metric("Best Trade", f"${max(pnl_values):.2f}")
                            st.metric("Worst Trade", f"${min(pnl_values):.2f}")
                            st.metric("P&L Std Dev", f"${np.std(pnl_values):.2f}")
                
                with col2:
                    st.subheader("Trade Distribution")
                    if pnl_values:
                        # Create P&L histogram
                        fig = go.Figure(data=[go.Histogram(x=pnl_values, nbinsx=20)])
                        fig.update_layout(
                            title="P&L Distribution",
                            xaxis_title="P&L (USD)",
                            yaxis_title="Count",
                            template='plotly_dark',
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info(f"No trades found for {selected_strategy}")
        
        else:
            st.warning("Please select a strategy to view trade history")
    
    with tab4:
        st.subheader("Strategy Settings")
        
        if selected_strategy:
            st.markdown(f"**Configuration for: {selected_strategy}**")
            
            # Strategy parameters (mock interface)
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Risk Parameters")
                
                max_position = st.number_input(
                    "Max Position Size (%)",
                    min_value=0.1,
                    max_value=10.0,
                    value=2.0,
                    step=0.1,
                    help="Maximum position size as percentage of portfolio"
                )
                
                stop_loss = st.number_input(
                    "Stop Loss (%)",
                    min_value=0.5,
                    max_value=10.0,
                    value=2.0,
                    step=0.1,
                    help="Stop loss threshold"
                )
                
                take_profit = st.number_input(
                    "Take Profit (%)",
                    min_value=0.5,
                    max_value=20.0,
                    value=4.0,
                    step=0.1,
                    help="Take profit threshold"
                )
            
            with col2:
                st.subheader("Strategy Parameters")
                
                confidence_min = st.slider(
                    "Minimum Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.6,
                    step=0.05,
                    help="Minimum signal confidence to trade"
                )
                
                cooldown_hours = st.number_input(
                    "Cooldown Period (hours)",
                    min_value=0.0,
                    max_value=168.0,
                    value=1.0,
                    step=0.5,
                    help="Minimum time between trades"
                )
                
                max_trades_per_day = st.number_input(
                    "Max Trades per Day",
                    min_value=1,
                    max_value=100,
                    value=10,
                    step=1,
                    help="Maximum number of trades per day"
                )
            
            # Update settings button
            if st.button("Update Strategy Settings", type="primary"):
                # Mock settings update
                st.success(f"Settings updated for {selected_strategy}")
                
                # Log the changes
                logger.info(f"Strategy {selected_strategy} settings updated: "
                           f"max_position={max_position}, stop_loss={stop_loss}, "
                           f"take_profit={take_profit}, confidence_min={confidence_min}")
            
            # Export settings
            if st.button("Export Strategy Settings"):
                settings_dict = {
                    'strategy_id': selected_strategy,
                    'max_position_pct': max_position,
                    'stop_loss_pct': stop_loss,
                    'take_profit_pct': take_profit,
                    'min_confidence': confidence_min,
                    'cooldown_hours': cooldown_hours,
                    'max_trades_per_day': max_trades_per_day,
                    'exported_at': datetime.now().isoformat()
                }
                
                st.json(settings_dict)
                st.info("Settings exported as JSON (copy to save)")
        
        else:
            st.warning("Please select a strategy to configure settings")
    
    # Auto-refresh
    if auto_refresh:
        time.sleep(5)
        st.rerun()


def main():
    """Main entry point"""
    if STREAMLIT_AVAILABLE:
        strategy_dashboard()
    else:
        print("Streamlit not available - strategy dashboard functionality limited")


if __name__ == '__main__':
    main()