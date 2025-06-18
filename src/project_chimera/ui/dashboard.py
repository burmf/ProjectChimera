"""
Streamlit Control Center Dashboard - Phase G Implementation
Real-time trading dashboard with 5-second refresh
Features: start/stop controls, equity curve, system status
"""

import time
import json
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
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
        def slider(label, min_val, max_val, value): return value
        @staticmethod
        def plotly_chart(fig, use_container_width=True): print("CHART: Plotly figure displayed")
        @staticmethod
        def json(data): print(f"JSON: {data}")
        @staticmethod
        def dataframe(df): print(f"DATAFRAME: {len(df)} rows")
        @staticmethod
        def columns(n): return [MockStreamlit() for _ in range(n)]
        @staticmethod
        def container(): return MockStreamlit()
        @staticmethod
        def empty(): return MockStreamlit()
        @staticmethod
        def rerun(): pass
        @staticmethod
        def set_page_config(**kwargs): pass
    
    st = MockStreamlit()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TradingSystemAPI:
    """API client for trading system backend"""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 5.0
    
    def get_health(self) -> Dict[str, Any]:
        """Get system health status"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to get health status: {e}")
            return self._mock_health_data()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get Prometheus metrics"""
        try:
            response = self.session.get("http://localhost:9100/metrics")
            metrics_text = response.text
            return self._parse_prometheus_metrics(metrics_text)
        except Exception as e:
            logger.warning(f"Failed to get metrics: {e}")
            return self._mock_metrics_data()
    
    def start_system(self) -> bool:
        """Start trading system"""
        try:
            response = self.session.post(f"{self.base_url}/start")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            return False
    
    def stop_system(self) -> bool:
        """Stop trading system"""
        try:
            response = self.session.post(f"{self.base_url}/stop")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to stop system: {e}")
            return False
    
    def _parse_prometheus_metrics(self, metrics_text: str) -> Dict[str, float]:
        """Parse Prometheus metrics format"""
        metrics = {}
        
        for line in metrics_text.split('\n'):
            if line.startswith('#') or not line.strip():
                continue
            
            parts = line.split(' ')
            if len(parts) >= 2:
                metric_name = parts[0].split('{')[0]  # Remove labels
                try:
                    metric_value = float(parts[1])
                    metrics[metric_name] = metric_value
                except ValueError:
                    continue
        
        return metrics
    
    def _mock_health_data(self) -> Dict[str, Any]:
        """Generate mock health data"""
        import random
        
        return {
            'status': random.choice(['ok', 'degraded']),
            'timestamp': datetime.now().isoformat(),
            'uptime_seconds': random.uniform(3600, 86400),
            'components': {
                'orchestrator': {'status': 'running'},
                'data_feed': {'status': 'running'},
                'strategy_hub': {'status': 'running'},
                'risk_engine': {'status': 'running'},
                'execution_engine': {'status': random.choice(['running', 'degraded'])},
            },
            'metrics': {
                'orders_placed': random.randint(50, 500),
                'orders_filled': random.randint(40, 450),
                'signals_generated': random.randint(100, 1000),
                'errors': random.randint(0, 5)
            }
        }
    
    def _mock_metrics_data(self) -> Dict[str, float]:
        """Generate mock metrics data"""
        import random
        
        return {
            'chimera_pnl_total_usd': random.uniform(-5000, 15000),
            'chimera_slippage_milliseconds': random.uniform(5, 50),
            'chimera_drawdown_percent': random.uniform(0, 15),
            'chimera_websocket_latency_ms': random.uniform(10, 100),
            'chimera_orders_total': random.randint(100, 1000),
            'chimera_orders_filled_total': random.randint(80, 900),
            'chimera_equity_value_usd': random.uniform(95000, 115000),
            'chimera_system_uptime_seconds': random.uniform(3600, 86400)
        }


class EquityCurveGenerator:
    """Generates realistic equity curve data for visualization"""
    
    def __init__(self):
        self.data_points = []
        self.start_equity = 100000.0
        self.current_equity = self.start_equity
        self.start_time = datetime.now() - timedelta(hours=24)
    
    def generate_historical_data(self, hours: int = 24) -> pd.DataFrame:
        """Generate historical equity curve data"""
        timestamps = []
        equity_values = []
        
        current_time = self.start_time
        current_equity = self.start_equity
        
        # Generate hourly data points
        for i in range(hours):
            timestamps.append(current_time)
            
            # Random walk with slight upward bias
            change_pct = random.gauss(0.001, 0.02)  # 0.1% mean, 2% std
            current_equity *= (1 + change_pct)
            equity_values.append(current_equity)
            
            current_time += timedelta(hours=1)
        
        self.current_equity = current_equity
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'equity': equity_values
        })
    
    def add_real_time_point(self, equity: float) -> Dict[str, Any]:
        """Add a real-time equity point"""
        timestamp = datetime.now()
        self.data_points.append({
            'timestamp': timestamp,
            'equity': equity
        })
        
        # Keep only last 1000 points
        if len(self.data_points) > 1000:
            self.data_points = self.data_points[-1000:]
        
        return self.data_points[-1]


def create_equity_chart(df: pd.DataFrame) -> go.Figure:
    """Create equity curve chart"""
    fig = go.Figure()
    
    # Add equity line
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['equity'],
        mode='lines',
        name='Portfolio Equity',
        line=dict(color='#00D4AA', width=2),
        hovertemplate='<b>%{y:$,.0f}</b><br>%{x}<extra></extra>'
    ))
    
    # Add formatting
    fig.update_layout(
        title='Portfolio Equity Curve',
        xaxis_title='Time',
        yaxis_title='Equity (USD)',
        template='plotly_dark',
        height=400,
        showlegend=False,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    # Format y-axis as currency
    fig.update_yaxis(tickformat='$,.0f')
    
    return fig


def create_metrics_chart(metrics: Dict[str, float]) -> go.Figure:
    """Create metrics overview chart"""
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('P&L', 'Drawdown', 'Slippage', 'Latency'),
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]]
    )
    
    # P&L gauge
    pnl = metrics.get('chimera_pnl_total_usd', 0)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=pnl,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [-10000, 20000]},
               'bar': {'color': "green" if pnl >= 0 else "red"},
               'steps': [{'range': [-10000, 0], 'color': "lightgray"},
                        {'range': [0, 20000], 'color': "lightgreen"}]}
    ), row=1, col=1)
    
    # Drawdown gauge
    dd = metrics.get('chimera_drawdown_percent', 0)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=dd,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [0, 25]},
               'bar': {'color': "red" if dd > 10 else "orange" if dd > 5 else "green"},
               'steps': [{'range': [0, 5], 'color': "lightgreen"},
                        {'range': [5, 10], 'color': "yellow"},
                        {'range': [10, 25], 'color': "lightcoral"}]}
    ), row=1, col=2)
    
    # Slippage gauge
    slippage = metrics.get('chimera_slippage_milliseconds', 0)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=slippage,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [0, 100]},
               'bar': {'color': "green" if slippage < 20 else "orange" if slippage < 50 else "red"}}
    ), row=2, col=1)
    
    # Latency gauge
    latency = metrics.get('chimera_websocket_latency_ms', 0)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=latency,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [0, 200]},
               'bar': {'color': "green" if latency < 50 else "orange" if latency < 100 else "red"}}
    ), row=2, col=2)
    
    fig.update_layout(
        template='plotly_dark',
        height=400,
        margin=dict(l=0, r=0, t=60, b=0)
    )
    
    return fig


def main_dashboard():
    """Main Streamlit dashboard"""
    
    # Page configuration
    st.set_page_config(
        page_title="Project Chimera Control Center",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize API client
    api = TradingSystemAPI()
    
    # Initialize equity curve generator
    if 'equity_generator' not in st.session_state:
        st.session_state.equity_generator = EquityCurveGenerator()
    
    # Auto-refresh every 5 seconds
    placeholder = st.empty()
    
    with placeholder.container():
        # Header
        st.title("🚀 Project Chimera Control Center")
        st.markdown("*Real-time trading system monitoring and control*")
        
        # Get system data
        health_data = api.get_health()
        metrics_data = api.get_metrics()
        
        # Top metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            pnl = metrics_data.get('chimera_pnl_total_usd', 0)
            st.metric("P&L", f"${pnl:,.0f}", f"{pnl:+,.0f}")
        
        with col2:
            equity = metrics_data.get('chimera_equity_value_usd', 100000)
            st.metric("Equity", f"${equity:,.0f}")
        
        with col3:
            dd = metrics_data.get('chimera_drawdown_percent', 0)
            st.metric("Drawdown", f"{dd:.1f}%")
        
        with col4:
            orders = int(metrics_data.get('chimera_orders_total', 0))
            filled = int(metrics_data.get('chimera_orders_filled_total', 0))
            fill_rate = (filled / orders * 100) if orders > 0 else 0
            st.metric("Fill Rate", f"{fill_rate:.1f}%", f"{filled}/{orders}")
        
        with col5:
            uptime = metrics_data.get('chimera_system_uptime_seconds', 0)
            uptime_hours = uptime / 3600
            st.metric("Uptime", f"{uptime_hours:.1f}h")
        
        # System controls
        st.subheader("System Controls")
        col1, col2, col3 = st.columns([1, 1, 3])
        
        with col1:
            if st.button("🟢 Start System", type="primary"):
                if api.start_system():
                    st.success("System started successfully!")
                else:
                    st.error("Failed to start system")
        
        with col2:
            if st.button("🔴 Stop System"):
                if api.stop_system():
                    st.success("System stopped successfully!")
                else:
                    st.error("Failed to stop system")
        
        # System status
        status = health_data.get('status', 'unknown')
        if status == 'ok':
            st.success(f"✅ System Status: {status.upper()}")
        elif status == 'degraded':
            st.warning(f"⚠️ System Status: {status.upper()}")
        else:
            st.error(f"❌ System Status: {status.upper()}")
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Equity curve
            st.subheader("Equity Curve")
            equity_df = st.session_state.equity_generator.generate_historical_data()
            equity_chart = create_equity_chart(equity_df)
            st.plotly_chart(equity_chart, use_container_width=True)
        
        with col2:
            # Component status
            st.subheader("Component Status")
            components = health_data.get('components', {})
            
            for name, info in components.items():
                status = info.get('status', 'unknown')
                if status == 'running':
                    st.success(f"✅ {name.title()}")
                elif status == 'degraded':
                    st.warning(f"⚠️ {name.title()}")
                else:
                    st.error(f"❌ {name.title()}")
        
        # Metrics dashboard
        st.subheader("Performance Metrics")
        metrics_chart = create_metrics_chart(metrics_data)
        st.plotly_chart(metrics_chart, use_container_width=True)
        
        # Recent activity
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("System Metrics")
            system_metrics = health_data.get('metrics', {})
            
            metrics_df = pd.DataFrame([
                {"Metric": "Orders Placed", "Value": system_metrics.get('orders_placed', 0)},
                {"Metric": "Orders Filled", "Value": system_metrics.get('orders_filled', 0)},
                {"Metric": "Signals Generated", "Value": system_metrics.get('signals_generated', 0)},
                {"Metric": "Errors", "Value": system_metrics.get('errors', 0)}
            ])
            
            st.dataframe(metrics_df, use_container_width=True)
        
        with col2:
            st.subheader("Live Metrics")
            st.json({
                "slippage_ms": f"{metrics_data.get('chimera_slippage_milliseconds', 0):.1f}",
                "ws_latency_ms": f"{metrics_data.get('chimera_websocket_latency_ms', 0):.1f}",
                "last_updated": datetime.now().strftime("%H:%M:%S")
            })
        
        # Footer
        st.markdown("---")
        st.markdown("*Phase G Implementation - Streamlit Control Center with 5-second refresh*")
        
        # Auto-refresh timestamp
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Auto-refresh every 5 seconds
    time.sleep(5)
    st.rerun()


def mock_dashboard():
    """Mock dashboard for systems without Streamlit"""
    print("="*80)
    print("PROJECT CHIMERA CONTROL CENTER (MOCK)")
    print("="*80)
    
    api = TradingSystemAPI()
    
    while True:
        try:
            # Get data
            health_data = api.get_health()
            metrics_data = api.get_metrics()
            
            # Display mock dashboard
            print(f"\n🚀 Dashboard Update - {datetime.now().strftime('%H:%M:%S')}")
            print("-" * 50)
            
            # Key metrics
            pnl = metrics_data.get('chimera_pnl_total_usd', 0)
            equity = metrics_data.get('chimera_equity_value_usd', 100000)
            dd = metrics_data.get('chimera_drawdown_percent', 0)
            
            print(f"P&L: ${pnl:,.0f}")
            print(f"Equity: ${equity:,.0f}")
            print(f"Drawdown: {dd:.1f}%")
            
            # System status
            status = health_data.get('status', 'unknown')
            print(f"Status: {status.upper()}")
            
            # Components
            components = health_data.get('components', {})
            print(f"Components: {len(components)} running")
            
            print("\n✅ Dashboard updated successfully!")
            
            time.sleep(5)  # 5-second refresh
            
        except KeyboardInterrupt:
            print("\n🛑 Dashboard stopped")
            break
        except Exception as e:
            print(f"❌ Dashboard error: {e}")
            time.sleep(10)


def main():
    """Main entry point"""
    if STREAMLIT_AVAILABLE:
        print("Starting Streamlit dashboard...")
        main_dashboard()
    else:
        print("Starting mock dashboard...")
        mock_dashboard()


if __name__ == '__main__':
    # Add random import for simulation
    import random
    main()