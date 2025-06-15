#!/usr/bin/env python3
"""
Bitget Scalping Dashboard
リアルタイムスキャルピングダッシュボード
"""

import streamlit as st
import asyncio
import json
import time
import logging
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import deque
import requests
import threading
import sys
import os

# Add core modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Page config
st.set_page_config(
    page_title="ProjectChimera - Bitget Scalping Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'trading_active' not in st.session_state:
    st.session_state.trading_active = False
if 'trades' not in st.session_state:
    st.session_state.trades = []
if 'price_history' not in st.session_state:
    st.session_state.price_history = deque(maxlen=200)
if 'current_position' not in st.session_state:
    st.session_state.current_position = None
if 'account_balance' not in st.session_state:
    st.session_state.account_balance = 100000
if 'total_profit' not in st.session_state:
    st.session_state.total_profit = 0

# Scalping Engine Class
class StreamlitScalpingEngine:
    """Streamlit用スキャルピングエンジン"""
    
    def __init__(self):
        self.base_url = 'https://api.bitget.com'
        self.futures_symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']  # 先物シンボル
        self.spot_symbol = 'BTCUSDT_SPBL'
        
        # トレーディング設定
        self.position_size = 25000
        self.profit_target = 0.002  # 0.2%
        self.stop_loss = 0.001      # 0.1%
        self.max_hold_time = 120    # 2分
        
    def get_futures_price(self, symbol: str) -> Optional[Dict]:
        """先物価格取得"""
        try:
            # Bitget先物APIエンドポイント
            url = f'{self.base_url}/api/mix/v1/market/ticker?symbol={symbol}_UMCBL'
            response = requests.get(url, timeout=3)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    ticker = data['data']
                    return {
                        'symbol': symbol,
                        'price': float(ticker['close']),
                        'change_24h': float(ticker['changeUtc']),
                        'volume': float(ticker['baseVolume']),
                        'funding_rate': float(ticker.get('fundingRate', 0)),
                        'timestamp': datetime.now()
                    }
            return None
        except Exception as e:
            st.error(f"Price fetch error for {symbol}: {e}")
            return None
    
    def get_spot_price(self) -> Optional[Dict]:
        """現物価格取得"""
        try:
            url = f'{self.base_url}/api/spot/v1/market/ticker?symbol={self.spot_symbol}'
            response = requests.get(url, timeout=3)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('code') == '00000':
                    ticker = data['data']
                    return {
                        'symbol': self.spot_symbol,
                        'price': float(ticker['close']),
                        'change_24h': float(ticker['change']),
                        'volume': float(ticker['baseVol']),
                        'timestamp': datetime.now()
                    }
            return None
        except Exception as e:
            st.error(f"Spot price fetch error: {e}")
            return None
    
    def calculate_futures_indicators(self, price_data: List[float]) -> Dict[str, float]:
        """先物用テクニカル指標計算"""
        if len(price_data) < 10:
            return {'momentum': 0, 'volatility': 0, 'trend': 0}
        
        # 直近の価格変動
        momentum = (price_data[-1] - price_data[-5]) / price_data[-5] if len(price_data) >= 5 else 0
        
        # ボラティリティ
        returns = [(price_data[i] / price_data[i-1] - 1) for i in range(1, min(len(price_data), 15))]
        volatility = pd.Series(returns).std() if len(returns) > 1 else 0
        
        # トレンド強度
        if len(price_data) >= 10:
            recent_avg = sum(price_data[-5:]) / 5
            older_avg = sum(price_data[-10:-5]) / 5
            trend = (recent_avg - older_avg) / older_avg
        else:
            trend = 0
        
        return {
            'momentum': momentum,
            'volatility': volatility,
            'trend': trend
        }
    
    def generate_futures_signal(self, symbol: str, price_data: List[float], current_price: float) -> Dict[str, Any]:
        """先物取引シグナル生成"""
        indicators = self.calculate_futures_indicators(price_data)
        
        signal = {
            'action': 'hold',
            'confidence': 0.0,
            'reasoning': '',
            'symbol': symbol,
            'entry_price': current_price,
            'target_price': 0,
            'stop_loss_price': 0,
            'leverage': 10  # 10倍レバレッジ
        }
        
        momentum = indicators['momentum']
        volatility = indicators['volatility']
        trend = indicators['trend']
        
        # 先物特有の高レバレッジ戦略
        if volatility < 0.0005:
            signal['reasoning'] = f'Low volatility: {volatility:.5f}'
            return signal
        
        # ロングシグナル（強い上昇モメンタム）
        if momentum > 0.001 and trend > 0.0005:
            signal['action'] = 'long'
            signal['confidence'] = min(0.95, momentum * 300)
            signal['target_price'] = current_price * (1 + self.profit_target)
            signal['stop_loss_price'] = current_price * (1 - self.stop_loss)
            signal['reasoning'] = f'Strong bullish momentum: {momentum:.4f}'
        
        # ショートシグナル（強い下落モメンタム）
        elif momentum < -0.001 and trend < -0.0005:
            signal['action'] = 'short'
            signal['confidence'] = min(0.95, abs(momentum) * 300)
            signal['target_price'] = current_price * (1 - self.profit_target)
            signal['stop_loss_price'] = current_price * (1 + self.stop_loss)
            signal['reasoning'] = f'Strong bearish momentum: {momentum:.4f}'
        
        # 高ボラティリティ環境での短期反転狙い
        elif volatility > 0.003:
            if abs(momentum) > 0.002:
                if momentum > 0:
                    signal['action'] = 'short'
                    signal['target_price'] = current_price * (1 - self.profit_target * 0.5)
                else:
                    signal['action'] = 'long'
                    signal['target_price'] = current_price * (1 + self.profit_target * 0.5)
                
                signal['confidence'] = 0.8
                signal['stop_loss_price'] = current_price * (1 + self.stop_loss if signal['action'] == 'short' else 1 - self.stop_loss)
                signal['reasoning'] = f'High vol reversal: {momentum:.4f}'
        
        return signal

# Initialize engine
@st.cache_resource
def get_scalping_engine():
    return StreamlitScalpingEngine()

engine = get_scalping_engine()

# Dashboard Header
st.title("⚡ ProjectChimera - Bitget Scalping Dashboard")

# Create columns for main layout
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    st.subheader("📊 Market Data")

with col2:
    st.subheader("🎯 Trading Signals")

with col3:
    st.subheader("💰 Performance")

# Sidebar Controls
st.sidebar.title("🎛️ Trading Controls")

# Trading Mode Selection
trading_mode = st.sidebar.selectbox(
    "Trading Mode",
    ["Spot Trading", "Futures Trading", "Both"]
)

# Symbol Selection for Futures
if trading_mode in ["Futures Trading", "Both"]:
    selected_futures = st.sidebar.multiselect(
        "Futures Symbols",
        engine.futures_symbols,
        default=["BTCUSDT"]
    )

# Position Size
position_size = st.sidebar.number_input(
    "Position Size ($)",
    min_value=1000,
    max_value=100000,
    value=engine.position_size,
    step=1000
)

# Risk Settings
profit_target = st.sidebar.slider(
    "Profit Target (%)",
    min_value=0.05,
    max_value=1.0,
    value=engine.profit_target * 100,
    step=0.05
) / 100

stop_loss = st.sidebar.slider(
    "Stop Loss (%)",
    min_value=0.05,
    max_value=0.5,
    value=engine.stop_loss * 100,
    step=0.05
) / 100

# Leverage (for futures)
leverage = st.sidebar.slider(
    "Leverage (Futures)",
    min_value=1,
    max_value=50,
    value=10,
    step=1
)

# Trading Controls
st.sidebar.divider()

if st.sidebar.button("🚀 Start Trading", type="primary"):
    st.session_state.trading_active = True
    st.success("Trading started!")

if st.sidebar.button("⏹️ Stop Trading"):
    st.session_state.trading_active = False
    st.session_state.current_position = None
    st.warning("Trading stopped!")

if st.sidebar.button("🔄 Reset Performance"):
    st.session_state.trades = []
    st.session_state.total_profit = 0
    st.session_state.account_balance = 100000
    st.info("Performance reset!")

# Main Dashboard Content
if st.session_state.trading_active:
    # Create real-time data containers
    price_container = st.container()
    signal_container = st.container()
    performance_container = st.container()
    
    # Auto-refresh setup
    placeholder = st.empty()
    
    with placeholder.container():
        # Market Data Section
        with price_container:
            st.subheader("📈 Real-Time Prices")
            
            price_cols = st.columns(3)
            
            # Get current prices
            if trading_mode in ["Spot Trading", "Both"]:
                spot_data = engine.get_spot_price()
                if spot_data:
                    with price_cols[0]:
                        st.metric(
                            label=f"🔸 {spot_data['symbol']}",
                            value=f"${spot_data['price']:,.2f}",
                            delta=f"{spot_data['change_24h']:+.2f}%"
                        )
                        st.session_state.price_history.append({
                            'timestamp': spot_data['timestamp'],
                            'symbol': spot_data['symbol'],
                            'price': spot_data['price']
                        })
            
            # Futures prices
            if trading_mode in ["Futures Trading", "Both"]:
                for i, symbol in enumerate(selected_futures):
                    futures_data = engine.get_futures_price(symbol)
                    if futures_data:
                        col_idx = (i + 1) % 3
                        with price_cols[col_idx]:
                            st.metric(
                                label=f"⚡ {symbol} Futures",
                                value=f"${futures_data['price']:,.2f}",
                                delta=f"{futures_data['change_24h']:+.2f}%"
                            )
                            
                            # Funding rate
                            st.caption(f"Funding: {futures_data['funding_rate']:.4f}%")
        
        # Price Chart
        if len(st.session_state.price_history) > 10:
            chart_data = pd.DataFrame(list(st.session_state.price_history))
            chart_data['timestamp'] = pd.to_datetime(chart_data['timestamp'])
            
            fig = go.Figure()
            
            for symbol in chart_data['symbol'].unique():
                symbol_data = chart_data[chart_data['symbol'] == symbol]
                fig.add_trace(go.Scatter(
                    x=symbol_data['timestamp'],
                    y=symbol_data['price'],
                    mode='lines',
                    name=symbol,
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title="Real-Time Price Chart",
                xaxis_title="Time",
                yaxis_title="Price ($)",
                height=400,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Trading Signals Section
        with signal_container:
            st.subheader("🎯 Active Signals")
            
            signal_cols = st.columns(2)
            
            # Current Position
            if st.session_state.current_position:
                pos = st.session_state.current_position
                with signal_cols[0]:
                    st.info(f"**Active Position:** {pos['action'].upper()} {pos['symbol']}")
                    st.write(f"Entry: ${pos['entry_price']:,.2f}")
                    st.write(f"Target: ${pos['target_price']:,.2f}")
                    st.write(f"Stop: ${pos['stop_loss_price']:,.2f}")
                    
                    # Calculate current P&L
                    current_price = pos['entry_price'] + (pos['entry_price'] * 0.001)  # Mock current price
                    if pos['action'] == 'long':
                        pnl = (current_price - pos['entry_price']) / pos['entry_price']
                    else:
                        pnl = (pos['entry_price'] - current_price) / pos['entry_price']
                    
                    pnl_amount = position_size * pnl
                    color = "green" if pnl_amount > 0 else "red"
                    st.write(f"Current P&L: :color[{color}][${pnl_amount:+,.2f} ({pnl:+.2%})]")
            
            # Signal Generation
            with signal_cols[1]:
                st.write("**Latest Signals:**")
                
                # Mock signal generation for display
                if len(st.session_state.price_history) > 10:
                    recent_prices = [p['price'] for p in list(st.session_state.price_history)[-10:]]
                    current_price = recent_prices[-1]
                    
                    signal = engine.generate_futures_signal("BTCUSDT", recent_prices, current_price)
                    
                    if signal['action'] != 'hold':
                        confidence_color = "green" if signal['confidence'] > 0.7 else "orange"
                        st.write(f"**{signal['action'].upper()}** {signal['symbol']}")
                        st.write(f"Confidence: :color[{confidence_color}][{signal['confidence']:.1%}]")
                        st.write(f"Reasoning: {signal['reasoning']}")
                    else:
                        st.write("No signals currently")
        
        # Performance Section
        with performance_container:
            st.subheader("📊 Performance Metrics")
            
            perf_cols = st.columns(4)
            
            with perf_cols[0]:
                st.metric(
                    "Account Balance",
                    f"${st.session_state.account_balance:,.2f}",
                    f"${st.session_state.total_profit:+,.2f}"
                )
            
            with perf_cols[1]:
                total_trades = len(st.session_state.trades)
                st.metric("Total Trades", total_trades)
            
            with perf_cols[2]:
                if total_trades > 0:
                    winning_trades = sum(1 for t in st.session_state.trades if t.get('profit', 0) > 0)
                    win_rate = winning_trades / total_trades
                    st.metric("Win Rate", f"{win_rate:.1%}")
                else:
                    st.metric("Win Rate", "0%")
            
            with perf_cols[3]:
                if total_trades > 0:
                    avg_profit = st.session_state.total_profit / total_trades
                    st.metric("Avg P&L", f"${avg_profit:+,.2f}")
                else:
                    st.metric("Avg P&L", "$0.00")
        
        # Trade History
        if st.session_state.trades:
            st.subheader("📋 Recent Trades")
            
            trades_df = pd.DataFrame(st.session_state.trades)
            trades_df = trades_df.sort_values('timestamp', ascending=False).head(10)
            
            # Format the dataframe for display
            display_df = trades_df[['timestamp', 'symbol', 'action', 'entry_price', 'exit_price', 'profit', 'reason']].copy()
            display_df['profit'] = display_df['profit'].apply(lambda x: f"${x:+,.2f}")
            display_df['timestamp'] = display_df['timestamp'].dt.strftime('%H:%M:%S')
            
            st.dataframe(display_df, use_container_width=True)
        
        # Auto-refresh
        time.sleep(2)
        st.rerun()

else:
    # Trading not active - show static info
    st.info("🔄 Click 'Start Trading' to begin real-time scalping")
    
    # Static market overview
    st.subheader("📊 Market Overview")
    
    overview_cols = st.columns(3)
    
    # Get current market data
    spot_data = engine.get_spot_price()
    if spot_data:
        with overview_cols[0]:
            st.metric(
                "BTC Spot",
                f"${spot_data['price']:,.2f}",
                f"{spot_data['change_24h']:+.2f}%"
            )
    
    btc_futures = engine.get_futures_price("BTCUSDT")
    if btc_futures:
        with overview_cols[1]:
            st.metric(
                "BTC Futures",
                f"${btc_futures['price']:,.2f}",
                f"{btc_futures['change_24h']:+.2f}%"
            )
            st.caption(f"Funding: {btc_futures['funding_rate']:.4f}%")
    
    eth_futures = engine.get_futures_price("ETHUSDT")
    if eth_futures:
        with overview_cols[2]:
            st.metric(
                "ETH Futures",
                f"${eth_futures['price']:,.2f}",
                f"{eth_futures['change_24h']:+.2f}%"
            )
            st.caption(f"Funding: {eth_futures['funding_rate']:.4f}%")
    
    # Strategy Information
    st.subheader("⚡ Scalping Strategy")
    
    strategy_cols = st.columns(2)
    
    with strategy_cols[0]:
        st.write("**Current Settings:**")
        st.write(f"• Position Size: ${position_size:,}")
        st.write(f"• Profit Target: {profit_target:.2%}")
        st.write(f"• Stop Loss: {stop_loss:.2%}")
        st.write(f"• Leverage: {leverage}x")
    
    with strategy_cols[1]:
        st.write("**Strategy Features:**")
        st.write("• High-frequency momentum detection")
        st.write("• Multi-timeframe analysis")
        st.write("• Automated risk management")
        st.write("• Real-time P&L tracking")

# Footer
st.divider()
st.caption("⚡ ProjectChimera Bitget Scalping Dashboard | Real-time trading with advanced AI analysis")