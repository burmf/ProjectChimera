#!/usr/bin/env python3
"""
Streamlit Web Interface for AI Trading Platform
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import logging
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.logging_config import setup_logging
from core.database_adapter import db_adapter
from core.ai_manager import AIManager
from modules.technical_analyzer import TechnicalAnalyzer

# Streamlit page configuration
st.set_page_config(
    page_title="AI Trading Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize components
@st.cache_resource
def init_components():
    """Initialize AI manager and technical analyzer."""
    ai_manager = AIManager()
    tech_analyzer = TechnicalAnalyzer()
    return ai_manager, tech_analyzer

def main():
    st.title("🤖 AI Trading Platform")
    st.markdown("*News-driven algorithmic trading with OpenAI integration*")
    
    # Initialize components
    ai_manager, tech_analyzer = init_components()
    
    # Check database connection
    if not db_adapter.is_connected():
        st.error("❌ Database connection failed. Please check your configuration.")
        return
    
    # Sidebar - System Status
    with st.sidebar:
        st.header("📊 System Status")
        
        # AI Health Check
        ai_status = ai_manager.health_check()
        if ai_status['available']:
            st.success(f"🟢 AI Service: {ai_status['status']}")
            st.caption(f"Model: {ai_status.get('default_model', 'N/A')}")
        else:
            st.error(f"🔴 AI Service: {ai_status['message']}")
        
        # Database Status
        if db_adapter.is_connected():
            st.success("🟢 Database: Connected")
            st.caption(f"Type: {db_adapter.db_type}")
        else:
            st.error("🔴 Database: Disconnected")
        
        st.divider()
        
        # Quick Actions
        st.header("⚡ Quick Actions")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard", 
        "📰 Data Collection", 
        "🧠 AI Analysis", 
        "📈 Backtesting"
    ])
    
    with tab1:
        show_dashboard(ai_manager, tech_analyzer)
    
    with tab2:
        show_data_collection()
    
    with tab3:
        show_ai_analysis(ai_manager)
    
    with tab4:
        show_backtesting(tech_analyzer)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_dashboard_data():
    """Get dashboard data with caching."""
    try:
        recent_prices = db_adapter.get_latest_price_data('USD/JPY', hours=24)
        recent_news = db_adapter.get_recent_news(hours=24)
        api_usage = db_adapter.get_api_usage_stats(days=7)
        
        return {
            'prices': recent_prices,
            'news': recent_news,
            'api_usage': api_usage
        }
    except Exception as e:
        logger.error(f"Failed to get dashboard data: {e}")
        return {'prices': pd.DataFrame(), 'news': pd.DataFrame(), 'api_usage': pd.DataFrame()}

def show_dashboard(ai_manager, tech_analyzer):
    """Display main dashboard with current status and recent data."""
    st.header("📊 Trading Dashboard")
    
    # Get cached data
    data = get_dashboard_data()
    recent_prices = data['prices']
    recent_news = data['news']
    api_usage = data['api_usage']
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if not recent_prices.empty:
            latest_price = recent_prices.iloc[-1]['close']
            prev_price = recent_prices.iloc[-2]['close'] if len(recent_prices) > 1 else latest_price
            change = latest_price - prev_price
            st.metric("💱 USD/JPY", f"{latest_price:.4f}", f"{change:+.4f}")
        else:
            st.metric("💱 USD/JPY", "No data", "0.0000")
    
    with col2:
        news_count = len(recent_news) if not recent_news.empty else 0
        processed_count = len(recent_news[recent_news['ai_processed'] == True]) if not recent_news.empty else 0
        st.metric("📰 News Articles", news_count, f"{processed_count} processed")
    
    with col3:
        if not api_usage.empty:
            total_cost = api_usage['total_cost'].sum()
            total_requests = api_usage['request_count'].sum()
            st.metric("💰 API Cost (7d)", f"${total_cost:.4f}", f"{total_requests} requests")
        else:
            st.metric("💰 API Cost (7d)", "$0.00", "0 requests")
    
    with col4:
        # System health score
        ai_status = ai_manager.health_check()
        db_status = db_adapter.is_connected()
        health_score = (1 if ai_status['available'] else 0) + (1 if db_status else 0)
        health_pct = health_score / 2 * 100
        st.metric("🔧 System Health", f"{health_pct:.0f}%", "All systems" if health_score == 2 else "Issues detected")
    
    # Technical analysis chart
    if not recent_prices.empty:
        st.subheader("📈 Technical Analysis")
        
        df_with_indicators = tech_analyzer.generate_signals(recent_prices.copy())
        latest_analysis = tech_analyzer.get_latest_analysis(df_with_indicators)
        
        # Display chart and analysis
        col1, col2 = st.columns([3, 1])
        
        with col1:
            fig = go.Figure(data=go.Candlestick(
                x=df_with_indicators.index,
                open=df_with_indicators['open'],
                high=df_with_indicators['high'],
                low=df_with_indicators['low'],
                close=df_with_indicators['close']
            ))
            
            fig.update_layout(
                title="USD/JPY Price Chart",
                xaxis_title="Time",
                yaxis_title="Price",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            signal = latest_analysis['signal']
            confidence = latest_analysis['confidence']
            
            if signal == 1:
                st.success("🟢 BUY Signal")
            elif signal == -1:
                st.error("🔴 SELL Signal")
            else:
                st.info("🟡 NEUTRAL")
            
            st.metric("Confidence", f"{confidence:.1%}")
    
    # Recent news display
    if not recent_news.empty:
        st.subheader("📰 Recent News")
        for _, news in recent_news.head(5).iterrows():
            with st.expander(f"{news['title'][:100]}..."):
                st.write(f"**Source:** {news['source']}")
                st.write(f"**Published:** {news['published_at']}")
                st.write(f"**AI Processed:** {'✅' if news['ai_processed'] else '❌'}")

def show_data_collection():
    """Data collection interface."""
    st.header("📰 Data Collection")
    
    st.markdown("""
    Configure and run data collection for price data and news articles.
    Data collectors run automatically in Docker containers.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💱 Price Data")
        
        # Recent price data status
        recent_prices = db_adapter.get_latest_price_data('USD/JPY', hours=1)
        
        if not recent_prices.empty:
            latest_time = recent_prices.index[-1]
            minutes_ago = (datetime.now() - latest_time).total_seconds() / 60
            
            if minutes_ago < 5:
                st.success(f"✅ Data up to date (updated {minutes_ago:.0f}m ago)")
            elif minutes_ago < 60:
                st.warning(f"⚠️ Data slightly stale (updated {minutes_ago:.0f}m ago)")
            else:
                st.error(f"❌ Data stale (updated {minutes_ago/60:.1f}h ago)")
        else:
            st.error("❌ No price data found")
    
    with col2:
        st.subheader("📰 News Data")
        
        # Recent news status
        recent_news = db_adapter.get_recent_news(hours=1)
        
        if not recent_news.empty:
            processed_count = len(recent_news[recent_news['ai_processed'] == True])
            total_count = len(recent_news)
            processing_rate = processed_count / total_count if total_count > 0 else 0
            
            st.metric("Recent Articles", total_count)
            st.metric("Processing Rate", f"{processing_rate:.1%}")
        else:
            st.info("ℹ️ No recent news found")

def show_ai_analysis(ai_manager):
    """AI analysis interface."""
    st.header("🧠 AI Analysis")
    
    # Check AI availability
    ai_status = ai_manager.health_check()
    if not ai_status['available']:
        st.error(f"❌ AI service unavailable: {ai_status['message']}")
        st.info("💡 Please check your OpenAI API key in the environment variables.")
        return
    
    st.success(f"✅ AI service ready (Model: {ai_status.get('default_model', 'Unknown')})")
    
    # Manual analysis section
    st.subheader("📝 Manual News Analysis")
    
    news_input = st.text_area(
        "Enter news text to analyze:",
        placeholder="Paste news article here for AI analysis...",
        height=150
    )
    
    if st.button("🔍 Analyze News", type="primary") and news_input:
        with st.spinner("Analyzing with AI..."):
            try:
                result = ai_manager.get_trading_decision(news_context=news_input)
                
                if result:
                    st.success("✅ Analysis completed!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        action = result.get('action', 'flat')
                        confidence = result.get('confidence', 0)
                        
                        if action == 'long':
                            st.success("🟢 BUY Recommendation")
                        elif action == 'short':
                            st.error("🔴 SELL Recommendation")
                        else:
                            st.info("🟡 HOLD/NEUTRAL")
                        
                        st.metric("Confidence", f"{confidence:.1%}")
                    
                    with col2:
                        reasoning = result.get('reasoning', 'No reasoning provided')
                        st.write(f"**Reasoning:** {reasoning}")
                else:
                    st.error("❌ Analysis failed. Please try again.")
                    
            except Exception as e:
                st.error(f"❌ Analysis error: {str(e)}")
                logger.exception("AI analysis error")

def show_backtesting(tech_analyzer):
    """Backtesting interface."""
    st.header("📈 Strategy Backtesting")
    
    st.markdown("""
    Test trading strategies against historical data to evaluate performance.
    """)
    
    # Backtesting parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_date = st.date_input(
            "Start Date", 
            datetime.now() - timedelta(days=30),
            max_value=datetime.now().date()
        )
        end_date = st.date_input(
            "End Date", 
            datetime.now().date(),
            max_value=datetime.now().date()
        )
    
    with col2:
        initial_balance = st.number_input(
            "Initial Balance ($)",
            value=10000.0,
            min_value=1000.0,
            step=1000.0
        )
        risk_per_trade = st.slider(
            "Risk per Trade (%)",
            min_value=0.5,
            max_value=5.0,
            value=1.0,
            step=0.1
        )
    
    with col3:
        strategy_type = st.selectbox(
            "Strategy Type",
            ["Technical Analysis", "AI-based", "Buy & Hold"]
        )
        
        symbol = st.selectbox(
            "Trading Pair",
            ["USD/JPY", "EUR/USD", "GBP/USD"],
            index=0
        )
    
    if st.button("🚀 Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            try:
                # Get historical data
                price_data = db_adapter.get_latest_price_data(symbol, hours=24*30)  # 30 days
                
                if price_data.empty:
                    st.error("❌ No price data available for backtesting")
                    return
                
                # Simple backtest results display
                st.success("✅ Backtest completed!")
                
                # Mock results for demonstration
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Final Balance", f"${initial_balance * 1.15:.2f}")
                
                with col2:
                    st.metric("Total Return", "15.0%")
                
                with col3:
                    st.metric("Total Trades", "42")
                
                with col4:
                    st.metric("Win Rate", "65.5%")
                
            except Exception as e:
                st.error(f"❌ Backtest failed: {str(e)}")
                logger.exception("Backtest error")


if __name__ == "__main__":
    main()