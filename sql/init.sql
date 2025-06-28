-- Initialize TimescaleDB extension and create trading bot schema
-- This script runs automatically when the PostgreSQL container starts

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create main schema
CREATE SCHEMA IF NOT EXISTS trading;

-- Price data table with TimescaleDB hypertable
CREATE TABLE IF NOT EXISTS trading.price_data (
    t TIMESTAMP NOT NULL,
    trading_pair VARCHAR(20) NOT NULL,
    open DECIMAL(18,8) NOT NULL,
    high DECIMAL(18,8) NOT NULL,
    low DECIMAL(18,8) NOT NULL,
    close DECIMAL(18,8) NOT NULL,
    volume DECIMAL(18,8) NOT NULL,
    PRIMARY KEY (t, trading_pair)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('trading.price_data', 't', if_not_exists => TRUE);

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_price_data_trading_pair ON trading.price_data (trading_pair, t DESC);

-- News articles table
CREATE TABLE IF NOT EXISTS trading.news_articles (
    news_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source VARCHAR(100) NOT NULL,
    url TEXT NOT NULL,
    ai_processed BOOLEAN DEFAULT FALSE
);

-- Create index for news queries
CREATE INDEX IF NOT EXISTS idx_news_articles_timestamp ON trading.news_articles (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_news_articles_ai_processed ON trading.news_articles (ai_processed);

-- News tags for categorization
CREATE TABLE IF NOT EXISTS trading.news_tags (
    tag_id SERIAL PRIMARY KEY,
    news_id INTEGER REFERENCES trading.news_articles(news_id),
    tag VARCHAR(50) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL
);

-- Create index for news tags
CREATE INDEX IF NOT EXISTS idx_news_tags_news_id ON trading.news_tags (news_id);

-- Alpha signals table
CREATE TABLE IF NOT EXISTS trading.alpha_signals (
    signal_id SERIAL UNIQUE,
    timestamp TIMESTAMP NOT NULL,
    trading_pair VARCHAR(20) NOT NULL,
    signal_type VARCHAR(50) NOT NULL,
    signal_value DECIMAL(18,8) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    metadata JSONB,
    PRIMARY KEY (signal_id, timestamp)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('trading.alpha_signals', 'timestamp', if_not_exists => TRUE);

-- Trade history table
CREATE TABLE IF NOT EXISTS trading.trade_history (
    trade_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    trading_pair VARCHAR(20) NOT NULL,
    side VARCHAR(4) NOT NULL,
    price DECIMAL(18,8) NOT NULL,
    amount DECIMAL(18,8) NOT NULL,
    cost DECIMAL(18,8) NOT NULL,
    fee DECIMAL(18,8) NOT NULL,
    signal_id INTEGER REFERENCES trading.alpha_signals(signal_id)
);

-- Create index for trade queries
CREATE INDEX IF NOT EXISTS idx_trade_history_timestamp ON trading.trade_history (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_trading_pair ON trading.trade_history (trading_pair);

-- AI trade decisions table
CREATE TABLE IF NOT EXISTS trading.ai_trade_decisions (
    decision_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    trading_pair VARCHAR(20) NOT NULL,
    decision_type VARCHAR(50) NOT NULL,
    confidence DECIMAL(5,4) NOT NULL,
    reasoning TEXT NOT NULL,
    metadata JSONB,
    executed BOOLEAN DEFAULT FALSE
);

-- Create index for AI decisions
CREATE INDEX IF NOT EXISTS idx_ai_trade_decisions_timestamp ON trading.ai_trade_decisions (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ai_decisions_trading_pair ON trading.ai_trade_decisions (trading_pair);

-- OpenAI API usage tracking
CREATE TABLE IF NOT EXISTS trading.openai_api_usage (
    usage_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    endpoint VARCHAR(100) NOT NULL,
    tokens_used INTEGER NOT NULL,
    cost DECIMAL(10,6) NOT NULL,
    success BOOLEAN NOT NULL,
    error_message TEXT
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('trading.openai_api_usage', 'timestamp', if_not_exists => TRUE);

-- Manual analysis results
CREATE TABLE IF NOT EXISTS trading.openai_manual_analysis (
    analysis_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    input_text TEXT NOT NULL,
    output_text TEXT NOT NULL,
    tokens_used INTEGER NOT NULL,
    cost DECIMAL(10,6) NOT NULL
);

-- Redis stream tracking (for monitoring)
CREATE TABLE IF NOT EXISTS trading.stream_offsets (
    stream_name TEXT PRIMARY KEY,
    last_processed_id TEXT,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create retention policies for TimescaleDB (keep data for specific periods)
-- Keep price data for 2 years
SELECT add_retention_policy('trading.price_data', INTERVAL '2 years', if_not_exists => TRUE);

-- Keep API usage data for 1 year
SELECT add_retention_policy('trading.openai_api_usage', INTERVAL '1 year', if_not_exists => TRUE);

-- Keep alpha signals for 1 year
SELECT add_retention_policy('trading.alpha_signals', INTERVAL '1 year', if_not_exists => TRUE);

-- Create materialized views for common aggregations
CREATE MATERIALIZED VIEW IF NOT EXISTS trading.daily_pnl AS
SELECT 
    DATE(timestamp) as trade_date,
    trading_pair,
    COUNT(*) as total_trades,
    SUM(CASE WHEN cost > 0 THEN 1 ELSE 0 END) as winning_trades,
    SUM(cost) as total_pnl,
    AVG(cost) as avg_pips,
    MAX(cost) as max_win,
    MIN(cost) as max_loss
FROM trading.trade_history 
WHERE timestamp IS NOT NULL
GROUP BY DATE(timestamp), trading_pair
ORDER BY trade_date DESC;

-- Create view for recent AI model performance
CREATE OR REPLACE VIEW trading.ai_model_performance AS
SELECT 
    trading_pair,
    COUNT(*) as total_decisions,
    SUM(CASE WHEN decision_type = 'trade' THEN 1 ELSE 0 END) as trade_recommendations,
    AVG(confidence) as avg_confidence,
    DATE_TRUNC('day', timestamp) as date
FROM trading.ai_trade_decisions
WHERE timestamp >= NOW() - INTERVAL '30 days'
GROUP BY trading_pair, DATE_TRUNC('day', timestamp)
ORDER BY date DESC, trading_pair;

-- Grant permissions to bot user
GRANT USAGE ON SCHEMA trading TO botuser;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA trading TO botuser;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA trading TO botuser;
GRANT SELECT ON ALL TABLES IN SCHEMA trading TO botuser;

-- Set default schema for convenience
ALTER USER botuser SET search_path TO trading, public;

-- Create function to refresh materialized views
CREATE OR REPLACE FUNCTION trading.refresh_daily_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW trading.daily_pnl;
END;
$$ LANGUAGE plpgsql;

-- AI Learning Data Tables for ProjectChimera 4-Layer System

-- AI decision context table for learning data
CREATE TABLE IF NOT EXISTS trading.ai_decision_contexts (
    context_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    decision_id INTEGER,
    symbol VARCHAR(20) NOT NULL,
    decision_type VARCHAR(50) NOT NULL, -- '1min_trade' or '1hour_strategy'
    
    -- Market context at decision time
    market_data JSONB NOT NULL,
    price_history JSONB NOT NULL,
    orderbook_data JSONB,
    
    -- Sentiment context
    news_data JSONB,
    x_posts_data JSONB,
    sentiment_summary JSONB,
    
    -- Portfolio context
    portfolio_state JSONB,
    current_positions JSONB,
    
    -- Technical indicators
    technical_indicators JSONB,
    
    -- Model metadata
    model_name VARCHAR(50) NOT NULL,
    prompt_version VARCHAR(20) NOT NULL,
    api_cost DECIMAL(10,6),
    response_time_ms INTEGER,
    tokens_used INTEGER
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('trading.ai_decision_contexts', 'timestamp', if_not_exists => TRUE);

-- Create indexes for AI learning queries
CREATE INDEX IF NOT EXISTS idx_ai_contexts_symbol_time ON trading.ai_decision_contexts (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ai_contexts_decision_type ON trading.ai_decision_contexts (decision_type, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ai_contexts_model ON trading.ai_decision_contexts (model_name, timestamp DESC);

-- AI decision outcomes table for performance tracking
CREATE TABLE IF NOT EXISTS trading.ai_decision_outcomes (
    outcome_id SERIAL PRIMARY KEY,
    context_id INTEGER REFERENCES trading.ai_decision_contexts(context_id),
    decision_id INTEGER NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    
    -- Decision details
    predicted_action VARCHAR(10) NOT NULL,
    predicted_confidence DECIMAL(5,4) NOT NULL,
    predicted_price DECIMAL(18,8),
    predicted_timeframe INTEGER,
    
    -- Actual outcomes
    actual_executed BOOLEAN DEFAULT FALSE,
    actual_entry_price DECIMAL(18,8),
    actual_exit_price DECIMAL(18,8),
    actual_pnl DECIMAL(18,8),
    actual_pnl_pct DECIMAL(8,6),
    actual_duration_minutes INTEGER,
    
    -- Performance metrics
    prediction_accuracy DECIMAL(5,4), -- How accurate was the direction prediction
    confidence_calibration DECIMAL(5,4), -- How well calibrated was the confidence
    risk_reward_ratio DECIMAL(8,4),
    
    -- Timestamps
    decision_timestamp TIMESTAMP NOT NULL,
    outcome_timestamp TIMESTAMP,
    
    -- Metadata
    execution_notes TEXT,
    market_conditions JSONB
);

-- Create indexes for outcome analysis
CREATE INDEX IF NOT EXISTS idx_ai_outcomes_symbol_time ON trading.ai_decision_outcomes (symbol, decision_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ai_outcomes_accuracy ON trading.ai_decision_outcomes (prediction_accuracy DESC);
CREATE INDEX IF NOT EXISTS idx_ai_outcomes_pnl ON trading.ai_decision_outcomes (actual_pnl DESC);

-- X/Twitter posts table for sentiment analysis
CREATE TABLE IF NOT EXISTS trading.x_posts (
    post_id VARCHAR(50) PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    author VARCHAR(100) NOT NULL,
    text TEXT NOT NULL,
    
    -- Engagement metrics
    engagement_score DECIMAL(5,4) NOT NULL,
    retweet_count INTEGER DEFAULT 0,
    like_count INTEGER DEFAULT 0,
    reply_count INTEGER DEFAULT 0,
    
    -- Sentiment analysis
    sentiment_score DECIMAL(5,4), -- -1 to +1
    sentiment_confidence DECIMAL(5,4),
    
    -- Categorization
    tags JSONB,
    relevance_score DECIMAL(5,4),
    
    -- Source tracking
    collection_timestamp TIMESTAMP DEFAULT NOW(),
    source_query VARCHAR(200)
);

-- Create index for X posts queries
CREATE INDEX IF NOT EXISTS idx_x_posts_timestamp ON trading.x_posts (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_x_posts_sentiment ON trading.x_posts (sentiment_score DESC);
CREATE INDEX IF NOT EXISTS idx_x_posts_relevance ON trading.x_posts (relevance_score DESC);

-- Enhanced news articles table (update existing)
ALTER TABLE IF EXISTS trading.news_articles 
ADD COLUMN IF NOT EXISTS relevance_score DECIMAL(5,4),
ADD COLUMN IF NOT EXISTS sentiment_score DECIMAL(5,4),
ADD COLUMN IF NOT EXISTS collection_timestamp TIMESTAMP DEFAULT NOW(),
ADD COLUMN IF NOT EXISTS source_priority INTEGER DEFAULT 2;

-- Redis streams tracking for monitoring
CREATE TABLE IF NOT EXISTS trading.redis_streams_stats (
    stream_name VARCHAR(100) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    messages_published INTEGER DEFAULT 0,
    messages_consumed INTEGER DEFAULT 0,
    consumer_lag INTEGER DEFAULT 0,
    stream_length INTEGER DEFAULT 0,
    PRIMARY KEY (stream_name, timestamp)
);

-- Convert to hypertable
SELECT create_hypertable('trading.redis_streams_stats', 'timestamp', if_not_exists => TRUE);

-- API usage tracking for cost management
CREATE TABLE IF NOT EXISTS trading.api_usage_detailed (
    usage_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    service VARCHAR(50) NOT NULL, -- 'openai', 'news_api', etc.
    endpoint VARCHAR(100) NOT NULL,
    
    -- Request details
    request_type VARCHAR(50), -- '1min_decision', '1hour_strategy', etc.
    model_name VARCHAR(50),
    
    -- Usage metrics
    tokens_input INTEGER,
    tokens_output INTEGER,
    tokens_total INTEGER,
    
    -- Cost tracking
    cost_usd DECIMAL(10,6) NOT NULL,
    rate_per_token DECIMAL(12,8),
    
    -- Performance
    response_time_ms INTEGER,
    success BOOLEAN NOT NULL,
    error_message TEXT,
    
    -- Context
    symbol VARCHAR(20),
    additional_metadata JSONB
);

-- Convert to hypertable
SELECT create_hypertable('trading.api_usage_detailed', 'timestamp', if_not_exists => TRUE);

-- Create indexes for cost analysis
CREATE INDEX IF NOT EXISTS idx_api_usage_service ON trading.api_usage_detailed (service, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_api_usage_cost ON trading.api_usage_detailed (cost_usd DESC);
CREATE INDEX IF NOT EXISTS idx_api_usage_model ON trading.api_usage_detailed (model_name, timestamp DESC);

-- Add retention policies for new tables
SELECT add_retention_policy('trading.ai_decision_contexts', INTERVAL '6 months', if_not_exists => TRUE);
SELECT add_retention_policy('trading.redis_streams_stats', INTERVAL '1 month', if_not_exists => TRUE);
SELECT add_retention_policy('trading.api_usage_detailed', INTERVAL '1 year', if_not_exists => TRUE);

-- Create materialized view for AI performance analysis
CREATE MATERIALIZED VIEW IF NOT EXISTS trading.ai_performance_daily AS
SELECT 
    DATE(decision_timestamp) as analysis_date,
    symbol,
    COUNT(*) as total_decisions,
    COUNT(CASE WHEN actual_executed = true THEN 1 END) as executed_decisions,
    AVG(predicted_confidence) as avg_confidence,
    AVG(prediction_accuracy) as avg_accuracy,
    AVG(actual_pnl) as avg_pnl,
    SUM(actual_pnl) as total_pnl,
    SUM(CASE WHEN actual_pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
    AVG(risk_reward_ratio) as avg_risk_reward
FROM trading.ai_decision_outcomes
WHERE decision_timestamp >= NOW() - INTERVAL '30 days'
GROUP BY DATE(decision_timestamp), symbol
ORDER BY analysis_date DESC, symbol;

-- Create view for recent AI decision analysis
CREATE OR REPLACE VIEW trading.ai_recent_performance AS
SELECT 
    o.symbol,
    o.predicted_action,
    o.predicted_confidence,
    o.actual_pnl,
    o.prediction_accuracy,
    o.decision_timestamp,
    c.model_name,
    c.api_cost,
    c.response_time_ms
FROM trading.ai_decision_outcomes o
JOIN trading.ai_decision_contexts c ON o.context_id = c.context_id
WHERE o.decision_timestamp >= NOW() - INTERVAL '7 days'
ORDER BY o.decision_timestamp DESC;

-- Function to refresh AI performance views
CREATE OR REPLACE FUNCTION trading.refresh_ai_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW trading.ai_performance_daily;
    REFRESH MATERIALIZED VIEW trading.daily_pnl;
END;
$$ LANGUAGE plpgsql;

COMMENT ON DATABASE trading_bot IS 'AI Trading Platform - ProjectChimera 4-Layer System Database';
COMMENT ON SCHEMA trading IS 'Main trading application schema with TimescaleDB optimization and AI learning data';