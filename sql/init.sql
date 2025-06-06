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
    signal_id SERIAL,
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

COMMENT ON DATABASE trading_bot IS 'AI Trading Platform - The Architect v2 Database';
COMMENT ON SCHEMA trading IS 'Main trading application schema with TimescaleDB optimization';