-- A/B Test System Database Schema
-- AIモデルA/Bテストシステム用データベーススキーマ

-- A/Bテスト設定テーブル
CREATE TABLE IF NOT EXISTS ab_test_configs (
    test_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    models_to_test TEXT NOT NULL,  -- JSON array of model names
    traffic_split TEXT NOT NULL,   -- JSON object with model -> percentage
    duration_days INTEGER NOT NULL,
    min_samples INTEGER NOT NULL,
    confidence_level REAL NOT NULL DEFAULT 0.95,
    success_metrics TEXT,          -- JSON array of metric names
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL
);

-- A/Bテスト結果テーブル
CREATE TABLE IF NOT EXISTS ab_test_results (
    result_id TEXT PRIMARY KEY,
    test_id TEXT NOT NULL,
    model_used TEXT NOT NULL,
    request_data TEXT NOT NULL,    -- JSON of input data
    ai_response TEXT NOT NULL,     -- JSON of AI response
    actual_outcome TEXT,           -- JSON of actual market outcome
    metrics TEXT NOT NULL,         -- JSON of calculated metrics
    timestamp TIMESTAMP NOT NULL,
    processing_time_ms REAL NOT NULL,
    cost_usd REAL NOT NULL,
    confidence_score REAL NOT NULL,
    success BOOLEAN,               -- True if prediction was correct
    FOREIGN KEY (test_id) REFERENCES ab_test_configs(test_id)
);

-- A/Bテストサマリーテーブル
CREATE TABLE IF NOT EXISTS ab_test_summaries (
    test_id TEXT PRIMARY KEY,
    total_samples INTEGER NOT NULL,
    model_performance TEXT NOT NULL,      -- JSON of model performance stats
    statistical_significance TEXT,        -- JSON of significance test results
    winning_model TEXT,                   -- Name of best performing model
    cost_analysis TEXT NOT NULL,          -- JSON of cost breakdown
    recommendations TEXT NOT NULL,        -- JSON array of recommendations
    generated_at TIMESTAMP NOT NULL,
    FOREIGN KEY (test_id) REFERENCES ab_test_configs(test_id)
);

-- モデル別日次パフォーマンス集計テーブル
CREATE TABLE IF NOT EXISTS ab_test_daily_stats (
    stat_id TEXT PRIMARY KEY,
    test_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    date DATE NOT NULL,
    total_requests INTEGER NOT NULL DEFAULT 0,
    total_cost REAL NOT NULL DEFAULT 0.0,
    avg_confidence REAL NOT NULL DEFAULT 0.0,
    avg_response_time REAL NOT NULL DEFAULT 0.0,
    success_count INTEGER NOT NULL DEFAULT 0,
    success_rate REAL NOT NULL DEFAULT 0.0,
    cost_efficiency REAL NOT NULL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(test_id, model_name, date),
    FOREIGN KEY (test_id) REFERENCES ab_test_configs(test_id)
);

-- インデックス作成
CREATE INDEX IF NOT EXISTS idx_ab_test_results_test_id ON ab_test_results(test_id);
CREATE INDEX IF NOT EXISTS idx_ab_test_results_model ON ab_test_results(model_used);
CREATE INDEX IF NOT EXISTS idx_ab_test_results_timestamp ON ab_test_results(timestamp);
CREATE INDEX IF NOT EXISTS idx_ab_test_results_success ON ab_test_results(success);

CREATE INDEX IF NOT EXISTS idx_ab_test_configs_status ON ab_test_configs(status);
CREATE INDEX IF NOT EXISTS idx_ab_test_configs_created ON ab_test_configs(created_at);

CREATE INDEX IF NOT EXISTS idx_ab_daily_stats_test_model ON ab_test_daily_stats(test_id, model_name);
CREATE INDEX IF NOT EXISTS idx_ab_daily_stats_date ON ab_test_daily_stats(date);

-- ビュー: モデル別パフォーマンス概要
CREATE VIEW IF NOT EXISTS v_model_performance_overview AS
SELECT 
    r.test_id,
    r.model_used,
    COUNT(*) as total_requests,
    AVG(r.confidence_score) as avg_confidence,
    AVG(r.processing_time_ms) as avg_response_time,
    AVG(r.cost_usd) as avg_cost,
    SUM(r.cost_usd) as total_cost,
    COUNT(CASE WHEN r.success = 1 THEN 1 END) as success_count,
    CAST(COUNT(CASE WHEN r.success = 1 THEN 1 END) AS REAL) / COUNT(*) as success_rate,
    AVG(r.confidence_score / r.cost_usd) as avg_cost_efficiency,
    MIN(r.timestamp) as first_request,
    MAX(r.timestamp) as last_request
FROM ab_test_results r
WHERE r.cost_usd > 0
GROUP BY r.test_id, r.model_used;

-- ビュー: テスト進行状況
CREATE VIEW IF NOT EXISTS v_test_progress AS
SELECT 
    c.test_id,
    c.name,
    c.status,
    c.min_samples,
    c.duration_days,
    c.created_at,
    COUNT(r.result_id) as current_samples,
    CAST(COUNT(r.result_id) AS REAL) / c.min_samples as sample_progress,
    CASE 
        WHEN julianday('now') - julianday(c.created_at) >= c.duration_days THEN 1.0
        ELSE (julianday('now') - julianday(c.created_at)) / c.duration_days
    END as time_progress
FROM ab_test_configs c
LEFT JOIN ab_test_results r ON c.test_id = r.test_id
GROUP BY c.test_id, c.name, c.status, c.min_samples, c.duration_days, c.created_at;

-- ビュー: コスト分析
CREATE VIEW IF NOT EXISTS v_cost_analysis AS
SELECT 
    r.test_id,
    r.model_used,
    COUNT(*) as request_count,
    SUM(r.cost_usd) as total_cost,
    AVG(r.cost_usd) as avg_cost_per_request,
    MIN(r.cost_usd) as min_cost,
    MAX(r.cost_usd) as max_cost,
    SUM(r.cost_usd) / COUNT(CASE WHEN r.success = 1 THEN 1 END) as cost_per_success,
    strftime('%Y-%m-%d', r.timestamp) as date
FROM ab_test_results r
WHERE r.cost_usd > 0
GROUP BY r.test_id, r.model_used, strftime('%Y-%m-%d', r.timestamp);

-- サンプルデータ（開発・テスト用）
-- 削除する場合は以下をコメントアウト

/*
-- サンプルテスト設定
INSERT OR IGNORE INTO ab_test_configs (
    test_id, name, description, models_to_test, traffic_split,
    duration_days, min_samples, confidence_level, success_metrics,
    status, created_at
) VALUES (
    'test_001',
    'GPT-4 vs O3-mini Performance Test',
    'Comparing GPT-4 and O3-mini for forex trading signal generation',
    '["gpt-4", "o3-mini"]',
    '{"gpt-4": 0.5, "o3-mini": 0.5}',
    7,
    100,
    0.95,
    '["accuracy", "confidence", "cost_efficiency", "response_time"]',
    'active',
    datetime('now')
);

-- サンプル結果データ
INSERT OR IGNORE INTO ab_test_results (
    result_id, test_id, model_used, request_data, ai_response,
    metrics, timestamp, processing_time_ms, cost_usd, confidence_score, success
) VALUES 
(
    'result_001',
    'test_001',
    'gpt-4',
    '{"news": "Fed raises interest rates", "pair": "USD/JPY"}',
    '{"trade_warranted": true, "direction": "long", "confidence": 0.8}',
    '{"confidence": 0.8, "cost_efficiency": 8.0, "time_efficiency": 4.0}',
    datetime('now', '-1 hour'),
    1500.0,
    0.10,
    0.8,
    1
),
(
    'result_002',
    'test_001',
    'o3-mini',
    '{"news": "Fed raises interest rates", "pair": "USD/JPY"}',
    '{"trade_warranted": true, "direction": "long", "confidence": 0.75}',
    '{"confidence": 0.75, "cost_efficiency": 25.0, "time_efficiency": 7.5}',
    datetime('now', '-30 minutes'),
    800.0,
    0.03,
    0.75,
    1
);
*/