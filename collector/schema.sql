-- llm-obs-mcp  TimescaleDB Schema
-- Run once against your TimescaleDB instance:
--   psql $TIMESCALEDB_URL -f collector/schema.sql

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ── 1. inference_metrics ──────────────────────────────────────────────────────
-- One row per (model, scrape tick). Written every 15 s by collector/scraper.py.
CREATE TABLE IF NOT EXISTS inference_metrics (
    time                    TIMESTAMPTZ      NOT NULL,
    model_name              TEXT             NOT NULL,
    model_type              TEXT             NOT NULL DEFAULT 'llm',
    -- vLLM latency histogram quantiles (converted to ms)
    latency_p50_ms          DOUBLE PRECISION,
    latency_p95_ms          DOUBLE PRECISION,
    latency_p99_ms          DOUBLE PRECISION,
    -- Time To First Token
    ttft_p50_ms             DOUBLE PRECISION,
    ttft_p95_ms             DOUBLE PRECISION,
    -- Time Per Output Token (mean, ms)
    tpot_mean_ms            DOUBLE PRECISION,
    -- Throughput (derived: delta success_total / scrape_interval)
    throughput_rps          DOUBLE PRECISION,
    -- Cumulative counters (kept for delta computation; not queried directly)
    prompt_tokens_total     BIGINT,
    generation_tokens_total BIGINT,
    request_success_total   BIGINT,
    -- Concurrency
    requests_running        INTEGER,
    requests_waiting        INTEGER,
    -- KV cache 0-100 %
    kv_cache_usage_pct      DOUBLE PRECISION,
    -- GPU metrics from DCGM exporter
    gpu_util_pct            DOUBLE PRECISION,
    gpu_mem_used_mb         DOUBLE PRECISION,
    gpu_mem_free_mb         DOUBLE PRECISION,
    gpu_temp_celsius        DOUBLE PRECISION,
    gpu_power_watts         DOUBLE PRECISION,
    -- K8s pod resource usage
    pod_cpu_millicores      INTEGER,
    pod_mem_mb              DOUBLE PRECISION,
    -- TTS-only: Real-Time Factor (< 1.0 = faster than real-time)
    rtf                     DOUBLE PRECISION
);
SELECT create_hypertable('inference_metrics', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_im_model_time
    ON inference_metrics (model_name, time DESC);
SELECT add_retention_policy('inference_metrics',
    INTERVAL '90 days', if_not_exists => TRUE);

-- ── 2. anomaly_events ─────────────────────────────────────────────────────────
-- Written every 60 s by analytics/anomaly.py.
-- Also used by the alert webhook to store threshold / drift alerts.
CREATE TABLE IF NOT EXISTS anomaly_events (
    computed_at             TIMESTAMPTZ      NOT NULL,
    model_name              TEXT             NOT NULL,
    anomaly_score           DOUBLE PRECISION NOT NULL,
    is_anomaly              BOOLEAN          NOT NULL,
    cold_start_mode         BOOLEAN          NOT NULL DEFAULT FALSE,
    -- JSON array of {feature, value, baseline, deviation_pct}
    contributing_dimensions JSONB,
    -- Optional fields used by alert webhook (threshold / drift alerts)
    alert_type              TEXT,           -- 'anomaly' | 'threshold' | 'drift'
    severity                TEXT,           -- 'warning' | 'critical'
    title                   TEXT,
    description             TEXT,
    resolved_at             TIMESTAMPTZ
);
SELECT create_hypertable('anomaly_events', 'computed_at', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_ae_model_computed_at
    ON anomaly_events (model_name, computed_at DESC);
SELECT add_retention_policy('anomaly_events',
    INTERVAL '90 days', if_not_exists => TRUE);

-- ── 3. drift_scores ───────────────────────────────────────────────────────────
-- Written every 5 min by analytics/drift.py.
CREATE TABLE IF NOT EXISTS drift_scores (
    computed_at             TIMESTAMPTZ      NOT NULL,
    model_name              TEXT             NOT NULL,
    metric                  TEXT             NOT NULL,
    reference_window        TEXT             NOT NULL,
    comparison_window       TEXT             NOT NULL,
    severity                TEXT,           -- 'none' | 'low' | 'medium' | 'high'
    ks_statistic            DOUBLE PRECISION,
    ks_p_value              DOUBLE PRECISION,
    psi                     DOUBLE PRECISION,
    reference_mean          DOUBLE PRECISION,
    comparison_mean         DOUBLE PRECISION,
    drift_pct               DOUBLE PRECISION,
    n_reference             INTEGER,
    n_comparison            INTEGER,
    UNIQUE (model_name, metric, reference_window, comparison_window)
);
SELECT create_hypertable('drift_scores', 'computed_at', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_ds_model_metric
    ON drift_scores (model_name, metric, computed_at DESC);
SELECT add_retention_policy('drift_scores',
    INTERVAL '90 days', if_not_exists => TRUE);

-- ── 4. forecast_snapshots ─────────────────────────────────────────────────────
-- Written every 30 min by analytics/forecaster.py.
CREATE TABLE IF NOT EXISTS forecast_snapshots (
    computed_at             TIMESTAMPTZ      NOT NULL,
    model_name              TEXT             NOT NULL,
    metric                  TEXT             NOT NULL,
    horizon_hours           INTEGER          NOT NULL,
    saturation_threshold    DOUBLE PRECISION,
    saturation_time         TIMESTAMPTZ,
    saturation_risk         TEXT,           -- 'low' | 'medium' | 'high' | 'critical'
    n_training_points       INTEGER,
    cold_start_mode         BOOLEAN          NOT NULL DEFAULT FALSE,
    -- JSON array of {timestamp, yhat, yhat_lower, yhat_upper}
    forecast_points_json    JSONB            NOT NULL
);
SELECT create_hypertable('forecast_snapshots', 'computed_at', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_fs_model_metric
    ON forecast_snapshots (model_name, metric, computed_at DESC);
SELECT add_retention_policy('forecast_snapshots',
    INTERVAL '30 days', if_not_exists => TRUE);

-- ── 5. model_artifacts ────────────────────────────────────────────────────────
-- Serialised IsolationForest + StandardScaler blobs per model.
CREATE TABLE IF NOT EXISTS model_artifacts (
    model_name              TEXT             NOT NULL,
    model_type              TEXT             NOT NULL DEFAULT 'llm',
    trained_at              TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    scaler_blob             BYTEA,
    model_blob              BYTEA            NOT NULL,
    feature_names           JSONB,
    baseline_means          JSONB,
    baseline_stds           JSONB,
    n_training_samples      INTEGER,
    PRIMARY KEY (model_name)
);
CREATE INDEX IF NOT EXISTS idx_ma_model_trained
    ON model_artifacts (model_name, trained_at DESC);
