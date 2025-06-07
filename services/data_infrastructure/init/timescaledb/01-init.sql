-- TimescaleDB Initialization Script for GCP Project
-- This script creates the necessary hypertables for time-series data

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "timescaledb";
CREATE EXTENSION IF NOT EXISTS "postgis";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS entropy;
CREATE SCHEMA IF NOT EXISTS coherence;
CREATE SCHEMA IF NOT EXISTS metrics;
CREATE SCHEMA IF NOT EXISTS events;

-- Set search path
SET search_path TO entropy, coherence, metrics, events, public;

-- Create entropy tables
CREATE TABLE IF NOT EXISTS entropy.raw_samples (
    ts TIMESTAMPTZ NOT NULL,
    node_id BIGINT NOT NULL,
    sample_hash BYTEA NOT NULL,
    quality FLOAT NOT NULL
);

SELECT create_hypertable('entropy.raw_samples', 'ts', chunk_time_interval => INTERVAL '1 hour');

CREATE TABLE IF NOT EXISTS entropy.quality_scores (
    ts TIMESTAMPTZ NOT NULL,
    node_id BIGINT NOT NULL,
    quality FLOAT NOT NULL,
    frequency_test FLOAT,
    runs_test FLOAT,
    poker_test FLOAT,
    autocorr_test FLOAT,
    entropy_bits FLOAT
);

SELECT create_hypertable('entropy.quality_scores', 'ts', chunk_time_interval => INTERVAL '1 hour');

-- Create coherence tables
CREATE TABLE IF NOT EXISTS coherence.metrics (
    ts TIMESTAMPTZ NOT NULL,
    node_a BIGINT NOT NULL,
    node_b BIGINT NOT NULL,
    metric TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL
);

SELECT create_hypertable('coherence.metrics', 'ts', chunk_time_interval => INTERVAL '1 hour');

CREATE TABLE IF NOT EXISTS coherence.global_metrics (
    ts TIMESTAMPTZ NOT NULL,
    metric TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    node_count INTEGER NOT NULL,
    region GEOGRAPHY
);

SELECT create_hypertable('coherence.global_metrics', 'ts', chunk_time_interval => INTERVAL '1 hour');

-- Create metrics tables
CREATE TABLE IF NOT EXISTS metrics.node_metrics (
    ts TIMESTAMPTZ NOT NULL,
    node_id BIGINT NOT NULL,
    metric TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL
);

SELECT create_hypertable('metrics.node_metrics', 'ts', chunk_time_interval => INTERVAL '1 hour');

CREATE TABLE IF NOT EXISTS metrics.system_metrics (
    ts TIMESTAMPTZ NOT NULL,
    host TEXT NOT NULL,
    service TEXT NOT NULL,
    metric TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL
);

SELECT create_hypertable('metrics.system_metrics', 'ts', chunk_time_interval => INTERVAL '1 hour');

-- Create events tables
CREATE TABLE IF NOT EXISTS events.detected_events (
    ts TIMESTAMPTZ NOT NULL,
    event_type TEXT NOT NULL,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT,
    geo_lat FLOAT,
    geo_lon FLOAT,
    geo_city TEXT,
    geo_country TEXT,
    confidence FLOAT NOT NULL,
    correlation_id UUID,
    metadata JSONB
);

SELECT create_hypertable('events.detected_events', 'ts', chunk_time_interval => INTERVAL '1 day');

CREATE TABLE IF NOT EXISTS events.spikes (
    ts TIMESTAMPTZ NOT NULL,
    metric TEXT NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    z_score DOUBLE PRECISION NOT NULL,
    region GEOGRAPHY,
    is_global BOOLEAN NOT NULL,
    event_id UUID
);

SELECT create_hypertable('events.spikes', 'ts', chunk_time_interval => INTERVAL '1 day');

-- Create continuous aggregates for common queries
CREATE MATERIALIZED VIEW IF NOT EXISTS entropy.hourly_quality
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', ts) AS bucket,
    node_id,
    avg(quality) AS avg_quality,
    min(quality) AS min_quality,
    max(quality) AS max_quality,
    count(*) AS sample_count
FROM entropy.quality_scores
GROUP BY bucket, node_id;

CREATE MATERIALIZED VIEW IF NOT EXISTS coherence.hourly_global_cec
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', ts) AS bucket,
    avg(value) AS avg_cec,
    min(value) AS min_cec,
    max(value) AS max_cec,
    count(*) AS sample_count
FROM coherence.global_metrics
WHERE metric = 'CEC'
GROUP BY bucket;

CREATE MATERIALIZED VIEW IF NOT EXISTS metrics.daily_system_metrics
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 day', ts) AS bucket,
    host,
    service,
    metric,
    avg(value) AS avg_value,
    min(value) AS min_value,
    max(value) AS max_value,
    count(*) AS sample_count
FROM metrics.system_metrics
GROUP BY bucket, host, service, metric;

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_raw_samples_node_ts ON entropy.raw_samples(node_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_quality_scores_node_ts ON entropy.quality_scores(node_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_coherence_metrics_nodes ON coherence.metrics(node_a, node_b, ts DESC);
CREATE INDEX IF NOT EXISTS idx_coherence_metrics_metric ON coherence.metrics(metric, ts DESC);
CREATE INDEX IF NOT EXISTS idx_global_metrics_metric ON coherence.global_metrics(metric, ts DESC);
CREATE INDEX IF NOT EXISTS idx_node_metrics_node ON metrics.node_metrics(node_id, metric, ts DESC);
CREATE INDEX IF NOT EXISTS idx_system_metrics_host ON metrics.system_metrics(host, service, ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_type ON events.detected_events(event_type, ts DESC);
CREATE INDEX IF NOT EXISTS idx_events_geo ON events.detected_events(geo_country, geo_city);
CREATE INDEX IF NOT EXISTS idx_spikes_metric ON events.spikes(metric, ts DESC);
CREATE INDEX IF NOT EXISTS idx_spikes_zscore ON events.spikes(z_score DESC, ts DESC);

-- Set retention policies
SELECT add_retention_policy('entropy.raw_samples', INTERVAL '7 days');
SELECT add_retention_policy('entropy.quality_scores', INTERVAL '30 days');
SELECT add_retention_policy('coherence.metrics', INTERVAL '90 days');
SELECT add_retention_policy('coherence.global_metrics', INTERVAL '365 days');
SELECT add_retention_policy('metrics.node_metrics', INTERVAL '30 days');
SELECT add_retention_policy('metrics.system_metrics', INTERVAL '30 days');
SELECT add_retention_policy('events.detected_events', INTERVAL '365 days');
SELECT add_retention_policy('events.spikes', INTERVAL '365 days');

-- Set refresh policies for continuous aggregates
SELECT add_continuous_aggregate_policy('entropy.hourly_quality',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

SELECT add_continuous_aggregate_policy('coherence.hourly_global_cec',
    start_offset => INTERVAL '2 days',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour');

SELECT add_continuous_aggregate_policy('metrics.daily_system_metrics',
    start_offset => INTERVAL '7 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day');

-- Grant privileges
GRANT ALL PRIVILEGES ON SCHEMA entropy TO gcp_user;
GRANT ALL PRIVILEGES ON SCHEMA coherence TO gcp_user;
GRANT ALL PRIVILEGES ON SCHEMA metrics TO gcp_user;
GRANT ALL PRIVILEGES ON SCHEMA events TO gcp_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA entropy TO gcp_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA coherence TO gcp_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA metrics TO gcp_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA events TO gcp_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA entropy TO gcp_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA coherence TO gcp_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA metrics TO gcp_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA events TO gcp_user;
