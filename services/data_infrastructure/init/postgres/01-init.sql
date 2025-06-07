-- PostgreSQL Initialization Script for GCP Project
-- This script creates the necessary databases, schemas, tables, and indexes for the relational data

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "postgis";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS camera_management;
CREATE SCHEMA IF NOT EXISTS experiments;
CREATE SCHEMA IF NOT EXISTS users;

-- Set search path
SET search_path TO camera_management, experiments, users, public;

-- Create camera_management tables
CREATE TABLE IF NOT EXISTS camera_management.cameras (
    id SERIAL PRIMARY KEY,
    url TEXT NOT NULL UNIQUE,
    source TEXT NOT NULL,
    discovered_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status VARCHAR(20) NOT NULL,
    fps FLOAT,
    resolution_width INTEGER,
    resolution_height INTEGER,
    latency_ms INTEGER,
    entropy_quality FLOAT,
    geo_lat FLOAT,
    geo_lon FLOAT,
    geo_city TEXT,
    geo_country TEXT,
    geo_continent TEXT,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS camera_management.validation_results (
    id SERIAL PRIMARY KEY,
    camera_id INTEGER NOT NULL REFERENCES camera_management.cameras(id) ON DELETE CASCADE,
    validated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    connection_successful BOOLEAN NOT NULL,
    connection_time_ms INTEGER,
    measured_fps FLOAT,
    frame_width INTEGER,
    frame_height INTEGER,
    stream_latency_ms INTEGER,
    entropy_score FLOAT,
    stability_score FLOAT,
    overall_quality FLOAT,
    error_type VARCHAR(100),
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS camera_management.health_checks (
    id SERIAL PRIMARY KEY,
    camera_id INTEGER NOT NULL REFERENCES camera_management.cameras(id) ON DELETE CASCADE,
    checked_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    is_available BOOLEAN NOT NULL,
    response_time_ms INTEGER,
    error_type VARCHAR(100),
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS camera_management.geographic_distribution (
    id SERIAL PRIMARY KEY,
    region_type VARCHAR(20) NOT NULL,
    region_name VARCHAR(100) NOT NULL,
    camera_count INTEGER NOT NULL DEFAULT 0,
    active_camera_count INTEGER NOT NULL DEFAULT 0,
    percentage FLOAT NOT NULL DEFAULT 0.0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS camera_management.discovery_jobs (
    id SERIAL PRIMARY KEY,
    source VARCHAR(50) NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    total_discovered INTEGER DEFAULT 0,
    new_cameras INTEGER DEFAULT 0,
    updated_cameras INTEGER DEFAULT 0,
    failed_urls INTEGER DEFAULT 0,
    status VARCHAR(20) NOT NULL DEFAULT 'running',
    error_message TEXT
);

-- Create experiments tables
CREATE TABLE IF NOT EXISTS experiments.experiments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name TEXT NOT NULL,
    start_ts TIMESTAMPTZ NOT NULL,
    end_ts TIMESTAMPTZ,
    region GEOGRAPHY,
    metric TEXT NOT NULL,
    target_delta DOUBLE PRECISION NOT NULL,
    p_value_target DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    current_p_value DOUBLE PRECISION,
    current_effect_size DOUBLE PRECISION,
    metadata JSONB
);

CREATE TABLE IF NOT EXISTS experiments.experiment_results (
    id SERIAL PRIMARY KEY,
    experiment_id UUID NOT NULL REFERENCES experiments.experiments(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    p_value DOUBLE PRECISION NOT NULL,
    effect_size DOUBLE PRECISION NOT NULL,
    sample_size INTEGER NOT NULL,
    is_significant BOOLEAN NOT NULL DEFAULT FALSE
);

-- Create users tables
CREATE TABLE IF NOT EXISTS users.users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(100) NOT NULL UNIQUE,
    email VARCHAR(255) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login TIMESTAMPTZ,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_admin BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS users.api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users.users(id) ON DELETE CASCADE,
    key_name VARCHAR(100) NOT NULL,
    key_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ,
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_cameras_status ON camera_management.cameras(status);
CREATE INDEX IF NOT EXISTS idx_cameras_source ON camera_management.cameras(source);
CREATE INDEX IF NOT EXISTS idx_cameras_geo_country ON camera_management.cameras(geo_country);
CREATE INDEX IF NOT EXISTS idx_cameras_geo_continent ON camera_management.cameras(geo_continent);
CREATE INDEX IF NOT EXISTS idx_cameras_entropy_quality ON camera_management.cameras(entropy_quality);
CREATE INDEX IF NOT EXISTS idx_cameras_status_quality ON camera_management.cameras(status, entropy_quality);
CREATE INDEX IF NOT EXISTS idx_cameras_geo ON camera_management.cameras(geo_country, geo_city);
CREATE INDEX IF NOT EXISTS idx_cameras_discovery ON camera_management.cameras(source, discovered_at);

CREATE INDEX IF NOT EXISTS idx_validation_camera_date ON camera_management.validation_results(camera_id, validated_at);
CREATE INDEX IF NOT EXISTS idx_health_camera_date ON camera_management.health_checks(camera_id, checked_at);
CREATE INDEX IF NOT EXISTS idx_geo_dist_unique ON camera_management.geographic_distribution(region_type, region_name);
CREATE INDEX IF NOT EXISTS idx_discovery_source ON camera_management.discovery_jobs(source);
CREATE INDEX IF NOT EXISTS idx_discovery_status ON camera_management.discovery_jobs(status);

CREATE INDEX IF NOT EXISTS idx_experiments_dates ON experiments.experiments(start_ts, end_ts);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments.experiments(status);
CREATE INDEX IF NOT EXISTS idx_experiment_results_experiment ON experiments.experiment_results(experiment_id);

CREATE INDEX IF NOT EXISTS idx_users_username ON users.users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users.users(email);
CREATE INDEX IF NOT EXISTS idx_api_keys_user ON users.api_keys(user_id);

-- Create admin user
INSERT INTO users.users (username, email, password_hash, full_name, is_admin)
VALUES ('admin', 'admin@gcp.org', '$2a$12$1234567890123456789012uGZACxF5WBI6gX5fGm4XSkmxCUhSRQm', 'GCP Admin', TRUE)
ON CONFLICT (username) DO NOTHING;

-- Grant privileges
GRANT ALL PRIVILEGES ON SCHEMA camera_management TO gcp_user;
GRANT ALL PRIVILEGES ON SCHEMA experiments TO gcp_user;
GRANT ALL PRIVILEGES ON SCHEMA users TO gcp_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA camera_management TO gcp_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA experiments TO gcp_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA users TO gcp_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA camera_management TO gcp_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA experiments TO gcp_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA users TO gcp_user;
