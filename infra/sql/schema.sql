-- Database schema for traffic forecasting platform

CREATE TABLE nodes (
    node_id TEXT PRIMARY KEY,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    road_name TEXT,
    road_type TEXT,
    attributes JSONB
);

CREATE TABLE forecasts (
    id SERIAL PRIMARY KEY,
    node_id TEXT REFERENCES nodes(node_id),
    ts_generated TIMESTAMPTZ DEFAULT NOW(),
    horizon_min INT,
    speed_kmh_pred FLOAT,
    flow_pred INT,
    congestion_pred INT,
    model_name TEXT,
    meta JSONB
);

CREATE TABLE events (
    event_id TEXT PRIMARY KEY,
    title TEXT,
    category TEXT,
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    venue_lat DOUBLE PRECISION,
    venue_lon DOUBLE PRECISION,
    source TEXT,
    metadata JSONB
);

CREATE TABLE ingest_log (
    id SERIAL PRIMARY KEY,
    source TEXT,
    ts_ingested TIMESTAMPTZ DEFAULT NOW(),
    records_count INT,
    status TEXT
);

-- Indexes
CREATE INDEX idx_forecasts_node_ts ON forecasts(node_id, ts_generated);
CREATE INDEX idx_events_venue ON events USING GIST (point(venue_lon, venue_lat));