# Data model v1 (proposal)
This document describes the initial data model for the traffic forecasting platform and notes for local/dev and Google Cloud deployment.
## Relational metadata (Postgres/Timescale)
- Table `nodes` (static topology / metadata)
- node_id TEXT PRIMARY KEY
- lat DOUBLE PRECISION
- lon DOUBLE PRECISION
- road_name TEXT
- road_type TEXT
- attributes JSONB
- Table `events`
- event_id TEXT PRIMARY KEY
- title TEXT
- category TEXT
- start_time TIMESTAMPTZ
- end_time TIMESTAMPTZ
- venue_lat DOUBLE PRECISION
- venue_lon DOUBLE PRECISION
- source TEXT
- metadata JSONB
- Table `forecasts`
- id SERIAL PRIMARY KEY
- node_id TEXT
- ts_generated TIMESTAMPTZ
- horizon_min INT
- speed_kmh_pred FLOAT
- flow_pred INT
- congestion_pred INT
- model_name TEXT
- meta JSONB
Indexes: (node_id, ts_generated) on forecasts; GiST index on (venue_lon, venue_lat) for events if PostGIS is used.
## Timeseries / feature store
- Raw snapshots: Parquet files partitioned by date (stored on GCS/S3 or locally in `data/parquet/`). Columns: node_id, ts, avg_speed_kmh, vehicle_count, raw_json.
- Feature table: per-node per-ts feature vector stored in Postgres (or feature store) with versioning.
## Local dev setup
- `docker-compose.yml` spins up app + Postgres (see repo root). Use DATABASE_URL env to connect.
- Run `python models/db.py` to initialize (or use alembic migration).
## Google Cloud deployment (recommended path)
1. Containerize app (Dockerfile provided). Build images and push to Artifact Registry.
2. Cloud SQL (Postgres) for metadata; use private IP or Cloud SQL Auth Proxy in Cloud Run.
3. Store raw Parquet time-series on GCS buckets.
4. Use Cloud Run for app serving (stateless), Cloud Scheduler / Cloud Tasks for periodic collectors, or a small GKE/Cloud Run job for crawling.
5. For heavy training jobs, use AI Platform / Vertex AI with GPUs and GCS / BigQuery as data sources.
## Notes
- Secure API keys and DB credentials with Secret Manager.
- Use Pub/Sub for event-driven ingestion (optionally) between collectors and feature pipeline.
