# Internal Data & Modeling Readiness Report (IR-05)

**Date:** 2025-10-25
**Author:** Le Quang That (THAT Le Quang) – SE183256
**Audience:** Cloud deployment and data engineering teammates  
**Scope:** Data structure, collection pipeline, model inventory, and key metrics for the Academic v4.0 stack.

---

## 1. Data Structure Overview

### 1.1 Logical Architecture

- **Relational metadata (Postgres/Timescale-ready)**:
  - `nodes(node_id, lat, lon, road_name, road_type, attributes)` — immutable topology metadata for each monitored intersection.
  - `events(event_id, title, category, start_time, end_time, venue_lat, venue_lon, source, metadata)` — optional traffic-impacting events.
  - `forecasts(id, node_id, ts_generated, horizon_min, speed_kmh_pred, flow_pred, congestion_pred, model_name, meta)` — prediction history; indexes on `(node_id, ts_generated)` keep lookups fast.
- **Time-series store**:
  - Raw snapshots saved as Parquet partitions (`node_id`, `ts`, `avg_speed_kmh`, `vehicle_count`, `temperature_c`, `rain_mm`, `wind_speed_kmh`, `raw_json`).
  - Enhanced feature tables (~60 engineered columns) land in `data/processed/` along with scaler objects (`models/feature_scaler.pkl`) and run metadata (`data/processed/metadata.json`).
- **Traffic history cache** — SQLite database `data/traffic_history.db` with `traffic_snapshots(timestamp, node_id, avg_speed_kmh, congestion_level, temperature_c, rain_mm, wind_speed_kmh, data_json)` plus indexes on `timestamp` and `(node_id, timestamp)`; default retention 7 days with rolling cleanup.

### 1.2 Repository Data Layout

- `data/` — latest JSON exports, SQLite history, feature CSV outputs.
- `data/parquet/` — partitioned raw timeseries (node_id, ts, avg_speed_kmh, vehicle_count, raw_json).
- `data/processed/` — train/validation splits, scalers, and metadata (`metadata.json`).
- `models/` — serialized estimators (`*.pkl`, `*.keras`), feature scaler, research artifacts.
  - `models/research/` – experimental deep learning artefacts; now includes ASTGCN saves (`*_config.pkl`, `*_adjacency.npy`, `.keras`).
- `cache/` — API cache (Overpass/Open-Meteo/Google) governed by TTLs in `configs/project_config.yaml`.

### 1.3 Configuration Sources

- `configs/project_config.yaml` — governs global area of interest, adaptive scheduler slots, collector options (Overpass, Google Directions, Open-Meteo), feature pipeline toggles, and default model registry entry.
- `.env` (from `.env.template`) — holds secrets for Google API keys, database URLs, and environment-specific overrides.
- `configs/nodes_schema_v2.json` — JSON schema used by validation layer to guarantee node payload shape before storage.

---

## 2. Data Collection Method & Platform

### 2.1 Collection Workflow

1. **Scheduler** — Adaptive by default (`scheduler.mode=adaptive`), implemented in `traffic_forecast/scheduler/adaptive_scheduler.py`:
   - Peak windows (06:30-07:30, 10:30-11:30, 13:00-13:30, 16:30-19:30) at 30-minute cadence.
   - Off-peak at 60 minutes; weekends optionally throttled to 90 minutes.
   - CLI entrypoint: `python scripts/collect_and_render.py --adaptive` (or `--print-schedule` for cost preview).
2. **Collectors** — orchestrated via `scripts/collect_and_render.py` / `scripts/collect_with_history.py`:
   - **Overpass** (`traffic_forecast/collectors/overpass/collector.py`) retrieves OSM topology, filtered by `node_selection` config (min_degree 6, min_importance 40, motorway/trunk/primary only).
   - **Google Directions** (`traffic_forecast/collectors/google/collector.py`) samples k=3 nearest edges per node within 1.024 km radius. Development uses mock responses (`use_mock_api: true`); production toggles to real API with cost envelope ~$720/month for 64 nodes.
   - **Open-Meteo** (`traffic_forecast/collectors/open_meteo/collector.py`) enriches current and short-horizon weather features (t+5 to t+60 minutes).
3. **Post-processing** — `traffic_forecast/pipelines` modules construct temporal, lag, and spatial features according to the YAML pipeline definitions. Traffic history DB supports lag queries to avoid recomputation.
4. **Storage & Manifest** — each run writes a manifest under `data_runs/` (configurable via `globals.output_base`) and, when history capture is enabled, appends to `traffic_history.db` plus feature CSV outputs.

### 2.2 Platform & Deployment Hooks

- **Local / CI** — Conda environment `dsp` (`environment.yml`), runnable via helper scripts (`scripts/quick_start.sh`). Mock API keeps costs at zero.
- **GVM / Cloud** — Production setup uses the provided shell scripts to configure a Google Cloud VM, install dependencies, and register system services:
  - Cloud SQL or TimescaleDB hosts relational layer.
  - Raw Parquet and model artefacts stored on GCS buckets.
  - Scheduling handled by systemd timer, cron, or Cloud Scheduler invoking `collect_and_render.py`.
  - `scripts/start_collection.sh` provides one-command bootstrap in production; health checks available via `scripts/health_check.sh`.
- **Retention & Maintenance** — `scripts/cleanup_runs.py --days 14` prunes historical runs; `TrafficHistoryStore.cleanup_old_data` enforces rolling SQLite retention.

---

## 3. Model Portfolio & Metrics

### 3.1 Feature Pipeline

- **Lag features** — 5, 15, 30, and 60 minute shifts plus rolling means, max/min, and variance; leverage traffic history storage to avoid recomputation.
- **Temporal features** — sine/cosine encodings for hour and day-of-week, rush-hour flags (morning, lunch, evening), weekend and holiday markers.
- **Spatial features** — neighbor averages, min/max, standard deviation, congestion counts derived from the Overpass adjacency graph.
- All feature groups can be enabled/disabled or retuned through `pipelines.preprocess` in `project_config.yaml`.

### 3.2 Production-Proven Models

| Model Family                      | Usage                             | Notes                                                       | Latest Metrics                       |
| --------------------------------- | --------------------------------- | ----------------------------------------------------------- | ------------------------------------ |
| Linear Regression                 | Baseline                          | Fast inference; default `pipelines.model.type`              | RMSE 8.2 km/h, MAE 6.1 km/h, R² 0.89 |
| Ridge / Lasso                     | Regularized baselines             | Available via registry                                      | Cross-val RMSE ≈ 8.5 ± 0.3           |
| Random Forest / Gradient Boosting | Tree ensembles                    | Used in stacking ensemble                                   | Contribute to final RMSE 8.2 km/h    |
| XGBoost                           | High-bias/high-variance component | Weight 0.45 in ensemble                                     | Improves non-linear capture          |
| Stacking Ensemble                 | Production best                   | Combines XGBoost, RF, GB                                    | RMSE 8.2 km/h, MAPE 12.5%            |
| LSTM (attention)                  | Deep temporal model               | `traffic_forecast/models/lstm_model.py`, 12-timestep window | RMSE ~8.2-8.5 km/h on validation     |

### 3.3 Research & Experimental Models

- **ASTGCN (Attention-based Spatial-Temporal GCN)** — three-component network (recent, daily, weekly) with temporal and spatial attention modules, Chebyshev graph convolutions, and learnable fusion; ships with helpers for tensor preparation, training, evaluation, and artifact export (`.keras`, `_config.pkl`, `_adjacency.npy`).
- **Feature Engineering R&D** — continuing experiments around lag tuning, temporal flags, and neighbor propagation metrics feed back into the pipeline configuration.

### 3.4 Experiment Tracking & Deployment

- **MLflow** support baked into training utilities (`traffic_forecast/models/advanced_training.py`).
- Model registry in `traffic_forecast/models/registry.py` exposes standardized builder interface (linear, RF, GB, etc.).
- Serialized models saved under `models/` and referenced by API service (`traffic_forecast/api/main.py`) and CLI tooling.
- Weekly retraining cadence recommended; model metadata recorded in `models/model_metadata.json` with scaler parity.

---

**Questions or action items?** Ping the data platform channel before enabling real API keys or modifying scheduler cadence.
