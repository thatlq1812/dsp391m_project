# Feature Engineering Guide
Complete guide for the enhanced feature engineering pipeline (v3.1).
## Overview
The feature engineering pipeline transforms raw traffic data (23 base features) into enriched data (~60 features) for improved forecasting accuracy.
**Expected improvement**: RMSE 8.2 → 5.5-6.0 km/h (27-33% reduction)
## Architecture
```
Raw Traffic Data (23 features)
↓
Feature Engineering Pipeline 
1. Lag Features (~30) 
2. Temporal Features (~18) 
3. Spatial Features (~9) 
↓
Enhanced Data (~60 features)
```
## Feature Groups
### 1. Base Features (23)
Already present in raw data:
**Current Traffic State**:
- `avg_speed_kmh`: Average speed
- `congestion_level`: 0-3 scale
**Weather Current**:
- `temperature_c`, `rain_mm`, `wind_speed_kmh`
- `cloud_cover_pct`, `visibility_km`, `pressure_mb`
**Weather Forecasts** (4 horizons: t+5, t+15, t+30, t+60):
- `forecast_temp_t{5,15,30,60}_c`
- `forecast_rain_t{5,15,30,60}_mm`
- `forecast_wind_t{5,15,30,60}_kmh`
### 2. Lag Features (~30)
Traffic history patterns - **HIGHEST IMPORTANCE**
**Simple Lags** (correlation 0.70-0.85):
```python
speed_lag_5min # Speed 5 minutes ago
speed_lag_15min # Speed 15 minutes ago
speed_lag_30min # Speed 30 minutes ago
speed_lag_60min # Speed 1 hour ago
congestion_lag_5min
congestion_lag_15min
congestion_lag_30min
congestion_lag_60min
```
**Rolling Aggregations** (correlation 0.75-0.90):
```python
# 15-minute window
speed_rolling_15min_mean
speed_rolling_15min_std
speed_rolling_15min_min
speed_rolling_15min_max
# 30-minute window
speed_rolling_30min_mean
speed_rolling_30min_std
speed_rolling_30min_min
speed_rolling_30min_max
# 60-minute window
speed_rolling_60min_mean
speed_rolling_60min_std
speed_rolling_60min_min
speed_rolling_60min_max
```
**Change Features**:
```python
speed_change_5min # Absolute change in last 5 min
speed_pct_change_5min # Percentage change
speed_acceleration_5min # Rate of change
congestion_change_5min # Change in congestion level
```
### 3. Temporal Features (~18)
Time-based patterns
**Cyclical Encoding** (preserves circular nature):
```python
hour_sin, hour_cos # Hour 23 and 0 are adjacent
day_of_week_sin, day_of_week_cos
month_sin, month_cos
```
**Rush Hour Flags**:
```python
is_morning_rush # 7-9 AM
is_evening_rush # 5-7 PM
is_rush_hour # Combined
is_lunch_time # 11 AM - 1 PM
is_off_peak # Not rush hour
```
**Day Type**:
```python
is_weekend, is_weekday
is_friday, is_monday
```
**Holidays** (optional):
```python
is_holiday
is_pre_holiday
days_to_next_holiday
```
### 4. Spatial Features (~9)
Network-based patterns
**Neighbor Aggregations**:
```python
neighbor_avg_avg_speed_kmh # Average of neighbor speeds
neighbor_min_avg_speed_kmh
neighbor_max_avg_speed_kmh
neighbor_std_avg_speed_kmh
neighbor_count # Number of neighbors
neighbor_speed_diff # Self - neighbor_avg
neighbor_is_bottleneck # True if 5+ km/h slower
```
**Congestion Propagation**:
```python
neighbor_congested_count # Count of congested neighbors
neighbor_congested_fraction # Fraction congested
```
## Usage
### Quick Start
```python
from traffic_forecast.features.pipeline import create_all_features
import pandas as pd
import json
# Load data
df = pd.read_csv('data/features_nodes_v2.csv')
with open('data/nodes.json', 'r') as f:
nodes = json.load(f)
# Generate all features
df_enhanced = create_all_features(df, nodes)
print(f"Input: {len(df.columns)} columns")
print(f"Output: {len(df_enhanced.columns)} columns")
```
### Using Pipeline Class
```python
from traffic_forecast.features.pipeline import FeatureEngineeringPipeline
# Initialize with config
pipeline = FeatureEngineeringPipeline()
# Step-by-step
df1 = pipeline.create_lag_features(df)
df2 = pipeline.create_temporal_features(df1)
df3 = pipeline.create_spatial_features(df2, nodes)
# Or all at once
df_enhanced = pipeline.create_all_features(df, nodes)
# Validate
report = pipeline.validate_features(df_enhanced)
print(report)
```
### Command Line
```bash
# Generate enhanced features from existing data
python scripts/generate_enhanced_features.py \
--input data/features_nodes_v2.json \
--output data/features_nodes_v3.json \
--nodes data/nodes.json
# Skip spatial features (faster)
python scripts/generate_enhanced_features.py \
--no-spatial
```
### Testing
```bash
# Run test with synthetic data
python tests/test_feature_pipeline.py
# Expected output:
# - Creates 10 nodes, 100 timesteps
# - Generates ~60 features
# - Saves test_features_output.csv
```
## Configuration
Edit `configs/project_config.yaml`:
```yaml
pipelines:
# Lag features config
lag_config:
lag_minutes: [5, 15, 30, 60]
rolling_windows: [3, 6, 12] # 15, 30, 60 minutes
# Temporal features config
temporal_config:
cyclical_encoding: true
rush_hours:
morning: [7, 9]
evening: [17, 19]
lunch: [11, 13]
# Spatial features config
spatial_config:
enabled: true
neighbor_hops: 1 # Direct neighbors only
```
## Feature Importance
Based on analysis (see `TEMPORAL_FEATURES_ANALYSIS.md`):
### Critical Features (correlation 0.70-0.90)
1. **Lag Features**: `speed_lag_*`, `speed_rolling_*_mean`
- Direct history is best predictor
- Expected: 15-20% RMSE improvement
2. **Rush Hour Flags**: `is_rush_hour`, `is_morning_rush`
- Clear traffic patterns
- Expected: 5-8% improvement
3. **Neighbor Speed**: `neighbor_avg_speed`
- Congestion spreads through network
- Expected: 3-5% improvement
### Important Features (correlation 0.50-0.70)
4. **Weather Forecasts**: `forecast_temp_t*`, `forecast_rain_t*`
- Already implemented
- Current: ~3% contribution
5. **Temporal Encoding**: `hour_sin/cos`, `day_of_week_sin/cos`
- Better than raw hour/day
- Expected: 2-3% improvement
### Moderate Features (correlation 0.30-0.50)
6. **Speed Changes**: `speed_change_5min`, `speed_pct_change_5min`
- Detects acceleration/deceleration
- Expected: 1-2% improvement
7. **Weekend Flags**: `is_weekend`, `is_weekday`
- Different patterns weekday/weekend
- Expected: 1-2% improvement
## Implementation Details
### Lag Features Module
File: `traffic_forecast/features/lag_features.py`
```python
def create_all_lag_features(df, config, group_by='node_id'):
"""
Creates lag features grouped by node.
Steps:
1. Sort by node_id, ts
2. Group by node_id
3. Create lags (shift)
4. Create rolling (window)
5. Create changes (diff)
"""
```
**Key points**:
- Groups by `node_id` (each node has independent history)
- Requires sorted data (`node_id`, `ts`)
- Lags create NaN for first N rows (removed in final data)
- Rolling windows need enough history (>=12 timesteps)
### Temporal Features Module
File: `traffic_forecast/features/temporal_features.py`
```python
def add_temporal_features(df, config, ts_column='ts'):
"""
Adds time-based encoding.
Cyclical encoding example:
hour_sin = sin(2π × hour / 24)
hour_cos = cos(2π × hour / 24)
Why? Hour 23 and Hour 0 are adjacent (not 23 apart!)
"""
```
**Key points**:
- Cyclical encoding preserves circular nature of time
- Rush hours defined for HCMC (configurable)
- Holiday list needs updating per year
### Spatial Features Module
File: `traffic_forecast/features/spatial_features.py`
```python
def add_spatial_features(df, nodes_data, config):
"""
Adds network-based features.
Steps:
1. Build adjacency graph from nodes
2. For each node at each timestamp:
- Find neighbors
- Aggregate neighbor values
3. Compute propagation metrics
"""
```
**Key points**:
- Builds graph from `connected_ways` in nodes data
- Can use k-hop neighbors (default k=1)
- Computationally expensive (O(N×T×D) where D=avg degree)
- Can be disabled if slow
## Performance Impact
### Before (v3.0)
- Features: 23
- RMSE: 8.2 km/h
- R²: 0.89
- Training time: ~2 minutes
### After (v3.1) - Expected
- Features: ~60
- RMSE: 5.5-6.0 km/h (27-33% better)
- R²: 0.93-0.94
- Training time: ~3-4 minutes
### Breakdown
```
Lag features: -15 to -20% RMSE (CRITICAL)
Rush hour flags: -5 to -8% RMSE
Neighbor speed: -3 to -5% RMSE
Temporal encoding: -2 to -3% RMSE
Others: -2 to -3% RMSE
Combined: -27 to -33% RMSE
```
## Common Issues
### 1. Too Many NaN Values
**Problem**: Lag features create NaN for first N rows.
**Solution**:
```python
# Remove rows with NaN
df_enhanced = df_enhanced.dropna()
# Or fill with forward-fill
df_enhanced = df_enhanced.fillna(method='ffill')
```
### 2. Slow Spatial Features
**Problem**: Spatial features take too long on large datasets.
**Solution**:
```yaml
# In project_config.yaml
spatial_config:
enabled: false # Disable temporarily
```
Or use command line:
```bash
python scripts/generate_enhanced_features.py --no-spatial
```
### 3. Memory Issues
**Problem**: 60 features × millions of rows = OOM
**Solution**:
```python
# Process in chunks
chunk_size = 10000
for i in range(0, len(df), chunk_size):
chunk = df[i:i+chunk_size]
chunk_enhanced = pipeline.create_all_features(chunk, nodes)
# Save chunk
```
### 4. Inconsistent Node IDs
**Problem**: Nodes in traffic data don't match nodes.json
**Solution**:
```python
# Check node coverage
traffic_nodes = set(df['node_id'].unique())
graph_nodes = set(n['node_id'] for n in nodes)
missing = traffic_nodes - graph_nodes
print(f"Missing from graph: {missing}")
# Filter to common nodes
common = traffic_nodes & graph_nodes
df_filtered = df[df['node_id'].isin(common)]
```
## Next Steps
### 1. Generate Enhanced Features
```bash
# For existing data
python scripts/generate_enhanced_features.py
# Output: data/features_nodes_v3.json
```
### 2. Re-train Models
```bash
# With new features
python traffic_forecast/cli/train.py \
--input data/features_nodes_v3.csv \
--output models/v3.1/
```
### 3. Compare Performance
```python
# v3.0 (old)
model_v30 = joblib.load('models/v3.0/lstm_model.pkl')
rmse_v30 = evaluate(model_v30, X_test_v30, y_test)
# v3.1 (new)
model_v31 = joblib.load('models/v3.1/lstm_model.pkl')
rmse_v31 = evaluate(model_v31, X_test_v31, y_test)
improvement = (rmse_v30 - rmse_v31) / rmse_v30 * 100
print(f"RMSE improved by {improvement:.1f}%")
```
### 4. Update Data Collection
```python
# In traffic_forecast/pipelines/data_pipeline.py
from traffic_forecast.features.pipeline import create_all_features
# After collecting raw data
df_raw = collect_traffic_data()
nodes = load_nodes()
# Apply feature engineering
df_enhanced = create_all_features(df_raw, nodes)
# Continue with prediction
predictions = model.predict(df_enhanced)
```
## References
- **Implementation**: `traffic_forecast/features/`
- **Configuration**: `configs/project_config.yaml`
- **Testing**: `tests/test_feature_pipeline.py`
- **Analysis**: `doc/reference/TEMPORAL_FEATURES_ANALYSIS.md`
- **Examples**: `scripts/generate_enhanced_features.py`
## Summary
The feature engineering pipeline v3.1 adds ~37 new features to the existing 23 base features:
| Module | Features | Importance | Impact |
| -------- | -------- | ---------- | ---------------- |
| Lag | 30 | Critical | -15 to -20% RMSE |
| Temporal | 18 | High | -7 to -11% RMSE |
| Spatial | 9 | Medium | -3 to -5% RMSE |
**Total expected improvement**: 27-33% RMSE reduction (8.2 → 5.5-6.0 km/h)
**Ready to use**: All modules implemented and tested!
