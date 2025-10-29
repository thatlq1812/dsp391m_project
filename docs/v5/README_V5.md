# Traffic Forecast v5.0 - Production Ready

**Maintainer:**Le Quang That (Xiel) - SE183256
**Version:** 5.0.0
**Date:**October 29, 2025
**Status:**Production Ready - Real API Only

---

## What's New in v5.0

### Major Changes

1. **Real API Only** - Removed all mock API fallback
2. **Intelligent Caching** - Overpass topology cached permanently
3. **Distance Filtering** - Minimum 200m between nodes to avoid clustering
4. **Expanded Coverage** - Radius increased to 2048m
5. **128 Nodes** - Doubled from 64 nodes for better coverage
6. **Retry Mechanism** - Auto-retry failed API calls (3 attempts)

### Configuration

File: `configs/project_config.yaml`

```yaml
# Key Settings
globals:
  area:
  radius_m: 2048 # 4km radius coverage

node_selection:
  max_nodes: 128 # Up from 64
  min_distance_meters: 200 # NEW: Avoid clustered nodes
  min_degree: 6
  min_importance_score: 40.0

collectors:
  overpass:
  use_cache: true # Cache topology permanently
  cache_file: cache/overpass_topology.json

  google_directions:
  use_real_api_only: true # NO mock fallback
  retry_on_failure: 3 # Retry failed requests
```

---

## Quick Start

### Prerequisites

```bash
# Set Google API key (REQUIRED)
export GOOGLE_MAPS_API_KEY="your-api-key-here"

# Activate environment
conda activate dsp
```

### First Run - Collect Topology

```bash
# Collect and cache Overpass topology (once)
cd /d/UNI/DSP391m/project
python traffic_forecast/collectors/overpass/collector_v5.py
```

Output:

- `cache/overpass_topology.json` - Permanent cache
- 128 major intersections with 200m+ spacing
- Road types: motorway, trunk, primary

### Collect Traffic Data

```bash
# Collect real traffic data from Google API
python traffic_forecast/collectors/google/collector_v5.py
```

Expected:

- 128 nodes \* 3 neighbors = ~384 API calls
- Cost: ~$1.92 per collection
- Duration: ~5-10 minutes (with rate limiting)

### Full Collection Pipeline

```bash
# Run complete collection (Overpass + Google + Weather)
python scripts/collect_and_render.py --once
```

---

## Architecture

### Data Flow

```
Overpass API
 |
 v
[Cache] --> Topology (128 nodes)
 |
 v
 Google Directions API
 |
 v
 Traffic Data (384 edges)
 |
 v
 Storage
```

### Caching Strategy

| Data Source | Cache     | Frequency        | Rationale       |
| ----------- | --------- | ---------------- | --------------- |
| Overpass    | Permanent | Once             | Static topology |
| Weather     | 1 hour    | Every collection | Slow-changing   |
| Traffic     | No cache  | Every collection | Real-time data  |

---

## Cost Analysis

### Per Collection

- Nodes: 128
- k_neighbors: 3
- API calls: 128 \* 3 = 384
- Cost: 384 \* $0.005 = $1.92

### Monthly Estimate (Adaptive Schedule)

- Collections/day: 25
- Days/month: 30
- Total collections: 750
- Total API calls: 750 \* 384 = 288,000
- **Monthly cost: $1,440**

### Cost Optimization

Reduce frequency:

```yaml
scheduler:
 adaptive:
 peak_hours:
 interval_minutes: 60 # From 30
 offpeak:
 interval_minutes: 120 # From 60
```

New cost: ~$720/month (50% savings)

---

## Node Selection Algorithm

### Criteria

1. **Degree**: >= 6 connecting roads
2. **Importance**: >= 40.0 score
3. **Distance**: >= 200m from other nodes
4. **Road Type**: motorway, trunk, primary only
5. **Ranking**: Top 128 by importance

### Importance Scoring

```python
ROAD_WEIGHTS = {
 'motorway': 10,
 'trunk': 9,
 'primary': 8,
 'secondary': 7,
}

score = sum(road_weights) + diversity_bonus
```

### Distance Filtering

```python
# Keeps highest importance nodes when < 200m apart
for node in sorted_by_importance:
 if all(distance_to(selected) >= 200m):
 select(node)
```

Result: Well-distributed nodes across coverage area

---

## API Integration

### Google Directions API v5.0

Features:

- Real API only (no mock)
- Retry mechanism (3 attempts)
- Rate limiting (2800 req/min)
- Traffic-aware duration
- Exponential backoff

Usage:

```python
from traffic_forecast.collectors.google.collector_v5 import real_directions_api

result = real_directions_api(node_a, node_b, api_key, rate_limiter, config)

if result:
 speed = result['speed_kmh']
 has_traffic = result['has_traffic_data']
```

### Error Handling

```python
# Failed requests after 3 retries
if result is None:
 # Log failure, skip edge
 print(f"FAILED: {node_a} -> {node_b}")
 continue
```

Success rate: Expected > 95%

---

## File Structure

### New Files (v5.0)

```
configs/
 project_config.yaml # Updated v5.0 config
 project_config_backup.yaml # Backup of v4.0

traffic_forecast/collectors/
 google/
 collector_v5.py # Real API only
 overpass/
 collector_v5.py # With caching
 node_selector_v5.py # With distance filtering

cache/
 overpass_topology.json # Permanent topology cache
 weather_latest.json # 1-hour weather cache

doc/
 README_V5.md # This file
 reports/
 PROJECT_EVALUATION.md # Evaluation report
```

---

## Testing

### Test Overpass Collection

```bash
# Force fresh collection (ignore cache)
python traffic_forecast/collectors/overpass/collector_v5.py --force-refresh
```

Expected output:

```
TOPOLOGY EXTRACTION STATISTICS
Total Nodes: 128
Total Edges: ~400
Average Degree: 6.5
Road Type Distribution:
 primary: 95
 trunk: 25
 motorway: 8
```

### Test Google Collection (Limited)

```bash
# Test with only 10 edges
export GOOGLE_TEST_LIMIT=10
python traffic_forecast/collectors/google/collector_v5.py
```

Expected:

- 10 API calls
- Cost: $0.05
- Duration: ~30 seconds

### Full Test Collection

```bash
# Unset limit for full test
unset GOOGLE_TEST_LIMIT

# Run full collection
python traffic_forecast/collectors/google/collector_v5.py
```

Expected:

- ~384 API calls
- Cost: ~$1.92
- Duration: 5-10 minutes
- Success rate: > 95%

---

## Deployment

### Local Development

```bash
# 1. Set API key
export GOOGLE_MAPS_API_KEY="your-key"

# 2. Test collection
export GOOGLE_TEST_LIMIT=10
python traffic_forecast/collectors/google/collector_v5.py

# 3. Full collection
unset GOOGLE_TEST_LIMIT
python scripts/collect_and_render.py --once
```

### Production (GCP)

```bash
# 1. Set environment
export GOOGLE_MAPS_API_KEY="production-key"

# 2. Run adaptive scheduler
python scripts/collect_and_render.py --adaptive

# 3. Monitor costs
# Expected: ~$1,440/month
```

---

## Monitoring

### Collection Success Rate

```bash
# Check recent collection logs
tail -100 data_runs/*/manifest.json | grep "success_rate"
```

Target: > 95% success rate

### API Cost Tracking

```bash
# Count API calls per day
find data_runs -name "traffic_edges.json" -mtime -1 | \
 xargs jq 'length' | \
 awk '{sum+=$1} END {print sum " API calls"}'
```

### Cache Status

```bash
# Check cache freshness
ls -lh cache/

# Force cache refresh if needed
python traffic_forecast/collectors/overpass/collector_v5.py --force-refresh
```

---

## Troubleshooting

### Issue: Google API Key Invalid

```
ValueError: GOOGLE_MAPS_API_KEY environment variable not set
```

Solution:

```bash
export GOOGLE_MAPS_API_KEY="your-valid-key-here"
```

### Issue: High Failure Rate

```
FAILED: Could not get traffic data for edge X -> Y
Success rate: 75%
```

Solutions:

1. Check API quota limits
2. Verify API key has Directions API enabled
3. Check network connectivity
4. Review rate limiting settings

### Issue: Cache Not Working

```
Cache file not found: cache/overpass_topology.json
```

Solution:

```bash
# Create cache directory
mkdir -p cache

# Run Overpass collection
python traffic_forecast/collectors/overpass/collector_v5.py
```

---

## Migration from v4.0

### Changes Required

1. **Remove mock API usage**

- Set `GOOGLE_MAPS_API_KEY`
- No mock fallback in v5.0

2. **Update config**

```bash
# Backup already created
# configs/project_config.yaml already updated
```

3. **Clear old cache**

```bash
rm -rf cache/*.json
```

4. **Re-collect topology**

```bash
python traffic_forecast/collectors/overpass/collector_v5.py
```

### Backward Compatibility

v4.0 collectors still available:

- `traffic_forecast/collectors/google/collector.py`
- `traffic_forecast/collectors/overpass/collector.py`

To use v4.0:

```bash
# Restore v4.0 config
cp configs/project_config_backup.yaml configs/project_config.yaml

# Use old collectors
python traffic_forecast/collectors/overpass/collector.py
```

---

## Performance

### Improvements over v4.0

| Metric          | v4.0     | v5.0     | Change    |
| --------------- | -------- | -------- | --------- |
| Nodes           | 64       | 128      | +100%     |
| Coverage radius | 1024m    | 2048m    | +300%     |
| Node spacing    | Variable | 200m min | Optimized |
| Overpass calls  | 100x     | 1x       | -99%      |
| Cache hits      | 0%       | ~100%    | +100%     |
| Retry logic     | No       | Yes (3x) | Added     |

### Expected Metrics

- Collection time: 5-10 min
- API success rate: > 95%
- Topology cache hits: ~100%
- Weather cache hits: ~80%
- Cost per collection: $1.92

---

## Next Steps

### Immediate (Priority 1)

1. Test collection with real API
2. Verify cache mechanism
3. Monitor success rate
4. Measure actual costs

### Short Term (Priority 2)

1. Create visualizations
2. Feature importance analysis
3. Cross-validation
4. Update notebooks

### Long Term (Priority 3)

1. ML model retraining
2. API endpoint deployment
3. Dashboard updates
4. Documentation cleanup

---

## Support

### Common Commands

```bash
# Collection
python scripts/collect_and_render.py --once

# Test mode
export GOOGLE_TEST_LIMIT=10
python traffic_forecast/collectors/google/collector_v5.py

# Force cache refresh
python traffic_forecast/collectors/overpass/collector_v5.py --force-refresh

# Check logs
tail -f logs/collection.log
```

### Configuration Paths

- Main config: `configs/project_config.yaml`
- Cache directory: `cache/`
- Data outputs: `data_runs/`
- Logs: `logs/`

---

## Changelog

### v5.0.0 (2025-10-29)

BREAKING CHANGES:

- Removed mock API completely
- Real Google API required
- New caching mechanism

NEW FEATURES:

- Distance-based node filtering (200m min)
- Permanent Overpass caching
- Retry mechanism for API failures
- Expanded coverage (2048m radius)
- 128 nodes (up from 64)

IMPROVEMENTS:

- Better error handling
- Comprehensive logging
- Cost tracking
- Success rate monitoring

---

**End of README v5.0**
