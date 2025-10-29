# Data Collection Schema v1.0

**Last Updated**: October 29, 2025  
**Status**: Production-Ready ✅

## Overview

Each collection cycle creates a timestamped directory containing complete traffic network snapshot data from 3 sources: OpenStreetMap topology, Open-Meteo weather, and Google Directions traffic.

## Directory Structure

```
data/
├── runs/
│   └── run_YYYYmmdd_HHMMSS/          # Timestamped collection
│       ├── nodes.json                 # 44 KB - Network nodes (topology)
│       ├── edges.json                 # 315 B - Network edges (topology)
│       ├── statistics.json            # 308 B - Collection metadata
│       ├── weather_snapshot.json      # 18 KB - Weather conditions
│       └── traffic_edges.json         # 290 B - Traffic data per edge
└── cache/
    └── overpass_topology.json         # Permanent topology cache
```

## File Specifications

### 1. `nodes.json` (44 KB)

**Source**: OpenStreetMap via Overpass API  
**Purpose**: Geographic nodes representing intersections and key road points  
**Update Frequency**: Cached permanently (topology rarely changes)

**Schema**:

```json
[
  {
    "node_id": "node-10.760533-106.677134",
    "lat": 10.760533,
    "lon": 106.677134,
    "osmid": 1234567890
  }
]
```

**Typical Size**: 78 nodes × ~580 bytes = 44 KB

---

### 2. `edges.json` (315 B)

**Source**: OpenStreetMap via Overpass API  
**Purpose**: Road segments connecting nodes  
**Update Frequency**: Cached permanently

**Schema**:

```json
[
  {
    "edge_id": "node-10.760533-106.677134-node-10.762243-106.676490",
    "start_node_id": "node-10.760533-106.677134",
    "end_node_id": "node-10.762243-106.676490",
    "distance_m": 202.77,
    "road_type": "secondary",
    "road_name": "Lê Hồng Phong",
    "way_id": 330374257
  }
]
```

**Typical Size**: 234 edges (full network), 1 edge (test mode)

---

### 3. `statistics.json` (308 B)

**Source**: Collection orchestrator  
**Purpose**: Metadata about collection process  
**Update Frequency**: Every collection

**Schema**:

```json
{
  "timestamp": "2025-10-29T13:42:27.123456",
  "run_id": "run_20251029_134227",
  "duration_seconds": 9.1,
  "collectors": {
    "overpass": {
      "status": "success",
      "nodes_count": 78,
      "edges_count": 1,
      "source": "cache"
    },
    "open_meteo": {
      "status": "success",
      "nodes_count": 78
    },
    "google_directions": {
      "status": "success",
      "edges_processed": 1,
      "success_rate": 100.0
    }
  }
}
```

---

### 4. `weather_snapshot.json` (18 KB)

**Source**: Open-Meteo API  
**Purpose**: Weather conditions at each node location  
**Update Frequency**: Every collection (real-time)

**Schema**:

```json
{
  "timestamp": "2025-10-29T13:42:30.123456",
  "nodes": [
    {
      "node_id": "node-10.760533-106.677134",
      "lat": 10.760533,
      "lon": 106.677134,
      "temperature_c": 28.5,
      "humidity_percent": 75,
      "precipitation_mm": 0.0,
      "wind_speed_kmh": 12.3,
      "weather_code": 3,
      "cloud_cover_percent": 60
    }
  ]
}
```

**Typical Size**: 78 nodes × ~230 bytes = 18 KB

---

### 5. `traffic_edges.json` (290 B - 68 KB full)

**Source**: Google Directions API  
**Purpose**: Real-time traffic data (duration, speed) for road segments  
**Update Frequency**: Every collection (real-time)

**Schema**:

```json
[
  {
    "node_a_id": "node-10.760533-106.677134",
    "node_b_id": "node-10.762243-106.676490",
    "distance_km": 0.722,
    "duration_sec": 157,
    "speed_kmh": 16.55,
    "has_traffic_data": true,
    "timestamp": "2025-10-29T13:42:36.312247",
    "api_type": "real"
  }
]
```

**Size Calculation**:

- Test mode: 1 edge × 290 bytes = 290 B
- Production: 234 edges × 290 bytes = 68 KB

---

## Collection Workflow

1. **Topology Collection** (Overpass API)

   - Check cache: `cache/overpass_topology.json`
   - If exists and fresh (< 30 days): Use cache ✅
   - Else: Fetch from Overpass API → Save to cache
   - Copy `nodes.json` and `edges.json` to run directory

2. **Weather Collection** (Open-Meteo API)

   - For each node: Query current weather
   - Aggregate into `weather_snapshot.json`
   - Save to run directory

3. **Traffic Collection** (Google Directions API)

   - Load edges from `edges.json`
   - For each edge: Query driving route with traffic
   - Extract duration → Calculate speed
   - Save to `traffic_edges.json`

4. **Metadata Generation**
   - Record collection stats
   - Save to `statistics.json`

**Total Time**: ~9 seconds (1 edge), ~15-20 min (234 edges)

---

## Data Sizes

| Collection Mode            | Total Size | Breakdown                                                        |
| -------------------------- | ---------- | ---------------------------------------------------------------- |
| **Test (1 edge)**          | 62 KB      | nodes(44) + edges(0.3) + stats(0.3) + weather(18) + traffic(0.3) |
| **Production (234 edges)** | 130 KB     | nodes(44) + edges(0.3) + stats(0.3) + weather(18) + traffic(68)  |

**3-Day Collection**: 54 runs × 130 KB = **7.02 MB**

---

## API Usage

### Open-Meteo (Free, No Key)

- **Requests**: 78 nodes/run × 54 runs = 4,212 requests
- **Cost**: $0 (free tier)

### Google Directions API (Paid)

- **Requests**: 234 edges/run × 54 runs = 12,636 requests
- **Cost**: 12,636 × $0.005 = **$63.18**

### Overpass API (Free, Cached)

- **Requests**: 1 initial + 0 cached = 1 total
- **Cost**: $0

---

## Quality Metrics

### Completeness

- ✅ **Topology**: 78 nodes, 234 edges (100% coverage)
- ✅ **Weather**: 78 locations (100% coverage)
- ✅ **Traffic**: 234 edges (100% coverage in production)

### Accuracy

- **Spatial**: ±10m (GPS precision)
- **Weather**: ±0.5°C temperature, ±5% humidity
- **Traffic**: Real-time Google data (highest accuracy)

### Freshness

- **Topology**: Cached (updates only if OSM changes)
- **Weather**: Real-time (< 1 min old)
- **Traffic**: Real-time (< 1 min old)

---

## Usage Examples

### Load a Collection

```python
import json
from pathlib import Path

# Load specific run
run_dir = Path('data/runs/run_20251029_134227')

# Topology
with open(run_dir / 'nodes.json') as f:
    nodes = json.load(f)
with open(run_dir / 'edges.json') as f:
    edges = json.load(f)

# Weather
with open(run_dir / 'weather_snapshot.json') as f:
    weather = json.load(f)

# Traffic
with open(run_dir / 'traffic_edges.json') as f:
    traffic = json.load(f)

print(f"Loaded {len(nodes)} nodes, {len(traffic)} traffic readings")
```

### Find Latest Collection

```python
from pathlib import Path
import re

runs_dir = Path('data/runs')
runs = sorted([d for d in runs_dir.iterdir() if d.is_dir()])
latest = runs[-1]

print(f"Latest collection: {latest.name}")
```

### Merge Multiple Collections

```python
import json
from pathlib import Path
from datetime import datetime

runs_dir = Path('data/runs')
all_traffic = []

for run in sorted(runs_dir.iterdir()):
    traffic_file = run / 'traffic_edges.json'
    if traffic_file.exists():
        with open(traffic_file) as f:
            data = json.load(f)
        all_traffic.extend(data)

print(f"Total traffic readings: {len(all_traffic)}")
```

---

## Version History

### v1.0 (2025-10-29)

- ✅ Standardized directory structure: `run_YYYYmmdd_HHMMSS/`
- ✅ 5-file schema: nodes, edges, statistics, weather, traffic
- ✅ Permanent topology caching
- ✅ Support for both test (1 edge) and production (234 edges) modes
- ✅ Complete Google Directions integration
- ✅ Validated on GCP VM deployment

---

## See Also

- [Data Model Reference](data_model.md)
- [Collection Scripts](../getting-started/SCRIPTS_REFERENCE.md)
- [GCP Deployment Guide](../../scripts/deployment/GCP_DEPLOYMENT_README.md)
