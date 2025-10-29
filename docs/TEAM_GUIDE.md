# Traffic Forecast Project - Team Guide

## Project Overview

This project collects real-time traffic data from Ho Chi Minh City roads using Google Maps API and Open-Meteo weather API. The system runs continuously on Google Cloud Platform, collecting data every 30-120 minutes depending on traffic patterns.

**Key Information:**

- Repository: https://github.com/thatlq1812/dsp391m_project
- GCP VM: traffic-forecast-collector (asia-southeast1-a)
- Coverage: 64 major intersections, 144 traffic routes
- Collection frequency: ~150-200 runs per day
- Data format: JSON files (nodes, edges, traffic, weather)

---

## For Team Members - Download Data

### Option 1: Using gcloud CLI (Recommended)

**Requirements:**

- Google Cloud SDK installed
- Authenticated with gcloud

**Steps:**

1. Install gcloud CLI (if not installed):

   - Download from: https://cloud.google.com/sdk/docs/install
   - Run: `gcloud init`

2. Authenticate:

   ```bash
   gcloud auth login
   ```

3. Download data using script:

   ```bash
   ./scripts/data/download_latest.sh
   ```

4. Choose download option:
   - Option 1: Latest run only (fastest)
   - Option 2: Last 10 runs
   - Option 3: Last 24 hours
   - Option 4: All data

**Output:**
Data will be downloaded to `./downloaded_data/` directory with structure:

```
downloaded_data/
├── run_20251030_032440/
│   ├── nodes.json
│   ├── edges.json
│   ├── traffic_edges.json
│   ├── weather_snapshot.json
│   └── statistics.json
└── run_20251030_032457/
    └── ...
```

### Option 2: Manual Download

If you have gcloud access but prefer manual download:

```bash
# List available runs
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="ls -lt ~/traffic-forecast/data/runs/ | head -10"

# Download specific run
RUN_ID="run_20251030_032440"
gcloud compute scp \
  --zone=asia-southeast1-a \
  --recurse \
  traffic-forecast-collector:~/traffic-forecast/data/runs/$RUN_ID \
  ./my_data/
```

### Option 3: HTTP Download (No Authentication)

**Note:** Admin must setup HTTP server on VM first.

Ask admin to run on VM:

```bash
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a
cd ~/traffic-forecast
./scripts/data/serve_data_public.sh
```

Then team members can download:

```bash
./scripts/data/download_simple.sh
```

---

## Data Structure

Each collection run contains 5 JSON files:

### 1. nodes.json

Contains road network nodes (intersections):

```json
[
  {
    "id": "123456789",
    "lat": 10.772465,
    "lon": 106.697794,
    "name": "Nguyen Hue - Le Loi",
    "degree": 4,
    "importance_score": 25.5,
    "road_types": ["primary", "trunk"]
  }
]
```

**Fields:**

- `id`: OpenStreetMap node ID
- `lat`, `lon`: GPS coordinates
- `name`: Street intersection name
- `degree`: Number of roads meeting at intersection
- `importance_score`: Node importance (higher = more important)
- `road_types`: Types of roads (motorway, trunk, primary)

### 2. edges.json

Contains node pairs for traffic collection:

```json
[
  {
    "node_a_id": "123456",
    "node_b_id": "789012",
    "node_a_lat": 10.77,
    "node_a_lon": 106.69,
    "node_b_lat": 10.78,
    "node_b_lon": 106.7
  }
]
```

### 3. traffic_edges.json

Contains actual traffic data:

```json
[
  {
    "node_a_id": "123456",
    "node_b_id": "789012",
    "distance_meters": 1250,
    "duration_minutes": 3.5,
    "duration_in_traffic_minutes": 8.2,
    "speed_kmh": 21.4,
    "congestion_level": "moderate",
    "timestamp": "2025-10-30T03:24:40+07:00"
  }
]
```

**Fields:**

- `distance_meters`: Route distance
- `duration_minutes`: Normal travel time (no traffic)
- `duration_in_traffic_minutes`: Actual travel time with current traffic
- `speed_kmh`: Average speed
- `congestion_level`: low, moderate, high, severe
- `timestamp`: Collection time (UTC+7)

### 4. weather_snapshot.json

Contains weather data at collection time:

```json
[
  {
    "node_id": "123456",
    "temperature_c": 28.5,
    "rain_mm": 0.0,
    "wind_speed_kmh": 12.3,
    "forecast_5min": {...},
    "forecast_15min": {...},
    "forecast_30min": {...},
    "forecast_60min": {...}
  }
]
```

### 5. statistics.json

Contains collection metadata:

```json
{
  "total_nodes": 64,
  "total_edges": 144,
  "avg_degree": 4.5,
  "collection_time": "2025-10-30T03:24:40+07:00",
  "duration_seconds": 15.2
}
```

---

## Data Analysis Examples

### Python - Load and Analyze

```python
import json
import pandas as pd
from pathlib import Path

# Load traffic data
run_dir = Path("downloaded_data/run_20251030_032440")
with open(run_dir / "traffic_edges.json") as f:
    traffic_data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(traffic_data)

# Basic statistics
print(f"Total edges: {len(df)}")
print(f"Average speed: {df['speed_kmh'].mean():.1f} km/h")
print(f"Congestion distribution:")
print(df['congestion_level'].value_counts())

# Filter slow traffic
slow_traffic = df[df['speed_kmh'] < 20]
print(f"\nSlow traffic routes: {len(slow_traffic)}")
```

### Merge Multiple Runs

```python
import json
from pathlib import Path
import pandas as pd

runs_dir = Path("downloaded_data")
all_traffic = []

for run_dir in sorted(runs_dir.glob("run_*")):
    traffic_file = run_dir / "traffic_edges.json"
    if traffic_file.exists():
        with open(traffic_file) as f:
            data = json.load(f)
            all_traffic.extend(data)

df = pd.DataFrame(all_traffic)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"Total data points: {len(df)}")
print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
```

### Time Series Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Group by hour
df['hour'] = df['timestamp'].dt.hour
hourly_speed = df.groupby('hour')['speed_kmh'].mean()

# Plot
plt.figure(figsize=(12, 6))
hourly_speed.plot(kind='bar')
plt.xlabel('Hour of Day')
plt.ylabel('Average Speed (km/h)')
plt.title('Average Traffic Speed by Hour')
plt.tight_layout()
plt.savefig('hourly_speed.png')
```

---

## Common Questions

**Q: How often is data collected?**
A: Peak hours (6:30-8:00, 10:30-11:30, 16:00-19:00): every 30 minutes
Off-peak hours: every 120 minutes

**Q: What is the data size?**
A: Each run is ~200-300KB. Daily total is ~50MB. 3-day collection is ~150MB.

**Q: How long will collection run?**
A: Currently configured for continuous 3-day collection (Oct 30 - Nov 2, 2025).

**Q: What if I don't have gcloud access?**
A: Ask admin to setup HTTP server (Option 3 above) for authentication-free download.

**Q: Can I download while collection is running?**
A: Yes, download does not affect the collection process.

**Q: What timezone is used?**
A: All timestamps are in UTC+7 (Vietnam timezone).

**Q: Why are some edges missing traffic data?**
A: Some routes may not be available or API may fail. Success rate is typically 60-70%.

---

## Support

**For data access issues:**

- Check you have gcloud installed and authenticated
- Verify VM is running: `gcloud compute instances list`
- Contact admin if HTTP server needed

**For data analysis help:**

- See example code above
- Check main README.md for detailed documentation
- Review traffic_forecast package documentation

**Repository:** https://github.com/thatlq1812/dsp391m_project
