# Data Collections

**Run-based data collection storage**

---

## Structure

```
data/
├── runs/                     # All collection runs
│   ├── run_20251029_084417/  # Run 1
│   │   ├── nodes.json        # Topology (78 nodes)
│   │   ├── edges.json        # Road connections
│   │   ├── statistics.json   # Topology stats
│   │   ├── traffic_edges.json  # Traffic data (234 edges)
│   │   └── weather_snapshot.json  # Weather (78 nodes)
│   │
│   ├── run_20251029_094417/  # Run 2
│   │   ├── nodes.json
│   │   ├── traffic_edges.json
│   │   └── weather_snapshot.json
│   │
│   ├── run_20251029_104417/  # Run 3
│   │   └── ...
│   │
│   └── ... (54 runs total after 3 days)
│
├── final/               # Merged datasets (after 3 days)
│   ├── traffic_complete.json      # All traffic merged
│   ├── weather_complete.json      # All weather merged
│   ├── traffic_timeseries.json    # Grouped by edge
│   └── weather_timeseries.json    # Grouped by node
│
└── cache/               # Permanent cache (shared)
    └── overpass_topology.json  # OSM topology cache
```

---

## How It Works

### During Collection (3 days)

**Each collection creates 1 timestamped directory:**

- `run_YYYYmmdd_HHMMSS/`

**Contains 5 files:**

- `nodes.json` - Topology (78 nodes, ~46 KB)
- `edges.json` - Connections (minimal)
- `statistics.json` - Stats
- `traffic_edges.json` - Traffic (234 edges, ~68 KB)
- `weather_snapshot.json` - Weather (78 nodes, ~19 KB)

**After 54 runs (3 days × 18/day):**

- 54 directories (~7.2 MB total)
- Each run ~133 KB

---

## Usage

### View Current Progress

```bash
python scripts/view_collections.py
```

**Shows:**

- Number of runs
- Time range
- Speed statistics
- Progress to 3-day target
- Storage usage

---

### Merge After 3 Days

```bash
python scripts/merge_collections.py
```

**Creates 4 files in `data/final/`:**

1. **`traffic_complete.json`** - All traffic records chronologically

   ```json
   [
     {"node_a_id": "...", "speed_kmh": 18.5, "timestamp": "2025-10-29T08:44:08"},
     {"node_a_id": "...", "speed_kmh": 22.1, "timestamp": "2025-10-29T09:44:08"},
     ...
   ]
   ```

2. **`weather_complete.json`** - All weather records chronologically

   ```json
   [
     {"node_id": "...", "temperature": 28.5, "timestamp": "2025-10-29T08:44:07"},
     ...
   ]
   ```

3. **`traffic_timeseries.json`** - Grouped by edge (for ML)

   ```json
   {
     "node-A-->node-B": [
       {"speed_kmh": 18.5, "timestamp": "2025-10-29T08:44:08"},
       {"speed_kmh": 22.1, "timestamp": "2025-10-29T09:44:08"},
       ...
     ],
     ...
   }
   ```

4. **`weather_timeseries.json`** - Grouped by node (for ML)
   ```json
   {
     "node-10.7777-106.6666": [
       {"temperature": 28.5, "timestamp": "2025-10-29T08:44:07"},
       ...
     ],
     ...
   }
   ```

---

## File Formats

### Traffic File Format

```json
[
  {
    "node_a_id": "node-10.802705-106.698490",
    "node_b_id": "node-10.801105-106.709186",
    "distance_km": 1.817,
    "duration_sec": 421,
    "speed_kmh": 15.537292161520188,
    "has_traffic_data": true,
    "timestamp": "2025-10-29T08:44:08.097516",
    "api_type": "real"
  }
]
```

**234 records per file** (one per edge)

---

### Weather File Format

```json
[
  {
    "node_id": "node-10.802705-106.698490",
    "temperature": 28.5,
    "humidity": 75,
    "precipitation": 0.0,
    "timestamp": "2025-10-29T08:44:07.123456"
  }
]
```

**78 records per file** (one per node)

---

## Expected Data Volume

### Per Collection

- Traffic: ~68 KB (234 edges)
- Weather: ~19 KB (78 nodes)
- Total: ~87 KB

### After 54 Collections (3 days)

- Traffic: ~3.6 MB (12,636 records)
- Weather: ~1 MB (4,212 records)
- Total: ~5 MB

### After Merge

- Complete datasets: ~3 MB compressed
- Time-series format: ~4 MB (optimized for ML)

---

## Cleanup

### During Collection

**Keep all files!** Don't delete anything during the 3-day collection.

### After Download from VM

```bash
# Download from VM
gcloud compute scp --recurse \
  traffic-forecast-collector:~/traffic-forecast/data/collections \
  ./data/collections --zone=asia-southeast1-a

# Merge locally
python scripts/merge_collections.py

# Now safe to delete VM and timestamped files
# Keep only data/final/ for ML training
```

### Manual Cleanup (if needed)

```bash
# Remove all timestamped files (keeps latest)
rm data/collections/traffic_edges_202*.json
rm data/collections/weather_snapshot_202*.json

# Keep only latest
ls data/collections/*latest.json
```

---

## Advanced: Custom Merge

### Merge Specific Date Range

```python
import glob
from datetime import datetime

# Find files from specific date
files = glob.glob('data/collections/traffic_edges_20251029*.json')

# Or time range
def filter_by_time(filename, start, end):
    # Extract timestamp from filename
    ts = filename.split('_')[-2] + '_' + filename.split('_')[-1].split('.')[0]
    dt = datetime.strptime(ts, '%Y%m%d_%H%M%S')
    return start <= dt <= end

start = datetime(2025, 10, 29, 8, 0)
end = datetime(2025, 10, 29, 18, 0)

filtered = [f for f in files if filter_by_time(f, start, end)]
```

### Extract Peak Hours Only

```python
# Get only 7-9 AM and 5-7 PM collections
import glob
from datetime import datetime

all_files = glob.glob('data/collections/traffic_edges_*.json')

peak_files = []
for f in all_files:
    ts = f.split('_')[-2] + '_' + f.split('_')[-1].split('.')[0]
    dt = datetime.strptime(ts, '%Y%m%d_%H%M%S')
    hour = dt.hour

    if (7 <= hour < 9) or (17 <= hour < 19):
        peak_files.append(f)

print(f"Peak hour collections: {len(peak_files)}")
```

---

## Troubleshooting

### Missing Files

**Expected:** 1 traffic + 1 weather per collection

**If missing:**

```bash
# Check logs
tail -50 logs/collection.log

# Check if script failed
grep FAILED logs/cron.log
```

### Duplicate Timestamps

Collections < 1 second apart may have same timestamp.

**Solution:** Already handled - uses microseconds in timestamp.

### Large File Sizes

**Normal:** 68 KB per traffic file, 19 KB per weather

**If larger:** Check if duplicate records in file.

---

**Last updated:** October 29, 2025  
**Version:** v5.1 (timestamped collections)
