# Collection Optimization v5.1 - Final Review

**Traffic Forecast Data Collection - Optimized for Cost & Quality**

---

## Optimization Summary

### 3 Major Improvements

1. **Adaptive Scheduling** - Collect smarter, not harder
2. **Weather Grid Caching** - Share weather data across nearby nodes
3. **Permanent Topology Caching** - Never re-fetch static data

---

## Results

### Cost Reduction

| Metric | Before | After v5.1 | Savings |
| ------------------------ | ------- | ---------- | ---------- |
| **Daily collections** | 24 | 18 | 25% fewer |
| **Daily API requests** | 5,616 | 4,212 | 25% fewer |
| **Daily cost** | $28.08 | $21.06 | **$7.02** |
| **Weekly cost (7 days)** | $196.56 | $147.42 | **$49.14** |

**Total savings: 25% cost reduction with BETTER data quality during peaks!**

---

## 1. Adaptive Scheduling

### Strategy

**Peak Hours (7-9 AM, 12-1 PM, 5-8 PM):** 30-minute intervals

- High traffic variability
- Need frequent samples to capture patterns
- 12 collections/day during peaks

**Off-Peak Hours (9 AM-12 PM, 1-5 PM, 8-11 PM):** 90-minute intervals

- Moderate traffic
- Less variability
- 6 collections/day during off-peak

**Night (11 PM-7 AM):**Skip completely

- Very stable traffic
- Minimal value for ML model
- 0 collections/night = **Major savings**

**Weekend:** 120-minute intervals

- Less traffic overall
- Relaxed schedule

### Benefits

**Captures rush hour dynamics** - 30min during peaks
**Saves money at night** - Skip when traffic stable
**Maintains coverage** - Still 18 collections/day
**Better ML training** - More data during important hours

### Configuration

```yaml
# configs/project_config.yaml
scheduler:
mode: adaptive
adaptive:
peak_hours:
time_ranges:
- start: "07:00"
end: "09:00"
- start: "12:00"
end: "13:00"
- start: "17:00"
end: "20:00"
interval_minutes: 30

offpeak:
interval_minutes: 90

night:
skip: true # No collection at night

weekend:
interval_minutes: 120
```

---

## 2. Weather Grid Caching

### Problem

**Before:**Fetch weather for EACH node

- 78 nodes = 78 API calls per collection
- Weather is homogeneous within 1km²
- **Wasteful!**

### Solution

**Grid-based caching:**

- Divide coverage area (4096m radius) into 1km² cells
- ~53 cells for entire coverage area
- Fetch weather once per cell, share with all nodes in that cell

### Savings

| Metric | Without Grid | With Grid | Reduction |
| ----------------- | ------------ | --------- | --------- |
| Weather API calls | 78 | 53 | **32%** |
| Cache duration | N/A | 30 min | Reuse! |

**Result:** 32% fewer weather API calls + smart caching!

### How It Works

```python
# traffic_forecast/collectors/weather_grid_cache.py

# Create grid cache
cache = WeatherGridCache(
center_lat=10.772465,
center_lon=106.697794,
radius_m=4096,
cell_size_m=1000, # 1km x 1km cells
cache_expiry_minutes=30
)

# For each node
for node in nodes:
# Check if weather already cached for this cell
weather = cache.get_weather(node['lat'], node['lon'])

if weather is None:
# Fetch weather for cell center
cell_lat, cell_lon = cache.get_cell_center(node)
weather = fetch_weather(cell_lat, cell_lon)

# Cache for all nodes in this cell
cache.set_weather(node['lat'], node['lon'], weather)

# Use cached weather
node['weather'] = weather
```

### Why This Works

**Weather is spatially homogeneous:**

- Temperature varies < 0.5°C within 1km²
- Precipitation same within 1km²
- Wind speed similar within 1km²

**Perfectly reasonable for traffic forecasting!**

---

## 3. Permanent Topology Caching

### Static Data

**Overpass (OSM) topology doesn't change:**

- Road network stable
- Node coordinates fixed
- Road types (primary/trunk) constant

### Cache Forever

```yaml
# configs/project_config.yaml
collectors:
overpass:
use_cache: true
cache_expiry_days: 30 # Re-fetch monthly for safety
cache_file: cache/overpass_topology.json
force_cache: true # Never re-fetch if cache exists
```

**Result:**

- Topology fetched once (already done!)
- Reuse `cache/overpass_topology.json` forever
- Zero Overpass API calls after first collection
- **100% savings on topology fetching**

---

## Collection Schedule Example

### Weekday Schedule

```
Time Range | Type | Interval | Collections
---------------|-----------|----------|------------
00:00 - 07:00 | Night | SKIP | 0
07:00 - 09:00 | Peak | 30 min | 4
09:00 - 12:00 | Off-peak | 90 min | 2
12:00 - 13:00 | Peak | 30 min | 2
13:00 - 17:00 | Off-peak | 90 min | 2
17:00 - 20:00 | Peak | 30 min | 6
20:00 - 23:00 | Off-peak | 90 min | 2
23:00 - 24:00 | Night | SKIP | 0
---------------|-----------|----------|------------
TOTAL | | | 18/day
```

### Weekend Schedule

```
Time Range | Interval | Collections
---------------|----------|------------
00:00 - 24:00 | 120 min | 12/day
```

---

## Cost Breakdown

### Per Collection

- **Edges:** 234 (78 nodes × 3 neighbors)
- **Traffic API calls:** 234
- **Weather API calls:** ~53 (grid-based)
- **Topology API calls:** 0 (cached)
- **Total API calls:** 234 (only Google Directions counts)
- **Cost per collection:** 234 × $0.005 = $1.17

### Daily Cost (Weekday)

```
Period Collections Cost
Peak 12 $14.04
Off-peak 6 $7.02
Night 0 $0.00
--------------------------------------
Total 18 $21.06/day
```

### Weekly Cost (5 weekdays + 2 weekend)

```
Weekdays: 5 × $21.06 = $105.30
Weekend: 2 × $14.04 = $28.08 (12 collections/day)
--------------------------------------
Total: $133.38/week
```

**For 7-day academic project: ~$133-147** (vs $197 before!)

---

## Implementation

### Files Modified

1. **`configs/project_config.yaml`**

- Added adaptive scheduler configuration
- Added weather grid cache settings
- Updated Overpass cache settings

2. **`traffic_forecast/collectors/weather_grid_cache.py`** (NEW)

- Grid-based weather caching
- 1km² cell management
- 30-minute cache expiry

3. **`traffic_forecast/scheduler/adaptive_scheduler.py`** (EXISTS)
- Time-based interval selection
- Peak/off-peak/night logic

### Testing

```bash
# Test weather grid cache
python traffic_forecast/collectors/weather_grid_cache.py

# Test adaptive scheduler
python tests/test_adaptive_scheduler.py
```

---

## Academic Considerations

### For DSP391m Project (7 days)

**Total budget:** ~$140-150 (down from $197!)

**Data quality:**

- Dense sampling during rush hours (30 min)
- Adequate off-peak coverage (90 min)
- Skip night (not useful for model)
- Total ~126-140 collections over 7 days

**Sufficient for:**

- Traffic pattern analysis
- Peak/off-peak comparison
- ML model training (XGBoost, RF, LightGBM)
- Temporal feature engineering
- Academic report/presentation

---

## Deployment Instructions

### 1. Verify Configuration

```bash
# Check config file
cat configs/project_config.yaml

# Test scheduler
python tests/test_adaptive_scheduler.py

# Test weather cache
python traffic_forecast/collectors/weather_grid_cache.py
```

### 2. Deploy to GCP VM

```python
# In notebooks/GCP_DEPLOYMENT.ipynb

# Upload updated config
upload_project_files()

# Deploy with new settings
deploy_on_vm()

# Setup cron (will use adaptive schedule)
setup_cron_collection()
```

### 3. Monitor Collection

```python
# Check logs for adaptive schedule
check_collection_logs(lines=100)

# Verify weather grid caching
ssh_to_vm('cat ~/traffic-forecast/cache/weather_grid.json | jq length')

# Validate cost
estimate_current_costs()
```

---

## Comparison: Before vs After

### Collection Strategy

| Aspect | Before v5.0 | After v5.1 |
| -------------------- | ---------------- | --------------------- |
| **Schedule** | Fixed 60 min | Adaptive (30/90/skip) |
| **Night collection** | Yes | No (skip) |
| **Peak sampling** | Same as off-peak | 2x more frequent |
| **Weather calls** | 78/collection | 53/collection (-32%) |
| **Topology calls** | Every time | Once (cached) |

### Cost & Quality

| Metric | Before | After | Change |
| --------------------- | ------- | ------- | -------------- |
| **Collections/day** | 24 | 18 | -25% |
| **Daily cost** | $28.08 | $21.06 | -$7.02 |
| **Weekly cost** | $196.56 | $147.42 | -$49.14 |
| **Peak coverage** | Medium | High | Better |
| **Off-peak coverage** | High | Medium | Adequate |
| **Night coverage** | High | None | Unnecessary |

**Result: Lower cost + better quality where it matters!**

---

## Rationale

### Why Adaptive Scheduling?

**Traffic patterns are time-dependent:**

- Rush hours (7-9 AM, 5-8 PM): High variability
- Mid-day: Moderate, stable
- Night: Very stable, predictable

**ML models benefit from:**

- Dense samples during high-variance periods
- Sparse samples during low-variance periods
- More useful data per dollar spent

### Why Weather Grid Caching?

**Weather is spatially smooth:**

- Temperature gradient: ~0.1°C/km in urban areas
- Precipitation: Same within several km
- Wind: Relatively uniform within city blocks

**1km² resolution is:**

- More than enough for traffic forecasting
- Meteorologically sound
- Cost-effective

### Why Skip Night Collection?

**Night traffic is:**

- Very predictable (low, stable)
- Minimal ML value (model easily learns)
- Not useful for rush hour forecasting

**Better to:**

- Save money on night collections
- Invest in denser peak coverage
- Get more useful training data

---

## Final Recommendations

### For 7-Day Academic Project

**Use adaptive schedule:**

- Peak: 30 min
- Off-peak: 90 min
- Night: Skip
- Weekend: 120 min

**Enable weather grid caching:**

- 1km² cells
- 30-minute expiry

**Use permanent topology cache:**

- Already cached at `cache/overpass_topology.json`
- Never re-fetch

**Expected outcome:**

- ~18 collections/day weekday
- ~12 collections/day weekend
- ~$21/day weekday, ~$14/day weekend
- **Total 7-day cost: ~$140-150**
- **126-140 total collections**
- **Dense coverage during important hours**

---

## Summary

### What We Optimized

1. **Scheduling** - Adaptive based on time/day
2. **Weather** - Grid caching (32% reduction)
3. **Topology** - Permanent caching (100% reduction)

### Cost Impact

- **Before:** $28/day, $197/week
- **After:** $21/day, $147/week
- **Savings:** 25% ($50/week)

### Quality Impact

- **Peak hours:**Better coverage (30 min vs 60 min)
- **Off-peak:**Adequate coverage (90 min)
- **Night:**None (not needed)
- **Overall:**Better data for ML training

### Implementation Status

- Configuration updated
- Weather grid cache created
- Adaptive scheduler exists
- Testing complete
- Ready for deployment

---

**Version:** 5.1
**Date:**January 2025
**Status:**Optimized and Ready for Production
