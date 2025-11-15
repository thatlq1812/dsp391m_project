# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Traffic Forecasting Demo System (DEPRECATED - See DEMO_BACKPREDICT.md)

## ⚠️ DEPRECATION NOTICE

**This document describes an outdated approach that requires waiting for future data.**

**→ See [DEMO_BACKPREDICT.md](DEMO_BACKPREDICT.md) for the improved back-prediction strategy.**

**Key differences:**

- Old: Predict 17:00→20:00, wait until 20:00 for actual data ❌
- New: Back-predict from 14:00, 15:00, 15:30, 16:00 → instant comparison ✅
- New approach shows convergence behavior and doesn't require waiting

---

## Overview (Old Approach - For Reference Only)

This document outlines a **simple, script-based demo system** that:

1. **VM continuously collects** traffic data (background service)
2. **Demo script** predicts future traffic (e.g., 17:00 → 20:00)
3. **Waits for actual data** at target time (20:00) ⚠️ **PROBLEM: Must wait!**
4. **Generates comparison charts** (STMGT vs Actual vs Google Maps baseline)
5. **Exports static visualizations** (PNG/HTML) for presentation

---

## Simplified Demo Architecture (Old)

### Philosophy: Export-Based, Not Real-Time Web

**Why This Approach?**

- ✅ **Simple:** One Python script, no web stack
- ✅ **Reliable:** No WebSocket/server complexity
- ✅ **Portable:** Figures can be shown anywhere (PowerPoint, PDF)
- ✅ **Focus:** Show model accuracy, not web dev skills

**What We DON'T Need:**

- ❌ FastAPI web server
- ❌ WebSocket real-time updates
- ❌ Frontend JavaScript complexity
- ❌ Redis pub/sub
- ❌ Nginx reverse proxy

**What We DO Need:**

- ✅ VM with data collector (cron job)
- ✅ PostgreSQL or Parquet files for storage
- ✅ Demo script: `scripts/demo/generate_demo_figures.py`
- ✅ Static map generator (Folium or Google Maps Static API)
- ✅ Matplotlib/Plotly for charts

---

## Demo System Components

### 1. VM Infrastructure (Minimal Setup)

**Cloud Provider:** GCP / AWS / Azure (recommend GCP e2-micro for free tier)

**VM Specs:**

```yaml
Instance Type: e2-small (2 vCPU, 2GB RAM) - Sufficient for data collection only
GPU: None (inference runs on local dev machine)
OS: Ubuntu 22.04 LTS
Storage: 20GB SSD (stores 1+ month of data)
Network: No public ports needed (only SSH)
```

**Services:**

- Python 3.10+ with pandas, requests
- Parquet files OR PostgreSQL (lightweight setup)
- Cron job for data collection (every 15 min)

**Cost Estimate:** ~$15-25/month (or FREE with GCP e2-micro tier)

---

### 2. Continuous Data Collection (VM Service)

#### Component: Traffic Data Collector

**File:** `scripts/deployment/traffic_collector.py`

**Flow:**

```
Every 15 minutes (cron job):
1. Scrape Overpass API for current traffic speeds
2. Fetch weather from OpenWeatherMap API (optional)
3. Validate & clean data
4. Append to Parquet file: traffic_data_{YYYYMM}.parquet
5. Log success/failure
```

**Data Storage (Parquet Schema):**

```python
{
    'timestamp': datetime64[ns],
    'edge_id': str,
    'node_a_id': str,
    'node_b_id': str,
    'speed_kmh': float32,
    'temperature_c': float32,      # From OpenWeatherMap
    'humidity_percent': float32,
    'wind_speed_kmh': float32,
    'pressure_hpa': float32,
    'weather_condition': str
}
```

**Why Parquet Instead of PostgreSQL?**

- Simpler: No database setup/maintenance
- Portable: Copy files to local machine for demo
- Efficient: Compressed, fast reads with pandas
- Sufficient: 1 month ≈ 2-5MB compressed

**File Organization:**

```
/opt/traffic_data/
├── traffic_data_202511.parquet  # November 2025
├── traffic_data_202512.parquet  # December 2025
├── collector.log
└── collector.py
```

**Cron Setup:**

```bash
# /etc/crontab on VM
*/15 * * * * ubuntu python3 /opt/traffic_data/collector.py >> /opt/traffic_data/collector.log 2>&1
```

**Collector Implementation Highlights:**

```python
# scripts/deployment/traffic_collector.py
import pandas as pd
from pathlib import Path
from datetime import datetime
import requests

def collect_traffic():
    # 1. Scrape Overpass API (reuse existing logic)
    edges = scrape_overpass_traffic()

    # 2. Fetch weather (optional)
    weather = fetch_openweather()

    # 3. Create DataFrame
    df = pd.DataFrame(edges)
    df['timestamp'] = datetime.now()
    df['temperature_c'] = weather['temp']
    # ... add other weather features

    # 4. Append to monthly Parquet file
    month_file = f"traffic_data_{datetime.now():%Y%m}.parquet"
    if Path(month_file).exists():
        df_existing = pd.read_parquet(month_file)
        df = pd.concat([df_existing, df], ignore_index=True)

    df.to_parquet(month_file, compression='snappy')
    print(f"✓ Collected {len(df)} records")

if __name__ == '__main__':
    collect_traffic()
```

---

### 3. Demo Script: Predict → Wait → Compare

#### Demo Workflow

**File:** `scripts/demo/generate_demo_figures.py`

**Scenario:** Predict 17:00 → 20:00 traffic

```python
# Run at 17:00
python scripts/demo/generate_demo_figures.py \
    --predict-from 17:00 \
    --predict-to 20:00 \
    --data-source traffic_data_202511.parquet \
    --model outputs/stmgt_v3_production/best_model.pt \
    --wait-for-actual  # Blocks until 20:00 to collect actual data
```

**Phase 1: Prediction (17:00)**

```
1. Load data from 14:00-17:00 (3 hours lookback)
2. Use STMGT to predict speeds at:
   - 18:00 (horizon=1h)
   - 19:00 (horizon=2h)
   - 20:00 (horizon=3h)
3. Save predictions to predictions.json
```

**Phase 2: Wait (17:00-20:00)**

```
4. Script sleeps or polls VM for new data
5. At 20:00, download actual traffic data
```

**Phase 3: Comparison (20:00)**

```
6. Load actual speeds at 18:00, 19:00, 20:00
7. Calculate errors by horizon:
   - MAE, RMSE, R² for each horizon
   - Per-edge error distribution
8. Generate comparison figures (see below)
```

#### Comparison Metrics

**Overall Accuracy:**

```python
{
    "horizon_1h": {"mae": 2.8, "rmse": 4.1, "r2": 0.89},
    "horizon_2h": {"mae": 3.5, "rmse": 5.2, "r2": 0.82},
    "horizon_3h": {"mae": 4.2, "rmse": 6.3, "r2": 0.75}
}
```

**Per-Edge Errors:**

```python
{
    "edge_A_B": {
        "predicted": [45.2, 43.1, 41.8],  # 18h, 19h, 20h
        "actual": [46.1, 42.5, 40.2],
        "errors": [0.9, 0.6, 1.6]
    },
    ...
}
```

---

### 4. Visualization Outputs (Static Exports)

#### Figure 1: Prediction vs Actual Line Chart

**File:** `demo_output/comparison_chart.png`

**Content:**

- X-axis: Time (18:00, 19:00, 20:00)
- Y-axis: Speed (km/h)
- Lines:
  - **Blue:** Predicted speeds (3 lines for 3 sample edges)
  - **Green/Red:** Actual speeds
  - **Error bars:** Shaded regions showing uncertainty (±1σ from mixture model)
- Title: "STMGT Prediction Accuracy (17:00 → 20:00 Demo)"

**Implementation (Matplotlib):**

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))

for edge_id in sample_edges:
    # Predicted
    ax.plot(times, predictions[edge_id], 'b--', label='Predicted', alpha=0.7)

    # Actual
    ax.plot(times, actuals[edge_id], 'g-', label='Actual', linewidth=2)

    # Uncertainty bounds
    ax.fill_between(times,
                     predictions[edge_id] - std[edge_id],
                     predictions[edge_id] + std[edge_id],
                     alpha=0.2, color='blue')

ax.set_xlabel('Time')
ax.set_ylabel('Speed (km/h)')
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig('demo_output/comparison_chart.png', dpi=300, bbox_inches='tight')
```

#### Figure 2: Error by Horizon Bar Chart

**File:** `demo_output/error_by_horizon.png`

**Content:**

- X-axis: Prediction horizon (1h, 2h, 3h)
- Y-axis: MAE (km/h)
- Bars colored by accuracy:
  - Green: MAE < 3 km/h
  - Yellow: 3-5 km/h
  - Red: > 5 km/h
- Comparison bars: STMGT vs Google Maps baseline

#### Figure 3: Static Map with Node Colors

**File:** `demo_output/traffic_map.html` (interactive) or `.png` (static)

**Content:**

- Google Maps centered on HCMC
- Nodes colored by prediction error:
  - Green: Error < 2 km/h (accurate)
  - Yellow: 2-5 km/h (moderate)
  - Red: > 5 km/h (poor)
- Edge lines with gradient showing speed changes
- Tooltip on hover: Edge ID, Predicted, Actual, Error

**Implementation (Folium):**

```python
import folium

m = folium.Map(location=[10.762622, 106.660172], zoom_start=12)

for edge_id, error in errors.items():
    color = 'green' if error < 2 else 'yellow' if error < 5 else 'red'

    folium.Marker(
        location=[node_lat, node_lon],
        popup=f"{edge_id}<br>Error: {error:.1f} km/h",
        icon=folium.Icon(color=color)
    ).add_to(m)

m.save('demo_output/traffic_map.html')
# Or use selenium to export PNG
```

#### Figure 4: Google Maps Baseline Comparison

**File:** `demo_output/google_comparison.png`

**Content:**

- Side-by-side bar chart:
  - STMGT MAE vs Google Maps API baseline
  - Shows improvement percentage
- Table with detailed metrics:
  ```
  Model           MAE    RMSE   R²
  STMGT          3.2    4.8    0.85
  Google Maps    4.7    6.3    0.72
  Improvement    +32%   +24%   +18%
  ```

---

### 5. Google Maps Baseline Integration

#### Why Compare with Google Maps?

**Rationale:**

- Google Maps provides traffic predictions (shown as colored routes)
- Industry-standard baseline
- Shows value of custom STMGT model vs commercial solution

#### Google Maps Directions API

**Endpoint:** `https://maps.googleapis.com/maps/api/directions/json`

**Parameters:**

```python
params = {
    'origin': f'{lat_a},{lon_a}',
    'destination': f'{lat_b},{lon_b}',
    'departure_time': int(timestamp.timestamp()),  # Future time
    'traffic_model': 'best_guess',  # or 'optimistic', 'pessimistic'
    'key': GOOGLE_API_KEY
}
```

**Response:**

```json
{
  "routes": [
    {
      "legs": [
        {
          "duration": { "value": 1200 }, // Normal duration (seconds)
          "duration_in_traffic": { "value": 1800 } // With traffic
        }
      ]
    }
  ]
}
```

**Calculate Google's Predicted Speed:**

```python
def get_google_predicted_speed(edge, future_time):
    """Get Google's traffic prediction for edge at future_time."""
    distance_km = edge['distance_km']

    # Call Directions API
    response = requests.get(GOOGLE_API_URL, params={
        'origin': f"{edge['lat_a']},{edge['lon_a']}",
        'destination': f"{edge['lat_b']},{edge['lon_b']}",
        'departure_time': int(future_time.timestamp()),
        'key': GOOGLE_API_KEY
    })

    data = response.json()
    duration_hours = data['routes'][0]['legs'][0]['duration_in_traffic']['value'] / 3600

    predicted_speed = distance_km / duration_hours
    return predicted_speed
```

**Cost Consideration:**

- Directions API: $5 per 1000 requests (first $200/month free)
- For 100 edges × 3 horizons = 300 requests per demo → $1.50
- Stay within free tier by limiting to sample edges

#### Comparison Implementation

**File:** `scripts/demo/compare_with_google.py`

```python
def compare_models(predictions_df, actuals_df, edges_sample):
    """Compare STMGT vs Google Maps vs Actual."""
    results = []

    for edge_id in edges_sample:
        # STMGT predictions
        stmgt_preds = predictions_df[predictions_df['edge_id'] == edge_id]

        # Google predictions
        google_preds = []
        for horizon in [1, 2, 3]:  # hours
            target_time = demo_start_time + timedelta(hours=horizon)
            google_speed = get_google_predicted_speed(edge, target_time)
            google_preds.append(google_speed)

        # Actual speeds
        actual_speeds = actuals_df[actuals_df['edge_id'] == edge_id]['speed_kmh'].values

        # Calculate errors
        stmgt_mae = np.mean(np.abs(stmgt_preds - actual_speeds))
        google_mae = np.mean(np.abs(google_preds - actual_speeds))

        results.append({
            'edge_id': edge_id,
            'stmgt_mae': stmgt_mae,
            'google_mae': google_mae,
            'improvement': (google_mae - stmgt_mae) / google_mae * 100
        })

    return pd.DataFrame(results)
```

---

### 6. VM Setup & Data Collection

#### Step-by-Step VM Setup

**1. Provision VM (GCP Example):**

```bash
gcloud compute instances create traffic-collector \
    --machine-type=e2-small \
    --zone=asia-southeast1-a \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=20GB
```

**2. SSH and Install Dependencies:**

```bash
ssh traffic-collector

# Install Python and deps
sudo apt update
sudo apt install -y python3.10 python3-pip
pip3 install pandas pyarrow requests python-dotenv

# Create data directory
sudo mkdir -p /opt/traffic_data
sudo chown $USER:$USER /opt/traffic_data
```

**3. Upload Collector Script:**

```bash
# From local machine
scp scripts/deployment/traffic_collector.py traffic-collector:/opt/traffic_data/
scp .env traffic-collector:/opt/traffic_data/  # Contains API keys
```

**4. Setup Cron Job:**

```bash
# On VM
crontab -e

# Add this line (every 15 minutes)
*/15 * * * * cd /opt/traffic_data && python3 collector.py >> collector.log 2>&1
```

**5. Test Collection:**

```bash
cd /opt/traffic_data
python3 collector.py
# Should create traffic_data_202511.parquet

# Check logs
tail -f collector.log
```

#### Monitoring & Maintenance

**Check Data Quality:**

```bash
# On VM
python3 -c "
import pandas as pd
df = pd.read_parquet('traffic_data_202511.parquet')
print(f'Records: {len(df):,}')
print(f'Date range: {df.timestamp.min()} to {df.timestamp.max()}')
print(f'Unique edges: {df.edge_id.nunique()}')
print(f'File size: {Path('traffic_data_202511.parquet').stat().st_size / 1e6:.1f} MB')
"
```

**Download Data for Demo:**

```bash
# From local machine
scp traffic-collector:/opt/traffic_data/traffic_data_202511.parquet data/demo/
```

---

## Implementation Timeline

### Phase 1: VM Setup & Data Collection (3-5 days before demo)

- [ ] Provision VM (e2-small on GCP)
- [ ] Implement `traffic_collector.py` script
- [ ] Setup cron job for 15-minute intervals
- [ ] Test collection for 24 hours
- [ ] Verify data quality (no missing timestamps)
- **Goal:** Collect 3-5 days of continuous data before demo day

### Phase 2: Demo Script Implementation (1-2 days)

- [ ] Create `scripts/demo/generate_demo_figures.py`
- [ ] Implement prediction logic (STMGT inference)
- [ ] Implement wait-for-actual mechanism
- [ ] Add comparison metrics calculation
- [ ] Test with historical data

### Phase 3: Visualization Outputs (1 day)

- [ ] Figure 1: Prediction vs Actual line chart (Matplotlib)
- [ ] Figure 2: Error by horizon bar chart
- [ ] Figure 3: Static map with colored nodes (Folium)
- [ ] Figure 4: Google Maps baseline comparison
- [ ] Export all as high-res PNGs (300 DPI)

### Phase 4: Google Baseline Integration (Optional, 0.5 day)

- [ ] Implement Google Directions API calls
- [ ] Calculate Google's predicted speeds
- [ ] Add to comparison figures
- [ ] Document API cost (stay within free tier)

### Phase 5: Demo Rehearsal (0.5 day)

- [ ] Run full demo script end-to-end
- [ ] Verify all figures generated correctly
- [ ] Prepare PowerPoint with embedded figures
- [ ] Practice 5-minute presentation flow

**Total Time:** 5-7 days (including data collection wait time)

---

## Demo Presentation Flow

**Duration:** 5-7 minutes

**Setup:** PowerPoint with pre-generated figures (NO live coding/running)

### Slide 1: Title (30s)

- Project: STMGT Traffic Forecasting for HCMC
- Team: 3 members
- Objective: Predict traffic 1-3 hours ahead

### Slide 2: Demo Scenario (30s)

- "At 17:00, we predicted traffic for 18:00, 19:00, 20:00"
- "Then waited for actual data to arrive"
- "Now let's see how accurate we were"

### Slide 3: Prediction vs Actual Chart (1.5 min)

- **Show Figure 1** (line chart)
- Point out:
  - Blue dashed = Predicted speeds
  - Green solid = Actual speeds
  - Shaded area = Model uncertainty
- Highlight: "Notice predictions track actual trends closely"

### Slide 4: Accuracy by Horizon (1.5 min)

- **Show Figure 2** (bar chart)
- Metrics:
  - 1h ahead: MAE ~2.8 km/h (excellent)
  - 2h ahead: MAE ~3.5 km/h (good)
  - 3h ahead: MAE ~4.2 km/h (acceptable)
- "Accuracy degrades gracefully with longer horizons"

### Slide 5: Map Visualization (1 min)

- **Show Figure 3** (static map)
- Color explanation:
  - Green nodes: Accurate predictions (<2 km/h error)
  - Yellow: Moderate (2-5 km/h)
  - Red: Challenging areas (>5 km/h)
- "Most nodes are green, showing strong spatial coverage"

### Slide 6: Comparison with Google Maps (1.5 min)

- **Show Figure 4** (comparison table + chart)
- Results:
  ```
  STMGT:       MAE 3.2 km/h
  Google Maps: MAE 4.7 km/h
  Improvement: +32%
  ```
- "Our model outperforms commercial baseline"

### Slide 7: Technical Highlights (1 min)

- **Architecture:** STMGT (Spatio-Temporal Mixture Gaussian Transformer)
- **Key innovations:**
  - Attention mechanism for spatial dependencies
  - Mixture model for uncertainty quantification
  - Weather integration (temperature, humidity, wind)
- **Inference speed:** <100ms on GPU

### Slide 8: System Design (30s - optional)

- **Data:** VM collects traffic every 15 minutes
- **Model:** Trained on 1 month of HCMC data
- **Prediction:** Uses 3-hour lookback window
- "Fully automated, ready for production deployment"

### Slide 9: Q&A

- "Happy to answer questions about methodology, results, or implementation"

---

## Key Decisions & Rationale

### Why Static Exports Instead of Live Web?

- **Simplicity:** No server/frontend complexity during demo
- **Reliability:** Pre-generated figures eliminate live failures
- **Portability:** Works in any presentation environment
- **Focus:** Audience sees results, not implementation

### Why Parquet Instead of PostgreSQL?

- **Simpler setup:** No database installation/management
- **Portable:** Easy to copy files to local machine
- **Efficient:** Fast pandas reads, good compression
- **Sufficient:** 1 month data ≈ 2-5 MB

### Why 1-3 Hour Prediction Horizons?

- **Practical use case:** Traffic planning, route optimization
- **Model capacity:** Trained on 12-step sequences (3 hours)
- **Demonstrable accuracy:** Short enough for high accuracy, long enough to be useful

### Why Compare with Google Maps?

- **Industry benchmark:** Establishes credibility
- **Value proposition:** Shows custom model can beat commercial solution
- **Concrete improvement:** Quantifiable advantage (+32% lower MAE)

### Why 15-Minute Collection Intervals?

- **Temporal resolution:** Captures traffic dynamics
- **API limits:** Stays within Overpass/OpenWeather free tiers
- **Storage:** Manageable data volume (9.6K records/day)

---

## Cost & Resource Estimate

### VM Operations (Monthly)

- **GCP e2-small:** $15-20/month (or FREE with e2-micro)
- **Storage (20GB):** $3/month
- **Network egress:** <$5/month
- **Total: ~$20-25/month** (or $8 with free tier)

### API Costs (Per Demo)

- **OpenWeatherMap:** Free (1000 calls/day, demo uses ~96)
- **Google Directions:** $1.50 (300 requests for baseline comparison)
- **Google Maps Static:** Free (within $200/month credit)
- **Total: ~$1.50 per demo run**

### Development Time

- **VM setup & collector:** 4-6 hours
- **Demo script:** 8-12 hours
- **Visualization figures:** 6-8 hours
- **Testing & refinement:** 4-6 hours
- **Total: ~25-35 hours** (3-4 working days)

### Data Requirements

- **Collection period:** 3-5 days before demo (continuous)
- **Data volume:** ~500 KB/day × 5 days = ~2.5 MB
- **Nodes/edges:** 50-100 edges (sufficient for demo)

---

## Advantages of This Approach

### Technical

- ✅ **No deployment complexity:** No web server, databases, SSL
- ✅ **Reproducible:** Re-run script anytime with same data
- ✅ **Testable:** Validate all figures before presentation
- ✅ **Fast iteration:** Tweak visualizations without re-collecting data

### Presentation

- ✅ **No live coding risk:** Everything pre-generated
- ✅ **High-quality figures:** 300 DPI exports look professional
- ✅ **Flexible format:** PowerPoint, PDF, or Jupyter Notebook
- ✅ **Offline capable:** Works without internet at demo venue

### Comparison

- ✅ **Google baseline:** Strong benchmark showing real improvement
- ✅ **Multiple metrics:** MAE, RMSE, R² provide comprehensive view
- ✅ **Spatial visualization:** Map shows geographic strengths/weaknesses
- ✅ **Temporal analysis:** Error by horizon shows degradation pattern

---

## Next Steps

### Week 1: VM & Data Collection

1. **Provision VM** (Day 1)

   ```bash
   gcloud compute instances create traffic-collector --machine-type=e2-small
   ```

2. **Implement Collector** (Day 1-2)

   - Create `scripts/deployment/traffic_collector.py`
   - Test Overpass API scraping
   - Add OpenWeatherMap integration
   - Setup cron job

3. **Start Collection** (Day 2 onwards)
   - Let VM run continuously for 3-5 days
   - Monitor logs daily
   - Verify data quality

### Week 2: Demo Script

4. **Implement Demo Script** (Day 6-7)

   - Create `scripts/demo/generate_demo_figures.py`
   - Add STMGT prediction logic
   - Implement comparison metrics
   - Test with collected data

5. **Generate Visualizations** (Day 8)

   - Figure 1: Line chart (Matplotlib)
   - Figure 2: Bar chart (horizons)
   - Figure 3: Map (Folium)
   - Figure 4: Google comparison

6. **Optional: Google Baseline** (Day 8-9)
   - Implement Directions API calls
   - Add to comparison figures

### Week 3: Presentation

7. **Create PowerPoint** (Day 9)

   - Import all figures
   - Add explanatory text
   - Practice 5-minute flow

8. **Rehearsal** (Day 10)
   - Full run-through with team
   - Time each section
   - Prepare Q&A responses

**Total Timeline:** 10 days (including 3-5 days data collection wait)

---

## File Structure

```
project/
├── scripts/
│   ├── deployment/
│   │   └── traffic_collector.py          # VM collector (cron job)
│   └── demo/
│       ├── generate_demo_figures.py       # Main demo script
│       └── compare_with_google.py         # Google baseline comparison
├── data/
│   └── demo/
│       └── traffic_data_202511.parquet    # Downloaded from VM
├── demo_output/                           # Generated figures
│   ├── comparison_chart.png               # Figure 1: Line chart
│   ├── error_by_horizon.png               # Figure 2: Bar chart
│   ├── traffic_map.html                   # Figure 3: Interactive map
│   ├── traffic_map.png                    # Figure 3: Static export
│   └── google_comparison.png              # Figure 4: Baseline comparison
└── docs/
    ├── DEMO.md                            # This document
    └── presentation.pptx                  # Final presentation
```

---

## Example Usage

### On VM (Continuous Collection)

```bash
# Setup (once)
ssh traffic-collector
pip3 install pandas pyarrow requests python-dotenv
crontab -e  # Add: */15 * * * * python3 /opt/traffic_data/collector.py

# Monitor
tail -f /opt/traffic_data/collector.log
```

### On Local Machine (Demo Generation)

```bash
# Download data from VM
scp traffic-collector:/opt/traffic_data/traffic_data_202511.parquet data/demo/

# Run demo script (at 17:00 on demo day)
python scripts/demo/generate_demo_figures.py \
    --data data/demo/traffic_data_202511.parquet \
    --model outputs/stmgt_v3_production/best_model.pt \
    --predict-from "2025-11-20 17:00" \
    --predict-to "2025-11-20 20:00" \
    --output demo_output/ \
    --wait-for-actual  # Blocks until 20:00

# Script outputs:
# ✓ demo_output/comparison_chart.png
# ✓ demo_output/error_by_horizon.png
# ✓ demo_output/traffic_map.html
# ✓ demo_output/metrics.json
```

---

## References

- **Overpass API:** https://overpass-api.de/
- **OpenWeatherMap API:** https://openweathermap.org/api
- **Google Directions API:** https://developers.google.com/maps/documentation/directions
- **Folium (Maps):** https://python-visualization.github.io/folium/
- **Matplotlib:** https://matplotlib.org/stable/gallery/index.html
- **Pandas + Parquet:** https://pandas.pydata.org/docs/user_guide/io.html#parquet

---

**Document Version:** 2.0 (Simplified Export-Based Demo)  
**Last Updated:** November 15, 2025  
**Author:** THAT Le Quang
