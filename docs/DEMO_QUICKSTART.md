# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Demo Quick Start Guide

Step-by-step guide to deploy VM, collect data, and generate demo figures.

## Timeline

- **VM Setup:** 1 hour
- **Data Collection:** 3-5 days (automatic)
- **Figure Generation:** 1 hour
- **Total:** ~5-7 days

## Prerequisites

1. **GCP Account** with billing enabled
2. **gcloud CLI** installed and configured
3. **API Keys:**
   - Google Maps API key (REQUIRED)
   - OpenWeatherMap API key (optional)

## Step 1: Deploy VM (1 hour)

### 1.1 Configure GCP Project

```bash
# Check current project
gcloud config get-value project

# Set project (if needed)
gcloud config set project sonorous-nomad-476606-g3

# Set default zone
gcloud config set compute/zone asia-southeast1-a
```

### 1.2 Run Deployment Script

```bash
cd /d/UNI/DSP391m/project
./scripts/deployment/deploy_demo_vm.sh
```

The script will:

- ✅ Create e2-micro VM (FREE tier)
- ✅ Install Miniconda and Python 3.10
- ✅ Clone repository
- ✅ Setup systemd timer (every 15 minutes)

**Time:** ~5 minutes

### 1.3 Configure API Keys

```bash
# SSH into VM
gcloud compute ssh traffic-demo-collector \
    --zone=asia-southeast1-a \
    --project=sonorous-nomad-476606-g3

# Edit .env file
cd ~/traffic-demo
nano .env
```

Add your keys:

```bash
GOOGLE_MAPS_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
OPENWEATHER_API_KEY=your_key_here  # Optional
```

Save: `Ctrl+O`, `Enter`, `Ctrl+X`

### 1.4 Build Topology Cache

```bash
# Still on VM
conda activate dsp
python scripts/data/01_collection/build_topology.py
```

This creates `cache/overpass_topology.json` with edge information.

**Time:** ~2-3 minutes

### 1.5 Start Collection Service

```bash
# Enable timer (start on boot)
sudo systemctl enable traffic-collector.timer

# Start timer now
sudo systemctl start traffic-collector.timer

# Check status
systemctl status traffic-collector.timer
```

Should see: `Active: active (waiting)`

### 1.6 Verify First Collection

```bash
# Wait 1-2 minutes, then check log
tail -f /opt/traffic_data/collector.log
```

Expected output:

```
2025-11-15 14:00:00 - INFO - ================================================================================
2025-11-15 14:00:00 - INFO - Starting traffic collection at 2025-11-15 14:00:00
2025-11-15 14:00:01 - INFO - Loaded 50 edges from topology
2025-11-15 14:00:02 - INFO - Fetching weather data...
2025-11-15 14:00:03 - INFO - Weather: 28.5°C, Clear
2025-11-15 14:00:03 - INFO - Collecting traffic data for 50 edges...
...
2025-11-15 14:02:30 - INFO - ✓ Collection completed
```

Press `Ctrl+C` to exit log view.

### 1.7 Exit VM

```bash
exit  # Logout from VM
```

**✅ VM is now collecting data every 15 minutes automatically!**

## Step 2: Wait for Data (3-5 days)

VM runs automatically. No action needed.

### Optional: Monitor Progress

```bash
# SSH back to VM
gcloud compute ssh traffic-demo-collector --zone=asia-southeast1-a

# Check data file
ls -lh /opt/traffic_data/

# Count records
conda activate dsp
python -c "
import pandas as pd
df = pd.read_parquet('/opt/traffic_data/traffic_data_202511.parquet')
print(f'Records: {len(df):,}')
print(f'Time range: {df.timestamp.min()} to {df.timestamp.max()}')
"

exit
```

### Expected Data Growth

- **Per hour:** 4 collections × 50 edges = 200 records
- **Per day:** 200 × 24 = 4,800 records
- **After 5 days:** 24,000 records (~2-5 MB compressed)

## Step 3: Download Data (5 minutes)

After 3-5 days of collection:

```bash
# Create demo data directory
cd /d/UNI/DSP391m/project
mkdir -p data/demo

# Download parquet file
gcloud compute scp \
    traffic-demo-collector:/opt/traffic_data/traffic_data_202511.parquet \
    ./data/demo/ \
    --zone=asia-southeast1-a \
    --project=sonorous-nomad-476606-g3
```

**✅ Data downloaded to `data/demo/traffic_data_202511.parquet`**

## Step 4: Generate Demo Figures (30 minutes)

### 4.1 Verify STMGT Model

```bash
# Check if model exists
ls outputs/stmgt_v2_20251110_123931/best_model.pt

# If not found, use latest model
ls outputs/stmgt_v*/best_model.pt | tail -1
```

### 4.2 Run Demo Script

```bash
# Basic run (no Google comparison)
python scripts/demo/generate_demo_figures.py \
    --data data/demo/traffic_data_202511.parquet \
    --model outputs/stmgt_v2_20251110_123931/best_model.pt \
    --demo-time "2025-11-20 17:00" \
    --output demo_output/

# With Google baseline comparison
python scripts/demo/generate_demo_figures.py \
    --data data/demo/traffic_data_202511.parquet \
    --model outputs/stmgt_v2_20251110_123931/best_model.pt \
    --demo-time "2025-11-20 17:00" \
    --include-google \
    --output demo_output/
```

**⚠️ NOTE:** Script currently has TODO items for:

- Model loading
- Prediction logic
- Metric calculations

These need to be implemented first! (See TODO section below)

### 4.3 Output Files

```bash
# Check output
ls -lh demo_output/

# Expected files:
# - figure1_multi_prediction.png (300 DPI)
# - figure2_variance_analysis.png (300 DPI)
# - figure3_traffic_map.html (interactive)
# - figure4_google_comparison.png (300 DPI, if --include-google)
# - metrics.json (raw data)
```

## Step 5: Prepare Presentation (30 minutes)

1. **Open PowerPoint**
2. **Insert figures:**
   - Slide 1: Title + Overview
   - Slide 2: Figure 1 (Multi-prediction convergence)
   - Slide 3: Figure 2 (Variance analysis)
   - Slide 4: Figure 3 screenshot (Traffic map)
   - Slide 5: Figure 4 (Google comparison)
   - Slide 6: Conclusions
3. **Add explanatory text**
4. **Practice presentation (5-7 minutes)**

## Cleanup (Optional)

### Stop VM (Keep Data)

```bash
# Stop VM to save costs
gcloud compute instances stop traffic-demo-collector \
    --zone=asia-southeast1-a \
    --project=sonorous-nomad-476606-g3

# Start again later
gcloud compute instances start traffic-demo-collector \
    --zone=asia-southeast1-a \
    --project=sonorous-nomad-476606-g3
```

### Delete VM (Remove Everything)

```bash
# Backup data first!
gcloud compute scp \
    traffic-demo-collector:/opt/traffic_data/*.parquet \
    ./data/demo/ \
    --zone=asia-southeast1-a \
    --project=sonorous-nomad-476606-g3

# Delete VM
gcloud compute instances delete traffic-demo-collector \
    --zone=asia-southeast1-a \
    --project=sonorous-nomad-476606-g3 \
    --quiet
```

## Troubleshooting

### Issue: API key not working

```bash
# Test API key
curl "https://maps.googleapis.com/maps/api/directions/json?origin=10.762622,106.660172&destination=10.772622,106.670172&key=YOUR_KEY"

# Should return JSON with "status": "OK"
```

### Issue: No data collected

```bash
# Check service status
systemctl status traffic-collector.timer
systemctl status traffic-collector.service

# Check logs
tail -50 /opt/traffic_data/collector.log
tail -50 /opt/traffic_data/collector_error.log

# Run manually to see errors
cd ~/traffic-demo
conda activate dsp
python scripts/deployment/traffic_collector.py
```

### Issue: Topology file missing

```bash
# Rebuild topology
cd ~/traffic-demo
conda activate dsp
python scripts/data/01_collection/build_topology.py

# Verify
ls -lh cache/overpass_topology.json
```

### Issue: Out of disk space

```bash
# Check disk usage
df -h

# Remove old logs
rm /opt/traffic_data/collector.log
rm /opt/traffic_data/collector_error.log

# Download and compress old data
gcloud compute scp traffic-demo-collector:/opt/traffic_data/traffic_data_202510.parquet ./
gcloud compute ssh traffic-demo-collector --command="gzip /opt/traffic_data/traffic_data_202510.parquet"
```

## TODO: Implement Demo Script

Before running `generate_demo_figures.py`, implement:

1. **Model Loading** (`load_stmgt_model()`)

   - Load checkpoint from .pt file
   - Initialize STMGTPredictor
   - Handle device (CPU/GPU)

2. **Prediction Logic** (`make_predictions()`)

   - Prepare 3-hour lookback window
   - Format data for STMGT input
   - Run predictions for each horizon
   - Extract speeds and uncertainties

3. **Metric Calculations** (`calculate_metrics()`)

   - MAE, RMSE, R² for each horizon
   - Per-edge errors
   - Variance across prediction points
   - Convergence analysis

4. **Map Visualization** (`generate_figure3_map()`)
   - Load node coordinates
   - Calculate per-edge accuracy
   - Color edges by error level
   - Add markers and legend

See `scripts/demo/generate_demo_figures.py` for detailed TODO comments.

## Cost Estimate

- **e2-micro (FREE tier):** $0/month (if under quota)
- **e2-micro (paid):** ~$8/month
- **e2-small:** ~$15-25/month
- **Storage (30GB):** ~$1/month
- **Network egress:** ~$0.5/month

**Total:** FREE or $10-30/month depending on tier

## References

- **VM Commands:** `scripts/deployment/vm_commands.md`
- **Demo Architecture:** `docs/DEMO_BACKPREDICT.md`
- **Traffic Collector:** `scripts/deployment/traffic_collector.py`
- **Demo Script:** `scripts/demo/generate_demo_figures.py`

## Support

For issues, check:

1. VM logs: `/opt/traffic_data/collector.log`
2. Service status: `systemctl status traffic-collector.timer`
3. GitHub Issues: https://github.com/thatlq1812/dsp391m_project/issues

---

**Author:** THAT Le Quang  
**Date:** November 15, 2025
