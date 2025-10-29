# Deployment Guide - Traffic Forecast v5.0

## Overview

This guide covers deploying the Traffic Forecast system to a **Google Cloud Platform (GCP) VM** for continuous data collection over 7-14 days.

---

## Deployment Objectives

1. **Automated Data Collection**: Collect traffic data every 15 minutes (96 times/day)
2. **Data Quality**: Ensure 100% success rate with real Google Directions API
3. **Cost Efficiency**: Stay within $200 free GCP credit (~18 days of collection)
4. **Reliability**: Use systemd + cron for fault-tolerant scheduling

---

## What You Need

### 1. GCP Account

- Sign up at [cloud.google.com](https://cloud.google.com)
- Enable $300 free credit (no charges during trial)

### 2. API Keys

- **Google Maps Directions API** key with billing enabled
- Already configured in `.env` file

### 3. Local Preparation

- 78 nodes cached (`cache/overpass_topology.json`)
- Config ready (`configs/project_config.yaml`)
- Collection tested locally (234/234 edges successful)

---

## GCP VM Setup

### Step 1: Create VM Instance

1. Go to **Compute Engine → VM Instances**
2. Click **Create Instance**
3. Configure:

```
Name: traffic-forecast-collector
Region: asia-southeast1 (Singapore) - closest to HCMC
Zone: asia-southeast1-a
Machine type: e2-micro (0.25-1 vCPU, 1GB RAM) - FREE TIER
Boot disk: Ubuntu 22.04 LTS (10GB standard persistent disk)
```

4. **Allow HTTP/HTTPS traffic** (if running dashboard)
5. Click **Create**

### Step 2: Connect to VM

```bash
# From local machine
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a

# Or use SSH in browser from GCP Console
```

---

## Deployment Process

### Option A: Automated Deployment (Recommended)

1. **Upload project files to VM**:

```bash
# From local machine
gcloud compute scp --recurse \
~/project/traffic_forecast \
~/project/configs \
~/project/cache \
~/project/.env \
~/project/scripts \
~/project/requirements.txt \
~/project/environment.yml \
traffic-forecast-collector:~/traffic-forecast/
```

2. **Run deployment script**:

```bash
# On VM
cd ~/traffic-forecast
chmod +x scripts/deploy_gcp_vm.sh
./scripts/deploy_gcp_vm.sh
```

This script will:

- Install Miniconda
- Create conda environment
- Install dependencies
- Test API connection
- Setup cron job (every 15 minutes)
- Create systemd timer (backup scheduling)

### Option B: Manual Deployment

<details>
<summary>Click to expand manual steps</summary>

```bash
# 1. Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# 2. Create environment
conda create -n dsp python=3.9 -y
conda activate dsp
pip install -r requirements.txt

# 3. Test collection
export GOOGLE_TEST_LIMIT=5
python traffic_forecast/collectors/google/collector.py

# 4. Setup cron
crontab -e
# Add this line:
*/15 * * * * cd ~/traffic-forecast && conda run -n dsp python traffic_forecast/collectors/google/collector.py >> logs/collection.log 2>&1
```

</details>

---

## Using CONTROL_PANEL Notebook

Open `notebooks/CONTROL_PANEL.ipynb` for **full pipeline control**:

### Available Functions

```python
# 1. Check configuration
update_config('collection', 'batch_size', 10)

# 2. View topology
refresh_topology(force=True)

# 3. Run collection
run_collection_test() # Test with 5 edges
run_collection_full() # All 234 edges

# 4. Validate data
validate_collection()

# 5. Visualize
plot_node_coverage()
plot_speed_distribution()

# 6. Estimate costs
estimate_api_costs(collections_per_day=96)

# 7. Create deployment package
create_deployment_package()

# 8. Generate cron script
create_cron_script()

# 9. Check logs
check_logs(last_n_lines=50)
```

---

## Cost Estimation

### Current Configuration

- **Nodes**: 78
- **Edges per collection**: 234 (78 nodes × 3 neighbors)
- **Frequency**: Every 15 minutes = 96 collections/day
- **API Cost**: $0.005 per request

### Daily Breakdown

```
Daily Requests: 234 × 96 = 22,464 requests
Daily Cost: 22,464 × $0.005 = $112.32
```

### Monthly Projection

```
Monthly Requests: 22,464 × 30 = 673,920 requests
Monthly Cost: $112.32 × 30 = $3,369.60
```

### Free Credit Coverage

```
$200 credit ÷ $112.32/day = 1.78 days
```

**IMPORTANT**: Free credit only covers ~2 days at this frequency!

### Recommended Adjustments

**For 7-day collection within $200 budget:**

```
Budget: $200
Days: 7
Daily budget: $200 / 7 = $28.57

Required collections/day: $28.57 / ($0.005 × 234) = 24.4 ≈ 24 collections/day
Interval: 24 hours / 24 = Every 60 minutes (hourly)
```

**Update in notebook:**

```python
# Option 1: Hourly collection
update_config('collection', 'interval_minutes', 60)

# Option 2: Peak hours only (7-9 AM, 5-7 PM = 4 hours/day, every 10 min = 24 collections)
# Manual cron setup required
```

---

## Monitoring

### Check Collection Status

**From VM:**

```bash
# View recent collections
tail -n 50 ~/traffic-forecast/logs/collection.log

# Check cron jobs
crontab -l

# Check systemd timer
sudo systemctl status traffic-collection.timer

# Count collected data
ls -lh ~/traffic-forecast/data/
```

**From Notebook:**

```python
# In CONTROL_PANEL.ipynb
validate_collection() # Check data quality
check_logs(last_n_lines=100) # View logs
```

### Data Quality Metrics

Monitor in collection logs:

- **Success Rate**: Should be 100%
- **Speed Range**: 10-60 km/h (HCMC traffic)
- **Timestamp Coverage**: No gaps > 15 minutes
- **API Errors**: Should be 0

---

## Troubleshooting

### Issue: API REQUEST_DENIED

**Solution:**

1. Check API key in `.env`
2. Verify Directions API is enabled in GCP Console
3. Confirm billing account is active
4. Test with single request:

```bash
export GOOGLE_TEST_LIMIT=1
python traffic_forecast/collectors/google/collector.py
```

### Issue: Cron Not Running

**Solution:**

```bash
# Check cron service
sudo systemctl status cron

# View cron logs
grep CRON /var/log/syslog

# Manually trigger
cd ~/traffic-forecast
conda run -n dsp python traffic_forecast/collectors/google/collector.py
```

### Issue: Out of Memory

**Solution:**

1. Upgrade VM to e2-small (2GB RAM)
2. Or reduce batch size:

```python
update_config('collection', 'batch_size', 5)
```

---

## Data Collection Timeline

### Recommended Schedule

**Week 1 (Days 1-7):**

- Collect every 60 minutes (24/day)
- Cost: ~$28/day
- Total: ~$196 for 7 days Within budget

**Data Downloaded:**

- 7 days × 24 collections = 168 snapshots
- 168 × 234 edges = 39,312 traffic measurements
- Sufficient for temporal patterns

### Data Directory Structure

```
data/
├── downloads/
│ ├── download_20251029_100000/
│ │ ├── traffic_edges.json
│ │ ├── normalized_traffic.json
│ │ └── metadata.json
│ ├── download_20251029_110000/
│ └── ...
├── traffic_edges.json # Latest collection
└── statistics.json
```

---

## Deployment Checklist

### Pre-Deployment

- [x] Topology cached (78 nodes)
- [x] API key tested locally
- [x] Config finalized (radius 2048m, min_distance 200m)
- [x] Cost estimated ($28/day for 7 days)

### During Deployment

- [ ] Create GCP VM (e2-micro, Ubuntu 22.04)
- [ ] Upload project files
- [ ] Run deployment script
- [ ] Test collection (5 edges)
- [ ] Verify cron setup

### Post-Deployment

- [ ] Monitor first 3 collections
- [ ] Check success rate = 100%
- [ ] Validate data format
- [ ] Set calendar reminder to stop VM after 7 days

### After 7 Days

- [ ] Download all data from VM
- [ ] Stop VM to prevent charges
- [ ] Delete VM instance
- [ ] Analyze collected data locally
- [ ] Train ML models

---

## Support

**If you encounter issues:**

1. **Check logs first**:

```python
# In CONTROL_PANEL.ipynb
check_logs(last_n_lines=100)
validate_collection()
```

2. **Common fixes**:

- API errors → Check GCP Console → APIs & Services
- Cron not running → `sudo systemctl restart cron`
- Out of disk → Clean old downloads

3. **Emergency stop**:

```bash
# Disable cron
crontab -r

# Stop systemd timer
sudo systemctl stop traffic-collection.timer
```

---

## Academic Notes

**For DSP391m Project:**

- 7 days of hourly data = **Sufficient for analysis**
- Covers weekdays + weekend patterns
- Captures peak hours (morning/evening rush)
- Within free credit budget

**Data will support:**

- Time series analysis
- Peak hour identification
- Weather correlation
- ML model training
- Cross-validation

---

**Good luck with your deployment! **
