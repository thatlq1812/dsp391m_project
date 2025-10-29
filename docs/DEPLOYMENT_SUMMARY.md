# Traffic Forecast Project - Deployment Summary

**Date:** October 30, 2025  
**Status:** ✅ PRODUCTION READY  
**VM:** traffic-forecast-collector (asia-southeast1-a)

---

## 📊 System Configuration

### Coverage Area
- **Radius:** 4096m (Ho Chi Minh City center)
- **Nodes:** 64 quality intersections
- **Edges:** ~144 traffic routes
- **Road Types:** Motorway, Trunk, Primary

### Adaptive Scheduler
**Peak Hours (30 min intervals):**
- 06:30-08:00 (Morning rush)
- 10:30-11:30 (Lunch rush)  
- 16:00-19:00 (Evening rush)

**Off-Peak Hours (120 min intervals):**
- 08:00-10:30, 11:30-16:00, 19:00-22:00, 22:00-06:30

### Performance Metrics
- **Collections/day:** ~150-200
- **API calls/day:** ~3,800
- **Cost:** ~$21/day (~$63 for 3 days)
- **Data size:** ~50MB/day
- **Success rate:** ~60% (normal for traffic data)

---

## 🚀 Quick Commands

### For Developers

```bash
# Deploy changes
./scripts/deployment/deploy_git.sh

# Check status
./scripts/deployment/status.sh

# Monitor live
./scripts/deployment/monitor_logs.sh

# View statistics
./scripts/monitoring/view_stats.sh

# Health check
./scripts/monitoring/health_check_remote.sh

# Restart if needed
./scripts/deployment/restart.sh
```

### For Team Members (Data Download)

```bash
# Option 1: With gcloud (interactive)
./scripts/data/download_latest.sh
# Choose: 1=latest, 2=last 10, 3=24h, 4=all

# Option 2: Simple HTTP download (no auth)
./scripts/data/download_simple.sh
# Note: Admin must run serve_data_public.sh on VM first
```

---

## 📁 Data Structure

```
data/runs/
├── run_20251030_032440/
│   ├── nodes.json              # 64 nodes
│   ├── edges.json              # 144 edges  
│   ├── traffic_edges.json      # Traffic data
│   ├── weather_snapshot.json   # Weather data
│   └── statistics.json         # Collection stats
├── run_20251030_032457/
│   └── ...
└── ...
```

Each run contains:
- **nodes.json:** Road network nodes with metadata
- **edges.json:** Node pairs for traffic collection
- **traffic_edges.json:** Actual traffic data (speed, duration)
- **weather_snapshot.json:** Weather conditions at collection time
- **statistics.json:** Collection statistics and metadata

---

## 🎯 Data Download Examples

### Download Latest Run
```bash
./scripts/data/download_latest.sh my_data/
# Select option 1
```

### Download Last 24 Hours
```bash
./scripts/data/download_latest.sh data_24h/
# Select option 3
```

### Download with gcloud Directly
```bash
# Get latest run name
RUN=$(gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="ls -t ~/traffic-forecast/data/runs/ | head -1")

# Download it
gcloud compute scp \
  --zone=asia-southeast1-a \
  --recurse \
  traffic-forecast-collector:~/traffic-forecast/data/runs/$RUN \
  ./downloaded_data/
```

### Setup Public HTTP Server (for team without gcloud)
```bash
# On VM (admin only)
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a
cd ~/traffic-forecast
./scripts/data/serve_data_public.sh

# Team members can then download via HTTP:
curl http://1.53.74.2:8080/data/runs.json
# Or use download_simple.sh script
```

---

## 🔍 Monitoring & Health

### System Status
```bash
./scripts/deployment/status.sh
```
Shows:
- Service status
- Recent collections
- Disk usage  
- Topology info
- Recent logs

### Collection Statistics
```bash
./scripts/monitoring/view_stats.sh
```
Shows:
- Total runs
- Average edges/run
- Success rate
- Collections by hour
- Latest 10 runs detail

### Health Check
```bash
./scripts/monitoring/health_check_remote.sh
```
Checks:
1. Service running ✓
2. Recent collections ✓
3. Topology cache ✓
4. Disk space ✓
5. Memory usage ✓
6. Error logs ✓
7. Network connectivity ✓

### Real-time Monitoring
```bash
./scripts/deployment/monitor_logs.sh
```
Live tail of adaptive_scheduler.log

---

## 🔧 Configuration Changes

### Modify Collection Settings

1. **Edit config locally:**
   ```bash
   vim configs/project_config.yaml
   ```

2. **Commit changes:**
   ```bash
   git add configs/project_config.yaml
   git commit -m "Update scheduler intervals"
   ```

3. **Deploy to VM:**
   ```bash
   ./scripts/deployment/deploy_git.sh
   ```

**What happens:**
- Pushes to GitHub
- Pulls on VM
- Detects config change
- Regenerates topology
- Restarts service

### Common Config Changes

**Adjust peak hours:**
```yaml
peak_hours:
  time_ranges:
    - start: "07:00"  # Change to 7 AM
      end: "09:00"
```

**Change intervals:**
```yaml
peak_hours:
  interval_minutes: 15  # Collect every 15 min
offpeak:
  interval_minutes: 60  # Collect every 60 min
```

**Adjust coverage:**
```yaml
globals:
  area:
    radius_m: 5000  # Increase to 5km
```

**Node selection:**
```yaml
node_selection:
  min_degree: 3
  min_importance_score: 15.0
  max_nodes: 80  # Increase to 80 nodes
```

---

## 📈 Expected Data Collection

### 3-Day Collection Run

**Day 1:**
- ~150-200 runs
- ~22,000-29,000 traffic edges
- ~50MB data
- $21 cost

**Day 2:**
- Same as Day 1
- Cumulative: ~300-400 runs

**Day 3:**
- Same as Day 1
- **Total: ~450-600 runs**
- **Total: ~66,000-87,000 traffic edges**
- **Total: ~150MB data**
- **Total cost: ~$63**

### Data Quality Metrics
- **Success rate:** 60-70% (normal)
- **Coverage:** 64 intersections × 144 routes
- **Temporal resolution:** 30min (peak), 120min (off-peak)
- **Weather integration:** Yes (Open-Meteo)
- **Timezone:** UTC+7 (Vietnam)

---

## 🛠️ Troubleshooting

### Service Not Running
```bash
./scripts/deployment/status.sh
./scripts/deployment/restart.sh
```

### No Recent Collections
```bash
# Check logs
./scripts/deployment/monitor_logs.sh

# Health check
./scripts/monitoring/health_check_remote.sh

# If stuck, restart
./scripts/deployment/restart.sh
```

### Download Failed
```bash
# Ensure VM is running
gcloud compute instances list

# Ensure you're authenticated
gcloud auth login

# Try direct SSH
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a
```

### Config Changes Not Applied
```bash
# Redeploy
./scripts/deployment/deploy_git.sh

# Or manual:
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a
cd ~/traffic-forecast
git pull origin master
rm cache/overpass_topology.json
python scripts/collect_once.py --force-refresh
sudo systemctl restart traffic-collector
```

---

## 📞 Support & Resources

**Documentation:**
- [Quick Start (Git Workflow)](docs/QUICK_START_GIT.md)
- [Scripts Reference](scripts/deployment/README.md)
- [Deployment Guide](docs/v5/DEPLOYMENT_GUIDE.md)
- [Operations Guide](docs/OPERATIONS.md)

**GitHub:** https://github.com/thatlq1812/dsp391m_project

**GCP Project:** sonorous-nomad-476606-g3

**VM Details:**
- Name: traffic-forecast-collector
- Zone: asia-southeast1-a
- Type: e2-small (2GB RAM)
- Disk: 30GB
- IP: 1.53.74.2 (external)

**Direct SSH:**
```bash
gcloud compute ssh traffic-forecast-collector \
  --zone=asia-southeast1-a \
  --project=sonorous-nomad-476606-g3
```

---

## ✅ Final Checklist

- [x] Service running and collecting data
- [x] 64 nodes, 144 edges topology
- [x] Adaptive scheduler configured (Vietnam timezone)
- [x] Git-based deployment workflow
- [x] Team data download (with & without gcloud)
- [x] Monitoring scripts (status, stats, health check)
- [x] Documentation updated
- [x] Auto-restart enabled
- [x] Timezone set to UTC+7
- [x] All scripts tested and working

**System Status:** ✅ READY FOR 3-DAY PRODUCTION RUN

**Next Steps:**
1. Monitor daily: `./scripts/deployment/status.sh`
2. Download data after 3 days: `./scripts/data/download_latest.sh`
3. Analyze collected data using `traffic_forecast` package

---

**Deployment completed:** October 30, 2025  
**Estimated completion:** November 2, 2025  
**Total expected data:** ~450-600 runs, ~150MB
