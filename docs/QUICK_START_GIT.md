# Quick Start Guide - Git-Based Workflow

## 🚀 For Developers

### Initial Setup
```bash
# Clone the repository
git clone https://github.com/thatlq1812/dsp391m_project.git
cd dsp391m_project

# Setup Python environment
conda env create -f environment.yml
conda activate dsp
```

### Make Changes & Deploy
```bash
# 1. Make your changes locally
vim configs/project_config.yaml

# 2. Test locally (optional)
python scripts/collect_once.py

# 3. Commit changes
git add .
git commit -m "Your commit message"

# 4. Deploy to VM (auto push + pull + restart)
./scripts/deployment/deploy_git.sh
```

### Monitor System
```bash
# Check system status
./scripts/deployment/status.sh

# View collection statistics
./scripts/monitoring/view_stats.sh

# Monitor real-time logs
./scripts/deployment/monitor_logs.sh

# Health check
./scripts/monitoring/health_check_remote.sh

# Restart service if needed
./scripts/deployment/restart.sh
```

---

## 📊 For Team Members (Data Access)

### Download Latest Data (with gcloud)
```bash
# Download latest run
./scripts/data/download_latest.sh

# Choose what to download:
#   1) Latest run only
#   2) Last 10 runs
#   3) Last 24 hours
#   4) All data
```

### Download Data (Simple - No Auth Required)
```bash
# Coming soon: Direct HTTP download
./scripts/data/download_simple.sh

# Admin needs to start data server on VM first:
# ssh to VM and run: ./scripts/data/serve_data_public.sh
```

---

## 🎯 Common Tasks

### Check if System is Running
```bash
./scripts/deployment/status.sh
```

### View Recent Collections
```bash
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a --command="
    ls -lht ~/traffic-forecast/data/runs/ | head -10
"
```

### Download Specific Run
```bash
RUN_ID="run_20251030_032440"
gcloud compute scp \
    --zone=asia-southeast1-a \
    --recurse \
    traffic-forecast-collector:~/traffic-forecast/data/runs/$RUN_ID \
    ./downloaded_data/
```

### Emergency Stop
```bash
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a --command="
    sudo systemctl stop traffic-collector
"
```

### Emergency Restart
```bash
./scripts/deployment/restart.sh
```

---

## 📁 Project Structure

```
dsp391m_project/
├── configs/
│   └── project_config.yaml       # Main configuration
├── scripts/
│   ├── deployment/               # Deployment tools
│   │   ├── deploy_git.sh        # Deploy from Git
│   │   ├── status.sh            # Check system status
│   │   ├── monitor_logs.sh      # Real-time logs
│   │   └── restart.sh           # Restart service
│   ├── data/                     # Data download tools
│   │   ├── download_latest.sh   # Download with gcloud
│   │   └── download_simple.sh   # Simple HTTP download
│   └── monitoring/               # Monitoring tools
│       ├── view_stats.sh        # Collection statistics
│       └── health_check_remote.sh # Health check
├── traffic_forecast/             # Main package
└── data/runs/                    # Collected data (on VM)
```

---

## ⚙️ Configuration

### Adaptive Scheduler (configs/project_config.yaml)

**Peak Hours** (30 min intervals):
- 06:30-08:00: Morning rush
- 10:30-11:30: Lunch rush
- 16:00-19:00: Evening rush

**Off-Peak Hours** (120 min intervals):
- 08:00-10:30, 11:30-16:00, 19:00-22:00, 22:00-06:30

**Coverage**:
- Radius: 4096m
- Nodes: 64 quality intersections
- Edges: ~144 traffic routes

---

## 🔧 Troubleshooting

### Service Not Running
```bash
./scripts/deployment/status.sh
./scripts/deployment/restart.sh
```

### No Recent Collections
```bash
# Check logs
./scripts/deployment/monitor_logs.sh

# Check health
./scripts/monitoring/health_check_remote.sh
```

### Config Changes Not Applied
```bash
# Re-deploy
./scripts/deployment/deploy_git.sh

# This will:
# 1. Push to GitHub
# 2. Pull on VM
# 3. Regenerate topology if config changed
# 4. Restart service
```

---

## 📞 Support

- **GitHub**: https://github.com/thatlq1812/dsp391m_project
- **VM**: traffic-forecast-collector (asia-southeast1-a)
- **SSH**: `gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a`

---

## 📈 Data Collection Info

- **Duration**: Continuous (3+ days)
- **Frequency**: ~150-200 collections/day
- **API Calls**: ~3,800/day
- **Cost**: ~$21/day
- **Data Size**: ~50MB/day
- **Retention**: All runs kept (cleanup manually if needed)
