# Traffic Forecast v5.1 - Quick Start Guide

Production-ready traffic forecasting with adaptive scheduling and cost optimization.

## 🚀 Quick Start (5 minutes)

### 1. Setup Environment

```bash
# Clone and enter project
git clone <repo-url>
cd project

# Create conda environment
conda env create -f environment.yml
conda activate dsp

# Install package
pip install -e .

# Configure API key
cp .env.example .env
# Edit .env and add your GOOGLE_MAPS_API_KEY
```

### 2. Run Single Collection

```bash
# Test collection (uses cached topology)
python scripts/collect_once.py

# Results in: data/runs/run_YYYYMMDD_HHMMSS/
```

### 3. Use Interactive Control Panel

```bash
# Local development dashboard
bash scripts/control_panel.sh

# Options:
#   1-4: Data collection (single, test, adaptive scheduler)
#   5-8: Data management (view, merge, cleanup, export)
#   9-12: Visualization
#   13-16: Testing & debugging
```

## 🌐 Deploy to GCP (Production)

### Option A: Interactive Wizard (Recommended)

```bash
bash scripts/deploy_wizard.sh

# Select:
#   A) AUTO: Full deployment (steps 1-7)
#   or step-by-step: 1→2→3→4→5→6→7
```

### Option B: Manual Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.

## 📊 Adaptive Scheduling

**Cost-optimized collection** (40% savings vs constant interval):

- **Peak hours** (6-9 AM, 4-7 PM): 15 min intervals
- **Off-peak** (9 AM-4 PM, 7-10 PM): 60 min intervals  
- **Night** (10 PM-6 AM): 120 min intervals

**3-day production run:**
- ~150 collections
- ~35,100 data points
- ~$45 total cost

## 📁 Project Structure

```
project/
├── traffic_forecast/     # Main Python package
│   ├── collectors/       # Data collectors (Google, Weather, Overpass)
│   ├── scheduler/        # Adaptive scheduler
│   ├── models/           # Data models
│   └── ml/               # Machine learning (future)
├── scripts/              # Utility scripts
│   ├── control_panel.sh  # Local development dashboard
│   ├── deploy_wizard.sh  # GCP deployment wizard
│   └── collect_once.py   # Single collection script
├── configs/              # Configuration files
│   └── project_config.yaml
├── data/                 # Data storage
│   └── runs/             # Collection runs
├── cache/                # Cached data (topology, weather grid)
└── docs/                 # Documentation
```

## 🔧 Configuration

Edit `configs/project_config.yaml`:

```yaml
scheduler:
  mode: adaptive  # or 'fixed'
  adaptive:
    peak_hours:
      interval_minutes: 15
    offpeak:
      interval_minutes: 60
    night:
      interval_minutes: 120
```

## 📚 Learn More

- [DEPLOYMENT.md](DEPLOYMENT.md) - Full deployment guide
- [OPERATIONS.md](OPERATIONS.md) - Daily operations & monitoring
- [CHANGELOG.md](CHANGELOG.md) - Version history
- [configs/project_config.yaml](configs/project_config.yaml) - Configuration reference

## 🆘 Troubleshooting

### Collection fails
```bash
# Check API key
grep GOOGLE_MAPS_API_KEY .env

# Test Google API
python tools/test_google_limited.py

# Test weather API
python -c "from traffic_forecast.collectors.weather_collector import WeatherCollector; import asyncio; asyncio.run(WeatherCollector(None).collect())"
```

### Deployment issues
```bash
# Check prerequisites
gcloud --version
gcloud auth list

# Verify cache files
ls -lh cache/
python tools/check_edges.py
```

## 📞 Support

- Issues: GitHub Issues
- Documentation: `docs/` directory
- Examples: `scripts/` directory

---

**Traffic Forecast v5.1** - Adaptive Scheduling • Cost Optimized • Production Ready
