# Traffic Forecast v5.1

Real-time traffic forecasting for Ho Chi Minh City with adaptive scheduling and cost optimization.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Production--Ready-success.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **DSP391m Data Science Project** - Academic project for real-time traffic prediction using Google Directions API, Open-Meteo weather data, and OpenStreetMap topology.

## ✨ Features

- **Adaptive Scheduling**: Peak/off-peak/night intervals (40% cost savings)
- **Smart Caching**: Weather grid caching (95% API reduction), permanent topology cache
- **Wide Coverage**: 4096m radius, 78 filtered nodes, 234 road segments
- **Production Ready**: Automated GCP deployment, systemd service, monitoring
- **Cost Optimized**: ~$45 for 3-day collection, ~$150 for 7 days

## 🚀 Quick Start

### 1. Setup (5 minutes)

```bash
# Clone repository
git clone <repo-url>
cd project

# Create environment
conda env create -f environment.yml
conda activate dsp

# Install package
pip install -e .

# Configure API key
echo "GOOGLE_MAPS_API_KEY=your_api_key_here" > .env
```

### 2. Test Collection

```bash
# Run single collection
python scripts/collect_once.py

# Use interactive dashboard
bash scripts/control_panel.sh
```

### 3. Deploy to GCP

```bash
# Interactive deployment wizard
bash scripts/deploy_wizard.sh

# Select option A for auto deployment
```

## 📊 Adaptive Scheduling

Cost-optimized collection strategy:

| Time Period | Interval | Rationale |
|------------|----------|-----------|
| **Peak** (6-9 AM, 4-7 PM) | 15 min | High traffic variability |
| **Off-peak** (9 AM-4 PM, 7-10 PM) | 60 min | Moderate traffic |
| **Night** (10 PM-6 AM) | 120 min | Stable traffic |

**3-Day Collection:**
- ~150 total collections
- ~35,100 data points
- ~$45 total cost (40% savings)

## 📁 Project Structure

```
project/
├── traffic_forecast/        # Main Python package
│   ├── collectors/          # Data collectors
│   ├── scheduler/           # Adaptive scheduler
│   └── models/              # Data models
├── scripts/                 # Utility scripts
│   ├── control_panel.sh     # Local dashboard
│   ├── deploy_wizard.sh     # GCP deployment
│   ├── collect_once.py      # Single collection
│   └── run_adaptive_collection.py  # Continuous collection
├── configs/                 # Configuration
│   └── project_config.yaml
├── data/                    # Data storage
│   └── runs/                # Collection outputs
├── cache/                   # Cached data
│   ├── overpass_topology.json
│   └── weather_grid.json
└── docs/                    # Documentation
```

## 📚 Documentation

- **[QUICK_START.md](QUICK_START.md)** - 5-minute getting started guide
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Full GCP deployment guide
- **[OPERATIONS.md](OPERATIONS.md)** - Daily operations & monitoring
- **[scripts/README.md](scripts/README.md)** - Scripts reference
- **[CHANGELOG.md](CHANGELOG.md)** - Version history

## 🔧 Configuration

Edit `configs/project_config.yaml`:

```yaml
scheduler:
  mode: adaptive  # or 'fixed'
  adaptive:
    peak_hours:
      time_ranges:
        - start: "06:00"
          end: "09:00"
        - start: "16:00"
          end: "19:00"
      interval_minutes: 15
    offpeak:
      interval_minutes: 60
    night:
      interval_minutes: 120
```

## 💰 Cost Estimation

**3-Day Production Run:**
- VM (e2-micro): ~$0.50
- Google Directions API: ~$45
- Total: **~$45**

**7-Day Production Run:**
- VM (e2-micro): ~$1.50
- Google Directions API: ~$150
- Total: **~$151**

**Cost Savings:**
- 40% vs constant 15-min intervals
- 95% reduction in weather API calls (grid caching)
- One-time topology fetch (permanent cache)

## 🛠️ Development

### Local Testing

```bash
# Interactive control panel
bash scripts/control_panel.sh

# Single collection
python scripts/collect_once.py

# View collections
python scripts/view_collections.py

# Merge collections
python scripts/merge_collections.py --output data/merged.json
```

### Production Deployment

```bash
# Deploy wizard
bash scripts/deploy_wizard.sh

# Monitor
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="tail -f ~/traffic-forecast/logs/service.log"

# Download data
gcloud compute scp --recurse \
  traffic-forecast-collector:~/traffic-forecast/data/runs \
  ./data-backup --zone=asia-southeast1-a
```

## 📊 Data Output

Each collection run creates:

```
data/runs/run_YYYYMMDD_HHMMSS/
├── nodes.json               # 78 network nodes
├── edges.json               # Network connectivity
├── weather_snapshot.json    # Weather data
├── traffic_edges.json       # 234 traffic segments
└── statistics.json          # Collection stats
```

## 🔍 Monitoring

### Check Service Status

```bash
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="sudo systemctl status traffic-collection.service"
```

### View Logs

```bash
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="tail -50 ~/traffic-forecast/logs/service.log"
```

### Count Collections

```bash
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="ls ~/traffic-forecast/data/runs/ | wc -l"
```

## 🆘 Troubleshooting

### Collection fails
```bash
# Check API key
grep GOOGLE_MAPS_API_KEY .env

# Test Google API
python tools/test_google_limited.py
```

### Service won't start
```bash
# Check logs
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="tail -100 ~/traffic-forecast/logs/service_error.log"
```

### See [OPERATIONS.md](OPERATIONS.md) for complete troubleshooting guide.

## 👥 Author

**Le Quang That (THAT Le Quang)** - SE183256
- Nickname: Xiel
- GitHub: [@thatlq1812](https://github.com/thatlq1812)
- Email: fxlqthat@gmail.com

**Course:** DSP391m - Data Science Project (Fall 2025)

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Google Directions API** for traffic data
- **Open-Meteo** for weather data
- **OpenStreetMap** via Overpass API for road topology
- **FPT University** for academic support

---

**Traffic Forecast v5.1** - Adaptive Scheduling • Cost Optimized • Production Ready

For detailed guides, see [QUICK_START.md](QUICK_START.md) | [DEPLOYMENT.md](DEPLOYMENT.md) | [OPERATIONS.md](OPERATIONS.md)
