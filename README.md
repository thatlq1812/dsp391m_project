# Traffic Forecast System - Ho Chi Minh City# Traffic Forecast v5.1

Real-time traffic data collection system for Ho Chi Minh City using Google Maps API, deployed on Google Cloud Platform.Real-time traffic forecasting for Ho Chi Minh City with adaptive scheduling and cost optimization.

## Project Overview[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

[![Status](https://img.shields.io/badge/Status-Production--Ready-success.svg)]()

This system continuously collects traffic data from 64 major intersections across Ho Chi Minh City, capturing:[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

- Real-time traffic speeds and congestion levels

- Weather conditions> **DSP391m Data Science Project** - Academic project for real-time traffic prediction using Google Directions API, Open-Meteo weather data, and OpenStreetMap topology.

- Road network topology

- Time-series traffic patterns## Features

**Collection Period:** October 30 - November 2, 2025 (3 days)- **Adaptive Scheduling**: Peak/off-peak/night intervals (40% cost savings)

**Data Points:** ~450-600 collection runs, ~150MB total data- **Smart Caching**: Weather grid caching (95% API reduction), permanent topology cache

**Coverage:** 64 intersections, 144 traffic routes, 4km radius from city center- **Wide Coverage**: 4096m radius, 78 filtered nodes, 234 road segments

- **Production Ready**: Automated GCP deployment, systemd service, monitoring

## Quick Start- **Cost Optimized**: ~$45 for 3-day collection, ~$150 for 7 days

### For Team Members (Data Download Only)## Quick Start

1. Install Google Cloud SDK: https://cloud.google.com/sdk/docs/install### 1. Setup (5 minutes)

2. Authenticate: `gcloud auth login`

3. Clone repository: `git clone https://github.com/thatlq1812/dsp391m_project.git````bash

4. Download data: `./scripts/data/download_latest.sh`# Clone repository

git clone <repo-url>

See detailed guide: **docs/TEAM_GUIDE.md**cd project

### For Developers (Deploy & Manage)# Create environment

conda env create -f environment.yml

1. Clone and setup:conda activate dsp

   ````bash

   git clone https://github.com/thatlq1812/dsp391m_project.git# Install package

   cd dsp391m_projectpip install -e .

   conda env create -f environment.yml

   conda activate dsp# Configure API key

   ```echo "GOOGLE_MAPS_API_KEY=your_api_key_here" > .env
   ````

````

2. Configure gcloud:

   ```bash### 2. Test Collection

   gcloud init

   gcloud auth login```bash

   ```# Run single collection

python scripts/collect_once.py

3. Deploy changes:

   ```bash# Use interactive dashboard

   ./scripts/deployment/deploy_git.shbash scripts/control_panel.sh

````

See detailed guide: **docs/DEVELOPER_GUIDE.md**### 3. Deploy to GCP

## Documentation```bash

# Interactive deployment wizard

- **TEAM_GUIDE.md** - How to download and analyze collected databash scripts/deploy_wizard.sh

- **DEVELOPER_GUIDE.md** - How to deploy changes and manage system

- **scripts/deployment/README.md** - Scripts reference# Select option A for auto deployment

````

## System Architecture

## Adaptive Scheduling

**Collection Flow:**

1. Adaptive Scheduler determines collection time based on traffic patternsCost-optimized collection strategy:

2. Overpass API provides road network topology (cached)

3. Open-Meteo API provides weather data| Time Period                       | Interval | Rationale                |

4. Google Maps Directions API provides real-time traffic data| --------------------------------- | -------- | ------------------------ |

5. Data saved to JSON files with timestamps| **Peak** (6-9 AM, 4-7 PM)         | 15 min   | High traffic variability |

| **Off-peak** (9 AM-4 PM, 7-10 PM) | 60 min   | Moderate traffic         |

**Deployment:**| **Night** (10 PM-6 AM)            | 120 min  | Stable traffic           |

- Platform: Google Cloud Platform

- VM: traffic-forecast-collector (e2-small, asia-southeast1-a)**3-Day Collection:**

- Service: systemd auto-restart enabled

- Timezone: UTC+7 (Vietnam)- ~150 total collections

- ~35,100 data points

## Data Collection Schedule- ~$45 total cost (40% savings)



**Peak Hours** (30-minute intervals):## Project Structure

- Morning: 06:30 - 08:00

- Lunch: 10:30 - 11:30```

- Evening: 16:00 - 19:00project/

├── traffic_forecast/        # Main Python package

**Off-Peak Hours** (120-minute intervals):│   ├── collectors/          # Data collectors

- Other weekday hours│   ├── scheduler/           # Adaptive scheduler

│   └── models/              # Data models

## Key Features├── scripts/                 # Utility scripts

│   ├── control_panel.sh     # Local dashboard

**Adaptive Scheduling:**│   ├── deploy_wizard.sh     # GCP deployment

- Automatic adjustment based on time of day│   ├── collect_once.py      # Single collection

- More frequent collection during peak hours│   └── run_adaptive_collection.py  # Continuous collection

- Reduced API costs during off-peak hours├── configs/                 # Configuration

│   └── project_config.yaml

**Weather Integration:**├── data/                    # Data storage

- Grid-based caching (95% API call reduction)│   └── runs/                # Collection outputs

- Current conditions + forecasts (5, 15, 30, 60 minutes)├── cache/                   # Cached data

│   ├── overpass_topology.json

**Quality Control:**│   └── weather_grid.json

- Node selection based on importance scores└── docs/                    # Documentation

- Minimum spacing between nodes```

- Road type filtering (motorway, trunk, primary only)

## 🤖 Machine Learning Models

**Deployment Automation:**This project uses **Deep Learning models** for traffic forecasting:

### LSTM (Long Short-Term Memory)
- Temporal sequence modeling for traffic prediction
- Handles time-series dependencies effectively
- Best for: Short-term traffic forecasting (15-60 minutes ahead)

### ATSCGN (Adaptive Traffic Spatial-Temporal Convolutional Graph Network)
- Combines spatial (graph) and temporal (sequence) modeling
- Captures road network topology and traffic flow patterns
- Best for: Complex multi-node traffic prediction

**Training & Deployment:**
```bash
# Train models
python -m traffic_forecast.ml.dl_trainer --model lstm --epochs 100
python -m traffic_forecast.ml.dl_trainer --model atscgn --epochs 100

# View training history
python -m traffic_forecast.ml.dl_trainer --model lstm --mode history
```

See **[docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md)** for detailed training workflow.

## 📚 Documentation

- Git-based workflow- **[QUICK_START.md](QUICK_START.md)** - 5-minute getting started guide

- Automatic topology regeneration- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Full GCP deployment guide

- Service auto-restart- **[OPERATIONS.md](OPERATIONS.md)** - Daily operations & monitoring

- Health monitoring- **[scripts/README.md](scripts/README.md)** - Scripts reference

- **[CHANGELOG.md](CHANGELOG.md)** - Version history

## Development Commands

## 🔧 Configuration

```bash

# Deploy changesEdit `configs/project_config.yaml`:

./scripts/deployment/deploy_git.sh

```yaml

# Check statusscheduler:

./scripts/deployment/status.sh  mode: adaptive # or 'fixed'

  adaptive:

# Monitor logs    peak_hours:

./scripts/deployment/monitor_logs.sh      time_ranges:

        - start: "06:00"

# View statistics          end: "09:00"

./scripts/monitoring/view_stats.sh        - start: "16:00"

          end: "19:00"

# Download data      interval_minutes: 15

./scripts/data/download_latest.sh    offpeak:

```      interval_minutes: 60

    night:

## Repository      interval_minutes: 120

````

**GitHub:** https://github.com/thatlq1812/dsp391m_project

## Cost Estimation

**GCP Details:**

- Project: sonorous-nomad-476606-g3**3-Day Production Run:**

- VM: traffic-forecast-collector

- Zone: asia-southeast1-a- VM (e2-micro): ~$0.50

- Google Directions API: ~$45

---- Total: **~$45**

For detailed documentation:**7-Day Production Run:**

- Team members: See **docs/TEAM_GUIDE.md**

- Developers: See **docs/DEVELOPER_GUIDE.md**- VM (e2-micro): ~$1.50

- Scripts: See **scripts/deployment/README.md**- Google Directions API: ~$150

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

## Data Output

Each collection run creates:

```
data/runs/run_YYYYMMDD_HHMMSS/
├── nodes.json               # 78 network nodes
├── edges.json               # Network connectivity
├── weather_snapshot.json    # Weather data
├── traffic_edges.json       # 234 traffic segments
└── statistics.json          # Collection stats
```

## Monitoring

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

## Troubleshooting

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

## Author

**Le Quang That (THAT Le Quang)** - SE183256

- Nickname: Xiel
- GitHub: [@thatlq1812](https://github.com/thatlq1812)
- Email: fxlqthat@gmail.com

**Course:** DSP391m - Data Science Project (Fall 2025)

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- **Google Directions API** for traffic data
- **Open-Meteo** for weather data
- **OpenStreetMap** via Overpass API for road topology
- **FPT University** for academic support

---

**Traffic Forecast v5.1** - Adaptive Scheduling • Cost Optimized • Production Ready
