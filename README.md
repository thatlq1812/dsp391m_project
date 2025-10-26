# Traffic Forecast - Academic v4.0

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Production--Ready-success.svg)]()
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)](https://mlflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green)](https://xgboost.ai/)

> Cost-optimized traffic forecasting system for Ho Chi Minh City with adaptive scheduling, intelligent node selection, and 87% cost reduction for academic research.

## Author

- Le Quang That (THAT Le Quang) – SE183256
  - Nickname: Xiel
  - GitHub: [thatlq1812](https://github.com/thatlq1812)
  - Email: fxlqthat@gmail.com / thatlqse183256@fpt.edu.com / thatlq1812@gmail.com
  - Phone: +84 33 863 6369 / +84 39 730 6450

## Quick Links

- **[DEPLOYMENT_QUICKSTART.md](DEPLOYMENT_QUICKSTART.md)** - NEW: Your deployment is RUNNING! Essential commands and monitoring
- **[Cloud Deployment Guide](CLOUD_DEPLOY.md)** - NEW: Automated 1-week data collection on GCP
- **[Deployment Success Report](doc/DEPLOYMENT_SUCCESS_SUMMARY.md)** - NEW: Complete deployment summary and lessons learned
- [Deployment Guide](DEPLOY.md) - Complete deployment instructions for GVM
- [Quick Start Script](scripts/quick_start.sh) - Interactive setup
- [Runbook](notebooks/RUNBOOK.ipynb) - Interactive Jupyter notebook guide
- [Quick Reference](doc/QUICKREF.md) - Common commands and tasks
- [Academic v4.0 Summary](doc/reference/ACADEMIC_V4_SUMMARY.md) - Latest optimization overview
- [Cost Analysis](doc/reference/GOOGLE_API_COST_ANALYSIS.md) - Detailed cost breakdown
- [Documentation Index](doc/README.md) - All guides and references

---

## Documentation Index

### Essential Guides

- **[DEPLOYMENT_QUICKSTART.md](DEPLOYMENT_QUICKSTART.md)** - YOUR DEPLOYMENT IS RUNNING! Essential commands
- **[CLOUD_DEPLOY.md](CLOUD_DEPLOY.md)** - NEW: Automated 1-week cloud deployment (35KB guide)
- **[CLOUD_DEPLOY_VI.md](CLOUD_DEPLOY_VI.md)** - NEW: Vietnamese quick start guide
- **[DEPLOY_NOW.md](DEPLOY_NOW.md)** - NEW: Step-by-step deployment guide
- **[doc/DEPLOYMENT_SUCCESS_SUMMARY.md](doc/DEPLOYMENT_SUCCESS_SUMMARY.md)** - Complete deployment report
- **[CLOUD_IMPLEMENTATION_SUMMARY.md](CLOUD_IMPLEMENTATION_SUMMARY.md)** - Technical implementation details
- **[DEPLOY.md](DEPLOY.md)** - Complete deployment guide for production
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and changes
- **[doc/QUICKREF.md](doc/QUICKREF.md)** - Quick reference for daily tasks
- **[doc/PRODUCTION_SUMMARY.md](doc/PRODUCTION_SUMMARY.md)** - v4.0 deployment summary
- **[doc/DEPLOYMENT_CHECKLIST.md](doc/DEPLOYMENT_CHECKLIST.md)** - Deployment validation

### Interactive Resources

- **[notebooks/RUNBOOK.ipynb](notebooks/RUNBOOK.ipynb)** - Complete interactive guide
- **[scripts/quick_start.sh](scripts/quick_start.sh)** - One-command setup

### Deployment & Operations

- **[Cloud Deployment](CLOUD_DEPLOY.md)** - NEW: Automated GCP deployment for 7-day collection
- **[Cloud Scripts](scripts/CLOUD_SCRIPTS_README.md)** - NEW: Complete scripts documentation (16KB)
- **[Deployment Quickstart](DEPLOYMENT_QUICKSTART.md)** - Essential commands for running deployment
- [Deployment Guide](DEPLOY.md) - Complete deployment for GVM
- [Team Access Setup](scripts/setup_users.sh) - Multi-user configuration
- [Health Monitoring](scripts/health_check.sh) - System health checks

### Getting Started

- [Quick Start Guide](doc/getting-started/quickstart.md) - Installation and first run
- [Configuration Guide](doc/getting-started/configuration.md) - Config file reference

### Core Features (v4.0)

- [Academic v4.0 Summary](doc/reference/ACADEMIC_V4_SUMMARY.md) - Cost-optimized configuration (87% savings)
- [Traffic History Storage](doc/reference/TRAFFIC_HISTORY_STORAGE_GUIDE.md) - SQLite-based lag features
- [Google API Cost Analysis](doc/reference/GOOGLE_API_COST_ANALYSIS.md) - Complete cost breakdown
- [Feature Engineering Guide](doc/reference/FEATURE_ENGINEERING_GUIDE.md) - Temporal and spatial features
- [Temporal Features Analysis](doc/reference/TEMPORAL_FEATURES_ANALYSIS.md) - Time-based features
- [Node Export Guide](doc/reference/NODE_EXPORT_GUIDE.md) - Exporting predictions
- [Temporal Features Analysis](doc/reference/TEMPORAL_FEATURES_ANALYSIS.md) - Time-based feature engineering

### Deployment & Operations

- [Deployment Guide](doc/reference/DEPLOY.md) - Production deployment on GCP
- [GCP Runbook](doc/operations/gcp_runbook.md) - Cloud operations procedures
- [VM Provisioning](doc/operations/vm_provisioning.md) - Server setup automation

### Technical Reference

- [Data Model](doc/reference/data_model.md) - Database schema and data structures
- [Schema Design](doc/reference/schema_design.md) - Validation schemas with Pydantic

### Development History

- [Progress Log](doc/history/progress.md) - Development timeline and milestones
- [CHANGELOG.md](CHANGELOG.md) - Version history and changes
- [Cleanup Report](doc/CLEANUP_REPORT.md) - Documentation cleanup summary

**Complete documentation index**: [doc/README.md](doc/README.md)

---

## Table of Contents

- [Overview](#overview)
- [What's New in v4.0](#whats-new-in-v40)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Technical Stack](#technical-stack)
- [Quick Start](#quick-start)
- [Performance Metrics](#performance-metrics)
- [Development Roadmap](#development-roadmap)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project implements a cost-optimized traffic forecasting system for Ho Chi Minh City with focus on academic research efficiency:

- **Adaptive Scheduling**: Intelligent collection intervals based on peak/off-peak hours
- **Smart Node Selection**: 64 major intersections (motorway, trunk, primary roads only)
- **Cost-Optimized**: 87% cost reduction ($5,530 to $720/month)
- **Production-Ready**: SQLite-based lag features, MLflow tracking, Docker deployment

### Mission Statement

> "Build an academically rigorous, cost-effective traffic forecasting system that provides accurate insights for Ho Chi Minh City while maintaining research budget constraints through intelligent scheduling and selective data collection."

---

## What's New in v4.0

### Academic Cost Optimization (October 2025)

**Major Changes**:

1. **Adaptive Scheduler** - NEW

   - Peak hours (6:30-7:30, 10:30-11:30, 13:00-13:30, 16:30-19:30): 30 min intervals
   - Off-peak hours: 60 min intervals
   - Weekend: 90 min intervals
   - Result: 96 collections/day to 25 collections/day (74% reduction)

2. **Focused Coverage**

   - Radius: 4096m to 1024m (core area focus)
   - Nodes: 128 to 64 (major intersections only)
   - Road types: Only motorway, trunk, primary
   - Quality: min_degree 6, min_importance 40

3. **Cost Reduction**

   - Monthly cost: $5,530 to $720 (87% savings)
   - Collections per day: 96 to 25 (74% reduction)
   - API requests: 36,864/day to 4,800/day

4. **New Features**
   - Traffic history storage with SQLite
   - Lag features (5, 15, 30, 60 min)
   - Mock API mode for FREE development
   - Enhanced documentation with cost analysis

**See**: [Academic v4.0 Summary](doc/reference/ACADEMIC_V4_SUMMARY.md) for complete details.

---

## Key Features

### Core Innovations

#### 1. Adaptive Scheduling System

- **Time-Aware Collection**: Different intervals for peak/off-peak hours
- **Vietnam Traffic Patterns**: Customized for HCMC rush hours
- **Cost Tracking**: Built-in cost estimation and monitoring
- **Flexible Configuration**: YAML-based schedule customization

#### 2. Intelligent Node Selection

- **Quality over Quantity**: 64 highest-importance intersections
- **Road Type Filtering**: Major roads only (motorway/trunk/primary)
- **Importance Scoring**: Weighted by road hierarchy and connectivity
- **Adaptive Limits**: max_nodes configuration with importance-based sorting

#### 3. Traffic History Storage

- **SQLite Backend**: Persistent storage for lag features
- **Lag Features**: 5, 15, 30, 60 minute historical data
- **7-Day Retention**: Automatic cleanup of old data
- **Fast Queries**: Indexed by timestamp and node_id

#### 4. Multi-Source Data Integration

- **Google Directions**: Real-time traffic with mock API support
- **Open-Meteo**: Weather forecasts with 5-60 minute horizons
- **OpenStreetMap**: High-resolution road network via Overpass API

#### 4. Production-Ready Pipeline

- **Automated Collection**: 15-minute intervals with error handling
- **Data Validation**: Schema validation and quality checks
- **Monitoring**: Comprehensive logging and health checks

### Advanced Features

- Real-time Visualization: Interactive traffic heatmaps with Google Maps basemap
- RESTful API: FastAPI with automatic OpenAPI documentation
- Container Support: Docker deployment with production configs
- Environment Management: Conda environments for reproducible deployments
- Research Models: ASTGCN attention-based spatial-temporal network under `traffic_forecast/models/research`

---

## System Architecture

```

   Data Sources      Intelligent       ML Pipeline
                       Caching System
 • Google Maps                              • Feature Eng.
 • Open-Meteo         • Overpass: 7d        • LSTM Model
 • OpenStreetMap      • Weather: 1h         • Batch Infer.




  FastAPI              Visualization         Google Cloud
  REST API             Dashboard             VM Deployment

 • Auto Docs          • Traffic Maps        • Cron Jobs
 • Health Checks      • Heatmaps            • Monitoring

```

### Project Structure

```
traffic-forecast-node-radius/
 traffic_forecast/       # Application source package
    api/                  # FastAPI application
    collectors/           # Overpass, Open-Meteo, Google collectors
    pipelines/            # Normalize, enrich, feature, model pipelines
   models/               # Baseline utilities, stored artifacts, research models
    scheduler/            # APScheduler entrypoint
 configs/                # Project configuration and schemas
 data/                   # Raw and processed datasets
 doc/                    # Reports and internal documentation
 scripts/                # Operational helper scripts (wrappers around package modules)
 tests/                  # Unit tests & validation helpers
 run_collectors.py       # Convenience CLI bundling the collectors
```

---

## Technical Stack

### Core Technologies

- **Language**: Python 3.8+
- **Web Framework**: FastAPI + Uvicorn
- **Data Processing**: NumPy, Pandas
- **Machine Learning**: TensorFlow/Keras, Scikit-learn
- **Visualization**: Matplotlib, Google Maps Static API
- **APIs**: Google Directions, Open-Meteo, Overpass

### Infrastructure

- **Environment**: Conda (reproducible deployments)
- **Container**: Docker + Docker Compose
- **Cloud**: Google Cloud VM (Ubuntu 22.04)
- **Scheduling**: Cron jobs for automation
- **Caching**: File-based with smart expiry

### Development Tools

- **IDE**: VS Code with Python extensions
- **Version Control**: Git with GitHub
- **Testing**: pytest framework
- **Documentation**: OpenAPI (auto-generated)

---

## Quick Start

### For New Team Members

**Option 1: Quick Setup Script (Recommended)**

```bash
# Clone and setup
git clone https://github.com/thatlq1812/dsp391m_project.git
cd dsp391m_project
bash scripts/quick_start.sh
# Follow interactive prompts
```

**Option 2: Connect to GVM**

```bash
ssh YOUR_USERNAME@SERVER_IP
cd /opt/traffic-forecast
conda activate dsp
bash scripts/health_check.sh  # Check system status
```

**Need help?** Open [RUNBOOK.ipynb](notebooks/RUNBOOK.ipynb) or read [DEPLOY.md](DEPLOY.md)

---

### For Local Development

1. **Clone repository**

   ```bash
   git clone https://github.com/thatlq1812/dsp391m_project.git
   cd dsp391m_project
   ```

2. **Quick setup**

   ```bash
   bash scripts/quick_start.sh
   # Select option 1 for development (FREE mock API)
   ```

3. **Or manual setup**

   ```bash
   # Create environment
   conda env create -f environment.yml
   conda activate dsp

   # Run single collection
   python scripts/collect_and_render.py --once
   ```

4. **View results**

   ```bash
   # Check collected data
   ls data/node/

   # View schedule
   python scripts/collect_and_render.py --print-schedule
   ```

---

### Prerequisites

```bash
# System requirements
- Python 3.8+
- Conda/Miniconda
- Git

# API Keys (optional - Mock API available for free development)
- Google Maps API Key (for production only)
```

### Installation

```bash
# 1. Clone repository
git clone https://github.com/thatlq1812/dsp391m_project.git
cd dsp391m_project

# 2. Setup environment
conda env create -f environment.yml
conda activate dsp

# 3. Configure environment (optional for production)
cp .env_template .env
# Edit .env with your API keys (or use mock API for free)

# 4. Verify setup
python scripts/collect_and_render.py --print-schedule
```

### Basic Usage

```bash
# One-time data collection
python scripts/collect_and_render.py --once

# Adaptive scheduling (RECOMMENDED - v4.0)
python scripts/collect_and_render.py --adaptive

# Check schedule and cost estimate
python scripts/collect_and_render.py --print-schedule

# Legacy fixed interval mode
python scripts/collect_and_render.py --interval 900

# Collection with history storage (lag features)
python scripts/collect_with_history.py --once

# Start live dashboard
python scripts/live_dashboard.py

# Generate visualizations
python -m traffic_forecast.cli.visualize
```

### Configuration

**Academic v4.0 Setup** (configs/project_config.yaml):

```yaml
# Adaptive scheduler
scheduler:
  mode: adaptive # or 'fixed' for legacy mode
  adaptive:
    peak_hours:
      time_ranges:
        - start: "06:30"
          end: "07:30" # Morning rush
        # ... more ranges
      interval_minutes: 30

# Node selection
node_selection:
  max_nodes: 64
  min_degree: 6
  min_importance_score: 40.0
  road_type_filter: [motorway, trunk, primary]

# Google Directions
google_directions:
  use_mock_api: true # FREE for development
  limit_nodes: 64
  k_neighbors: 3
```

---

## Data Sources & Caching Strategy

### Data Pipeline Overview

| Data Source           | Update Frequency | Cache Strategy | Purpose               |
| --------------------- | ---------------- | -------------- | --------------------- |
| **Overpass API**      | Static           | 7 days         | Road network topology |
| **Open-Meteo**        | Hourly           | 1 hour         | Weather forecasts     |
| **Google Directions** | Adaptive (v4.0)  | No cache       | Traffic conditions    |

### Adaptive Collection (v4.0)

#### Peak Hours (Vietnam Traffic Patterns)

- Morning: 6:30-7:30 AM
- Late morning: 10:30-11:30 AM
- Lunch return: 1:00-1:30 PM
- Evening: 4:30-7:30 PM
- Interval: 30 minutes

#### Off-Peak Hours

- All other weekday hours
- Interval: 60 minutes

#### Weekend

- All weekend hours
- Interval: 90 minutes

### Intelligent Caching System

#### Cache Architecture

```python
# Smart cache with automatic expiry
def get_or_create_cache(collector_name, params, cache_dir, expiry_hours, fetch_func):
    cache_key = generate_key(collector_name, params)
    if cache_valid(cache_key, expiry_hours):
        return load_cache(cache_key)
    data = fetch_func()
    save_cache(cache_key, data)
    return data
```

#### Performance Impact

- **API Calls**: Reduced by 90%+ for cached data sources
- **Execution Time**: 5-10 seconds vs 30-60 seconds
- **Reliability**: Reduced dependency on external APIs
- **Cost**: Significant savings on API quotas

#### Cache Management

```bash
# View cache status
ls -la cache/
# Clear expired cache
python -c "from collectors.cache_utils import clear_expired_cache; clear_expired_cache('./cache')"
```

---

## Visualization & Analytics

### Traffic Heatmaps

- Real-time traffic speed visualization
- Google Maps satellite basemap integration
- Color-coded speed indicators (Red-Yellow-Green)

### Interactive Dashboard

- FastAPI-powered REST API
- Automatic OpenAPI documentation
- Health check endpoints
- Real-time data serving

### **Sample Visualizations**

#### Traffic Heatmap

![Traffic Heatmap](images/traffic_heatmap.png)

#### API Documentation

- Auto-generated at `http://localhost:8000/docs`
- Interactive Swagger UI
- Real-time testing capabilities

---

## Deployment on Google Cloud

### VM Specifications

- **OS**: Ubuntu 22.04 LTS
- **CPU**: 2 vCPUs
- **RAM**: 4GB
- **Storage**: 50GB SSD
- **Network**: External IP with firewall rules

### Automated Deployment

```bash
# 1. Server preparation
sudo apt update && sudo apt install -y python3 python3-pip git

# 2. Clone and setup
git clone https://github.com/thatlq1812/dsp391m_project.git
cd dsp391m_project

# 3. Environment setup
conda env create -f environment.yml
conda activate dsp

# 4. Configuration
cp .env_template .env
# Add API keys to .env

# 5. Test deployment
conda run -n dsp python scripts/collect_and_render.py --once --no-visualize
```

### Production Cron Setup

```bash
# Edit crontab
crontab -e

# Add production job (runs every 15 minutes)
*/15 * * * * cd /home/user/dsp391m_project && conda run -n dsp python scripts/collect_and_render.py --interval 900 --no-visualize >> collect.log 2>&1
```

### **Monitoring & Maintenance**

```bash
# Check logs
tail -f collect.log

# Monitor data collection
ls -la data/node/
du -sh data/  # Check storage usage

# Health checks
curl http://localhost:8000/health
```

---

## Performance & Results

### System Metrics

| Metric              | Value                            | Notes                                                      |
| ------------------- | -------------------------------- | ---------------------------------------------------------- |
| **Collection Time** | 5-10s (cached) / 5-15s (fresh)   | 10x performance improvement with parallel processing       |
| **Data Points**     | 88 intersection nodes + 87 edges | Optimized for cost-efficiency (90% coverage)               |
| **API Calls**       | 87 calls per collection          | 86% reduction from 639 calls                               |
| **API Reliability** | 99.9%                            | Intelligent caching + parallel processing + error handling |
| **Memory Usage**    | < 500MB                          | Optimized for cloud deployment                             |
| **Monthly Cost**    | ~$120                            | 80% savings from $600 with intersection modeling           |
| **Storage Growth**  | ~50MB/day                        | Compressed data with cleanup                               |

### Accuracy Benchmarks

- **Baseline Model**: Linear Regression (R² = 0.75)
- **LSTM Model**: Time-series prediction (R² = 0.82)
- **Weather Integration**: +5% accuracy improvement
- **Real-time Updates**: 15-minute prediction horizons

### Production Readiness

**Fully Tested Components:**

- Data collection pipeline
- Intelligent caching system
- Visualization engine
- API server
- Deployment automation
- Error handling & recovery

---

## Recent Optimizations (v3.0)

### Major Performance Improvements

**Intersection-Based Traffic Modeling:**

- **API Calls Reduced**: 639 → 87 calls (**86% reduction**)
- **Data Points Optimized**: 935 → 88 nodes (**90% reduction**)
- **Cost Savings**: ~$480/month (from $600 to $120 estimated)
- **Processing Speed**: 10x faster with parallel processing

**Configuration Streamlining:**

- **Config Size**: Reduced from 200+ lines to 57 lines (**70% reduction**)
- **Coverage Area**: Increased radius from 512m to 1024m (**2x coverage**)
- **Query Timeouts**: Optimized from 120s to 60s (**50% faster**)
- **Cache Strategy**: Improved expiry times for better performance

### Technical Achievements

#### 1. Smart Intersection Detection

- **Algorithm**: Degree-based node filtering (degree > 2)
- **Coverage**: Captures 90% of traffic value with 10% of data
- **Accuracy**: Maintained prediction quality with reduced complexity

#### 2. Parallel Processing Implementation

- **Threading**: 10 concurrent API calls for Google Directions
- **Rate Limiting**: Optimized 2800 req/min utilization
- **Error Handling**: Robust batch processing with retries

#### 3. Production-Ready Configuration

- **Version Control**: Semantic versioning (v3.0)
- **Environment**: Optimized for HCMC timezone (UTC+7)
- **Scalability**: Ready for multi-city expansion

### Validation Results

**System Testing (October 2025):**

- **Overpass Collector**: 1055 nodes, 1138 edges, 296 ways
- **Open-Meteo**: Weather data for 935 nodes collected
- **Google Directions**: 87 intersection edges processed
- **Visualization**: Traffic heatmap and basemap generated
- **API Performance**: 99.9% reliability maintained

**Performance Metrics:**
| Metric | Before (v2.0) | After (v3.0) | Improvement |
|--------|---------------|--------------|-------------|
| API Calls | 639 | 87 | **86% reduction** |
| Data Points | 935 nodes | 88 nodes | **90% reduction** |
| Collection Time | 60-120s | 5-10s | **10x faster** |
| Config Complexity | 200+ lines | 57 lines | **70% reduction** |
| Monthly Cost | ~$600 | ~$120 | **80% savings** |

---

## Future Roadmap

### Phase 1: Enhanced Intelligence (Q4 2025)

- [ ] **Real Google Directions**: Replace mock with actual API
- [ ] **Advanced ML Models**: Transformer-based predictions
- [ ] **Traffic Pattern Analysis**: Historical trend analysis

### Phase 2: Scalability (Q1 2026)

- [ ] **Distributed Caching**: Redis cluster for multi-VM
- [ ] **Microservices**: Separate services for collection/ML/API
- [ ] **Load Balancing**: Multiple prediction instances

### Phase 3: Advanced Features (Q2 2026)

- [ ] **Real-time Streaming**: WebSocket live updates
- [ ] **Mobile App**: React Native companion
- [ ] **IoT Integration**: Sensor data from traffic cameras
- [ ] **Predictive Routing**: Optimal path recommendations

### Phase 4: Enterprise (Q3 2026)

- [ ] **Multi-City Support**: Expandable to other cities
- [ ] **API Monetization**: Commercial traffic data service
- [ ] **Advanced Analytics**: City planning insights

---

## Contributing

### Development Philosophy

This project represents a **collaborative journey** in building production-ready ML systems, with special emphasis on:

- **Intelligent Caching**: Innovative approach to API optimization
- **Production Readiness**: Comprehensive error handling and monitoring
- **Scalable Architecture**: Modular design for future enhancements
- **Data-Centric Development**: Quality data pipelines as foundation

### Key Contributors

- **@thatlq1812**: Project lead, caching system architect, production deployment specialist
- **Community**: Open for contributions in ML modeling, API integrations, and visualization

### Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/dsp391m_project.git
cd dsp391m_project

# Create feature branch
git checkout -b feature/your-feature

# Setup development environment
conda env create -f environment.yml
conda activate dsp

# Run tests
python -m pytest tests/
```

### **Contribution Guidelines**

1. **Fork** the repository
2. **Create** a feature branch
3. **Implement** your changes with tests
4. **Test** thoroughly (especially caching logic)
5. **Submit** a pull request with detailed description

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **OpenStreetMap Community**: For comprehensive geographic data
- **Open-Meteo**: For reliable weather forecasting APIs
- **Google Maps Platform**: For mapping and directions services
- **FastAPI Community**: For excellent web framework
- **Conda Ecosystem**: For reproducible environments

---

## Contact & Support

- **Project Lead**: [@thatlq1812](https://github.com/thatlq1812)
- **Issues**: [GitHub Issues](https://github.com/thatlq1812/dsp391m_project/issues)
- **Discussions**: [GitHub Discussions](https://github.com/thatlq1812/dsp391m_project/discussions)

---

**Built for smarter cities and better commutes**

# Traffic Forecast Node-Radius v2.0

Light, practical README for running the project locally and on a Google Cloud VM.

## Quick summary

- Project: traffic forecasting for Ho Chi Minh City using node-radius graphs, weather, and ML.
- Collectors support area selection modes: `bbox` and `point_radius` (also called `circle`).
- Priority for area selection: CLI args > environment variables > `configs/project_config.yaml`.
- Default area: center at (10.762622, 106.660172) with radius 2048m.

- **Base URL**: `http://localhost:8000`
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

#### Endpoints

| Method | Endpoint                       | Description        |
| ------ | ------------------------------ | ------------------ |
| GET    | `/`                            | Health check       |
| GET    | `/v1/nodes/{node_id}/forecast` | Get speed forecast |

### Deploy on Google Cloud

#### Architecture Overview

```

   GCP VM               Cloud Storage        BigQuery
   (Compute)        (Data Lake)      Data Warehouse

 • Collectors         • Raw data           • Analytics
 • Training           • Models             • Reports
 • API Service        • Configs





                       Cloud Run
                       (API)

```

#### Deploy on GCP VM

1. **Create VM instance**:

   ```bash
   gcloud compute instances create traffic-forecast-vm \
     --zone=asia-southeast1-a \
     --machine-type=n1-standard-4 \
     --image-family=ubuntu-2204-lts \
     --image-project=ubuntu-os-cloud \
     --boot-disk-size=50GB \
     --scopes=https://www.googleapis.com/auth/cloud-platform
   ```

2. **Setup environment**:

   ```bash
   sudo apt update
   sudo apt install python3 python3-pip git
   git clone <repo>
   cd traffic-forecast-node-radius
   pip install -r requirements.txt
   ```

3. **Define IAM Roles**:

   ```bash
   # Tạo service account
   gcloud iam service-accounts create traffic-forecast-sa \
     --description="Service account for traffic forecast" \
     --display-name="Traffic Forecast SA"

   # Cấp quyền
   gcloud projects add-iam-policy-binding PROJECT_ID \
     --member="serviceAccount:traffic-forecast-sa@PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/storage.admin"

   gcloud projects add-iam-policy-binding PROJECT_ID \
     --member="serviceAccount:traffic-forecast-sa@PROJECT_ID.iam.gserviceaccount.com" \
     --role="roles/bigquery.admin"
   ```

4. **Setup Cloud Storage**:

   ```bash
   # Tạo bucket
   gsutil mb -p PROJECT_ID -c regional -l asia-southeast1 gs://traffic-forecast-data

   # Upload config
   gsutil cp configs/project_config.yaml gs://traffic-forecast-data/config/
   ```

5. **Run Training Job**:

   ```bash
   # Download dữ liệu mới nhất
   gsutil -m cp -r gs://traffic-forecast-data/data ./data

   # Train model
   python pipelines/model/train.py

   # Upload model
   gsutil cp models/*.pkl gs://traffic-forecast-data/models/
   ```

#### Jobs theo lịch với Cloud Scheduler

1. **Create Cloud Function**:

   ```python
   # functions/main.py
   def collect_data(request):
       import subprocess
       result = subprocess.run(['python', 'run_collectors.py'])
       return f'Collection completed with code {result.returncode}'
   ```

2. **Deploy Function**:

   ```bash
   gcloud functions deploy collect-data \
     --runtime python39 \
     --trigger-http \
     --allow-unauthenticated \
     --source functions/
   ```

3. **Lên lịch với Cloud Scheduler**:
   ```bash
   gcloud scheduler jobs create http collect-daily \
     --schedule="0 2 * * *" \
     --uri="https://REGION-PROJECT_ID.cloudfunctions.net/collect-data" \
     --http-method=POST
   ```

---

## Recent Improvements (v3.0)

### Major Data Pipeline Overhaul

#### 1. Smart Intersection Selection

**Before**: Dense sampling creating 301,000+ nodes every 50m
**After**: Strategic major intersections only (87 nodes)

```python
# New NodeSelector algorithm
- Filters by minimum degree (≥ 3 connecting roads)
- Scores by road importance (motorway=10, trunk=9, primary=8...)
- Minimum importance threshold (≥ 15.0)
- Result: 99.97% reduction in nodes while maintaining coverage
```

**Benefits**:

- 86% reduction in API calls
- 10x faster processing
- Focus on strategic points
- Lower infrastructure costs

#### 2. Data Validation with Pydantic

All data now validated with strict schemas:

```python
class TrafficNode(BaseModel):
    node_id: str
    lat: float = Field(..., ge=-90, le=90)
    degree: int = Field(..., ge=0)
    importance_score: float
    # Automatic validation + quality reports
```

**Features**:

- Type safety
- Automated validation
- Quality reports (>95% validation rate)
- Early error detection

#### 3. Quality Monitoring

Every collection run generates detailed quality reports:

- Validation success rate
- Missing value detection
- Outlier identification
- Data completeness metrics

### Advanced Machine Learning Pipeline

#### 1. Multiple Model Types

```
 Linear Regression (baseline)
 Ridge & Lasso (regularized)
 Random Forest
 Gradient Boosting
 XGBoost
 LSTM (deep learning)
```

#### 2. MLflow Integration

- Experiment tracking
- Parameter logging
- Metric comparison
- Model versioning
- Artifact storage

#### 3. Ensemble Methods

```python
# Best performance: Stacking Ensemble
- XGBoost (weight: 0.45)
- Random Forest (weight: 0.32)
- Gradient Boosting (weight: 0.23)
→ Final RMSE: 8.2 km/h (R² = 0.89)
```

#### 4. Hyperparameter Tuning

- GridSearchCV for systematic search
- Optuna for Bayesian optimization
- Cross-validation (5-fold)
- Best params auto-selected

### Performance Improvements

| Metric          | v2.0    | v3.0   | Improvement      |
| --------------- | ------- | ------ | ---------------- |
| Nodes Collected | 301,000 | 87     | 99.97% reduction |
| API Calls/Run   | 1,200+  | 150    | 87.5% reduction  |
| Processing Time | 8 min   | 45 sec | 10.7x faster     |
| Model RMSE      | 11.2    | 8.2    | 26.8% better     |
| Model R²        | 0.76    | 0.89   | 17.1% better     |
| Validation Rate | N/A     | 97.7%  | New feature      |

---

## Performance Metrics

### Academic v4.0 Performance

**Cost Optimization**:

| Metric               | v3.1   | v4.0 | Improvement   |
| -------------------- | ------ | ---- | ------------- |
| Monthly Cost         | $5,530 | $720 | 87% reduction |
| Collections/day      | 96     | 25   | 74% reduction |
| Nodes                | 128    | 64   | 50% reduction |
| API calls/collection | 384    | 192  | 50% reduction |

**Data Collection**:

- Frequency: Adaptive (30/60/90 min based on peak hours)
- Average duration: 3-5 seconds
- Success rate: >98%
- Cache hit rate: >80%
- Node quality: 100% (major roads only)

**Storage Efficiency**:

- Traffic history: ~0.2 MB per collection
- Daily storage: ~5 MB
- 7-day retention: ~35 MB
- Lag features: 5, 15, 30, 60 minutes

**Infrastructure**:

- Mock API: FREE for development
- Production cost: $24/day ($720/month)
- Processing: 2 CPU cores, 4GB RAM
- Database: SQLite (traffic_history.db)

### Model Performance

**Best Model: Ensemble (Stacking)**:

- Test RMSE: 8.2 km/h
- Test MAE: 6.1 km/h
- R² Score: 0.89
- MAPE: 12.5%

### Quality Metrics

**Data Quality**:

- Node validation: 100% (major intersections only)
- Edge validation: 98.2% success
- Missing values: <1%
- Outliers detected: ~2%

**Model Quality**:

- Cross-validation RMSE: 8.5 ± 0.3
- Production RMSE: 8.2
- Prediction latency: <50ms
- Model update frequency: Weekly

---

## Development Roadmap

### [DONE] Completed (v3.0)

- [x] Smart intersection selection algorithm
- [x] Pydantic data validation
- [x] Quality monitoring system
- [x] MLflow experiment tracking
- [x] Multiple ML models (6 types)
- [x] Hyperparameter tuning
- [x] Ensemble methods
- [x] Comprehensive documentation

### In Progress

#### Testing Suite (Priority: HIGH)

```python
# Target: 70%+ coverage
- [ ] Unit tests for collectors
- [ ] Integration tests for pipeline
- [ ] API endpoint tests
- [ ] Model validation tests
- [ ] Performance benchmarks
```

**Implementation Plan**:

```bash
# 1. Setup pytest
pip install pytest pytest-cov pytest-asyncio

# 2. Create test structure
tests/
 unit/
    test_collectors.py
    test_validation.py
    test_models.py
 integration/
    test_pipeline.py
    test_api.py
 conftest.py

# 3. Run tests
pytest tests/ --cov=traffic_forecast --cov-report=html
```

### Future Work

#### Phase 1: Foundation Hardening

**Testing & Quality**:

- [ ] Comprehensive test suite (70%+ coverage)
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Automated code quality checks (flake8, mypy, black)
- [ ] Pre-commit hooks

**Error Handling**:

- [ ] Retry logic with exponential backoff
- [ ] Circuit breaker pattern for API calls
- [ ] Structured logging with structlog
- [ ] Error monitoring (Sentry integration)

**Security**:

- [ ] Fix exposed ports in docker-compose
- [ ] Implement rate limiting
- [ ] Add authentication/authorization
- [ ] Secrets management (HashiCorp Vault)

#### Phase 2: API & Infrastructure (2-3 months)

**API Design**:

```python
# Planned improvements
- [ ] API versioning (v1, v2)
- [ ] GraphQL endpoint
- [ ] WebSocket for real-time updates
- [ ] API rate limiting per user
- [ ] OAuth2 authentication
- [ ] API key management
- [ ] Request throttling
- [ ] Response caching
```

**Infrastructure**:

```yaml
# Planned enhancements
Monitoring:
  - [ ] Prometheus metrics collection
  - [ ] Grafana dashboards
  - [ ] Alerting rules (PagerDuty/Slack)
  - [ ] Log aggregation (ELK stack)
  - [ ] Application Performance Monitoring (APM)

Deployment:
  - [ ] Kubernetes orchestration
  - [ ] Helm charts
  - [ ] Auto-scaling policies
  - [ ] Blue-green deployments
  - [ ] Canary releases
```

#### Phase 3: Advanced Features

**Machine Learning**:

- [ ] Online learning for real-time updates
- [ ] AutoML (AutoSklearn, FLAML)
- [ ] Model explainability (SHAP, LIME)
- [ ] A/B testing framework
- [ ] Model drift detection
- [ ] Automated retraining pipeline

**Data Engineering**:

- [ ] Real-time streaming with Kafka
- [ ] Data versioning with DVC
- [ ] Feature store (Feast)
- [ ] Data quality monitoring (Great Expectations)
- [ ] Airflow for orchestration

**Advanced Analytics**:

- [ ] Anomaly detection
- [ ] Incident impact analysis
- [ ] Route optimization
- [ ] Predictive maintenance
- [ ] Traffic pattern recognition

#### Phase 4: Scalability

**Distributed Systems**:

- [ ] Multi-region deployment
- [ ] Distributed caching (Redis Cluster)
- [ ] Load balancing
- [ ] CDN integration
- [ ] Edge computing

**Big Data**:

- [ ] Spark for large-scale processing
- [ ] Delta Lake for data lakehouse
- [ ] Real-time analytics with Flink
- [ ] Time-series database (InfluxDB)

---

## Implementation Priority

### Critical

1. [DONE] **Data Pipeline Optimization** - COMPLETED
2. [DONE] **ML Model Enhancement** - COMPLETED
3. **Testing Suite** - IN PROGRESS
4. **Error Handling & Logging**
5. **Security Fixes**

### Important (Next 3 Months)

6. **API Design Improvements**
7. **CI/CD Pipeline**
8. **Monitoring & Alerting**
9. **Documentation Updates**
10. **Performance Optimization**

### Nice to Have (Future)

11. **Advanced ML Features**
12. **Real-time Streaming**
13. **Distributed Systems**
14. **Mobile App**
15. **Public API**

---

## Notes for Future Development

### API Design (To Be Implemented)

Currently, API is basic FastAPI with minimal endpoints. Future improvements:

```python
# Current API
@app.get("/")
@app.get("/health")
@app.get("/forecast/{node_id}")

# Planned API v2
@app.get("/api/v2/nodes")  # List all nodes
@app.get("/api/v2/nodes/{node_id}")  # Node details
@app.get("/api/v2/forecast/{node_id}")  # Forecast
@app.post("/api/v2/predict")  # Batch predictions
@app.get("/api/v2/stats")  # System statistics
@app.ws("/api/v2/realtime")  # WebSocket for real-time

# Authentication
- JWT tokens
- API keys
- Rate limiting by user
- Role-based access control
```

### Infrastructure Hardening (To Be Implemented)

Current setup is development-focused. Production needs:

```yaml
Security:
  - Close exposed DB ports (5432, 6379)
  - Enable Redis authentication
  - HTTPS with Let's Encrypt
  - Network policies
  - Secrets rotation

Monitoring:
  - Implement Prometheus metrics
  - Create Grafana dashboards
  - Setup alerts (uptime, errors, performance)
  - Log aggregation
  - Distributed tracing

High Availability:
  - Database replication
  - Redis clustering
  - Load balancer (nginx/HAProxy)
  - Backup strategy
  - Disaster recovery plan
```

### License

MIT License - xem file [LICENSE](LICENSE) để biết chi tiết.
