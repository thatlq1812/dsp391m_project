# Deployment Guide - Traffic Forecast v5.1

Complete guide for deploying to Google Cloud Platform.

## 📋 Prerequisites

### Required Software
- **gcloud CLI**: [Install here](https://cloud.google.com/sdk/docs/install)
- **Git**: For cloning repository
- **Python 3.10+**: Local testing

### Required Accounts
- **Google Cloud Account** with billing enabled
- **Google Maps API Key** with Directions API enabled

### Required Files
```bash
# Verify these exist before deploying:
configs/project_config.yaml
cache/overpass_topology.json
.env (with GOOGLE_MAPS_API_KEY)
```

## 🚀 Deployment Methods

### Method 1: Interactive Wizard (Recommended)

```bash
# Run deployment wizard
bash scripts/deploy_wizard.sh

# Select option A for auto deployment
# Or choose steps 1-7 individually:
#   1) Check Prerequisites
#   2) Select GCP Project
#   3) Create VM
#   4) Upload Code
#   5) Setup Environment
#   6) Test Collection
#   7) Start Adaptive Scheduler
```

The wizard will:
- ✅ Verify prerequisites
- ✅ Create VM (e2-micro, free tier eligible)
- ✅ Upload and configure code
- ✅ Install dependencies
- ✅ Test collection
- ✅ Start systemd service

### Method 2: Manual Deployment

#### Step 1: Authenticate with GCP

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

#### Step 2: Create VM Instance

```bash
gcloud compute instances create traffic-forecast-collector \
  --zone=asia-southeast1-a \
  --machine-type=e2-micro \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=30GB \
  --boot-disk-type=pd-standard
```

#### Step 3: Upload Code

```bash
# Create archive
tar -czf traffic-forecast.tar.gz \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  --exclude='.git' \
  --exclude='data/runs/*' \
  traffic_forecast/ scripts/ configs/ cache/ .env \
  requirements.txt pyproject.toml setup.py

# Upload to VM
gcloud compute scp traffic-forecast.tar.gz \
  traffic-forecast-collector:~/ \
  --zone=asia-southeast1-a

# Extract on VM
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="
    mkdir -p ~/traffic-forecast
    tar -xzf ~/traffic-forecast.tar.gz -C ~/traffic-forecast
    rm ~/traffic-forecast.tar.gz
  "
```

#### Step 4: Setup Environment

```bash
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a

# On VM:
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh

# Create environment
conda create -n dsp python=3.10 -y
conda activate dsp

# Install packages
cd ~/traffic-forecast
pip install -r requirements.txt
pip install --no-deps -e .

# Create directories
mkdir -p logs data/runs
```

#### Step 5: Test Collection

```bash
# Still on VM
cd ~/traffic-forecast
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dsp

python scripts/collect_once.py
```

#### Step 6: Create Systemd Service

```bash
# Create service file
sudo tee /etc/systemd/system/traffic-collection.service > /dev/null << 'EOF'
[Unit]
Description=Traffic Forecast Adaptive Collection v5.1
After=network.target

[Service]
Type=simple
User=YOUR_USERNAME
WorkingDirectory=/home/YOUR_USERNAME/traffic-forecast
ExecStart=/home/YOUR_USERNAME/miniconda3/envs/dsp/bin/python scripts/run_adaptive_collection.py
Restart=always
RestartSec=60
StandardOutput=append:/home/YOUR_USERNAME/traffic-forecast/logs/service.log
StandardError=append:/home/YOUR_USERNAME/traffic-forecast/logs/service_error.log

[Install]
WantedBy=multi-user.target
EOF

# Start service
sudo systemctl daemon-reload
sudo systemctl enable traffic-collection.service
sudo systemctl start traffic-collection.service

# Check status
sudo systemctl status traffic-collection.service
```

## 📊 Monitoring

### Check Service Status

```bash
# Service status
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="sudo systemctl status traffic-collection.service"

# View logs
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="tail -f ~/traffic-forecast/logs/service.log"

# Check collections
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="ls -la ~/traffic-forecast/data/runs/ | tail -20"
```

### Monitor Costs

```bash
# View current costs
gcloud billing accounts list
gcloud billing projects describe YOUR_PROJECT_ID

# Set budget alert (recommended)
# Go to: https://console.cloud.google.com/billing/budgets
```

### Collection Statistics

```bash
# SSH to VM
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a

# Check collection count
ls ~/traffic-forecast/data/runs/ | wc -l

# View latest collection
ls -ltr ~/traffic-forecast/data/runs/ | tail -1

# Disk usage
du -sh ~/traffic-forecast/data/runs/
```

## 💾 Data Download

### Download All Collections

```bash
# Download to local machine
gcloud compute scp --recurse \
  traffic-forecast-collector:~/traffic-forecast/data/runs \
  ./data-downloaded-$(date +%Y%m%d) \
  --zone=asia-southeast1-a
```

### Download Specific Run

```bash
# List runs first
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="ls ~/traffic-forecast/data/runs/"

# Download specific run
gcloud compute scp --recurse \
  traffic-forecast-collector:~/traffic-forecast/data/runs/run_YYYYMMDD_HHMMSS \
  ./local-run \
  --zone=asia-southeast1-a
```

## 🛑 Cleanup

### Stop VM (Keep Data)

```bash
gcloud compute instances stop traffic-forecast-collector \
  --zone=asia-southeast1-a
```

### Restart Stopped VM

```bash
gcloud compute instances start traffic-forecast-collector \
  --zone=asia-southeast1-a

# Restart service
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="sudo systemctl restart traffic-collection.service"
```

### Delete VM (Remove Everything)

```bash
# DANGER: This deletes all data!
gcloud compute instances delete traffic-forecast-collector \
  --zone=asia-southeast1-a
```

## 💰 Cost Estimation

### 3-Day Production Run

**VM Costs (e2-micro):**
- Compute: $0.006738/hour × 24h × 3 days = **$0.49**
- Disk (30GB): $0.040/GB/month × 30GB × (3/30) = **$0.12**

**API Costs:**
- Google Directions: ~150 collections × 234 edges × $0.005 = **~$180** ⚠️
- Weather (Open-Meteo): Free
- Overpass: Free (cached)

**Estimated Total: ~$180** (mostly API calls)

💡 **Cost Optimization:**
- Adaptive scheduling saves 40% vs constant interval
- Weather grid caching reduces API calls by 95%
- Permanent topology cache (no repeat calls)

### 7-Day Production Run

- VM: ~$1.40
- APIs: ~$420
- **Total: ~$421**

## 🔧 Troubleshooting

### Service Won't Start

```bash
# Check logs
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="tail -100 ~/traffic-forecast/logs/service_error.log"

# Check service status
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="sudo systemctl status traffic-collection.service -l"

# Restart service
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="sudo systemctl restart traffic-collection.service"
```

### Collection Fails

```bash
# Test collection manually
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="
    cd ~/traffic-forecast
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate dsp
    python scripts/collect_once.py
  "
```

### API Key Issues

```bash
# Check API key on VM
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="grep GOOGLE_MAPS_API_KEY ~/traffic-forecast/.env"

# Test API
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="
    cd ~/traffic-forecast
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate dsp
    python tools/test_google_limited.py
  "
```

### Disk Full

```bash
# Check disk usage
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="df -h"

# Clean old runs
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="
    cd ~/traffic-forecast
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate dsp
    python scripts/cleanup_runs.py --days 7
  "
```

## 📝 Best Practices

1. **Before Deployment:**
   - ✅ Test collection locally first
   - ✅ Verify API key works
   - ✅ Check cache files exist
   - ✅ Set budget alerts in GCP

2. **During Collection:**
   - 📊 Monitor logs daily
   - 💰 Check costs regularly
   - 💾 Download data incrementally
   - 🔄 Restart service if errors

3. **After Collection:**
   - 💾 Download all data
   - 🛑 Stop or delete VM
   - 📊 Analyze results
   - 📝 Document findings

## 🆘 Support

- **Deployment Wizard**: `bash scripts/deploy_wizard.sh`
- **Control Panel**: `bash scripts/control_panel.sh`
- **Documentation**: `docs/` directory
- **Issues**: GitHub Issues

---

**Traffic Forecast v5.1** - Production Deployment Guide
