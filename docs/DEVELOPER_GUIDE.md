# Developer Guide - Git-Based Workflow

## Overview

This guide is for developers who need to deploy code changes, modify configuration, or manage the data collection system on GCP VM.

**Prerequisites:**

- Git installed
- Google Cloud SDK (gcloud) installed and configured
- Access to GitHub repository: thatlq1812/dsp391m_project
- Python 3.10+ with conda

---

## Quick Reference

```bash
# Deploy changes to VM
./scripts/deployment/deploy_git.sh

# Check system status
./scripts/deployment/status.sh

# Monitor real-time logs
./scripts/deployment/monitor_logs.sh

# View collection statistics
./scripts/monitoring/view_stats.sh

# Health check
./scripts/monitoring/health_check_remote.sh

# Restart service
./scripts/deployment/restart.sh

# Download data
./scripts/data/download_latest.sh
```

---

## Initial Setup

### 1. Clone Repository

```bash
git clone https://github.com/thatlq1812/dsp391m_project.git
cd dsp391m_project
```

### 2. Setup Python Environment

```bash
# Create conda environment
conda env create -f environment.yml

# Activate environment
conda activate dsp
```

### 3. Configure gcloud

```bash
# Install gcloud CLI if not installed
# Download from: https://cloud.google.com/sdk/docs/install

# Initialize and login
gcloud init
gcloud auth login

# Set project
gcloud config set project sonorous-nomad-476606-g3
```

### 4. Test SSH Access

```bash
# Test connection to VM
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a

# If successful, exit
exit
```

---

## Development Workflow

### Making Configuration Changes

**Example: Change collection intervals**

1. Edit configuration locally:

```bash
vim configs/project_config.yaml
```

2. Modify scheduler settings:

```yaml
scheduler:
  adaptive:
    peak_hours:
      interval_minutes: 20 # Changed from 30
    offpeak:
      interval_minutes: 90 # Changed from 120
```

3. Commit changes:

```bash
git add configs/project_config.yaml
git commit -m "Update collection intervals: peak 20min, offpeak 90min"
```

4. Deploy to VM:

```bash
./scripts/deployment/deploy_git.sh
```

**What happens:**

- Script checks for uncommitted changes (fails if any)
- Pushes commits to GitHub
- Connects to VM via SSH
- Pulls latest code from GitHub
- Detects config changes
- Regenerates topology if needed
- Restarts traffic-collector service
- Verifies deployment

### Making Code Changes

**Example: Update data collector**

1. Make changes locally:

```bash
vim traffic_forecast/collectors/google/collector.py
```

2. Test locally (optional):

```bash
python scripts/collect_once.py
```

3. Commit and deploy:

```bash
git add .
git commit -m "Fix: Improve error handling in Google collector"
./scripts/deployment/deploy_git.sh
```

### Checking Deployment Status

After deployment, verify everything is working:

```bash
# Quick status check
./scripts/deployment/status.sh

# Detailed health check
./scripts/monitoring/health_check_remote.sh

# Watch logs in real-time
./scripts/deployment/monitor_logs.sh
```

---

## System Monitoring

### Status Check

Shows service status, recent collections, disk usage, topology info:

```bash
./scripts/deployment/status.sh
```

Output example:

```
SERVICE STATUS
--------------
Active: active (running) since Thu 2025-10-30 03:24:56
Main PID: 5879

RECENT COLLECTIONS
------------------
run_20251030_032457
run_20251030_032440
run_20251030_032246

DISK USAGE
----------
1.8M    data/runs/
Total runs: 7

TOPOLOGY
--------
Nodes: 64, Edges: 144, Avg: 2.2 edges/node
```

### Collection Statistics

View detailed collection statistics:

```bash
./scripts/monitoring/view_stats.sh
```

Shows:

- Total runs and time range
- Average edges per run
- Success rate
- Collections by hour (histogram)
- Latest 10 runs detail

### Health Check

7-point system health check:

```bash
./scripts/monitoring/health_check_remote.sh
```

Checks:

1. Service running
2. Recent collections (< 3 hours old)
3. Topology cache exists
4. Disk space (< 80%)
5. Memory usage (< 90%)
6. Error logs (< 5 errors in last 100 lines)
7. Network connectivity

Returns:

- Exit code 0: System healthy
- Exit code 1: Issues detected

### Real-time Monitoring

Monitor logs live:

```bash
./scripts/deployment/monitor_logs.sh
```

Press Ctrl+C to stop.

---

## Common Tasks

### Restart Service

If service is stuck or after manual VM changes:

```bash
./scripts/deployment/restart.sh
```

### Download Data for Analysis

```bash
./scripts/data/download_latest.sh
```

Choose option:

1. Latest run only
2. Last 10 runs
3. Last 24 hours
4. All data
5. Custom number of runs

Data downloaded to `./downloaded_data/`

### Manual VM Access

Direct SSH to VM:

```bash
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a
```

Common commands on VM:

```bash
# Check service status
sudo systemctl status traffic-collector

# View logs
tail -f ~/traffic-forecast/logs/adaptive_scheduler.log

# List runs
ls -lt ~/traffic-forecast/data/runs/ | head

# Check topology
cat ~/traffic-forecast/cache/overpass_topology.json | jq '.nodes | length'

# Manual collection (for testing)
cd ~/traffic-forecast
source ~/miniconda3/bin/activate dsp
python scripts/collect_once.py
```

### Emergency Stop

Stop data collection:

```bash
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="sudo systemctl stop traffic-collector"
```

### Emergency Restart

If deployment script fails, manual restart:

```bash
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="
    cd ~/traffic-forecast
    git pull origin master
    rm -f cache/overpass_topology.json
    source ~/miniconda3/bin/activate dsp
    python scripts/collect_once.py --force-refresh
    sudo systemctl restart traffic-collector
  "
```

---

## Configuration Reference

### Main Configuration File

Location: `configs/project_config.yaml`

**Key sections:**

#### Scheduler Configuration

```yaml
scheduler:
  enabled: true
  mode: adaptive # or 'fixed'

  adaptive:
    # Peak hours - frequent collection
    peak_hours:
      enabled: true
      time_ranges:
        - start: "06:30"
          end: "08:00"
        - start: "10:30"
          end: "11:30"
        - start: "16:00"
          end: "19:00"
      interval_minutes: 30

    # Off-peak hours - less frequent
    offpeak:
      enabled: true
      interval_minutes: 120
```

#### Coverage Area

```yaml
globals:
  area:
    mode: point_radius
    center: [106.697794, 10.772465] # HCMC center
    radius_m: 4096 # 4km radius
```

#### Node Selection

```yaml
node_selection:
  min_degree: 3 # Min roads at intersection
  min_importance_score: 17.5 # Quality threshold
  max_nodes: 64 # Target number of nodes
  min_distance_meters: 200 # Spacing between nodes
  road_type_filter:
    - motorway
    - trunk
    - primary
  min_street_name_count: 2 # Min unique street names
```

### Environment Variables

Location: `.env` (on VM only)

```bash
GOOGLE_MAPS_API_KEY=your_api_key_here
```

Never commit .env file to git!

---

## Troubleshooting

### Deployment Failed

**Error: Uncommitted changes**

```bash
# Check what changed
git status

# Option 1: Commit changes
git add .
git commit -m "Your message"

# Option 2: Discard changes
git restore <file>
```

**Error: SSH connection failed**

```bash
# Check VM is running
gcloud compute instances list

# Start VM if stopped
gcloud compute instances start traffic-forecast-collector \
  --zone=asia-southeast1-a
```

**Error: Permission denied**

```bash
# Re-authenticate
gcloud auth login
```

### Service Not Running

```bash
# Check status
./scripts/deployment/status.sh

# View recent errors
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a \
  --command="tail -50 ~/traffic-forecast/logs/adaptive_scheduler.log"

# Restart
./scripts/deployment/restart.sh
```

### No Recent Collections

Possible causes:

1. Service stopped - check status and restart
2. API quota exceeded - check Google Cloud Console
3. Network issues - check VM connectivity
4. Config error - verify project_config.yaml

Debug:

```bash
# Health check
./scripts/monitoring/health_check_remote.sh

# Watch logs
./scripts/deployment/monitor_logs.sh

# Manual test collection
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a
cd ~/traffic-forecast
source ~/miniconda3/bin/activate dsp
python scripts/collect_once.py
```

### Topology Changed Unexpectedly

If node/edge count differs from expected:

```bash
# Check config
cat configs/project_config.yaml | grep -A 10 node_selection

# Regenerate topology manually
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a
cd ~/traffic-forecast
rm cache/overpass_topology.json
source ~/miniconda3/bin/activate dsp
python scripts/collect_once.py --force-refresh
```

### API Errors

Check API key and quota:

```bash
# On VM, verify API key
gcloud compute ssh traffic-forecast-collector --zone=asia-southeast1-a
cat ~/traffic-forecast/.env | grep GOOGLE_MAPS_API_KEY

# Check Google Cloud Console for quota
# https://console.cloud.google.com/apis/api/directions-backend.googleapis.com
```

---

## Best Practices

### Git Workflow

1. Always pull before making changes:

   ```bash
   git pull origin master
   ```

2. Use descriptive commit messages:

   ```bash
   git commit -m "Fix: Correct timezone handling in scheduler"
   ```

3. Test locally before deploying (when possible)

4. Deploy during off-peak hours if making major changes

### Configuration Changes

1. Always backup current config:

   ```bash
   cp configs/project_config.yaml configs/project_config.yaml.backup
   ```

2. Test with small changes first

3. Monitor logs after deployment:

   ```bash
   ./scripts/deployment/monitor_logs.sh
   ```

4. Document why changes were made in commit message

### Monitoring

1. Check status daily:

   ```bash
   ./scripts/deployment/status.sh
   ```

2. Review statistics weekly:

   ```bash
   ./scripts/monitoring/view_stats.sh
   ```

3. Run health check before/after deployments:
   ```bash
   ./scripts/monitoring/health_check_remote.sh
   ```

---

## Scripts Reference

### Deployment Scripts

| Script            | Purpose                   | When to Use                        |
| ----------------- | ------------------------- | ---------------------------------- |
| `deploy_git.sh`   | Full Git-based deployment | After commits, config changes      |
| `status.sh`       | Check system status       | Daily monitoring, after deployment |
| `monitor_logs.sh` | Real-time log streaming   | Debugging, verify collections      |
| `restart.sh`      | Restart service only      | Service stuck, minor issues        |

### Monitoring Scripts

| Script                   | Purpose               | Output                                  |
| ------------------------ | --------------------- | --------------------------------------- |
| `view_stats.sh`          | Collection statistics | Runs, success rate, hourly distribution |
| `health_check_remote.sh` | 7-point health check  | Pass/fail with details                  |

### Data Scripts

| Script                 | Purpose                   | Use Case                    |
| ---------------------- | ------------------------- | --------------------------- |
| `download_latest.sh`   | Interactive data download | Get data for analysis       |
| `download_simple.sh`   | HTTP download (no auth)   | Team members without gcloud |
| `serve_data_public.sh` | Start HTTP server on VM   | Enable HTTP downloads       |

---

## Machine Learning Model Training

### Available Models

This project uses **Deep Learning models** for traffic forecasting:

1. **LSTM (Long Short-Term Memory)**

   - Temporal sequence modeling
   - Best for short-term forecasting (15-60 minutes)
   - Handles time-series dependencies

2. **ATSCGN (Adaptive Traffic Spatial-Temporal Convolutional Graph Network)**
   - Spatial-temporal graph modeling
   - Captures road network topology
   - Best for multi-node traffic prediction

### Training Workflow

#### 1. Prepare Data

Ensure you have collected traffic data:

```bash
# Download latest data
./scripts/data/download_latest.sh

# Or collect locally
python scripts/collect_once.py
```

#### 2. Train Models

```bash
# Activate environment
conda activate dsp

# Train LSTM model
python -m traffic_forecast.ml.dl_trainer \
  --model lstm \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 0.001

# Train ATSCGN model
python -m traffic_forecast.ml.dl_trainer \
  --model atscgn \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 0.001
```

#### 3. Monitor Training

Training progress is saved to:

- `models/checkpoints/` - Model checkpoints
- `models/history/` - Training history (loss, metrics)
- TensorBoard logs (if enabled)

```bash
# View training history
python -m traffic_forecast.ml.dl_trainer --model lstm --mode history

# Or use TensorBoard
tensorboard --logdir models/logs
```

#### 4. Evaluate Models

```bash
# Evaluate on test set
python -m traffic_forecast.ml.dl_trainer \
  --model lstm \
  --mode evaluate \
  --checkpoint models/checkpoints/lstm_best.h5
```

#### 5. Deploy Models

```bash
# Export for production
python -m traffic_forecast.ml.dl_trainer \
  --model lstm \
  --mode export \
  --checkpoint models/checkpoints/lstm_best.h5 \
  --output models/production/lstm_v1.0
```

### Model Configuration

Edit training parameters in `configs/project_config.yaml`:

```yaml
ml:
  lstm:
    hidden_units: 64
    num_layers: 2
    dropout: 0.2
    sequence_length: 12 # 3 hours at 15-min intervals

  atscgn:
    gcn_units: 64
    temporal_units: 64
    num_layers: 3
    dropout: 0.3
    sequence_length: 12

  training:
    batch_size: 32
    epochs: 100
    learning_rate: 0.001
    early_stopping_patience: 10
    validation_split: 0.2
```

### Training Best Practices

1. **Data Quality**

   - Ensure sufficient data (minimum 3 days of collections)
   - Check for missing values
   - Verify data consistency

2. **Hyperparameter Tuning**

   - Start with default parameters
   - Use validation set for tuning
   - Monitor for overfitting

3. **Model Selection**

   - Use LSTM for temporal patterns
   - Use ATSCGN when network topology is important
   - Compare both models on validation set

4. **Production Deployment**
   - Always validate on test set before deployment
   - Keep multiple model versions
   - Document model performance metrics

### Troubleshooting Training

**Out of Memory:**

```bash
# Reduce batch size
python -m traffic_forecast.ml.dl_trainer --model lstm --batch-size 16
```

**Slow Training:**

```bash
# Check GPU availability
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Use CPU-only TensorFlow if needed (already installed)
```

**Poor Performance:**

- Check data quality and preprocessing
- Increase model capacity (more layers/units)
- Collect more training data
- Tune hyperparameters

---

## Architecture Overview

```
Local Machine (You)
    |
    | git push
    v
GitHub Repository
    |
    | git pull
    v
GCP VM (asia-southeast1-a)
    |
    +-- traffic-collector.service (systemd)
    |       |
    |       +-- run_adaptive_collection.py
    |               |
    |               +-- collect_once.py
    |                       |
    |                       +-- Overpass API (topology)
    |                       +-- Open-Meteo API (weather)
    |                       +-- Google Maps API (traffic)
    |
    +-- Data Storage
            |
            +-- data/runs/run_YYYYMMDD_HHMMSS/
                    - nodes.json
                    - edges.json
                    - traffic_edges.json
                    - weather_snapshot.json
                    - statistics.json
```

---

## Additional Resources

- **GitHub Repository:** https://github.com/thatlq1812/dsp391m_project
- **Team Guide:** See TEAM_GUIDE.md for data download instructions
- **GCP Console:** https://console.cloud.google.com
- **Project ID:** sonorous-nomad-476606-g3
- **VM Name:** traffic-forecast-collector
- **VM Zone:** asia-southeast1-a
