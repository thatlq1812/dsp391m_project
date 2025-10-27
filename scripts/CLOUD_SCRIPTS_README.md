# Cloud Deployment Scripts

Collection of automated scripts for deploying and managing the Traffic Forecast system on Google Cloud Platform.

## Overview

| Script | Purpose | Usage |
| --------------------------- | ---------------------------------------------- | ------------------------------------- |
| `preflight_check.sh` | Verify prerequisites before deployment | `./scripts/preflight_check.sh` |
| `deploy_week_collection.sh` | Full automated deployment for 7-day collection | `./scripts/deploy_week_collection.sh` |
| `monitor_collection.sh` | Quick status check of running collection | `./scripts/monitor_collection.sh` |
| `download_data.sh` | Download collected data from VM | `./scripts/download_data.sh` |
| `cloud_quickref.sh` | Quick reference for common commands | `./scripts/cloud_quickref.sh` |

## Quick Start

### 1. Pre-flight Check

Before deploying, verify all prerequisites:

```bash
chmod +x scripts/*.sh
./scripts/preflight_check.sh
```

**What it checks:**

- gcloud CLI installed
- Authenticated with GCP
- Project ID configured
- Billing enabled
- Required files exist
- API key (if using real API)

### 2. Deploy Collection System

**Option A: Mock API (FREE - Recommended for testing)**

```bash
export GCP_PROJECT_ID="your-project-id"
./scripts/deploy_week_collection.sh
```

**Option B: Real Google API (~$168 for 7 days)**

```bash
export GCP_PROJECT_ID="your-project-id"
export GOOGLE_MAPS_API_KEY="your-api-key"
USE_REAL_API=true ./scripts/deploy_week_collection.sh
```

### 3. Monitor Collection

```bash
# Quick status
./scripts/monitor_collection.sh

# View logs live
gcloud compute ssh traffic-collector-v4 --zone=asia-southeast1-b \
 --command="tail -f ~/dsp391m_project/logs/collector.log"
```

### 4. Download Data

After 7 days (or anytime):

```bash
./scripts/download_data.sh
```

## Detailed Script Documentation

### `preflight_check.sh`

**Purpose:** Validate environment before deployment

**Checks:**

- gcloud CLI installation and version
- GCP authentication status
- Project ID configuration
- Billing enabled
- API availability
- Configuration files
- Deployment scripts
- Disk space

**Exit codes:**

- 0: All checks passed
- 1: Critical checks failed

**Example output:**

```

 Pre-flight Check for Cloud Deployment


 gcloud CLI installed (version: 450.0.0)
 Authenticated as: your-email@gmail.com
 GCP_PROJECT_ID set: traffic-forecast-391
 Configuration file exists
 Deployment script ready
 Billing status unknown
 Using Mock API (FREE)

Checks passed: 6
Checks failed: 0
Warnings: 1

 All critical checks passed!
```

---

### `deploy_week_collection.sh`

**Purpose:** Complete automated deployment

**What it does:**

1. Creates GCP VM instance (e2-standard-2, Ubuntu 22.04)
2. Installs Miniconda and dependencies
3. Clones repository
4. Creates conda environment
5. Configures environment variables
6. Sets up systemd service
7. Starts data collection
8. Provides monitoring commands

**Environment variables:**

```bash
GCP_PROJECT_ID # Required: Your GCP project ID
GCP_ZONE # Optional: Default asia-southeast1-b
GCP_REGION # Optional: Default asia-southeast1
INSTANCE_NAME # Optional: Default traffic-collector-v4
MACHINE_TYPE # Optional: Default e2-standard-2
USE_REAL_API # Optional: true/false (default: false)
GOOGLE_MAPS_API_KEY # Required if USE_REAL_API=true
```

**Example:**

```bash
# Full custom deployment
GCP_PROJECT_ID="my-project" \
GCP_ZONE="us-central1-a" \
INSTANCE_NAME="my-collector" \
MACHINE_TYPE="e2-standard-4" \
./scripts/deploy_week_collection.sh
```

**Output files:**

- `deployment_info.txt` - Deployment details and commands

---

### `monitor_collection.sh`

**Purpose:** Quick status check

**Usage:**

```bash
# Default instance
./scripts/monitor_collection.sh

# Custom instance
./scripts/monitor_collection.sh my-instance-name my-zone
```

**Shows:**

- Systemd service status
- Recent log entries (last 20 lines)
- Disk usage
- Data files count

**Example output:**

```
=== Traffic Collection Status ===
 traffic-collector.service - Traffic Forecast Data Collection
 Loaded: loaded (/etc/systemd/system/traffic-collector.service)
 Active: active (running) since Fri 2025-10-25 14:00:00 UTC

=== Recent Logs ===
2025-10-25 14:30:15 - INFO - Collection cycle completed
2025-10-25 14:30:16 - INFO - 64 nodes, 192 edges collected
2025-10-25 14:30:17 - INFO - Next collection: 15:00:15

=== Disk Usage ===
/dev/sda1 50G 5.2G 42G 12% /

=== Data Files ===
run_20251025_140015/
run_20251025_143015/
run_20251025_150015/
```

---

### `download_data.sh`

**Purpose:** Download collected data from VM

**Usage:**

```bash
# Default (creates timestamped directory)
./scripts/download_data.sh

# Custom output directory
./scripts/download_data.sh traffic-collector-v4 asia-southeast1-b ./my_data
```

**Downloads:**

- `data/` - All collected data files
- `traffic_history.db` - SQLite database
- `logs/` - Collection logs
- Creates `README.md` with summary

**Example:**

```bash
./scripts/download_data.sh

# Output:
# Downloading data from traffic-collector-v4...
# Output directory: ./data_downloaded_20251101_120000
# Downloading data files...
# Downloading database...
# Downloading logs...
# Creating summary...
# Download complete!
# Location: ./data_downloaded_20251101_120000
```

---

### `cloud_quickref.sh`

**Purpose:** Display quick reference guide

**Usage:**

```bash
./scripts/cloud_quickref.sh
```

**Shows:**

- Deployment commands
- Monitoring commands
- Data download commands
- Control commands (start/stop)
- Cost management
- Troubleshooting tips
- Expected metrics
- Documentation links

---

## Common Workflows

### Workflow 1: First-time Deployment

```bash
# 1. Check prerequisites
./scripts/preflight_check.sh

# 2. Set project ID
export GCP_PROJECT_ID="your-project-id"

# 3. Deploy with mock API (free)
./scripts/deploy_week_collection.sh

# 4. Monitor for a few minutes
./scripts/monitor_collection.sh

# 5. Check logs
gcloud compute ssh traffic-collector-v4 --zone=asia-southeast1-b \
 --command="tail -f ~/dsp391m_project/logs/collector.log"
```

### Workflow 2: Production Deployment (Real API)

```bash
# 1. Check prerequisites
./scripts/preflight_check.sh

# 2. Configure
export GCP_PROJECT_ID="your-project-id"
export GOOGLE_MAPS_API_KEY="your-api-key"

# 3. Deploy with real API
USE_REAL_API=true ./scripts/deploy_week_collection.sh

# 4. Monitor daily
./scripts/monitor_collection.sh
```

### Workflow 3: Data Download After 7 Days

```bash
# 1. Check final status
./scripts/monitor_collection.sh

# 2. Download all data
./scripts/download_data.sh

# 3. Verify download
cd data_downloaded_*/
ls -lh data/
sqlite3 traffic_history.db "SELECT COUNT(*) FROM traffic_snapshots;"

# 4. Stop VM (stop billing)
gcloud compute instances stop traffic-collector-v4 --zone=asia-southeast1-b

# 5. Delete VM when done
gcloud compute instances delete traffic-collector-v4 --zone=asia-southeast1-b
```

### Workflow 4: Troubleshooting

```bash
# 1. View quick reference
./scripts/cloud_quickref.sh

# 2. Check service status
gcloud compute ssh traffic-collector-v4 --zone=asia-southeast1-b \
 --command="sudo systemctl status traffic-collector"

# 3. View error logs
gcloud compute ssh traffic-collector-v4 --zone=asia-southeast1-b \
 --command="tail -50 ~/dsp391m_project/logs/collector.error.log"

# 4. Manual test
gcloud compute ssh traffic-collector-v4 --zone=asia-southeast1-b
cd ~/dsp391m_project
conda activate dsp
python scripts/collect_and_render.py --once --no-visualize

# 5. Restart service if needed
sudo systemctl restart traffic-collector
```

---

## Configuration

### Custom Instance Configuration

Edit `deploy_week_collection.sh` variables:

```bash
# VM Configuration
ZONE="asia-southeast1-b" # Closest to Vietnam
MACHINE_TYPE="e2-standard-2" # 2 vCPU, 8GB RAM
DISK_SIZE="50GB" # Storage size
IMAGE_FAMILY="ubuntu-2204-lts" # OS image (Ubuntu 22.04 LTS)

# Collection Configuration
COLLECTION_DURATION_DAYS=7 # Collection period
USE_REAL_API="false" # true/false
```

### Custom Schedule

After deployment, SSH to VM and edit:

```bash
vim ~/dsp391m_project/configs/project_config.yaml

# Change intervals:
scheduler:
 adaptive:
 peak_hours:
 interval_minutes: 30 # Change to 15, 30, 60, etc.
 offpeak:
 interval_minutes: 60
 weekend:
 interval_minutes: 90

# Restart service
sudo systemctl restart traffic-collector
```

---

## Expected Results

### Collections per Day (Adaptive Schedule)

| Period | Interval | Collections | Percentage |
| ------------------- | -------- | ----------- | ---------- |
| Peak (5h) | 30 min | 10 | 29% |
| Off-peak (19h) | 60 min | 19 | 56% |
| Weekend (48h total) | 90 min | ~32 | 15% |
| **Total/Week** | - | **~235** | 100% |

### Data Volume

| Component | Per Collection | 7 Days |
| ------------ | -------------- | ---------- |
| Nodes JSON | ~50KB | ~12MB |
| Edges JSON | ~150KB | ~35MB |
| Weather JSON | ~5KB | ~1MB |
| Database | grows | ~100MB |
| Logs | ~10KB | ~2MB |
| **Total** | ~215KB | **~150MB** |

### Costs

**Mock API Mode:**

- VM: $12 (7 days)
- API: $0
- **Total: $12**

**Real API Mode:**

- VM: $12 (7 days)
- API: $168 (7 days)
- **Total: $180**

---

## Support

### Documentation

- [CLOUD_DEPLOY.md](../CLOUD_DEPLOY.md) - Full deployment guide
- [DEPLOY.md](../DEPLOY.md) - General deployment
- [README.md](../README.md) - Project overview

### Quick Help

```bash
./scripts/cloud_quickref.sh
```

### Contact

- **Author:** THAT Le Quang (Xiel)
- **Email:** fxlqthat@gmail.com
- **GitHub:** [thatlq1812](https://github.com/thatlq1812)

---

**Last Updated:** October 25, 2025 
**Version:** Academic v4.0
