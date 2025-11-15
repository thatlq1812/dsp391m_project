# Deployment Scripts

Scripts for VM deployment and demo data collection.

## Quick Start (Demo VM)

```bash
# 1. Deploy VM on GCP
./scripts/deployment/deploy_demo_vm.sh

# 2. SSH and configure
gcloud compute ssh traffic-demo-collector --zone=asia-southeast1-a
cd ~/traffic-demo
nano .env  # Add API keys

# 3. Build topology
conda activate dsp
python scripts/data/01_collection/build_topology.py

# 4. Start collection
sudo systemctl enable traffic-collector.timer
sudo systemctl start traffic-collector.timer

# 5. Wait 3-5 days, then download data
gcloud compute scp traffic-demo-collector:/opt/traffic_data/traffic_data_*.parquet ./data/demo/
```

## Scripts Overview

### Demo VM Deployment

**`deploy_demo_vm.sh`** - Automated VM deployment (GCP)

- Creates e2-micro instance (FREE tier)
- Installs Miniconda, Python 3.10
- Clones repository
- Sets up systemd timer (every 15 minutes)
- Usage: `./scripts/deployment/deploy_demo_vm.sh`

**`setup_demo_vm_manual.sh`** - Manual VM setup

- Run ON the VM if auto-deploy fails
- Step-by-step installation
- Usage: `bash setup_demo_vm_manual.sh` (on VM)

**`vm_commands.md`** - VM management reference

- All gcloud commands
- Service management
- Monitoring and debugging
- Common issues and solutions

**`traffic_collector.py`** - Data collection script

- Collects traffic every 15 minutes
- Saves to monthly Parquet files
- Includes weather data
- Runs via systemd timer

### Old Scripts (Archived)

See `scripts/archive/deployment/` for old web API deployment scripts.

## Deployment Scripts (`deployment/`)

### `deploy_git.sh` ⭐ **Main deployment script**

- Git-based deployment workflow
- Auto push → pull → restart
- Regenerates topology if config changed

### `status.sh` - System status check

- Service status
- Recent collections
- Disk usage
- Topology info

### `monitor_logs.sh` - Real-time logs

- Tail adaptive_scheduler.log
- Watch collections live

### `restart.sh` - Restart service

- Stop → Start → Status
- Safe restart with verification

## Data Scripts (`data/`)

### `download_latest.sh` ⭐ **Download data**

- Interactive download
- Choose: latest / last 10 / 24h / all
- Requires gcloud

### `download_simple.sh` - No-auth download

- HTTP-based download
- No gcloud needed
- Requires VM server running

### `serve_data_public.sh` - Public server

- Run on VM
- Serves data via HTTP
- Port 8080

## Monitoring Scripts (`monitoring/`)

### `view_stats.sh` - Collection statistics

- Total runs
- Success rate
- Collections by hour
- Latest runs detail

### `health_check_remote.sh` - Health check

- 7-point system check
- Service, disk, memory, network
- Returns healthy/issues status

## See Full Documentation

- [QUICK_START_GIT.md](../docs/QUICK_START_GIT.md) - Quick start guide
- [README.md](./README.md) - Full scripts reference
- [DEPLOYMENT_GUIDE.md](../docs/v5/DEPLOYMENT_GUIDE.md) - Detailed deployment

## Migration from Old Scripts

| Old Script                   | New Script                          | Notes                    |
| ---------------------------- | ----------------------------------- | ------------------------ |
| `deploy_wizard.sh`           | `deployment/deploy_git.sh`          | Git-based, automated     |
| `control_panel.sh`           | `deployment/status.sh`              | Simpler, remote-friendly |
| `data/download_data.sh`      | `data/download_latest.sh`           | Interactive, better UX   |
| `monitoring/health_check.sh` | `monitoring/health_check_remote.sh` | Remote execution         |

Old scripts still work but new scripts are recommended for Git workflow.
