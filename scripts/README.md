# Maintainer Profile

**Full name:**THAT Le Quang
**Nickname:**Xiel

- **Role:**AI & DS Major Student
- **GitHub:** [thatlq1812](https://github.com/thatlq1812)
- **Primary email:** fxlqthat@gmail.com
- **Academic email:** thatlqse183256@fpt.edu.com
- **Alternate email:** thatlq1812@gmail.com
- **Phone (VN):** +84 33 863 6369 / +84 39 730 6450

---

# Scripts Directory

This directory contains utility scripts for the Traffic Forecast v5.0 project, organized by functional groups.

## Directory Structure

```
scripts/
├── analysis/ # ML and data analysis scripts
├── data/ # Data management and cleanup
├── deployment/ # GCP deployment automation
├── monitoring/ # Health checks and dashboards
└── README.md # This file
```

---

## Analysis Scripts (`analysis/`)

Machine learning analysis and model evaluation tools.

### `feature_importance_analysis.py`

Analyze feature importance from trained models.

**Usage:**

```bash
conda run -n dsp python scripts/analysis/feature_importance_analysis.py \
--model models/xgboost_model.pkl \
--output reports/feature_importance.png
```

### `cross_validation.py`

Perform k-fold cross-validation on ML models.

**Usage:**

```bash
conda run -n dsp python scripts/analysis/cross_validation.py \
--data data/training/features.csv \
--model xgboost \
--folds 5
```

### `quick_summary.py`

Generate quick summary statistics from collected data.

**Usage:**

```bash
conda run -n dsp python scripts/analysis/quick_summary.py \
--input data/downloads/download_20250128_140502/
```

---

## Data Management (`data/`)

Scripts for cleaning, backing up, and managing collected data.

### `cleanup_runs.py`

Remove data runs older than specified days to free disk space.

**Usage:**

```bash
# Remove runs older than 14 days
conda run -n dsp python scripts/data/cleanup_runs.py --days 14

# Dry run (preview only)
conda run -n dsp python scripts/data/cleanup_runs.py --days 14 --dry-run
```

**Configuration:**

- Default retention: 14 days
- Keeps: Latest run + runs within retention period
- Removes: `data/downloads/download_*` directories

### `backup.sh`

Backup collected data to cloud storage (Google Cloud Storage).

**Usage:**

```bash
# Backup all data
bash scripts/data/backup.sh

# Backup specific run
bash scripts/data/backup.sh data/downloads/download_20250128_140502
```

**Requirements:**

- `gsutil` installed and configured
- GCS bucket: `gs://traffic-forecast-backups/`

### `download_data.sh`

Download data from remote VM to local machine.

**Usage:**

```bash
# Download latest run
bash scripts/data/download_data.sh

# Download specific run
bash scripts/data/download_data.sh download_20250128_140502
```

**Configuration:**

- VM name: From `.env` → `VM_NAME`
- GCP zone: From `.env` → `GCP_ZONE`
- Destination: `data/vm_downloads/`

### `download_data_compressed.sh`

Download data from VM with compression (faster for large datasets).

**Usage:**

```bash
# Compress and download all data
bash scripts/data/download_data_compressed.sh
```

**Process:**

1. SSH to VM
2. Create tarball: `data/downloads.tar.gz`
3. Download compressed file
4. Extract locally
5. Clean up remote tarball

---

## Deployment (`deployment/`)

Automated deployment to Google Cloud Platform.

### `gcp_vm_deploy.sh`

One-command deployment of Traffic Forecast to GCP VM.

**Usage:**

```bash
# Full deployment (create VM + setup + start collection)
bash scripts/deployment/gcp_vm_deploy.sh

# Setup only (VM already exists)
bash scripts/deployment/gcp_vm_deploy.sh --setup-only

# Dry run (check configuration)
bash scripts/deployment/gcp_vm_deploy.sh --dry-run
```

**What it does:**

1. Creates GCP VM instance (e2-micro, Ubuntu 22.04)
2. Uploads project files (excludes `data/`)
3. Installs Python environment (conda/venv)
4. Installs package dependencies
5. Configures cron job for hourly collection
6. Verifies API connectivity
7. Starts first collection run

**Prerequisites:**

- `gcloud` CLI installed and authenticated
- `.env` file configured:
```bash
GCP_PROJECT_ID=your-project-id
GCP_ZONE=asia-southeast1-a
VM_NAME=traffic-collector-v5
GOOGLE_MAPS_API_KEY=your-api-key
```

**Output:**

- VM external IP
- SSH command for access
- Collection logs location
- Cron job status

**For detailed step-by-step deployment guide, see:**

- `doc/v5/NEW_GCP_PROJECT_DEPLOYMENT.md` - Full deployment guide for new GCP projects
- `notebooks/GCP_DEPLOYMENT.ipynb` - Interactive VM management notebook

---

## Monitoring (`monitoring/`)

Health checks, dashboards, and collection monitoring.

### `health_check.sh`

Check system health and collection status.

**Usage:**

```bash
# Run health check
bash scripts/monitoring/health_check.sh

# JSON output (for automation)
bash scripts/monitoring/health_check.sh --json
```

**Checks:**

- VM status (running/stopped)
- Disk usage (warn if > 80%)
- Last collection timestamp
- Collection success rate (last 24 hours)
- API quota usage
- Cron job status

### `live_dashboard.py`

FastAPI-based web dashboard for real-time monitoring.

**Usage:**

```bash
# Start dashboard (local)
conda run -n dsp python scripts/monitoring/live_dashboard.py

# Start on VM (accessible from outside)
conda run -n dsp python scripts/monitoring/live_dashboard.py --host 0.0.0.0 --port 8080
```

**Access:**

- Local: http://localhost:8000
- VM: http://VM_EXTERNAL_IP:8080

**Features:**

- Real-time collection metrics
- Success rate trends (hourly/daily)
- Node coverage map
- Cost tracking
- Recent logs
- Alert notifications

**API Endpoints:**

- `GET /` - Dashboard HTML
- `GET /api/status` - System status JSON
- `GET /api/metrics` - Collection metrics
- `GET /api/logs` - Recent logs

### `monitor_collection.sh`

Monitor collection process and send alerts.

**Usage:**

```bash
# Run once
bash scripts/monitoring/monitor_collection.sh

# Run in background (check every 5 minutes)
bash scripts/monitoring/monitor_collection.sh --daemon --interval 300
```

**Alerts:**

- Collection failed (success rate < 95%)
- High API usage (> 80% quota)
- Low disk space (< 2GB free)
- Cron job not running

**Notification channels:**

- Email (via sendmail)
- Slack webhook (if configured)
- Log file: `logs/monitor.log`

---

## VS Code Tasks

Pre-configured tasks available in `.vscode/tasks.json`:

### `Collect Only`

Run one collection cycle without visualization.

```bash
# Press: Ctrl+Shift+B → Select "Collect Only"
```

### `Collect Loop (1m)`

Run collection loop every 1 minute (for testing).

```bash
# Press: Ctrl+Shift+P → Tasks: Run Task → "Collect Loop (1m)"
```

### `Collect Loop (15m)`

Run collection loop every 15 minutes.

### `Visualize Latest`

Generate visualization for latest run.

### `Run Dashboard`

Start FastAPI live dashboard.

### `Cleanup Old Runs`

Remove runs older than 14 days.

---

## Quick Reference

### Common Workflows

**1. Local Testing**

```bash
# Run one collection
conda run -n dsp python -m traffic_forecast.cli collect --once

# Validate results
conda run -n dsp python scripts/analysis/quick_summary.py

# Visualize
conda run -n dsp python -m traffic_forecast.cli visualize
```

**2. Deploy to GCP**

```bash
# One-command deployment
bash scripts/deployment/gcp_vm_deploy.sh

# Or use interactive notebook
jupyter notebook notebooks/GCP_DEPLOYMENT.ipynb
```

**3. Monitor Production**

```bash
# Check health
bash scripts/monitoring/health_check.sh

# Start dashboard
conda run -n dsp python scripts/monitoring/live_dashboard.py

# Download latest data
bash scripts/data/download_data.sh
```

**4. Data Cleanup**

```bash
# Remove old runs (keep 14 days)
conda run -n dsp python scripts/data/cleanup_runs.py --days 14

# Backup to cloud
bash scripts/data/backup.sh
```

---

## Environment Setup

All scripts assume:

1. **Conda environment 'dsp'** activated:

```bash
conda activate dsp
```

2. **`.env` file** configured:

```bash
GOOGLE_MAPS_API_KEY=your-api-key
GCP_PROJECT_ID=your-project-id
GCP_ZONE=asia-southeast1-a
VM_NAME=traffic-collector-v5
```

3. **Project installed** in development mode:
```bash
pip install -e .
```

---

## Troubleshooting

### Script Fails with "ModuleNotFoundError"

**Solution:**

```bash
# Ensure package is installed
pip install -e .

# Verify installation
python -c "from traffic_forecast import __version__; print(__version__)"
```

### Cannot Connect to VM

**Solution:**

```bash
# Check VM status
gcloud compute instances list

# Start VM if stopped
gcloud compute instances start traffic-collector-v5 --zone=asia-southeast1-a

# Verify SSH access
gcloud compute ssh traffic-collector-v5 --zone=asia-southeast1-a
```

### Cleanup Script Removes Too Much Data

**Solution:**

```bash
# Always dry-run first
python scripts/data/cleanup_runs.py --days 14 --dry-run

# Adjust retention period
python scripts/data/cleanup_runs.py --days 30 # Keep 30 days
```

### Dashboard Not Accessible

**Solution:**

```bash
# Check if running
ps aux | grep live_dashboard

# Check port availability
netstat -tulpn | grep 8000

# Open firewall (on VM)
sudo ufw allow 8000
```

---

## Migration from Old Scripts

If you have old collection scripts, here's the migration guide:

| Old Script | New Equivalent | Notes |
| ------------------------ | ------------------------------------------------- | ------------------- |
| `collect_and_render.py` | `python -m traffic_forecast.cli collect` | Use CLI instead |
| `collect_optimized.py` | `python -m traffic_forecast.cli collect --no-viz` | Same functionality |
| `validate_collection.py` | `scripts/analysis/quick_summary.py` | Enhanced validation |
| `deploy_wizard.sh` | `scripts/deployment/gcp_vm_deploy.sh` | Fully automated |
| `vm_setup.sh` | Part of `gcp_vm_deploy.sh` | Integrated |
| Old cleanup scripts | `scripts/data/cleanup_runs.py` | Safer with dry-run |

**To migrate:**

1. Update your workflows to use new script paths
2. Test with `--dry-run` flags first
3. Remove old scripts after verification

---

## Documentation

- **Main docs:** `/doc/v5/`
- **Deployment guide:** `/doc/v5/NEW_GCP_PROJECT_DEPLOYMENT.md`
- **Architecture:** `/doc/v5/BAO_CAO_CAI_TIEN_V5.md`
- **API reference:** `/doc/v5/README_V5.md`

---

## Support

For issues or questions:

1. Check `/doc/v5/` documentation
2. Review error logs in `logs/`
3. Use `--help` flag on any script
4. Check VS Code tasks for pre-configured workflows

**Version:** 5.0.0
**Last updated:**January 2025

Scripts for system monitoring and dashboards.

- `health_check.sh` - Check system health status
- `live_dashboard.py` - FastAPI-based live dashboard for data visualization

### utilities/

Utility scripts for maintenance and fixes.

- `backfill_overpass_data.py` - Backfill missing Overpass data
- `fix_overpass_cache.sh` - Fix Overpass cache issues
- `quick_start.sh` - Interactive quick start wizard

### deprecated/

Old scripts kept for reference but no longer actively used.

## Common Usage Patterns

### Local Development

```bash
# Quick start (interactive)
bash scripts/utilities/quick_start.sh

# Single collection
conda run -n dsp python scripts/collection/collect_and_render.py --once

# Start live dashboard
conda run -n dsp python scripts/monitoring/live_dashboard.py
```

### Data Management

```bash
# Download latest data from GCP
bash scripts/data_management/download_data_compressed.sh

# Cleanup old runs (keep last 14 days)
conda run -n dsp python scripts/data_management/cleanup_runs.py --days 14

# Backup data
bash scripts/data_management/backup.sh
```

### Deployment

```bash
# Pre-deployment check
bash scripts/deployment/deploy_preflight.sh

# Deploy to GCP
bash scripts/deployment/deploy_week_collection.sh

# Setup new VM
bash scripts/deployment/vm_setup.sh
```

### Monitoring

```bash
# Check system health
bash scripts/monitoring/health_check.sh

# Monitor collection
bash scripts/collection/collection_monitor.sh
```

## Environment Requirements

Most scripts require the `dsp` conda environment. Use:

```bash
conda activate dsp
# or
conda run -n dsp <command>
```

## Notes

- All scripts use bash shell (not sh)
- Scripts are designed to be run from project root directory
- Most Python scripts have `--help` option for detailed usage
- See `CLOUD_SCRIPTS_README.md` for cloud-specific documentation
