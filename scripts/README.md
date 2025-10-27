# Maintainer Profile

**Full name:** THAT Le Quang  
**Nickname:** Xiel

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812](https://github.com/thatlq1812)
- **Primary email:** fxlqthat@gmail.com
- **Academic email:** thatlqse183256@fpt.edu.com
- **Alternate email:** thatlq1812@gmail.com
- **Phone (VN):** +84 33 863 6369 / +84 39 730 6450

---

# Scripts Organization

This directory contains all operational scripts for the Traffic Forecast project, organized by function.

## Directory Structure

### deployment/

Scripts for deploying and setting up the system on GCP or local environments.

- `deploy_preflight.sh` - Pre-deployment checks and validation
- `deploy_week_collection.sh` - Deploy week-long data collection to GCP
- `install_dependencies.sh` - Install system dependencies
- `vm_setup.sh` - Initial VM setup on GCP
- `vm_users_setup.sh` - Setup user access on GCP VM

### data_management/

Scripts for data download, backup, and cleanup operations.

- `download_data_compressed.sh` - Download compressed data from GCP (recommended)
- `download_data_legacy.sh` - Legacy download method (slower)
- `backup.sh` - Create backup of project data
- `cleanup_runs.py` - Remove old data collection runs
- `cleanup_failed_deployment.sh` - Clean up failed deployment artifacts

### collection/

Scripts for running data collection tasks.

- `collect_and_render.py` - Main collection script with visualization
- `collect_with_history.py` - Collection with historical data tracking
- `collection_start.sh` - Helper script to start collection
- `collection_monitor.sh` - Monitor running collection processes

### monitoring/

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
