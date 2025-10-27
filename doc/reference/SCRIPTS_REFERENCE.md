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
# Scripts Reference Guide
This document describes all scripts in the `scripts/` directory, organized by category.
**Last Updated:** October 27, 2025 
**Version:** Academic v4.0.2
---
## Table of Contents
1. [Quick Reference Table](#quick-reference-table)
2. [Deployment Scripts](#deployment-scripts)
3. [Data Collection Scripts](#data-collection-scripts)
4. [Data Management Scripts](#data-management-scripts)
5. [Maintenance Scripts](#maintenance-scripts)
6. [Monitoring Scripts](#monitoring-scripts)
7. [Utility Scripts](#utility-scripts)
8. [Deprecated Scripts](#deprecated-scripts)
---
## Quick Reference Table
| Script | Category | Purpose | Usage |
| ------------------------------ | ----------- | ----------------------------- | ---------------------------------------------------- |
| `deploy_week_collection.sh` | Deployment | Full automated GCP deployment | `bash scripts/deploy_week_collection.sh` |
| `vm_setup.sh` | Deployment | Setup VM environment | Auto-called by deploy script |
| `deploy_preflight.sh` | Deployment | Pre-deployment checks | `bash scripts/deploy_preflight.sh` |
| `vm_users_setup.sh` | Deployment | Add team SSH access | `bash scripts/vm_users_setup.sh` |
| `cleanup_failed_deployment.sh` | Deployment | Cleanup failed deploy | `bash scripts/cleanup_failed_deployment.sh` |
| `collect_and_render.py` | Collection | Main collection script | `python scripts/collect_and_render.py --once` |
| `collection_start.sh` | Collection | Start collection service | `bash scripts/collection_start.sh` |
| `collection_monitor.sh` | Monitoring | Monitor collection status | `bash scripts/collection_monitor.sh` |
| `download_data_compressed.sh` | Data | Fast compressed download | `bash scripts/download_data_compressed.sh` |
| `download_data_legacy.sh` | Data | Old download (deprecated) | Not recommended |
| `backfill_overpass_data.py` | Data | Backfill missing data | `python scripts/backfill_overpass_data.py --dry-run` |
| `cleanup_runs.py` | Data | Remove old collections | `python scripts/cleanup_runs.py --days 14` |
| `backup.sh` | Maintenance | Backup data/config | `bash scripts/backup.sh` |
| `fix_overpass_cache.sh` | Maintenance | Fix cache bug | `bash scripts/fix_overpass_cache.sh` |
| `health_check.sh` | Monitoring | System health check | `bash scripts/health_check.sh` |
| `live_dashboard.py` | Monitoring | Web dashboard | `python scripts/live_dashboard.py` |
| `quick_start.sh` | Utility | Interactive setup | `bash scripts/quick_start.sh` |
---
6. [Utility Scripts](#utility-scripts)
7. [Deprecated Scripts](#deprecated-scripts)
---
## Deployment Scripts
### `deploy_week_collection.sh`
**Purpose:** Fully automated 7-day data collection deployment to GCP VM
**Usage:**
```bash
bash scripts/deploy_week_collection.sh
```
**Features:**
- Creates and configures GCP VM instance
- Installs all dependencies
- Sets up systemd service
- Starts data collection automatically
- Estimated time: 10-15 minutes
**Prerequisites:**
- GCP CLI configured (`gcloud auth login`)
- Project ID set (`gcloud config set project PROJECT_ID`)
- Billing enabled
**What it does:**
1. Creates e2-standard-2 VM in asia-southeast1-b
2. Uploads project code
3. Runs `vm_setup.sh` for environment setup
4. Configures systemd service for auto-start
5. Starts collection with adaptive scheduling
---
### `vm_setup.sh`
**Purpose:** Setup script that runs ON the VM to install dependencies
**Usage:**
```bash
# Called automatically by deploy_week_collection.sh
# Or manually on VM:
bash scripts/vm_setup.sh
```
**What it does:**
1. Installs Miniconda3
2. Creates conda environment from `environment.yml`
3. Installs Python packages
4. Sets up directory structure
5. Configures environment variables
**Time:** ~5-7 minutes
---
### `deploy_preflight.sh`
**Purpose:** Pre-deployment validation checklist
**Usage:**
```bash
bash scripts/deploy_preflight.sh
```
**Checks:**
- GCP authentication status
- Project configuration
- Required files present
- Environment file validity
- Code syntax (Python files)
- Config file structure
**Exit codes:**
- 0: All checks passed
- 1: One or more checks failed
---
### `cleanup_failed_deployment.sh`
**Purpose:** Cleanup resources after failed deployment
**Usage:**
```bash
bash scripts/cleanup_failed_deployment.sh [instance_name]
```
**What it does:**
- Deletes VM instance
- Removes firewall rules
- Cleans up disk snapshots
- Lists remaining resources
---
### `vm_users_setup.sh`
**Purpose:** Add team members SSH access to GCP VM
**Usage:**
```bash
bash scripts/vm_users_setup.sh
```
**Features:**
- Adds SSH keys for team members
- Sets up proper permissions
- Configures sudo access
---
## Data Collection Scripts
### `collect_and_render.py`
**Purpose:** Main data collection script with scheduling
**Usage:**
```bash
# Single collection
python scripts/collect_and_render.py --once
# Continuous with interval
python scripts/collect_and_render.py --interval 900 # 15 minutes
# Print schedule only
python scripts/collect_and_render.py --print-schedule
# Disable visualization
python scripts/collect_and_render.py --interval 1800 --no-visualize
```
**Arguments:**
- `--once`: Run one collection cycle then exit
- `--interval SECONDS`: Collection interval (default: 1800)
- `--adaptive`: Use adaptive scheduling
- `--no-visualize`: Skip visualization generation
- `--print-schedule`: Show schedule and exit
**Collectors:**
1. Overpass - Road network topology (40 major nodes)
2. Google Directions - Traffic conditions (120 edges)
3. Open-Meteo - Weather data
---
### `collect_with_history.py`
**Purpose:** Collection with SQLite database storage (alternative to file-based)
**Usage:**
```bash
python scripts/collect_with_history.py
```
**Note:** Currently using file-based storage. This is for future SQLite implementation.
---
### `collection_start.sh`
**Purpose:** Helper script to start collection service
**Usage:**
```bash
bash scripts/collection_start.sh
```
**What it does:**
- Activates conda environment
- Starts collection with default settings
- Runs in background with logging
---
### `collection_monitor.sh`
**Purpose:** Monitor active collection status
**Usage:**
```bash
bash scripts/collection_monitor.sh
```
**Shows:**
- Current collection run
- Last collection timestamp
- Service status
- Recent log entries
---
## Data Management Scripts
### `download_data_compressed.sh`
**Purpose:** Download data from VM as compressed archive (RECOMMENDED)
**Usage:**
```bash
# Default (tar.gz)
bash scripts/download_data_compressed.sh
# Custom format and location
bash scripts/download_data_compressed.sh \
traffic-collector-v4 \
asia-southeast1-b \
./data/downloads/my_data \
zip
```
**Arguments:**
1. Instance name (default: traffic-collector-v4)
2. Zone (default: asia-southeast1-b)
3. Output directory (default: ./data/downloads/download_TIMESTAMP)
4. Format: tar.gz or zip (default: tar.gz)
**Performance:**
- 5-10x faster than recursive scp
- ~90% compression ratio
- Auto-cleanup on VM and local
**Steps:**
1. Creates archive on VM
2. Downloads single file
3. Extracts locally
4. Cleans up archive on VM
5. Generates README.md
---
### `download_data_legacy.sh`
**Purpose:** Old download method using recursive scp (DEPRECATED)
**Usage:**
```bash
bash scripts/download_data_legacy.sh
```
**Note:** Shows deprecation warning and prompts to use compressed method.
---
### `backfill_overpass_data.py`
**Purpose:** Backfill missing Overpass data to existing collections
**Usage:**
```bash
# Dry run first
python scripts/backfill_overpass_data.py --data-dir data --dry-run
# Apply changes
python scripts/backfill_overpass_data.py --data-dir data
```
**What it does:**
1. Finds a collection with valid Overpass data
2. Validates data structure (40 nodes, edges)
3. Copies to all collections missing Overpass data
4. Reports statistics
**Use case:** Fix collections that failed Overpass collection due to cache bug
---
### `cleanup_runs.py`
**Purpose:** Remove old collection runs to free disk space
**Usage:**
```bash
# Remove runs older than 14 days
python scripts/cleanup_runs.py --days 14
# Dry run
python scripts/cleanup_runs.py --days 7 --dry-run
```
**Arguments:**
- `--days N`: Remove runs older than N days
- `--dry-run`: Show what would be deleted without deleting
---
### `backup.sh`
**Purpose:** Backup data and configuration files
**Usage:**
```bash
bash scripts/backup.sh
```
**Backs up:**
- All data files
- Configuration files
- Database (if exists)
- Logs
**Output:** Creates timestamped archive in backups/
---
## Maintenance Scripts
### `fix_overpass_cache.sh`
**Purpose:** Deploy cache bug fix to production VM
**Usage:**
```bash
bash scripts/fix_overpass_cache.sh
```
**What it does:**
1. Uploads fixed `cache_utils.py`
2. Uploads backfill script
3. Clears corrupted cache
4. Restarts collection service
5. Runs backfill (with confirmation)
**Use case:** Fix KeyError 'nodes' cache bug on production
---
### `health_check.sh`
**Purpose:** Comprehensive system health check
**Usage:**
```bash
bash scripts/health_check.sh
```
**Checks:**
- VM status and uptime
- Service status
- Disk usage
- Recent collections
- Log errors
- Data integrity
**Exit codes:**
- 0: All healthy
- 1: Issues detected
---
### `install_dependencies.sh`
**Purpose:** Install system-level dependencies on VM
**Usage:**
```bash
bash scripts/install_dependencies.sh
```
**Installs:**
- Python build tools
- System libraries
- Git
- Other required packages
**Note:** Called automatically by `vm_setup.sh`
---
## Monitoring Scripts
### `live_dashboard.py`
**Purpose:** FastAPI-based live dashboard for collection monitoring
**Usage:**
```bash
python scripts/live_dashboard.py
```
**Features:**
- Real-time collection status
- Data statistics
- Recent runs
- Error tracking
- Web interface on http://localhost:8000
**Note:** Experimental - not used in production yet
---
## Utility Scripts
### `quick_start.sh`
**Purpose:** Interactive quick start guide
**Usage:**
```bash
bash scripts/quick_start.sh
```
**Features:**
- Interactive menu
- Guides through setup
- Runs basic tests
- Shows common commands
---
### `CLOUD_SCRIPTS_README.md`
**Purpose:** Legacy documentation for cloud scripts
**Note:** Information now consolidated into this reference guide.
---
## Deprecated Scripts
Located in `scripts/deprecated/` - kept for reference but not actively maintained.
### `add_teammate_access.sh`
- Empty placeholder
- Functionality moved to `vm_users_setup.sh`
### `check_images.sh`
- Image validation script
- No longer needed (visualization disabled in production)
### `cleanup.sh`
- Old cleanup script
- Replaced by `cleanup_runs.py`
### `cloud_quickref.sh`
- Quick reference display script
- Information moved to `doc/QUICKREF.md`
### `deploy_wizard.sh`
- Interactive deployment wizard
- Replaced by `deploy_week_collection.sh`
### `fix_nodes_issue.sh`
- Emergency fix for nodes.json issue (Oct 26, 2025)
- Replaced by `fix_overpass_cache.sh`
---
## Script Naming Conventions
**Format:** `category_action.sh` or `action_target.py`
**Categories:**
- `deploy_*` - Deployment related
- `collection_*` - Data collection related
- `vm_*` - VM setup and management
- `download_*` - Data download methods
**Examples:**
- `deploy_week_collection.sh` - Deploy for week-long collection
- `collection_monitor.sh` - Monitor collection status
- `vm_setup.sh` - Setup VM environment
- `download_data_compressed.sh` - Download with compression
---
## Common Workflows
### Initial Deployment
```bash
# 1. Preflight check
bash scripts/deploy_preflight.sh
# 2. Deploy to GCP
bash scripts/deploy_week_collection.sh
# 3. Monitor
bash scripts/collection_monitor.sh
```
### Download Data
```bash
# Fast compressed download (recommended)
bash scripts/download_data_compressed.sh
# Verify
ls -lh data/downloads/
```
### Fix Production Issues
```bash
# Fix cache bug
bash scripts/fix_overpass_cache.sh
# Health check
bash scripts/health_check.sh
# Backfill missing data
python scripts/backfill_overpass_data.py --dry-run
python scripts/backfill_overpass_data.py
```
### Maintenance
```bash
# Cleanup old runs
python scripts/cleanup_runs.py --days 14
# Backup data
bash scripts/backup.sh
# Check health
bash scripts/health_check.sh
```
---
## Environment Variables
Most scripts use these environment variables (from `.env` file):
```bash
# GCP Configuration
GCP_PROJECT_ID=your-project-id
GCP_ZONE=asia-southeast1-b
GCP_INSTANCE_NAME=traffic-collector-v4
# Collection Configuration
COLLECTION_INTERVAL=1800
USE_MOCK_API=true
# Paths
DATA_DIR=data
CACHE_DIR=cache
LOG_DIR=logs
```
---
## Error Handling
All scripts follow these conventions:
1. **Exit codes:**
- 0 = Success
- 1 = Error
- 2 = Invalid arguments
2. **Logging:**
- Info messages to stdout
- Error messages to stderr
- Detailed logs to files
3. **Error messages:**
- Clear description
- Suggested fix
- Related documentation link
---
## Best Practices
1. **Always run preflight check** before deployment
2. **Use dry-run mode** for destructive operations
3. **Monitor logs** after deployment
4. **Download data regularly** for backup
5. **Use compressed download** for speed
6. **Check health** weekly
---
## Troubleshooting
### Script won't run
```bash
# Make executable
chmod +x scripts/script_name.sh
# Check syntax
bash -n scripts/script_name.sh
```
### GCP authentication issues
```bash
# Re-authenticate
gcloud auth login
# Set project
gcloud config set project YOUR_PROJECT_ID
```
### Python script errors
```bash
# Activate environment
conda activate dsp
# Check dependencies
conda env export
```
### Download fails
```bash
# Use compressed method
bash scripts/download_data_compressed.sh
# Check VM status
gcloud compute instances list
```
---
## Additional Resources
- [QUICKREF.md](../QUICKREF.md) - Quick reference commands
- [DEPLOY.md](../../DEPLOY.md) - Complete deployment guide
- [TROUBLESHOOTING_NODES_MISSING.md](../TROUBLESHOOTING_NODES_MISSING.md) - Cache bug fix
- [CHANGELOG.md](../../CHANGELOG.md) - Version history
---
## Support
For issues or questions:
- Email: fxlqthat@gmail.com
- GitHub: [thatlq1812](https://github.com/thatlq1812)
- Phone: +84 33 863 6369
