# Maintainer Profile# Documentation Structure

**Full name:** THAT Le Quang This file describes the organization of project documentation.

**Nickname:** Xiel

**Last Updated**: October 26, 2025

- **Role:** AI & DS Major Student**Version**: Academic v4.0

- **GitHub:** [thatlq1812](https://github.com/thatlq1812)

- **Primary email:** fxlqthat@gmail.com---

- **Academic email:** thatlqse183256@fpt.edu.com

- **Alternate email:** thatlq1812@gmail.com## Root Level Documentation

- **Phone (VN):** +84 33 863 6369 / +84 39 730 6450

### Essential Files

---

- **[README.md](../README.md)** - Project overview and main entry point

# Documentation Hub- **[DEPLOY.md](../DEPLOY.md)** - Complete deployment guide for GCP (500+ lines)

- **[CHANGELOG.md](../CHANGELOG.md)** - Version history and changes

Complete documentation for the Traffic Forecast System project.

### Reference Documentation (in doc/)

**Last Updated**: October 27, 2025

**Version**: Academic v4.0- **[QUICKREF.md](QUICKREF.md)** - Quick reference for common tasks

- **[PRODUCTION_SUMMARY.md](PRODUCTION_SUMMARY.md)** - v4.0 deployment summary

## Quick Start- **[DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)** - Deployment validation checklist

- **[CLEANUP_REPORT.md](CLEANUP_REPORT.md)** - Documentation cleanup report

### Essential Commands- **[TROUBLESHOOTING_NODES_MISSING.md](TROUBLESHOOTING_NODES_MISSING.md)** - Fix nodes.json missing error

#### Environment Setup### Interactive Resources

```bash- **[notebooks/RUNBOOK.ipynb](../notebooks/RUNBOOK.ipynb)** - Complete Jupyter notebook runbook

# Activate conda environment- **[scripts/quick_start.sh](../scripts/quick_start.sh)** - Interactive setup script

conda activate dsp

---

# Update environment

conda env update -f environment.yml## Documentation Directory Structure

```

````

#### Data Collectiondoc/

├── getting-started/          # Beginner guides

```bash│   ├── quickstart.md        # Quick start guide

# Single collection (local)│   └── configuration.md     # Configuration reference

conda run -n dsp python scripts/collection/collect_and_render.py --once│

├── history/                  # Development history

# Start collection loop (15 minutes)│   └── progress.md          # Development timeline

conda run -n dsp python scripts/collection/collect_and_render.py --interval 900│

├── reference/               # Technical references

# View schedule│   ├── ACADEMIC_V4_SUMMARY.md              # v4.0 optimization overview

conda run -n dsp python scripts/collection/collect_and_render.py --print-schedule│   ├── GOOGLE_API_COST_ANALYSIS.md         # Cost analysis and optimization

```│   ├── FEATURE_ENGINEERING_GUIDE.md        # Feature engineering documentation

│   ├── TEMPORAL_FEATURES_ANALYSIS.md       # Temporal features analysis

#### Data Management│   ├── TRAFFIC_HISTORY_STORAGE_GUIDE.md    # SQLite storage guide

│   ├── NODE_EXPORT_GUIDE.md                # Node export and selection

```bash│   ├── SCRIPTS_REFERENCE.md                # Complete scripts documentation

# Download latest data from GCP (recommended)│   ├── data_model.md                       # Database schema

bash scripts/data_management/download_data_compressed.sh│   └── schema_design.md                    # Validation schemas

│

# Cleanup old runs (keep last 14 days)├── reports/                 # Generated reports

conda run -n dsp python scripts/data_management/cleanup_runs.py --days 14│   └── internal_report_05.md               # Internal progress report

│

# Backup data└── archive/                 # Historical documentation

bash scripts/data_management/backup.sh    ├── DEPLOYMENT_SUCCESS_SUMMARY.md       # Oct 25 deployment details

```    └── TEAM_ACCESS_GUIDE.md                # Team access (deprecated)

````

#### Monitoring

---

```bash

# Check system health## Documentation Categories

bash scripts/monitoring/health_check.sh

### 1. Getting Started (For New Users)

# Start live dashboard

conda run -n dsp python scripts/monitoring/live_dashboard.pyStart here if you're new to the project:

```

1. [README.md](../README.md) - Overview

### Configuration2. [scripts/quick_start.sh](../scripts/quick_start.sh) - Quick setup (run this!)

3. [getting-started/quickstart.md](getting-started/quickstart.md) - Detailed guide

**Main config**: `configs/project_config.yaml`4. [notebooks/RUNBOOK.ipynb](../notebooks/RUNBOOK.ipynb) - Interactive tutorial

Key settings:### 2. Deployment (For Production)

- `google_directions.use_mock_api`: true = FREE, false = PAID

- `scheduler.mode`: adaptive or fixedDeploy to GCP or server:

- `node_selection.max_nodes`: number of monitoring nodes

1. [DEPLOY.md](../DEPLOY.md) - Complete deployment guide

## Documentation Structure2. [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) - Validation checklist

3. [PRODUCTION_SUMMARY.md](PRODUCTION_SUMMARY.md) - What changed in v4.0

### 📖 Getting Started4. [TROUBLESHOOTING_NODES_MISSING.md](TROUBLESHOOTING_NODES_MISSING.md) - Common issues

5. [reference/SCRIPTS_REFERENCE.md](reference/SCRIPTS_REFERENCE.md) - All scripts explained

New to the project? Start here:

### 3. Configuration (For Customization)

- **[getting-started/quickstart.md](getting-started/quickstart.md)** - Quick start guide

- **[getting-started/configuration.md](getting-started/configuration.md)** - Configuration referenceCustomize the system:

- **[../notebooks/RUNBOOK.ipynb](../notebooks/RUNBOOK.ipynb)** - Interactive Jupyter runbook

1. [getting-started/configuration.md](getting-started/configuration.md) - Config reference

### 📚 Reference Documentation2. [reference/ACADEMIC_V4_SUMMARY.md](reference/ACADEMIC_V4_SUMMARY.md) - v4.0 settings

3. [.env.template](../.env.template) - Environment variables

Technical specifications and guides:

### 4. Cost Optimization (For Budgeting)

- **[reference/ACADEMIC_V4_SUMMARY.md](reference/ACADEMIC_V4_SUMMARY.md)** - v4.0 optimization overview

- **[reference/GOOGLE_API_COST_ANALYSIS.md](reference/GOOGLE_API_COST_ANALYSIS.md)** - Cost analysis and optimizationUnderstand and control costs:

- **[reference/FEATURE_ENGINEERING_GUIDE.md](reference/FEATURE_ENGINEERING_GUIDE.md)** - Feature engineering documentation

- **[reference/TEMPORAL_FEATURES_ANALYSIS.md](reference/TEMPORAL_FEATURES_ANALYSIS.md)** - Temporal features analysis1. [reference/GOOGLE_API_COST_ANALYSIS.md](reference/GOOGLE_API_COST_ANALYSIS.md) - Complete cost analysis

- **[reference/TRAFFIC_HISTORY_STORAGE_GUIDE.md](reference/TRAFFIC_HISTORY_STORAGE_GUIDE.md)** - SQLite storage guide2. [reference/ACADEMIC_V4_SUMMARY.md](reference/ACADEMIC_V4_SUMMARY.md) - 87% cost reduction guide

- **[reference/NODE_EXPORT_GUIDE.md](reference/NODE_EXPORT_GUIDE.md)** - Node export and selection

- **[reference/SCRIPTS_REFERENCE.md](reference/SCRIPTS_REFERENCE.md)** - Scripts documentation### 5. Technical Reference (For Developers)

- **[reference/data_model.md](reference/data_model.md)** - Data model specification

- **[reference/schema_design.md](reference/schema_design.md)** - Schema designDeep dive into technical details:

### 📊 Reports1. [reference/FEATURE_ENGINEERING_GUIDE.md](reference/FEATURE_ENGINEERING_GUIDE.md) - Features

2. [reference/TEMPORAL_FEATURES_ANALYSIS.md](reference/TEMPORAL_FEATURES_ANALYSIS.md) - Time features

Analysis reports and findings:3. [reference/TRAFFIC_HISTORY_STORAGE_GUIDE.md](reference/TRAFFIC_HISTORY_STORAGE_GUIDE.md) - Storage

4. [reference/NODE_EXPORT_GUIDE.md](reference/NODE_EXPORT_GUIDE.md) - Node selection

- **[reports/internal_report_05.md](reports/internal_report_05.md)** - Latest internal report5. [reference/data_model.md](reference/data_model.md) - Database schema

6. [reference/schema_design.md](reference/schema_design.md) - Validation schemas

### 📜 History

### 6. Daily Operations (For Maintenance)

Development timeline and progress:

Day-to-day tasks:

- **[history/progress.md](history/progress.md)** - Development timeline

- **[../CHANGELOG.md](../CHANGELOG.md)** - Version history and changes1. [QUICKREF.md](QUICKREF.md) - Quick command reference

2. [scripts/health_check.sh](../scripts/health_check.sh) - Health check

### 🗄️ Archive3. [scripts/cleanup.sh](../scripts/cleanup.sh) - Cleanup

4. [scripts/backup.sh](../scripts/backup.sh) - Backup

Historical documentation (for reference only):

---

- **[archive/ARCHIVE_README.md](archive/ARCHIVE_README.md)** - Archive overview

- **[archive/DEPLOYMENT_SUCCESS_SUMMARY.md](archive/DEPLOYMENT_SUCCESS_SUMMARY.md)** - v4.0 deployment summary## Quick Access by Task

- **[archive/TEAM_ACCESS_GUIDE.md](archive/TEAM_ACCESS_GUIDE.md)** - Team access setup

- **[archive/QUICKREF.md](archive/QUICKREF.md)** - Legacy quick reference (merged into this doc)### I want to...

- **[archive/OLD_README.md](archive/OLD_README.md)** - Previous README version

**...set up the project for the first time**

## Project Root Documentation→ Run `bash scripts/quick_start.sh` OR read [DEPLOY.md](../DEPLOY.md)

Essential files at project root:**...understand the cost optimization**

→ Read [reference/GOOGLE_API_COST_ANALYSIS.md](reference/GOOGLE_API_COST_ANALYSIS.md)

- **[../README.md](../README.md)** - Project overview and main entry point

- **[../CHANGELOG.md](../CHANGELOG.md)** - Version history and changes**...configure peak hours**

- **[../PROJECT_SPEC.yaml](../PROJECT_SPEC.yaml)** - Project specification→ Read [getting-started/configuration.md](getting-started/configuration.md)

## Scripts Organization**...deploy to production**

→ Follow [DEPLOY.md](../DEPLOY.md) and use [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)

All operational scripts are organized in `../scripts/`:

**...add team members**

- **deployment/** - Deployment and setup scripts→ Run `sudo bash scripts/setup_users.sh`

- **data_management/** - Download, backup, cleanup scripts

- **collection/** - Data collection scripts**...check system health**

- **monitoring/** - Monitoring and dashboard scripts→ Run `bash scripts/health_check.sh`

- **utilities/** - Maintenance and utility scripts

- **deprecated/** - Old scripts (for reference)**...find a specific command**

→ Check [QUICKREF.md](QUICKREF.md)

See **[../scripts/README.md](../scripts/README.md)** for detailed scripts documentation.

**...learn about features**

## Cost Information→ Read [reference/FEATURE_ENGINEERING_GUIDE.md](reference/FEATURE_ENGINEERING_GUIDE.md)

### Academic v4.0 (Current Configuration)**...understand the data model**

→ Read [reference/data_model.md](reference/data_model.md)

- **Nodes**: 64 monitoring points

- **Collections/day**: ~25 (adaptive schedule)---

- **Monthly cost**: $720 (real API) or $0 (mock API)

## Documentation Standards

### Toggle Mock API

All documentation follows these guidelines:

Edit `configs/project_config.yaml`:

1. **American English spelling** (optimize, not optimise)

````yaml2. **No emojis** in documentation

google_directions:3. **Clear headings** with proper hierarchy

  use_mock_api: true  # true = FREE, false = PAID4. **Code blocks** with language specification

```5. **Links** to related documentation

6. **Last updated** date at bottom

## Troubleshooting

---

### Common Issues

## Removed Documentation

**Import Errors**

```bashThe following files were removed in v4.0 cleanup (2025-10-25):

conda activate dsp

pip install -r requirements.txt --upgrade**Duplicates**:

````

- `doc/reference/DEPLOY.md` (duplicate of root DEPLOY.md)

**Permission Errors**

````bash**Outdated Operations**:

chmod +x scripts/**/*.sh

chmod 755 scripts/**/*.py- `doc/operations/deployment.md` (consolidated into DEPLOY.md)

```- `doc/operations/gcp_runbook.md` (consolidated into DEPLOY.md)

- `doc/operations/vm_provisioning.md` (consolidated into DEPLOY.md)

**Database Locked**

```bash**Old Versions**:

pkill -f collect_and_render.py

rm traffic_history.db-journal- `doc/reference/IMPLEMENTATION_SUMMARY_V31.md` (superseded by v4.0)

```- `doc/reference/IMPLEMENTATION_SUMMARY_V4.md` (consolidated)

- `doc/reference/COMPREHENSIVE_ANALYSIS.md` (info in ACADEMIC_V4_SUMMARY.md)

**API Errors**- `doc/reference/UPGRADE_SUMMARY.md` (info in CHANGELOG.md)

Switch to mock API in `configs/project_config.yaml`- `doc/reference/PROJECT_COMPLETION_REPORT.md` (outdated)



### Check System Health**Consolidated**:



```bash- `doc/reference/README_DATAPIPELINE.md` (info in feature guides)

# Local- `doc/reference/README_MODEL.md` (info in feature guides)

bash scripts/monitoring/health_check.sh

All information from removed files has been consolidated into current documentation.

# View logs

tail -f logs/service.log---

````

**Maintained by**: Project team

## Team Collaboration**Questions**: Check [README.md](../README.md) or [DEPLOY.md](../DEPLOY.md)

### GCP VM Access

```bash
# SSH to VM
gcloud compute ssh traffic-collector-v4 --zone=asia-southeast1-b

# Activate environment
conda activate dsp

# Check status
bash scripts/monitoring/health_check.sh
```

## Support

For issues or questions:

- Check logs: `tail -f logs/service.log`
- Run health check: `bash scripts/monitoring/health_check.sh`
- Review documentation in this folder
- Contact maintainer (see profile above)

---

**Project**: Traffic Forecast System  
**Version**: Academic v4.0  
**Last Updated**: October 27, 2025
