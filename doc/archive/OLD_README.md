# Documentation Structure
This file describes the organization of project documentation.
**Last Updated**: October 26, 2025 
**Version**: Academic v4.0
---
## Root Level Documentation
### Essential Files
- **[README.md](../README.md)** - Project overview and main entry point
- **[DEPLOY.md](../DEPLOY.md)** - Complete deployment guide for GCP (500+ lines)
- **[CHANGELOG.md](../CHANGELOG.md)** - Version history and changes
### Reference Documentation (in doc/)
- **[QUICKREF.md](QUICKREF.md)** - Quick reference for common tasks
- **[PRODUCTION_SUMMARY.md](PRODUCTION_SUMMARY.md)** - v4.0 deployment summary
- **[DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)** - Deployment validation checklist
- **[CLEANUP_REPORT.md](CLEANUP_REPORT.md)** - Documentation cleanup report
- **[TROUBLESHOOTING_NODES_MISSING.md](TROUBLESHOOTING_NODES_MISSING.md)** - Fix nodes.json missing error
### Interactive Resources
- **[notebooks/RUNBOOK.ipynb](../notebooks/RUNBOOK.ipynb)** - Complete Jupyter notebook runbook
- **[scripts/quick_start.sh](../scripts/quick_start.sh)** - Interactive setup script
---
## Documentation Directory Structure
```
doc/
getting-started/ # Beginner guides
quickstart.md # Quick start guide
configuration.md # Configuration reference
history/ # Development history
progress.md # Development timeline
reference/ # Technical references
ACADEMIC_V4_SUMMARY.md # v4.0 optimization overview
GOOGLE_API_COST_ANALYSIS.md # Cost analysis and optimization
FEATURE_ENGINEERING_GUIDE.md # Feature engineering documentation
TEMPORAL_FEATURES_ANALYSIS.md # Temporal features analysis
TRAFFIC_HISTORY_STORAGE_GUIDE.md # SQLite storage guide
NODE_EXPORT_GUIDE.md # Node export and selection
SCRIPTS_REFERENCE.md # Complete scripts documentation
data_model.md # Database schema
schema_design.md # Validation schemas
reports/ # Generated reports
internal_report_05.md # Internal progress report
archive/ # Historical documentation
DEPLOYMENT_SUCCESS_SUMMARY.md # Oct 25 deployment details
TEAM_ACCESS_GUIDE.md # Team access (deprecated)
```
---
## Documentation Categories
### 1. Getting Started (For New Users)
Start here if you're new to the project:
1. [README.md](../README.md) - Overview
2. [scripts/quick_start.sh](../scripts/quick_start.sh) - Quick setup (run this!)
3. [getting-started/quickstart.md](getting-started/quickstart.md) - Detailed guide
4. [notebooks/RUNBOOK.ipynb](../notebooks/RUNBOOK.ipynb) - Interactive tutorial
### 2. Deployment (For Production)
Deploy to GCP or server:
1. [DEPLOY.md](../DEPLOY.md) - Complete deployment guide
2. [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md) - Validation checklist
3. [PRODUCTION_SUMMARY.md](PRODUCTION_SUMMARY.md) - What changed in v4.0
4. [TROUBLESHOOTING_NODES_MISSING.md](TROUBLESHOOTING_NODES_MISSING.md) - Common issues
5. [reference/SCRIPTS_REFERENCE.md](reference/SCRIPTS_REFERENCE.md) - All scripts explained
### 3. Configuration (For Customization)
Customize the system:
1. [getting-started/configuration.md](getting-started/configuration.md) - Config reference
2. [reference/ACADEMIC_V4_SUMMARY.md](reference/ACADEMIC_V4_SUMMARY.md) - v4.0 settings
3. [.env.template](../.env.template) - Environment variables
### 4. Cost Optimization (For Budgeting)
Understand and control costs:
1. [reference/GOOGLE_API_COST_ANALYSIS.md](reference/GOOGLE_API_COST_ANALYSIS.md) - Complete cost analysis
2. [reference/ACADEMIC_V4_SUMMARY.md](reference/ACADEMIC_V4_SUMMARY.md) - 87% cost reduction guide
### 5. Technical Reference (For Developers)
Deep dive into technical details:
1. [reference/FEATURE_ENGINEERING_GUIDE.md](reference/FEATURE_ENGINEERING_GUIDE.md) - Features
2. [reference/TEMPORAL_FEATURES_ANALYSIS.md](reference/TEMPORAL_FEATURES_ANALYSIS.md) - Time features
3. [reference/TRAFFIC_HISTORY_STORAGE_GUIDE.md](reference/TRAFFIC_HISTORY_STORAGE_GUIDE.md) - Storage
4. [reference/NODE_EXPORT_GUIDE.md](reference/NODE_EXPORT_GUIDE.md) - Node selection
5. [reference/data_model.md](reference/data_model.md) - Database schema
6. [reference/schema_design.md](reference/schema_design.md) - Validation schemas
### 6. Daily Operations (For Maintenance)
Day-to-day tasks:
1. [QUICKREF.md](QUICKREF.md) - Quick command reference
2. [scripts/health_check.sh](../scripts/health_check.sh) - Health check
3. [scripts/cleanup.sh](../scripts/cleanup.sh) - Cleanup
4. [scripts/backup.sh](../scripts/backup.sh) - Backup
---
## Quick Access by Task
### I want to...
**...set up the project for the first time**
→ Run `bash scripts/quick_start.sh` OR read [DEPLOY.md](../DEPLOY.md)
**...understand the cost optimization**
→ Read [reference/GOOGLE_API_COST_ANALYSIS.md](reference/GOOGLE_API_COST_ANALYSIS.md)
**...configure peak hours**
→ Read [getting-started/configuration.md](getting-started/configuration.md)
**...deploy to production**
→ Follow [DEPLOY.md](../DEPLOY.md) and use [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)
**...add team members**
→ Run `sudo bash scripts/setup_users.sh`
**...check system health**
→ Run `bash scripts/health_check.sh`
**...find a specific command**
→ Check [QUICKREF.md](QUICKREF.md)
**...learn about features**
→ Read [reference/FEATURE_ENGINEERING_GUIDE.md](reference/FEATURE_ENGINEERING_GUIDE.md)
**...understand the data model**
→ Read [reference/data_model.md](reference/data_model.md)
---
## Documentation Standards
All documentation follows these guidelines:
1. **American English spelling** (optimize, not optimise)
2. **No emojis** in documentation
3. **Clear headings** with proper hierarchy
4. **Code blocks** with language specification
5. **Links** to related documentation
6. **Last updated** date at bottom
---
## Removed Documentation
The following files were removed in v4.0 cleanup (2025-10-25):
**Duplicates**:
- `doc/reference/DEPLOY.md` (duplicate of root DEPLOY.md)
**Outdated Operations**:
- `doc/operations/deployment.md` (consolidated into DEPLOY.md)
- `doc/operations/gcp_runbook.md` (consolidated into DEPLOY.md)
- `doc/operations/vm_provisioning.md` (consolidated into DEPLOY.md)
**Old Versions**:
- `doc/reference/IMPLEMENTATION_SUMMARY_V31.md` (superseded by v4.0)
- `doc/reference/IMPLEMENTATION_SUMMARY_V4.md` (consolidated)
- `doc/reference/COMPREHENSIVE_ANALYSIS.md` (info in ACADEMIC_V4_SUMMARY.md)
- `doc/reference/UPGRADE_SUMMARY.md` (info in CHANGELOG.md)
- `doc/reference/PROJECT_COMPLETION_REPORT.md` (outdated)
**Consolidated**:
- `doc/reference/README_DATAPIPELINE.md` (info in feature guides)
- `doc/reference/README_MODEL.md` (info in feature guides)
All information from removed files has been consolidated into current documentation.
---
**Maintained by**: Project team 
**Questions**: Check [README.md](../README.md) or [DEPLOY.md](../DEPLOY.md)
