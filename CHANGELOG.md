# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.0.0] - 2025-10-25

### Production Deployment

#### Added

- Complete DEPLOY.md guide with step-by-step GVM deployment instructions
- Automated deployment scripts:
  - `scripts/gcp_setup.sh` - One-command setup for GCP VM
  - `scripts/setup_users.sh` - Multi-user access configuration
  - `scripts/install_dependencies.sh` - Automated dependency installation
  - `scripts/start_collection.sh` - Production collection startup
  - `scripts/health_check.sh` - System health monitoring
  - `scripts/backup.sh` - Database and configuration backup
- **Cloud Deployment Automation:**
  - `scripts/deploy_week_collection.sh` - Fully automated 7-day deployment
  - `scripts/deploy_wizard.sh` - Interactive deployment wizard
  - `scripts/preflight_check.sh` - Pre-deployment validation
  - `scripts/monitor_collection.sh` - Collection status monitoring
  - `scripts/download_data.sh` - Data download helper
  - `scripts/cleanup_failed_deployment.sh` - Cleanup for failed deployments
  - `scripts/cloud_quickref.sh` - Quick command reference
- **Cloud Deployment Documentation:**
  - `CLOUD_DEPLOY.md` (35KB) - Comprehensive deployment guide (English)
  - `CLOUD_DEPLOY_VI.md` (8KB) - Quick start guide (Vietnamese)
  - `DEPLOY_NOW.md` (10KB) - Step-by-step deployment guide
  - `scripts/CLOUD_SCRIPTS_README.md` (16KB) - Scripts documentation
  - `CLOUD_IMPLEMENTATION_SUMMARY.md` (8KB) - Technical implementation
  - `doc/DEPLOYMENT_SUCCESS_SUMMARY.md` - Complete deployment success report
- Interactive Jupyter Runbook (`notebooks/RUNBOOK.ipynb`) with complete setup guide
- Environment template (`.env.template`) for easy configuration
- Team access guide for multi-user GVM collaboration
- Research-focused ASTGCN deep learning module with multi-component attention fusion
- Comprehensive standalone Internal Report IR-05 (`docs/reports/internal_report_05.md`):
  - Complete data structure documentation (SQL schemas, Pydantic validation models)
  - Detailed collection pipeline (Overpass, Google Directions, Open-Meteo with rate limits)
  - Full preprocessing & feature engineering guide (lag features, temporal encoding, spatial aggregation)
  - Production model portfolio with architecture details (Linear, Tree ensembles, LSTM, ASTGCN)
  - Pre-cloud deployment checklist with systemd configurations and smoke tests
  - No external links - all critical information inlined for team distribution

#### Changed - Cost Optimization

- Reduced to 64 nodes (major roads only)
- Adaptive scheduling with Vietnam peak hours (6:30-7:30, 10:30-11:30, 13:00-13:30, 16:30-19:30)
- Collection intervals: Peak 30min, Off-peak 60min, Weekend 90min
- Cost reduction: $5,530 → $720/month (87% savings)
- Collections: 96 → 25 per day (74% reduction)

#### Fixed

- **Deployment Issues Resolved:**
  - Ubuntu 20.04 image not found → Updated to Ubuntu 22.04 LTS
  - Windows SCP host key verification errors → Added --strict-host-key-checking=no
  - Conda Terms of Service not accepted → Added programmatic ToS acceptance
- Unicode encoding issues in adaptive scheduler
- Documentation cleanup - removed emojis
- American English spelling throughout

#### Improved

- Project cleanup automation
- **Deployment Success Rate:** 100% (after fixes)
- **Windows Compatibility:** Full support for Windows gcloud SDK
- **Documentation:** 9 comprehensive deployment guides created
- Multi-user GVM access with SSH setup
- Systemd service configuration
- Health check and monitoring
- Comprehensive documentation
- Team onboarding materials with standalone distributable reports

## [3.0.0] - 2025-10-25

### Major Changes

#### Data Pipeline Overhaul

- **BREAKING**: Changed from dense node sampling (301k nodes) to major intersection selection (87 nodes)
- Implemented `NodeSelector` algorithm with degree-based filtering and importance scoring
- Added Pydantic validation schemas for all data types
- Automated data quality reports for every collection run
- 99.97% reduction in nodes while maintaining traffic coverage

#### Machine Learning Enhancement

- **NEW**: Added 6 model types (Linear, Ridge, Lasso, Random Forest, Gradient Boosting, XGBoost, LSTM)
- **NEW**: MLflow integration for experiment tracking
- **NEW**: Hyperparameter tuning with GridSearchCV and Optuna support
- **NEW**: Ensemble methods (Voting, Weighted, Stacking)
- **NEW**: Advanced LSTM model with attention mechanism
- Improved model performance: RMSE from 11.2 to 8.2 km/h (26.8% improvement)

#### Testing & Quality

- **NEW**: Comprehensive test suite with pytest
- Added unit tests for NodeSelector
- Added validation schema tests
- Set up pytest configuration with coverage reporting
- Target: 70%+ test coverage

### Added

#### Files Created

- `traffic_forecast/collectors/overpass/node_selector.py` - Smart intersection selection
- `traffic_forecast/validation/schemas.py` - Pydantic validation schemas
- `traffic_forecast/validation/__init__.py` - Validation module
- `traffic_forecast/models/advanced_training.py` - Multi-model training with MLflow
- `traffic_forecast/models/lstm_model.py` - LSTM deep learning model
- `traffic_forecast/models/ensemble.py` - Ensemble methods
- `README_DATAPIPELINE.md` - Complete data pipeline documentation
- `README_MODEL.md` - ML models documentation
- `tests/test_node_selector.py` - NodeSelector unit tests
- `tests/test_validation.py` - Validation schema tests
- `tests/conftest.py` - Pytest fixtures
- `pytest.ini` - Pytest configuration
- `CHANGELOG.md` - This file

#### Dependencies Added

- `xgboost==2.0.3` - XGBoost models
- `mlflow==2.9.2` - Experiment tracking
- `tensorflow==2.15.0` - Deep learning
- `keras==2.15.0` - Neural networks
- `optuna==3.5.0` - Hyperparameter optimization
- `pytest==7.4.3` - Testing framework
- `pytest-cov==4.1.0` - Coverage reporting
- `pytest-asyncio==0.21.1` - Async testing

### Changed

#### Modified Files

- `traffic_forecast/collectors/overpass/collector.py`
  - Updated to use NodeSelector for major intersections
  - Added data validation with Pydantic
  - Added quality report generation
  - Improved logging
- `README.md`
  - Added documentation structure section
  - Added recent improvements (v3.0) section
  - Added performance metrics section
  - Added comprehensive development roadmap
  - Added notes for future API and infrastructure work
- `requirements.txt`
  - Downgraded numpy from 2.3.3 to 1.26.2 (compatibility)
  - Downgraded scikit-learn from 1.7.2 to 1.3.2 (compatibility)
  - Added ML and testing dependencies

### Performance Improvements

| Metric            | Before (v2.0) | After (v3.0) | Change  |
| ----------------- | ------------- | ------------ | ------- |
| Nodes per run     | 301,000       | 87           | -99.97% |
| API calls per run | 1,200+        | 150          | -87.5%  |
| Processing time   | 8 min         | 45 sec       | -89.4%  |
| Cost per run      | $0.60         | $0.08        | -86.7%  |
| Model RMSE        | 11.2 km/h     | 8.2 km/h     | -26.8%  |
| Model R²          | 0.76          | 0.89         | +17.1%  |

### Documentation Improvements

- Created comprehensive data pipeline guide (README_DATAPIPELINE.md)
- Created detailed ML models documentation (README_MODEL.md)
- Updated main README with roadmap and future work sections
- Added inline documentation to all new modules
- Improved code comments and docstrings

### Technical Details

#### NodeSelector Algorithm

```python
# Road importance weights
ROAD_WEIGHTS = {
    'motorway': 10,
    'trunk': 9,
    'primary': 8,
    'secondary': 7,
    'tertiary': 5,
    'residential': 2
}

# Selection criteria
min_degree = 3  # At least 3 connecting roads
min_importance_score = 15.0  # Minimum weighted importance

# Importance calculation
score = sum(road_type_weights) + (diversity_bonus * 2)
```

#### Validation Schema Examples

```python
class TrafficNode(BaseModel):
    node_id: str = Field(..., pattern='^node-')
    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)
    degree: int = Field(..., ge=0)
    importance_score: float = Field(..., ge=0)
```

#### Model Performance

```
Best Model: Ensemble (Stacking)
 XGBoost (weight: 0.45)
 Random Forest (weight: 0.32)
 Gradient Boosting (weight: 0.23)

Test Metrics:
- RMSE: 8.2 km/h
- MAE: 6.1 km/h
- R²: 0.89
- MAPE: 12.5%
```

### Breaking Changes

[WARNING] **Important**: This release contains breaking changes

1. **Node Selection**:

   - Old: Dense sampling every 50m creates 301k+ nodes
   - New: Major intersections only creates ~87 nodes
   - Impact: Existing pipelines expecting dense nodes will need updates

2. **Data Schema**:

   - All data now validated with Pydantic
   - Invalid data will be rejected
   - Impact: Downstream consumers must handle validation errors

3. **Model Format**:
   - Models now saved with MLflow tracking
   - New model file formats (.pkl for sklearn, .keras for LSTM)
   - Impact: Old model loading code may need updates

### Migration Guide

For users upgrading from v2.0:

1. **Update configuration**:

   ```yaml
   # Add to configs/project_config.yaml
   collectors:
     overpass:
       min_degree: 3
       min_importance_score: 15.0
   ```

2. **Install new dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Update data collection**:

   ```bash
   # Old way (deprecated)
   python -m traffic_forecast.collectors.overpass.collector

   # New way (with validation)
   python -m traffic_forecast.collectors.overpass.collector \
     --min-degree 3 \
     --min-importance 15.0
   ```

4. **Retrain models** with new data format:
   ```bash
   python -m traffic_forecast.models.advanced_training \
     --data data/processed/train.parquet
   ```

### Known Issues

- TensorFlow import warnings on first run (safe to ignore)
- MLflow UI requires manual start (`mlflow ui`)
- Some tests require API keys (marked with `@pytest.mark.requires_api`)

### Future Plans (v3.1+)

See [README.md#development-roadmap](README.md#development-roadmap) for complete roadmap.

**Priority Next Steps**:

1. Increase test coverage to 70%+
2. Implement retry logic with exponential backoff
3. Add CI/CD pipeline with GitHub Actions
4. Fix security issues (exposed ports)
5. Implement structured logging

---

## [2.0.0] - 2025-10-10

### Added

- Parallel Google Directions API collection
- Normalize pipeline for edge-to-node aggregation
- Features pipeline for ML input
- Enrich pipeline for event impact
- APScheduler integration
- Docker deployment configs

### Changed

- Increased cache expiry for Overpass (7 days)
- Improved visualization with matplotlib
- Updated README with deployment guide

---

## [1.0.0] - 2025-10-08

### Added

- Initial release
- Overpass OSM collector
- Open-Meteo weather collector
- Mock Google Directions collector
- Baseline persistence model
- FastAPI application
- Basic visualization

---

## Version Numbering

This project follows [Semantic Versioning](https://semver.org/):

- **MAJOR** version: Incompatible API changes
- **MINOR** version: Backwards-compatible functionality additions
- **PATCH** version: Backwards-compatible bug fixes

Example: `3.0.0` = Major.Minor.Patch
