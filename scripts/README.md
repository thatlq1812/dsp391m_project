# Maintainer Profile# Maintainer Profile

**Name:** THAT Le Quang**Name:** THAT Le Quang

- **Role:** AI & DS Major Student- **Role:** AI & DS Major Student

- **GitHub:** [thatlq1812]- **GitHub:** [thatlq1812]

---

# Scripts Directory# Scripts Directory

Essential scripts for training, data processing, analysis, and deployment.Scripts for training, data processing, analysis, and monitoring.

---## Structure

## Directory Structure- **training/** - Training scripts for STMGT v2

- **data/** - Data preparation and augmentation

````- **analysis/** - Data and model analysis

scripts/- **monitoring/** - Training monitoring tools

├── training/           # Model training- **maintenance/** - Project maintenance utilities

├── data/              # Data processing

├── analysis/          # Analysis utilities## Main Scripts

├── deployment/        # Deployment scripts

├── monitoring/        # Monitoring tools### Training

├── run_api.sh         # API launcher

├── stmgt.sh           # CLI wrapper```bash

└── archive/           # Historical scriptsconda run -n dsp --no-capture-output \

```	python scripts/training/train_stmgt.py --config configs/training_config.json

````

---

### Data Augmentation

## Quick Reference

````bash

### Trainingconda run -n dsp python scripts/data/augment_data_advanced.py \

```bash	--input data/processed/all_runs_combined.parquet \

python scripts/training/train_stmgt.py --config configs/train_normalized_v3.json	--output data/processed/all_runs_augmented.parquet \

```	--validate

````

### Data Processing

````bash### Monitoring

python scripts/data/preprocess_runs.py

python scripts/data/augment_safe.py --preset moderate```bash

```conda run -n dsp python scripts/monitoring/monitor_training.py

````

### API Server

```bashSee `docs/WORKFLOW.md` for complete workflows.

python scripts/deployment/start_api.py

# Or: ./scripts/stmgt.sh api start

````

### Analysis
```bash
python scripts/analysis/analyze_training_results.py
python scripts/analysis/compare_augmentation_methods.py
````

---

## Documentation

**See comprehensive guides:**

- [TRAINING_WORKFLOW.md](../TRAINING_WORKFLOW.md) - Complete pipeline
- [docs/TRAINING.md](../docs/TRAINING.md) - Training guide
- [docs/DEPLOYMENT.md](../docs/DEPLOYMENT.md) - Deployment
- [docs/DATA.md](../docs/DATA.md) - Data processing
