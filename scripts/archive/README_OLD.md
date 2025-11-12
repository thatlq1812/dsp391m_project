# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Scripts Directory

Scripts for training, data processing, analysis, and monitoring.

## Structure

- **training/** - Training scripts for STMGT v2
- **data/** - Data preparation and augmentation
- **analysis/** - Data and model analysis
- **monitoring/** - Training monitoring tools
- **maintenance/** - Project maintenance utilities

## Main Scripts

### Training

```bash
conda run -n dsp --no-capture-output \
	python scripts/training/train_stmgt.py --config configs/training_config.json
```

### Data Augmentation

```bash
conda run -n dsp python scripts/data/augment_data_advanced.py \
	--input data/processed/all_runs_combined.parquet \
	--output data/processed/all_runs_augmented.parquet \
	--validate
```

### Monitoring

```bash
conda run -n dsp python scripts/monitoring/monitor_training.py
```

See `docs/WORKFLOW.md` for complete workflows.
