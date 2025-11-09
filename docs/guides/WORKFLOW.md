# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# STMGT Project - Standardized Workflow

**Last Updated:** November 2, 2025  
**Status:** Phase 1 (Model Hardening) Complete

---

## PROJECT OVERVIEW

### Current State (Nov 2, 2025)

- **Primary Model:** STMGT (Spatio-Temporal Multi-Graph Transformer)
- **Latest Benchmark:** MAE 3.05 km/h, R² 0.77, MAPE 22.9% on validation (run `outputs/stmgt_v2_20251101_012257/`)
- **Dataset Source:** Validated processed parquet declared in `configs/project_config.yaml` and registry entries
- **Status:** Phase 1 deliverables finished (validation enforced, registry hardened, analytics tooling refreshed)

### Key Assets

```
Training Script: scripts/training/train_stmgt.py
Registry Schema: traffic_forecast/core/registry.py
Config Loader: traffic_forecast/core/config_loader.py
Dataset Validator: traffic_forecast/data/validators.py
Validated Analytics: scripts/analysis/analyze_augmentation_strategy.py
```

---

## STANDARDIZED WORKFLOWS

### 1. TRAINING WORKFLOW

#### Start New Training

```bash
# Activate environment
conda activate dsp

# Launch training with dataset validation and config loader
conda run -n dsp --no-capture-output \
    python scripts/training/train_stmgt.py \
    --config configs/training_config.json

# Monitor progress (separate terminal)
conda run -n dsp python scripts/monitoring/monitor_training.py
```

#### Monitor GPU Usage

```bash
# Check GPU utilization
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv

# Real-time monitoring
python scripts/monitoring/gpu_monitor.py
```

#### Training Configuration

- **Batch Size:** Defined per config (`configs/training_config.json` by default)
- **Workers:** Auto-resolved with fallback to 0 on Windows laptops
- **Mixed Precision:** Enabled (AMP with GradScaler, TF32 toggled when available)
- **Early Stopping:** Patience 20 epochs (configurable)
- **Optimizer:** AdamW with ReduceLROnPlateau scheduler
- **Dataset Validation:** Enforced before dataloaders instantiate; failures stop the run early

---

### 2. DATA WORKFLOW

#### Combine New Runs

```bash
# Combine multiple collection runs (validation optional)
conda run -n dsp python scripts/data/combine_runs.py \
    --runs-dir data/runs \
    --output-file data/processed/all_runs_combined.parquet \
    --validate
```

#### Augment Data

**Basic Augmentation (23.4x):**

```bash
conda run -n dsp python scripts/data/augment_data_advanced.py \
    --input data/processed/all_runs_combined.parquet \
    --output data/processed/all_runs_augmented.parquet \
    --validate
```

**Extreme Augmentation (48.4x):**

```bash
conda run -n dsp python scripts/data/augment_extreme.py \
    --input data/processed/all_runs_combined.parquet \
    --output data/processed/all_runs_extreme_augmented.parquet
```

#### Validate Augmentation

```bash
conda run -n dsp python scripts/analysis/analyze_augmentation_strategy.py \
    --dataset data/processed/all_runs_combined.parquet
```

---

### 3. ANALYSIS WORKFLOW

#### Training Progress

```bash
# Quick progress check (summaries last run)
conda run -n dsp python scripts/analysis/quick_progress_check.py \
    --runs-root outputs

# Detailed analysis (plots, statistics)
conda run -n dsp python scripts/analysis/analyze_training_results.py \
    --runs-root outputs --run-id stmgt_v2_20251101_012257
```

#### Data Distribution

```bash
conda run -n dsp python scripts/analysis/analyze_data_distribution.py \
    --dataset data/processed/all_runs_combined.parquet \
    --output-dir outputs/data_analysis_results

# Optional: inspect generated PNG/HTML in outputs/data_analysis_results/
```

---

### 4. MODEL EVALUATION

#### Load Best Model

```python
# Prepare input tensors using traffic_forecast.data.stmgt_dataset utilities
batch = next(iter(dataloader))
with torch.no_grad():
    params = model(
        batch['traffic'],
        batch['edge_index'],
        batch['weather'],
        batch['temporal_features'],
    )
means = params['means']
weights = torch.softmax(params['logits'], dim=-1)
pred_mean = torch.sum(means * weights, dim=-1)
```

````

#### Make Predictions

```python
# Prepare input
# x_traffic: [B, N=62, T=12, 1]
# x_weather: [B, T=12, 3]
# edge_index: [2, 144]
# temporal_features: dict

with torch.no_grad():
    pred_params = model(x_traffic, edge_index, x_weather, temporal_features)

# Get mean prediction
means = pred_params['means']  # [B, N, P, K=3]
stds = pred_params['stds']
weights = torch.softmax(pred_params['logits'], dim=-1)

pred_mean = torch.sum(means * weights, dim=-1)  # [B, N, P]
````

---

### 5. MAINTENANCE WORKFLOW

#### Project Cleanup

```bash
conda run -n dsp python scripts/maintenance/cleanup_project.py
```

#### Update Documentation

```bash
# Update relevant markdown files in docs/
# Ensure Maintainer Profile header remains intact and log changes in docs/CHANGELOG.md

# Optional: verify workspace hygiene
conda run -n dsp python scripts/maintenance/verify_cleanup.py
```

---

## BEST PRACTICES

### Training

- Use `train_stmgt.py` (latest stable) with explicit configs
- Monitor GPU usage to avoid underutilization (see monitoring scripts)
- Review validation metrics each epoch via CSV and dashboard widgets
- Persist best model based on validation MAE and confirm artifacts in `outputs/`
- Keep early stopping parameters aligned with dataset size to prevent overfitting

### Data

- Always run augmentation validators before promoting new parquet files
- Inspect correlation and weather-speed relationships after augmentation
- Track augmentation provenance in `outputs/data_analysis_results/`
- Store raw collections (`data/runs/`) separately from processed parquet outputs

### Code

- Follow existing structure in `traffic_forecast/`
- Add docstrings to all functions
- Use type hints where applicable
- Write unit tests for new features

### Git

- Commit frequently with clear messages
- Don't commit model checkpoints (.pt files)
- Don't commit large data files (use .gitignore)
- Update CHANGELOG.md for major changes

---

## PERFORMANCE TARGETS

| Metric        | Target          | Current   | Status    |
| ------------- | --------------- | --------- | --------- |
| MAE           | < 5.0 km/h      | 3.05 km/h | Exceeded  |
| R²            | > 0.45          | 0.769     | Exceeded  |
| MAPE          | < 30%           | 22.98%    | Exceeded  |
| Training Time | < 2h/100 epochs | ~1.5h     | Met       |
| GPU Usage     | > 50%           | Variable  | ⚠ Monitor |

---

## TROUBLESHOOTING

### Common Issues

**Issue: GPU not utilized**

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Monitor during training
nvidia-smi -l 2
```

**Issue: Out of memory**

```python
# Reduce batch size in config
config['batch_size'] = 16  # or 8
```

**Issue: Multiprocessing error on Windows**

```python
# Set num_workers to 0
config['num_workers'] = 0
```

**Issue: Training too slow**

```python
# Enable mixed precision
config['use_amp'] = True

# Increase batch size if VRAM allows
config['batch_size'] = 48
```

---

## Contacts

- **Maintainer:** THAT Le Quang
- **Repository:** https://github.com/thatlq1812/dsp391m_project
- **Documentation Index:** README.md

---

## CHANGELOG

### 2025-11-02

- Phase 1 hardening complete (dataset validation, registry schema, CLI refresh)
- Updated dashboard quick start/reference and root README
- Restored STMGT regression tests (`tests/test_stmgt_utils.py`)

### 2025-11-01

- Completed project cleanup
- Achieved target metrics (MAE=3.05, R²=0.769)
- Organized documentation
- Standardized workflows

### 2025-10-31

- Implemented extreme augmentation (48.4x)
- Created STMGT model
- Validated data distribution

---

_This workflow is continuously updated. Last review: 2025-11-01_
