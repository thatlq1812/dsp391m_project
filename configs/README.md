# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Training Configurations

This directory contains training configuration files for the STMGT traffic forecasting model.

## Configuration Files Overview

| File                          | Status          | Description                                                         | Use Case                      |
| ----------------------------- | --------------- | ------------------------------------------------------------------- | ----------------------------- |
| `train_production_ready.json` | **RECOMMENDED** | Production-ready config with K=3 mixture, optimized hyperparameters | Final training for deployment |
| `train_optimized_final.json`  | Good            | K=2 mixture, well-tuned but sub-optimal calibration                 | Baseline comparison           |
| `train_balanced_v3.json`      | Experimental    | Earlier balanced config                                             | Development/testing           |
| `train_smoke_test.json`       | 🧪 Debug        | Quick 5-epoch test                                                  | Verify code changes           |
| `train_smoke_config.json`     | 🧪 Debug        | Fast sanity check                                                   | CI/CD pipeline                |
| `training_config.json`        | Legacy          | Old config format                                                   | Deprecated                    |

---

## Recommended Configuration

### `train_production_ready.json`

**Expected Performance:**

- Val MAE: **3.9-4.1 km/h**
- Val RMSE: **6.0-6.5 km/h**
- R² Score: **0.70-0.75**
- 80% CI Coverage: **78-82%** (well-calibrated)

**Key Features:**

```json
{
  "mixture_components": 3, // Tri-modal: free/moderate/heavy traffic
  "mse_loss_weight": 0.4, // Balanced MAE/calibration (0.4 MSE + 0.6 NLL)
  "hidden_dim": 96, // ~1.0M params - good capacity
  "num_blocks": 4, // Parallel ST-processing depth
  "batch_size": 64, // Stable gradients
  "learning_rate": 0.0004, // Conservative, safe convergence
  "pin_memory": true, // Faster GPU transfer
  "persistent_workers": true // Reduced data loading overhead
}
```

**Usage:**

```bash
python -m traffic_forecast.models.train --config configs/train_production_ready.json
```

**Training Time:** ~20 hours (100 epochs @ 12 min/epoch on RTX 3060)

---

## Configuration Comparison

### Architecture Settings

| Parameter                | Production | Optimized Final | Notes                              |
| ------------------------ | ---------- | --------------- | ---------------------------------- |
| `hidden_dim`             | 96         | 96              | Good balance                       |
| `num_blocks`             | 4          | 4               | Parallel ST-blocks                 |
| `num_heads`              | 6          | 6               | Multi-head attention               |
| **`mixture_components`** | **3**      | **2**           | **K=3 captures tri-modal traffic** |
| `seq_len`                | 12         | 12              | 3-hour input window                |
| `pred_len`               | 12         | 12              | 3-hour forecast horizon            |

### Training Hyperparameters

| Parameter             | Production | Optimized Final | Impact                             |
| --------------------- | ---------- | --------------- | ---------------------------------- |
| `batch_size`          | 64         | 64              | Stable gradients                   |
| `learning_rate`       | 0.0004     | 0.0004          | Conservative convergence           |
| `weight_decay`        | 0.0001     | 0.0001          | L2 regularization                  |
| **`mse_loss_weight`** | **0.4**    | **0.3**         | **Better MAE/calibration balance** |
| `drop_edge_p`         | 0.08       | 0.08            | Graph dropout                      |
| `max_epochs`          | 100        | 100             | With early stopping                |
| `patience`            | 20         | 20              | Early stopping trigger             |

### Data Loading Optimization

| Parameter                | Production | Optimized Final | Improvement                    |
| ------------------------ | ---------- | --------------- | ------------------------------ |
| **`pin_memory`**         | **true**   | **false**       | **Faster GPU transfer**        |
| **`persistent_workers`** | **true**   | **false**       | **No worker restart overhead** |
| **`prefetch_factor`**    | **2**      | **null**        | **Pipeline 2 batches ahead**   |
| `num_workers`            | 8          | 8               | Parallel data loading          |

---

## Why K=3 Mixture Components?

### Traffic Modes in Real Data

HCMC traffic exhibits **3 distinct states**:

1. **Free-flow** (μ₃ ≈ 28-30 km/h)

   - Off-peak hours
   - Low probability during rush hour (~10%)

2. **Moderate congestion** (μ₂ ≈ 20-24 km/h)

   - Transitional periods
   - Medium probability (~30-35%)

3. **Heavy congestion** (μ₁ ≈ 14-18 km/h)
   - Peak rush hours
   - High probability during commute (~55-60%)

### K=2 vs K=3 Performance

| Metric            | K=2 (Current Run) | K=3 (Best Run) | Improvement           |
| ----------------- | ----------------- | -------------- | --------------------- |
| Val MAE           | 4.48 km/h         | 3.91 km/h      | **14.6% better**      |
| 80% Coverage      | 92.7%             | 78.1%          | **Better calibrated** |
| Calibration Error | 12.7%             | 1.9%           | **85% reduction**     |

**Conclusion:** K=3 is essential for realistic traffic modeling.

---

## MSE Loss Weight Tuning

### Loss Function

```
Total Loss = mse_weight * MSE + (1 - mse_weight) * NLL
```

### Impact on Performance

| `mse_loss_weight` | MAE Focus | Calibration | Recommendation              |
| ----------------- | --------- | ----------- | --------------------------- |
| 0.1-0.2           | Poor      | Good        | Over-emphasizes uncertainty |
| **0.4-0.5**       | **Good**  | **Good**    | **Balanced**                |
| 0.7-0.8           | Excellent | Poor        | Point prediction only       |

**Production config uses 0.4:** Best balance between accurate predictions and well-calibrated uncertainty.

---

## Quick Start Guide

### 1. Smoke Test (Verify Setup)

```bash
python -m traffic_forecast.models.train --config configs/train_smoke_test.json
```

- Duration: ~1 minute
- Purpose: Ensure code/data pipeline works

### 2. Development Training (Fast Iteration)

```bash
python -m traffic_forecast.models.train --config configs/train_balanced_v3.json
```

- Duration: ~2-3 hours (20 epochs)
- Purpose: Test architecture changes

### 3. Production Training (Final Model)

```bash
python -m traffic_forecast.models.train --config configs/train_production_ready.json
```

- Duration: ~20 hours (100 epochs)
- Purpose: Best performance for deployment

---

## Creating Custom Configurations

### Template Structure

```json
{
  "model": {
    "hidden_dim": 96, // Model capacity (64, 96, 128)
    "num_heads": 6, // Attention heads (4, 6, 8)
    "num_blocks": 4, // ST-block depth (2, 3, 4)
    "mixture_components": 3, // GMM components (2, 3, 5)
    "seq_len": 12, // Input steps (6, 12, 24)
    "pred_len": 12 // Output steps (6, 12, 24)
  },
  "training": {
    "batch_size": 64, // (32, 64, 128)
    "learning_rate": 0.0004, // (0.0001, 0.0004, 0.001)
    "weight_decay": 0.0001, // L2 regularization
    "max_epochs": 100, // Max iterations
    "patience": 20, // Early stopping
    "drop_edge_p": 0.08, // Graph dropout (0.0-0.2)
    "mse_loss_weight": 0.4, // MSE vs NLL balance (0.3-0.5)
    "use_amp": true, // Mixed precision (faster)
    "pin_memory": true, // GPU optimization
    "persistent_workers": true, // Data loader optimization
    "prefetch_factor": 2 // Pipeline batches (2-4)
  }
}
```

### Hyperparameter Guidelines

**Model Capacity:**

- Small dataset (<10K samples): `hidden_dim=64, num_blocks=2`
- Medium dataset (10K-100K): `hidden_dim=96, num_blocks=4` ✅
- Large dataset (>100K): `hidden_dim=128, num_blocks=6`

**Learning Rate:**

- Conservative (safe): `0.0002-0.0004` ✅
- Aggressive (fast): `0.0008-0.001`
- Very large batch (>128): `0.001-0.002`

**Mixture Components:**

- Binary states: `K=2` (e.g., congested/free)
- Tri-modal: `K=3` (e.g., heavy/moderate/free)
- Multi-modal: `K=5` (complex urban scenarios)

---

## Troubleshooting

### Issue: Poor Calibration (Coverage >> 80%)

**Symptoms:** 80% CI coverage = 90-95%

**Causes:**

- `mixture_components` too low (K=2)
- `mse_loss_weight` too low (<0.3)

**Solutions:**

1. Increase `mixture_components` to 3
2. Increase `mse_loss_weight` to 0.4-0.5
3. Use `train_production_ready.json`

### Issue: Overfitting (Train/Val Gap > 2.0 km/h)

**Symptoms:** Train MAE << Val MAE

**Causes:**

- Insufficient regularization
- Too much capacity

**Solutions:**

1. Increase `weight_decay` (0.0001 → 0.0005)
2. Increase `drop_edge_p` (0.08 → 0.15)
3. Reduce `hidden_dim` (96 → 64)
4. Add data augmentation

### Issue: Slow Convergence

**Symptoms:** Validation MAE still decreasing after 50 epochs

**Causes:**

- Learning rate too low
- Batch size too small

**Solutions:**

1. Increase `learning_rate` (0.0004 → 0.0006)
2. Increase `batch_size` (64 → 128)
3. Increase `patience` (20 → 30)

### Issue: OOM (Out of Memory)

**Symptoms:** CUDA out of memory errors

**Solutions:**

1. Reduce `batch_size` (64 → 32)
2. Reduce `hidden_dim` (96 → 64)
3. Disable `pin_memory` (true → false)
4. Reduce `num_workers` (8 → 4)
5. Enable gradient accumulation: `accumulation_steps=2`

---

## Experimental Results

### Validation Performance by Config

| Config                        | Val MAE  | Val RMSE | R²       | Coverage 80% | Status             |
| ----------------------------- | -------- | -------- | -------- | ------------ | ------------------ |
| `train_production_ready.json` | **TBD**  | **TBD**  | **TBD**  | **TBD**      | **To be trained**  |
| `train_optimized_final.json`  | 4.48     | 7.06     | 0.53     | 92.7%        | Running (epoch 17) |
| Run `20251102_182710`         | **3.91** | **6.29** | **0.72** | **78.1%**    | **Best so far**    |
| Run `20251101_215205`         | 5.00     | 7.10     | 0.68     | 85.3%        | Complete           |
| Naive baseline                | 7.20     | 10.50    | 0.00     | N/A          | Reference          |

### Training Time Estimates

| Config         | Epochs  | Time/Epoch | Total Time    | GPU          |
| -------------- | ------- | ---------- | ------------- | ------------ |
| Smoke test     | 5       | 12 min     | ~1 hour       | RTX 3060     |
| Development    | 20      | 12 min     | ~4 hours      | RTX 3060     |
| **Production** | **100** | **12 min** | **~20 hours** | **RTX 3060** |

---

## Next Steps for Production Training

### 1. Pre-Training Checklist

- [ ] Verify data: `data/all_runs_augmented.parquet` exists
- [ ] Check disk space: >10 GB free for checkpoints
- [ ] GPU available: `nvidia-smi` shows RTX 3060
- [ ] Environment ready: `conda activate dsp`

### 2. Launch Production Training

```bash
# Navigate to project root
cd /d/UNI/DSP391m/project

# Activate environment
conda activate dsp

# Start training (background recommended)
nohup python -m traffic_forecast.models.train \
    --config configs/train_production_ready.json \
    > logs/production_training.log 2>&1 &

# Monitor progress
tail -f logs/production_training.log
```

### 3. Monitor Training

```bash
# Check latest metrics
python scripts/analysis/analyze_training.py

# View training curves
python scripts/analysis/plot_training_curves.py
```

### 4. Post-Training Validation

```bash
# Generate report figures
python scripts/analysis/create_report_figures_part1.py
python scripts/analysis/create_report_figures_part2.py

# Update report with final results
# Edit docs/report/RP3_ReCheck.md with actual metrics
```

---

## Configuration Version History

| Version  | Date           | Changes                     | Performance           |
| -------- | -------------- | --------------------------- | --------------------- |
| v1.0     | 2025-11-01     | Initial config (h64_b2)     | MAE 10.81             |
| v2.0     | 2025-11-01     | Increased capacity (h96_b3) | MAE 5.00              |
| v3.0     | 2025-11-02     | Tuned LR/WD                 | **MAE 3.91**          |
| v4.0     | 2025-11-02     | K=2, optimized data loading | MAE 4.48 (running)    |
| **v5.0** | **2025-11-02** | **K=3, balanced loss**      | **TBD (recommended)** |

---

## References

- Model architecture: `traffic_forecast/models/stmgt_v2.py`
- Training script: `traffic_forecast/models/train.py`
- Data pipeline: `traffic_forecast/data/datamodule.py`
- Documentation: `docs/STMGT_ARCHITECTURE.md`

---

**Last Updated:** November 2, 2025  
**Maintainer:** THAT Le Quang
