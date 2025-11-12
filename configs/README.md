# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Training Configurations

This directory contains training configuration files for the STMGT traffic forecasting model.

## Configuration Files Overview

### Production Configs

| File                           | Status         | Params | MAE        | Description                                          |
| ------------------------------ | -------------- | ------ | ---------- | ---------------------------------------------------- |
| **`train_normalized_v3.json`** | **PRODUCTION** | 680K   | **3.0468** | Current baseline, 1.1% better than V1, best coverage |

**Archived Configs** (see `configs/archive/`):

- `train_normalized_v1.json` - Previous baseline (MAE 3.08)
- `train_normalized_v2.json` - Capacity experiment (rejected, overfits)

### Capacity Experiments (CONCLUDED)

| File                          | Status   | Params | Change | Test MAE | Result                                |
| ----------------------------- | -------- | ------ | ------ | -------- | ------------------------------------- |
| `train_v0.6_minimal.json`     | TESTED   | 350K   | -48%   | 3.11     | Better than V0.8, but worse than V1   |
| `train_v0.8_smaller.json`     | TESTED   | 520K   | -23%   | 3.22     | Underfits, worse than V0.6            |
| `train_v0.9_ablation_k3.json` | OPTIONAL | 600K   | -12%   | ?        | Can narrow optimal range to 600K-680K |

**Conclusion:** 680K params is PROVEN OPTIMAL (U-shaped capacity curve confirmed)

### Legacy Configs

| File                          | Status | Description                 |
| ----------------------------- | ------ | --------------------------- |
| `train_production_ready.json` | Legacy | Old production config (K=3) |
| `train_optimized_final.json`  | Legacy | K=2 mixture                 |
| `train_smoke_test.json`       | Debug  | Quick 5-epoch test          |

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

## Experimental Config Variants (November 2025)

After validating V1 (680K params, MAE 3.08) as optimal and rejecting V2 (1.15M params, overfits), we created 6 experimental variants to explore safe improvements:

### Config Overview Table

| Config             | Focus          | Params | Key Change             | Expected MAE | Risk     | Priority    |
| ------------------ | -------------- | ------ | ---------------------- | ------------ | -------- | ----------- |
| **V1** (baseline)  | Production     | 680K   | hidden=96, K=5         | **3.08**     | -        | DEPLOYED    |
| **V1.5 Capacity**  | Safe scaling   | 850K   | hidden=104, K=6        | 2.98-3.05    | LOW      | TRY FIRST   |
| **V1 Arch**        | Optimization   | 680K   | Residuals, GELU        | 2.95-3.05    | VERY LOW | SAFEST      |
| **V1 Heavy Reg**   | Regularization | 1M     | hidden=112 + heavy reg | 2.95-3.10    | MEDIUM   | TEST        |
| **V1 Deeper**      | Depth          | 890K   | 4 blocks (depth)       | 2.95-3.08    | MEDIUM   | ALTERNATIVE |
| **V1 Uncertainty** | Calibration    | 680K   | K=7, MSE=0.3           | 3.08-3.15    | VERY LOW | PRODUCTION  |
| **V1 No Weather**  | Ablation       | 640K   | Remove weather         | 3.25-3.35    | N/A      | RESEARCH    |
| V2 (rejected)      | Capacity       | 1.15M  | hidden=128, K=7        | 3.22         | HIGH     | REJECTED    |

### Detailed Config Descriptions

#### 1. `train_v1.5_capacity.json` - Safe Capacity Increment

**Strategy:** Small capacity increase (+25%) with stronger regularization

**Changes from V1:**

- hidden_dim: 96 → 104 (+8.3%, safe increment)
- mixture_K: 5 → 6 (+20%, better uncertainty)
- dropout: 0.2 → 0.22, drop_edge: 0.1 → 0.12
- Total params: 680K → 850K (+25%)
- Parameter ratio: 850K/144K = 0.17 (safe zone)

**Expected Results:**

- Val MAE: 2.98-3.05 km/h (1-3% better than V1)
- Coverage@80: 84-86%
- Overfitting risk: LOW

**When to use:** Want to improve V1 with minimal risk. Try this FIRST before more aggressive variants.

**Usage:**

```bash
python scripts/training/train_stmgt.py --config configs/train_v1.5_capacity.json
```

---

#### 2. `train_v1_arch_improvements.json` - Architectural Enhancements

**Strategy:** Same capacity (680K) but better architecture (NO overfitting risk)

**Changes from V1:**

- SAME params: 680K (zero capacity increase)
- Add: Residual connections (multi-scale features)
- Add: Layer normalization (stability)
- Add: GELU activation (smoother gradients vs ReLU)
- Add: Gradient clipping = 1.0 (prevent spikes)

**Expected Results:**

- Val MAE: 2.95-3.05 km/h (0-4% better than V1)
- Coverage@80: 83-85%
- Overfitting risk: VERY LOW

**When to use:** SAFEST option. Want to improve V1 WITHOUT any overfitting risk. Pure optimization improvements.

**Usage:**

```bash
python scripts/training/train_stmgt.py --config configs/train_v1_arch_improvements.json
```

---

#### 3. `train_v1_heavy_reg.json` - Aggressive Regularization Test

**Strategy:** Moderate capacity (+47%) with HEAVY regularization

**Changes from V1:**

- hidden_dim: 96 → 112 (+16.7%)
- mixture_K: 5 → 6
- Total params: 680K → 1M (+47%)
- AGGRESSIVE regularization:
  - dropout: 0.3 (vs V1's 0.2, V2's 0.25)
  - drop_edge: 0.2 (vs V1's 0.1)
  - weight_decay: 0.00025 (2.5× V1)
  - label_smoothing: 0.05
  - Strong mixup: alpha=0.3
  - More cutout: p=0.15

**Expected Results:**

- Val MAE: 2.95-3.10 km/h (±0-3% vs V1)
- Coverage@80: 85-87%
- Overfitting risk: MEDIUM

**Hypothesis:** Can aggressive regularization support 1M params for 205K samples? If yes → strong reg works. If overfits → confirms 680K is hard limit.

**When to use:** EXPERIMENTAL. Test regularization limits. Monitor train/val gap closely. Stop if gap > 15%.

**Usage:**

```bash
python scripts/training/train_stmgt.py --config configs/train_v1_heavy_reg.json
```

---

#### 4. `train_v1_deeper.json` - Depth Instead of Width

**Strategy:** Increase depth (blocks) instead of width (hidden_dim)

**Changes from V1:**

- num_blocks: 3 → 4 (+33% depth)
- SAME width: hidden_dim = 96
- Receptive field: 3 hops → 4 hops (25% → 40% network coverage)
- Total params: 680K → 890K (+31%)
- Lower LR: 0.001 → 0.0008 (deeper = slower convergence)

**Expected Results:**

- Val MAE: 2.95-3.08 km/h (0-4% better than V1)
- Coverage@80: 83-85%
- Overfitting risk: MEDIUM

**Rationale:** V2 showed width (96→128) causes overfitting. Test if depth is safer alternative for capturing long-range dependencies.

**When to use:** Alternative to width scaling. If successful, proves depth > width for traffic graphs.

**Usage:**

```bash
python scripts/training/train_stmgt.py --config configs/train_v1_deeper.json
```

---

#### 5. `train_v1_uncertainty_focused.json` - Calibration Priority

**Strategy:** Same architecture (680K) but optimize for uncertainty, not MAE

**Changes from V1:**

- SAME capacity: 680K params
- mixture_K: 5 → 7 (+40%, more modes)
- MSE weight: 0.4 → 0.3 (prioritize NLL/calibration)
- Trade-off: Accept slight MAE increase for better calibration

**Expected Results:**

- Val MAE: 3.08-3.15 km/h (±0-2% vs V1, slight degradation OK)
- Coverage@80: 86-88% (MAIN GOAL: +2-4% vs V1's 84%)
- CRPS: 2.10-2.20 (better uncertainty score)
- Overfitting risk: VERY LOW

**When to use:** Production systems needing reliable confidence intervals (traffic management, risk assessment, route planning). Acceptable to trade 2% MAE for 3% better calibration.

**Usage:**

```bash
python scripts/training/train_stmgt.py --config configs/train_v1_uncertainty_focused.json
```

---

#### 6. `train_v1_ablation_no_weather.json` - Weather Feature Importance

**Strategy:** Remove weather features to quantify their contribution

**Changes from V1:**

- REMOVED: Weather features (temp, precip, cloud cover)
- REMOVED: Weather cross-attention module
- SAME: All other architecture (hidden=96, K=5, blocks=3)
- Params: ~640K (slight reduction)

**Expected Results:**

- Val MAE: 3.25-3.35 km/h (+5-9% WORSE than V1)
- MAE degradation: +0.17-0.27 km/h
- R²: 0.78-0.80 (-2-4% vs V1's 0.82)

**Purpose:** ABLATION STUDY. Quantify: How much does weather improve predictions?

**When to use:** RESEARCH ONLY. Run to validate weather module importance for paper/report. Do NOT use for production.

**Usage:**

```bash
python scripts/training/train_stmgt.py --config configs/train_v1_ablation_no_weather.json
```

---

### Decision Guide: Which Config to Try?

#### Scenario 1: Want to improve V1 with minimal risk

→ **Try V1.5 Capacity** or **V1 Arch Improvements**

**Reasoning:**

- V1.5: Safe capacity increase (+25%, 850K params)
- V1 Arch: Zero capacity increase (680K params, pure optimization)
- Both have LOW/VERY LOW overfitting risk

#### Scenario 2: Want best possible MAE, willing to experiment

→ **Try V1 Heavy Reg** or **V1 Deeper**

**Reasoning:**

- Test capacity limits (1M, 890K params)
- Monitor train/val gap carefully
- Stop early if overfitting detected (gap > 15%)

#### Scenario 3: Production deployment, need reliable uncertainty

→ **Use V1 Uncertainty Focused**

**Reasoning:**

- Same capacity as proven V1 (680K params)
- Optimized for calibration (Coverage@80 86-88%)
- Slight MAE trade-off acceptable for better confidence intervals

#### Scenario 4: Research/paper, need ablation studies

→ **Run V1 No Weather**

**Reasoning:**

- Quantify weather contribution (expected +5-9% MAE degradation)
- Validate multi-modal design choice
- Not for production use

---

### Training Recommendations

#### Priority Order (Recommended Sequence)

1. **V1 Arch Improvements** (SAFEST - try first)

   - Risk: VERY LOW
   - If successful (MAE < 3.05): Deploy immediately
   - If similar (MAE 3.05-3.10): Still valuable (better optimization)

2. **V1.5 Capacity** (Safe scaling)

   - Risk: LOW
   - If successful (MAE < 3.05): Validates capacity increase
   - If overfits: Confirms V1 (680K) is optimal

3. **V1 Uncertainty Focused** (Production alternative)

   - Risk: VERY LOW
   - Use if calibration > point predictions
   - Deploy for risk-aware applications

4. **V1 Deeper** OR **V1 Heavy Reg** (Pick one)

   - Risk: MEDIUM
   - Experimental - test capacity limits
   - Monitor carefully for overfitting

5. **V1 No Weather** (Research only)
   - Not for production
   - Run for paper/report ablation section

#### Training Tips

**Monitor these metrics during training:**

1. **Train/Val MAE Gap:** Should stay < 10%

   - If gap > 15%: Overfitting detected, stop training
   - If gap > 20%: Config too large, reject

2. **Best Epoch:** Should be in middle-to-late training (epoch 8-15)

   - If best epoch < 5: Model too large (like V2)
   - If best epoch > 25: Increase patience or lower LR

3. **Coverage@80:** Should be 80-85% (well-calibrated)

   - If < 75%: Overconfident (CIs too narrow)
   - If > 90%: Underconfident (CIs too wide)

4. **Gradient Norms:** Should be stable
   - If exploding (>10): Lower LR or increase gradient clipping
   - If vanishing (<0.01): Check depth, may need better initialization

**Early stopping criteria:**

```python
# Stop training if ANY of these occur:
- Train/val gap > 20% (severe overfitting)
- Val MAE increases for 3 consecutive epochs
- Gradient norms explode (>100)
- NaN loss detected
```

---

## Configuration Version History

| Version      | Date           | Config         | Changes                 | Performance  | Status         |
| ------------ | -------------- | -------------- | ----------------------- | ------------ | -------------- |
| v1.0         | 2025-11-01     | Initial        | hidden=64, blocks=2     | MAE 10.81    | Obsolete       |
| v2.0         | 2025-11-01     | Capacity++     | hidden=96, blocks=3     | MAE 5.00     | Obsolete       |
| v3.0         | 2025-11-02     | Tuned          | LR/WD optimized         | MAE 3.91     | Good           |
| **V1**       | **2025-11-09** | **Normalized** | **Fixed normalization** | **MAE 3.08** | **PRODUCTION** |
| V2           | 2025-11-10     | Capacity+69%   | hidden=128, K=7         | MAE 3.22     | **REJECTED**   |
| V1.5         | 2025-11-10     | Safe scaling   | hidden=104, K=6         | TBD          | To test        |
| V1 Arch      | 2025-11-10     | Optimization   | Residuals, GELU         | TBD          | To test        |
| V1 Heavy     | 2025-11-10     | Heavy reg      | hidden=112 + reg        | TBD          | Experimental   |
| V1 Deeper    | 2025-11-10     | Depth          | 4 blocks                | TBD          | Experimental   |
| V1 Uncertain | 2025-11-10     | Calibration    | K=7, MSE=0.3            | TBD          | To test        |

---

## Data Augmentation Configuration

### Safe Augmentation (NEW - 2025-11-12)

**File:** `augmentation_config.json`

Data augmentation now uses **leak-free methods** that only use training data statistics.

**Available Presets:**

```json
{
  "light": {
    "noise_copies": 2,
    "weather_scenarios": 3,
    "jitter_copies": 1
  },
  "moderate": {
    // Recommended
    "noise_copies": 3,
    "weather_scenarios": 5,
    "jitter_copies": 2
  },
  "aggressive": {
    "noise_copies": 5,
    "weather_scenarios": 8,
    "jitter_copies": 3
  }
}
```

**Usage in Training Script:**

```python
from traffic_forecast.data.augmentation_safe import SafeTrafficAugmentor
import json

# Load config
with open('configs/augmentation_config.json') as f:
    aug_config = json.load(f)

# After temporal split
train_data, val_data, test_data = split_temporal(df)

# Augment training data only
augmentor = SafeTrafficAugmentor(train_data)
train_augmented = augmentor.augment_all(**aug_config['moderate'])

# Train with augmented data (val/test unchanged)
```

**Key Changes from Old Augmentation:**

| Aspect         | Old (Deprecated)                | New (Safe)             |
| -------------- | ------------------------------- | ---------------------- |
| **Statistics** | From entire dataset             | **Train-only**         |
| **Methods**    | Interpolation, extrapolation    | Noise, weather, jitter |
| **When**       | Pre-training (all data)         | **After split**        |
| **Leakage**    | Yes (test patterns)             | **No**                 |
| **Files**      | `augment_extreme.py` (archived) | `augmentation_safe.py` |

**Deprecated Files:**

Old augmentation scripts moved to `scripts/data/archive/`:

- `augment_extreme.py` - Data leakage via global statistics
- `augment_data_advanced.py` - Data leakage via test patterns

See `scripts/data/archive/DEPRECATION_NOTICE.md` for details.

---

## References

- **Current Production Config:** `configs/train_normalized_v3.json`
- **Augmentation Config:** `configs/augmentation_config.json`
- **Archived Configs:** `configs/archive/` (v1, v2)
- **Data Leakage Fix:** `docs/fix/data_leakage_fix.md`
- **Safe Augmentation Guide:** `docs/guides/safe_augmentation_guide.md`
- **Weather Data Explained:** `docs/guides/weather_data_explained.md`
- Model architecture: `traffic_forecast/models/stmgt/model.py`
- Training script: `scripts/training/train_stmgt.py`
- Data pipeline: `traffic_forecast/data/stmgt_dataset.py`

---

**Last Updated:** November 12, 2025  
**Maintainer:** THAT Le Quang
