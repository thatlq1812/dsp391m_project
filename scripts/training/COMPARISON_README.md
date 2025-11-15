# Fair Model Comparison for Final Report

This directory contains scripts to train all models with identical conditions for fair comparison.

---

## Quick Start

```bash
# Train all 3 models with identical conditions
cd /d/UNI/DSP391m/project
bash scripts/training/train_all_for_comparison.sh
```

**Time:** ~90 minutes for all models

---

## What This Does

Trains 3 models with **identical conditions:**

1. **STMGT V3** - Multi-modal spatial-temporal (current best)
2. **LSTM Baseline** - Temporal only (no spatial info)
3. **GraphWaveNet** - Adaptive graph + temporal

**Identical Conditions:**

- Dataset: `data/processed/all_runs_gapfilled_week.parquet`
- Split: 70% train / 15% val / 15% test (temporal)
- Epochs: 100 with early stopping
- Evaluation: Same metrics (MAE, RMSE, R², MAPE, CRPS)

---

## Output Structure

```
outputs/final_comparison/run_YYYYMMDD_HHMMSS/
├── config.txt                      # Training configuration
├── comparison_report.json          # Summary comparison
├── lstm_baseline/
│   └── run_*/
│       ├── results.json
│       ├── training_history.csv
│       └── model files...
├── graphwavenet/
│   └── run_*/
└── stmgt_v3/
    ├── best_model.pt
    ├── test_results.json
    └── ...
```

---

## Individual Model Training

If you want to train models separately:

```bash
# LSTM Baseline
python scripts/training/train_lstm_baseline.py \
    --dataset data/processed/all_runs_gapfilled_week.parquet \
    --epochs 100

# GraphWaveNet
python scripts/training/train_graphwavenet_baseline.py \
    --dataset data/processed/all_runs_gapfilled_week.parquet \
    --epochs 100

# STMGT V3
python scripts/training/train_stmgt.py \
    --config configs/train_normalized_v3.json
```

---

## Comparison Report

After training, run:

```bash
python scripts/training/compare_models.py \
    --comparison-dir outputs/final_comparison/run_YYYYMMDD_HHMMSS \
    --output outputs/final_comparison/run_YYYYMMDD_HHMMSS/comparison_report.json
```

**Report includes:**

- Performance table (MAE, RMSE, R², MAPE)
- Improvements over baselines
- Best model identification
- Full configuration details

---

## Expected Results

Based on previous training runs:

| Model        | MAE (km/h) | R²        | Training Time |
| ------------ | ---------- | --------- | ------------- |
| **STMGT V3** | **~3.08**  | **~0.82** | ~15 min       |
| GraphWaveNet | ~3.95      | ~0.71     | ~30 min       |
| LSTM         | ~4.85      | ~0.64     | ~20 min       |

---

## For Final Report

Use results from `comparison_report.json` to update:

1. **Section 08: Evaluation** - Update performance table
2. **Section 09: Results** - Add comparison figures
3. **Generate figures:**
   ```bash
   python scripts/visualization/04_results_figures.py
   ```

---

## Troubleshooting

**Issue:** Script fails on specific model

**Solution:** Comment out that model in `train_all_for_comparison.sh` and train others

**Issue:** Out of memory

**Solution:** Reduce batch size in script (change `BATCH_SIZE=32` to `BATCH_SIZE=16`)

**Issue:** Training takes too long

**Solution:** Reduce epochs (change `EPOCHS=100` to `EPOCHS=50`)

---

**Maintainer:** THAT Le Quang (thatlq1812)  
**Created:** November 12, 2025
