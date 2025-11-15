# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# STMGT Training Workflow

Complete workflow for training STMGT model from data collection to evaluation with leak-free augmentation.

**Last Updated:** November 12, 2024

**Status:** Active - Uses SafeTrafficAugmentor (no data leakage)

---

## Overview

This guide provides step-by-step instructions for the complete training pipeline after data collection is complete. The workflow ensures no data leakage by using only training set statistics for augmentation.

### Key Principles

1. **Temporal Split First:** Always split data before augmentation
2. **Train-Only Statistics:** Augmentation uses only training set statistics
3. **No Future Information:** No test set information leaks into training
4. **Weather is NOT Leakage:** Weather is exogenous (available via forecasts)

---

## Prerequisites

### Environment Setup

```bash
# Activate conda environment
conda activate dsp391m

# Verify Python packages
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
```

### Required Files

- ✓ Raw data in `data/runs/` (JSON format from data collection)
- ✓ Graph topology in `cache/overpass_topology.json`
- ✓ Configuration files in `configs/`

### Check Data Collection

```bash
# List collected runs
ls -lh data/runs/

# Expected: run_YYYYMMDD_HHMMSS/ directories with JSON files:
#   - traffic_edges.json (traffic data)
#   - nodes.json (node information)
#   - weather_snapshot.json (weather data)
#   - edges.json (edge information)
#   - statistics.json (run statistics)
```

---

## Step 1: Data Preprocessing

**Purpose:** Convert raw JSON files to Parquet format with validation and cleaning.

### Run Preprocessing

```bash
# Preprocess all runs
python scripts/data/preprocess_runs.py

# Or preprocess specific runs
python scripts/data/preprocess_runs.py --runs run_20251030_032457 run_20251030_032440

# Force reprocess all (ignore cache)
python scripts/data/preprocess_runs.py --force
```

### What Happens

1. **Load raw JSON:** Reads JSON files from each run directory
   - `traffic_edges.json` - Traffic speed data
   - `nodes.json` - Node information (or generated from traffic)
   - `weather_snapshot.json` - Weather data (or uses defaults)
2. **Data cleaning:**
   - Parse timestamps to datetime
   - Add temporal features (hour, minute, day_of_week, is_weekend)
   - Add congestion levels and speed categories
   - Merge traffic with weather data
   - Remove duplicates
   - Validate speed values (non-negative, no NaN)
3. **Combine runs:** Merge all runs into single dataframe
4. **Save Parquet:** Write to `data/processed/all_runs_combined.parquet`
5. **Generate metadata:** Create `metadata.json` and `summary.json`

### Verify Preprocessing

```bash
# Validate processed dataset
python scripts/data/validate_processed_dataset.py

# Check output files
ls -lh data/processed/
# Expected: all_runs_combined.parquet, metadata.json, summary.json
```

### Output Files

```
data/processed/
├── all_runs_combined.parquet  # Combined traffic data
├── metadata.json               # Dataset metadata (columns, dtypes)
└── summary.json                # Summary statistics
```

---

## Step 2: Training Without Augmentation (Baseline)

**Purpose:** Establish baseline performance before applying augmentation.

### Configure Training

Edit `configs/train_normalized_v3.json`:

```json
{
  "data": {
    "data_path": "data/processed/all_runs_combined.parquet",
    "graph_path": "cache/overpass_topology.json",
    "seq_len": 12,
    "pred_len": 12,
    "train_ratio": 0.7,
    "val_ratio": 0.15
  },
  "training": {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "use_amp": true,
    "early_stopping_patience": 15
  }
}
```

### Run Training

```bash
# Train baseline model
python scripts/training/train_stmgt.py --config configs/train_normalized_v3.json

# Training will:
# 1. Load data from data/processed/all_runs_combined.parquet
# 2. Split into train/val/test (70/15/15 by runs)
# 3. Normalize using TRAIN-ONLY statistics
# 4. Train STMGT model
# 5. Save checkpoints to outputs/stmgt_v2_TIMESTAMP/
```

### Monitor Training

```bash
# Watch training progress (in another terminal)
tail -f outputs/stmgt_v2_*/training.log

# Check TensorBoard (if configured)
tensorboard --logdir outputs/stmgt_v2_*/tensorboard
```

### Expected Baseline Performance

Based on recent runs:

- **Train MAE:** ~2.8-3.0 km/h
- **Val MAE:** ~3.0-3.2 km/h
- **Test MAE:** ~3.0-3.5 km/h

### Output Files

```
outputs/stmgt_v2_TIMESTAMP/
├── checkpoints/
│   ├── best_model.pt          # Best validation checkpoint
│   └── last_model.pt          # Last epoch checkpoint
├── config.json                # Training configuration
├── training_history.csv       # Epoch-by-epoch metrics
├── evaluation_results.json    # Final test set evaluation
└── training.log               # Training logs
```

---

## Step 3: Data Augmentation (Leak-Free)

**Purpose:** Generate augmented training data using ONLY training set statistics.

### Understand SafeTrafficAugmentor

The new augmentation system (`traffic_forecast/data/augmentation_safe.py`) ensures no data leakage:

1. **Train-Only Statistics:** Learns patterns from training set only
2. **No Test Information:** Never sees validation or test data
3. **Temporal Integrity:** Maintains temporal ordering
4. **Exogenous Variables:** Weather augmentation uses training ranges (NOT leakage)

### Augmentation Methods

**Available methods:**

- `augment_noise_injection()` - Add Gaussian noise (train std)
- `augment_weather_scenarios()` - Synthetic weather conditions (train ranges)
- `augment_temporal_jitter()` - Small time shifts (train patterns)

### Configure Augmentation

Edit `configs/augmentation_config.json`:

```json
{
  "augmentation_presets": {
    "light": {
      "noise_copies": 2,
      "weather_scenarios": 3,
      "jitter_copies": 1,
      "noise_scale": 0.05,
      "jitter_max_minutes": 5
    },
    "moderate": {
      "noise_copies": 3,
      "weather_scenarios": 5,
      "jitter_copies": 2,
      "noise_scale": 0.1,
      "jitter_max_minutes": 10
    },
    "aggressive": {
      "noise_copies": 5,
      "weather_scenarios": 8,
      "jitter_copies": 3,
      "noise_scale": 0.15,
      "jitter_max_minutes": 15
    }
  },
  "default_preset": "moderate"
}
```

### Create Augmentation Script

Create `scripts/data/augment_safe.py`:

```python
"""
Safe data augmentation script using SafeTrafficAugmentor.

Usage:
    python scripts/data/augment_safe.py --preset moderate
    python scripts/data/augment_safe.py --preset aggressive --output data/processed/augmented.parquet
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from traffic_forecast.data.augmentation_safe import SafeTrafficAugmentor


def main():
    parser = argparse.ArgumentParser(description="Augment traffic data safely")
    parser.add_argument(
        "--preset",
        type=str,
        default="moderate",
        choices=["light", "moderate", "aggressive"],
        help="Augmentation preset to use"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default="data/processed/all_runs_combined.parquet",
        help="Input parquet file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="data/processed/all_runs_augmented.parquet",
        help="Output parquet file"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio for split"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio for split"
    )
    args = parser.parse_args()

    # Load augmentation config
    config_path = PROJECT_ROOT / "configs" / "augmentation_config.json"
    with open(config_path) as f:
        config = json.load(f)

    preset = config["augmentation_presets"][args.preset]
    print(f"\nUsing preset: {args.preset}")
    print(f"Configuration: {preset}")

    # Load data
    print(f"\nLoading data from {args.input}...")
    df = pd.read_parquet(args.input)
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"Loaded {len(df)} samples")

    # CRITICAL: Split data first (temporal split by runs)
    print("\nSplitting data (temporal split by runs)...")
    runs = df['run_id'].unique()
    n_train = int(len(runs) * args.train_ratio)
    n_val = int(len(runs) * args.val_ratio)

    train_runs = runs[:n_train]
    val_runs = runs[n_train:n_train+n_val]
    test_runs = runs[n_train+n_val:]

    df_train = df[df['run_id'].isin(train_runs)].copy()
    df_val = df[df['run_id'].isin(val_runs)].copy()
    df_test = df[df['run_id'].isin(test_runs)].copy()

    print(f"Train: {len(df_train)} samples ({len(train_runs)} runs)")
    print(f"Val: {len(df_val)} samples ({len(val_runs)} runs)")
    print(f"Test: {len(df_test)} samples ({len(test_runs)} runs)")

    # Initialize augmentor with TRAIN data only
    print("\nInitializing SafeTrafficAugmentor with training data only...")
    augmentor = SafeTrafficAugmentor(df_train)

    # Augment training data
    print("\nAugmenting training data...")
    df_train_augmented = augmentor.augment_all(
        noise_copies=preset["noise_copies"],
        weather_scenarios=preset["weather_scenarios"],
        jitter_copies=preset["jitter_copies"]
    )
    print(f"Augmented training data: {len(df_train_augmented)} samples")

    # Validate no leakage
    print("\nValidating no data leakage...")
    augmentor.validate_no_leakage(df_train_augmented, df_val, df_test)

    # Combine: augmented train + original val + original test
    print("\nCombining datasets...")
    df_combined = pd.concat([
        df_train_augmented,
        df_val,
        df_test
    ], ignore_index=True)

    print(f"Combined dataset: {len(df_combined)} samples")
    print(f"  Train (augmented): {len(df_train_augmented)}")
    print(f"  Val (original): {len(df_val)}")
    print(f"  Test (original): {len(df_test)}")

    # Save augmented dataset
    print(f"\nSaving to {args.output}...")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_parquet(args.output, index=False)
    print("Done!")

    # Print augmentation summary
    print("\n" + "="*60)
    print("AUGMENTATION SUMMARY")
    print("="*60)
    print(f"Preset: {args.preset}")
    print(f"Original train size: {len(df_train)}")
    print(f"Augmented train size: {len(df_train_augmented)}")
    print(f"Augmentation factor: {len(df_train_augmented) / len(df_train):.2f}x")
    print(f"Val/Test: Unchanged (no augmentation)")
    print(f"Output: {args.output}")
    print("="*60)


if __name__ == "__main__":
    main()
```

### Run Augmentation

```bash
# Create the augmentation script first
# (copy the script above to scripts/data/augment_safe.py)

# Run with moderate preset (recommended)
python scripts/data/augment_safe.py --preset moderate

# Or use aggressive preset for more data
python scripts/data/augment_safe.py --preset aggressive

# Or light preset for minimal augmentation
python scripts/data/augment_safe.py --preset light
```

### Verify Augmentation

The script will:

1. Load original data
2. Split into train/val/test (70/15/15)
3. Learn statistics from **training set only**
4. Augment training set (val/test unchanged)
5. Validate no leakage
6. Save to `data/processed/all_runs_augmented.parquet`

**Expected output:**

```
Original train size: ~7000 samples
Augmented train size: ~35000-70000 samples (depending on preset)
Augmentation factor: 5-10x
```

---

## Step 4: Training With Augmentation

**Purpose:** Train model on augmented data to improve generalization.

### Update Config

Edit `configs/train_normalized_v3.json`:

```json
{
  "data": {
    "data_path": "data/processed/all_runs_augmented.parquet", // Changed
    "graph_path": "cache/overpass_topology.json",
    "seq_len": 12,
    "pred_len": 12,
    "train_ratio": 0.7,
    "val_ratio": 0.15
  },
  "training": {
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "use_amp": true,
    "early_stopping_patience": 15
  }
}
```

### Run Training

```bash
# Train with augmented data
python scripts/training/train_stmgt.py --config configs/train_normalized_v3.json
```

### Expected Performance Improvement

With proper augmentation:

- **Train MAE:** May increase slightly (more diverse data)
- **Val MAE:** Should decrease (better generalization)
- **Test MAE:** Target improvement 5-10%

**Example:**

- Baseline: Test MAE 3.20
- Augmented: Test MAE 2.90-3.05 (improvement)

---

## Step 5: Evaluation and Comparison

**Purpose:** Compare baseline vs augmented model performance.

### Evaluate Models

```bash
# Baseline model
python scripts/training/train_stmgt.py --config configs/train_normalized_v3.json --evaluate-only --checkpoint outputs/stmgt_v2_BASELINE_TIMESTAMP/checkpoints/best_model.pt

# Augmented model
python scripts/training/train_stmgt.py --config configs/train_normalized_v3.json --evaluate-only --checkpoint outputs/stmgt_v2_AUGMENTED_TIMESTAMP/checkpoints/best_model.pt
```

### Compare Results

Check `evaluation_results.json` in each output directory:

```json
{
  "test_mae": 3.05,
  "test_rmse": 4.12,
  "test_mape": 8.5,
  "train_mae": 2.88,
  "val_mae": 3.02
}
```

### Create Comparison Report

```bash
# Compare two runs
python scripts/analysis/compare_training_runs.py \
    --run1 outputs/stmgt_v2_BASELINE_TIMESTAMP \
    --run2 outputs/stmgt_v2_AUGMENTED_TIMESTAMP \
    --output comparison_report.md
```

---

## Workflow Summary

```
1. Data Collection (manual)
   └─> data/runs/run_YYYYMMDD_HHMMSS/
       ├─ traffic_edges.json
       ├─ nodes.json
       ├─ weather_snapshot.json
       ├─ edges.json
       └─ statistics.json

2. Preprocessing
   └─> python scripts/data/preprocess_runs.py
       └─> data/processed/all_runs_combined.parquet

3. Baseline Training (no augmentation)
   └─> python scripts/training/train_stmgt.py --config configs/train_normalized_v3.json
       └─> outputs/stmgt_v2_TIMESTAMP/ (baseline checkpoint)

4. Data Augmentation (leak-free)
   └─> python scripts/data/augment_safe.py --preset moderate
       └─> data/processed/all_runs_augmented.parquet

5. Training with Augmentation
   └─> python scripts/training/train_stmgt.py --config configs/train_normalized_v3.json
       └─> outputs/stmgt_v2_TIMESTAMP/ (augmented checkpoint)

6. Evaluation & Comparison
   └─> Compare baseline vs augmented performance
```

---

## Complete Command Sequence

```bash
# Step 1: Preprocess data
python scripts/data/preprocess_runs.py
python scripts/data/validate_processed_dataset.py

# Step 2: Train baseline
python scripts/training/train_stmgt.py --config configs/train_normalized_v3.json

# Step 3: Create augmentation script (if not exists)
# Copy the augment_safe.py script to scripts/data/

# Step 4: Run augmentation
python scripts/data/augment_safe.py --preset moderate

# Step 5: Update config to use augmented data
# Edit configs/train_normalized_v3.json: change data_path to all_runs_augmented.parquet

# Step 6: Train with augmentation
python scripts/training/train_stmgt.py --config configs/train_normalized_v3.json

# Step 7: Compare results
# Check evaluation_results.json in both output directories
```

---

## Troubleshooting

### Issue: "No data found in data/runs/"

**Solution:** Run data collection first or verify data paths.

### Issue: "KeyError: temperature_c"

**Solution:** Preprocessing should fill missing weather columns. Run validation:

```bash
python scripts/data/validate_processed_dataset.py
```

### Issue: "Augmentation increases training loss"

**Solution:** Normal behavior. Augmented data is more diverse. Check validation loss instead.

### Issue: "Out of memory during training"

**Solution:** Reduce batch size in config:

```json
{
  "training": {
    "batch_size": 16 // Reduced from 32
  }
}
```

### Issue: "Model not improving with augmentation"

**Possible causes:**

1. Too aggressive augmentation (try "light" preset)
2. Not enough epochs (augmented data needs more training)
3. Learning rate too high (reduce to 0.0005)

---

## Data Leakage Prevention Checklist

Before training, verify:

- ✓ Split data BEFORE augmentation
- ✓ Augmentor initialized with train data only
- ✓ Validation set NOT augmented
- ✓ Test set NOT augmented
- ✓ Normalization uses train-only statistics
- ✓ No future information in temporal features
- ✓ Weather data from forecasts (NOT historical actuals from test period)

**How to verify:**

```python
# In augmentation script, validate_no_leakage() checks:
# 1. No test timestamps in augmented train
# 2. No test node patterns leaked
# 3. Statistical distributions reasonable
```

---

## Next Steps

After successful training:

1. **Model Deployment:** Export best checkpoint for API
2. **API Integration:** Update `traffic_api/` to use new model
3. **Monitoring:** Track real-world performance
4. **Iteration:** Collect more data, retrain with augmentation

---

## Related Documentation

- [AUGMENTATION_MIGRATION_GUIDE.md](docs/guides/AUGMENTATION_MIGRATION_GUIDE.md) - Detailed augmentation documentation
- [data_leakage_fix.md](docs/fix/data_leakage_fix.md) - Data leakage analysis and fix
- [weather_data_explained.md](docs/guides/weather_data_explained.md) - Why weather is NOT leakage
- [CLEANUP_SUMMARY.md](docs/CLEANUP_SUMMARY.md) - Recent project cleanup details

---

## Questions or Issues

If you encounter issues:

1. Check preprocessing logs: `python scripts/data/validate_processed_dataset.py`
2. Verify augmentation: Check `validate_no_leakage()` output
3. Review training logs: `outputs/stmgt_v2_*/training.log`
4. Compare with baseline: Use comparison scripts

For data leakage concerns, refer to [data_leakage_fix.md](docs/fix/data_leakage_fix.md).
