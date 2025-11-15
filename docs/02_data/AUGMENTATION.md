# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Migration Guide: Old to Safe Augmentation

**Date:** 2025-11-12  
**Version:** 2.0  
**Status:** Production Ready

## Overview

This guide helps you migrate from the old augmentation system (with data leakage) to the new safe augmentation system.

## What Changed?

### Old System (Deprecated)

```
Scripts: augment_extreme.py, augment_data_advanced.py
Config: Old augmentation_config.json (basic/extreme)
When: Pre-training (augment entire dataset)
Statistics: From ALL data (including test)
Problem: Data leakage via test set patterns
```

### New System (Safe)

```
Script: traffic_forecast/data/augmentation_safe.py
Config: augmentation_config.json (light/moderate/aggressive)
When: After temporal split (augment train only)
Statistics: From TRAIN data only
Benefit: No data leakage
```

---

## Migration Steps

### Step 1: Understand What Was Wrong

**Old approach created data leakage:**

```python
# OLD (WRONG):
# 1. Load entire dataset
df = pd.read_parquet('all_runs_combined.parquet')

# 2. Compute patterns from ALL data (including test)
hourly_patterns = df.groupby('hour')['speed'].mean()  # ← Uses test data

# 3. Create augmented dataset
augmented = apply_patterns(df, hourly_patterns)
augmented.to_parquet('all_runs_augmented.parquet')

# 4. Split for training
train, val, test = split_temporal(augmented)  # ← Train contains test patterns
```

**Why this is bad:**

- Model learns patterns that include test set statistics
- Artificially inflates performance
- Doesn't generalize to truly unseen data

### Step 2: Update Your Training Script

**New approach (CORRECT):**

```python
# NEW (CORRECT):
from traffic_forecast.data.augmentation_safe import SafeTrafficAugmentor
import json

# 1. Load dataset
df = pd.read_parquet('all_runs_combined.parquet')

# 2. Split FIRST
train_df, val_df, test_df = split_temporal(df)

# 3. Load augmentation config
with open('configs/augmentation_config.json') as f:
    aug_config = json.load(f)

# 4. Initialize with TRAIN data only
augmentor = SafeTrafficAugmentor(
    train_df=train_df,
    random_seed=42
)

# 5. Augment train only
train_augmented = augmentor.augment_all(
    **aug_config['moderate']  # or 'light', 'aggressive'
)

# 6. Create datasets
train_dataset = STMGTDataset(train_augmented, ...)
val_dataset = STMGTDataset(val_df, ...)      # No augmentation
test_dataset = STMGTDataset(test_df, ...)    # No augmentation
```

### Step 3: Update Configuration Files

**Old config format (deprecated):**

```json
{
  "basic": {
    "noise_std_speed": 2.0,
    "interpolation_steps": 2
  }
}
```

**New config format:**

```json
{
  "moderate": {
    "noise_copies": 3,
    "noise_level": 0.05,
    "weather_scenarios": 5,
    "jitter_copies": 2,
    "jitter_max_minutes": 15,
    "include_original": true
  }
}
```

### Step 4: Remove Old Augmented Datasets

**Clean up old files:**

```bash
# Remove datasets with data leakage
rm data/processed/all_runs_augmented.parquet
rm data/processed/all_runs_extreme_augmented.parquet

# Use original data for new augmentation
# Augmentation now happens in training script
```

### Step 5: Verify No Leakage

**Add validation in your training script:**

```python
from traffic_forecast.data.augmentation_safe import validate_no_leakage

# After augmentation
leakage_free = validate_no_leakage(
    train_augmented,
    val_df,
    test_df
)

if not leakage_free:
    raise RuntimeError("Data leakage detected!")
```

---

## Configuration Presets

### Light Augmentation (Fast Experiments)

```python
augmentor.augment_all(**aug_config['light'])
```

- Augmentation factor: ~2x
- Training time: Minimal increase
- Use for: Quick experiments, debugging

### Moderate Augmentation (Recommended)

```python
augmentor.augment_all(**aug_config['moderate'])
```

- Augmentation factor: ~3-4x
- Training time: 20-30% increase
- Use for: Production training, best results

### Aggressive Augmentation (Maximum Diversity)

```python
augmentor.augment_all(**aug_config['aggressive'])
```

- Augmentation factor: ~6-8x
- Training time: 50-80% increase
- Use for: Small datasets, overfitting issues

---

## Example: Complete Training Script Migration

### Before (Old System)

```python
# train_old.py
import pandas as pd
from traffic_forecast.data.stmgt_dataset import STMGTDataset

# Load pre-augmented data (WITH LEAKAGE)
df = pd.read_parquet('data/processed/all_runs_augmented.parquet')

# Split
train, val, test = split_temporal(df)

# Create datasets
train_dataset = STMGTDataset(train)
val_dataset = STMGTDataset(val)
test_dataset = STMGTDataset(test)

# Train model...
```

### After (New System)

```python
# train_new.py
import pandas as pd
import json
from traffic_forecast.data.augmentation_safe import SafeTrafficAugmentor
from traffic_forecast.data.stmgt_dataset import STMGTDataset

# Load original data (NO pre-augmentation)
df = pd.read_parquet('data/processed/all_runs_combined.parquet')

# Split FIRST
train, val, test = split_temporal(df)

# Load config
with open('configs/augmentation_config.json') as f:
    aug_config = json.load(f)

# Augment train only (NO LEAKAGE)
augmentor = SafeTrafficAugmentor(train, random_seed=42)
train_augmented = augmentor.augment_all(**aug_config['moderate'])

# Create datasets
train_dataset = STMGTDataset(train_augmented)
val_dataset = STMGTDataset(val)  # Original
test_dataset = STMGTDataset(test)  # Original

# Train model...
```

---

## Troubleshooting

### Issue: "SafeTrafficAugmentor not found"

```bash
# Make sure module is in correct location
ls traffic_forecast/data/augmentation_safe.py

# If missing, check if you pulled latest code
git pull origin master
```

### Issue: "Performance dropped after migration"

**This is expected and GOOD:**

- Old system: Artificially inflated performance due to leakage
- New system: True generalization performance
- Drop of 0.1-0.3 MAE is normal and indicates leakage was removed

**Compare:**

- Old MAE with leakage: 3.0 km/h
- New MAE (honest): 3.2 km/h ← More trustworthy

### Issue: "Training time increased significantly"

**Options:**

1. Use 'light' preset instead of 'aggressive'
2. Reduce num_workers if data loading is bottleneck
3. Use smaller batch_size if GPU memory is full
4. Consider training longer with less augmentation

### Issue: "Original data not found"

```bash
# Make sure you have the base dataset
ls data/processed/all_runs_combined.parquet

# If missing, regenerate from raw data
python scripts/data/process_data.py
```

---

## Performance Expectations

### Before Migration (With Leakage)

```
Test MAE: 3.0 km/h ← Artificially good
Generalization: Poor (doesn't hold on new data)
Scientific validity: Low
```

### After Migration (Leak-Free)

```
Test MAE: 3.2 km/h ← Honest performance
Generalization: Good (holds on new data)
Scientific validity: High
```

**The small performance drop is a FEATURE, not a bug:**

- Shows true model capability
- More reliable for deployment
- Scientifically sound for publication

---

## Validation Checklist

After migration, verify:

- [ ] No more pre-augmented parquet files in use
- [ ] Augmentation happens after temporal split
- [ ] Statistics computed from train data only
- [ ] Validation check passes: `validate_no_leakage()`
- [ ] Test performance is realistic (not artificially inflated)
- [ ] Training time acceptable (20-50% increase expected)
- [ ] Model still converges properly

---

## Files Location Reference

### New Files (Use These)

```
traffic_forecast/data/augmentation_safe.py      ← Safe augmentation module
configs/augmentation_config.json                ← New config format
docs/guides/safe_augmentation_guide.md          ← Usage guide
scripts/analysis/compare_augmentation_methods.py ← Validation tool
```

### Deprecated Files (Archived)

```
scripts/data/archive/augment_extreme.py              ← Old (has leakage)
scripts/data/archive/augment_data_advanced.py        ← Old (has leakage)
scripts/data/archive/DEPRECATION_NOTICE.md           ← Why deprecated
configs/archive/train_normalized_v1.json             ← Old config
configs/archive/train_normalized_v2.json             ← Old config
```

---

## Support and Documentation

### Full Documentation

- `docs/fix/data_leakage_fix.md` - Complete technical analysis
- `docs/guides/safe_augmentation_guide.md` - User guide
- `docs/guides/weather_data_explained.md` - Exogenous vs endogenous variables
- `docs/DATA_LEAKAGE_QUICK_REF.md` - Quick reference

### Key Concepts

1. **Exogenous variables** (weather, time) - NOT leakage
2. **Endogenous variables** (traffic) - CAN leak if used from test
3. **Temporal split** - ALWAYS split before augmentation
4. **Train-only statistics** - Core principle of safe augmentation

### Testing

Run comparison to verify improvement:

```bash
python scripts/analysis/compare_augmentation_methods.py \
  --dataset data/processed/all_runs_combined.parquet \
  --output-dir outputs/migration_validation
```

---

## Summary

**Key Changes:**

1. ✓ Augmentation moved from pre-training to in-training
2. ✓ Statistics computed from train data only
3. ✓ Old scripts archived with deprecation notice
4. ✓ New config format with presets
5. ✓ Validation tools to prevent leakage

**Benefits:**

1. ✓ No data leakage
2. ✓ True generalization performance
3. ✓ Scientifically valid results
4. ✓ Publication ready
5. ✓ Deployment ready

**Action Items:**

1. Update training scripts to use SafeTrafficAugmentor
2. Remove old augmented datasets
3. Update config files to new format
4. Run validation to verify no leakage
5. Retrain models with new system

---

**Migration completed?** Test with:

```bash
# Verify clean state
ls data/processed/*.parquet  # Should only see all_runs_combined.parquet
ls scripts/data/archive/     # Should see old augmentation scripts

# Run training with new system
python scripts/training/train_stmgt.py --config configs/train_normalized_v3.json
```

**Questions?** See `docs/fix/data_leakage_fix.md` for detailed technical explanation.
