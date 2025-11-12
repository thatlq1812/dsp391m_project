# Deprecated Augmentation Scripts

**Date:** 2025-11-12  
**Reason:** Data leakage in augmentation patterns

## Files Deprecated

- `augment_extreme.py` - Uses global statistics from entire dataset
- `augment_data_advanced.py` - Uses patterns from test set

## Why Deprecated?

These scripts compute augmentation patterns from the entire dataset (including test data), which creates information leakage:

1. **Pattern Leakage**: Hourly/DOW profiles include test set patterns
2. **Interpolation Leakage**: Creates runs between train/test boundaries
3. **Statistical Leakage**: Edge-specific stats computed from all data

## Replacement

Use `traffic_forecast/data/augmentation_safe.py` instead:

```python
from traffic_forecast.data.augmentation_safe import SafeTrafficAugmentor

# After temporal split
augmentor = SafeTrafficAugmentor(train_df)
train_augmented = augmentor.augment_all(
    noise_copies=3,
    weather_scenarios=5,
    jitter_copies=2
)
```

## Migration Guide

### Old Approach (Deprecated)

```bash
# Pre-augment entire dataset
python scripts/data/augment_extreme.py
python scripts/data/augment_data_advanced.py

# Use augmented dataset
python scripts/training/train_stmgt.py --dataset data/processed/all_runs_augmented.parquet
```

### New Approach (Recommended)

```python
# Augment AFTER temporal split in training script
from traffic_forecast.data.augmentation_safe import SafeTrafficAugmentor

# Split first
train_data, val_data, test_data = split_temporal(df)

# Augment train only
augmentor = SafeTrafficAugmentor(train_data)
train_augmented = augmentor.augment_all()

# Train with augmented data
# val and test remain original (no augmentation)
```

## Documentation

See:

- `docs/fix/data_leakage_fix.md` - Full assessment
- `docs/guides/safe_augmentation_guide.md` - Usage guide
- `docs/guides/weather_data_explained.md` - Exogenous vs endogenous variables

## Files Kept for Reference

These files are archived in `scripts/data/archive/` for:

- Historical reference
- Comparison experiments
- Understanding what was wrong

**Do not use in production or new experiments.**
