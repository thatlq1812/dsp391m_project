# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Data Leakage Quick Reference

Date: 2025-11-12

## TL;DR

**Problem:** Augmentation uses global stats from entire dataset (including test data)

**Severity:** MODERATE (not severe)

**Why Not Severe:** Model normalization uses train-only stats correctly

**Status:** Safe augmentation module implemented, integration pending

---

## Key Files

| File                                               | Purpose                                  |
| -------------------------------------------------- | ---------------------------------------- |
| `docs/fix/data_leakage_fix.md`                     | Full assessment (6 sections, ~400 lines) |
| `traffic_forecast/data/augmentation_safe.py`       | Leak-free augmentation module            |
| `docs/guides/safe_augmentation_guide.md`           | User guide and best practices            |
| `scripts/analysis/compare_augmentation_methods.py` | Validation tool                          |

---

## Quick Actions

### For Current Experiments

Use original data without augmentation:

```bash
# Use this dataset
data/processed/all_runs_combined.parquet

# Avoid these (contain leakage)
data/processed/all_runs_augmented.parquet
data/processed/all_runs_extreme_augmented.parquet
```

### For New Experiments

Use safe augmentation:

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

---

## What's Safe, What's Not

### SAFE ✓

- Normalization in `STMGTDataset` (uses train-only stats)
- Temporal splitting in `UnifiedEvaluator` (timestamp-based)
- Graph structure (fixed, domain knowledge)
- **Weather data as input** (forecasts available at prediction time)

### LEAKY ⚠️

- `augment_hourly_interpolation` (interpolates between all runs)
- `augment_temporal_extrapolation` (uses full dataset patterns)
- `augment_pattern_variations` (uses global edge profiles)

### NEW SAFE METHODS ✓

- `augment_noise_injection` (Gaussian noise from train std)
- `augment_weather_scenarios` (train weather ranges)
- `augment_temporal_jitter` (train hourly patterns)

---

## Testing

Run comparison:

```bash
python scripts/analysis/compare_augmentation_methods.py \
  --dataset data/processed/all_runs_combined.parquet \
  --output-dir outputs/augmentation_comparison
```

---

## Impact on Results

**Expected changes when fixing:**

- Test MAE may increase slightly (more realistic)
- Generalization gap should be more honest
- Training may need more epochs
- Model should be more robust

**Not expected:**

- Dramatic performance drop
- Complete retraining needed
- Current results invalidated

---

## Timeline

- **Now**: Use original data for critical experiments
- **This week**: Avoid interpolation/extrapolation methods
- **Next 2 weeks**: Integrate safe augmentation into pipeline
- **Month**: Validate with full experiments

---

## FAQ

**Q: Is using weather data from the dataset also leakage?**

A: NO. Weather data is:

- Observable at prediction time (from forecast APIs)
- Exogenous variable (external input, not what we predict)
- Similar to using "hour of day" or "day of week"
- In deployment: comes from weather forecast services

See Section 4 in full doc for detailed explanation.

**Q: What's the difference between weather and traffic patterns?**

A:

- Weather: Available as forecasts → NOT leakage
- Traffic: Must be predicted → Leakage if using future patterns

---

## Questions?

See full documentation: `docs/fix/data_leakage_fix.md`
