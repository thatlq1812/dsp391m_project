# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Super Dataset Quick Start Guide

Fast-track guide to generate the 1-year challenging traffic dataset.

---

## Prerequisites

```bash
# Ensure you have the project environment
conda activate dsp

# Required packages (already in environment)
- pandas
- numpy
- pyyaml
- tqdm
```

---

## Step 1: Review Configuration (5 minutes)

```bash
# Open configuration file
code configs/super_dataset_config.yaml
```

**Key parameters to verify:**

```yaml
temporal:
  start_date: "2024-01-01" # Adjust if needed
  duration_days: 365 # Full year

incidents:
  rate_per_week: 3 # More = harder dataset

construction:
  num_zones_per_year: 8 # Long-term disruptions

weather:
  rain:
    probability_by_month: # Seasonal realism
      6: 0.30 # June (rainy season)
      12: 0.05 # December (dry season)
```

---

## Step 2: Dry Run (1 minute)

Test configuration without generating full dataset:

```bash
python scripts/data/generate_super_dataset.py --dry-run
```

**Expected output:**

```
[SuperDataset] Initialized with config: configs/super_dataset_config.yaml
  Duration: 365 days
  Interval: 10 minutes
  Total timestamps: 52,560
[DRY RUN] Configuration loaded successfully!
Remove --dry-run to start generation.
```

---

## Step 3: Generate Dataset (30-60 minutes)

**Full generation with visualization:**

```bash
python scripts/data/generate_super_dataset.py \
    --config configs/super_dataset_config.yaml \
    --output data/processed/super_dataset_1year.parquet \
    --visualize
```

**Progress stages:**

```
[Step 1/8] Initializing speed matrix...
[Step 2/8] Generating base traffic patterns...
[Step 3/8] Applying seasonal patterns...
[Step 4/8] Injecting traffic incidents...
[Step 5/8] Adding construction zones...
[Step 6/8] Applying weather effects...
[Step 7/8] Injecting special events...
[Step 8/8] Applying holiday patterns...
[Assembly] Building final DataFrame...
[Validation] Checking dataset quality...
[Save] Writing to data/processed/super_dataset_1year.parquet...
```

---

## Step 4: Verify Output

**Check generated files:**

```bash
ls -lh data/processed/super_dataset*

# Expected:
# super_dataset_1year.parquet      (~500MB)
# super_dataset_metadata.json      (events, holidays)
# super_dataset_statistics.json    (validation metrics)
# super_dataset_splits.json        (train/val/test)
```

**Quick inspection:**

```python
import pandas as pd

# Load dataset
df = pd.read_parquet('data/processed/super_dataset_1year.parquet')

print(f"Shape: {df.shape}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Speed range: [{df['speed_kmh'].min():.2f}, {df['speed_kmh'].max():.2f}] km/h")
print(f"\nColumns: {list(df.columns)}")

# Check event coverage
print(f"\nIncidents: {df['is_incident'].sum():,}")
print(f"Construction: {df['is_construction'].sum():,}")
print(f"Holidays: {df['is_holiday'].sum():,}")
```

**Expected output:**

```
Shape: (7,568,640, 13)
Date range: 2024-01-01 00:00:00 to 2024-12-31 23:50:00
Speed range: [3.37, 52.84] km/h

Columns: ['timestamp', 'node_a_id', 'node_b_id', 'speed_kmh',
          'is_incident', 'is_construction', 'weather_condition',
          'is_holiday', 'event_type', 'temperature_c',
          'precipitation_mm', 'wind_speed_kmh', 'humidity_percent']

Incidents: 156,000
Construction: 1,152,000
Holidays: 151,200
```

---

## Step 5: Train Models on New Dataset

**Update training scripts to use new dataset:**

```bash
# GraphWaveNet
python scripts/training/train_graphwavenet_baseline.py \
    --dataset data/processed/super_dataset_1year.parquet \
    --output-dir outputs/graphwavenet_super_dataset

# LSTM
python scripts/training/train_lstm_baseline.py \
    --dataset data/processed/super_dataset_1year.parquet \
    --output-dir outputs/lstm_super_dataset

# STMGT
python scripts/training/train_stmgt.py \
    --dataset data/processed/super_dataset_1year.parquet \
    --output-dir outputs/stmgt_super_dataset
```

**Expected training time:**

- GraphWaveNet: 3-4 hours (longer due to more data)
- LSTM: 2-3 hours
- STMGT: 4-5 hours

---

## Step 6: Evaluate & Compare

**Run evaluation suite:**

```bash
python scripts/evaluation/compare_all_models.py \
    --dataset data/processed/super_dataset_1year.parquet \
    --models graphwavenet_super_dataset lstm_super_dataset stmgt_super_dataset \
    --output outputs/final_comparison_super_dataset
```

**Expected performance:**

| Model        | MAE (km/h) | RMSE (km/h) | Notes                        |
| ------------ | ---------- | ----------- | ---------------------------- |
| GraphWaveNet | 4-6        | 5-7         | No autocorrelation shortcuts |
| LSTM         | 5-7        | 6-8         | Temporal only                |
| **STMGT**    | **3-4**    | **4-5**     | Spatial-temporal advantage   |

---

## Troubleshooting

### Issue: Out of Memory

**Solution:** Reduce chunk size in config:

```yaml
output:
  chunk_size: 5000 # Default: 10000
```

### Issue: Generation Too Slow

**Solution:** Use parallel processing (if implemented):

```bash
python scripts/data/generate_super_dataset.py \
    --config configs/super_dataset_config.yaml \
    --output data/processed/super_dataset_1year.parquet \
    --workers 4  # Use 4 CPU cores
```

### Issue: Unrealistic Patterns

**Solution:** Adjust validation thresholds:

```yaml
validation:
  max_jump_per_timestep: 15.0 # Increase if needed
  min_autocorr_lag12: 0.70 # Lower = more chaotic
  max_autocorr_lag12: 0.95 # Higher = more predictable
```

---

## Customization Examples

### Example 1: More Challenging Dataset

```yaml
incidents:
  rate_per_week: 5 # More frequent incidents

construction:
  num_zones_per_year: 12 # More construction

weather:
  rain:
    probability_by_month:
      6: 0.50 # Increase rainy season
```

### Example 2: Longer Duration

```yaml
temporal:
  duration_days: 730 # 2 years
  total_timestamps: 105120 # Update accordingly
```

### Example 3: Different Topology

```yaml
spatial:
  topology_file: data/processed/custom_topology.parquet
  num_edges: 200 # More edges
  num_nodes: 80
```

---

## Next Steps After Generation

1. **Visualize sample weeks** to verify patterns
2. **Check autocorrelation** to ensure challenge level
3. **Train baseline models** (GraphWaveNet, LSTM)
4. **Train STMGT** on new dataset
5. **Compare results** - STMGT should maintain advantage
6. **Document findings** in final report

---

## Timeline

| Phase  | Duration | Task                               |
| ------ | -------- | ---------------------------------- |
| Week 1 | 5 days   | Implement base patterns + seasonal |
| Week 2 | 5 days   | Event injection + propagation      |
| Week 3 | 5 days   | Integration + validation           |
| Week 4 | 5 days   | Training + evaluation              |

**Total:** ~4 weeks to complete dataset and retrain all models

---

## Resources

- **Design Doc:** `docs/SUPER_DATASET_DESIGN.md`
- **Config:** `configs/super_dataset_config.yaml`
- **Generator:** `scripts/data/generate_super_dataset.py`
- **README:** `scripts/data/README.md`

---

**Status:** 📋 Ready for implementation  
**Approval:** Required before starting  
**Priority:** High (critical for proper evaluation)
