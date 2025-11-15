# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# STMGT Data Guide

Complete guide to data concepts, schemas, and pipeline for STMGT Traffic Forecasting System.

**Last Updated:** November 12, 2025

---

## Quick Reference

**Data Format:** JSON (raw) → Parquet (processed)
**Collection:** 29 days, 15-minute intervals
**Coverage:** 62 nodes, 144 edges in HCMC district
**Size:** ~205K samples after preprocessing

---

## 1. Data Collection

### 1.1 Raw Data Structure

Each run directory contains:

```
data/runs/run_YYYYMMDD_HHMMSS/
├── traffic_edges.json      # Traffic speed data (primary)
├── nodes.json              # Node information
├── weather_snapshot.json   # Weather conditions
├── edges.json              # Edge metadata
└── statistics.json         # Run statistics
```

### 1.2 Traffic Data Schema

**traffic_edges.json:**

```json
[
  {
    "node_a_id": 123,
    "node_b_id": 456,
    "speed_kmh": 25.5,
    "distance_m": 350.2,
    "timestamp": "2025-11-01T14:30:00Z"
  }
]
```

**Key fields:**

- `node_a_id`, `node_b_id`: Edge endpoints (OSM node IDs)
- `speed_kmh`: Average speed on segment (target variable)
- `distance_m`: Segment length
- `timestamp`: Collection time (UTC)

### 1.3 Weather Data Schema

**weather_snapshot.json:**

```json
[
  {
    "node_id": 123,
    "timestamp": "2025-11-01T14:30:00Z",
    "temperature_c": 28.5,
    "wind_speed_kmh": 12.3,
    "precipitation_mm": 0.0
  }
]
```

**Why weather is NOT data leakage:**

- Weather is **exogenous variable** (not dependent on traffic)
- Forecasts available at prediction time
- Similar to time-of-day features

**Full explanation:** [Weather Data Explained](archive/guides/weather_data_explained.md)

---

## 2. Data Preprocessing

### 2.1 Preprocessing Pipeline

**Script:** `scripts/data/preprocess_runs.py`

**Steps:**

1. **Load JSON files** from each run directory
2. **Parse timestamps** to datetime objects
3. **Add temporal features:**
   - hour (0-23)
   - minute (0, 15, 30, 45)
   - day_of_week (0-6)
   - is_weekend (boolean)
4. **Add traffic categories:**
   - congestion_level (heavy/moderate/light/free_flow)
   - speed_category (very_slow to very_fast)
5. **Merge traffic + weather** by node_id
6. **Combine all runs** into single dataframe
7. **Save Parquet** format for fast loading

**Output:** `data/processed/all_runs_combined.parquet`

### 2.2 Data Validation

**Script:** `scripts/data/validate_processed_dataset.py`

**Checks:**

- Required columns present
- No NaN in critical fields (speed, timestamp)
- Speed values reasonable (0-150 km/h)
- Temporal features valid (hour 0-23, dow 0-6)
- Weather features filled (use means for missing)

**Output:** Validation report with warnings/errors

---

## 3. Dataset Characteristics

### 3.1 Traffic Statistics

**Speed Distribution:**

```
Min: 3.37 km/h (heavy congestion)
Max: 52.84 km/h (free flow)
Mean: 18.72 ± 7.03 km/h
Median: 17.68 km/h
25th percentile: 13.88 km/h (congested)
75th percentile: 22.19 km/h (moderate)
```

**Coefficient of Variation:** 37% (high variability, NOT stable traffic)

**Multi-modal Distribution:**

1. Free-flow: 40-50 km/h (highways, off-peak)
2. Moderate: 15-25 km/h (normal urban)
3. Congested: <13 km/h (peak hours)

### 3.2 Temporal Patterns

**Rush Hours:**

- Morning: 7-9 AM (speed drops 30-50%)
- Evening: 5-7 PM (speed drops 30-50%)

**Day of Week:**

- Weekday: Average 18.2 km/h
- Weekend: Average 21.0 km/h (+15%)

**Weather Impact:**

- Clear: Average 19.5 km/h
- Rain: Average 15.8 km/h (-19%)

### 3.3 Spatial Characteristics

**Network:**

- 62 nodes (intersections)
- 144 edges (road segments)
- Graph diameter: 12 hops
- Coverage: ~4 km² district

**Node Types:**

- Major intersections: High connectivity (5-8 edges)
- Minor intersections: Low connectivity (2-3 edges)
- Highway nodes: High speed variance

---

## 4. Data Splits

### 4.1 Temporal Split Strategy

**Method:** Blocked time split by runs (no shuffle!)

**Rationale:**

- Preserve temporal ordering
- Prevent information leakage
- Simulate real-world deployment

**Split Ratios:**

- Train: 70% (first runs chronologically)
- Val: 15% (middle runs)
- Test: 15% (last runs)

**Example:**

```
29 days of data, 1 run every 3 hours = 232 runs
Train: runs 1-162 (70%)
Val: runs 163-197 (15%)
Test: runs 198-232 (15%)
```

### 4.2 Why NOT Random Split?

**Problem with random split:**

- Test samples from same time periods as train
- Model memorizes patterns, not learns
- Inflated performance metrics

**Correct approach (blocked split):**

- Test samples from FUTURE time periods
- Model must generalize to unseen temporal patterns
- Realistic performance evaluation

---

## 5. Data Augmentation

### 5.1 SafeTrafficAugmentor

**Purpose:** Increase training data diversity without information leakage

**Key Principle:** Use ONLY training set statistics

**Methods:**

1. **Noise Injection:** Add Gaussian noise (train std)
2. **Weather Scenarios:** Synthetic weather (train ranges)
3. **Temporal Jitter:** Small time shifts (train patterns)

**Critical:** Augment AFTER split, only on train set

### 5.2 Augmentation Workflow

```python
# 1. Split data first
train_df, val_df, test_df = temporal_split(data)

# 2. Initialize with train only
augmentor = SafeTrafficAugmentor(train_df)

# 3. Augment training data
train_augmented = augmentor.augment_all(
    noise_copies=3,
    weather_scenarios=5,
    jitter_copies=2
)

# 4. Validate no leakage
augmentor.validate_no_leakage(train_augmented, val_df, test_df)

# 5. Combine (augmented train + original val/test)
final_data = concat([train_augmented, val_df, test_df])
```

**Full guide:** [AUGMENTATION.md](AUGMENTATION.md)

---

## 6. Feature Engineering

### 6.1 Traffic Features

**Raw:**

- `speed_kmh`: Target variable
- `distance_m`: Segment length

**Engineered:**

- `congestion_level`: Categorical (heavy/moderate/light/free_flow)
- `speed_category`: Categorical (very_slow to very_fast)

### 6.2 Temporal Features

**Cyclic Encoding:**

```python
hour_sin = sin(2π * hour / 24)
hour_cos = cos(2π * hour / 24)
dow_sin = sin(2π * dow / 7)
dow_cos = cos(2π * dow / 7)
```

**Binary:**

- `is_weekend`: 0 (weekday) or 1 (weekend)
- `is_rush_hour`: 0 (off-peak) or 1 (7-9 AM, 5-7 PM)

### 6.3 Weather Features

**Continuous:**

- `temperature_c`: Temperature in Celsius
- `wind_speed_kmh`: Wind speed
- `precipitation_mm`: Rainfall

**Normalized:**

```python
temp_norm = (temp - temp_mean) / temp_std
wind_norm = (wind - wind_mean) / wind_std
precip_norm = (precip - precip_mean) / precip_std
```

---

## 7. Normalization

### 7.1 Critical Rule

**ALWAYS use train-only statistics for normalization**

```python
# CORRECT: Compute stats from train only
train_mean = train_df['speed_kmh'].mean()
train_std = train_df['speed_kmh'].std()

# Apply to all splits
train_normalized = (train_df - train_mean) / train_std
val_normalized = (val_df - train_mean) / train_std  # Use train stats!
test_normalized = (test_df - train_mean) / train_std  # Use train stats!
```

**Why?**

- Test set statistics are "future information"
- Using test stats = data leakage
- Model overfits to test distribution

### 7.2 Normalization in Code

**Location:** `traffic_forecast/data/stmgt_dataset.py`

```python
class STMGTDataset:
    def __init__(self, split='train', ...):
        # Compute normalization from FULL dataset first
        self.speed_mean = df['speed_kmh'].mean()
        self.speed_std = df['speed_kmh'].std()

        # Then split
        if split == 'train':
            df_split = df[df['run_id'].isin(train_runs)]
        # ...

        # Normalize using TRAIN stats (stored in self)
        df_split['speed_normalized'] = (
            df_split['speed_kmh'] - self.speed_mean
        ) / self.speed_std
```

**Note:** Current implementation has minor issue - computes stats from full dataset. Should compute from train only. See [FIXES.md](FIXES.md) for details.

---

## 8. Data Loading

### 8.1 DataLoader Configuration

**Batch Size:** 32 (default)

- Small network (62 nodes) fits in memory
- Larger batches → faster training
- Smaller batches → better generalization

**Num Workers:** 2-4 (default: auto-detect)

- Windows: Use 0 or 2 (multiprocessing issues)
- Linux: Use 4-8 for speed

**Shuffle:**

- Train: True (within temporal blocks)
- Val/Test: False (preserve order)

### 8.2 Sliding Window

**Sequence Creation:**

```python
# For each timestamp t:
history = data[t-12:t]      # 12 timesteps = 3 hours
target = data[t:t+12]       # 12 timesteps = 3 hours ahead
```

**Sequence Length:** 12 timesteps (15 min interval = 3 hours)
**Prediction Length:** 12 timesteps (3 hours forecast)

---

## 9. Data Pipeline Summary

```
1. Collection
   ├─ Raw JSON (traffic + weather + nodes)
   └─ 15-minute intervals, 29 days

2. Preprocessing
   ├─ Parse timestamps
   ├─ Add temporal features
   ├─ Merge traffic + weather
   ├─ Add categories
   └─ Save Parquet

3. Validation
   ├─ Check required columns
   ├─ Validate ranges
   └─ Report issues

4. Split (Temporal)
   ├─ Train: 70% (early runs)
   ├─ Val: 15% (middle runs)
   └─ Test: 15% (late runs)

5. Augmentation (Optional)
   ├─ Initialize with train only
   ├─ Augment training set
   └─ Validate no leakage

6. Normalization
   ├─ Compute stats from train
   └─ Apply to all splits

7. DataLoader
   ├─ Create sliding windows
   ├─ Batch and shuffle
   └─ Feed to model
```

---

## 10. Common Issues & Solutions

### Issue: Missing Weather Data

**Solution:**

```python
# Fill with column means
weather_cols = ['temperature_c', 'wind_speed_kmh', 'precipitation_mm']
for col in weather_cols:
    df[col] = df[col].fillna(df[col].mean())
```

### Issue: Data Leakage

**Check:**

- Augmentation uses train-only stats? ✓
- Normalization uses train-only stats? ✓
- Split before augmentation? ✓
- No test info in features? ✓

**Validation:** Run `SafeTrafficAugmentor.validate_no_leakage()`

### Issue: Imbalanced Speed Distribution

**Solution:** Augmentation handles this

- Over-sample rare speed ranges
- Add synthetic congestion scenarios
- Balance training data

---

## 11. Related Documentation

**Preprocessing:**

- [TRAINING_WORKFLOW.md](../TRAINING_WORKFLOW.md) - Step-by-step preprocessing

**Augmentation:**

- [AUGMENTATION.md](AUGMENTATION.md) - Augmentation guide
- [FIXES.md](FIXES.md) - Data leakage fix

**Training:**

- [TRAINING.md](TRAINING.md) - Training guide
- [MODEL.md](MODEL.md) - Model overview

**API:**

- [API.md](API.md) - Data schemas in API responses

---

## Questions or Issues

For data-related questions, refer to:

- Preprocessing issues → [TRAINING_WORKFLOW.md](../TRAINING_WORKFLOW.md)
- Augmentation → [AUGMENTATION.md](AUGMENTATION.md)
- Leakage concerns → [FIXES.md](FIXES.md)
- Weather concepts → [archive/guides/weather_data_explained.md](archive/guides/weather_data_explained.md)
