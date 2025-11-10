# Maintainer Profile

**Name:** THAT Le Quang
- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Section 6: Data Cleaning & Preprocessing

## 6.1 Data Cleaning Steps

### 6.1.1 Outlier Detection and Removal
- Speed outliers: Remove if <0 or >120 km/h
- Weather outliers: Temperature <15°C or >45°C flagged
- Missing data handling: Forward-fill weather, drop missing speeds

### 6.1.2 Normalization

**Speed Normalization:**
```python
# Z-score normalization
speed_mean = 18.72  # km/h
speed_std = 7.03    # km/h
speed_normalized = (speed - speed_mean) / speed_std
```

**Weather Normalization:**
```python
# [PLACEHOLDER: Add actual weather normalization stats]
temp_mean = 27.49
temp_std = 2.15
# wind, precipitation normalization
```

## 6.2 Graph Construction

**Adjacency Matrix:**
- 62×62 binary matrix
- Edge exists if road segment connects two nodes
- Stored in `cache/adjacency_matrix.npy`

## 6.3 Sequence Creation

**Sliding Window:**
- seq_len=12 (3 hours history, 15-min intervals)
- pred_len=12 (3 hours forecast)
- Stride=1 (overlapping windows)

## 6.4 Data Augmentation

**Strategy:** Extreme augmentation for small dataset
- Time jitter: ±1 timestep
- Node masking: Drop 10% nodes randomly
- Details in `configs/augmentation_config.json`

## 6.5 Train/Val/Test Split

**Split Strategy:**
- Train: 70% (first 11,500 samples)
- Val: 15% (next 2,400 samples)
- Test: 15% (last 2,400 samples)
- **Temporal split** (no shuffling to prevent leakage)

**[PLACEHOLDER: Add exact sample counts after query]**

---

**Next:** [Exploratory Data Analysis →](05_eda.md)
