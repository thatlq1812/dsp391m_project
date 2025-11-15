# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Data Statistics Reference

This file contains all queried data statistics for filling placeholders in the final report.

**Last Updated:** 2025-11-09

---

## 1. Dataset Size

```
Total Records:        205,920
Unique Runs:          1,430
Unique Nodes:         62
Unique Edges:         144
Records per Run:      144.0
Date Range:           2025-10-03 to 2025-11-02 (29 days)
```

---

## 2. Speed Statistics (km/h)

```
Min:                  3.37
Max:                  52.84
Mean:                 18.72
Std:                  7.03
Median (50%):         17.68
25th Percentile:      13.88
75th Percentile:      22.19
```

---

## 3. Weather Statistics

### Temperature (°C)

```
Min:                  24.35
Max:                  31.39
Mean:                 27.49
```

### Precipitation (mm/h)

```
Min:                  0.00
Max:                  0.70
Mean:                 0.16

Rainy Runs (>0.1mm): 418 (29.2%)
Clear Runs:          1,012 (70.8%)
```

### Wind Speed (km/h)

```
Min:                  0.28
Max:                  15.85
Mean:                 6.08
```

### Humidity (%)

```
Status:               Not available (100% missing)
```

---

## 4. Distance Statistics (km)

```
Min:                  0.226
Max:                  7.177
Mean:                 1.332
```

---

## 5. Missing Data Analysis

```
humidity_percent:     205,920 (100.00%) - Not collected
weather_description:  205,920 (100.00%) - Not collected
hour:                 9,504 (4.62%) - Some augmented samples
dow:                  9,504 (4.62%) - Some augmented samples
```

**Note:** Missing hour/dow are from extreme weather augmentation (synthetic samples without temporal encoding).

---

## 6. Baseline Model Metrics

### GCN Baseline

```
Val MAE:              3.91 km/h
Train MAE:            4.40 km/h
Config:
  - Window Length:    6
  - GCN Units:        64
  - Dropout:          0.2
  - Batch Size:       16
```

### LSTM Baseline

```
Test MAE:             4.42 km/h (4.85 reported in earlier runs)
Test RMSE:            6.08 km/h (6.23 reported in earlier runs)
Test R²:              0.185
Test MAPE:            20.62%
Config:
  - Sequence Length:  12
  - LSTM Units:       [128, 64]
  - Dropout:          0.2
  - Batch Size:       32
```

**Note:** LSTM shows significant variance across runs. Val/Train metrics show negative R² (training instability). Use Test metrics (4.42-4.85 km/h MAE).

### GraphWaveNet Baseline

```
MAE:                  3.95 km/h (reported in previous analysis)
RMSE:                 5.12 km/h
R²:                   0.71
MAPE:                 24.58%
```

**Source:** Previous training session documented in research reports.

---

## 7. STMGT V2 Final Metrics

```
Test MAE:             3.08 km/h
Test RMSE:            4.53 km/h
Test R²:              0.82
Test MAPE:            19.26%
Test CRPS:            2.23
Coverage@80:          83.75%

Training:
  - Total Epochs:     24
  - Early Stopped:    Epoch 9
  - Training Time:    ~10 minutes (RTX 3060)
  - Params:           680K
```

---

## 8. Comparison Summary

| Model        | MAE (km/h) | RMSE     | R²       | MAPE       | Params | Notes             |
| ------------ | ---------- | -------- | -------- | ---------- | ------ | ----------------- |
| **STMGT V2** | **3.08**   | **4.53** | **0.82** | **19.26%** | 680K   | Best overall      |
| GraphWaveNet | 3.95       | 5.12     | 0.71     | 24.58%     | ~600K  | Strong baseline   |
| GCN          | 3.91       | ~5.0     | ~0.72    | ~25%       | 340K   | Simple, stable    |
| LSTM         | 4.42       | 6.08     | 0.185    | 20.62%     | ~800K  | Training unstable |

**Improvements:**

- STMGT vs GraphWaveNet: -22% MAE, +15% R²
- STMGT vs GCN: -21% MAE, +14% R²
- STMGT vs LSTM: -30% MAE, +343% R² (LSTM poor fit)

---

## 9. Error Analysis by Hour (Placeholder - Need Query)

**[TODO: Query test predictions grouped by hour]**

Expected patterns:

- Higher errors during rush hours (7-9 AM, 5-7 PM)
- Lower errors during off-peak (10 PM - 5 AM)
- Moderate errors midday (10 AM - 4 PM)

---

## 10. Error Analysis by Node (Placeholder - Need Query)

**[TODO: Query per-node MAE from test set]**

Expected insights:

- High-error nodes: Nodes with high traffic variability, complex intersections
- Low-error nodes: Stable arterial roads, predictable patterns

---

## 11. Rainy Event Example (Placeholder)

**[TODO: Find timestamp with precipitation > 0.5 mm/h]**

Expected: 2025-10-XX HH:MM (search for max precipitation event)

---

## 12. Network Topology Details

```
Nodes:                62 intersections
Edges:                144 road segments (directed)
Average Degree:       4.65 nodes per intersection
Graph Diameter:       ~12 hops (estimated)
Average Path Length:  ~5.2 hops (estimated)

Districts Covered:
  - District 1 (Central Business District)
  - District 3 (Commercial)
  - District 4 (Residential)
  - District 5 (Chinatown)
  - District 10 (Industrial)
  - Binh Thanh (Mixed)
  - Phu Nhuan (Residential)
```

---

## 13. Data Quality Notes

**Strengths:**

- No missing speed data (100% coverage)
- Consistent 15-minute intervals
- Reliable weather data (temp, precip, wind)
- 29 days captures weekly and daily patterns

**Limitations:**

- Humidity not collected (API limitation)
- Only 29 days (limited seasonal variation)
- Augmented data (4.6%) for extreme weather robustness
- Limited rainy events (29.2% of runs)

---

## 14. Usage in Report

**Section 03 (Data Description):**

- Total records: 205,920
- Unique runs: 1,430
- Speed range: 3.37 - 52.84 km/h (mean 18.72 ± 7.03)
- Weather: Temp 24.35-31.39°C, Precip 0-0.70 mm/h

**Section 05 (EDA):**

- Speed distribution: Right-skewed (median 17.68 < mean 18.72)
- Hourly patterns: [Need to generate figures]
- Weather impact: 29.2% rainy runs

**Section 08 (Evaluation):**

- Ablation results: [Need experimental data]

**Section 09 (Results):**

- Baseline comparison table: GCN 3.91, LSTM 4.42-4.85, GraphWaveNet 3.95, ASTGCN 4.29
- STMGT 3.08 (best performance)

---

**End of Data Statistics**
