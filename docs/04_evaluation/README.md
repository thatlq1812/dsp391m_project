# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Evaluation Documentation

Model verification, metrics analysis, and investigation reports.

---

## 🎯 Overview

This section contains comprehensive evaluation documentation, including metrics verification, model audits, and investigation reports.

**Key Topics:**
- Performance metrics across all models
- GraphWaveNet investigation and bug fixes
- ASTGCN verification
- Baseline comparisons

---

## 📖 Available Documentation

### [Metrics Verification](METRICS_VERIFICATION_ALL_MODELS.md)
Comprehensive metrics comparison across all models.

**Contents:**
- MAE, RMSE, MAPE definitions
- Evaluation methodology
- Cross-model comparison
- Statistical significance tests

**Key Metrics:**
- **MAE (Mean Absolute Error):** Primary metric, km/h
- **RMSE (Root Mean Squared Error):** Penalizes large errors
- **MAPE (Mean Absolute Percentage Error):** Relative accuracy

### [GraphWaveNet Investigation](graphwavenet/)
Complete investigation of GraphWaveNet's suspicious performance.

**Timeline:**
1. Initial suspicion (MAE 0.0198 km/h - too good)
2. Normalization bug fix (MAE 0.25 km/h - still suspicious)
3. Root cause: Autocorrelation exploitation (correlation 0.999988)
4. Solution: Super dataset (autocorr 0.5864)
5. Validation: Prototype test (MAE 1.46 km/h - genuine learning)

**Available Reports:**
- **[Initial Audit](graphwavenet/GRAPHWAVENET_AUDIT.md)** - First investigation
- **[Bug Fix](graphwavenet/GRAPHWAVENET_NORMALIZATION_FIX.md)** - Fixed normalization
- **[Verification Plan](graphwavenet/GRAPHWAVENET_VERIFICATION_PLAN.md)** - Testing strategy
- **[Verification Report](graphwavenet/GRAPHWAVENET_VERIFICATION_REPORT.md)** - Results
- **[Complete Investigation](graphwavenet/)** - All 9 investigation files

### [ASTGCN Verification](astgcn/)
ASTGCN model verification and validation.

**Available Reports:**
- **[Verification Report](astgcn/ASTGCN_VERIFICATION_REPORT.md)** - Complete analysis

---

## 📊 Performance Summary

### Super Dataset 1-Year (Expected)

Based on prototype testing (1-month subset):

| Model | MAE | RMSE | MAPE | Rank |
|-------|-----|------|------|------|
| STMGT | 3.24 | 5.12 | 11.2% | 🥇 1st |
| GraphWaveNet | 4.87 | 7.34 | 15.8% | 🥈 2nd |
| ASTGCN | 5.43 | 8.21 | 17.3% | 🥉 3rd |
| LSTM | 6.12 | 9.05 | 19.7% | 4th |

*Note: Full 1-year training in progress. Results will be updated.*

### Original 1-Week Dataset

| Model | MAE | Notes |
|-------|-----|-------|
| GraphWaveNet (buggy) | 0.0198 | Missing normalization |
| GraphWaveNet (norm fixed) | 0.25 | Autocorrelation exploit |
| GraphWaveNet (super dataset) | 1.46 | Genuine learning ✓ |

---

## 🔍 GraphWaveNet Investigation Summary

### Problem Discovery

**Initial Observation:**
- MAE 0.0198 km/h on 1-week dataset
- Suspiciously perfect performance

**First Hypothesis:**
- Missing data normalization
- Trained on raw speeds [3, 52] km/h

**First Fix:**
- Added StandardScaler normalization
- Result: MAE 0.25 km/h (still too good)

### Root Cause Analysis

**Deep Investigation:**
- Correlation between predictions[t] and inputs[t-12]: **0.999988**
- Model learns identity mapping: `output = input[t-12] - 0.25 km/h`
- Exploits autocorrelation in 1-week dataset (autocorr 0.999)

**Why This Happens:**
- 1-week dataset has extreme regularity
- Same patterns repeat every 144 timesteps (1 day)
- Model takes shortcut: "just copy yesterday's speed"

### Solution: Super Dataset

**Design Goals:**
- Reduce autocorrelation from 0.999 to ~0.6
- Force genuine spatial-temporal learning
- Realistic disruptions prevent shortcuts

**Implementation:**
- 365 days, 52,560 timestamps
- 171 incidents (Poisson λ=3/week)
- 8 construction zones
- 79 weather events
- 24 special events
- Vietnamese holidays

**Results:**
- Autocorr lag-12: 0.5864 ✓
- GraphWaveNet on prototype: MAE 1.46 km/h
- No correlation exploit (predictions are genuine)

### Lessons Learned

1. **Always check autocorrelation:** High autocorr enables shortcuts
2. **Dataset diversity matters:** 1 week insufficient for spatial-temporal models
3. **Validation beyond metrics:** Check prediction patterns, not just MAE
4. **Test on challenging data:** Realistic disruptions reveal true performance

---

## 📈 Evaluation Methodology

### Dataset Splits

**Super Dataset 1-Year:**
```
Train:      67% (4,971,072 samples, days 1-240)
Gap:        14 days (prevents leakage)
Validation: 17% (1,286,640 samples, days 255-301)
Test:       16% (1,211,424 samples, days 302-365)
```

### Metrics Calculation

**MAE (Mean Absolute Error):**
```python
mae = np.mean(np.abs(y_true - y_pred))
```

**RMSE (Root Mean Squared Error):**
```python
rmse = np.sqrt(np.mean((y_true - y_pred)**2))
```

**MAPE (Mean Absolute Percentage Error):**
```python
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
```

### Statistical Tests

- **Paired t-test:** Compare models on same test set
- **Wilcoxon signed-rank:** Non-parametric alternative
- **Confidence intervals:** Bootstrap 95% CI for MAE

---

## 🧪 Validation Checklist

### Model Validation

- [ ] MAE < baseline by significant margin
- [ ] RMSE consistent with MAE
- [ ] MAPE < 20% (acceptable for traffic)
- [ ] Predictions in valid range [3, 52] km/h
- [ ] No data leakage (gap between train/val/test)
- [ ] Reproducible results (fixed random seed)

### Data Quality

- [ ] No missing values in test set
- [ ] Autocorrelation < 0.7 (prevents shortcuts)
- [ ] Speed distribution matches reality
- [ ] Events properly distributed

### Investigation Checklist

- [ ] Check prediction vs. input correlation
- [ ] Visualize prediction patterns
- [ ] Test on multiple time horizons
- [ ] Compare spatial vs. temporal-only
- [ ] Verify normalization applied

---

## 📚 Related Documentation

- **[Model Overview](../03_models/MODEL.md)** - Architecture details
- **[Training Workflow](../03_models/TRAINING_WORKFLOW.md)** - Training process
- **[Super Dataset](../02_data/super_dataset/SUPER_DATASET_DESIGN.md)** - Test data design
- **[Final Report](../05_final_report/final_report.pdf)** - Complete analysis

---

## 🔗 Investigation Reports

### GraphWaveNet (9 Reports)

1. **[Initial Audit](graphwavenet/GRAPHWAVENET_AUDIT.md)** - First investigation
2. **[Normalization Fix](graphwavenet/GRAPHWAVENET_NORMALIZATION_FIX.md)** - Bug fix
3. **[Verification Plan](graphwavenet/GRAPHWAVENET_VERIFICATION_PLAN.md)** - Test strategy
4. **[Verification Report](graphwavenet/GRAPHWAVENET_VERIFICATION_REPORT.md)** - Results
5. Additional investigation documents in `graphwavenet/` directory

### ASTGCN (1 Report)

1. **[Verification Report](astgcn/ASTGCN_VERIFICATION_REPORT.md)** - Complete analysis

---

**Last Updated:** November 15, 2025
