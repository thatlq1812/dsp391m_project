# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# GraphWaveNet Audit Summary

**Date:** November 14, 2025  
**Auditor:** THAT Le Quang  
**Status:** 🚨 CRITICAL BUGS FOUND - RESULTS INVALID

## Quick Summary

GraphWaveNet implementation has **critical bugs** that make results invalid:

- Reports impossible MAE of **0.0198 km/h** (should be 3-4 km/h)
- Missing data normalization
- Potential data leakage in preprocessing

**Action Required:** Fix bugs and retrain before using results

## The Problem

### Current Results (INVALID)

```
Train:  MAE = 0.29 km/h   R² = 0.992
Val:    MAE = 0.02 km/h   R² = 0.9999  ← Impossible!
Test:   MAE = 0.02 km/h   R² = 0.9999  ← Impossible!
```

### Baseline Comparison

```
LSTM (correct):   MAE = 4.42 km/h   ← Realistic
STMGT (correct):  MAE = 1.88 km/h   ← Realistic
GraphWaveNet:     MAE = 0.02 km/h   ← Impossible!
```

## Root Causes

### Bug 1: No Data Normalization ⚠️ CRITICAL

**Problem:** Model trains on raw speed values without standardization

**Evidence:**

```python
# graph_wavenet.py line 324-325
self.scaler_mean = np.mean(y_train)  # Stored but NEVER used!
self.scaler_std = np.std(y_train)    # Stored but NEVER used!

# Trains directly on raw data
history = self.model.fit(X_train, y_train, ...)  # No normalization!
```

**What LSTM does correctly:**

```python
# lstm_traffic.py line 137-141
self.scaler_X = StandardScaler()
self.scaler_y = StandardScaler()
X_train_scaled = self.scaler_X.fit_transform(X_train)  ← Normalizes!
y_train_scaled = self.scaler_y.fit_transform(y_train)  ← Normalizes!
```

**Impact:**

- Neural network struggles with large raw values (0-120 km/h)
- Metrics are meaningless
- Cannot compare with other models

### Bug 2: Data Leakage ⚠️ HIGH

**Problem:** Statistics calculated from entire training set, including future

**Evidence:**

```python
# graphwavenet_wrapper.py line 180-182
# pivot contains ALL training timestamps
means = pivot.mean(skipna=True)  # Includes future data!
self.edge_means = means.fillna(global_mean)

# Used for imputation later
filled = filled.fillna(self.edge_means)  # Leaks future info!
```

**Timeline Example:**

```
Training data: T1, T2, T3, ..., T100
Creating sequence at T10: [T1-T9] → T10

When filling missing values at T5:
- Uses edge_means from T1-T100  ← Includes T6-T100 = FUTURE!
```

**Impact:**

- Model sees future statistics when predicting past
- Explains why val/test better than train (backwards!)

## Data Quality: ✅ GOOD

Ran comprehensive data verification - **data is fine:**

```
Dataset: 96,768 samples, 144 edges, 62 nodes
Speed range: 3.37-52.84 km/h (realistic)
No missing values in speed
Consistent temporal sampling
No distribution shift (4.85%)
```

**Conclusion:** Problem is in CODE, not DATA

## Required Fixes

### 1. Add Normalization (2-3 hours)

```python
from sklearn.preprocessing import StandardScaler

# In fit():
self.scaler_X = StandardScaler()
self.scaler_y = StandardScaler()
X_scaled = self.scaler_X.fit_transform(X_reshaped)
y_scaled = self.scaler_y.fit_transform(y_reshaped)

# In predict():
X_scaled = self.scaler_X.transform(X_reshaped)
y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
```

### 2. Fix Leakage (1-2 hours)

Options:

- A) Calculate stats per-sequence (conservative, no leakage)
- B) Use global stats consistently (acceptable for baseline, document clearly)

### 3. Retrain & Verify (2-3 hours)

- Retrain with fixes
- Verify realistic metrics (3.5-4.0 km/h)
- Update comparison reports

**Total Time:** 5-8 hours

## Expected Results After Fix

```
Train:  MAE = 3.2-3.8 km/h   R² = 0.60-0.75
Val:    MAE = 3.5-4.0 km/h   R² = 0.55-0.70
Test:   MAE = 3.5-4.2 km/h   R² = 0.50-0.70
```

**Reasoning:** GraphWaveNet should perform between LSTM (4.42) and STMGT (1.88)

## Documentation Created

1. **`docs/GRAPHWAVENET_CRITICAL_BUGS.md`** - Full technical analysis with code examples
2. **`scripts/analysis/verify_data_quality.py`** - Data verification tool
3. **`docs/CHANGELOG.md`** - Updated with findings

## Files with Bugs

- `traffic_forecast/models/graph/graph_wavenet.py` - No normalization
- `traffic_forecast/evaluation/graphwavenet_wrapper.py` - Data leakage
- `scripts/training/train_graphwavenet_baseline.py` - Training script (needs update after fix)

## Comparison: Buggy vs Correct

| Model        | Implementation              | MAE (km/h) | Realistic? |
| ------------ | --------------------------- | ---------- | ---------- |
| LSTM         | ✅ Correct (StandardScaler) | 4.42       | ✅ Yes     |
| STMGT        | ✅ Correct (Normalization)  | 1.88       | ✅ Yes     |
| GraphWaveNet | ❌ Buggy (No normalization) | 0.02       | ❌ NO!     |

## Lessons Learned

1. **Always normalize** neural network inputs
2. **Verify metrics** - if too good, investigate
3. **Compare baselines** - catch anomalies early
4. **Add tests** - prevent preprocessing bugs
5. **Document assumptions** - make preprocessing explicit

## Next Steps

- [ ] Implement fixes in `graph_wavenet.py`
- [ ] Implement fixes in `graphwavenet_wrapper.py`
- [ ] Add unit tests for normalization
- [ ] Retrain model
- [ ] Verify realistic metrics
- [ ] Update comparison reports
- [ ] Update documentation

## References

- Full bug report: `docs/GRAPHWAVENET_CRITICAL_BUGS.md`
- Data verification: `scripts/analysis/verify_data_quality.py`
- Correct LSTM: `traffic_forecast/models/lstm_traffic.py`
- CHANGELOG: `docs/CHANGELOG.md`
