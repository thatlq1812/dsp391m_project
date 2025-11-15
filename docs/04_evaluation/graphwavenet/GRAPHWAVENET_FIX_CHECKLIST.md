# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# GraphWaveNet Bug Fix Checklist

**Created:** November 14, 2025  
**Estimated Time:** 5-8 hours  
**Priority:** CRITICAL

## Overview

This checklist guides fixing critical bugs in GraphWaveNet implementation that cause impossible prediction accuracy (MAE 0.02 km/h vs expected 3-4 km/h).

## Pre-Fix Verification

- [x] Document current bugs
- [x] Verify data quality is good
- [x] Compare with correct LSTM implementation
- [x] Create bug report documentation
- [x] Update CHANGELOG

## Priority 1: Add Data Normalization (2-3 hours) ✅ COMPLETED

### Task 1.1: Update GraphWaveNetTrafficPredictor class ✅

**File:** `traffic_forecast/models/graph/graph_wavenet.py`

- [x] Import StandardScaler at top of file
- [x] Add scalers to `__init__` method
- [x] Update `fit()` method to normalize data
  - [x] Reshape X_train and y_train for scaling
  - [x] Fit and transform training data
  - [x] Transform validation data (if provided)
  - [x] Train on normalized data
  - [x] Keep existing callbacks and early stopping
- [x] Update `predict()` method to handle normalization
  - [x] Normalize input X
  - [x] Make prediction on normalized data
  - [x] Denormalize output predictions
- [x] Update `save()` method to save scalers
  - [x] Import joblib
  - [x] Save scaler_X.pkl
  - [x] Save scaler_y.pkl
  - [x] Keep legacy scaler.npz for backward compatibility
- [x] Update `load()` classmethod to load scalers
  - [x] Load scaler_X.pkl if exists
  - [x] Load scaler_y.pkl if exists
  - [x] Load legacy scaler.npz for backward compatibility

### Task 1.2: Test normalization implementation ✅

- [x] Create test script `tests/test_graphwavenet_normalization.py`
  - [x] Test fit normalizes data
  - [x] Test predict denormalizes output
  - [x] Test save/load preserves scalers
  - [x] Test predictions are in realistic range
  - [x] Test no impossible accuracy
- [x] Run tests and verify all pass - **ALL 5 TESTS PASSED**

## Priority 2: Fix Data Leakage (1-2 hours)

### Task 2.1: Review current statistics calculation

**File:** `traffic_forecast/evaluation/graphwavenet_wrapper.py`

- [ ] Review `_ensure_edge_order()` method (line 171-190)
- [ ] Review `_fill_missing_values()` method (line 192-202)
- [ ] Document current leakage mechanism

### Task 2.2: Choose fix approach

**Option A: Per-sequence statistics (most conservative)**

- [ ] Calculate stats only from past data for each sequence
- [ ] Modify `_fill_missing_values()` to accept current index
- [ ] Use only `pivot.iloc[:current_idx+1]` for stats
- Pros: No leakage, true online scenario
- Cons: More complex, slower

**Option B: Global statistics (acceptable for baseline)**

- [ ] Keep current global statistics approach
- [ ] Add clear documentation about methodology
- [ ] Note this is acceptable for offline baseline comparison
- Pros: Simple, fast, works for baselines
- Cons: Not suitable for real-time deployment

**Decision:** [ ] Option A [ ] Option B

Recommendation: **Option B** for baseline, document clearly

### Task 2.3: Implement chosen approach

If Option A:

- [ ] Modify `_fill_missing_values()` signature
- [ ] Update all callers to pass current index
- [ ] Calculate statistics from past only
- [ ] Test with small dataset

**Decision: Option B** ✅

- [x] Add comprehensive docstrings explaining methodology
- [x] Note acceptable for offline baseline comparison
- [x] Document not suitable for real-time without modification
- [x] Keep current implementation (it's consistent)

### Task 2.4: Test and verify ✅

- [x] Review code - statistics calculated correctly from training split
- [x] Verify consistent application across all splits
- [x] Document methodology clearly in code
- [x] Note: Primary issue was missing normalization, not statistics

## Priority 3: Update Training Script (30 mins) ✅ COMPLETED

**File:** `scripts/training/train_graphwavenet_baseline.py`

- [x] Remove manual scaler_std access
- [x] Trust evaluator metrics completely
- [x] Update comments about normalization
- [x] Clarify that training history is in normalized scale

## Priority 4: Retrain and Validate (2-3 hours)

### Task 4.1: Clean old results

- [ ] Backup old results
  ```bash
  mv outputs/final_comparison/run_20251114_190346 \
     outputs/final_comparison/run_20251114_190346_BUGGY_BACKUP
  ```

### Task 4.2: Retrain GraphWaveNet 🔄 IN PROGRESS

- [x] Run training script (outputs/graphwavenet_baseline_fixed)
- [x] Monitor training progress - Training running, MAE decreasing normally
- [ ] Wait for completion (100 epochs)
- [ ] Verify training curves are smooth
- [ ] Check for convergence

### Task 4.3: Validate results

Expected results:

- Train MAE: 3.2-3.8 km/h
- Val MAE: 3.5-4.0 km/h
- Test MAE: 3.5-4.2 km/h
- R²: 0.50-0.75

Checks:

- [ ] Metrics are in realistic range
- [ ] Val/test MAE >= train MAE (normal generalization)
- [ ] No near-perfect R² (0.9999)
- [ ] Performance between LSTM (4.42) and STMGT (1.88)

### Task 4.4: Compare with baselines

- [ ] Extract metrics from all models
- [ ] Create comparison table
- [ ] Verify GraphWaveNet is between LSTM and STMGT
- [ ] Document improvements over LSTM (if any)

## Priority 5: Documentation (1 hour)

### Task 5.1: Update verification report

**File:** `docs/GRAPHWAVENET_VERIFICATION_REPORT.md`

- [ ] Document fixes applied
- [ ] Show before/after metrics
- [ ] Explain normalization approach
- [ ] Explain statistics calculation choice
- [ ] Mark verification as PASSED (after fixes)

### Task 5.2: Update CHANGELOG

**File:** `docs/CHANGELOG.md`

- [ ] Add entry for bug fixes
- [ ] Document new results
- [ ] Mark old results as invalid
- [ ] Link to bug reports

### Task 5.3: Update comparison report

**File:** `outputs/final_comparison/run_YYYYMMDD_HHMMSS/comparison_report.json`

- [ ] Regenerate with new GraphWaveNet results
- [ ] Include all three models (LSTM, GraphWaveNet, STMGT)
- [ ] Update improvements calculations
- [ ] Verify all metrics are consistent

### Task 5.4: Create summary

- [ ] Update README with new results
- [ ] Create visual comparison chart
- [ ] Document model ranking
- [ ] Highlight key findings

## Testing Checklist

### Unit Tests

- [ ] Test normalization (fit/predict/save/load)
- [ ] Test statistics calculation
- [ ] Test sequence preparation
- [ ] Test prediction alignment
- [ ] All tests pass

### Integration Tests

- [ ] Train on small dataset (10 samples)
- [ ] Verify can save and load
- [ ] Verify predictions are reproducible
- [ ] Verify metrics are computed correctly

### Regression Tests

- [ ] Compare new results with expected range
- [ ] Verify no performance degradation vs LSTM
- [ ] Verify improvements are documented
- [ ] Check no new bugs introduced

## Final Validation

### Code Review

- [ ] Review all changed files
- [ ] Verify no debugging code left
- [ ] Check code style and formatting
- [ ] Verify docstrings are updated
- [ ] Run linter (if available)

### Documentation Review

- [ ] All documentation updated
- [ ] CHANGELOG is complete
- [ ] Bug reports are clear
- [ ] Fix guide is accurate

### Metrics Validation

- [ ] All metrics in realistic range
- [ ] No impossible accuracy (MAE < 1)
- [ ] No perfect predictions (R² > 0.99)
- [ ] Proper train < val < test ordering

### Comparison Validation

- [ ] LSTM: 4.42 km/h (baseline)
- [ ] GraphWaveNet: 3.5-4.0 km/h (fixed)
- [ ] STMGT: 1.88 km/h (best)
- [ ] Ordering makes sense

## Completion Criteria

- [ ] All bugs documented
- [ ] All fixes implemented
- [ ] All tests passing
- [ ] Model retrained
- [ ] Metrics validated
- [ ] Documentation updated
- [ ] Comparison table updated
- [ ] No critical TODOs remaining

## Time Tracking

| Task            | Estimated | Actual | Notes |
| --------------- | --------- | ------ | ----- |
| Normalization   | 2-3h      |        |       |
| Leakage fix     | 1-2h      |        |       |
| Training script | 0.5h      |        |       |
| Retraining      | 2-3h      |        |       |
| Documentation   | 1h        |        |       |
| **Total**       | **5-8h**  |        |       |

## Notes

Add any notes, issues, or decisions here:

-
-
-

## Sign-off

- [ ] All fixes implemented and tested
- [ ] Results validated and realistic
- [ ] Documentation complete
- [ ] Ready for use in comparisons

**Completed by:** **\*\***\_\_\_\_**\*\***  
**Date:** **\*\***\_\_\_\_**\*\***  
**Review by:** **\*\***\_\_\_\_**\*\***  
**Date:** **\*\***\_\_\_\_**\*\***
