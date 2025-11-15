# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# GraphWaveNet Critical Bugs Report

**Date:** November 14, 2025  
**Status:** CRITICAL - Model Results Invalid

## Executive Summary

GraphWaveNet model shows suspiciously perfect performance (MAE = 0.0198 km/h) due to **two critical bugs**:

1. **Missing data normalization** - Model trains on raw speed values without standardization
2. **Potential data leakage** - Statistics calculated from entire pivot table during training

These bugs make the current GraphWaveNet results **completely invalid** and cannot be used for comparison.

## Observed Anomalies

### Unrealistic Performance

From `outputs/final_comparison/run_20251114_190346/graphwavenet/run_20251114_204714/results.json`:

```json
{
  "results": {
    "train": {
      "mae": 0.29227309119155903,
      "rmse": 0.5742412674651616,
      "r2": 0.9922356748846209
    },
    "val": {
      "mae": 0.019796429195619327,
      "rmse": 0.03495599616615141,
      "r2": 0.9999642092337879
    },
    "test": {
      "mae": 0.01979642919561933,
      "rmse": 0.03495599616615141,
      "r2": 0.9999642092337879
    }
  }
}
```

**Problems:**

- Test MAE of **0.0198 km/h** is impossible for real traffic data
- R² of **0.9999** indicates near-perfect prediction
- Val/Test MAE is **15x better** than train MAE (should be worse!)
- LSTM baseline has realistic MAE of **4.42 km/h**

## Bug 1: Missing Data Normalization

### Current Implementation

**File:** `traffic_forecast/models/graph/graph_wavenet.py`

```python
def fit(self, X_train, y_train, X_val=None, y_val=None, ...):
    # Store normalization params (BUT NEVER USE THEM!)
    self.scaler_mean = np.mean(y_train)  # Line 324
    self.scaler_std = np.std(y_train)    # Line 325

    # Train directly on raw data - NO NORMALIZATION!
    history = self.model.fit(
        X_train, y_train,  # Raw speed values (0-120 km/h)
        validation_data=validation_data,
        ...
    )
```

**File:** `traffic_forecast/evaluation/graphwavenet_wrapper.py`

```python
def fit(self, train_data, val_data, ...):
    # Prepare sequences - NO NORMALIZATION
    X_train, y_train = self._prepare_sequences(train_data, speed_col, add_noise=True)
    X_val, y_val = self._prepare_sequences(val_data, speed_col, add_noise=False)

    # Train on raw speed values
    history = self.model.fit(X_train, y_train, X_val, y_val, ...)
```

### Correct Implementation (LSTM)

**File:** `traffic_forecast/models/lstm_traffic.py`

```python
from sklearn.preprocessing import StandardScaler

def fit(self, X_train, y_train, X_val=None, y_val=None, ...):
    # Create and fit scalers
    self.scaler_X = StandardScaler()
    self.scaler_y = StandardScaler()

    # Normalize training data
    X_train_scaled = self.scaler_X.fit_transform(X_train)
    y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

    # Normalize validation data
    if X_val is not None:
        X_val_scaled = self.scaler_X.transform(X_val)
        y_val_scaled = self.scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()

    # Train on normalized data
    history = self.model.fit(X_train_scaled, y_train_scaled, ...)

def predict(self, X):
    # Normalize input
    X_scaled = self.scaler_X.transform(X)
    y_pred_scaled = self.model.predict(X_scaled_seq)

    # Denormalize output
    y_pred = self.scaler_y.inverse_transform(y_pred_scaled).ravel()
    return y_pred
```

### Impact

1. **Neural network trains poorly** - Large raw values (0-120 km/h) without normalization
2. **Metrics meaningless** - Reported MAE in wrong scale
3. **Cannot compare** - LSTM uses normalized scale, GraphWaveNet uses raw scale

## Bug 2: Potential Data Leakage in Statistics Calculation

### Problematic Code

**File:** `traffic_forecast/evaluation/graphwavenet_wrapper.py` (Lines 171-190)

```python
def _ensure_edge_order(self, pivot: pd.DataFrame, data: pd.DataFrame, speed_col: str):
    if self.edge_order is None:
        # First time - calculate statistics from entire pivot table
        self.edge_order = sorted(pivot.columns.tolist())

        # PROBLEM: pivot contains ALL timestamps in training split
        # This includes future values relative to any sequence!
        means = pivot.mean(skipna=True).reindex(self.edge_order)  # Line 180
        self.edge_means = means.fillna(global_mean)

        stds = pivot.std(skipna=True).reindex(self.edge_order)    # Line 183
        self.edge_stds = stds.fillna(global_std).replace(0, global_std)

    return pivot.reindex(columns=self.edge_order)
```

### The Problem

When preparing training sequences:

1. **Pivot table created** from entire training split (e.g., 67,680 samples)
2. **Statistics calculated** from this entire pivot table:
   - `pivot.mean()` - Mean across all training timestamps
   - `pivot.std()` - Std across all training timestamps
3. **Used for imputation** when filling missing values

**Example Timeline:**

```
Training data: T1, T2, T3, ..., T100
Creating sequence at T10: [T1-T9] -> T10

When filling missing values at T5:
- Uses edge_means calculated from T1-T100 (includes T6-T100 = FUTURE!)
- This leaks future information into past predictions
```

### Where Leakage Occurs

**File:** `traffic_forecast/evaluation/graphwavenet_wrapper.py` (Lines 192-202)

```python
def _fill_missing_values(self, pivot: pd.DataFrame):
    filled = pivot.copy()

    # Limited interpolation (OK - only uses nearby past/future)
    filled = filled.interpolate(method='time', limit=self.max_interp_gap)
    filled = filled.ffill(limit=self.max_interp_gap)

    # LEAKAGE: Uses edge_means calculated from entire training set
    # This includes future values!
    filled = filled.fillna(self.edge_means)  # Line 199

    return filled, missing_mask
```

### Impact Assessment

**Severity:** Medium to High

- Edge means include future traffic patterns in same split
- When imputing missing values, model sees future statistics
- Explains why val/test metrics are better than training metrics
- However, interpolation limit (3 steps) somewhat mitigates this

## Bug 3: Inconsistent Metric Reporting

### Training History vs Evaluation

From training script output:

```python
# Training history reports normalized MAE
train_mae_hist = history.history['mae'][-1]  # 0.292 (normalized scale?)

# But evaluation uses raw predictions
metrics = evaluator.calculate_metrics(
    split_data['speed'].values,  # Raw speed (km/h)
    preds                        # Raw predictions (km/h)
)
# Returns: MAE = 0.0198 km/h
```

**The inconsistency suggests:**

- Model trained on one scale (possibly partially normalized internally by Keras)
- Evaluated on another scale (raw km/h)
- Creates confusing results

## Comparison with Other Models

### LSTM Baseline (Correct Implementation)

```json
{
  "train": { "mae": 4.35, "rmse": 5.92, "r2": 0.18 },
  "val": { "mae": 4.28, "rmse": 5.84, "r2": -0.0 },
  "test": { "mae": 4.42, "rmse": 6.02, "r2": -0.06 }
}
```

**Observations:**

- Realistic traffic prediction errors (4-5 km/h)
- Validation slightly better than training (expected with dropout)
- Test slightly worse than validation (normal generalization)
- Uses proper StandardScaler normalization

### STMGT V3 (Correct Implementation)

```json
{
  "test": { "mae": 1.88, "rmse": 3.07, "r2": 0.89 }
}
```

**Observations:**

- Good performance (1.88 km/h MAE)
- Realistic for advanced spatial-temporal model
- Better than LSTM due to graph structure

### GraphWaveNet (Buggy Implementation)

```json
{
  "train": { "mae": 0.29, "rmse": 0.57, "r2": 0.99 },
  "val": { "mae": 0.02, "rmse": 0.03, "r2": 0.9999 },
  "test": { "mae": 0.02, "rmse": 0.03, "r2": 0.9999 }
}
```

**Observations:**

- **Impossible accuracy** - Too good to be true!
- Val/test better than train (red flag)
- Near-perfect R² (0.9999) indicates severe overfitting or leakage

## Required Fixes

### Priority 1: Add Data Normalization

**File:** `traffic_forecast/models/graph/graph_wavenet.py`

```python
class GraphWaveNetTrafficPredictor:
    def __init__(self, ...):
        # Add scalers
        from sklearn.preprocessing import StandardScaler
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

    def fit(self, X_train, y_train, X_val=None, y_val=None, ...):
        # Reshape for scaling
        n_samples, seq_len, n_nodes, n_features = X_train.shape
        X_train_2d = X_train.reshape(-1, n_features)
        y_train_2d = y_train.reshape(-1, 1)

        # Fit and transform training data
        X_train_scaled = self.scaler_X.fit_transform(X_train_2d)
        X_train_scaled = X_train_scaled.reshape(n_samples, seq_len, n_nodes, n_features)

        y_train_scaled = self.scaler_y.fit_transform(y_train_2d)
        y_train_scaled = y_train_scaled.reshape(y_train.shape)

        # Transform validation data
        if X_val is not None:
            X_val_2d = X_val.reshape(-1, n_features)
            X_val_scaled = self.scaler_X.transform(X_val_2d)
            X_val_scaled = X_val_scaled.reshape(X_val.shape)

            y_val_2d = y_val.reshape(-1, 1)
            y_val_scaled = self.scaler_y.transform(y_val_2d)
            y_val_scaled = y_val_scaled.reshape(y_val.shape)

            validation_data = (X_val_scaled, y_val_scaled)

        # Train on normalized data
        history = self.model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=validation_data,
            ...
        )

        return history

    def predict(self, X):
        # Normalize input
        n_samples, seq_len, n_nodes, n_features = X.shape
        X_2d = X.reshape(-1, n_features)
        X_scaled = self.scaler_X.transform(X_2d)
        X_scaled = X_scaled.reshape(n_samples, seq_len, n_nodes, n_features)

        # Predict
        y_pred_scaled = self.model.predict(X_scaled, verbose=0)

        # Denormalize output
        y_pred_2d = y_pred_scaled.reshape(-1, 1)
        y_pred = self.scaler_y.inverse_transform(y_pred_2d)
        y_pred = y_pred.reshape(y_pred_scaled.shape)

        return y_pred
```

### Priority 2: Fix Statistics Calculation

**Option A: Calculate statistics per-sequence (more conservative)**

```python
def _fill_missing_values(self, pivot: pd.DataFrame, current_idx: int):
    """Fill missing values using only past data."""
    # Only use data up to current_idx for statistics
    past_pivot = pivot.iloc[:current_idx+1]

    # Calculate statistics from past only
    edge_means = past_pivot.mean(skipna=True)

    # Fill only current sequence
    filled = pivot.copy()
    filled = filled.interpolate(method='time', limit=self.max_interp_gap)
    filled = filled.ffill(limit=self.max_interp_gap)
    filled = filled.fillna(edge_means)

    return filled
```

**Option B: Use global statistics from full dataset (acceptable for baselines)**

```python
def _ensure_edge_order(self, pivot: pd.DataFrame, data: pd.DataFrame, speed_col: str):
    """Calculate global statistics - OK for baseline since applied consistently."""
    if self.edge_order is None:
        self.edge_order = sorted(pivot.columns.tolist())

        # Use global mean from raw data (not pivot)
        self.global_speed_mean = float(data[speed_col].mean())

        # Calculate per-edge statistics from full training set
        # NOTE: This is acceptable for a baseline if documented,
        # but not for production/real-time systems
        means = pivot.mean(skipna=True).reindex(self.edge_order)
        self.edge_means = means.fillna(self.global_speed_mean)

    return pivot.reindex(columns=self.edge_order)
```

### Priority 3: Save/Load Scalers

**File:** `traffic_forecast/models/graph/graph_wavenet.py`

```python
def save(self, save_dir: Path):
    """Save model and scalers."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    self.model.save(save_dir / 'graphwavenet_model.keras')

    # Save scalers
    import joblib
    if self.scaler_X is not None:
        joblib.dump(self.scaler_X, save_dir / 'scaler_X.pkl')
    if self.scaler_y is not None:
        joblib.dump(self.scaler_y, save_dir / 'scaler_y.pkl')

@classmethod
def load(cls, load_dir: Path):
    """Load model and scalers."""
    import joblib

    # Load model
    model = keras.models.load_model(...)

    # Load scalers
    predictor = cls.__new__(cls)
    predictor.model = model

    scaler_x_path = load_dir / 'scaler_X.pkl'
    if scaler_x_path.exists():
        predictor.scaler_X = joblib.load(scaler_x_path)

    scaler_y_path = load_dir / 'scaler_y.pkl'
    if scaler_y_path.exists():
        predictor.scaler_y = joblib.load(scaler_y_path)

    return predictor
```

## Testing Plan

After fixes are applied:

1. **Retrain GraphWaveNet** with normalization
2. **Verify metrics** are in reasonable range (2-4 km/h MAE)
3. **Compare training curves** - should be smooth and converge
4. **Validation check** - val MAE should be similar to or slightly worse than train MAE
5. **Cross-validation** - ensure consistent performance across folds

### Expected Results After Fix

```json
{
  "train": {"mae": 3.2-3.8, "rmse": 4.5-5.5, "r2": 0.60-0.75},
  "val":   {"mae": 3.5-4.0, "rmse": 5.0-6.0, "r2": 0.55-0.70},
  "test":  {"mae": 3.5-4.2, "rmse": 5.0-6.2, "r2": 0.50-0.70}
}
```

**Reasoning:**

- GraphWaveNet should perform between LSTM (4.42) and STMGT (1.88)
- Adaptive adjacency learning helps but not as much as full STMGT architecture
- Target range: 3.5-4.0 km/h MAE

## Impact on Project

### Current Status

- **GraphWaveNet results are INVALID** - Cannot be used for comparison
- All analysis comparing GraphWaveNet performance must be discarded
- Model comparison table needs to be updated after fixes

### Action Items

- [ ] Fix normalization in `graph_wavenet.py`
- [ ] Fix scaler save/load in `graph_wavenet.py`
- [ ] Review and potentially fix statistics calculation in `graphwavenet_wrapper.py`
- [ ] Retrain GraphWaveNet baseline with fixes
- [ ] Update comparison report
- [ ] Update documentation to explain fixes
- [ ] Add unit tests for normalization

### Estimated Time

- Implementation: 2-3 hours
- Testing: 1-2 hours
- Retraining: 1-2 hours (depending on hardware)
- Documentation: 1 hour

**Total: 5-8 hours**

## Lessons Learned

1. **Always normalize data** for neural networks - Especially important for regression tasks
2. **Verify metrics make sense** - If results are too good, investigate thoroughly
3. **Compare implementations** - Cross-reference with known working baselines (LSTM)
4. **Test incrementally** - Should have caught this earlier with unit tests
5. **Document assumptions** - Statistics calculation method should be explicit

## References

- LSTM implementation: `traffic_forecast/models/lstm_traffic.py` (correct normalization)
- GraphWaveNet paper: "Graph WaveNet for Deep Spatial-Temporal Graph Modeling"
- Best practices: Always normalize inputs for neural networks
- Expected traffic prediction MAE: 2-6 km/h for baseline models
