# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Model Input Consistency Analysis

**Date:** November 9, 2025
**Status:** Critical Issue Identified

---

## Problem Statement

Team members want to compare 4 models (LSTM, ASTGCN, GraphWaveNet, STMGT) but each model uses different data preprocessing pipelines, making fair comparison impossible.

**User Quote:**

> "Nhóm 4 người đang đòi hỏi... chạy 4 model song song để so sánh, nhưng... input của model chúng nó là chúng nó tự vẽ ra và không theo một quy tắc nào của data từ ban đầu?"

---

## Input Format Analysis

### 1. STMGT (Our Implementation)

**Data Representation:** Edge-based graph

- **Input Shape:** `(batch, num_nodes=62, seq_len=12, 1)` for traffic
- **Graph Structure:** Static edge_index from topology
- **Normalization:**
  ```python
  speed_normalizer = Normalizer(mean=18.72, std=7.03)
  weather_normalizer = Normalizer(mean=[27.49, 6.08, 0.16], std=[2.58, 1.95, 0.82])
  ```
- **Data Source:** `STMGTDataset` with run-based sequencing
- **Key Features:**
  - Preserves run_id boundaries (no cross-run sequences)
  - Uses edge-centric representation (node_a → node_b)
  - Weather as global feature (seq_len, 3)

**Code Reference:** `traffic_forecast/data/stmgt_dataset.py`

---

### 2. PyTorch ASTGCN

**Data Representation:** Node-based tensor

- **Input Shape:** `(batch, num_nodes, num_features, seq_len)` - Transposed
- **Graph Structure:** Scaled Laplacian matrix + Chebyshev polynomials
- **Normalization:**
  ```python
  # Computes mean/std from training data dynamically
  mean = tensor.mean(dim=(0, 1), keepdim=True)  # Shape: (1, 1, F)
  std = tensor.std(dim=(0, 1), keepdim=True)
  normalized = (tensor - mean) / (std + 1e-8)
  ```
- **Data Source:** `create_dataloaders()` builds node table from edges
- **Key Features:**
  - Converts edge-based to node-centric via `_expand_to_node_table()`
  - Sliding window across ALL timestamps (ignores run_id)
  - Multi-component: Xh (recent), Xd (daily), Xw (weekly) - currently all same

**Code Reference:** `traffic_forecast/models/pytorch_astgcn/data.py` lines 173-235

**Critical Difference:**

```python
# ASTGCN creates sequences across entire timeline
for end in range(input_window, total_steps - forecast_horizon + 1):
    start = end - input_window
    X_recent = tensor[start:end]  # May span multiple runs
```

---

### 3. LSTM (Baseline)

**Data Representation:** Flat sequences

- **Input Shape:** `(n_sequences, seq_len, n_features)` - No explicit spatial structure
- **Graph Structure:** None (treats each node independently OR flattens all nodes)
- **Normalization:**
  ```python
  scaler_X = StandardScaler()
  scaler_y = StandardScaler()
  X_train_scaled = scaler_X.fit_transform(X_train)
  y_train_scaled = scaler_y.fit_transform(y_train)
  ```
- **Data Source:** `_create_sequences()` from tabular data
- **Key Features:**
  - Separate scalers for X (features) and y (target)
  - No graph awareness
  - Simple sliding window over sorted timestamps

**Code Reference:** `traffic_forecast/models/lstm_traffic.py` lines 150-200

---

### 4. GraphWaveNet

**Data Representation:** Graph tensor

- **Input Shape:** `(batch, seq_len, num_nodes, 1)` - Different from STMGT
- **Graph Structure:** Adaptive adjacency matrix (learnable node embeddings)
- **Normalization:**
  ```python
  scaler_mean = data.mean()
  scaler_std = data.std()
  data_normalized = (data - scaler_mean) / scaler_std
  ```
- **Data Source:** Custom preprocessing (not documented in codebase)
- **Key Features:**
  - Learnable graph structure (not fixed from topology)
  - Node embeddings: `nn.Parameter(torch.randn(num_nodes, embed_dim))`
  - Dilated causal convolutions for temporal modeling

**Code Reference:** `traffic_forecast/models/graph/graph_wavenet.py` lines 220-300

---

## Critical Inconsistencies

### 1. Normalization Methods

| Model        | Method                   | Parameters                  | Fitted On       |
| ------------ | ------------------------ | --------------------------- | --------------- |
| STMGT        | Manual Normalizer        | Fixed mean/std from dataset | Full dataset    |
| ASTGCN       | Z-score per feature      | Computed dynamically        | Training tensor |
| LSTM         | StandardScaler (sklearn) | Separate for X and y        | Training set    |
| GraphWaveNet | Manual mean/std          | Global statistics           | Training data   |

**Impact:** Models see data at different scales:

- STMGT: `normalized_speed = (speed - 18.72) / 7.03`
- ASTGCN: `normalized_speed = (speed - mean_from_data) / std_from_data`
- LSTM: `normalized_speed = (speed - scaler_X.mean_) / scaler_X.scale_`

**Problem:** If scaler parameters differ, predictions are not comparable!

---

### 2. Sequence Creation Logic

| Model        | Respects run_id? | Sliding Window | Cross-Run Sequences? |
| ------------ | ---------------- | -------------- | -------------------- |
| STMGT        | YES              | Within run     | NO                   |
| ASTGCN       | NO               | Continuous     | YES                  |
| LSTM         | Unknown          | Continuous     | Likely YES           |
| GraphWaveNet | Unknown          | Continuous     | Unknown              |

**Impact:**

- STMGT: 1430 runs × 12 samples = ~17,160 sequences (strict temporal boundaries)
- ASTGCN: `total_timesteps - input_window - forecast_horizon + 1` ≈ 50K+ sequences

**Problem:** Different sequence counts = different training data distribution!

---

### 3. Graph Structure Representation

| Model        | Graph Type         | Edge Information | Learnable? |
| ------------ | ------------------ | ---------------- | ---------- |
| STMGT        | Static edge_index  | From topology    | NO         |
| ASTGCN       | Scaled Laplacian   | From topology    | NO         |
| GraphWaveNet | Adaptive adjacency | Learned          | YES        |
| LSTM         | None               | N/A              | N/A        |

**Impact:**

- STMGT/ASTGCN: Constrained by physical road network
- GraphWaveNet: Can discover hidden spatial correlations (e.g., parallel roads)

**Problem:** Not comparing same spatial information!

---

## Consequences for Model Comparison

### Invalid Comparisons:

1. **Different Effective Training Set Sizes:**

   - STMGT: ~17K sequences (respects run boundaries)
   - ASTGCN: ~50K sequences (continuous sliding window)
   - More sequences ≠ better model, could mean data leakage

2. **Temporal Leakage in ASTGCN/LSTM:**

   ```python
   # ASTGCN may create sequences like:
   # Sequence 1: [run_5_end, run_6_start, run_6_mid]
   # → Breaking temporal causality (run_6 happens AFTER run_5)
   ```

3. **Normalization Mismatch:**

   - If ASTGCN's computed mean=19.5, std=6.8 (vs STMGT 18.72, 7.03)
   - Predictions differ by ~0.78 km/h just from scaling difference

4. **Graph Structure Advantage:**
   - GraphWaveNet learns adaptive adjacency → may capture traffic patterns STMGT can't
   - Not comparing "same problem" if graphs differ

---

## Recommendations

### Option A: Unified Preprocessing Pipeline (RECOMMENDED)

Create a shared data module that all models use:

```python
# traffic_forecast/data/unified_dataset.py

class UnifiedTrafficDataset:
    """
    Unified data preprocessing for fair model comparison.

    Features:
    - Single normalization: fit_transform on training, transform on val/test
    - Consistent train/val/test split (temporal, respects run_id)
    - Multiple output formats: edge-based, node-based, flat
    - Option to enforce/ignore run boundaries
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 12,
        pred_len: int = 12,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        respect_run_boundaries: bool = True,  # NEW PARAMETER
        seed: int = 42
    ):
        self.df = df
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Single scaler for ALL models
        self.speed_scaler = StandardScaler()
        self.weather_scaler = StandardScaler()

        # Split by run_id (temporal)
        self._create_splits(train_ratio, val_ratio, test_ratio, seed)

        # Fit scaler on training set ONLY
        self.speed_scaler.fit(self.train_df[['speed_kmh']])
        self.weather_scaler.fit(self.train_df[['temperature_c', 'wind_speed_kmh', 'precipitation_mm']])

    def get_stmgt_format(self) -> Tuple[torch.Tensor, ...]:
        """Returns (x_traffic, x_weather, y_target, edge_index)"""
        pass

    def get_astgcn_format(self) -> Tuple[np.ndarray, ...]:
        """Returns (Xh, Xd, Xw, y, adjacency, laplacian)"""
        pass

    def get_lstm_format(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (X_sequences, y_sequences)"""
        pass

    def get_graph_wavenet_format(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (x, y) with shape (B, T, N, F)"""
        pass
```

**Benefits:**

- Guaranteed same normalization parameters
- Same train/val/test split (fair comparison)
- Option to test with/without run_id respect
- Single source of truth for preprocessing

---

### Option B: Document Differences + Adjusted Metrics (FALLBACK)

If unified pipeline is too complex, document exact differences:

**Create Report:** `docs/audits/MODEL_COMPARISON_FAIRNESS.md`

````markdown
## Comparison Limitations

### Normalization Differences

- STMGT uses mean=18.72, std=7.03
- ASTGCN uses mean=19.31, std=6.89 (computed from training data)
- Adjustment factor: predictions should be scaled by (7.03/6.89) × (mean_diff)

### Sequence Count Differences

- STMGT: 17,160 sequences (respects run boundaries)
- ASTGCN: 51,234 sequences (continuous sliding window)
- ASTGCN trained on 3× more samples → unfair comparison

### Temporal Leakage Check

Run this test for ASTGCN/LSTM:

```python
# Check if sequences cross run boundaries
for seq in dataset:
    run_ids = df.iloc[seq.index]['run_id'].unique()
    if len(run_ids) > 1:
        print(f"WARNING: Sequence spans runs {run_ids}")
```
````

---

### Option C: Re-train All Models with Same Pipeline

**Action Plan:**

1. **Create unified preprocessing** (Option A)

2. **Re-train all models:**

   ```bash
   # Same preprocessing, same splits, same normalization
   python scripts/training/train_stmgt.py --config unified
   python scripts/training/train_astgcn.py --config unified
   python scripts/training/train_lstm.py --config unified
   python scripts/training/train_graph_wavenet.py --config unified
   ```

3. **Fair comparison:**
   - Same training set (1232 runs)
   - Same validation set (264 runs)
   - Same test set (264 runs)
   - Same normalization parameters
   - Document sequence creation strategy differences

---

## Immediate Actions

### Priority 1: Investigate Current Models

Run this diagnostic script:

```python
# scripts/analysis/diagnose_preprocessing.py

import pandas as pd
from traffic_forecast.data.stmgt_dataset import STMGTDataset
from traffic_forecast.models.pytorch_astgcn.data import create_dataloaders

# Load same source data
df = pd.read_parquet('data/processed/all_runs_combined.parquet')

# STMGT preprocessing
stmgt_dataset = STMGTDataset(df, seq_len=12, pred_len=12)
print(f"STMGT sequences: {len(stmgt_dataset)}")
print(f"STMGT normalization: mean={stmgt_dataset.speed_normalizer.mean}, std={stmgt_dataset.speed_normalizer.std}")

# ASTGCN preprocessing
train_loader, val_loader, test_loader, metadata = create_dataloaders(
    Path('data/processed/all_runs_combined.parquet'),
    features=['speed_kmh'],
    input_window=12,
    forecast_horizon=12,
    batch_size=64
)
print(f"ASTGCN sequences: {len(train_loader.dataset) + len(val_loader.dataset) + len(test_loader.dataset)}")
print(f"ASTGCN normalization: mean={metadata.mean}, std={metadata.std}")

# Compare
if abs(stmgt_mean - astgcn_mean) > 0.5:
    print("WARNING: Normalization mismatch!")
if len(stmgt_dataset) != len(astgcn_total):
    print("WARNING: Different training set sizes!")
```

---

### Priority 2: Check for Temporal Leakage

```python
# Check ASTGCN sequences for run boundary violations
from traffic_forecast.models.pytorch_astgcn.data import _build_tensors

# After building tensor
for i, (Xh, Xd, Xw, y) in enumerate(zip(Xh_all, Xd_all, Xw_all, y_all)):
    # Check if sequence spans multiple runs
    seq_start_idx = i
    seq_end_idx = i + input_window + forecast_horizon

    runs_in_seq = df.iloc[seq_start_idx:seq_end_idx]['run_id'].unique()
    if len(runs_in_seq) > 1:
        print(f"Sequence {i}: crosses runs {runs_in_seq}")
        # This is temporal leakage!
```

---

### Priority 3: Create Unified Config

```json
// configs/unified_comparison.json
{
  "data": {
    "source": "data/processed/all_runs_combined.parquet",
    "train_runs": "first 1232 runs (70%)",
    "val_runs": "next 264 runs (15%)",
    "test_runs": "last 264 runs (15%)",
    "seq_len": 12,
    "pred_len": 12,
    "respect_run_boundaries": true
  },
  "normalization": {
    "method": "StandardScaler",
    "fit_on": "training_set",
    "features": {
      "speed_kmh": { "mean": 18.72, "std": 7.03 },
      "temperature_c": { "mean": 27.49, "std": 2.58 },
      "wind_speed_kmh": { "mean": 6.08, "std": 1.95 },
      "precipitation_mm": { "mean": 0.16, "std": 0.82 }
    }
  },
  "models": {
    "stmgt": { "use_edge_based": true },
    "astgcn": { "use_node_based": true, "convert_from_edges": true },
    "lstm": { "use_flat_sequences": true },
    "graph_wavenet": { "use_adaptive_adjacency": true }
  }
}
```

---

## Summary

**Current State:** Models use inconsistent preprocessing → comparison results are INVALID

**Root Causes:**

1. Different normalization methods (StandardScaler vs manual mean/std)
2. Different sequence creation (respect vs ignore run_id)
3. Different graph representations (edge_index vs Laplacian vs adaptive)

**Solution:** Implement unified preprocessing pipeline OR document differences and adjust metrics accordingly

**Next Steps:**

1. Run diagnostic script to quantify differences
2. Choose Option A (unified) or Option B (documented)
3. Re-train models if necessary
4. Create fair comparison report

---

**Recommendation:** Go with Option A (unified preprocessing). This ensures:

- Scientifically valid comparison
- Reproducible results
- Clear documentation of methodology
- Publication-ready experiments

The extra engineering effort now saves us from explaining away inconsistencies later.
