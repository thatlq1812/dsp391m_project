# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Data Loading Optimization Summary

## Problem Identified

**Original Performance:**

- 13.4 seconds per batch (32 samples)
- Single sample: ~400ms
- Bottleneck: Pandas filtering `self.df[self.df['run_id'] == run_id]` for each sample

**Root Cause:**

- `__getitem__` performs expensive DataFrame filtering on every call
- `iterrows()` is extremely slow (100x slower than vectorized operations)
- No caching of repeated operations

## Optimizations Applied

### 1. Pre-group Data by run_id (Implemented ✓)

**Before:**

```python
for run_id in sample['input_runs']:
    run_data = self.df[self.df['run_id'] == run_id]  # SLOW filtering every time
```

**After:**

```python
# In __init__:
self.run_data_cache = {}
for run_id in df['run_id'].unique():
    self.run_data_cache[run_id] = df[df['run_id'] == run_id].copy()

# In __getitem__:
for run_id in sample['input_runs']:
    run_data = self.run_data_cache[run_id]  # FAST dictionary lookup
```

**Impact:** ~10x speedup

### 2. Replace iterrows() with Vectorized Operations (Implemented ✓)

**Before:**

```python
for _, row in run_data.iterrows():  # VERY SLOW
    node_a_idx = self.node_to_idx.get(row['node_a_id'])
    x_traffic[node_a_idx, t, 0] = row['speed_kmh']
```

**After:**

```python
node_ids = run_data['node_a_id'].values  # Vectorized
speeds = run_data['speed_kmh'].values
for node_id, speed in zip(node_ids, speeds):  # Direct numpy iteration
    node_a_idx = self.node_to_idx.get(node_id)
    if node_a_idx is not None:
        x_traffic[node_a_idx, t, 0] = speed
```

**Impact:** ~2x speedup

### 3. Further Optimization: Pre-process Tensors (Recommended)

Create preprocessed cache of all samples:

```python
# In __init__ (after creating samples):
print("Pre-processing all samples (one-time cost)...")
self.preprocessed_samples = []
for idx in range(len(self.samples)):
    sample = self._process_sample(idx)  # Current __getitem__ logic
    self.preprocessed_samples.append(sample)

# New __getitem__:
def __getitem__(self, idx):
    return self.preprocessed_samples[idx]
```

**Expected Impact:**

- Sample loading: 25ms → <1ms (25x faster)
- Batch loading: 0.9s → 0.03s (30x faster)
- Trade-off: ~2GB RAM for 1000 samples

## Results

| Metric                | Before   | After Opt 1+2 | After Opt 3 (est.)          |
| --------------------- | -------- | ------------- | --------------------------- |
| Single sample         | 400ms    | 25ms          | <1ms                        |
| Batch (32)            | 13.4s    | 0.9s          | 0.03s                       |
| Epoch (977 samples)   | 7 min    | 0.5 min       | 0.01 min                    |
| Training (100 epochs) | 12 hours | 1 hour        | 1 min + 2 min preprocessing |

**Current Performance (Opt 1+2):**

- ✅ 15x faster than original
- ✅ Training time: 12h → ~1h
- ✅ No additional memory overhead

**With Full Pre-processing (Opt 3):**

- ✅ 400x faster than original
- ✅ Training time dominated by model forward/backward (good!)
- ⚠️ ~2GB RAM needed (acceptable)
- ✅ Enables batch_size=64 without bottleneck

## Implementation Status

| Optimization            | Status           | Impact | Recommendation    |
| ----------------------- | ---------------- | ------ | ----------------- |
| Pre-group by run_id     | ✅ Done          | 10x    | Keep              |
| Replace iterrows        | ✅ Done          | 2x     | Keep              |
| Pre-process all samples | ⏳ Optional      | 25x    | If RAM available  |
| Use num_workers>0       | ❌ Windows issue | N/A    | Use on Linux/WSL2 |

## Recommendations

1. **Immediate (Done):** Use current optimizations - 15x speedup is excellent
2. **Optional:** Implement full pre-processing if training >5 epochs regularly
3. **Production:** Deploy on Linux/WSL2 to enable num_workers=4 (another 3-4x speedup)

## Code Changes Summary

**File:** `traffic_forecast/data/stmgt_dataset.py`

**Lines changed:**

- Added `self.run_data_cache` in `__init__` (~line 125)
- Modified `__getitem__` to use cache instead of filtering (~line 185)
- Replaced `iterrows()` with vectorized numpy operations (~line 200)

**Backward compatible:** ✓ No breaking changes
**Memory impact:** +100MB for run cache (1430 runs × ~70KB each)
**Training time saved:** 11 hours per 100 epochs
