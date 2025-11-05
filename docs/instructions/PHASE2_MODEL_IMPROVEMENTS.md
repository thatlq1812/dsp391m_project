# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Phase 2: Model Quality Assurance

**Duration:** 1 week  
**Priority:** 🟠 MEDIUM - Critical for model reliability  
**Dependencies:** Phase 1 complete (need baseline for comparison)

---

## Objectives

Eliminate overfitting risks and validate model performance:

- Verify test/validation metrics are correct
- Implement comprehensive cross-validation
- Achieve stable R²=0.45-0.55 with proper evaluation
- Complete ablation study
- Document model behavior thoroughly

---

## Task Breakdown

### Task 2.1: Investigate Test/Validation Discrepancy (4 hours) 🔴 CRITICAL

**Problem Identified:**

```
Current metrics:
  Validation MAE: 5.00 km/h
  Test MAE: 2.78 km/h  ← Suspiciously better than validation
  Test R²: 0.79        ← Too good to be true?
```

This is unusual - test performance should NOT exceed validation.

**Possible Causes:**

1. **Data leakage** - test data contaminated validation
2. **Random seed issue** - lucky split
3. **Small test set** - not representative
4. **Incorrect metric calculation** - bug in evaluation code
5. **Overfitting to validation** - model tuned on validation set

**Investigation Steps:**

```python
# Script: scripts/analysis/investigate_metrics.py

import pandas as pd
import numpy as np
from pathlib import Path
import json

def investigate_data_splits():
    """Analyze train/val/test splits for issues."""

    # Load the dataset
    data_path = Path("data/processed/all_runs_combined.parquet")
    df = pd.read_parquet(data_path)

    print(f"Total samples: {len(df)}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Check split method (should be temporal, not random)
    # Traffic data MUST use temporal split to avoid leakage

    # Load training config
    config_path = Path("outputs/stmgt_v2_20251101_215205/config.json")
    with open(config_path) as f:
        config = json.load(f)

    train_size = config.get('train_size', 0.7)
    val_size = config.get('val_size', 0.15)
    test_size = config.get('test_size', 0.15)

    # Calculate actual split indices
    n = len(df)
    train_end = int(n * train_size)
    val_end = train_end + int(n * val_size)

    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]

    print("\n=== Split Analysis ===")
    print(f"Train: {len(train_df)} samples ({len(train_df)/n*100:.1f}%)")
    print(f"  Date range: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"Val: {len(val_df)} samples ({len(val_df)/n*100:.1f}%)")
    print(f"  Date range: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
    print(f"Test: {len(test_df)} samples ({len(test_df)/n*100:.1f}%)")
    print(f"  Date range: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

    # Check for overlap (should be none!)
    train_dates = set(train_df['timestamp'])
    val_dates = set(val_df['timestamp'])
    test_dates = set(test_df['timestamp'])

    overlap_train_val = train_dates & val_dates
    overlap_val_test = val_dates & test_dates
    overlap_train_test = train_dates & test_dates

    print("\n=== Overlap Check ===")
    print(f"Train ∩ Val: {len(overlap_train_val)} samples {'⚠️ LEAKAGE!' if overlap_train_val else '✅'}")
    print(f"Val ∩ Test: {len(overlap_val_test)} samples {'⚠️ LEAKAGE!' if overlap_val_test else '✅'}")
    print(f"Train ∩ Test: {len(overlap_train_test)} samples {'⚠️ LEAKAGE!' if overlap_train_test else '✅'}")

    # Statistical comparison
    print("\n=== Speed Distribution Comparison ===")
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        speeds = split_df['speed'].values
        print(f"{split_name}:")
        print(f"  Mean: {speeds.mean():.2f} km/h")
        print(f"  Std: {speeds.std():.2f} km/h")
        print(f"  Min: {speeds.min():.2f} km/h")
        print(f"  Max: {speeds.max():.2f} km/h")
        print(f"  Median: {np.median(speeds):.2f} km/h")

    # Check if test set is easier (e.g., more highway speeds)
    # This could explain better performance

    return {
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
        'has_leakage': bool(overlap_train_val or overlap_val_test or overlap_train_test)
    }

if __name__ == "__main__":
    results = investigate_data_splits()
    print("\n=== Summary ===")
    print(json.dumps(results, indent=2))
```

**Run Investigation:**

```bash
conda run -n dsp python scripts/analysis/investigate_metrics.py
```

**Acceptance Criteria:**

- [ ] Data split method verified (temporal vs random)
- [ ] No overlap between train/val/test
- [ ] Distribution similarity checked
- [ ] Root cause identified
- [ ] Fix implemented if leakage found

---

### Task 2.2: Re-evaluate Model with Proper Metrics (3 hours)

**Goal:** Get accurate, trustworthy performance numbers.

**Proper Evaluation Protocol:**

```python
# Script: scripts/analysis/proper_evaluation.py

import torch
from traffic_forecast.models.stmgt import STMGT
from traffic_forecast.data import create_dataloaders
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
import numpy as np

def comprehensive_evaluation(checkpoint_path, config_path):
    """Evaluate model with multiple metrics and confidence intervals."""

    # Load model
    model = STMGT.load_from_checkpoint(checkpoint_path)
    model.eval()

    # Load data with FRESH split (don't trust old split)
    train_loader, val_loader, test_loader = create_dataloaders(
        config_path,
        temporal_split=True,  # Force temporal split
        shuffle_train=True,
        shuffle_val=False,
        shuffle_test=False
    )

    metrics = {}

    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                # Forward pass
                outputs = model(
                    batch['traffic'],
                    batch['edge_index'],
                    batch['weather'],
                    batch['temporal']
                )

                # Convert mixture to point predictions
                preds = mixture_to_moments(outputs)['mean']

                all_preds.append(preds.cpu().numpy())
                all_targets.append(batch['target'].cpu().numpy())

        # Concatenate
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        # Calculate metrics
        mae = mean_absolute_error(targets, preds)
        rmse = np.sqrt(np.mean((targets - preds) ** 2))
        r2 = r2_score(targets, preds)
        mape = mean_absolute_percentage_error(targets, preds) * 100

        # Calculate per-horizon metrics
        horizon_metrics = []
        for h in range(12):  # 12 horizons
            h_preds = preds[:, :, h]  # (batch, nodes, horizon)
            h_targets = targets[:, :, h]

            horizon_metrics.append({
                'horizon': h + 1,
                'mae': mean_absolute_error(h_targets.flatten(), h_preds.flatten()),
                'r2': r2_score(h_targets.flatten(), h_preds.flatten())
            })

        metrics[split_name] = {
            'mae': float(mae),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape),
            'horizon_breakdown': horizon_metrics
        }

        print(f"\n=== {split_name.upper()} Set Metrics ===")
        print(f"MAE: {mae:.3f} km/h")
        print(f"RMSE: {rmse:.3f} km/h")
        print(f"R²: {r2:.3f}")
        print(f"MAPE: {mape:.2f}%")

    # Sanity check: test should NOT be better than validation
    if metrics['test']['mae'] < metrics['val']['mae'] * 0.9:
        print("\n⚠️ WARNING: Test MAE is suspiciously better than validation!")
        print("Possible issues: data leakage, small test set, or overfitting to val")

    return metrics

if __name__ == "__main__":
    checkpoint = "outputs/stmgt_v2_20251101_215205/best_model.pt"
    config = "configs/training_config.json"

    metrics = comprehensive_evaluation(checkpoint, config)

    # Save metrics
    import json
    with open("outputs/proper_evaluation_results.json", "w") as f:
        json.dump(metrics, indent=2, fp=f)
```

**Acceptance Criteria:**

- [ ] Metrics recalculated with fresh data split
- [ ] Val/test relationship makes sense (test ≥ val usually)
- [ ] Per-horizon breakdown computed
- [ ] Results saved to `outputs/proper_evaluation_results.json`
- [ ] If metrics change significantly, update README and docs

---

### Task 2.3: Cross-Validation (6 hours)

**Goal:** Get robust performance estimates with confidence intervals.

**Method:** Time-series cross-validation (not k-fold - would leak future data!)

```python
# Script: scripts/analysis/cross_validation.py

import numpy as np
from pathlib import Path
import pandas as pd
from datetime import timedelta

def time_series_cv(n_splits=5):
    """
    Time-series cross-validation with expanding window.

    Example with 5 splits on 16K samples:

    Split 1: Train [0:3K]    Val [3K:4K]   Test [4K:5K]
    Split 2: Train [0:6K]    Val [6K:7K]   Test [7K:8K]
    Split 3: Train [0:9K]    Val [9K:10K]  Test [10K:11K]
    Split 4: Train [0:12K]   Val [12K:13K] Test [13K:14K]
    Split 5: Train [0:15K]   Val [15K:16K] Test [16K:17K] (final)
    """

    df = pd.read_parquet("data/processed/all_runs_combined.parquet")
    df = df.sort_values('timestamp')  # Ensure temporal order

    n = len(df)
    split_size = n // (n_splits + 1)

    cv_results = []

    for i in range(1, n_splits + 1):
        train_end = split_size * (i + 1)
        val_end = train_end + split_size
        test_end = val_end + split_size

        if test_end > n:
            break

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:test_end]

        print(f"\n=== Split {i}/{n_splits} ===")
        print(f"Train: {len(train_df)} samples")
        print(f"Val: {len(val_df)} samples")
        print(f"Test: {len(test_df)} samples")

        # Train model on this split
        # (Simplified - actual training would go here)
        # For now, evaluate existing model on different test sets

        # ... training code ...

        cv_results.append({
            'split': i,
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'val_mae': None,  # Fill from training
            'test_mae': None,
            'val_r2': None,
            'test_r2': None
        })

    # Calculate mean and std across splits
    val_maes = [r['val_mae'] for r in cv_results if r['val_mae'] is not None]
    test_maes = [r['test_mae'] for r in cv_results if r['test_mae'] is not None]

    print("\n=== Cross-Validation Summary ===")
    print(f"Validation MAE: {np.mean(val_maes):.3f} ± {np.std(val_maes):.3f} km/h")
    print(f"Test MAE: {np.mean(test_maes):.3f} ± {np.std(test_maes):.3f} km/h")

    return cv_results

if __name__ == "__main__":
    results = time_series_cv(n_splits=5)

    import json
    with open("outputs/cross_validation_results.json", "w") as f:
        json.dump(results, indent=2, fp=f)
```

**Note:** Full CV requires retraining 5 times - this is time-consuming but necessary for reliable estimates.

**Acceptance Criteria:**

- [ ] 5-fold time-series CV implemented
- [ ] Mean ± std reported for all metrics
- [ ] Confidence intervals computed (95% CI)
- [ ] Results show consistent performance across splits
- [ ] Document in `docs/MODEL_EVALUATION.md`

---

### Task 2.4: Overfitting Mitigation (8 hours)

**Current Risk:** 16K samples vs 267K parameters = 0.06× ratio (need >10×)

**Strategies to Implement:**

#### 2.4.1 Increase Dropout (1 hour)

```python
# In traffic_forecast/models/stmgt/model.py

# Current: dropout=0.2
# Try: dropout=0.3

class ParallelSTBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, dropout=0.3, drop_edge_rate=0.2):  # Increase to 0.3
        super().__init__()
        self.dropout = dropout  # Apply more aggressively
```

Test with `dropout=0.25, 0.3, 0.35` and compare validation curves.

#### 2.4.2 Add DropEdge (2 hours)

```python
# Randomly drop graph edges during training to prevent over-reliance on specific connections

def drop_edge(edge_index, edge_attr=None, p=0.2, training=True):
    """
    Randomly drop edges from graph.

    Args:
        edge_index: (2, num_edges) edge indices
        p: probability of dropping each edge
        training: only drop during training
    """
    if not training or p == 0:
        return edge_index, edge_attr

    mask = torch.rand(edge_index.size(1)) > p
    edge_index = edge_index[:, mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return edge_index, edge_attr

# Use in forward pass:
if self.training:
    edge_index, _ = drop_edge(edge_index, p=self.drop_edge_rate)
```

Test with `drop_edge_rate=0.1, 0.15, 0.2`.

#### 2.4.3 Aggressive Data Augmentation (3 hours)

Current augmentation might be insufficient. Add:

```python
# In traffic_forecast/augmentation/

# 1. Gaussian noise injection (already have, but increase?)
# 2. Speed value jittering
# 3. Temporal shifts
# 4. Node masking (randomly mask some nodes during training)

def augment_batch(batch, config):
    """Apply multiple augmentation techniques."""

    # 1. Add noise to speeds
    if config.noise_std > 0:
        noise = torch.randn_like(batch['traffic']) * config.noise_std
        batch['traffic'] = batch['traffic'] + noise

    # 2. Random temporal jitter (shift by ±1 timestep)
    if config.temporal_jitter and torch.rand(1) > 0.5:
        shift = torch.randint(-1, 2, (1,)).item()
        if shift != 0:
            batch['traffic'] = torch.roll(batch['traffic'], shifts=shift, dims=2)

    # 3. Node masking (randomly zero out some nodes)
    if config.node_mask_prob > 0:
        mask = torch.rand(batch['traffic'].size(1)) > config.node_mask_prob
        batch['traffic'][:, ~mask, :, :] = 0

    return batch
```

#### 2.4.4 Early Stopping Tuning (1 hour)

```python
# More aggressive early stopping

early_stopping = EarlyStopping(
    patience=5,  # Reduce from 10 to 5
    min_delta=0.001,  # Stricter improvement threshold
    monitor='val_loss'  # Monitor NLL loss, not just MAE
)
```

#### 2.4.5 Weight Decay (1 hour)

```python
# Increase L2 regularization

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=1e-3  # Increase from 1e-4
)
```

**Experiment Grid:**

| Experiment | Dropout | DropEdge | Weight Decay | Augmentation | Expected Result     |
| ---------- | ------- | -------- | ------------ | ------------ | ------------------- |
| Baseline   | 0.2     | 0.05     | 1e-4         | Current      | Current performance |
| Exp1       | 0.3     | 0.05     | 1e-4         | Current      | Less overfitting    |
| Exp2       | 0.3     | 0.15     | 1e-4         | Current      | More robust         |
| Exp3       | 0.3     | 0.15     | 1e-3         | Aggressive   | Best generalization |

**Acceptance Criteria:**

- [ ] All experiments logged to `outputs/overfitting_experiments/`
- [ ] Training curves plotted (train vs val loss)
- [ ] Best config identified
- [ ] Val/test gap reduced to <15%
- [ ] No overfitting in final 10 epochs

---

### Task 2.5: Ablation Study (12 hours)

**Goal:** Understand which components contribute to performance.

**Ablations to Test:**

```python
# 1. Remove weather cross-attention
model_no_weather = STMGT(use_weather_attention=False)

# 2. Remove parallel blocks (use sequential instead)
model_sequential = STMGT(parallel_st=False)

# 3. Remove Gaussian mixture (use single Gaussian)
model_single_gaussian = STMGT(num_mixture_components=1)

# 4. Remove temporal encoding
model_no_temporal = STMGT(use_temporal_encoding=False)

# 5. Simpler graph network (GCN instead of GAT)
model_gcn = STMGT(spatial_layer='gcn')

# 6. No attention (use simple averaging)
model_no_attention = STMGT(use_attention=False)
```

**Ablation Matrix:**

| Model Variant  | Components Removed | Expected MAE | Expected R² |
| -------------- | ------------------ | ------------ | ----------- |
| **Full STMGT** | None               | 2.78         | 0.79        |
| -Weather       | Weather cross-attn | +0.3-0.5     | -0.05       |
| -Parallel      | Parallel ST blocks | +0.2-0.4     | -0.03       |
| -Mixture       | Gaussian mixture   | +0.1-0.2     | -0.02       |
| -Temporal      | Temporal encoding  | +0.4-0.6     | -0.08       |
| -GAT           | GAT → GCN          | +0.2-0.3     | -0.03       |
| -Attention     | All attention      | +0.5-0.8     | -0.10       |

**Script Template:**

```python
# scripts/analysis/ablation_study.py

def run_ablation(ablation_name, model_config):
    """Train model with specific ablation."""

    print(f"\n{'='*50}")
    print(f"Running Ablation: {ablation_name}")
    print(f"{'='*50}")

    # Create model with ablation
    model = create_ablated_model(model_config)

    # Train for fewer epochs (ablation doesn't need full training)
    trainer = create_trainer(max_epochs=30)

    # Train
    trainer.fit(model, train_loader, val_loader)

    # Evaluate
    metrics = evaluate_model(model, test_loader)

    return {
        'ablation': ablation_name,
        'test_mae': metrics['mae'],
        'test_r2': metrics['r2'],
        'test_mape': metrics['mape'],
        'num_params': count_parameters(model)
    }

ablations = [
    ('full', {'all_components': True}),
    ('no_weather', {'use_weather_attention': False}),
    ('no_parallel', {'parallel_st': False}),
    ('no_mixture', {'num_mixture_components': 1}),
    ('no_temporal', {'use_temporal_encoding': False}),
    ('gcn', {'spatial_layer': 'gcn'}),
]

results = []
for name, config in ablations:
    result = run_ablation(name, config)
    results.append(result)

# Create comparison table
df = pd.DataFrame(results)
print("\n=== Ablation Study Results ===")
print(df.to_string(index=False))

# Save
df.to_csv("outputs/ablation_study_results.csv", index=False)
```

**Acceptance Criteria:**

- [ ] All 6 ablations trained and evaluated
- [ ] Results table created showing MAE/R² degradation
- [ ] Statistical significance tested (t-test)
- [ ] Visualization created (bar chart of component importance)
- [ ] Documented in `docs/ABLATION_STUDY.md`

---

### Task 2.6: Model Card Documentation (4 hours)

**Goal:** Create comprehensive model documentation following best practices.

**Create:** `docs/MODEL_CARD.md`

```markdown
# Model Card: STMGT Traffic Forecasting

## Model Details

**Model Name:** STMGT (Spatial-Temporal Multi-Modal Graph Transformer)  
**Version:** v2.0  
**Date:** November 2025  
**Developers:** THAT Le Quang (thatlq1812)  
**Model Type:** Graph Neural Network for Time Series Forecasting

### Architecture

- **Input:** 12 timesteps × 62 nodes × (1 traffic + 3 weather features)
- **Output:** 12 future timesteps × 62 nodes × (mean, std, mixture logits)
- **Parameters:** 267,392 (2.7 MB)
- **Key Components:**
  - Traffic encoder (Linear 1→96)
  - Weather encoder (Linear 3→96)
  - Temporal encoder (embeddings + cyclical)
  - 3× Parallel Spatial-Temporal blocks (GATv2 + Multi-head Attention)
  - Weather cross-attention
  - Gaussian Mixture Head (K=3)

## Intended Use

**Primary Use:** Traffic speed forecasting for Ho Chi Minh City road network

**Intended Users:**

- City traffic management agencies
- Navigation app developers
- Urban planning researchers
- Transportation engineers

**Out-of-Scope:**

- Real-time accident detection
- Long-term (>3 hours) forecasting
- Other cities without retraining
- Individual vehicle routing

## Training Data

**Dataset:** HCMC Traffic Dataset (October 2025)

- **Size:** 16,328 samples
- **Nodes:** 62 major intersections
- **Edges:** 144 road segments
- **Features:**
  - Traffic: speed (km/h)
  - Weather: temperature, precipitation, wind speed
  - Temporal: hour of day, day of week, weekend flag
- **Collection:** Google Directions API (real-world data)
- **Split:** 70% train / 15% val / 15% test (temporal split)

**Data Limitations:**

- Single city (Ho Chi Minh City only)
- Single month (October 2025)
- May not generalize to:
  - Different seasons (e.g., rainy vs dry)
  - Special events (festivals, holidays)
  - Infrastructure changes (new roads)

## Performance

### Validation Set (1,500 samples)

| Metric | Value          | Interpretation                    |
| ------ | -------------- | --------------------------------- |
| MAE    | 3.5 ± 0.4 km/h | Average error in speed prediction |
| RMSE   | 5.2 ± 0.6 km/h | Emphasizes larger errors          |
| R²     | 0.52 ± 0.08    | Explains 52% of variance          |
| MAPE   | 25 ± 5%        | Relative error percentage         |

### Test Set (1,500 samples)

| Metric | Value          | Interpretation                 |
| ------ | -------------- | ------------------------------ |
| MAE    | 3.8 ± 0.5 km/h | Slightly worse than validation |
| R²     | 0.48 ± 0.10    | Consistent with validation     |

### Per-Horizon Performance

| Horizon | Time Ahead | MAE | R²   |
| ------- | ---------- | --- | ---- |
| 1       | 15 min     | 2.5 | 0.65 |
| 4       | 1 hour     | 3.2 | 0.58 |
| 8       | 2 hours    | 4.1 | 0.50 |
| 12      | 3 hours    | 5.0 | 0.42 |

## Limitations

### Model Limitations

1. **Data Scarcity:** Only 16K samples for 267K parameters (overfitting risk)
2. **Temporal Coverage:** Single month - may not capture seasonal patterns
3. **Spatial Coverage:** 62 nodes - limited to major roads only
4. **Uncertainty Quantification:** Mixture model helps, but calibration not verified

### Known Failure Modes

1. **Sudden Events:** Cannot predict accidents, road closures, protests
2. **Extreme Weather:** Limited training data for heavy rain/flooding
3. **Weekend Patterns:** Less training data for weekends (collection bias)
4. **Night Hours:** Fewer samples → less reliable predictions

### Biases

1. **Temporal Bias:** More data from weekday peak hours
2. **Spatial Bias:** Urban core over-represented vs suburbs
3. **Weather Bias:** Mostly dry weather (October in HCMC)

## Ethical Considerations

### Potential Harms

- **Traffic Redistribution:** If widely adopted, could shift congestion rather than reduce it
- **Privacy:** Node locations could be correlated with sensitive areas
- **Equity:** May optimize for high-traffic areas, neglecting underserved neighborhoods

### Mitigation Strategies

- Predictions aggregated to intersection level (no individual tracking)
- Model outputs probabilities, not deterministic routes
- Regular retraining to adapt to changing patterns

## Recommendations

### When to Use

✅ Traffic planning and forecasting  
✅ Research on traffic patterns  
✅ Benchmarking other models  
✅ Educational demonstrations

### When NOT to Use

❌ Safety-critical real-time decisions (use with human oversight)  
❌ Legal/contractual commitments (predictions not guaranteed)  
❌ Other cities without domain adaptation  
❌ Long-term infrastructure planning (>1 year)

## Maintenance

**Retraining Schedule:** Every 1-2 weeks with fresh data  
**Monitoring:** Track validation MAE; retrain if >10% degradation  
**Updates:** Model versioning with semantic versioning (v2.0, v2.1, etc.)

## References

[Include relevant papers, especially Graph WaveNet, MTGNN, etc.]

## Contact

**Maintainer:** THAT Le Quang  
**GitHub:** thatlq1812  
**Project:** DSP391m Traffic Forecasting
```

**Acceptance Criteria:**

- [ ] Model card follows standard format
- [ ] All sections filled with accurate info
- [ ] Limitations clearly stated
- [ ] Ethical considerations addressed
- [ ] Linked from main README

---

## Phase 2 Success Criteria

✅ **Test/validation discrepancy resolved**  
✅ **Cross-validation completed (5 folds)**  
✅ **Overfitting mitigation strategies tested**  
✅ **Ablation study shows component importance**  
✅ **Model card documented**  
✅ **R² stable at 0.45-0.55 with verified metrics**

---

## Deliverables

1. `outputs/proper_evaluation_results.json` - Verified metrics
2. `outputs/cross_validation_results.json` - CV results with confidence intervals
3. `outputs/overfitting_experiments/` - Grid search results
4. `outputs/ablation_study_results.csv` - Component importance
5. `docs/MODEL_CARD.md` - Comprehensive model documentation
6. `docs/ABLATION_STUDY.md` - Detailed ablation analysis

---

## Next Steps

After Phase 2 completion:

1. Update CHANGELOG.md with model improvements
2. Incorporate findings into Report 4
3. Begin Phase 3 - Production Hardening

---

## Time Tracking

| Task                       | Estimated | Actual | Notes |
| -------------------------- | --------- | ------ | ----- |
| 2.1 Investigate Metrics    | 4h        |        |       |
| 2.2 Proper Evaluation      | 3h        |        |       |
| 2.3 Cross-Validation       | 6h        |        |       |
| 2.4 Overfitting Mitigation | 8h        |        |       |
| 2.5 Ablation Study         | 12h       |        |       |
| 2.6 Model Card             | 4h        |        |       |
| **Total**                  | **37h**   |        |       |

Spread across 1 week = 5-7 hours per day (intense but doable).

---

**Note:** This phase is the most technically rigorous. Take your time and don't rush the validation - bad metrics will undermine everything else! 🔬
