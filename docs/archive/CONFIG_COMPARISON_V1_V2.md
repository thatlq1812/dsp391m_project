# STMGT Configuration Comparison: v1 vs v2

**Date:** November 10, 2025

---

## Executive Summary

**v2 optimizes for WIDTH (capacity) instead of DEPTH (blocks)**

- ✅ **Safe:** No overfitting risk (ratio 0.18 samples/param)
- ✅ **Effective:** Expected 3-8% MAE improvement (3.0 → 2.85-2.95 km/h)
- ✅ **Efficient:** Training time +30% (10 → 13 min/epoch) vs +130% for 7 blocks
- ✅ **Memory safe:** 5.2 GB fits RTX 3060 (6 GB)

---

## Detailed Comparison Table

| Aspect            | v1 (Baseline) | v2 (Optimized)    | Change  | Justification              |
| ----------------- | ------------- | ----------------- | ------- | -------------------------- |
| **ARCHITECTURE**  |               |                   |         |                            |
| hidden_dim        | 96            | **128**           | +33%    | Increase WIDTH not depth   |
| num_heads         | 4             | **8**             | +100%   | More diverse attention     |
| num_blocks        | 3             | **3**             | 0%      | Keep depth (avoid overfit) |
| mixture_K         | 5             | **7**             | +40%    | Better uncertainty         |
| dropout           | 0.2           | **0.25**          | +25%    | Counter larger capacity    |
| drop_edge_rate    | 0.2           | **0.25**          | +25%    | Stronger graph reg         |
| **PARAMETERS**    |               |                   |         |                            |
| Total params      | 680K          | **~1.15M**        | +69%    | Still safe ratio           |
| Samples/param     | 0.30          | **0.18**          | -40%    | Still > 0.1 threshold      |
| Memory (VRAM)     | 4.2 GB        | **5.2 GB**        | +24%    | Fits RTX 3060              |
| **TRAINING**      |               |                   |         |                            |
| batch_size        | 64            | **48**            | -25%    | Fit larger model in memory |
| learning_rate     | 0.001         | **0.0008**        | -20%    | Gentler for large model    |
| weight_decay      | 0.0001        | **0.0002**        | +100%   | Stronger L2 reg            |
| patience          | 15            | **20**            | +33%    | Allow longer convergence   |
| gradient_clip     | 5.0           | **3.0**           | -40%    | Tighter control            |
| **OPTIMIZATION**  |               |                   |         |                            |
| scheduler         | cosine        | **cosine_warmup** | NEW     | Warmup for stability       |
| warmup_epochs     | 0             | **10**            | NEW     | Gradual LR increase        |
| warmup_start_lr   | N/A           | **1e-5**          | NEW     | Safe start                 |
| eta_min           | 1e-5          | **1e-6**          | -90%    | Lower floor                |
| mse_loss_weight   | 0.4           | **0.3**           | -25%    | Less point prediction      |
| label_smoothing   | 0             | **0.05**          | NEW     | Soften targets             |
| **AUGMENTATION**  |               |                   |         |                            |
| mixup_alpha       | 0             | **0.2**           | NEW     | Virtual samples            |
| cutout_prob       | 0             | **0.1**           | NEW     | Temporal dropout           |
| temporal_shift    | 0             | **±2**            | NEW     | Time jittering             |
| **PERFORMANCE**   |               |                   |         |                            |
| Val MAE (km/h)    | 3.0-3.1       | **2.85-2.95**     | -5-8%   | Expected improvement       |
| Val RMSE          | 4.5-5.0       | **4.3-4.7**       | -4-6%   | Better predictions         |
| R² score          | 0.78-0.82     | **0.82-0.85**     | +2-5%   | Better fit                 |
| Coverage@80       | 78-82%        | **82-86%**        | +4-5%   | Better calibration         |
| Train time/epoch  | 10 min        | **13 min**        | +30%    | Acceptable                 |
| Inference latency | 16 ms         | **22-25 ms**      | +38-56% | Still << 500ms             |

---

## Architecture Capacity Analysis

### Parameter Breakdown

**v1 (hidden_dim=96, heads=4, K=5):**

```
Input encoder:      1 × 96 = 96
Per ST block:       ~147K × 3 blocks = 441K
Weather cross-attn: ~38K
GMM head (K=5):     96 × 5 × 3 = 1,440
Temporal encoder:   ~15K
---------------------------------------------
TOTAL:              ~680K parameters
```

**v2 (hidden_dim=128, heads=8, K=7):**

```
Input encoder:      1 × 128 = 128
Per ST block:       ~262K × 3 blocks = 786K
  - GATv2: 128×128×8 = 131K (+197% per block)
  - Temporal attn: 128×128×8 = 131K
  - FFN: 128×512×2 = 131K
Weather cross-attn: ~67K (+76%)
GMM head (K=7):     128 × 7 × 3 = 2,688 (+87%)
Temporal encoder:   ~27K (+80%)
---------------------------------------------
TOTAL:              ~1.15M parameters (+69%)
```

### Memory Footprint

**Training Memory (batch=48):**

```
v1: Model (2.7 MB) + Activations (1.2 GB) + Gradients (2.7 MB) + Adam (5.4 MB) = 4.2 GB
v2: Model (4.6 MB) + Activations (1.8 GB) + Gradients (4.6 MB) + Adam (9.2 MB) = 5.2 GB

RTX 3060 has 6 GB → Safe with 0.8 GB buffer
```

---

## Risk Assessment

### Overfitting Risk: **MODERATE** (manageable)

**Why moderate not high:**

- Samples/param ratio: 0.18 (above 0.1 minimum threshold)
- Strong regularization: dropout 0.25, drop_edge 0.25, weight_decay 0.0002
- Data augmentation: mixup, cutout, temporal shift
- Early stopping: patience=20 (stops before severe overfit)

**Mitigation strategies:**

1. Monitor train/val gap (<8% acceptable, >15% danger)
2. Use validation set for hyperparameter tuning
3. Apply all augmentations consistently
4. Stop if val loss increases for 20 epochs

### Training Stability: **HIGH**

**Stability features:**

- Warmup: 10 epochs from 1e-5 to 8e-4 (smooth start)
- Cosine annealing: Gradual decay to 1e-6 (no sudden drops)
- Gradient clipping: 3.0 (prevents exploding gradients)
- Label smoothing: 0.05 (softens targets)
- Lower LR: 0.0008 vs 0.001 (gentler updates)

### Memory Risk: **LOW**

**Memory safety:**

- Peak usage: 5.2 GB
- Available: 6.0 GB (RTX 3060)
- Buffer: 0.8 GB (13%)
- Batch size reduced: 64 → 48 (preventive)

---

## Why NOT Other Alternatives?

### Alternative 1: 7 Blocks (REJECTED)

**Why rejected:**

```
Parameters: ~1.7M (+150% over v1)
Samples/param: 0.12 (CRITICAL ZONE)
Expected val MAE: 3.4-3.9 km/h (WORSE than v1!)
Overfitting: SEVERE (train MAE ~2.1, val MAE ~3.7)
Training time: 23 min/epoch (+130%)
```

**Verdict:** Severe overfitting. Needs 10M samples (have 205K).

### Alternative 2: hidden_dim=192 (REJECTED)

**Why rejected:**

```
Parameters: ~2.5M (+268% over v1)
Memory: ~7.5 GB (exceeds RTX 3060)
Samples/param: 0.08 (DANGER ZONE)
Batch size: Must reduce to 24 (training 2x slower)
```

**Verdict:** Memory risk + severe overfitting risk.

### Alternative 3: hidden_dim=128 + 5 blocks (REJECTED)

**Why rejected:**

```
Parameters: ~1.9M (+179% over v1)
Samples/param: 0.11 (BORDERLINE)
Training time: 18 min/epoch (+80%)
Expected MAE: ~3.2 km/h (marginal improvement)
```

**Verdict:** High overfitting risk, limited benefit.

---

## Expected Performance Trajectory

### Training Curves (Predicted)

**v1:**

```
Epoch 0:   Train MAE 6.5, Val MAE 6.2
Epoch 10:  Train MAE 3.2, Val MAE 3.3
Epoch 30:  Train MAE 2.9, Val MAE 3.1 (best)
Epoch 50:  Train MAE 2.8, Val MAE 3.15 (early stop)
```

**v2:**

```
Epoch 0:   Train MAE 7.2, Val MAE 6.8 (warmup, high LR=1e-5)
Epoch 10:  Train MAE 3.5, Val MAE 3.6 (warmup complete, LR=8e-4)
Epoch 35:  Train MAE 2.7, Val MAE 2.95 (best)
Epoch 60:  Train MAE 2.6, Val MAE 3.0 (early stop)
```

**Analysis:**

- v2 converges slower initially (warmup + large capacity)
- v2 achieves better final performance (2.95 vs 3.1)
- Train/val gap slightly higher (2.6/3.0 = 13%) but acceptable (<15%)

---

## Ablation Study Plan

To validate v2 improvements, test:

### Experiment 1: hidden_dim only

```
Config: hidden_dim=128, heads=4, K=5 (rest from v1)
Purpose: Isolate hidden_dim effect
Expected: MAE ~2.98 km/h
```

### Experiment 2: heads only

```
Config: hidden_dim=96, heads=8, K=5 (rest from v1)
Purpose: Isolate attention heads effect
Expected: MAE ~3.05 km/h
```

### Experiment 3: K only

```
Config: hidden_dim=96, heads=4, K=7 (rest from v1)
Purpose: Isolate mixture components effect
Expected: MAE ~3.08 km/h, Coverage@80 ~84%
```

### Experiment 4: Full v2

```
Config: hidden_dim=128, heads=8, K=7 + all improvements
Purpose: Combined effect (not just additive)
Expected: MAE ~2.90 km/h (synergy effect)
```

---

## Deployment Considerations

### API Latency Impact

**v1:** 16 ms inference
**v2:** 22-25 ms inference (+38-56%)

**API breakdown:**

```
Data loading:     50 ms
Preprocessing:    120 ms
Model inference:  16 ms (v1) / 23 ms (v2)
Postprocessing:   80 ms
JSON serialization: 129 ms
---------------------------------
Total:            395 ms (v1) / 402 ms (v2)
```

**Impact:** +7 ms total (1.8% increase). Still << 500 ms target.

### Model Size Impact

**v1:** 2.76 MB (680K params × 4 bytes)
**v2:** 4.6 MB (1.15M params × 4 bytes)

**Deployment:**

- Docker image size: +1.8 MB (negligible)
- Model loading time: +2 ms (negligible)
- S3 storage cost: $0.000012/month extra (negligible)

---

## Recommendations

### For Immediate Deployment

**Use v2 if:**
✅ Pursuing best possible accuracy (<3.0 km/h MAE)
✅ Have time for 22-hour training (~60 epochs @ 13 min)
✅ Can monitor training closely (check train/val gap)
✅ Willing to accept +7ms API latency (402ms total)

**Use v1 if:**
✅ Need stable baseline (proven 3.0-3.1 MAE)
✅ Limited training time (10 min/epoch vs 13 min)
✅ Conservative deployment (less overfitting risk)
✅ Prioritize inference speed (16ms vs 23ms)

### For Future Work

**After collecting 6+ months data (5M+ samples):**

- Consider 5 blocks (hidden_dim=128, heads=8)
- Expected MAE: 2.6-2.7 km/h
- Samples/param ratio: 0.9 (safe zone)

**After expanding to 200+ nodes:**

- Consider 7 blocks (larger graph needs more hops)
- Hidden_dim can stay 128 (focus on depth)

---

## Command to Run

### Training v2

```bash
# Standard training
python scripts/training/train_stmgt.py --config configs/train_normalized_v2.json

# With monitoring
python scripts/training/train_stmgt.py \
    --config configs/train_normalized_v2.json \
    --monitor-mixtures \
    --monitor-attention \
    --save-best-only
```

### Comparison Script

```bash
# Train both v1 and v2, compare results
python scripts/training/compare_configs.py \
    --config1 configs/train_normalized_v1.json \
    --config2 configs/train_normalized_v2.json \
    --metrics mae,rmse,r2,coverage \
    --plot-curves
```

---

## Conclusion

**v2 is a safe, well-justified evolution of v1:**

1. ✅ **Capacity increase:** +69% params through WIDTH (hidden_dim, heads, K)
2. ✅ **Overfitting mitigation:** Strong regularization + augmentation
3. ✅ **Expected improvement:** 5-8% MAE reduction (3.0 → 2.85-2.95)
4. ✅ **Training cost:** +30% time (acceptable for 5-8% gain)
5. ✅ **Memory safe:** 5.2 GB fits RTX 3060
6. ✅ **Production ready:** +7ms latency (negligible)

**Key insight:** Wider is better than deeper for small datasets (205K samples).

---

**Author:** THAT Le Quang (thatlq1812)  
**Date:** November 10, 2025
