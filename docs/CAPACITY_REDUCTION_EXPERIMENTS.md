# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# STMGT Capacity Experiments Summary

**Date:** November 10, 2025  
**Finding:** 680K params (V1) is optimal. Larger models overfit, need to test SMALLER models.

---

## Experimental Results

### Capacity Scaling Experiments (COMPLETED)

| Model  | Params   | Change       | Test MAE | Test R²  | Coverage@80 | Status          |
| ------ | -------- | ------------ | -------- | -------- | ----------- | --------------- |
| **V1** | **680K** | **baseline** | **3.08** | **0.82** | **83.75%**  | **OPTIMAL** ✓   |
| V1.5   | 850K     | +25%         | 3.18     | 0.804    | 84.14%      | WORSE           |
| V2     | 1.15M    | +69%         | 3.22     | 0.796    | 84.09%      | WORSE, OVERFITS |

**Key Finding:** ALL capacity increases performed WORSE than V1. Direction should be REDUCING capacity.

---

## New Experiments: Capacity Reduction

### Hypothesis

V1 (680K) may still be TOO LARGE for 205K samples. Test smaller capacities to find true optimal.

### New Configs (TO TEST)

#### 1. V0.9 - Ablation K=3 (600K, -12%)

```bash
cd /d/UNI/DSP391m/project
conda run -n dsp --no-capture-output python scripts/training/train_stmgt.py --config configs/train_v0.9_ablation_k3.json
```

**Changes:**

- SAME architecture (hidden=96, blocks=3, heads=4)
- Only change: K=5 → K=3 mixtures (-40%)
- Params: 680K → 600K (-12%)

**Expected:**

- MAE: 3.08-3.15 (similar to V1)
- Coverage@80: 79-82% (may be 2-4% worse)
- Purpose: Isolate mixture component impact

**Hypothesis:** If MAE similar but Coverage worse → K=5 is justified. If both similar → K=3 sufficient.

---

#### 2. V0.8 - Smaller Model (520K, -23%)

```bash
cd /d/UNI/DSP391m/project
conda run -n dsp --no-capture-output python scripts/training/train_stmgt.py --config configs/train_v0.8_smaller.json
```

**Changes:**

- hidden_dim: 96 → 88 (-8.3%)
- mixture K: 5 → 4 (-20%)
- Params: 680K → 520K (-23%)
- dropout: 0.2 → 0.15, drop_edge: 0.1 → 0.08

**Expected:**

- MAE: 3.05-3.15 (may be better!)
- Best epoch: 12-18 (later than V1's epoch 9)
- Parameter/sample ratio: 0.28 (better than V1's 0.21)

**Hypothesis:** Smaller model with better data/param ratio may generalize better and NOT overfit early.

---

#### 3. V0.6 - Minimal Model (350K, -48%)

```bash
cd /d/UNI/DSP391m/project
conda run -n dsp --no-capture-output python scripts/training/train_stmgt.py --config configs/train_v0.6_minimal.json
```

**Changes:**

- hidden_dim: 96 → 80 (-16.7%)
- num_blocks: 3 → 2 (-33%)
- mixture K: 5 → 3 (-40%)
- Params: 680K → 350K (-48%)
- dropout: 0.2 → 0.1, drop_edge: 0.1 → 0.05

**Expected:**

- MAE: 3.10-3.25 (may underfit slightly)
- Best epoch: 15-25 (late convergence, no overfitting)
- Parameter/sample ratio: 0.41 (2× samples per param vs V1)

**Hypothesis:** Test lower bound. May underfit but establish minimum viable capacity.

---

## Training Priority

### Recommended Order:

1. **V0.9 (600K)** - Safest, clean ablation study
2. **V0.8 (520K)** - Main experiment, likely sweet spot
3. **V0.6 (350K)** - Lower bound exploration

### Expected Outcomes:

**Scenario A: V0.8 or V0.9 better than V1**

- Finding: V1 was still too large!
- Action: Use smaller model for production
- Conclusion: Optimal capacity is 520K-600K for 205K samples

**Scenario B: All similar to V1 (MAE 3.05-3.15)**

- Finding: Wide optimal range (350K-680K)
- Action: Use smallest model (faster inference)
- Conclusion: Model capacity not critical factor

**Scenario C: All worse than V1 (MAE > 3.15)**

- Finding: V1 (680K) is the sweet spot
- Action: Keep V1 for production
- Conclusion: 680K params is optimal for 205K samples

---

## Scientific Value

This experiment tests **both directions**:

- ✓ Capacity UP: V1.5 (+25%), V2 (+69%) → WORSE
- ? Capacity DOWN: V0.9 (-12%), V0.8 (-23%), V0.6 (-48%) → TO TEST

Will establish **optimal capacity range** through systematic exploration.

---

## Monitoring Checklist

During training, watch for:

### 1. Best Epoch Location

- V1: Best epoch 9 (early, slight overfit)
- V1.5: Best epoch ? (need to check)
- V2: Best epoch 4 (very early, severe overfit)
- **Expected for smaller models:** Best epoch 12-20 (later = less overfitting)

### 2. Train/Val Gap

- V1: ~5% (healthy)
- V2: 34.4% (severe overfitting)
- **Expected for smaller models:** < 5% (better generalization)

### 3. MAE Trajectory

- **Good sign:** Smooth convergence, late best epoch
- **Bad sign:** Early peak, then degradation

### 4. Coverage@80

- V1: 83.75% (well-calibrated)
- Target: 80-85% (slightly worse OK for smaller K)

---

## Quick Start

### Run all experiments sequentially:

```bash
cd /d/UNI/DSP391m/project
bash scripts/training/run_capacity_experiments.sh
```

### Run individual experiments:

```bash
# V0.9 (600K, -12%)
conda run -n dsp --no-capture-output python scripts/training/train_stmgt.py --config configs/train_v0.9_ablation_k3.json

# V0.8 (520K, -23%)
conda run -n dsp --no-capture-output python scripts/training/train_stmgt.py --config configs/train_v0.8_smaller.json

# V0.6 (350K, -48%)
conda run -n dsp --no-capture-output python scripts/training/train_stmgt.py --config configs/train_v0.6_minimal.json
```

---

## Expected Timeline

- V0.9 (600K): ~9-10 hours
- V0.8 (520K): ~9 hours
- V0.6 (350K): ~7 hours

**Total:** ~25-26 hours for all 3 experiments

---

## Files Created

**Configs:**

- `configs/train_v0.9_ablation_k3.json` (600K params)
- `configs/train_v0.8_smaller.json` (520K params)
- `configs/train_v0.6_minimal.json` (350K params)

**Scripts:**

- `scripts/training/run_capacity_experiments.sh` (automated training)

**Removed configs (failed experiments):**

- ~~train_v1.5_capacity.json~~ (worse than V1)
- ~~train_v1_arch_improvements.json~~ (not tested, risky)
- ~~train_v1_heavy_reg.json~~ (likely to overfit)
- ~~train_v1_deeper.json~~ (likely to overfit)
- ~~train_v1_uncertainty_focused.json~~ (not priority)
- ~~train_v1_ablation_no_weather.json~~ (defer to later)

---

**Status:** Ready to run capacity reduction experiments  
**Next Action:** Start with V0.9 (safest ablation study)
