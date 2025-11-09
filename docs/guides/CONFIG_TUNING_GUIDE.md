# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# STMGT Configuration Analysis & Recommendations

## Current Model Analysis

### Current Configuration (Nov 9, 2025)

```json
{
  "hidden_dim": 64,
  "num_blocks": 3,
  "num_heads": 4,
  "mixture_components": 3,
  "dropout": 0.2,
  "batch_size": 32,
  "learning_rate": 0.001
}
```

**Performance:**

- Parameters: 304,236
- Test MAE: 3.44 km/h
- R² Score: 0.76
- Training Time: ~7 min/epoch (100 epochs ≈ 12 hours)

### Hidden Dim Impact Analysis

**Parameter Count vs Hidden Dim:**

- hidden_dim=64: 304K params
- hidden_dim=96: ~680K params (+123%)
- hidden_dim=128: ~1.2M params (+294%)

**Expected Impact:**

| Aspect           | hidden_dim=64 | hidden_dim=96   | hidden_dim=128  |
| ---------------- | ------------- | --------------- | --------------- |
| Capacity         | Base          | +50%            | +100%           |
| Parameters       | 304K          | 680K            | 1.2M            |
| Memory           | 1.2 MB        | 2.7 MB          | 4.8 MB          |
| Training Time    | 7 min/epoch   | 10-12 min/epoch | 15-18 min/epoch |
| Expected MAE     | 3.44          | 3.2-3.3         | 3.1-3.2         |
| Overfitting Risk | Low           | Medium          | High            |

**Recommendation:**

- **Current (64)**: Good balance for 62 nodes, 144 edges
- **Upgrade to 96**: Reasonable if aiming for <3.3 MAE
- **128+**: Likely overkill, risk overfitting with only 1430 runs

### Other Hyperparameter Tuning

**High Impact Changes:**

1. **Learning Rate Schedule**

   - Current: Fixed 0.001
   - Recommendation: Cosine annealing or OneCycleLR
   - Expected: -5% MAE improvement

   ```python
   scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
       optimizer, T_max=100, eta_min=1e-5
   )
   ```

2. **Batch Size**

   - Current: 32
   - Try: 64 (faster training, similar accuracy)
   - Expected: -15% training time, ±0.1 MAE

3. **Mixture Components**

   - Current: K=3 (tri-modal)
   - Try: K=5 (more expressive)
   - Expected: -0.1 to -0.2 MAE, better uncertainty

4. **Data Augmentation**
   - Already using extreme augmentation ✓
   - Could add: Mixup between runs
   - Expected: -0.1 MAE, better generalization

**Low Impact Changes:**

- num_blocks: 3 is optimal (4+ gives diminishing returns)
- num_heads: 4 is good (2-8 range tested)
- dropout: 0.2 appropriate

### Recommended Next Config

```json
{
  "model": {
    "hidden_dim": 96,
    "num_blocks": 3,
    "num_heads": 4,
    "mixture_components": 5,
    "dropout": 0.2
  },
  "training": {
    "batch_size": 64,
    "learning_rate": 0.001,
    "scheduler": "cosine",
    "scheduler_params": {
      "T_max": 100,
      "eta_min": 1e-5
    }
  }
}
```

**Expected Results:**

- Test MAE: 3.1-3.2 km/h (vs current 3.44)
- Training Time: 8-10 min/epoch (vs 7 min)
- Total Time: ~15 hours (vs 12 hours)
- ROI: 7% MAE improvement for 25% more time

### Cost-Benefit Analysis

| Config           | MAE     | Training Time | Worth It?           |
| ---------------- | ------- | ------------- | ------------------- |
| Current (64)     | 3.44    | 12h           | ✓ Baseline          |
| hidden_96        | 3.2-3.3 | 15h           | ✓ Good ROI          |
| hidden_128       | 3.1-3.2 | 20h           | ⚠️ Marginal         |
| hidden_96 + K=5  | 3.0-3.1 | 18h           | ✓✓ Best ROI         |
| hidden_128 + K=5 | 2.9-3.0 | 25h           | ❌ Overfitting risk |

## Conclusion

**Immediate Recommendations:**

1. **Retrain with fixed normalization first** (most important!)
2. **If <3.3 MAE needed**: Upgrade to hidden_dim=96
3. **If <3.1 MAE needed**: hidden_dim=96 + K=5 mixtures + cosine LR
4. **Don't go beyond 128**: Overfitting risk with current data size

**Data Collection Priority:**

- More training data (2000+ runs) would justify hidden_dim=128+
- Current 1430 runs optimal for hidden_dim=64-96
