# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# GraphWaveNet Performance Verification Plan

**Purpose:** Kiểm thử thực tế để xác định performance claim (MAE 0.91 km/h) của hunglm có đáng tin hay không

**Date:** November 12, 2025  
**Status:** PENDING VERIFICATION

---

## 🎯 MỤC TIÊU KIỂM THỬ

### Câu hỏi cần trả lời:

1. ✅ **MAE 0.91 km/h có thật không?**
   - Có thể reproduce được không?
   - Data có bị leak không?
   - Metrics có được tính đúng không?

2. ✅ **So sánh công bằng:**
   - Performance của hunglm vs current project
   - Cùng dataset, cùng split, cùng evaluation

3. ✅ **Lesson learned:**
   - Data augmentation có hiệu quả như claim không?
   - Architecture nào tốt hơn?
   - Có gì học được từ implementation của hunglm?

---

## 📋 VERIFICATION STRATEGY

### Strategy A: Quick Verification (Recommended - 2 giờ)

**Kiểm tra nhanh với current codebase:**

```bash
# 1. Branch mới cho verification
git checkout -b verify-hunglm-performance

# 2. Test current GraphWaveNet implementation với data khác nhau
# a) Test với baseline data (no augmentation)
python scripts/training/train_graphwavenet_baseline.py \
  --dataset data/processed/all_runs_combined.parquet \
  --output-dir outputs/gwn_verify_baseline \
  --epochs 50

# b) Test với augmented data (nếu có)
python scripts/training/train_graphwavenet_baseline.py \
  --dataset data/processed/all_runs_extreme_augmented.parquet \
  --output-dir outputs/gwn_verify_augmented \
  --epochs 50
```

**Expected outcomes:**

- Baseline: MAE ~11-15 km/h (realistic)
- Augmented: MAE ~10-13 km/h (if augmentation helps)
- **KHÔNG nên thấy 0.91 km/h** với data thật

### Strategy B: Deep Dive (Thorough - 1-2 ngày)

**Reproduce hunglm's exact setup:**

#### Phase 1: Setup hunglm's environment (2 hours)

```bash
cd archive/experimental/Traffic-Forecasting-GraphWaveNet

# Check if requirements work
cat requirements.txt

# Try to setup (may need fixing)
python -m venv venv_hunglm
source venv_hunglm/bin/activate  # Windows: venv_hunglm\Scripts\activate
pip install -r requirements.txt
```

#### Phase 2: Inspect hunglm's data (1 hour)

```bash
# Check what data exists
ls -lh data/runs/
ls -lh data/runs_holdout_test/

# Count samples
python -c "
import pandas as pd
df = pd.read_parquet('data/processed/train_val_combined.parquet')
print(f'Train/Val samples: {len(df)}')
print(f'Unique timestamps: {df[\"timestamp\"].nunique()}')
print(f'Speed range: {df[\"speed_kmh\"].min():.1f} - {df[\"speed_kmh\"].max():.1f}')
print(f'Speed mean: {df[\"speed_kmh\"].mean():.1f} km/h')
print(f'Speed std: {df[\"speed_kmh\"].std():.1f} km/h')
"

# RED FLAGS to look for:
# - Speed std < 3 km/h (too smooth, likely preprocessed)
# - Speed range < 20 km/h (not realistic traffic)
# - Very few unique timestamps (not enough data)
```

#### Phase 3: Review training code (1 hour)

```python
# Check train.py for suspicious patterns:

# RED FLAG 1: Using test data in normalization
scaler = StandardScaler()
scaler.fit(all_data)  # ❌ BAD! Should only fit on train

# RED FLAG 2: No proper train/test split
model.fit(X, y)  # ❌ No split!

# RED FLAG 3: Using augmented test set
test_data = pd.read_parquet('test_augmented.parquet')  # ❌ Test shouldn't be augmented

# RED FLAG 4: Evaluation on validation instead of test
final_mae = val_mae  # ❌ Should be test_mae

# RED FLAG 5: Normalized metrics reported as km/h
mae_loss = 0.0071  # Loss in normalized scale
print(f"MAE: {mae_loss} km/h")  # ❌ Wrong! Should denormalize first
```

#### Phase 4: Run verification experiments (3-6 hours)

**Experiment 1: Current data with hunglm's architecture**

```bash
# Port hunglm's exact architecture to our TensorFlow code
# Train on our data/processed/all_runs_combined.parquet
# Expected: MAE 10-15 km/h (realistic)
```

**Experiment 2: Check if data is too easy**

```python
# Simple baseline: Previous speed
df['pred'] = df.groupby('edge_id')['speed_kmh'].shift(1)
mae = (df['speed_kmh'] - df['pred']).abs().mean()
print(f"Naive baseline MAE: {mae:.2f} km/h")

# If naive MAE < 2 km/h → Data is TOO SMOOTH/PREDICTABLE
# If MAE 0.91 beats naive by <1 km/h → Model not adding much value
```

**Experiment 3: Check normalization math**

```python
# Load hunglm's scaler
mean = scaler.mean_  # or from saved npz
std = scaler.std_

# Check if metrics make sense
val_loss = 0.0071  # From report
denormalized_mae = val_loss * std + mean  # If they normalized with StandardScaler

print(f"If val_loss is normalized MAE:")
print(f"  Mean: {mean:.2f} km/h")
print(f"  Std: {std:.2f} km/h")
print(f"  Denormalized MAE: {denormalized_mae:.2f} km/h")

# Expected: std = 5-10 km/h for normal traffic
# If std > 50 → Something wrong with data
# If std < 2 → Data too smooth
```

---

## 🔬 VERIFICATION CHECKLIST

### Data Quality Checks

- [ ] Check speed distribution (mean, std, range)
- [ ] Check temporal coverage (how many days? hours?)
- [ ] Check spatial coverage (how many edges?)
- [ ] Verify train/test split is temporal (no shuffle!)
- [ ] Check for data leakage (test timestamps in train?)

### Training Process Checks

- [ ] Verify normalization only uses train statistics
- [ ] Check early stopping (best epoch vs final epoch)
- [ ] Review loss curves (train vs val divergence?)
- [ ] Confirm evaluation on holdout test set
- [ ] Check if metrics are denormalized correctly

### Performance Reality Checks

- [ ] Compare MAE to naive baselines (previous speed, mean speed)
- [ ] Check if MAE matches loss value (normalized vs denormalized?)
- [ ] Verify R² score makes sense (0.93 means 93% variance explained)
- [ ] Check if predictions are clipped/bounded (artificial smoothing?)
- [ ] Compare to literature (SOTA traffic models usually MAE 3-5 km/h)

### Code Quality Checks

- [ ] No test data in training loop
- [ ] No global statistics computed on full dataset
- [ ] Proper cross-validation or holdout
- [ ] No data augmentation on test set
- [ ] Clear separation of train/val/test

---

## 📊 EXPECTED FINDINGS

### Scenario 1: Claims are TRUE (unlikely ~5%)

**If MAE really is 0.91 km/h:**

- Data must be extremely smooth/predictable
- Likely heavy preprocessing/smoothing
- Model may be learning trivial patterns
- **Action:** Analyze why data is so easy, not useful for real traffic

### Scenario 2: Metrics CONFUSION (likely ~60%)

**If 0.91 is normalized loss, real MAE is ~5-10 km/h:**

- Common mistake: report loss without denormalization
- `val_loss = 0.0071` in normalized space
- Real MAE = `0.0071 × std + mean` (if MinMaxScaler/StandardScaler)
- **Action:** Document confusion, update report with corrected metrics

### Scenario 3: Data LEAKAGE (possible ~25%)

**If test data was seen during training:**

- Augmentation used test statistics
- Normalization computed on full dataset
- Validation set used for augmentation patterns
- **Action:** Document leakage, mark as "unverified", use our implementation

### Scenario 4: EVALUATION ERROR (possible ~10%)

**If evaluated on validation instead of test:**

- Report says "test" but actually validation
- Or used augmented validation set
- **Action:** Re-evaluate on true holdout if possible

---

## 🎬 EXECUTION PLAN

### Timeline: 1-2 days

**Day 1 Morning (2 hours) - Quick Check:**

1. ✅ Create verification branch
2. ✅ Run Strategy A (Quick Verification)
3. ✅ Check results against expectations
4. ⏸️ **Decision point:** If MAE ~10-15 km/h → STOP, conclusion clear
5. ⏸️ If MAE < 5 km/h → Continue to Strategy B

**Day 1 Afternoon (3 hours) - Deep Dive (if needed):**

6. Setup hunglm's environment
7. Inspect data quality
8. Review training code for red flags
9. Document findings

**Day 2 (optional, 4-6 hours) - Experiments:**

10. Run verification experiments
11. Analyze results
12. Write comprehensive report

### Deliverables

1. **Verification Report:** `docs/GRAPHWAVENET_VERIFICATION_REPORT.md`
   - Methodology
   - Findings
   - Conclusion (verified/suspicious/rejected)
   - Recommendations

2. **Updated Analysis:** Update `docs/GRAPHWAVENET_CONTRIBUTION_ANALYSIS.md`
   - Add "Verification" section
   - Adjust rating based on findings
   - Clear conclusion on performance claims

3. **Training Logs:** Save to `outputs/graphwavenet_verification/`
   - Baseline run results
   - Augmented run results (if tested)
   - Comparison with claimed performance

---

## 💡 DECISION RULES

### When to STOP verification early:

✅ **STOP if Strategy A shows MAE > 10 km/h**

- Conclusion: Claims not reproducible with reasonable data
- Likely explanation: Metrics confusion or very different data
- **Action:** Document in report, mark as "suspicious but unverified"

✅ **STOP if obvious red flags found in code review**

- Example: `scaler.fit(all_data)` or no train/test split
- Conclusion: Data leakage confirmed
- **Action:** Document issue, no need to reproduce

✅ **STOP if data not available**

- hunglm's exact data not accessible
- Can't reproduce exact setup
- **Action:** Document as "unable to verify"

### When to CONTINUE to Strategy B:

⚠️ **CONTINUE if Strategy A shows MAE < 5 km/h**

- Unexpected result, needs investigation
- May indicate we're missing something
- Worth deep dive to understand

⚠️ **CONTINUE if results are ambiguous**

- Performance varies widely between runs
- Unclear if problem is data or model
- Need more controlled experiments

---

## 📝 REPORT TEMPLATE

```markdown
# GraphWaveNet Performance Verification Report

**Date:** [Date]
**Tester:** [Name]
**Branch:** verify-hunglm-performance

## Executive Summary

- **Claim:** MAE 0.91 km/h, R² 0.93
- **Verification Status:** [VERIFIED / SUSPICIOUS / REJECTED / UNABLE_TO_VERIFY]
- **Conclusion:** [One sentence conclusion]

## Methodology

[What was tested, how]

## Results

### Quick Verification (Strategy A)

- Baseline run: MAE [X] km/h
- Augmented run: MAE [Y] km/h
- Comparison: [Analysis]

### Deep Dive (Strategy B) - if conducted

[Detailed findings]

## Analysis

### Data Quality

[Findings about data]

### Training Process

[Findings about training]

### Performance Claims

[Analysis of 0.91 km/h claim]

## Conclusion

### Rating Update

- Previous: 4.5/5
- Updated: [X]/5
- Rationale: [Why]

### Recommendations

1. [Recommendation 1]
2. [Recommendation 2]

## Lessons Learned

[What we learned from this verification]
```

---

## 🚀 GETTING STARTED

### Step 1: Create Branch

```bash
cd D:\UNI\DSP391m\project
git checkout -b verify-hunglm-performance
git add docs/GRAPHWAVENET_VERIFICATION_PLAN.md
git commit -m "docs: add GraphWaveNet verification plan"
```

### Step 2: Run Quick Verification

```bash
# Test current implementation with baseline data
python scripts/training/train_graphwavenet_baseline.py \
  --dataset data/processed/all_runs_combined.parquet \
  --output-dir outputs/gwn_verify_baseline \
  --epochs 50 \
  --batch-size 16

# Check results
cat outputs/gwn_verify_baseline/run_*/results.json | grep mae
```

### Step 3: Analyze & Decide

- If MAE > 10 km/h → Document & merge back to master
- If MAE < 5 km/h → Continue to Strategy B
- If error → Debug and retry

### Step 4: Update Reports

```bash
# Create verification report
code docs/GRAPHWAVENET_VERIFICATION_REPORT.md

# Update analysis with findings
code docs/GRAPHWAVENET_CONTRIBUTION_ANALYSIS.md

# Update CHANGELOG
code docs/CHANGELOG.md
```

---

## ⚖️ SUCCESS CRITERIA

### Verification is SUCCESSFUL if:

✅ We can definitively answer: "Is MAE 0.91 km/h real?"

✅ We document findings clearly (verified/suspicious/rejected)

✅ We update analysis report with evidence-based rating

✅ We extract lessons learned for our project

### Verification is COMPLETE when:

✅ Report written and reviewed

✅ Branch merged back to master

✅ Team understands hunglm's contribution realistically

✅ We can cite findings in final report

---

**Next Action:** Run `git checkout -b verify-hunglm-performance` and start Strategy A! 🚀
