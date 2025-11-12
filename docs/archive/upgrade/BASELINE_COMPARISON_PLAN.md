# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Baseline Model Comparison Plan

**Objective:** Prove STMGT superiority through comprehensive baseline comparison

**Timeline:** Week 1 (November 9-15, 2025)

---

## Model Comparison Strategy

### 4 Models to Compare

1. **LSTM** (Temporal-only baseline) ✅

   - Simple RNN architecture
   - Only temporal patterns
   - No spatial modeling
   - **Status:** COMPLETED - Val MAE 3.94 km/h

2. **GraphWaveNet** (Adaptive graph learning)

   - Learns adjacency matrix from data
   - Dilated causal convolutions
   - Skip connections
   - **Status:** NEXT - In progress

3. **ASTGCN** (Spatial-temporal attention)

   - Spatial attention on graph
   - Temporal attention on sequences
   - Combined attention mechanism
   - **Status:** PENDING

4. **STMGT** (Hybrid: Graph + Transformer + Weather) ✅
   - Graph module for spatial relationships
   - Transformer for long-range dependencies
   - Weather fusion for external factors
   - **Status:** COMPLETED - Val MAE 3.69 km/h

---

## Training Configuration

**Dataset:** `data/processed/all_runs_combined.parquet`

- Total samples: 9,504
- Unique timestamps: 66
- Edges per timestamp: 144

**Data Splits (Temporal):**

- Train: 70% (46 timestamps, 6,624 samples)
- Val: 15% (9 timestamps, 1,296 samples)
- Test: 15% (11 timestamps, 1,584 samples)

**Training Settings:**

- Epochs: 100 (with early stopping)
- Batch size: 16-32 (model-dependent)
- Window length: 12 timesteps
- Learning rate: 0.001
- Optimizer: Adam

**Evaluation Metrics:**

- Primary: MAE (km/h)
- Secondary: RMSE, R², MAPE
- Advanced: CRPS, 80% Coverage

---

## Expected Results

| Model        | Expected MAE | Architecture Type          | Key Feature              |
| ------------ | ------------ | -------------------------- | ------------------------ |
| LSTM         | ~4.0 km/h    | Temporal RNN               | Baseline (temporal only) |
| GraphWaveNet | ~3.8 km/h    | Adaptive graph + TCN       | Learn graph structure    |
| ASTGCN       | ~3.7 km/h    | Spatial-temporal attention | Attention mechanisms     |
| **STMGT**    | **3.69**     | Hybrid (Graph+Transformer) | Multi-modal fusion       |

---

## Analysis Plan

### 1. Performance Comparison

- Bar charts: MAE comparison across models
- Learning curves: Training vs validation loss
- Error distribution: Analyze prediction errors

### 2. Architecture Analysis

- Model complexity: Parameter counts
- Training efficiency: Time per epoch
- Inference speed: Predictions per second

### 3. Strengths & Weaknesses

**LSTM:**

- ✅ Simple, fast training
- ✅ Good temporal pattern learning
- ❌ No spatial modeling
- ❌ Limited long-range dependencies

**GraphWaveNet:**

- ✅ Learns graph structure
- ✅ Efficient TCN architecture
- ✅ Skip connections
- ❌ May overfit with limited spatial data

**ASTGCN:**

- ✅ Explicit attention mechanisms
- ✅ Spatial-temporal modeling
- ❌ Complex, slower training
- ❌ Attention may not add value without true topology

**STMGT:**

- ✅ Best performance
- ✅ Multi-modal (traffic + weather)
- ✅ Graph + Transformer hybrid
- ✅ Probabilistic predictions
- ⚠️ Most complex, longer training

### 4. Conclusion

- Why STMGT is superior for this use case
- Component contribution analysis
- Production readiness assessment

---

## Implementation Status

- [x] Evaluation framework (UnifiedEvaluator)
- [x] LSTM baseline (3.94 km/h)
- [ ] GraphWaveNet baseline (in progress)
- [ ] ASTGCN baseline (pending)
- [x] STMGT validation (3.69 km/h)
- [ ] Comprehensive comparison report
- [ ] Visualization (charts, graphs)

---

## Notes

### GCN Abandonment Reason

- GCN requires full graph snapshots `(batch, timesteps, num_nodes, features)`
- Our data: Independent edge time series (144 edges)
- Result: Only 40 training sequences (insufficient)
- **Lesson:** Architecture must match data structure

### Edge-level vs Node-level Prediction

- Our problem: Predict speed for each edge separately
- LSTM approach: Each edge is independent time series
- Graph models must handle this appropriately
- GraphWaveNet and ASTGCN will use edge-level prediction
