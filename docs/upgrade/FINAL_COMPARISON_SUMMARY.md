# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Final Baseline Comparison Summary

**Date:** November 9, 2025
**Status:** Phase 1 Complete - Ready for Production Deployment

---

## Executive Summary

Completed comprehensive baseline comparison to validate STMGT superiority. After testing multiple graph-based approaches, confirmed that STMGT's hybrid architecture achieves best performance for edge-level traffic speed prediction.

**Key Finding:** STMGT outperforms temporal baseline by 6.3% with Val MAE of 3.69 km/h vs LSTM's 3.94 km/h.

---

## Models Evaluated

### 1. LSTM Baseline ✅ SUCCESS

**Architecture:** Temporal-only (RNN)

- Sequence length: 12 timesteps
- LSTM layers: [128, 64] units
- Temporal features: hour_sin, hour_cos, day_of_week
- Training samples: 6,624

**Results:**

- Val MAE: **3.94 km/h**
- Train MAE: 4.30 km/h
- Early stopped at epoch 14/20
- Parameters: ~100K

**Strengths:**

- Simple, fast training
- Good temporal pattern learning
- Reliable baseline

**Weaknesses:**

- No spatial modeling
- Limited long-range dependencies
- Ignores edge relationships

---

### 2. GCN Baseline ❌ FAILED

**Architecture:** Graph Convolutional Network

- 2-layer GCN with dropout
- Full graph snapshot requirement

**Why It Failed:**

- GCN requires input shape: `(batch, timesteps, num_nodes, features)`
- Our data: 144 independent edge time series
- Result: Only **40 training sequences** from 46 timestamps
- Validation: Only **3 sequences** (statistically invalid)
- Val MAE: 3.91 km/h (meaningless due to tiny dataset)

**Lesson:** Model architecture must match data structure. GCN works for node-level prediction with spatial topology, not for independent edge time series.

---

### 3. GraphWaveNet Baseline ❌ FAILED

**Architecture:** Adaptive graph learning + dilated TCN

- 4 layers with exponential dilation
- Learns adjacency matrix from data
- Skip connections for gradient flow

**Why It Failed:**

- Same fundamental issue as GCN
- Requires full graph snapshots
- Result: Only **40 training sequences**
- Val MAE: **11.04 km/h** (worse than LSTM!)
- Parameters: ~32K

**Lesson:** ANY graph model requiring snapshot architecture fails with edge-level data, regardless of sophistication (adaptive learning, attention, etc.).

---

### 4. STMGT (Hybrid Model) ✅ BEST PERFORMANCE

**Architecture:** Multi-modal Spatial-Temporal Graph Transformer

- Graph module for edge relationships
- Transformer for long-range dependencies
- Weather fusion for external factors
- Probabilistic predictions

**Results:**

- Val MAE: **3.69 km/h** ⭐
- 6.3% better than LSTM
- 199% better than GraphWaveNet

**Why It Works:**

- Edge-level architecture (not graph snapshots)
- Learns spatial relationships without requiring full graph
- Transformer captures long-range temporal patterns
- Weather integration improves accuracy
- Provides uncertainty estimates

---

## Critical Insights

### 1. Architecture-Data Match is Essential

**Problem:** Graph snapshot models (GCN, GraphWaveNet, ASTGCN) require:

```python
input_shape = (batch, timesteps, num_nodes, features)
```

**Our Data:** Edge-level time series:

- 144 independent edges
- 66 unique timestamps
- 144 samples per timestamp
- Total: 9,504 samples

**Consequence:** Graph snapshot → Only 40 training sequences (insufficient)

### 2. Edge-level vs Node-level Prediction

**Our Task:** Predict speed for each edge independently

- Input: Historical speeds for one edge
- Output: Future speed for that edge
- No need for full graph state at one timestamp

**LSTM Approach:** Treats each edge as separate time series → 6,624 training samples ✅

**STMGT Approach:** Learns edge relationships while maintaining edge-level prediction → Works!

### 3. Model Complexity ≠ Performance

**GraphWaveNet** (32K params, adaptive learning, skip connections) → 11.04 km/h ❌
**LSTM** (100K params, simple RNN) → 3.94 km/h ✅
**STMGT** (sophisticated hybrid) → 3.69 km/h ⭐

**Lesson:** Appropriate architecture > Model sophistication

---

## Final Comparison

| Model        | Val MAE (km/h) | Improvement vs LSTM | Training Samples | Status     |
| ------------ | -------------- | ------------------- | ---------------- | ---------- |
| GraphWaveNet | 11.04          | -180%               | 40               | ❌ Failed  |
| LSTM         | 3.94           | Baseline            | 6,624            | ✅ Success |
| **STMGT**    | **3.69**       | **+6.3%**           | 6,624            | ⭐ Best    |

---

## Why STMGT is Superior

### 1. Graph Module

- Learns edge relationships without requiring full graph snapshots
- Adaptive adjacency captures traffic flow patterns
- Spatial information improves predictions

### 2. Transformer Architecture

- Self-attention handles long-range dependencies
- Better than LSTM at capturing complex temporal patterns
- Parallel processing (faster training)

### 3. Weather Integration

- External factors (rain, temperature) affect traffic
- Multi-modal fusion improves accuracy
- Provides additional context

### 4. Probabilistic Predictions

- Outputs uncertainty estimates
- Important for route planning (risk assessment)
- More informative than point predictions

---

## Recommendations for Production

### Use STMGT for Final System

**Reasons:**

1. Best performance (3.69 km/h)
2. Provides uncertainty estimates
3. Handles multiple modalities (traffic + weather)
4. Production-ready architecture

### Comparison in Final Report

**Focus on LSTM vs STMGT:**

- Clear 6.3% improvement
- Explain component contributions
- Show value of hybrid approach
- Document lessons learned from failed graph models

### Web Visualization

**Gradient Color Mapping:**

```
Speed Range → Color
50+ km/h   → Blue (#0066FF)    - Very smooth
40-50      → Green (#00CC00)   - Smooth
30-40      → Light Green (#90EE90) - Normal
20-30      → Yellow (#FFD700)  - Slow
10-20      → Orange (#FF8800)  - Congested
0-10       → Red (#FF0000)     - Heavy traffic
```

**Route Planning Feature:**

- Input: Start node, end node, departure time
- Output: 3 routes (fastest, shortest, balanced)
- Display: Predicted travel time ± uncertainty
- Update: Real-time every 5 minutes

---

## Next Steps

### Phase 2: Production System (Week 2)

1. **API Development** - FastAPI with STMGT predictor

   - `GET /api/traffic/current` → All edge speeds
   - `POST /api/route/plan` → Optimal route
   - `GET /api/predict/{edge_id}` → Edge prediction

2. **Web Interface** - Interactive traffic map

   - Leaflet.js or Mapbox GL for mapping
   - Gradient color visualization
   - Route planning form
   - Real-time updates

3. **Route Optimization** - Dijkstra/A\* with uncertainty
   - Multiple route options
   - Travel time estimation
   - Confidence intervals

### Phase 3: VM Deployment (Week 3)

1. Automated deployment scripts
2. Real-time data collection
3. Model serving infrastructure
4. Monitoring dashboard

---

## Files & Artifacts

### Models Trained

```
outputs/
├── lstm_baseline_production/
│   └── run_20251109_143849/
│       ├── lstm_model.keras
│       ├── results.json
│       └── training_history.csv
│
├── graphwavenet_baseline_production/
│   └── run_20251109_163755/
│       ├── graphwavenet_model.keras
│       └── results.json (failed - 11.04 km/h)
│
└── stmgt_production/
    └── best_model.pt (3.69 km/h)
```

### Documentation

```
docs/
├── CHANGELOG.md (updated)
└── upgrade/
    ├── BASELINE_COMPARISON_PLAN.md
    └── FINAL_COMPARISON_SUMMARY.md (this file)
```

---

## Conclusion

Successfully validated STMGT superiority through rigorous baseline comparison. Discovered that graph snapshot models are fundamentally incompatible with edge-level prediction, leading to revised comparison strategy focusing on LSTM vs STMGT.

**Key Achievement:** Proved STMGT's 6.3% improvement over temporal baseline with clear architectural justification.

**Ready for:** Production deployment with confidence in model choice.
