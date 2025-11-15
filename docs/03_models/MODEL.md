# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# STMGT Model Overview

Complete guide to STMGT (Spatio-Temporal Multi-Modal Graph Transformer) model capabilities, performance, and limitations.

**Model Version:** V1 (680K parameters)
**Status:** Production - Optimal Configuration
**Last Updated:** November 12, 2025

---

## Quick Facts

- **Performance:** MAE 3.08 km/h (beats SOTA by 21-28%)
- **Architecture:** Graph Neural Network + Transformer
- **Uncertainty:** Probabilistic forecasting with 84% calibration
- **Coverage:** 62 nodes, 144 edges in Ho Chi Minh City district
- **Training Data:** 205K samples, 29-day collection period

---

## 1. Model Performance

### 1.1 Benchmark Comparison

| Model                    | MAE (km/h) | Improvement over STMGT | Status                |
| ------------------------ | ---------- | ---------------------- | --------------------- |
| Naive (last value)       | 7.20       | +134% worse            | Baseline              |
| LSTM Sequential          | 4.42-4.85  | +43-57% worse          | Traditional           |
| GCN Graph                | 3.91       | +27% worse             | Graph-based           |
| GraphWaveNet (SOTA 2019) | 3.95       | +28% worse             | SOTA                  |
| ASTGCN                   | 4.29       | +39% worse             | Failed on 29-day data |
| **STMGT V1**             | **3.08**   | **BEST**               | **This work**         |

**Key Achievement:** Beats current SOTA (GraphWaveNet 2019) by **21-28%**

### 1.2 Detailed Metrics

**Test Set Performance:**

- **MAE:** 3.08 km/h (mean absolute error)
- **RMSE:** 4.50 km/h (root mean squared error)
- **R²:** 0.820 (82% variance explained)
- **MAPE:** 19.68% (mean absolute percentage error)

**Uncertainty Quantification:**

- **Coverage@80:** 83.75% (target: 80%)
- **CRPS:** 2.23 (continuous ranked probability score)
- **Calibration:** Well-calibrated (only 3.75% deviation)

### 1.3 Recent Training Results (Nov 12, 2025)

**Baseline (No Augmentation):**

- Test MAE: 3.1068 km/h
- Test R²: 0.8157
- Best Val MAE: 3.1713 km/h

**With SafeTrafficAugmentor:**

- Test MAE: 3.0774 km/h ✓ (0.95% improvement)
- Test R²: 0.8175 ✓
- Best Val MAE: 3.1570 km/h
- Improvement: Consistent across all metrics

---

## 2. Model Architecture

### 2.1 Design Rationale

**Goal:** Optimal balance between capacity and generalization

**Architecture:** 3-layer Graph Neural Network + Transformer Attention

- **Parameters:** 680K (optimal from capacity experiments)
- **GNN Blocks:** 3 (provides 3-hop spatial propagation)
- **Attention Heads:** Multi-head transformer for temporal patterns
- **Uncertainty:** Gaussian Mixture Model (K=5 modes)

**Why 680K Parameters?**

Tested 5 model sizes (350K to 1.15M parameters):

- 350K (V0.6): Too simple, MAE 3.11
- 520K (V0.8): Inefficient, MAE 3.22
- **680K (V1): OPTIMAL, MAE 3.08** ✓
- 850K (V1.5): Overfitting, MAE 3.18
- 1.15M (V2): Severe overfit, MAE 3.22

**Result:** U-shaped performance curve, 680K is global optimum

### 2.2 Key Components

**Graph Neural Network (3 hops):**

- Captures spatial dependencies between road segments
- 3-hop propagation: accident affects neighbors up to 3 edges away
- Graph diameter: 12 hops (25% coverage)

**Transformer Attention:**

- Learns temporal patterns (rush hour, weekday vs weekend)
- Detects sudden changes (accidents, weather events)
- Sequence length: 12 timesteps (3 hours history)

**Gaussian Mixture Model:**

- 5 mixture components for uncertainty quantification
- Produces probabilistic forecasts (not just point predictions)
- Enables confidence intervals and risk assessment

---

## 3. Real-World Capabilities

### 3.1 Traffic Variability Handling

**Speed Distribution:**

- Range: 3.37 to 52.84 km/h (14× variation)
- Mean: 18.72 ± 7.03 km/h (37% coefficient of variation)
- Multi-modal: Free-flow (40-50), Moderate (15-25), Congested (<13)

**Temporal Patterns:**

- Rush hours: 30-50% speed reduction
- Weather impact: 10-20% speed reduction in rain
- Weekend vs weekday: 15% difference

**Spatial Patterns:**

- 62 nodes with different characteristics
- 144 edges with varying traffic patterns
- Heterogeneous network (not uniform grid)

### 3.2 Accident Response Example

**Scenario:** Accident at 14:00, edge A→B blocked (speed: 25 → 1 km/h)

**Model Response:**

**1-hop neighbors (direct connections):**

- Predicted impact: 30-50% speed drop
- Reason: Traffic spillover from blocked edge

**2-hop neighbors (alternative routes):**

- Predicted impact: 10-20% speed drop
- Reason: Rerouting traffic increases congestion

**3-hop neighbors (distant edges):**

- Predicted impact: 5-10% speed drop
- Reason: Ripple effect through network

**Beyond 3 hops:**

- No direct spatial propagation (model limitation)
- Only temporal patterns used

**Uncertainty Quantification:**

```
Gaussian Mixture for blocked edge at 14:15:
- Mode 1 (60%): μ=2 km/h → "Still blocked"
- Mode 2 (25%): μ=5 km/h → "Partial clearance"
- Mode 3 (8%): μ=15 km/h → "Fully cleared"
- Mode 4 (5%): μ=8 km/h → "Slow clearance"
- Mode 5 (2%): μ=25 km/h → "Back to normal"
```

---

## 4. Limitations

### 4.1 Spatial Limitations

**Coverage:** 2048m radius district

- Current: 62 nodes, 144 edges (~4 km²)
- City-wide: Would need 2,000+ nodes, 5,000+ edges (~2,095 km²)
- Scale factor: 32× more nodes, 35× more edges

**Propagation Range:** 3 hops

- Network diameter: 12 hops
- Current coverage: 25% of network
- Solution: Increase to 10-15 blocks OR use hierarchical GNN

### 4.2 Temporal Limitations

**Training Period:** 29 days

- Limited seasonal coverage (single month)
- May not capture annual patterns
- Solution: Collect 3-6 months of data

**Forecast Horizon:** 3 hours (12 timesteps)

- Beyond 3 hours: Accuracy degrades significantly
- Current: Focuses on short-term operational forecasting

### 4.3 Data Quality Limitations

**Weather Integration:** Simplified

- Only 3 features: temperature, wind, precipitation
- Missing: humidity, visibility, air quality
- Weather impact: 10-20% in model (should be 30-40%)

**Missing Events:**

- No accident data (must infer from speed drops)
- No construction schedules
- No event information (concerts, sports)

### 4.4 Uncertainty Limitations

**Calibration Quality:** 83.75% coverage

- Target: 80% (model achieves 83.75%)
- Slightly overconfident in some scenarios
- Could improve with more diverse training data

---

## 5. Research Contributions

### 5.1 Academic Value

**Novel Contributions:**

1. **Capacity Analysis:** First systematic study of GNN capacity for traffic forecasting
2. **Uncertainty Quantification:** Rare in traffic forecasting literature (most papers don't report CRPS)
3. **Real-World Validation:** 29-day continuous collection, not synthetic data
4. **Engineering Quality:** Production-ready implementation with proper validation

**Comparison with Literature:**

- Most papers: 1-7 day experiments
- This work: 29-day continuous collection
- Most papers: No uncertainty quantification
- This work: Calibrated probabilistic forecasts

### 5.2 Practical Value

**Production Deployment:**

- FastAPI REST API for real-time predictions
- CLI tool for operations management
- Docker containerization ready
- Monitoring and logging integrated

**Cost-Effective:**

- 680K parameters (runs on single GPU)
- Inference time: <100ms per prediction
- No expensive infrastructure required

---

## 6. Future Work

### 6.1 Short-Term (3-6 months)

**Data Collection:**

- Extend to 3-6 months for seasonal patterns
- Add more nodes (100-200) for larger coverage
- Collect accident and event data

**Model Improvements:**

- Increase GNN blocks to 5-7 (better spatial propagation)
- Add hierarchical structure (district → city levels)
- Improve weather integration (more features)

### 6.2 Medium-Term (6-12 months)

**City-Scale Expansion:**

- 2,000+ nodes, 5,000+ edges
- Hierarchical GNN architecture
- Multi-district coordination

**Advanced Features:**

- Real-time incident detection
- Route optimization integration
- Multi-modal transport (bus, metro)

**Production Hardening:**

- A/B testing framework
- Performance monitoring dashboard
- Automatic retraining pipeline

### 6.3 Long-Term (1-2 years)

**Research Directions:**

- Transfer learning to other cities
- Federated learning for privacy
- Causal inference for interventions

**Integration:**

- Google Maps / Waze integration
- Government traffic management systems
- Public transportation optimization

---

## 7. Using This Model

### 7.1 When to Use STMGT

**Good fit:**

- Short-term forecasting (15 min to 3 hours)
- District-level coverage (50-100 nodes)
- Operational traffic management
- Route optimization
- Uncertainty-aware applications

**Not suitable:**

- Long-term planning (>1 day)
- City-wide coverage without hierarchical structure
- Applications requiring >5 hour forecasts
- Extremely sparse networks (<20 nodes)

### 7.2 Performance Expectations

**Typical Accuracy:**

- Urban district: MAE 3-4 km/h
- Highway segments: MAE 5-7 km/h (higher speeds → larger errors)
- Rush hour: MAE 4-5 km/h (more variability)
- Off-peak: MAE 2-3 km/h (stable traffic)

**Uncertainty Calibration:**

- 80% confidence interval: Expect 80-85% coverage
- Use for risk-aware routing
- Good for probabilistic applications

---

## 8. Related Documentation

**Architecture Details:**

- [ARCHITECTURE.md](ARCHITECTURE.md) - Technical design
- [RESEARCH.md](RESEARCH.md) - Academic research summary

**Training & Deployment:**

- [TRAINING_WORKFLOW.md](../TRAINING_WORKFLOW.md) - Complete training pipeline
- [TRAINING.md](TRAINING.md) - Advanced training guide
- [DEPLOYMENT.md](DEPLOYMENT.md) - Production deployment

**Data & Augmentation:**

- [DATA.md](DATA.md) - Data concepts and schemas
- [AUGMENTATION.md](AUGMENTATION.md) - Data augmentation guide
- [FIXES.md](FIXES.md) - Critical fixes (data leakage)

**API & Tools:**

- [API.md](API.md) - REST API reference
- [CLI.md](CLI.md) - Command-line interface

---

## Questions or Issues

For model-specific questions, refer to:

- Technical questions → [ARCHITECTURE.md](ARCHITECTURE.md)
- Training issues → [TRAINING.md](TRAINING.md)
- Deployment issues → [DEPLOYMENT.md](DEPLOYMENT.md)
- Data issues → [FIXES.md](FIXES.md)
