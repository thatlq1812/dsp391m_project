# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Section 4: Literature Review

## Overview

This section reviews existing approaches to traffic forecasting, from classical statistical methods to modern deep learning architectures. We examine 60+ academic papers to identify gaps in current knowledge and justify our STMGT architecture.

---

## 4.1 Classical Traffic Forecasting Methods

### 4.1.1 Statistical Approaches

**ARIMA (AutoRegressive Integrated Moving Average)** [15]

- **Strengths:** Simple, interpretable, works for short-term univariate forecasting
- **Limitations:** Cannot model spatial dependencies, fails with non-linear patterns, requires stationary data
- **Performance:** MAE ~5-8 km/h on simple road segments

**Kalman Filters**

- **Strengths:** Real-time updates, handles noise well
- **Limitations:** Assumes linear dynamics, no spatial modeling
- **Usage:** Still used in some commercial GPS systems for state estimation

**Vector Autoregression (VAR)**

- **Strengths:** Models multiple time series, captures some spatial correlation
- **Limitations:** Scales poorly to large networks (O(N²) parameters), linear assumptions
- **Performance:** Marginally better than ARIMA for small networks (<20 nodes)

**Why Classical Methods Fall Short:**

- Traffic exhibits **strong non-linearity** (congestion cascades, bottlenecks)
- **Spatial dependencies** are complex graph-structured, not grid-based
- **Multi-modal influences** (weather, events) require flexible feature integration

---

## 4.2 Early Deep Learning Approaches

### 4.2.1 Recurrent Neural Networks

**LSTM (Long Short-Term Memory)** [1]

- **Architecture:** Gated RNN with memory cells for long-term dependencies
- **Traffic Applications:** Duan et al. (2016), Ma et al. (2015)
- **Strengths:** Captures temporal patterns, handles sequences naturally
- **Limitations:**
  - No spatial modeling → treats each road independently
  - Sequential processing slow for inference
  - Vanishing gradients for very long sequences
- **Performance:** MAE ~4-6 km/h (single-node forecasting)

**GRU (Gated Recurrent Units)**

- Similar to LSTM but fewer parameters
- Slightly faster training, comparable accuracy
- Still lacks spatial modeling

**Bidirectional LSTM**

- Processes sequences forward and backward
- Better context but not suitable for real-time forecasting (requires full future sequence)

**Our LSTM Baseline Results:**

- **MAE:** 4.85 km/h (test set)
- **R²:** 0.64
- **Issue:** Cannot leverage road network structure

---

## 4.3 Graph Neural Networks for Traffic

### 4.3.1 Spatial Graph Convolution

**Graph Convolutional Networks (GCN)** [5]

- **Key Idea:** Generalize convolution to graph-structured data (Kipf & Welling, 2017)
- **Message Passing:** `h'_v = σ(Σ_{u∈N(v)} W · h_u / √(deg(u)·deg(v)))`
- **Limitations:**
  - No temporal modeling
  - Fixed graph structure
  - Over-smoothing with many layers

**ChebNet** [4]

- Uses Chebyshev polynomials for efficient spectral graph convolution (Defferrard et al., 2016)
- **Complexity:** O(K·|E|) where K is filter order
- **Advantage:** Localized filters, efficient computation

### 4.3.2 Graph Attention Networks

**Graph Attention Networks (GAT)** [6]

- **Key Innovation:** Learns attention weights for neighbors (Veličković et al., 2018)
- **Formula:** `α_{ij} = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))`
- **Advantage:** Adaptive neighbor importance, handles varying graph structures
- **Limitation:** O(N²) memory for full graph

**GATv2** [7]

- Fixes expressiveness limitation of original GAT (Brody et al., 2022)
- **More dynamic attention:** `α_{ij} = softmax(a^T LeakyReLU(W[h_i || h_j]))`
- **Our Choice:** Used in STMGT for spatial modeling

---

## 4.4 Spatio-Temporal Graph Models (SOTA)

### 4.4.1 STGCN (2018) - First ST-GCN [11]

**Reference:** Yu et al., IJCAI 2018

**Architecture:**

- **Spatial:** ChebNet [4] graph convolution
- **Temporal:** 1D CNN (temporal convolution)
- **Structure:** ST-Block = (Time-Conv → Graph-Conv → Time-Conv)

**Performance:** METR-LA dataset

- MAE: 2.96 mph
- RMSE: 5.87 mph
- MAPE: 7.89%
- **R² (estimated):** 0.76

**Limitations:**

- Sequential spatial-temporal processing suboptimal
- No attention mechanism
- No weather/external factors

### 4.4.2 Graph WaveNet (2019) - Adaptive Graph Learning [13]

**Reference:** Wu et al., IJCAI 2019

**Key Innovations:**

1. **Adaptive Adjacency Matrix:** Learns graph structure from data
   ```
   A_adaptive = softmax(ReLU(E1 · E2^T))
   ```
2. **TCN (Temporal Convolutional Networks):** Dilated causal convolutions
3. **Parallel Processing:** Combines multiple graph convolutions

**Performance:** METR-LA

- **MAE: 2.69 mph** (best at time of publication)
- RMSE: 5.15 mph
- MAPE: 6.78%
- **R² (estimated):** 0.83

**Our Baseline Results:**

- MAE: 3.95 km/h
- R²: 0.71
- **Analysis:** Strong baseline, but lacks weather integration

### 4.4.3 MTGNN (2020) - Multi-Faceted Graph Learning [14]

**Reference:** Wu et al., KDD 2020

**Key Ideas:**

- **Uni-directional graph:** Traffic flow direction matters
- **Mix-hop propagation:** Combines K-hop neighborhoods
- **Dilated inception:** Multi-scale temporal patterns

**Performance:** METR-LA

- MAE: 2.72 mph
- MAPE: 6.85%
- **R² (estimated):** 0.82

### 4.4.4 ASTGCN (2019) - Spatial-Temporal Attention [12]

**Reference:** Guo et al., AAAI 2019

**Architecture:**

- **Spatial Attention:** Learn node importance dynamically
- **Temporal Attention:** Weighted historical time steps
- **Recent/Daily/Weekly Components:** Multi-scale temporal modeling

**Performance:** PeMSD4

- MAE: 2.88 mph
- MAPE: 7.42%
- **R² (estimated):** 0.78

**Our ASTGCN Baseline Results:**

- **MAE: 4.29 km/h**
- **R²: 0.023** (very poor)
- **Issue:** Implementation complexity, sensitive to hyperparameters

### 4.4.5 GMAN (2020) - Attention-Based ST Network

**Reference:** Zheng et al., AAAI 2020

**Architecture:**

- **ST-Attention:** Parallel spatial and temporal multi-head attention
- **Transform Attention:** Encoder-decoder architecture
- **Gated Fusion:** Learned combination of spatial and temporal features

**Performance:** METR-LA

- MAE: 2.73 mph
- **Key Insight:** Parallel processing beats sequential by 5-8%

### 4.4.6 DGCRN (2022) - Dynamic Graph Convolution

**Reference:** Li et al., AAAI 2022

**Current SOTA:**

- **MAE: 2.59 mph** (best on METR-LA)
- MAPE: 5.82%
- **R² (estimated):** 0.85

**Innovation:** Dynamic graph construction at each time step

---

## 4.5 Uncertainty Quantification in Traffic Forecasting

### 4.5.1 Bayesian Neural Networks

**Approach:** Variational inference for weight posteriors

- **Pros:** Principled uncertainty quantification
- **Cons:** Slow inference, difficult to scale

### 4.5.2 Dropout as Bayesian Approximation

**MC Dropout (Gal & Ghahramani, 2016)**

- Enable dropout at inference, sample multiple predictions
- **Our testing:** Underestimates uncertainty for traffic data

### 4.5.3 Gaussian Mixture Models

**Mixture Density Networks** [2]

- Output parameters of K Gaussian components (Bishop, 1994)
- **Application:** Traffic speeds exhibit multi-modal distributions
  - **Free-flow:** ~40-50 km/h
  - **Moderate:** ~20-30 km/h
  - **Congested:** <15 km/h

**Our Choice:** K=5 Gaussian components

- Captures traffic state transitions
- CRPS loss [3] for proper scoring

---

## 4.6 Multi-Modal Fusion for Traffic

### 4.6.1 Weather Integration

**Existing Approaches:**

1. **Simple Concatenation:** Add weather as extra node features
   - Used in most papers but suboptimal
2. **FiLM (Perez et al., 2018):** Feature-wise Linear Modulation
   - `γ, β = MLP(weather)` then `output = γ · features + β`
3. **Cross-Attention:** Query traffic with weather context
   - **Our approach:** More expressive than concatenation

**Evidence for Weather Impact:**

- Rain: -15% speed reduction (Zhang et al., 2019)
- Heavy rain: -30% speed reduction
- Temperature extremes: -8% speed reduction

### 4.6.2 Attention Mechanisms

**Transformers** [8]

- **Self-Attention:** `Attention(Q,K,V) = softmax(QK^T/√d_k)V` (Vaswani et al., 2017)
- **Multi-Head:** Multiple attention patterns in parallel
- **Positional Encoding:** Sin/cos or learnable embeddings

**Application to Traffic:**

- Temporal self-attention for historical sequences [9]
- Cross-attention for weather conditioning
- **Challenge:** O(T²) complexity for long sequences

---

## 4.7 Research Gaps & STMGT Motivation

### 4.7.1 Identified Gaps

Based on review of 60+ papers, key limitations in existing work:

1. **Spatial-Temporal Processing:**

   - Most models use **sequential** (spatial → temporal) processing
   - Parallel processing shown superior but underexplored
   - **Gap:** Need architecture with parallel ST blocks + learned fusion

2. **Uncertainty Quantification:**

   - Most papers report only point predictions (MAE, RMSE)
   - Few use proper probabilistic metrics (CRPS, NLL, calibration)
   - **Gap:** Need probabilistic model with well-calibrated uncertainties

3. **Multi-Modal Integration:**

   - Weather typically concatenated, not fused intelligently
   - No cross-attention for context-dependent weather effects
   - **Gap:** Need adaptive fusion mechanism (cross-attention)

4. **Real-World Deployment:**

   - Most papers evaluate on public datasets (METR-LA, PeMS-BAY)
   - Limited work on emerging markets (Vietnam, Southeast Asia)
   - **Gap:** Need production-ready system with API deployment

5. **Small Network Challenges:**
   - SOTA models validated on large networks (200+ nodes, 30K+ samples)
   - Limited guidance for smaller networks (50-100 nodes, <20K samples)
   - **Gap:** Need realistic performance targets and regularization strategies

### 4.7.2 STMGT Design Rationale

Our architecture addresses these gaps through:

| Component                     | Addresses Gap              | Innovation                                    |
| ----------------------------- | -------------------------- | --------------------------------------------- |
| **Parallel ST Blocks**        | Sequential processing      | GATv2 ‖ Transformer with gated fusion         |
| **Gaussian Mixture (K=5)**    | Uncertainty quantification | Multi-modal speed distribution modeling       |
| **Weather Cross-Attention**   | Multi-modal fusion         | Context-dependent weather effects             |
| **Aggressive Regularization** | Small network overfitting  | Dropout 0.2, DropEdge 0.05, weight decay 1e-4 |
| **FastAPI Deployment**        | Production readiness       | <400ms inference, REST API, Docker            |

---

## 4.8 Benchmark Summary

**METR-LA Dataset (207 nodes, 15-min intervals):**

| Model             | Year | MAE (mph) | MAPE  | R²   | Architecture Type   |
| ----------------- | ---- | --------- | ----- | ---- | ------------------- |
| **DGCRN**         | 2022 | **2.59**  | 5.82% | 0.85 | Dynamic Graph       |
| **Graph WaveNet** | 2019 | 2.69      | 6.78% | 0.83 | Adaptive + TCN      |
| **MTGNN**         | 2020 | 2.72      | 6.85% | 0.82 | Multi-faceted Graph |
| **ASTGCN**        | 2019 | 2.88      | 7.42% | 0.78 | ST Attention        |
| **STGCN**         | 2018 | 2.96      | 7.89% | 0.76 | First ST-GCN        |

**Our Expected Performance (62 nodes, 16K samples):**

- **MAE Target:** 2.0-3.5 km/h
- **R² Target:** 0.45-0.55 (scaled from SOTA)
- **Rationale:** Smaller network + less data → lower R² but acceptable MAE

---

## 4.9 Key Takeaways

1. **Sequential → Parallel Processing:** 5-12% improvement demonstrated
2. **Graph Attention:** Adaptive neighbor weighting outperforms fixed graph convolution
3. **Uncertainty:** Gaussian mixtures appropriate for multi-modal traffic distributions
4. **Weather:** Cross-attention more expressive than simple concatenation
5. **Regularization:** Critical for small datasets (dropout, DropEdge, early stopping)

**Next Steps:** Apply these findings to STMGT architecture design and implementation.

---

**Next:** [Data Description →](03_data_description.md)
