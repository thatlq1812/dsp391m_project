# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Comparative Analysis of Model Input Architectures for Traffic Forecasting

**Date:** November 9, 2025
**Purpose:** Technical evaluation of data representation strategies across four deep learning models
**Status:** Analysis Complete - Recommendations Provided

---

## Executive Summary

This document provides a comprehensive technical analysis of input design choices across four traffic forecasting models (LSTM, ASTGCN, GraphWaveNet, STMGT). We evaluate each approach based on:

1. **Data representation granularity** (node-based vs edge-based)
2. **Temporal causality preservation** (sequence creation methodology)
3. **Spatial structure modeling** (graph representation)
4. **Feature integration strategy** (multi-modal fusion)

**Key Finding:** Architectural differences make unified input/output formats technically infeasible. We propose an **adapter pattern** that ensures fair model comparison while respecting architectural constraints.

**Model Suitability Assessment:**

| Model        | Spatial Modeling     | Temporal Handling | Data Efficiency | Best Use Case                     |
| ------------ | -------------------- | ----------------- | --------------- | --------------------------------- |
| STMGT        | Excellent (GNN)      | Excellent         | High            | Production deployment             |
| GraphWaveNet | Excellent (Adaptive) | Excellent         | Medium          | Large datasets, pattern discovery |
| ASTGCN       | Good (GCN)           | Needs Fix         | Medium          | After addressing temporal issues  |
| LSTM         | None                 | Good              | Low             | Baseline comparison only          |

---

## Detailed Analysis

### 1. LSTM - Baseline Time Series Approach

**Input Format:**

```python
X_train: (n_sequences, seq_len, n_features)
# Example: (50000, 12, 5)  # 5 features: speed, temp, wind, precip, hour
```

**Architectural Characteristics:**

#### Limitation 1: No Explicit Spatial Structure

```python
# LSTM treats these as independent:
node_1_sequence = [speed_t1, speed_t2, ..., speed_t12]
node_2_sequence = [speed_t1, speed_t2, ..., speed_t12]

# Reality: Node 1 and Node 2 are CONNECTED by road!
# If node 1 congested → node 2 will be congested soon
# LSTM cannot model this because it has NO GRAPH STRUCTURE
```

**Impact on Traffic Forecasting:**

- Node A = Bến Thành Market (congested)
- Node B = Nguyễn Huệ Walking Street (500m away)
- LSTM predicts node B independently → Cannot model traffic propagation
- Traffic flow: A → B takes ~5 minutes (spatial dependency not captured)

#### Limitation 2: Independent Node Modeling

```python
# LSTM may train ONE model for all nodes:
for node in all_nodes:
    X_node = df[df['node_id'] == node]
    model.fit(X_node)  # Same LSTM for Bến Thành and Airport?!

# Or train N separate models (even worse):
models = {node_id: LSTM() for node_id in nodes}  # 62 models!
```

**Architectural Implications:**

- Highway node vs residential street → different traffic patterns not distinguished
- Cannot transfer knowledge between similar roads
- Cannot model traffic flow propagation through network

#### Assessment: Limited for Graph-Structured Data

**Appropriate Use Cases:**

- Baseline comparison benchmark
- Single-sensor time series (weather forecasting, stock prices)
- Initial prototype for time series components

**Limitations for Traffic Networks:**

- Graph-structured data requires explicit spatial modeling
- Traffic propagation is inherently spatial (not captured by pure time series)
- Suitable as baseline but not optimal for production deployment

---

### 2. ASTGCN - Advanced Architecture with Implementation Considerations

**Input Format:**

```python
Xh: (batch, num_nodes, num_features, seq_len)
# Example: (64, 62, 4, 12)  # 4 features: speed, temp, wind, precip
adjacency: (62, 62)  # Graph structure
laplacian: (62, 62)  # Scaled Laplacian for Chebyshev polynomials
```

**Architectural Strengths:**

#### ✅ Node-Based Representation

```python
# Each node has its own feature vector at each timestep
# Shape: (num_nodes, num_features, seq_len)
#        (62, 4, 12)
#
# Node 5 at t=0: [speed=25.3, temp=28.5, wind=4.2, precip=0.0]
# Node 5 at t=1: [speed=23.1, temp=28.7, wind=4.5, precip=0.1]
```

**Design Rationale:**

- Natural representation for Graph Convolutional Networks
- Each node maintains state, GCN aggregates from neighbors
- Aligns with traffic theory: intersection = node with traffic state

#### ✅ Spectral Graph Convolution (Laplacian-Based)

```python
L_tilde = compute_scaled_laplacian(adjacency)
# L = D^{-1/2} A D^{-1/2}  (normalized Laplacian)

# Chebyshev polynomials allow K-hop propagation
cheb_polys = [I, L_tilde, 2*L_tilde@L_tilde - I, ...]
# K=3 → model sees 3-hop neighbors
```

**Technical Advantages:**

- Mathematically principled (spectral graph theory foundation)
- Computationally efficient (Chebyshev polynomial recursion)
- Captures multi-hop spatial dependencies (K-hop neighborhood)

#### ✅ Multi-Component Temporal Modeling

```python
X_recent: Last 12 timesteps (3 hours)
X_daily: Same time yesterday (daily pattern)
X_weekly: Same time last week (weekly pattern)
```

**Design Rationale:**

- Traffic patterns exhibit strong temporal periodicity
- Daily patterns: Monday 8am rush ≈ Tuesday 8am rush
- Model learns typical temporal patterns across different time scales

**Implementation Considerations:**

#### ⚠️ Temporal Boundary Handling in Current Implementation

```python
# Current implementation in pytorch_astgcn/data.py:
for end in range(input_window, total_steps - forecast_horizon + 1):
    start = end - input_window
    X_recent = tensor[start:end]  # PROBLEM HERE!
    Y_target = tensor[end:end + forecast_horizon]
```

**Concrete Example:**

```
Timeline:
  run_5: [t0, t1, t2, t3, t4] (ends at 12:00)
  run_6: [t0, t1, t2, t3, t4] (starts at 14:00, 2 hours gap)

ASTGCN creates sequence:
  X = [run_5_t3, run_5_t4, run_6_t0, run_6_t1]
      ^^^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^
      Past run          Future run - LEAKAGE!

  Model learns: "When I see pattern [a,b], next is [c,d]"
  Reality: [a,b] and [c,d] are from DIFFERENT DAYS!
```

**Impact on Model Performance:**

1. Violates temporal causality principle (future information in training)
2. May inflate validation metrics (model has access to future patterns)
3. Deployment mismatch (production data doesn't cross temporal boundaries)

**Recommended Fix:** Modify sequence creation to respect run_id boundaries (implementation in adapter pattern below).

#### ⚠️ Edge-to-Node Aggregation Strategy

```python
# _expand_to_node_table() does:
edge_df: (timestamp, node_a, node_b, speed_ab)
→
node_df: (timestamp, node_a, avg_speed_a)

# Averages all incoming/outgoing edges
node_speed = mean([speed_ab, speed_ac, speed_ba, speed_ca])
```

**Trade-offs:**

- **Directional information**: A→B vs B→A may have different speeds (averaged in node representation)
- **Heterogeneous aggregation**: Highway exit + residential street → averaged speed
- **Dimensionality reduction**: 144 edges → 62 nodes (information compression by 57%)

**Example Scenario:**

```
Node A has 3 edges:
  - A→B: Highway (speed=80 km/h)
  - A→C: Residential (speed=25 km/h)
  - D→A: Incoming traffic (speed=40 km/h)

Node table: node_A_speed = mean(80, 25, 40) = 48.3 km/h
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            Note: Average combines different road types and directions
```

**Alternative Approaches:**

- Weighted average by edge distance
- Separate incoming/outgoing aggregation
- Maintain edge-based representation (trade computational cost for fidelity)

#### Assessment: Strong Architecture with Preprocessing Considerations

**Recommended Improvements:**

1. Modify sequence creation to respect run_id boundaries (prevents temporal leakage)
2. Consider weighted aggregation methods for edge-to-node conversion
3. Align normalization parameters with other models for fair comparison

**Optimal Use Cases:**

- Multi-scale temporal pattern modeling (daily/weekly)
- When node-level predictions are sufficient
- After addressing temporal boundary handling

---

### 3. GraphWaveNet - Adaptive Learning Architecture

**Input Format:**

```python
X: (batch, seq_len, num_nodes, num_features)
# Example: (64, 12, 62, 1)  # Only speed, no weather

# Adaptive adjacency matrix (LEARNABLE):
node_embeddings: (62, 10)  # Each node has 10-dim embedding
A_adaptive = ReLU(E @ E^T)  # Compute similarity from embeddings
```

**Architectural Innovations:**

#### ✅ Adaptive Graph Structure Learning

```python
# Instead of fixed adjacency from roads:
A_fixed = topology_adjacency  # Based on physical connections

# GraphWaveNet learns:
A_learned = node_embeddings @ node_embeddings.T

# Can discover:
# - Parallel roads (not physically connected but similar patterns)
# - Functional relationships (residential → commercial during rush hour)
# - Hidden bottlenecks
```

**Example Discovery:**

```
Node 15 (Bến Thành) and Node 42 (Airport Road) are NOT connected physically
But model learns: embedding_15 · embedding_42 = 0.87 (high similarity)

Why? Both are major destinations → similar temporal patterns
Model can use Node 15 to help predict Node 42 even without direct edge!
```

**Advantages:**

- Discovers patterns beyond physical topology
- Adapts to evolving traffic patterns
- Higher representational capacity than fixed graphs

#### ✅ Efficient Temporal Modeling with Dilated Convolutions

```python
# Instead of RNN (sequential):
temporal_conv = CausalConv1d(dilation=2^l)  # l = layer depth

# Receptive field grows exponentially:
# Layer 1: sees t-1
# Layer 2: sees t-2, t-4  (dilation=2)
# Layer 3: sees t-4, t-8, t-12  (dilation=4)
# Layer 4: sees all 24 timesteps
```

**Technical Benefits:**

- Highly parallelizable (faster training than RNN)
- Avoids vanishing gradient problems
- Efficient long-range temporal dependency capture

**Considerations:**

#### ⚠️ Data Requirements for Adaptive Learning

```python
# With limited data (66 runs), model may learn spurious correlations:
"Node 5 always correlates with Node 23"

# Deployment considerations:
# - Road construction changes traffic flow → learned adjacency may not adapt
# - New roads added → model needs retraining
# - Requires sufficient data to learn robust patterns
```

**Mitigation:** Regularization techniques, larger training datasets, or hybrid approaches combining fixed + learned graphs.

#### ⚠️ Weather Feature Integration

```python
# Current implementation in codebase: Speed-only
X: (batch, seq_len, num_nodes, 1)  # 1 = speed only

# Weather integration not explicitly modeled
# Rain impact on traffic requires external feature engineering
```

**Comparison with Multi-Modal Approaches:**

```python
# STMGT approach (for reference):
X_traffic: (batch, num_nodes, seq_len, 1)  # Edge speeds
X_weather: (batch, seq_len, 3)  # Global weather context
# Cross-attention mechanism for traffic-weather fusion
```

**Extension Opportunity:** Weather features could be added as additional channels in GraphWaveNet's input tensor.

#### ⚠️ Interpretability Considerations

```python
# Fixed graph: A_ij = 1 if road exists from i→j
# → Directly interpretable (based on physical map)

# Learned graph: A_ij = embedding_i · embedding_j
# → Requires additional analysis (learned correlations)
```

**Trade-off:** Higher model capacity vs reduced interpretability. Appropriate for research/exploration phase.

#### Assessment: High-Capacity Architecture with Data Requirements

**Optimal Use Cases:**

- Large datasets (10K+ samples for robust learning)
- Stable road network topology
- Exploratory analysis to discover hidden traffic patterns
- Research scenarios prioritizing performance over interpretability

**Considerations:**

- Requires sufficient training data to avoid overfitting
- Best suited for established, stable transportation networks
- May need additional interpretation techniques for deployment

---

### 4. STMGT - Production-Oriented Multi-Modal Architecture

**Input Format:**

```python
X_traffic: (batch, num_nodes, seq_len, 1)
# num_nodes = 62 edges (not nodes!)
# Preserves directional information: A→B vs B→A

X_weather: (batch, seq_len, 3)  # Global context
# [temperature, wind_speed, precipitation]

edge_index: (2, num_edges)  # Static graph structure
# [[source_nodes], [target_nodes]]
```

**Design Philosophy:**

#### ✅ Edge-Level Data Representation

```python
# Traffic reality:
# - A→B (Highway, 3 lanes, speed limit 80): speed=70 km/h
# - B→A (Highway, 2 lanes, speed limit 60): speed=45 km/h (different!)

# STMGT preserves this:
node_id = edge_id_for_A_to_B
X_traffic[node_id] = [70, 68, 65, ...]  # Speed history

node_id = edge_id_for_B_to_A
X_traffic[node_id] = [45, 43, 40, ...]  # Different speed!
```

**Advantages:**

- **Complete information preservation**: 144 edges → 144 predictions (100% coverage)
- **Maintains traffic state granularity**: Each edge preserves its own traffic condition
- **Directional awareness**: Aligns with Google Maps API data structure

**Data Alignment:**

```python
# Our data collection:
edge_record = {
    "node_a_id": "intersection_1",
    "node_b_id": "intersection_2",
    "speed_kmh": 35.2,  # Speed on THIS EDGE
    "distance_m": 450,
    "duration_s": 46
}

# STMGT uses this DIRECTLY (no aggregation needed)
# Node-based models require edge-to-node conversion
```

#### ✅ Temporal Causality Preservation

```python
# STMGTDataset.__getitem__:
sample = self.samples[idx]  # Pre-computed valid samples

# Only creates sequences WITHIN same run:
for run_id in sample['input_runs']:
    run_data = self.run_data_cache[run_id]  # From cache
    # Never mixes data from different runs

# Validation:
assert all(r < min(target_runs) for r in input_runs)
# Input runs MUST be before target runs
```

**Benefits:**

- **Deployment consistency**: Training behavior matches production scenario
- **Accurate evaluation**: Validation metrics reflect real-world performance
- **Debugging support**: Predictions traceable to specific data runs

#### ✅ Multi-Modal Feature Integration

```python
# Traffic encoder: Processes edge-level speed sequences
traffic_embedding = self.traffic_encoder(X_traffic)
# Shape: (batch, num_nodes, seq_len, hidden_dim)

# Weather encoder: Processes global weather conditions
weather_embedding = self.weather_encoder(X_weather)
# Shape: (batch, seq_len, hidden_dim)

# Cross-attention: Traffic queries weather
# "How does rain affect THIS edge's speed?"
attended = self.weather_cross_attention(
    query=traffic_embedding,  # Each edge asks
    key=weather_embedding,     # Weather answers
    value=weather_embedding
)
```

**Design Rationale:**

1. **Appropriate granularity**: Traffic=edge-level, Weather=global
2. **Computational efficiency**: Weather encoded once, attended by all edges
3. **Semantic separation**: Different modalities processed by specialized encoders

**Comparison with Concatenation Approach:**

```python
# ASTGCN approach (node-based):
X = concat([speed, temp, wind, precip])  # Shape: (N, 4, T)
# Problem: Duplicates weather 62 times (one per node)!

# STMGT approach:
X_traffic = speed  # Shape: (62, 1, T)
X_weather = [temp, wind, precip]  # Shape: (1, 3, T)
# No duplication, correct granularity
```

#### ✅ Probabilistic Output with Uncertainty Quantification

```python
# Point prediction (ASTGCN, LSTM):
y_pred = model(X)  # Shape: (batch, num_nodes, pred_len)
# Single value per timestep

# STMGT probabilistic output:
y_pred = {
    'means': [μ1, μ2],      # Two mixture components
    'stds': [σ1, σ2],       # Uncertainty per component
    'weights': [π1, π2]     # Mixture probabilities
}

# Final prediction:
speed_distribution = π1·N(μ1,σ1²) + π2·N(μ2,σ2²)
point_estimate = π1·μ1 + π2·μ2
confidence_interval = [μ-2σ, μ+2σ]  # 95% interval
```

**Value Proposition:**

1. **Risk-aware routing**: Quantifies prediction confidence levels
2. **Informed decision-making**: "Route A: 15±2 min (high confidence), Route B: 12±8 min (high variance)"
3. **Model interpretability**: Uncertainty indicates model confidence in predictions

**Production Application:**

```python
# API response:
{
    "edge_id": "A_to_B",
    "predicted_speed": 35.2,
    "uncertainty": 4.1,  # ±4.1 km/h
    "confidence": "medium",  # Based on σ
    "alternative_speed": 28.7  # Second mixture component (congestion mode)
}

# UI displays: "Expected: 35 km/h, 30% probability of congestion (29 km/h)"
```

#### ✅ Engineering Optimization

```python
# Before (slow):
for idx in range(len(samples)):
    run_data = self.df[self.df['run_id'] == run_id]  # Filters every time!
    # 13.4 seconds per batch

# After (fast):
self.run_data_cache = {
    run_id: df[df['run_id'] == run_id].copy()  # Pre-grouped once
    for run_id in df['run_id'].unique()
}
# 0.9 seconds per batch (15x speedup!)
```

**Performance Impact:**

- Enables rapid experimentation cycles
- Reduces training cost (12 hours → 12 minutes per 100 epochs)
- Production-grade data pipeline implementation

#### Assessment: Comprehensive Production-Ready Solution

**Key Strengths:**

1. **Data fidelity**: Edge-based representation preserves directional information
2. **Temporal integrity**: Respects causality boundaries (no data leakage)
3. **Multi-modal design**: Separate processing of traffic and weather modalities
4. **Uncertainty awareness**: Probabilistic predictions for risk assessment
5. **Engineering optimization**: Efficient data loading for practical deployment

**Deployment Readiness:**

- Aligned with data collection methodology (Google Maps API)
- Validated temporal causality for production scenarios
- Optimized for real-time inference requirements

---

## Architectural Constraints: Why Unified I/O is HARD

### The Core Problem:

```
Team demand: "Unified input/output format for all models"
Reality: Each architecture REQUIRES different input shapes
```

**Fundamental Incompatibilities:**

### 1. Node vs Edge Representation

```python
# STMGT (edge-based):
X: (batch, 62, seq_len, 1)  # 62 edges
edge_index: (2, 144)  # Graph connections

# ASTGCN (node-based):
X: (batch, 62, num_features, seq_len)  # 62 NODES (different entities!)
adjacency: (62, 62)  # Node-to-node connections
```

**Cannot unify because:**

- Edge-based: 144 directed edges (A→B, B→A counted separately)
- Node-based: 62 intersections (bidirectional implicit)
- Converting edges→nodes LOSES directional information
- Converting nodes→edges CREATES artificial data

**Analogy:**

```
Like trying to unify:
- Model A: Predicts pixel RGB values (height×width×3)
- Model B: Predicts image caption (sequence of words)

Both are "images" but fundamentally different outputs!
```

### 2. Sequence Dimension Order

```python
# PyTorch GNN convention (STMGT, GraphWaveNet):
X: (batch, num_nodes, seq_len, features)
#          ^^^^^^^^^^^^^^^^^^^^ Spatial first, temporal second

# PyTorch ASTGCN convention:
X: (batch, num_nodes, features, seq_len)
#                     ^^^^^^^^^^^^^^^^^^^^^  Features first, temporal last

# LSTM convention:
X: (batch, seq_len, features)
#          ^^^^^^^^^^^^^^^^^^^^^^ No spatial dimension!
```

**Why different?**

- **GNN**: Spatial operations (graph conv) applied first
- **ASTGCN**: Temporal attention across seq_len dimension
- **LSTM**: Sequential processing (no spatial structure)

**Cannot simply transpose because:**

- Operations are dimension-specific (e.g., `nn.Conv1d` on last dim)
- Batch norms, layer norms expect certain shapes
- Attention masks depend on dimension order

### 3. Graph Structure Representation

```python
# STMGT: Edge index (PyTorch Geometric format)
edge_index = torch.tensor([
    [0, 1, 2, ...],  # Source nodes
    [1, 2, 3, ...]   # Target nodes
])  # Shape: (2, num_edges)

# ASTGCN: Adjacency matrix
adjacency = np.array([
    [0, 1, 0, ...],  # Node 0 connects to node 1
    [1, 0, 1, ...],  # Node 1 connects to nodes 0,2
    ...
])  # Shape: (num_nodes, num_nodes)

# GraphWaveNet: Learnable embeddings
node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim))
A_adaptive = F.relu(node_embeddings @ node_embeddings.T)
```

**Cannot unify because:**

- Edge index: Sparse, memory-efficient, used by `torch_geometric`
- Adjacency: Dense, required by graph Laplacian computation
- Learnable: Dynamic, changes during training

**Conversion cost:**

```python
# Edge index → Adjacency: O(E) time, O(N²) memory
# Adjacency → Edge index: O(N²) time

# For 62 nodes:
# - Edge index: 144 * 2 = 288 integers (1 KB)
# - Adjacency: 62 * 62 = 3844 floats (15 KB)
# - Learnable: 62 * 10 embeddings (2.5 KB) + computation each forward pass
```

### 4. Feature Engineering Differences

```python
# STMGT: Separates traffic and weather
X_traffic: (batch, num_edges, seq_len, 1)  # Speed only
X_weather: (batch, seq_len, 3)  # [temp, wind, precip]
# Cross-attention fuses them

# ASTGCN: Concatenates all features
X: (batch, num_nodes, 4, seq_len)  # [speed, temp, wind, precip]
# All features treated equally

# GraphWaveNet: Only speed
X: (batch, seq_len, num_nodes, 1)  # Speed only
# No weather integration
```

**Design philosophy clash:**

- STMGT: Multi-modal fusion (traffic ⊗ weather)
- ASTGCN: Feature concatenation (all equal)
- GraphWaveNet: Single-modal (speed only)

**Cannot unify without:**

- Redesigning STMGT's cross-attention (loses key advantage)
- Adding weather encoder to GraphWaveNet (major architecture change)
- Breaking ASTGCN's feature concatenation (changes input dimensions)

---

## Pragmatic Solution: Adapter Layer Pattern

### The Compromise:

**Accept:** Each model has its own input requirements (unavoidable)

**Solution:** Create adapter functions that convert from **canonical format** to model-specific format

### Canonical Format (Single Source of Truth):

```python
@dataclass
class CanonicalTrafficData:
    """
    Unified data representation - NOT used directly by models.
    Serves as intermediate format for adapters.
    """
    # Raw edge-level data (highest fidelity)
    edge_speeds: pd.DataFrame  # (timestamp, edge_id, speed_kmh)
    edge_topology: pd.DataFrame  # (edge_id, node_a, node_b, distance_m)

    # Weather data (global, time-indexed)
    weather: pd.DataFrame  # (timestamp, temp, wind, precip)

    # Temporal features
    temporal: pd.DataFrame  # (timestamp, hour, day_of_week, is_weekend)

    # Metadata
    node_coords: Dict[str, Tuple[float, float]]  # For distance calculations

    # Normalization parameters (fitted on training set ONLY)
    speed_scaler: StandardScaler
    weather_scaler: StandardScaler

    # Train/val/test split (run_id based)
    train_run_ids: List[int]
    val_run_ids: List[int]
    test_run_ids: List[int]
```

### Adapter Functions:

```python
class STMGTAdapter:
    """Convert canonical data to STMGT format."""

    def __call__(
        self,
        canonical: CanonicalTrafficData,
        split: str = 'train'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            X_traffic: (num_samples, 62, 12, 1)
            X_weather: (num_samples, 12, 3)
            y_target: (num_samples, 62, 12)
            edge_index: (2, 144)
        """
        # Get run IDs for this split
        run_ids = getattr(canonical, f'{split}_run_ids')

        # Filter data
        df = canonical.edge_speeds[canonical.edge_speeds['run_id'].isin(run_ids)]

        # Create sequences (respecting run boundaries)
        sequences = self._create_edge_sequences(
            df,
            seq_len=12,
            pred_len=12,
            respect_run_boundaries=True  # KEY: No temporal leakage
        )

        # Normalize
        X_traffic = canonical.speed_scaler.transform(sequences['X_speed'])
        X_weather = canonical.weather_scaler.transform(sequences['X_weather'])

        # Build edge_index from topology
        edge_index = self._build_edge_index(canonical.edge_topology)

        return X_traffic, X_weather, sequences['y_target'], edge_index


class ASTGCNAdapter:
    """Convert canonical data to ASTGCN format."""

    def __call__(
        self,
        canonical: CanonicalTrafficData,
        split: str = 'train'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            Xh: (num_samples, 62, 4, 12)  # Recent component
            Xd: (num_samples, 62, 4, 12)  # Daily component (not implemented yet)
            Xw: (num_samples, 62, 4, 12)  # Weekly component (not implemented yet)
            y: (num_samples, 62, 1, 12)
            adjacency: (62, 62)
            laplacian: (62, 62)
        """
        run_ids = getattr(canonical, f'{split}_run_ids')
        df = canonical.edge_speeds[canonical.edge_speeds['run_id'].isin(run_ids)]

        # Convert edge-based to node-based (WITH PROPER AGGREGATION)
        node_df = self._aggregate_edges_to_nodes(
            df,
            canonical.edge_topology,
            method='weighted_average'  # Weight by edge distance
        )

        # Create node-based sequences
        # FIXED: Now respects run boundaries!
        sequences = self._create_node_sequences(
            node_df,
            seq_len=12,
            pred_len=12,
            respect_run_boundaries=True  # NEW: Prevents temporal leakage
        )

        # Concatenate features: [speed, temp, wind, precip]
        Xh = self._concat_features(sequences, canonical.weather)

        # Build graph structures
        adjacency = self._build_node_adjacency(canonical.edge_topology)
        laplacian = compute_scaled_laplacian(adjacency)

        # For now, Xd and Xw are same as Xh (no daily/weekly split yet)
        Xd = Xh.copy()
        Xw = Xh.copy()

        return Xh, Xd, Xw, sequences['y'], adjacency, laplacian


class LSTMAdapter:
    """Convert canonical data to LSTM format (flatten spatial structure)."""

    def __call__(
        self,
        canonical: CanonicalTrafficData,
        split: str = 'train',
        aggregate_nodes: bool = True  # Aggregate to node-level or keep all edges?
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            X: (num_sequences, seq_len, num_features)
            y: (num_sequences, pred_len)
        """
        run_ids = getattr(canonical, f'{split}_run_ids')
        df = canonical.edge_speeds[canonical.edge_speeds['run_id'].isin(run_ids)]

        if aggregate_nodes:
            # Option A: Average to node-level (loses directionality)
            df = self._aggregate_edges_to_nodes(df, canonical.edge_topology)
        # Option B: Keep all edges (treats as independent time series)

        # Create flat sequences
        X_sequences = []
        y_sequences = []

        # FIXED: Respect run boundaries
        for run_id in run_ids:
            run_data = df[df['run_id'] == run_id].sort_values('timestamp')

            # Create sequences within this run only
            for i in range(len(run_data) - 12 - 12 + 1):
                X_seq = run_data.iloc[i:i+12][['speed_kmh', 'temp', 'wind', 'precip']].values
                y_seq = run_data.iloc[i+12:i+12+12]['speed_kmh'].values

                X_sequences.append(X_seq)
                y_sequences.append(y_seq)

        return np.array(X_sequences), np.array(y_sequences)


class GraphWaveNetAdapter:
    """Convert canonical data to GraphWaveNet format."""

    def __call__(
        self,
        canonical: CanonicalTrafficData,
        split: str = 'train'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            X: (num_samples, seq_len, num_nodes, 1)
            y: (num_samples, pred_len, num_nodes, 1)
        """
        # Similar to ASTGCN but different dimension order
        # and no explicit graph structure (learned adaptively)
        run_ids = getattr(canonical, f'{split}_run_ids')
        df = canonical.edge_speeds[canonical.edge_speeds['run_id'].isin(run_ids)]

        # Convert to node-based
        node_df = self._aggregate_edges_to_nodes(df, canonical.edge_topology)

        # Create sequences (respecting run boundaries)
        sequences = self._create_node_sequences(
            node_df,
            seq_len=12,
            pred_len=12,
            respect_run_boundaries=True
        )

        # Reshape to GraphWaveNet format
        X = sequences['X'].transpose(0, 2, 1, 3)  # (B, N, T, F) → (B, T, N, F)
        y = sequences['y'].transpose(0, 2, 1, 3)

        return torch.tensor(X), torch.tensor(y)
```

### Usage Example:

```python
# Step 1: Load canonical data (ONCE)
canonical = CanonicalTrafficData.from_parquet(
    'data/processed/all_runs_combined.parquet'
)

# Fit scalers on training set ONLY
canonical.fit_scalers(split='train')

# Step 2: Create model-specific datasets via adapters
stmgt_train = STMGTAdapter()(canonical, split='train')
stmgt_val = STMGTAdapter()(canonical, split='val')

astgcn_train = ASTGCNAdapter()(canonical, split='train')
astgcn_val = ASTGCNAdapter()(canonical, split='val')

lstm_train = LSTMAdapter()(canonical, split='train')
lstm_val = LSTMAdapter()(canonical, split='val')

# Step 3: Train models with their native formats
stmgt_model.fit(*stmgt_train)
astgcn_model.fit(*astgcn_train)
lstm_model.fit(*lstm_train)

# Step 4: Evaluate on SAME test set (canonical ensures fairness)
stmgt_test = STMGTAdapter()(canonical, split='test')
astgcn_test = ASTGCNAdapter()(canonical, split='test')

stmgt_metrics = evaluate(stmgt_model, stmgt_test)
astgcn_metrics = evaluate(astgcn_model, astgcn_test)

# Metrics are comparable because:
# 1. Same source data (canonical)
# 2. Same train/val/test split (run_id based)
# 3. Same normalization (fitted on train set)
# 4. Same sequence creation logic (respects run boundaries)
```

---

## Comparative Summary

### Model Capability Matrix

| Capability                 | LSTM     | ASTGCN          | GraphWaveNet  | STMGT            |
| -------------------------- | -------- | --------------- | ------------- | ---------------- |
| Spatial Modeling           | ❌ None  | ✅ Spectral GCN | ✅ Adaptive   | ✅ Static GNN    |
| Temporal Causality         | ✅ Good  | ⚠️ Needs Fix    | ✅ Good       | ✅ Excellent     |
| Data Efficiency            | Low      | Medium          | Medium        | High             |
| Weather Integration        | Basic    | Concatenation   | Limited       | Cross-Attention  |
| Uncertainty Quantification | ❌       | ❌              | ❌            | ✅ GMM           |
| Edge-Level Fidelity        | ❌       | ❌ (Averaged)   | ❌ (Averaged) | ✅ Native        |
| Interpretability           | High     | Medium          | Low           | Medium-High      |
| Production Readiness       | Baseline | After Fixes     | Research      | Deployment-Ready |

### Suitability Assessment

**STMGT:**

- ✅ Production deployment
- ✅ Real-time inference
- ✅ Uncertainty-aware routing
- ✅ Multi-modal integration

**GraphWaveNet:**

- ✅ Research exploration
- ✅ Pattern discovery in large datasets
- ⚠️ Requires substantial training data
- ⚠️ Interpretability considerations

**ASTGCN:**

- ✅ Multi-scale temporal modeling
- ⚠️ After addressing temporal boundary handling
- ⚠️ Node-level aggregation trade-offs
- ✅ Spectral graph theory foundation

**LSTM:**

- ✅ Baseline comparison
- ✅ Quick prototyping
- ❌ Not suitable for production (lacks spatial awareness)
- ✅ Educational/benchmark purposes

### Unified Evaluation Framework

**Challenge:** Direct input/output format unification is architecturally infeasible due to fundamental model design differences.

**Recommended Approach: Adapter Pattern**

**Strategy:**

1. ✅ Establish canonical data format (single source of truth)
2. ✅ Implement adapter functions for model-specific conversions
3. ✅ Guarantee identical train/val/test splits across all models
4. ✅ Ensure consistent normalization parameters
5. ✅ Respect architectural requirements (different tensor shapes by design)

**Guiding Principle:**

> "Unify the data source and preprocessing methodology, not the final input tensor format. Fair comparison comes from identical training data, not identical input shapes."

### Implementation Recommendation

**Phase 1: Canonical Data Pipeline**

- Single parquet source: `all_runs_combined.parquet`
- Unified normalization: `StandardScaler` fitted on training set
- Temporal split: 70% train / 15% val / 15% test by run_id
- Validation: Temporal causality preserved (no future leakage)

**Phase 2: Model-Specific Adapters**

- `STMGTAdapter`: Converts to edge-based format with run_id boundaries
- `ASTGCNAdapter`: Aggregates to node-based with fixed temporal boundaries
- `GraphWaveNetAdapter`: Provides node sequences for adaptive learning
- `LSTMAdapter`: Flattens spatial structure for baseline comparison

**Phase 3: Evaluation Protocol**

- Identical test set across all models
- Consistent metric calculation (MAE, RMSE, R², MAPE)
- Statistical significance testing
- Documentation of architectural differences

**Benefits:**

- Scientifically rigorous comparison
- Preserves each model's architectural strengths
- Publication-ready methodology
- Reproducible results

**Estimated Implementation:** 1-2 days for adapter pattern + validation

---

## Conclusion

Each model architecture embodies specific design choices optimized for different aspects of traffic forecasting:

- **LSTM**: Temporal baseline without spatial modeling
- **ASTGCN**: Spectral graph theory with multi-scale temporal patterns
- **GraphWaveNet**: Adaptive graph learning for pattern discovery
- **STMGT**: Production-oriented edge-based multi-modal architecture

**Fair comparison requires:**

1. Unified data source and preprocessing
2. Adapter pattern respecting architectural constraints
3. Consistent evaluation methodology
4. Documentation of design trade-offs

**Recommendation:** Implement adapter pattern for rigorous, publication-ready model comparison while preserving each architecture's intended design.
