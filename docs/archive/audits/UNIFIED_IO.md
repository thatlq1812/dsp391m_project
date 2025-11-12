# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Unified Input/Output Format: Technical Analysis & Proposal

**Date:** November 9, 2025

---

## Why Direct Unification is Technically Infeasible

### 1. Fundamental Architectural Differences

**Input Shape Incompatibility:**

```python
# STMGT (Edge-based GNN):
X_traffic: (batch, 62, 12, 1)  # 62 edges, 12 timesteps
X_weather: (batch, 12, 3)      # Global weather
edge_index: (2, 144)           # Graph structure

# ASTGCN (Node-based GCN):
X: (batch, 62, 4, 12)          # 62 NODES, 4 features, 12 timesteps
adjacency: (62, 62)            # Dense adjacency matrix
laplacian: (62, 62)            # Scaled Laplacian

# GraphWaveNet (Adaptive graph):
X: (batch, 12, 62, 1)          # Different dimension order!
node_embeddings: (62, 10)      # Learnable graph structure

# LSTM (No spatial structure):
X: (batch, 12, features)       # Flattened, no graph
```

**Why these differences exist:**

- **STMGT**: Needs edge_index format for PyTorch Geometric operations
- **ASTGCN**: Requires dense matrices for Laplacian computation
- **GraphWaveNet**: Learns adaptive adjacency (no fixed graph needed)
- **LSTM**: No graph awareness by design

**Forcing same shape would require:**

1. Redesigning all model architectures (weeks of work, breaks validated implementations)
2. Information loss (edges→nodes loses 57% data granularity)
3. Artificial data generation (nodes→edges creates fake values)

### 2. Data Representation Trade-offs

**Edge-based (STMGT) vs Node-based (ASTGCN):**

```
Real data: 144 directed edges (A→B, B→A are different)

Edge-based:
  - Edge A→B: 70 km/h
  - Edge B→A: 45 km/h
  → Preserves directional information ✓

Node-based (requires aggregation):
  - Node A: mean([70, 45, ...]) = 52 km/h
  → Loses directionality, averages heterogeneous roads ✗
```

**Converting between formats:**

- Edges → Nodes: Information loss (directional data averaged)
- Nodes → Edges: Information fabrication (creating non-existent directional data)

### 3. Graph Structure Representations

| Model        | Graph Type           | Storage            | Operations            |
| ------------ | -------------------- | ------------------ | --------------------- |
| STMGT        | Static edge_index    | Sparse (2×144)     | PyTorch Geometric GNN |
| ASTGCN       | Laplacian matrix     | Dense (62×62)      | Spectral convolution  |
| GraphWaveNet | Learnable embeddings | Parameters (62×10) | Adaptive adjacency    |
| LSTM         | None                 | N/A                | Sequential only       |

**Cannot unify because:** Each representation is hardcoded into model operations (GNN layers, Laplacian multiplication, embedding lookups).

---

## Recommended Solution: Adapter Pattern

### Core Concept

**Don't unify input shapes → Unify data source and preprocessing**

```
┌─────────────────────────────────┐
│  Canonical Data (Single Source) │
│  - Same parquet file            │
│  - Same train/val/test split    │
│  - Same normalization params    │
└────────────┬────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
┌───▼────┐       ┌───▼────┐       ┌──────────┐       ┌───────┐
│STMGT   │       │ASTGCN  │       │GraphWave │       │LSTM   │
│Adapter │       │Adapter │       │Net       │       │Adapter│
│        │       │        │       │Adapter   │       │       │
└───┬────┘       └───┬────┘       └────┬─────┘       └───┬───┘
    │                │                 │                 │
┌───▼─────┐     ┌───▼──────┐      ┌────▼────┐       ┌────▼───┐
│Edge-    │     │Node-     │      │Adaptive │       │Flat    │
│based    │     │based     │      │graph    │       │sequence│
│tensors  │     │tensors   │      │tensors  │       │        │
└─────────┘     └──────────┘      └─────────┘       └────────┘
```

### Implementation

**Step 1: Canonical Data Class**

```python
@dataclass
class CanonicalTrafficData:
    """Single source of truth for all models."""

    # Raw data (highest fidelity)
    edge_df: pd.DataFrame  # (timestamp, edge_id, speed, ...)
    weather_df: pd.DataFrame  # (timestamp, temp, wind, precip)
    topology_df: pd.DataFrame  # (edge_id, node_a, node_b)

    # Fitted scalers (on training set ONLY)
    speed_scaler: StandardScaler
    weather_scaler: StandardScaler

    # Train/val/test splits (same for all models)
    train_run_ids: List[int]  # [0, 1, 2, ..., 999]
    val_run_ids: List[int]    # [1000, 1001, ..., 1213]
    test_run_ids: List[int]   # [1214, 1215, ..., 1429]

    @classmethod
    def from_parquet(cls, path: str) -> 'CanonicalTrafficData':
        """Load from parquet and create splits."""
        df = pd.read_parquet(path)

        # Create temporal splits by run_id
        run_ids = sorted(df['run_id'].unique())
        n_train = int(len(run_ids) * 0.7)
        n_val = int(len(run_ids) * 0.15)

        # Fit scalers on training set ONLY
        train_df = df[df['run_id'].isin(run_ids[:n_train])]
        speed_scaler = StandardScaler()
        weather_scaler = StandardScaler()
        speed_scaler.fit(train_df[['speed_kmh']])
        weather_scaler.fit(train_df[['temperature_c', 'wind_speed_kmh', 'precipitation_mm']])

        return cls(
            edge_df=df,
            weather_df=df[['timestamp', 'temperature_c', 'wind_speed_kmh', 'precipitation_mm']].drop_duplicates(),
            topology_df=df[['edge_id', 'node_a_id', 'node_b_id']].drop_duplicates(),
            speed_scaler=speed_scaler,
            weather_scaler=weather_scaler,
            train_run_ids=run_ids[:n_train],
            val_run_ids=run_ids[n_train:n_train+n_val],
            test_run_ids=run_ids[n_train+n_val:]
        )
```

**Step 2: Model-Specific Adapters**

```python
class STMGTAdapter:
    """Convert canonical data to STMGT format."""

    def __call__(
        self,
        canonical: CanonicalTrafficData,
        split: str = 'train'
    ) -> STMGTDataLoader:
        """
        Returns DataLoader with:
        - X_traffic: (batch, 62, 12, 1) edge speeds
        - X_weather: (batch, 12, 3) global weather
        - y_target: (batch, 62, 12) predictions
        - edge_index: (2, 144) static graph
        """
        run_ids = getattr(canonical, f'{split}_run_ids')
        df = canonical.edge_df[canonical.edge_df['run_id'].isin(run_ids)]

        # Create edge-based sequences (respects run_id boundaries)
        dataset = STMGTDataset(
            df=df,
            seq_len=12,
            pred_len=12,
            speed_normalizer=canonical.speed_scaler,
            weather_normalizer=canonical.weather_scaler
        )

        return DataLoader(dataset, batch_size=64, shuffle=(split=='train'))


class ASTGCNAdapter:
    """Convert canonical data to ASTGCN format."""

    def __call__(
        self,
        canonical: CanonicalTrafficData,
        split: str = 'train'
    ) -> ASTGCNDataLoader:
        """
        Returns DataLoader with:
        - Xh: (batch, 62, 4, 12) node features [speed, temp, wind, precip]
        - Xd: (batch, 62, 4, 12) daily component (placeholder)
        - Xw: (batch, 62, 4, 12) weekly component (placeholder)
        - y: (batch, 62, 1, 12) predictions
        - adjacency: (62, 62) node adjacency matrix
        - laplacian: (62, 62) scaled Laplacian
        """
        run_ids = getattr(canonical, f'{split}_run_ids')
        df = canonical.edge_df[canonical.edge_df['run_id'].isin(run_ids)]

        # Convert edges to nodes (weighted aggregation)
        node_df = self._aggregate_edges_to_nodes(df, canonical.topology_df)

        # Create node-based sequences (NOW respects run_id boundaries - FIXED!)
        sequences = self._create_node_sequences(
            node_df,
            seq_len=12,
            pred_len=12,
            respect_run_boundaries=True  # KEY FIX
        )

        # Concatenate features
        X = np.stack([
            canonical.speed_scaler.transform(sequences['speed']),
            canonical.weather_scaler.transform(sequences['weather'])
        ], axis=2)  # (batch, nodes, features, time)

        # Build graph structures
        adjacency = self._build_adjacency(canonical.topology_df)
        laplacian = compute_scaled_laplacian(adjacency)

        return DataLoader(
            ASTGCNDataset(X, X.copy(), X.copy(), sequences['y'], adjacency, laplacian),
            batch_size=64,
            shuffle=(split=='train')
        )


class GraphWaveNetAdapter:
    """Convert canonical data to GraphWaveNet format."""

    def __call__(
        self,
        canonical: CanonicalTrafficData,
        split: str = 'train'
    ) -> GraphWaveNetDataLoader:
        """
        Returns DataLoader with:
        - X: (batch, 12, 62, 1) node speeds (different dim order!)
        - y: (batch, 12, 62, 1) predictions
        Note: No explicit graph structure (learned adaptively)
        """
        run_ids = getattr(canonical, f'{split}_run_ids')
        df = canonical.edge_df[canonical.edge_df['run_id'].isin(run_ids)]

        # Convert to nodes
        node_df = self._aggregate_edges_to_nodes(df, canonical.topology_df)

        # Create sequences
        sequences = self._create_node_sequences(
            node_df,
            seq_len=12,
            pred_len=12,
            respect_run_boundaries=True
        )

        # Normalize and reshape to (batch, time, nodes, features)
        X = canonical.speed_scaler.transform(sequences['X'])
        X = X.transpose(0, 2, 1, 3)  # (B, N, T, F) → (B, T, N, F)

        return DataLoader(
            GraphWaveNetDataset(X, sequences['y']),
            batch_size=64,
            shuffle=(split=='train')
        )


class LSTMAdapter:
    """Convert canonical data to LSTM format (flatten spatial structure)."""

    def __call__(
        self,
        canonical: CanonicalTrafficData,
        split: str = 'train'
    ) -> LSTMDataLoader:
        """
        Returns DataLoader with:
        - X: (batch, 12, 5) flat sequences [speed, temp, wind, precip, hour]
        - y: (batch, 12) predictions
        """
        run_ids = getattr(canonical, f'{split}_run_ids')
        df = canonical.edge_df[canonical.edge_df['run_id'].isin(run_ids)]

        # Flatten: treat each edge as independent time series
        X_sequences = []
        y_sequences = []

        for run_id in run_ids:
            run_data = df[df['run_id'] == run_id].sort_values('timestamp')

            for i in range(len(run_data) - 12 - 12 + 1):
                X_seq = run_data.iloc[i:i+12][['speed_kmh', 'temperature_c', 'wind_speed_kmh', 'precipitation_mm']].values
                y_seq = run_data.iloc[i+12:i+24]['speed_kmh'].values

                X_sequences.append(X_seq)
                y_sequences.append(y_seq)

        X = canonical.speed_scaler.transform(np.array(X_sequences).reshape(-1, 4)).reshape(-1, 12, 4)
        y = canonical.speed_scaler.transform(np.array(y_sequences).reshape(-1, 1)).reshape(-1, 12)

        return DataLoader(
            LSTMDataset(X, y),
            batch_size=64,
            shuffle=(split=='train')
        )
```

**Step 3: Unified Training & Evaluation**

```python
# Load canonical data ONCE
canonical = CanonicalTrafficData.from_parquet('data/processed/all_runs_combined.parquet')

# Create adapters
stmgt_adapter = STMGTAdapter()
astgcn_adapter = ASTGCNAdapter()
graph_wavenet_adapter = GraphWaveNetAdapter()
lstm_adapter = LSTMAdapter()

# Get model-specific data loaders
stmgt_train = stmgt_adapter(canonical, split='train')
stmgt_val = stmgt_adapter(canonical, split='val')
stmgt_test = stmgt_adapter(canonical, split='test')

astgcn_train = astgcn_adapter(canonical, split='train')
astgcn_val = astgcn_adapter(canonical, split='val')
astgcn_test = astgcn_adapter(canonical, split='test')

# Train models with their native formats
stmgt_model.fit(stmgt_train, stmgt_val)
astgcn_model.fit(astgcn_train, astgcn_val)

# Evaluate on SAME test set (canonical ensures fairness)
stmgt_results = evaluate(stmgt_model, stmgt_test)
astgcn_results = evaluate(astgcn_model, astgcn_test)

# Compare results (now fair because same source data!)
comparison = pd.DataFrame({
    'Model': ['STMGT', 'ASTGCN', 'GraphWaveNet', 'LSTM'],
    'MAE': [stmgt_results.mae, astgcn_results.mae, gwn_results.mae, lstm_results.mae],
    'RMSE': [stmgt_results.rmse, astgcn_results.rmse, gwn_results.rmse, lstm_results.rmse]
})
```

---

## What This Achieves

### Fair Comparison Guarantees

**Same training data**

- All models use identical run_ids for train/val/test
- No temporal leakage (adapters enforce run_id boundaries)

**Same normalization**

- Single StandardScaler fitted on training set
- Applied consistently to all models

**Same evaluation protocol**

- Identical test set (same timestamps, same edges/nodes)
- Consistent metric calculation (MAE, RMSE, R², MAPE)

**Preserves architectural strengths**

- STMGT: Keeps edge-level granularity
- ASTGCN: Uses node-based Laplacian operations
- GraphWaveNet: Learns adaptive adjacency
- LSTM: Baseline with no spatial constraints

### Comparison is Valid Because:

**Same input data source** → Different tensor shapes → **Same evaluation targets**

```
Example: Predicting speed on edge "A→B" at timestamp T:

STMGT predicts:    edge_62[T] = 35.2 km/h (direct)
ASTGCN predicts:   node_A[T] = 38.1 km/h (averaged with other edges)
GraphWaveNet:      node_A[T] = 37.5 km/h (learned aggregation)
LSTM predicts:     edge_62[T] = 34.8 km/h (no spatial context)

Ground truth:      edge_62[T] = 36.0 km/h (from canonical test set)

MAE calculation:
- STMGT: |35.2 - 36.0| = 0.8 km/h
- ASTGCN: |38.1 - 36.0| = 2.1 km/h (node averaging introduces error)
- GraphWaveNet: |37.5 - 36.0| = 1.5 km/h
- LSTM: |34.8 - 36.0| = 1.2 km/h
```

**Comparison is fair because:**

- All models predict same physical quantity (traffic speed)
- All models evaluated on same ground truth values
- Differences reflect architectural choices, not data inconsistencies

---
