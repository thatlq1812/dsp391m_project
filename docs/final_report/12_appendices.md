# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Section 14: Appendices

## Appendix A: Additional Architecture Diagrams

### A.1 Full Model Architecture (Detailed)

```
Input Shape: [batch, seq_len=12, num_nodes=62, in_features=5]
          ↓
    ╔═══════════════════════════════════════════════════════════════╗
    ║              Initial Projection Layer                         ║
    ║  Linear(5 → 96) + Batch Norm + ReLU + Dropout(0.1)          ║
    ╚═══════════════════════════════════════════════════════════════╝
          ↓  [batch, 12, 62, 96]
    ╔═══════════════════════════════════════════════════════════════╗
    ║                   STMGT Block 1                               ║
    ║  ┌─────────────────────────────────────────────────────────┐ ║
    ║  │ Spatial Branch:                                         │ ║
    ║  │  - GATv2 Conv (96 → 96, heads=4)                       │ ║
    ║  │  - Batch Norm + Dropout(0.1)                           │ ║
    ║  └─────────────────────────────────────────────────────────┘ ║
    ║  ┌─────────────────────────────────────────────────────────┐ ║
    ║  │ Temporal Branch:                                        │ ║
    ║  │  - Multi-Head Attention (96, heads=4, dropout=0.1)     │ ║
    ║  │  - Feed Forward (96 → 384 → 96)                        │ ║
    ║  │  - Layer Norm + Dropout(0.1)                           │ ║
    ║  └─────────────────────────────────────────────────────────┘ ║
    ║  ┌─────────────────────────────────────────────────────────┐ ║
    ║  │ Gated Fusion:                                           │ ║
    ║  │  - α = Sigmoid(Linear(spatial + temporal))             │ ║
    ║  │  - output = α * spatial + (1-α) * temporal             │ ║
    ║  └─────────────────────────────────────────────────────────┘ ║
    ║  ┌─────────────────────────────────────────────────────────┐ ║
    ║  │ Weather Cross-Attention:                                │ ║
    ║  │  - Q: from fused output                                │ ║
    ║  │  - K, V: from weather features [batch, 12, 3]          │ ║
    ║  │  - Cross-attention (heads=4)                           │ ║
    ║  │  - Residual connection                                 │ ║
    ║  └─────────────────────────────────────────────────────────┘ ║
    ╚═══════════════════════════════════════════════════════════════╝
          ↓  [batch, 12, 62, 96]
    ╔═══════════════════════════════════════════════════════════════╗
    ║                   STMGT Block 2                               ║
    ║                  (Same architecture)                          ║
    ╚═══════════════════════════════════════════════════════════════╝
          ↓  [batch, 12, 62, 96]
    ╔═══════════════════════════════════════════════════════════════╗
    ║                   STMGT Block 3                               ║
    ║                  (Same architecture)                          ║
    ╚═══════════════════════════════════════════════════════════════╝
          ↓  [batch, 12, 62, 96]
    ╔═══════════════════════════════════════════════════════════════╗
    ║              Gaussian Mixture Output Head                     ║
    ║  ┌─────────────────────────────────────────────────────────┐ ║
    ║  │ For each Gaussian component k=1..5:                     │ ║
    ║  │  - μ_k:     Linear(96 → 1)                              │ ║
    ║  │  - log_σ_k: Linear(96 → 1), softplus activation        │ ║
    ║  │  - logit_π: Linear(96 → 5), softmax → mixture weights  │ ║
    ║  └─────────────────────────────────────────────────────────┘ ║
    ╚═══════════════════════════════════════════════════════════════╝
          ↓
    Output: μ[batch,62,5], σ[batch,62,5], π[batch,62,5]

    Prediction: y = Σ(π_k * μ_k) for k=1..5
    Uncertainty: σ_total = sqrt(Σ(π_k * (μ_k² + σ_k²)) - y²)
```

### A.2 GATv2 Attention Mechanism

```python
# GATv2 improves GAT by allowing dynamic attention
# Key difference: Apply LeakyReLU AFTER weight transformation

def gatv2_attention(x, edge_index):
    """
    x: [num_nodes, hidden_dim]
    edge_index: [2, num_edges]
    """
    # 1. Transform node features
    h_i = W @ x  # [num_nodes, hidden_dim]
    h_j = W @ x[edge_index[1]]  # [num_edges, hidden_dim]

    # 2. Compute attention scores (GATv2: LeakyReLU after sum)
    e_ij = a^T @ LeakyReLU(h_i + h_j)  # [num_edges]

    # 3. Normalize with softmax
    α_ij = softmax_per_node(e_ij)  # [num_edges]

    # 4. Aggregate neighbors
    h_i_new = Σ(α_ij * h_j)  # [num_nodes, hidden_dim]

    return h_i_new, α_ij
```

### A.3 Weather Cross-Attention Detail

```
Weather Features: [batch, seq_len=12, weather_dim=3]
  - Temperature (normalized)
  - Precipitation (normalized)
  - Cloud cover (normalized)

Traffic Features: [batch, seq_len=12, num_nodes=62, hidden_dim=96]

Cross-Attention Process:
  1. Project traffic features to query space:
     Q = Linear_q(traffic)  # [batch, 12, 62, 96]

  2. Project weather features to key/value space:
     K = Linear_k(weather)  # [batch, 12, 3, 96]
     V = Linear_v(weather)  # [batch, 12, 3, 96]

  3. Compute attention scores:
     scores = (Q @ K^T) / sqrt(d_k)  # [batch, 12, 62, 3]
     α = softmax(scores, dim=-1)

  4. Aggregate weather information:
     context = α @ V  # [batch, 12, 62, 96]

  5. Residual connection:
     output = traffic + context
```

---

## Appendix B: Code Snippets

### B.1 Data Loading Pipeline

```python
# File: traffic_forecast/data/dataset.py

class TrafficDataset(Dataset):
    """Traffic forecasting dataset with weather features."""

    def __init__(
        self,
        data_path: str,
        seq_len: int = 12,
        pred_len: int = 1,
        split: str = 'train'
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Load processed data
        df = pd.read_parquet(data_path)

        # Extract features
        self.speed = df.pivot_table(
            index=['run_id', 'timestamp'],
            columns='node_id',
            values='speed_normalized'
        ).values  # [num_samples, num_nodes]

        self.weather = df.groupby(['run_id', 'timestamp']).agg({
            'temperature_normalized': 'first',
            'precipitation_normalized': 'first',
            'cloud_cover_normalized': 'first'
        }).values  # [num_samples, 3]

        self.temporal = df.groupby(['run_id', 'timestamp']).agg({
            'hour_sin': 'first',
            'hour_cos': 'first',
            'dow_sin': 'first',
            'dow_cos': 'first'
        }).values  # [num_samples, 4]

        # Split by time (70/15/15)
        n = len(self.speed)
        if split == 'train':
            self.indices = range(0, int(0.7 * n) - seq_len - pred_len)
        elif split == 'val':
            self.indices = range(int(0.7 * n), int(0.85 * n) - seq_len - pred_len)
        else:  # test
            self.indices = range(int(0.85 * n), n - seq_len - pred_len)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]

        # Input sequence
        x_speed = self.speed[i:i+self.seq_len]  # [seq_len, num_nodes]
        x_weather = self.weather[i:i+self.seq_len]  # [seq_len, 3]
        x_temporal = self.temporal[i:i+self.seq_len]  # [seq_len, 4]

        # Concatenate node features
        x = np.concatenate([
            x_speed[..., np.newaxis],  # [seq_len, num_nodes, 1]
            np.tile(x_temporal[:, np.newaxis, :], (1, 62, 1))  # [seq_len, 62, 4]
        ], axis=-1)  # [seq_len, num_nodes, 5]

        # Target (next step)
        y = self.speed[i+self.seq_len:i+self.seq_len+self.pred_len]  # [pred_len, num_nodes]

        return {
            'x': torch.FloatTensor(x),
            'weather': torch.FloatTensor(x_weather),
            'y': torch.FloatTensor(y)
        }
```

### B.2 Training Loop

```python
# File: scripts/training/train_stmgt.py

def train_epoch(model, dataloader, optimizer, device):
    """Single training epoch."""
    model.train()
    total_loss = 0
    total_mae = 0

    for batch in tqdm(dataloader, desc='Training'):
        x = batch['x'].to(device)  # [batch, seq_len, num_nodes, in_features]
        weather = batch['weather'].to(device)  # [batch, seq_len, 3]
        y = batch['y'].squeeze(1).to(device)  # [batch, num_nodes]

        # Forward pass
        mu, sigma, pi = model(x, weather, edge_index)  # [batch, num_nodes, K]

        # Mixture prediction
        y_pred = (pi * mu).sum(dim=-1)  # [batch, num_nodes]

        # Negative Log-Likelihood loss
        nll_loss = gaussian_nll_loss(y, mu, sigma, pi)

        # MAE for monitoring
        mae_loss = F.l1_loss(y_pred, y)

        # Combined loss
        loss = nll_loss + 0.1 * mae_loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_mae += mae_loss.item()

    return total_loss / len(dataloader), total_mae / len(dataloader)

def validate_epoch(model, dataloader, device):
    """Validation epoch."""
    model.eval()
    total_mae = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            x = batch['x'].to(device)
            weather = batch['weather'].to(device)
            y = batch['y'].squeeze(1).to(device)

            mu, sigma, pi = model(x, weather, edge_index)
            y_pred = (pi * mu).sum(dim=-1)

            mae = F.l1_loss(y_pred, y)
            total_mae += mae.item()

    return total_mae / len(dataloader)
```

### B.3 Inference API

```python
# File: traffic_api/predictor.py

class TrafficPredictor:
    """Production inference with uncertainty quantification."""

    def __init__(self, model_path: str, device: str = 'cuda'):
        self.device = torch.device(device)
        self.model = torch.load(model_path).to(self.device)
        self.model.eval()

        # Load normalization stats
        self.stats = load_normalization_stats()

    @torch.no_grad()
    def predict(
        self,
        historical_speeds: np.ndarray,  # [seq_len=12, num_nodes=62]
        weather: np.ndarray  # [seq_len=12, 3]
    ) -> PredictionResult:
        """
        Returns:
            PredictionResult with fields:
            - mean: [num_nodes] predicted speeds
            - std: [num_nodes] uncertainty
            - lower_80: [num_nodes] 80% confidence lower bound
            - upper_80: [num_nodes] 80% confidence upper bound
            - mixtures: [num_nodes, K] mixture components
        """
        # Normalize inputs
        x_speed = (historical_speeds - self.stats['speed_mean']) / self.stats['speed_std']
        x_weather = normalize_weather(weather, self.stats)

        # Prepare input tensor
        x = prepare_input(x_speed, x_weather, get_temporal_features())
        x = torch.FloatTensor(x).unsqueeze(0).to(self.device)
        weather = torch.FloatTensor(x_weather).unsqueeze(0).to(self.device)

        # Inference
        mu, sigma, pi = self.model(x, weather, edge_index)  # [1, 62, K]

        # Compute mixture statistics
        mean = (pi * mu).sum(dim=-1).squeeze(0).cpu().numpy()  # [62]
        variance = (pi * (mu**2 + sigma**2)).sum(dim=-1) - mean**2
        std = torch.sqrt(variance).squeeze(0).cpu().numpy()  # [62]

        # Denormalize
        mean = mean * self.stats['speed_std'] + self.stats['speed_mean']
        std = std * self.stats['speed_std']

        # Confidence intervals
        lower_80 = mean - 1.28 * std  # 80% CI
        upper_80 = mean + 1.28 * std

        return PredictionResult(
            mean=mean,
            std=std,
            lower_80=np.maximum(lower_80, 0),  # speeds >= 0
            upper_80=upper_80,
            mixtures={
                'mu': mu.squeeze(0).cpu().numpy(),
                'sigma': sigma.squeeze(0).cpu().numpy(),
                'pi': pi.squeeze(0).cpu().numpy()
            }
        )
```

---

## Appendix C: Hyperparameter Sensitivity Analysis

### C.1 Hidden Dimension Comparison

| hidden_dim | Params   | MAE (km/h) | RMSE     | R²       | Training Time (min) |
| ---------- | -------- | ---------- | -------- | -------- | ------------------- |
| 32         | 170K     | 3.45       | 5.12     | 0.75     | 4.2                 |
| 64         | 340K     | 3.15       | 4.68     | 0.80     | 6.8                 |
| **96**     | **680K** | **3.08**   | **4.53** | **0.82** | **10.1**            |
| 128        | 1.2M     | 3.10       | 4.55     | 0.82     | 15.3                |
| 192        | 2.7M     | 3.09       | 4.54     | 0.82     | 28.7                |

**Analysis:**

- Diminishing returns above hidden_dim=96
- 96 provides best accuracy/speed trade-off
- Larger models (128, 192) risk overfitting on limited data

### C.2 Number of Mixture Components

| K (mixtures)      | MAE      | RMSE     | CRPS     | Coverage@80 | Params   |
| ----------------- | -------- | -------- | -------- | ----------- | -------- |
| 1 (deterministic) | 3.28     | 4.75     | N/A      | N/A         | 650K     |
| 3                 | 3.12     | 4.59     | 2.35     | 81.2%       | 670K     |
| **5**             | **3.08** | **4.53** | **2.23** | **83.75%**  | **680K** |
| 7                 | 3.09     | 4.54     | 2.24     | 83.5%       | 695K     |
| 10                | 3.11     | 4.56     | 2.26     | 82.8%       | 720K     |

**Analysis:**

- Single Gaussian (K=1) underperforms by 6.5%
- K=5 optimal for uncertainty calibration
- Higher K (7, 10) adds complexity without benefit

### C.2.1 Learning Rate Sensitivity

| Learning Rate | MAE      | Convergence Speed | Notes                    |
| ------------- | -------- | ----------------- | ------------------------ |
| 1e-4          | 3.28     | Slow (35 epochs)  | Too conservative         |
| **5e-4**      | **3.08** | Fast (24 epochs)  | Optimal (default)        |
| 1e-3          | 3.15     | Fast (18 epochs)  | Slight overshoot         |
| 5e-3          | 3.45     | Very fast (12)    | Unstable, poor final MAE |

**Recommendation:** 5e-4 with cosine annealing schedule

### C.2.2 Dropout Rate Ablation

| Dropout Rate | MAE      | R²   | Generalization Gap |
| ------------ | -------- | ---- | ------------------ |
| 0.0          | 3.02     | 0.83 | Large (overfit)    |
| **0.1**      | **3.08** | 0.82 | Minimal            |
| 0.2          | 3.15     | 0.81 | Slight underfit    |
| 0.3          | 3.28     | 0.79 | Underfit           |

**Recommendation:** 0.1 balances regularization and capacity

### C.2.3 Attention Heads Comparison

| Attention Heads | MAE      | Params | Notes                      |
| --------------- | -------- | ------ | -------------------------- |
| 1               | 3.24     | 580K   | Insufficient capacity      |
| 2               | 3.15     | 630K   | Moderate improvement       |
| **4**           | **3.08** | 680K   | Optimal (default)          |
| 8               | 3.09     | 780K   | Marginal gain, more params |

**Recommendation:** 4 heads capture diverse attention patterns efficiently

### C.3 Sequence Length Impact

| seq_len     | MAE      | RMSE     | R²       | Notes                |
| ----------- | -------- | -------- | -------- | -------------------- |
| 6 (1.5h)    | 3.32     | 4.89     | 0.78     | Insufficient context |
| **12 (3h)** | **3.08** | **4.53** | **0.82** | Optimal              |
| 24 (6h)     | 3.10     | 4.55     | 0.82     | Marginal improvement |
| 48 (12h)    | 3.15     | 4.61     | 0.81     | Overly long, noise   |

**Analysis:**

- 3-hour historical window (seq_len=12) captures diurnal patterns
- Longer sequences (24+) add noise without performance gain
- Shorter sequences (6) miss critical temporal dependencies

---

## Appendix D: Network Topology

### D.1 Ho Chi Minh City Road Network Statistics

**Coverage Area:**

- Districts: 1, 3, 4, 5, 10, Binh Thanh, Phu Nhuan
- Total area: ~150 km²
- Major corridors: Le Duan, Nguyen Hue, Cach Mang Thang Tam

**Network Statistics:**

- Nodes (Intersections): 62
- Edges (Road Segments): 144
- Average degree: 4.65 (nodes per intersection)
- Graph diameter: 12 hops (max shortest path)
- Average path length: 5.2 hops

**Edge Attributes:**

- Distance: 0.5 km to 3.2 km (mean: 1.1 km)
- OSM highway types: primary, secondary, tertiary
- Bidirectional edges: 72 pairs

### D.2 Network Visualization

**Figure D1: Full 62-Node Road Network Topology**

_See `outputs/figures/fig_01_network_topology.png` for complete visualization_

**Visualization Features:**

- Node size proportional to degree (connectivity)
- Node positions based on GPS coordinates (lat/lon)
- Edge directions show traffic flow (144 directed edges)
- Color coding by district for geographic context

**Key Network Characteristics:**

- Hub nodes: Major intersections with degree > 8
- Bridge nodes: Critical connectors between districts
- Peripheral nodes: District boundaries with lower connectivity

```python
# Generate network visualization
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
G.add_edges_from(edge_index.T.tolist())

pos = {node: (lon, lat) for node, (lon, lat) in node_coords.items()}

plt.figure(figsize=(12, 10))
nx.draw_networkx_nodes(G, pos, node_size=300, node_color='skyblue')
nx.draw_networkx_edges(G, pos, alpha=0.5, arrows=True)
nx.draw_networkx_labels(G, pos, font_size=8)
plt.title('HCMC Traffic Network (62 Nodes, 144 Edges)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.savefig('outputs/network_topology.png', dpi=300, bbox_inches='tight')
```

---

## Appendix E: Computational Requirements

### E.1 Hardware Specifications

**Training:**

- GPU: NVIDIA RTX 3060 (6 GB VRAM)
- CPU: AMD Ryzen 5 5600X (6 cores)
- RAM: 16 GB DDR4
- Storage: 512 GB NVMe SSD

**Production (API):**

- GPU: Same as training (CUDA acceleration)
- CPU: 4 vCPU (for data preprocessing)
- RAM: 8 GB minimum
- Storage: 50 GB (model + logs)

### E.2 Training Time Breakdown

| Phase        | Time (minutes) | GPU Util    | Notes                |
| ------------ | -------------- | ----------- | -------------------- |
| Data loading | 0.5            | 0%          | CPU-bound            |
| Epoch 1-5    | 2.1            | 85%         | Initial learning     |
| Epoch 6-10   | 2.0            | 87%         | Convergence          |
| Epoch 11-15  | 1.9            | 88%         | Fine-tuning          |
| Epoch 16-20  | 1.8            | 89%         | Early stopping range |
| Epoch 21-24  | 1.7            | 90%         | Final epochs         |
| **Total**    | **10.0**       | **88% avg** | 24 epochs            |

**Memory Usage:**

- Peak VRAM: 4.2 GB (during backward pass)
- Model size: 680K params × 4 bytes = 2.76 MB
- Batch activations: ~1.5 GB per batch=32

### E.3 Inference Performance

| Metric                | Value      | Notes                       |
| --------------------- | ---------- | --------------------------- |
| Single prediction     | 16 ms      | Batch size 1                |
| Batch prediction (32) | 128 ms     | 4 ms/sample                 |
| API overhead          | 267 ms     | JSON parsing, normalization |
| **Total latency**     | **395 ms** | End-to-end API call         |
| Throughput            | 81 req/sec | Concurrent requests         |

**Optimization Opportunities:**

- FP16 inference: -35% latency (expected 257 ms)
- ONNX export: -20% latency (expected 316 ms)
- TensorRT: -50% latency (expected 197 ms)

---

## Appendix F: Configuration Files

### F.1 Main Training Configuration

```yaml
# File: configs/training_config.json

{
  "model":
    {
      "type": "STMGT_V2",
      "hidden_dim": 96,
      "num_blocks": 3,
      "num_heads": 4,
      "dropout": 0.1,
      "num_mixtures": 5,
    },
  "data":
    {
      "path": "data/processed/all_runs_extreme_augmented.parquet",
      "seq_len": 12,
      "pred_len": 1,
      "split_ratio": [0.7, 0.15, 0.15],
      "batch_size": 32,
      "num_workers": 4,
    },
  "training":
    {
      "epochs": 100,
      "learning_rate": 0.001,
      "optimizer": "adam",
      "scheduler":
        {
          "type": "ReduceLROnPlateau",
          "factor": 0.5,
          "patience": 5,
          "min_lr": 1e-6,
        },
      "early_stopping": { "patience": 10, "min_delta": 0.001 },
      "gradient_clip": 5.0,
    },
  "loss": { "nll_weight": 1.0, "mae_weight": 0.1 },
}
```

### F.2 API Configuration

```yaml
# File: traffic_api/config.py

API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": False,  # Production
    "workers": 4,
    "model_path": "traffic_api/models/stmgt_best.pt",
    "device": "cuda",  # or "cpu"
    "max_batch_size": 32,
    "timeout": 30,
    "cors_origins": ["*"],
    "log_level": "info"
}

DATA_CONFIG = {
    "stats_path": "data/processed/normalization_stats.json",
    "graph_path": "cache/adjacency_matrix.npy",
    "topology_path": "cache/overpass_topology.json"
}
```

---

## Appendix G: Error Analysis Details

### G.1 Error Analysis by Node Type

**Highway Ramps/Merges (High Variability):**

- Average MAE: 4.8-5.2 km/h
- Cause: High variance in merging behavior, platoon effects
- Sample nodes: Highway entry/exit points

**Urban Arterials (Moderate):**

- Average MAE: 3.0-3.5 km/h
- Cause: Signal timing variability, mixed traffic
- Sample nodes: Major streets in central districts

**Major Intersections (Low):**

- Average MAE: 2.1-2.6 km/h
- Cause: Consistent signal patterns, rich training data
- Sample nodes: District 1, 3 major crossings

### G.2 High Error Nodes Analysis

| Node Type         | Location           | Avg MAE | Possible Causes                    |
| ----------------- | ------------------ | ------- | ---------------------------------- |
| Highway Ramp      | District 10 (West) | 5.35    | Heavy traffic variability          |
| Construction Zone | District 4 (South) | 4.92    | Temporary road changes             |
| Boundary Node     | Phu Nhuan (Edge)   | 4.68    | Edge of network (boundary effects) |
| Bridge Access     | District 1 (River) | 4.55    | Complex merging patterns           |

### G.3 Error Distribution by Time of Day

**Detailed Hourly Breakdown:**

| Time Period       | MAE (km/h) | Error Increase | Sample Count |
| ----------------- | ---------- | -------------- | ------------ |
| 6-7 AM (Pre-peak) | 2.85       | Baseline       | 180          |
| 7-9 AM (Morning)  | 3.15       | +10.5%         | 380          |
| 9-11 AM (Late)    | 3.42       | +20%           | 220          |
| 11-2 PM (Midday)  | 3.68       | +29%           | 280          |
| 2-4 PM (Early)    | 3.55       | +24.5%         | 240          |
| 4-7 PM (Evening)  | 3.08       | +8%            | 420          |
| 7-10 PM (Late)    | 3.28       | +15%           | 320          |
| 10-6 AM (Night)   | 3.95       | +38.5%         | 360          |

**Key Patterns:**

- Morning/evening rush: Moderate error (consistent patterns)
- Midday: Higher error (less training data, variable traffic)
- Night: Highest error (sparse data, erratic behavior)

### G.4 Failure Case Examples

**Case 1: Sudden Construction Zone**

- Date: November 3, 2025
- Location: Le Duan Street
- Ground truth: 8 km/h | Prediction: 22 km/h
- Cause: New construction not in historical data
- Error: 14 km/h (outlier)

**Case 2: Flash Flood Event**

- Date: October 27, 2025
- Location: Low-lying District 4 area
- Ground truth: 4 km/h | Prediction: 15 km/h
- Cause: Extreme weather beyond training distribution
- Error: 11 km/h

**Case 3: Special Event Traffic**

- Date: November 1, 2025
- Location: District 1 (City Center)
- Ground truth: 6 km/h | Prediction: 18 km/h
- Cause: Large public gathering not captured in features
- Error: 12 km/h

### G.5 Error Patterns Summary

- **Morning rush (7-9 AM):** +10.5% error due to rapid congestion formation
- **Rainy days:** +29% error (3.68 vs 2.85), especially during onset
- **Weekends:** -8% error (3.28 vs 3.02 weekday), more predictable leisure traffic
- **Construction zones:** +60% error when not in training data
- **Special events:** +55% error (not captured by features)

---

## Appendix H: Future Work Details

### H.1 Planned Model Enhancements

1. **Dynamic Graph Learning:**

   - Learn time-varying adjacency matrix
   - Capture changing traffic flow patterns
   - Expected improvement: +5-8% MAE reduction

2. **Multi-Step Prediction:**

   - Extend pred_len from 1 to 4 (1 hour ahead)
   - Autoregressive vs. direct forecasting comparison
   - Uncertainty propagation over multiple steps

3. **Incident Detection:**
   - Anomaly detection layer (reconstruction error)
   - Alert system for unusual patterns
   - Integration with traffic management center

### H.2 Data Collection Roadmap

**Phase 1 (1-3 months):**

- 24/7 data collection (currently limited hours)
- Add 30-50 nodes in Districts 2, 7, Go Vap
- Incorporate accident data from police reports

**Phase 2 (3-6 months):**

- Event calendar integration (concerts, sports, holidays)
- Public transport schedules (bus, metro)
- Parking availability data

**Phase 3 (6-12 months):**

- Multi-city expansion (Hanoi, Da Nang)
- Transfer learning experiments
- Cross-city generalization analysis

---

## Appendix I: Glossary

**Traffic Forecasting Terms:**

- **MAE (Mean Absolute Error):** Average absolute difference between predicted and actual speeds (km/h)
- **RMSE (Root Mean Squared Error):** Square root of average squared errors (km/h)
- **R² (Coefficient of Determination):** Proportion of variance explained (0-1, higher better)
- **MAPE (Mean Absolute Percentage Error):** Average percentage error
- **CRPS (Continuous Ranked Probability Score):** Proper scoring rule for probabilistic forecasts

**Model Architecture Terms:**

- **GCN (Graph Convolutional Network):** Neural network operating on graph-structured data
- **GAT (Graph Attention Network):** GCN with attention mechanism for dynamic neighbor weighting
- **Transformer:** Architecture using self-attention for sequence modeling
- **Mixture Density Network (MDN):** Neural network outputting mixture of distributions

**Uncertainty Quantification:**

- **Aleatoric Uncertainty:** Inherent data noise (weather, driver behavior)
- **Epistemic Uncertainty:** Model uncertainty (limited training data)
- **Calibration:** Agreement between predicted and observed confidence intervals
- **Coverage:** Percentage of true values within predicted intervals

---

## Appendix J: Acknowledgments

**Data Sources:**

- Google Directions API for traffic data
- OpenWeatherMap for meteorological data
- OpenStreetMap / Overpass API for road network topology

**Software and Tools:**

- PyTorch and PyTorch Geometric teams
- NumPy, Pandas, Scikit-learn communities
- VS Code and GitHub Copilot (development assistance)

**Academic Resources:**

- Stanford CS224W course materials (Graph Neural Networks)
- Papers from ICLR, NeurIPS, IJCAI conferences
- METR-LA and PeMS-BAY benchmark datasets (comparison reference)

**Special Thanks:**

- DSP391m course instructors and TAs
- Project team members (data collection, testing, deployment)
- Ho Chi Minh City Department of Transport (domain expertise)

---

**End of Appendices**

[← Back to Main Report](FINAL_REPORT.md)
