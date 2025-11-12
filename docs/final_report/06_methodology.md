# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Section 8: Methodology

## 8.1 Model Selection Rationale

### 8.1.1 Why Graph Neural Networks?

**Traffic as Graph-Structured Data:**

- **Nodes:** Intersections (62 nodes)
- **Edges:** Road segments (144 edges)
- **Advantage:** Captures spatial dependencies through message passing

### 8.1.2 Why Spatio-Temporal Architecture?

**Dual Nature of Traffic:**

- **Spatial:** Speed on adjacent roads influences each other
- **Temporal:** Traffic patterns evolve over time
- **Solution:** Parallel processing of spatial and temporal dimensions

### 8.1.3 Why STMGT Over Baselines?

| Model        | Spatial        | Temporal             | Weather             | Uncertainty  |
| ------------ | -------------- | -------------------- | ------------------- | ------------ |
| LSTM         | ❌             | ✅ (RNN)             | ✅ (concat)         | ❌           |
| GCN          | ✅ (GCN)       | ❌                   | ✅ (concat)         | ❌           |
| GraphWaveNet | ✅ (adaptive)  | ✅ (TCN)             | ❌                  | ❌           |
| ASTGCN       | ✅ (attention) | ✅ (attention)       | ❌                  | ❌           |
| **STMGT**    | ✅ (GATv2) [7] | ✅ (Transformer) [8] | ✅ (cross-attn) [8] | ✅ (GMM) [2] |

**Key Advantages:**

1. **Parallel ST Processing:** 5-12% improvement over sequential [11, 13]
2. **Weather Cross-Attention:** Context-dependent weather effects [8]
3. **Gaussian Mixture:** Probabilistic predictions with calibrated uncertainty [2, 3]
4. **Adaptive Graph:** GATv2 [7] learns neighbor importance dynamically

---

## 8.2 Data Splitting Strategy

### 8.2.1 Temporal Split (No Shuffling)

**Rationale:**

- Traffic data has temporal autocorrelation
- Shuffled split causes **data leakage** (training on future to predict past)
- Temporal split simulates real deployment scenario

**Split Configuration:**

```python
# Chronological split
total_samples = 16,328
train_samples = int(0.70 * total_samples)  # 11,430
val_samples = int(0.15 * total_samples)    # 2,449
test_samples = total_samples - train_samples - val_samples  # 2,449

# No overlap between splits
train_data = data[:train_samples]
val_data = data[train_samples:train_samples+val_samples]
test_data = data[train_samples+val_samples:]
```

### 8.2.2 Validation Strategy

**Early Stopping:**

- Monitor validation MAE every epoch
- Patience: 10 epochs (stop if no improvement)
- Restore best model weights

**Cross-Validation:**

- Not applicable due to temporal dependencies
- Use fixed temporal split to prevent leakage

---

## 8.3 Feature Engineering and Selection

### 8.3.1 Graph Features

**Node Features (per node, per timestep):**

1. **Speed:** Historical speed (normalized)
2. **Weather:** Temperature, wind, precipitation (normalized)
3. **Total:** 4 features per node

**Edge Features:**

- Initially: None (only graph structure)
- Future work: Road type, lanes, speed limit

**Graph Structure:**

- **Adjacency Matrix:** 62×62 binary matrix
- **Edge Index:** COO format for PyTorch Geometric
  ```python
  edge_index = torch.tensor([[src_nodes], [dst_nodes]], dtype=torch.long)
  # Shape: [2, 144] for 144 directed edges
  ```

### 8.3.2 Temporal Features

**Cyclical Encoding (Hour-of-Day):**

```python
hour_sin = np.sin(2 * np.pi * hour / 24)
hour_cos = np.cos(2 * np.pi * hour / 24)
```

**Day-of-Week Embedding:**

- One-hot encoding: 7-dimensional vector
- Or learnable embedding: 7 → 16 dimensions

**Rush Hour Binary:**

```python
is_rush = ((hour >= 7) & (hour <= 9)) | ((hour >= 17) & (hour <= 19))
```

### 8.3.3 Weather Features

**Normalization:**

```python
# Z-score normalization
temp_normalized = (temp - temp_mean) / temp_std
wind_normalized = (wind - wind_mean) / wind_std
precip_normalized = (precip - precip_mean) / precip_std
```

**Weather Encoding for Cross-Attention:**

```python
weather_features = [temp_norm, wind_norm, precip_norm]  # 3D vector
weather_encoded = MLP(weather_features)  # Project to hidden_dim=96
```

### 8.3.4 Feature Selection Rationale

| Feature                | Included? | Rationale                              |
| ---------------------- | --------- | -------------------------------------- |
| **Speed (historical)** | ✅        | Primary signal, strong autocorrelation |
| **Temperature**        | ✅        | Extreme heat reduces speed (-5-8%)     |
| **Wind speed**         | ✅        | Correlates with rain events            |
| **Precipitation**      | ✅        | Strong impact (-15-30% speed)          |
| **Hour-of-day**        | ✅        | Rush hour patterns critical            |
| **Day-of-week**        | ✅        | Weekday vs weekend differences         |
| **Road type**          | ❌        | Not available in current dataset       |
| **Accidents**          | ❌        | No reliable real-time data source      |
| **Events**             | ❌        | Future work (require event calendar)   |

---

## 8.4 Sequence Representation

### 8.4.1 Input Sequence

**Shape:** `[batch, seq_len=12, num_nodes=62, features=4]`

```
Historical window: 12 timesteps × 15 min = 3 hours
Feature vector per node: [speed, temp, wind, precip]
Graph structure: 62 nodes, 144 edges
```

**Example:**

```
Timestamp    Node           Speed  Temp  Wind  Precip
07:00       node-10.737-...  16.0   28.5   5.2    0.0
07:15       node-10.737-...  14.5   28.6   5.1    0.0
...         ...              ...    ...    ...    ...
09:00       node-10.737-...  12.3   29.1   4.8    2.5
```

### 8.4.2 Output Sequence

**Shape:** `[batch, pred_len=12, num_nodes=62, mixture_params]`

```
Forecast window: 12 timesteps × 15 min = 3 hours
Mixture parameters: 5 components × (μ, σ, π) = 15 parameters
Per node prediction: [μ1, σ1, π1, μ2, σ2, π2, ..., μ5, σ5, π5]
```

**Gaussian Mixture Output:**

```python
# For each node at each timestep:
means = [μ1, μ2, μ3, μ4, μ5]  # 5 mixture means
stds = [σ1, σ2, σ3, σ4, σ5]   # 5 mixture std devs
weights = [π1, π2, π3, π4, π5]  # 5 mixture weights (sum to 1)

# Final prediction (point estimate):
predicted_speed = Σ(πi * μi)

# Uncertainty (confidence interval):
lower_bound = percentile(mixture_distribution, 10%)
upper_bound = percentile(mixture_distribution, 90%)
```

---

## 8.5 Model Architecture Overview

**High-Level Architecture:**

```
Input: [B, 12, 62, 4]
  ↓
Parallel Processing:
  ├─ Spatial Branch (GATv2)
  │   ├─ Multi-head attention (4 heads)
  │   └─ Message passing on graph
  │
  └─ Temporal Branch (Transformer)
      ├─ Self-attention over time
      └─ Positional encoding
  ↓
Gated Fusion:
  α = sigmoid(MLP([spatial || temporal]))
  fused = α * spatial + (1-α) * temporal
  ↓
Weather Cross-Attention:
  Query: fused features
  Key/Value: weather_encoded
  ↓
Mixture Output Head:
  → [μ1, σ1, π1, ..., μ5, σ5, π5]
  ↓
Output: [B, 12, 62, 15]  # 5 mixtures × 3 params
```

**Key Components:**

1. **GATv2:** Graph attention for spatial dependencies
2. **Transformer:** Self-attention for temporal patterns
3. **Gated Fusion:** Learnable combination of spatial and temporal
4. **Weather Cross-Attention:** Context-dependent weather effects
5. **Gaussian Mixture Head:** Probabilistic output

---

**Next:** [Model Development →](07_model_development.md)
