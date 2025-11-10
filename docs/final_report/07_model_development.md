# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Section 9: Model Development

## 9.1 STMGT Architecture

### 9.1.1 Model Overview

**STMGT (Spatio-Temporal Multi-Modal Graph Transformer)**

**Final Architecture Specifications:**

- **Parameters:** 680,000 (680K)
- **Hidden Dimension:** 96
- **Number of Blocks:** 3
- **Attention Heads:** 4
- **Mixture Components:** 5 (Gaussian)
- **Sequence Length:** 12 (input)
- **Prediction Length:** 12 (output)
- **Dropout:** 0.2
- **DropEdge:** 0.05

### 9.2 Detailed Architecture Components

#### 9.2.1 Input Embedding Layer

```python
class InputEmbedding(nn.Module):
    def __init__(self, in_dim=1, hidden_dim=96):
        super().__init__()
        self.linear = nn.Linear(in_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [batch, seq_len, num_nodes, in_dim]
        return self.norm(self.linear(x))
```

#### 9.2.2 Spatial Branch (GATv2)

**Graph Attention Network v2:**

```python
class SpatialBranch(nn.Module):
    def __init__(self, hidden_dim=96, heads=4, dropout=0.2):
        super().__init__()
        self.gat = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // heads,
            heads=heads,
            dropout=dropout,
            concat=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        # x: [batch * seq_len * num_nodes, hidden_dim]
        # edge_index: [2, num_edges]

        h = self.gat(x, edge_index)  # Multi-head attention
        h = self.dropout(h)
        h = self.norm(h + x)  # Residual connection
        return h
```

**GATv2 Attention Mechanism:**

```
α_ij = softmax_j(a^T · LeakyReLU(W [h_i || h_j]))

h'_i = σ(Σ_{j∈N(i)} α_ij · W · h_j)
```

**Key Features:**

- **Dynamic Attention:** Learns neighbor importance per sample
- **Multi-Head:** 4 parallel attention patterns
- **Residual Connection:** Prevents gradient vanishing

#### 9.2.3 Temporal Branch (Transformer)

**Self-Attention over Time:**

```python
class TemporalBranch(nn.Module):
    def __init__(self, hidden_dim=96, heads=4, dropout=0.2):
        super().__init__()
        self.pos_encoding = PositionalEncoding(hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: [batch, seq_len, num_nodes, hidden_dim]
        B, T, N, D = x.shape

        # Reshape to process all nodes' time series
        x = x.transpose(1, 2).reshape(B * N, T, D)  # [B*N, T, D]

        # Add positional encoding
        x = self.pos_encoding(x)

        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        # Reshape back
        x = x.reshape(B, N, T, D).transpose(1, 2)  # [B, T, N, D]
        return x
```

**Self-Attention Formula:**

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Q = W_Q · x
K = W_K · x
V = W_V · x
```

#### 9.2.4 Gated Fusion

**Learnable Combination of Spatial and Temporal:**

```python
class GatedFusion(nn.Module):
    def __init__(self, hidden_dim=96):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, spatial, temporal):
        # spatial, temporal: [B, T, N, D]

        # Concatenate and compute gate
        concat = torch.cat([spatial, temporal], dim=-1)  # [B, T, N, 2D]
        alpha = self.gate(concat)  # [B, T, N, D]
        beta = 1 - alpha

        # Fused = α * spatial + (1-α) * temporal
        fused = alpha * spatial + beta * temporal
        return fused, alpha  # Return alpha for analysis
```

**Rationale:**

- Learned weights adapt to data (some timesteps more spatial, others more temporal)
- Residual-like structure (α + β = 1)
- Interpretable (can visualize α to see spatial vs temporal importance)

#### 9.2.5 Weather Cross-Attention

**Context-Dependent Weather Integration:**

```python
class WeatherCrossAttention(nn.Module):
    def __init__(self, hidden_dim=96, weather_dim=3, dropout=0.2):
        super().__init__()
        self.weather_proj = nn.Linear(weather_dim, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, traffic_features, weather):
        # traffic_features: [B, T, N, D]
        # weather: [B, T, 3]  # [temp, wind, precip]

        B, T, N, D = traffic_features.shape

        # Project weather to hidden_dim
        weather_encoded = self.weather_proj(weather)  # [B, T, D]
        weather_encoded = weather_encoded.unsqueeze(2).expand(B, T, N, D)

        # Reshape for attention
        query = traffic_features.reshape(B * T, N, D)
        key_value = weather_encoded.reshape(B * T, N, D)

        # Cross-attention: Query traffic with weather context
        attn_out, _ = self.cross_attn(query, key_value, key_value)
        attn_out = attn_out.reshape(B, T, N, D)

        # Residual connection
        output = self.norm(traffic_features + attn_out)
        return output
```

**Why Cross-Attention over Concatenation:**

- Weather effects are **state-dependent** (rain impacts congested roads more than free-flow)
- Cross-attention learns **when** and **where** weather matters
- Literature shows **8-12% improvement** over simple concatenation

#### 9.2.6 Gaussian Mixture Output Head

**Probabilistic Prediction:**

```python
class GaussianMixtureHead(nn.Module):
    def __init__(self, hidden_dim=96, num_mixtures=5):
        super().__init__()
        self.num_mixtures = num_mixtures

        # Output: means, log_stds, logits for each mixture
        self.output_layer = nn.Linear(
            hidden_dim,
            num_mixtures * 3  # (μ, log_σ, logit) × K
        )

    def forward(self, x):
        # x: [B, pred_len, N, D]
        B, T, N, D = x.shape

        out = self.output_layer(x)  # [B, T, N, K*3]
        out = out.reshape(B, T, N, self.num_mixtures, 3)

        means = out[..., 0]  # [B, T, N, K]
        log_stds = out[..., 1]  # [B, T, N, K]
        logits = out[..., 2]  # [B, T, N, K]

        # Ensure positive std devs with floor
        stds = torch.exp(log_stds) + 0.01  # σ >= 0.01

        # Normalize mixture weights
        weights = F.softmax(logits, dim=-1)  # [B, T, N, K], sum to 1

        return means, stds, weights
```

**Mixture Distribution:**

```
p(y|x) = Σ_{k=1}^K π_k · N(μ_k, σ_k²)

where:
  π_k: Mixture weight (Σπ_k = 1)
  μ_k: Mean of k-th component
  σ_k: Std dev of k-th component
```

**Point Prediction:**

```python
predicted_speed = torch.sum(weights * means, dim=-1)  # Weighted average
```

**Uncertainty Quantification:**

```python
# Sample from mixture
samples = sample_gaussian_mixture(means, stds, weights, num_samples=1000)
lower_bound = torch.quantile(samples, 0.1, dim=-1)  # 10th percentile
upper_bound = torch.quantile(samples, 0.9, dim=-1)  # 90th percentile
```

---

## 9.3 Complete Forward Pass

```python
class STMGT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = InputEmbedding(in_dim=1, hidden_dim=96)

        # 3 ST-blocks
        self.blocks = nn.ModuleList([
            STBlock(hidden_dim=96, heads=4, dropout=0.2)
            for _ in range(3)
        ])

        self.weather_attn = WeatherCrossAttention(hidden_dim=96)
        self.mixture_head = GaussianMixtureHead(hidden_dim=96, num_mixtures=5)

    def forward(self, x, edge_index, weather):
        # x: [B, seq_len=12, N=62, 1]
        # edge_index: [2, 144]
        # weather: [B, seq_len=12, 3]

        # Embed input
        h = self.embedding(x)  # [B, 12, 62, 96]

        # Pass through ST-blocks
        for block in self.blocks:
            h = block(h, edge_index)  # Parallel spatial + temporal

        # Weather cross-attention
        h = self.weather_attn(h, weather)  # [B, 12, 62, 96]

        # Mixture output
        means, stds, weights = self.mixture_head(h)

        return means, stds, weights  # Each [B, 12, 62, 5]
```

---

## 9.4 Training Procedure

### 9.4.1 Loss Function

**Negative Log-Likelihood (NLL) for Gaussian Mixture:**

```python
def mixture_nll_loss(y_true, means, stds, weights):
    """
    y_true: [B, T, N] ground truth speeds
    means, stds, weights: [B, T, N, K] mixture parameters
    """
    # Compute log probability for each component
    log_probs = []
    for k in range(K):
        log_prob_k = -0.5 * ((y_true - means[..., k]) / stds[..., k])**2
        log_prob_k -= torch.log(stds[..., k])
        log_prob_k -= 0.5 * np.log(2 * np.pi)
        log_probs.append(log_prob_k + torch.log(weights[..., k]))

    # Log-sum-exp trick for numerical stability
    log_probs = torch.stack(log_probs, dim=-1)
    nll = -torch.logsumexp(log_probs, dim=-1)

    return nll.mean()
```

**Regularization Terms:**

```python
total_loss = nll_loss + λ1 * variance_regularization + λ2 * entropy_regularization

# Variance regularization (prevent collapse to single component)
var_reg = -torch.log(stds).mean()

# Entropy regularization (encourage diverse weights)
entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
entropy_reg = -entropy

# Final loss
loss = nll_loss + 0.01 * var_reg + 0.001 * entropy_reg
```

### 9.4.2 Optimizer Configuration

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,  # L2 regularization
    betas=(0.9, 0.999)
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,  # Restart every 10 epochs
    T_mult=2,
    eta_min=1e-6
)
```

### 9.4.3 Training Hyperparameters

| Hyperparameter              | Value | Rationale                         |
| --------------------------- | ----- | --------------------------------- |
| **Batch Size**              | 32    | Fit in 6GB GPU memory             |
| **Learning Rate**           | 1e-3  | Standard for AdamW                |
| **Weight Decay**            | 1e-4  | Prevent overfitting (16K samples) |
| **Dropout**                 | 0.2   | Aggressive regularization         |
| **DropEdge**                | 0.05  | Random edge dropping              |
| **Max Epochs**              | 100   | Early stopping at ~24 epochs      |
| **Early Stopping Patience** | 10    | Stop if val MAE no improvement    |
| **Gradient Clipping**       | 1.0   | Prevent exploding gradients       |

### 9.4.4 Training Loop Pseudocode

```python
for epoch in range(max_epochs):
    model.train()
    for batch in train_loader:
        x, y, edge_index, weather = batch

        # Forward pass
        means, stds, weights = model(x, edge_index, weather)

        # Compute loss
        loss = mixture_nll_loss(y, means, stds, weights)
        loss += 0.01 * variance_reg(stds)
        loss += 0.001 * entropy_reg(weights)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    # Validation
    model.eval()
    val_mae = evaluate(model, val_loader)

    # Early stopping
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        torch.save(model.state_dict(), 'best_model.pt')
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= 10:
            print("Early stopping triggered")
            break

    # Learning rate scheduling
    scheduler.step()
```

---

## 9.5 Implementation Details

### 9.5.1 Hardware and Software

**Hardware:**

- **GPU:** NVIDIA RTX 3060 Laptop (6GB VRAM)
- **CPU:** Intel Core i7
- **RAM:** 16GB

**Software:**

- **Framework:** PyTorch 2.0.1
- **Graph Library:** PyTorch Geometric 2.3.1
- **Python:** 3.10.18
- **CUDA:** 11.7

### 9.5.2 Training Time

**Per Epoch:**

- Forward pass: ~15 seconds
- Backward pass: ~10 seconds
- Total: ~25 seconds/epoch

**Total Training:**

- 24 epochs × 25 sec = ~10 minutes
- Early stopped at epoch 9 (best validation MAE)

### 9.5.3 Model Size

```
Total parameters: 680,000
Model file size: 2.76 MB (best_model.pt)
Config file size: 3.1 KB (config.json)
```

---

**Next:** [Model Evaluation & Fine-Tuning →](08_evaluation_tuning.md)
