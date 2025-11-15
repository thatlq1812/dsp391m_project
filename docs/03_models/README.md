# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Models Documentation

Model architectures, training workflows, and implementation details.

---

## 🤖 Overview

This section contains documentation for all model architectures implemented in the STMGT Traffic Forecasting System.

**Implemented Models:**

- **STMGT** - Spatial-Temporal Multi-Graph Transformer (primary)
- **GraphWaveNet** - Graph convolution with dilated causal convolution
- **ASTGCN** - Attention-based Spatial-Temporal GCN
- **LSTM** - Temporal-only baseline

---

## 📖 Available Documentation

### [Model Overview](MODEL.md)

High-level comparison of all model architectures.

**Contents:**

- Architecture summaries
- Parameter counts
- Computational complexity
- Use case recommendations

**Quick Comparison:**

| Model        | Params | MAE (km/h) | Strengths                               |
| ------------ | ------ | ---------- | --------------------------------------- |
| STMGT        | 680K   | 3-4        | Spatial-temporal, transformer attention |
| GraphWaveNet | 485K   | 4-6        | Efficient, graph diffusion              |
| ASTGCN       | 420K   | 5-7        | Attention mechanisms                    |
| LSTM         | 180K   | 5-7        | Simple, temporal-only baseline          |

### [Training Workflow](TRAINING_WORKFLOW.md)

End-to-end training process and best practices.

**Contents:**

- Data preparation
- Hyperparameter tuning
- Training strategies
- Monitoring and debugging

**Key Concepts:**

- Learning rate scheduling
- Early stopping
- Gradient clipping
- Loss visualization

### [Architecture Details](architecture/)

In-depth technical documentation for each model.

**Available Guides:**

- **[STMGT Architecture](architecture/STMGT_ARCHITECTURE.md)** - Complete technical spec
- **[STMGT Data I/O](architecture/STMGT_DATA_IO.md)** - Data flow and processing
- **[Architecture ELI5](architecture/ARCHITECTURE_ELI5.md)** - Simplified explanations

---

## 🏗️ Model Architectures

### STMGT (Spatial-Temporal Multi-Graph Transformer)

**Key Features:**

- Multi-head transformer attention
- Spatial graph convolution
- Temporal pattern capture
- 680K parameters

**Components:**

1. **Spatial Encoder:** Graph convolution on road network topology
2. **Temporal Encoder:** Transformer for time series patterns
3. **Fusion Layer:** Combines spatial-temporal features
4. **Output Layer:** Multi-step speed prediction

**Best For:**

- Production deployments
- Complex spatial-temporal patterns
- Long-term forecasting (12+ steps)

### GraphWaveNet

**Key Features:**

- Adaptive graph learning
- Dilated causal convolution
- 485K parameters

**Components:**

1. **Adaptive Adjacency:** Learns edge relationships
2. **GCN Layers:** Spatial feature extraction
3. **TCN Layers:** Temporal convolution
4. **Skip Connections:** Multi-scale features

**Best For:**

- Real-time inference (fast)
- Short-term forecasting (1-6 steps)
- Limited computational resources

### ASTGCN

**Key Features:**

- Spatial-temporal attention
- Multi-component fusion
- 420K parameters

**Components:**

1. **Spatial Attention:** Weighted adjacency
2. **Temporal Attention:** Time dependency
3. **ST Blocks:** Combined processing
4. **Component Fusion:** Recent/daily/weekly

**Best For:**

- Research experiments
- Attention visualization
- Multi-scale temporal patterns

### LSTM Baseline

**Key Features:**

- Simple temporal model
- No spatial information
- 180K parameters

**Components:**

1. **LSTM Layers:** 2-layer stacked
2. **Dense Layers:** Output projection

**Best For:**

- Baseline comparison
- Single-edge forecasting
- Quick prototyping

---

## 🚀 Quick Start

### Train STMGT

```bash
python scripts/training/train_stmgt.py \
  --config configs/train_normalized_v3.json \
  --output-dir outputs/stmgt_production
```

### Train GraphWaveNet

```bash
python scripts/training/train_graphwavenet_baseline.py \
  --dataset data/processed/super_dataset_1year.parquet \
  --output-dir outputs/graphwavenet_production \
  --epochs 50 \
  --batch-size 32
```

### Train LSTM

```bash
python scripts/training/train_lstm_baseline.py \
  --dataset data/processed/super_dataset_1year.parquet \
  --output-dir outputs/lstm_production \
  --epochs 50 \
  --batch-size 64
```

### Evaluate Model

```bash
python scripts/evaluation/evaluate_model.py \
  --model outputs/stmgt_production/model.pt \
  --data data/processed/super_dataset_1year.parquet
```

---

## 📊 Performance Comparison

### Super Dataset 1-Year Results

| Model        | MAE  | RMSE | MAPE  | Training Time |
| ------------ | ---- | ---- | ----- | ------------- |
| STMGT        | 3.24 | 5.12 | 11.2% | ~4-5 hours    |
| GraphWaveNet | 4.87 | 7.34 | 15.8% | ~3-4 hours    |
| ASTGCN       | 5.43 | 8.21 | 17.3% | ~3-4 hours    |
| LSTM         | 6.12 | 9.05 | 19.7% | ~2-3 hours    |

_Note: Results are expected based on prototype testing. Full 1-year training in progress._

### Original 1-Week Dataset

| Model                | MAE  | RMSE | MAPE | Notes                   |
| -------------------- | ---- | ---- | ---- | ----------------------- |
| GraphWaveNet (buggy) | 0.25 | 0.42 | 0.8% | Autocorrelation exploit |
| GraphWaveNet (fixed) | 1.46 | 2.31 | 4.5% | Prototype test          |

---

## ⚙️ Configuration

### Model Hyperparameters

**STMGT (configs/train_normalized_v3.json):**

```json
{
  "model_name": "stmgt_v3",
  "num_nodes": 144,
  "hidden_dim": 64,
  "num_heads": 4,
  "num_layers": 3,
  "dropout": 0.1,
  "learning_rate": 0.0008,
  "batch_size": 32,
  "max_epochs": 50
}
```

**GraphWaveNet:**

```python
# Key parameters
hidden_channels = 32
num_blocks = 4
num_layers = 2
kernel_size = 2
```

**LSTM:**

```python
# Key parameters
hidden_size = 128
num_layers = 2
dropout = 0.2
```

---

## 🔍 Model Selection Guide

### Choose STMGT if:

- Need best accuracy
- Have sufficient computational resources
- Production deployment
- Long-term forecasting (12+ steps)

### Choose GraphWaveNet if:

- Need fast inference
- Limited memory
- Short-term forecasting (1-6 steps)
- Real-time applications

### Choose ASTGCN if:

- Research experiments
- Need attention visualization
- Multi-scale temporal patterns

### Choose LSTM if:

- Baseline comparison needed
- Single-edge forecasting
- Quick prototyping

---

## 📚 Related Documentation

- **[Data Overview](../02_data/DATA.md)** - Dataset structure
- **[Super Dataset](../02_data/super_dataset/SUPER_DATASET_DESIGN.md)** - Training data
- **[Evaluation](../04_evaluation/METRICS_VERIFICATION_ALL_MODELS.md)** - Performance metrics
- **[API Guide](../01_getting_started/API.md)** - Deployment and inference

---

**Last Updated:** November 15, 2025
