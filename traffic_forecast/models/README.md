# Traffic Forecast Models

This directory contains all machine learning and deep learning models for traffic speed prediction.

## Directory Structure

```
models/
 __init__.py # Model registry and imports
 baseline.py # Baseline persistence model
 ensemble.py # Ensemble model wrapper
 lstm_traffic.py # LSTM neural network (NEW)
 graph/
 __init__.py # Graph neural networks module
 astgcn_traffic.py # ASTGCN implementation (NEW)
```

## Models Overview

### Traditional ML Models

Located in: `traffic_forecast/ml/trainer.py`

- **Random Forest** - Ensemble of decision trees
- **XGBoost** - Gradient boosting (BEST: R²=0.8908)
- **LightGBM** - Fast gradient boosting
- **Gradient Boosting** - sklearn implementation

### Deep Learning Models

#### LSTM Traffic Predictor

**File:** `lstm_traffic.py` 
**Class:** `LSTMTrafficPredictor`

Features:

- Sequence-based recurrent neural network
- Attention mechanism
- Multi-layer LSTM [128, 64]
- Dropout + BatchNormalization
- Multi-horizon forecasting

Usage:

```python
from traffic_forecast.models.lstm_traffic import LSTMTrafficPredictor

model = LSTMTrafficPredictor(
 sequence_length=12,
 lstm_units=[128, 64],
 dropout_rate=0.2
)

history = model.fit(X_train, y_train, X_val, y_val, epochs=50)
predictions = model.predict(X_test)
metrics = model.evaluate(X_test, y_test)
```

#### ASTGCN Traffic Model

**File:** `graph/astgcn_traffic.py` 
**Class:** `ASTGCNTrafficModel`

Features:

- Attention-based Spatial-Temporal Graph Convolutional Network
- Chebyshev graph convolutions
- Temporal and spatial attention
- Multi-component modeling (recent/daily/weekly)
- Learnable fusion

Usage:

```python
from traffic_forecast.models.graph import ASTGCNTrafficModel, ASTGCNConfig
import numpy as np

# Define adjacency matrix (num_nodes x num_nodes)
adjacency = np.eye(10) # Example: 10 nodes

# Configure model
config = ASTGCNConfig(
 num_nodes=10,
 input_dim=1,
 horizon=12,
 attention_units=32
)

# Create and train
model = ASTGCNTrafficModel(adjacency, config)
model.build_model()
history = model.fit(X_train, y_train, X_val, y_val, epochs=100)
predictions = model.predict(X_test)
```

### Baseline Models

**File:** `baseline.py`

Simple persistence forecast (last known speed).

## Model Performance

Based on testing with 120 samples, 31 features:

| Model | R² Score | RMSE | MAE | MAPE |
| ----------------- | -------- | ---- | ---- | ----- |
| XGBoost | 0.8908 | 2.42 | 1.93 | 6.73% |
| LightGBM | 0.8769 | 2.57 | 2.05 | 7.12% |
| Random Forest | 0.8682 | 2.66 | 2.13 | 7.45% |
| Gradient Boosting | 0.8590 | 2.75 | 2.20 | 7.68% |

LSTM and ASTGCN require more data for training.

## Saving and Loading

### Traditional ML

```python
from traffic_forecast.ml.trainer import ModelTrainer

trainer = ModelTrainer(model_type='xgboost')
trainer.train(X_train, y_train)
trainer.save('models/production/xgboost_model.pkl')

# Load
trainer = ModelTrainer.load('models/production/xgboost_model.pkl')
```

### LSTM

```python
model.save(Path('models/production/lstm'))
loaded_model = LSTMTrafficPredictor.load(Path('models/production/lstm'))
```

### ASTGCN

```python
model.save(Path('models/production/astgcn'))
loaded_model = ASTGCNTrafficModel.load(Path('models/production/astgcn'))
```

## Dependencies

**Traditional ML:**

- scikit-learn
- xgboost
- lightgbm
- numpy
- pandas

**Deep Learning:**

- tensorflow (>= 2.10)
- keras
- numpy

## Development

### Adding New Models

1. Create model file in appropriate directory
2. Implement standard interface (fit, predict, evaluate, save, load)
3. Register in `__init__.py`
4. Add tests in `tests/`
5. Update this README

### Code Standards

- Use type hints
- Write comprehensive docstrings
- Follow PEP 8 style guide
- 4-space indentation
- No emoji characters
- American English spelling

## See Also

- [ML Training Notebook](../../notebooks/ML_TRAINING.ipynb)
- [Control Panel](../../notebooks/CONTROL_PANEL.ipynb)
- [Model Results Report](../../doc/reports/MODEL_RESULTS.md)
- [Feature Engineering Guide](../../doc/reference/FEATURE_ENGINEERING_GUIDE.md)
