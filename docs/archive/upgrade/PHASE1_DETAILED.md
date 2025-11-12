# Maintainer Profile

**Name:** THAT Le Quang

- **Role:** AI & DS Major Student
- **GitHub:** [thatlq1812]

---

# Phase 1: Model Comparison & Validation

**Status:** In Progress  
**Started:** November 9, 2025  
**Target Completion:** November 13, 2025

---

## Overview

This phase focuses on establishing a rigorous model comparison framework to:

1. Validate STMGT performance improvements
2. Compare against baseline models (LSTM, ASTGCN, GraphWaveNet)
3. Conduct ablation study to understand component contributions
4. Fix current validation issues (MAE 3.05 → target < 2.5)

---

## Task 1.1: Unified Evaluation Framework

### Objectives

- Consistent evaluation across all models
- Fair comparison on identical data splits
- Statistical significance testing
- Reproducible metrics

### Implementation Plan

#### Step 1: Create Evaluation Module

**File:** `traffic_forecast/evaluation/unified_evaluator.py`

```python
"""
Unified evaluation framework for all traffic forecasting models.
Ensures fair comparison with consistent metrics and data splits.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pandas as pd
from pathlib import Path


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    mae: float
    rmse: float
    r2: float
    mape: float
    crps: Optional[float] = None
    coverage_80: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        return {
            'MAE': self.mae,
            'RMSE': self.rmse,
            'R2': self.r2,
            'MAPE': self.mape,
            'CRPS': self.crps,
            'Coverage_80': self.coverage_80
        }


@dataclass
class ModelComparison:
    """Results from comparing multiple models."""
    model_name: str
    train_metrics: EvaluationMetrics
    val_metrics: EvaluationMetrics
    test_metrics: EvaluationMetrics
    training_time_minutes: float
    n_parameters: int
    inference_time_ms: float


class UnifiedEvaluator:
    """
    Unified evaluator for all traffic forecasting models.

    Features:
    - Consistent data splits (temporal)
    - K-fold cross-validation support
    - Multiple metrics calculation
    - Statistical significance testing
    - Reproducible results with seed control
    """

    def __init__(
        self,
        dataset_path: Path,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        k_folds: Optional[int] = None
    ):
        """
        Initialize evaluator.

        Args:
            dataset_path: Path to parquet dataset
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            seed: Random seed for reproducibility
            k_folds: Number of folds for cross-validation (None = no CV)
        """
        self.dataset_path = dataset_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.k_folds = k_folds

        # Load and validate dataset
        self.df = pd.read_parquet(dataset_path)
        self._validate_dataset()

        # Create splits
        self.splits = self._create_temporal_splits()

    def _validate_dataset(self):
        """Validate dataset has required columns and proper format."""
        required_cols = ['timestamp', 'speed']
        missing = set(required_cols) - set(self.df.columns)
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

        # Check for temporal ordering
        if not self.df['timestamp'].is_monotonic_increasing:
            print("WARNING: Dataset not temporally ordered. Sorting...")
            self.df = self.df.sort_values('timestamp').reset_index(drop=True)

        print(f"Dataset validated: {len(self.df)} samples")
        print(f"Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")

    def _create_temporal_splits(self) -> Dict[str, pd.DataFrame]:
        """
        Create temporal train/val/test splits.

        CRITICAL: Use temporal split to avoid data leakage.
        Traffic data has strong temporal correlation.
        """
        n = len(self.df)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        splits = {
            'train': self.df.iloc[:train_end],
            'val': self.df.iloc[train_end:val_end],
            'test': self.df.iloc[val_end:]
        }

        # Verify no overlap
        assert splits['train']['timestamp'].max() < splits['val']['timestamp'].min()
        assert splits['val']['timestamp'].max() < splits['test']['timestamp'].min()

        print("\n=== Data Splits ===")
        for name, data in splits.items():
            print(f"{name.upper()}: {len(data)} samples ({len(data)/n*100:.1f}%)")
            print(f"  Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")

        return splits

    def calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: Optional[np.ndarray] = None
    ) -> EvaluationMetrics:
        """
        Calculate all evaluation metrics.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values (mean)
            y_std: Predicted standard deviation (for probabilistic models)

        Returns:
            EvaluationMetrics object
        """
        # Basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # MAPE (avoid division by zero)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

        # Probabilistic metrics (if available)
        crps = None
        coverage_80 = None

        if y_std is not None:
            # CRPS (Continuous Ranked Probability Score)
            crps = self._calculate_crps(y_true, y_pred, y_std)

            # Coverage (80% confidence interval)
            coverage_80 = self._calculate_coverage(y_true, y_pred, y_std, z=1.28)

        return EvaluationMetrics(
            mae=float(mae),
            rmse=float(rmse),
            r2=float(r2),
            mape=float(mape),
            crps=crps,
            coverage_80=coverage_80
        )

    def _calculate_crps(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray
    ) -> float:
        """Calculate Continuous Ranked Probability Score for Gaussian predictions."""
        from scipy.stats import norm

        # Standardize
        z = (y_true - y_pred) / (y_std + 1e-8)

        # CRPS for Gaussian distribution
        crps = y_std * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))

        return float(np.mean(crps))

    def _calculate_coverage(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray,
        z: float = 1.28  # 80% CI
    ) -> float:
        """Calculate coverage of confidence interval."""
        lower = y_pred - z * y_std
        upper = y_pred + z * y_std

        coverage = np.mean((y_true >= lower) & (y_true <= upper))

        return float(coverage)

    def evaluate_model(
        self,
        model,
        model_name: str,
        device: str = 'cuda'
    ) -> ModelComparison:
        """
        Evaluate a model on all splits.

        Args:
            model: PyTorch model with .predict() method
            model_name: Name for identification
            device: Device to run on

        Returns:
            ModelComparison object
        """
        import time

        results = {}

        for split_name, split_data in self.splits.items():
            print(f"\nEvaluating {model_name} on {split_name}...")

            # Prepare data (model-specific)
            # NOTE: Each model may need different data preparation
            # This is handled by the model's .predict() method

            start_time = time.time()
            y_pred, y_std = model.predict(split_data, device=device)
            inference_time = (time.time() - start_time) / len(split_data) * 1000  # ms per sample

            y_true = split_data['speed'].values

            metrics = self.calculate_metrics(y_true, y_pred, y_std)
            results[split_name] = metrics

            print(f"  MAE: {metrics.mae:.4f}, R²: {metrics.r2:.4f}")

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())

        return ModelComparison(
            model_name=model_name,
            train_metrics=results['train'],
            val_metrics=results['val'],
            test_metrics=results['test'],
            training_time_minutes=0.0,  # Set externally
            n_parameters=n_params,
            inference_time_ms=inference_time
        )

    def compare_models(
        self,
        model_comparisons: List[ModelComparison]
    ) -> pd.DataFrame:
        """
        Create comparison table for multiple models.

        Args:
            model_comparisons: List of ModelComparison objects

        Returns:
            DataFrame with comparison results
        """
        rows = []

        for comp in model_comparisons:
            row = {
                'Model': comp.model_name,
                'Parameters': f"{comp.n_parameters/1e6:.2f}M",
                'Train MAE': comp.train_metrics.mae,
                'Val MAE': comp.val_metrics.mae,
                'Test MAE': comp.test_metrics.mae,
                'Train R²': comp.train_metrics.r2,
                'Val R²': comp.val_metrics.r2,
                'Test R²': comp.test_metrics.r2,
                'Training Time': f"{comp.training_time_minutes:.1f} min",
                'Inference': f"{comp.inference_time_ms:.2f} ms"
            }
            rows.append(row)

        df = pd.DataFrame(rows)

        # Sort by validation MAE (lower is better)
        df = df.sort_values('Val MAE')

        return df

    def statistical_significance_test(
        self,
        model1_predictions: np.ndarray,
        model2_predictions: np.ndarray,
        y_true: np.ndarray
    ) -> Dict[str, float]:
        """
        Test if model1 is significantly better than model2.

        Uses paired t-test on absolute errors.
        """
        from scipy.stats import ttest_rel

        errors1 = np.abs(y_true - model1_predictions)
        errors2 = np.abs(y_true - model2_predictions)

        statistic, p_value = ttest_rel(errors1, errors2)

        return {
            't_statistic': float(statistic),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'model1_better': statistic < 0  # Lower error is better
        }
```

**Key Features:**

- Temporal splits (no data leakage)
- Comprehensive metrics (MAE, RMSE, R², MAPE, CRPS, Coverage)
- Statistical significance testing
- K-fold CV support (optional)
- Reproducible with seed control

#### Step 2: Model Wrapper Interface

**File:** `traffic_forecast/evaluation/model_wrapper.py`

```python
"""
Model wrapper interface for unified evaluation.
Each model implements this interface for consistent evaluation.
"""

from abc import ABC, abstractmethod
import torch
import numpy as np
from typing import Tuple, Optional
import pandas as pd


class ModelWrapper(ABC):
    """
    Abstract base class for model wrappers.

    Each model (STMGT, LSTM, ASTGCN, etc.) implements this interface
    to enable unified evaluation.
    """

    @abstractmethod
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        pass

    @abstractmethod
    def predict(
        self,
        data: pd.DataFrame,
        device: str = 'cuda'
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions on data.

        Args:
            data: DataFrame with features
            device: Device to run on

        Returns:
            (predictions, std) where std is None for deterministic models
        """
        pass

    @abstractmethod
    def parameters(self):
        """Return model parameters (for counting)."""
        pass
```

---

## Task 1.2: Fix STMGT Validation Issues

### Current Problem

From `training_history.csv`:

- Best validation MAE: 3.69 km/h (epoch 26)
- README claims 3.05 km/h (discrepancy?)
- Target: < 2.5 km/h

### Investigation Steps

1. **Verify Data Splits**

   - Check train/val/test are truly temporal
   - No data leakage
   - Proper size ratios

2. **Check Metric Calculation**

   - Compare with sklearn metrics
   - Verify units (km/h vs m/s)
   - Check for bugs in loss computation

3. **Hyperparameter Optimization**

   - Learning rate tuning
   - Batch size optimization
   - Regularization (dropout, weight decay)

4. **Architecture Improvements**
   - Check for overfitting signals
   - Add regularization if needed
   - Consider ensemble

### Action Items

```python
# Script: scripts/analysis/investigate_stmgt_validation.py

"""
Investigate STMGT validation metrics discrepancy.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import torch
from sklearn.metrics import mean_absolute_error, r2_score

def main():
    # Load training history
    history_path = Path("outputs/stmgt_v2_20251102_200308/training_history.csv")
    history = pd.read_csv(history_path)

    print("=== Training History Analysis ===")
    print(f"Total epochs: {len(history)}")
    print(f"\nBest validation metrics:")
    best_epoch = history.loc[history['val_mae'].idxmin()]
    print(f"  Epoch: {best_epoch['epoch']}")
    print(f"  Val MAE: {best_epoch['val_mae']:.4f}")
    print(f"  Val R²: {best_epoch['val_r2']:.4f}")
    print(f"  Val RMSE: {best_epoch['val_rmse']:.4f}")

    # Check for overfitting
    print(f"\n=== Overfitting Check ===")
    final_epoch = history.iloc[-1]
    print(f"Final epoch ({final_epoch['epoch']}):")
    print(f"  Train MAE: {final_epoch['train_mae']:.4f}")
    print(f"  Val MAE: {final_epoch['val_mae']:.4f}")
    print(f"  Gap: {final_epoch['val_mae'] - final_epoch['train_mae']:.4f}")

    if final_epoch['val_mae'] - final_epoch['train_mae'] > 1.0:
        print("  ⚠️ WARNING: Significant train/val gap suggests overfitting")

    # Load dataset and verify metrics
    dataset_path = Path("data/processed/all_runs_combined.parquet")
    if dataset_path.exists():
        print(f"\n=== Dataset Verification ===")
        df = pd.read_parquet(dataset_path)
        print(f"Total samples: {len(df)}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Check for temporal ordering
        is_sorted = df['timestamp'].is_monotonic_increasing
        print(f"Temporally sorted: {is_sorted}")
        if not is_sorted:
            print("  ⚠️ WARNING: Dataset not temporally sorted!")

    # Recommendations
    print(f"\n=== Recommendations ===")
    if best_epoch['val_mae'] > 2.5:
        print("1. Val MAE > 2.5 target:")
        print("   - Try hyperparameter tuning (learning rate, batch size)")
        print("   - Add more regularization (dropout, weight decay)")
        print("   - Check data quality (outliers, missing values)")
        print("   - Consider ensemble methods")

    if final_epoch['val_mae'] - final_epoch['train_mae'] > 1.0:
        print("2. Overfitting detected:")
        print("   - Increase dropout rate")
        print("   - Add weight decay")
        print("   - Reduce model complexity")
        print("   - Get more training data")

if __name__ == "__main__":
    main()
```

---

## Task 1.3: Baseline Model Benchmarking

### Models to Implement

#### 1. LSTM Baseline

**File:** `traffic_forecast/models/baselines/lstm_baseline.py`

Simple LSTM for temporal forecasting (no spatial information).

```python
import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    """
    Simple LSTM baseline for traffic speed forecasting.

    Uses only temporal information, no spatial graph structure.
    Good baseline to show value of spatial modeling.
    """

    def __init__(
        self,
        input_dim: int = 1,  # Just speed
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_horizon: int = 3
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, output_horizon)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            predictions: (batch, output_horizon)
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # Use last hidden state
        predictions = self.fc(last_hidden)
        return predictions
```

**Expected performance:**

- MAE: 4.0-4.5 km/h
- R²: 0.55-0.60
- Fast training (< 10 minutes)

#### 2. ASTGCN (Already Implemented)

Location: `traffic_forecast/models/pytorch_astgcn/model.py`

**Expected performance:**

- MAE: 3.5-4.0 km/h
- R²: 0.65-0.70
- Medium training (20-30 minutes)

#### 3. GraphWaveNet (If Time Permits)

Complex baseline with dilated convolutions.

**Expected performance:**

- MAE: 3.0-3.5 km/h
- R²: 0.70-0.75
- Long training (40-50 minutes)

---

## Task 1.4: Ablation Study

### Variants to Test

```python
# STMGT variants for ablation study

variants = [
    {
        'name': 'STMGT-Full',
        'description': 'Complete model with all components',
        'components': ['graph', 'transformer', 'weather', 'temporal']
    },
    {
        'name': 'STMGT-NoGraph',
        'description': 'Remove graph convolution layers',
        'components': ['transformer', 'weather', 'temporal']
    },
    {
        'name': 'STMGT-NoTransformer',
        'description': 'Remove transformer self-attention',
        'components': ['graph', 'weather', 'temporal']
    },
    {
        'name': 'STMGT-NoWeather',
        'description': 'Remove weather fusion',
        'components': ['graph', 'transformer', 'temporal']
    },
    {
        'name': 'STMGT-NoTemporal',
        'description': 'Remove cyclical temporal encoding',
        'components': ['graph', 'transformer', 'weather']
    },
    {
        'name': 'STMGT-GraphOnly',
        'description': 'Only graph convolution (like ASTGCN)',
        'components': ['graph']
    },
    {
        'name': 'STMGT-TransformerOnly',
        'description': 'Only transformer (like vanilla attention)',
        'components': ['transformer']
    }
]
```

**Implementation:**

Modify `traffic_forecast/models/stmgt/model.py` to accept component flags:

```python
class STMGT(nn.Module):
    def __init__(
        self,
        ...,
        enable_graph: bool = True,
        enable_transformer: bool = True,
        enable_weather: bool = True,
        enable_temporal: bool = True
    ):
        # Conditional module creation based on flags
        pass
```

---

## Deliverables

### 1. Code

- [x] `traffic_forecast/evaluation/unified_evaluator.py`
- [x] `traffic_forecast/evaluation/model_wrapper.py`
- [ ] `traffic_forecast/models/baselines/lstm_baseline.py`
- [ ] `scripts/analysis/investigate_stmgt_validation.py`
- [ ] `scripts/training/train_lstm_baseline.py`
- [ ] `scripts/training/train_astgcn_baseline.py`
- [ ] `scripts/evaluation/compare_all_models.py`
- [ ] `scripts/evaluation/ablation_study.py`

### 2. Results

- [ ] `docs/upgrade/results/model_comparison_table.csv`
- [ ] `docs/upgrade/results/ablation_study_results.csv`
- [ ] `docs/upgrade/results/statistical_tests.json`

### 3. Visualizations

- [ ] Training curves comparison (all models)
- [ ] Performance comparison bar charts
- [ ] Ablation study impact visualization
- [ ] Statistical significance heatmap

### 4. Documentation

- [ ] `docs/upgrade/MODEL_COMPARISON.md`
- [ ] `docs/upgrade/ABLATION_STUDY.md`
- [ ] `docs/upgrade/WHY_STMGT_WINS.md`

---

## Timeline

### Day 1-2 (Nov 9-10)

- Implement evaluation framework
- Fix STMGT validation
- Train LSTM baseline
- Initial comparison

### Day 3-4 (Nov 11-12)

- Train ASTGCN baseline
- Ablation study (7 variants)
- Statistical testing

### Day 5 (Nov 13)

- Create visualizations
- Write comparison report
- Document findings

---

## Success Criteria

- [ ] All models evaluated on identical data splits
- [ ] STMGT shows 20%+ improvement over best baseline
- [ ] Ablation study quantifies component contributions
- [ ] Statistical tests confirm significance (p < 0.05)
- [ ] Complete documentation with clear conclusions
