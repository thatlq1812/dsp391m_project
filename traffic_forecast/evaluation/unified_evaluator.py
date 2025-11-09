"""
Unified evaluation framework for all traffic forecasting models.

Ensures fair comparison with consistent metrics and data splits across
different model architectures.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json


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
        """Convert to dictionary, handling None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        s = f"MAE: {self.mae:.4f}, RMSE: {self.rmse:.4f}, R²: {self.r2:.4f}, MAPE: {self.mape:.2f}%"
        if self.crps is not None:
            s += f", CRPS: {self.crps:.4f}"
        if self.coverage_80 is not None:
            s += f", Coverage: {self.coverage_80:.2%}"
        return s


@dataclass
class ModelComparison:
    """Results from comparing a single model."""
    model_name: str
    train_metrics: EvaluationMetrics
    val_metrics: EvaluationMetrics
    test_metrics: EvaluationMetrics
    training_time_minutes: float
    n_parameters: int
    inference_time_ms: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'model_name': self.model_name,
            'train_metrics': self.train_metrics.to_dict(),
            'val_metrics': self.val_metrics.to_dict(),
            'test_metrics': self.test_metrics.to_dict(),
            'training_time_minutes': self.training_time_minutes,
            'n_parameters': self.n_parameters,
            'inference_time_ms': self.inference_time_ms
        }


class UnifiedEvaluator:
    """
    Unified evaluator for all traffic forecasting models.
    
    Features:
    - Consistent temporal data splits (no leakage)
    - K-fold cross-validation support
    - Multiple evaluation metrics
    - Statistical significance testing
    - Reproducible results with seed control
    
    Example:
        >>> evaluator = UnifiedEvaluator("data/processed/all_runs_combined.parquet")
        >>> results = evaluator.evaluate_model(stmgt_wrapper, "STMGT")
        >>> comparison_df = evaluator.compare_models([results])
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
            train_ratio: Training set ratio (default 0.7)
            val_ratio: Validation set ratio (default 0.15)
            test_ratio: Test set ratio (default 0.15)
            seed: Random seed for reproducibility
            k_folds: Number of folds for cross-validation (None = single split)
        """
        self.dataset_path = Path(dataset_path)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.k_folds = k_folds
        
        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if not np.isclose(total, 1.0):
            raise ValueError(f"Ratios must sum to 1.0, got {total}")
        
        # Load and validate dataset
        print(f"Loading dataset from {self.dataset_path}...")
        self.df = pd.read_parquet(self.dataset_path)
        self._validate_dataset()
        
        # Create splits
        self.splits = self._create_temporal_splits()
        
    def _validate_dataset(self):
        """Validate dataset has required columns and proper format."""
        # Check for speed column (handle both 'speed' and 'speed_kmh')
        speed_col = None
        if 'speed_kmh' in self.df.columns:
            speed_col = 'speed_kmh'
        elif 'speed' in self.df.columns:
            speed_col = 'speed'
        else:
            raise ValueError("Dataset missing speed column (expected 'speed' or 'speed_kmh')")
        
        # Rename to standardized 'speed' for internal use
        if speed_col == 'speed_kmh':
            self.df = self.df.rename(columns={'speed_kmh': 'speed'})
        
        required_cols = ['timestamp', 'speed']
        missing = set(required_cols) - set(self.df.columns)
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")
        
        # Check for NaN values
        nan_counts = self.df[required_cols].isna().sum()
        if nan_counts.any():
            print(f"WARNING: Found NaN values: {nan_counts[nan_counts > 0].to_dict()}")
            
        # Check for temporal ordering
        if not self.df['timestamp'].is_monotonic_increasing:
            print("WARNING: Dataset not temporally ordered. Sorting...")
            self.df = self.df.sort_values('timestamp').reset_index(drop=True)
            
        print(f"[OK] Dataset validated: {len(self.df):,} samples")
        print(f"  Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        print(f"  Speed range: [{self.df['speed'].min():.2f}, {self.df['speed'].max():.2f}] km/h")
        
    def _create_temporal_splits(self) -> Dict[str, pd.DataFrame]:
        """
        Create temporal train/val/test splits.
        
        CRITICAL: Uses temporal split to avoid data leakage.
        Traffic data has strong temporal correlation, so random splits
        would allow the model to "cheat" by learning future patterns.
        
        IMPORTANT: Graph data has multiple edges per timestamp.
        We split by unique timestamps, not by row index.
        
        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames
        """
        # Get unique timestamps and split them
        unique_times = sorted(self.df['timestamp'].unique())
        n_times = len(unique_times)
        
        train_end = int(n_times * self.train_ratio)
        val_end = train_end + int(n_times * self.val_ratio)
        
        train_times = unique_times[:train_end]
        val_times = unique_times[train_end:val_end]
        test_times = unique_times[val_end:]
        
        splits = {
            'train': self.df[self.df['timestamp'].isin(train_times)].copy(),
            'val': self.df[self.df['timestamp'].isin(val_times)].copy(),
            'test': self.df[self.df['timestamp'].isin(test_times)].copy()
        }
        
        # Verify no temporal overlap (should be guaranteed by timestamp splitting)
        assert splits['train']['timestamp'].max() < splits['val']['timestamp'].min(), \
            "Train and val sets have temporal overlap!"
        assert splits['val']['timestamp'].max() < splits['test']['timestamp'].min(), \
            "Val and test sets have temporal overlap!"
        
        print("\n=== Data Splits (Temporal) ===")
        print(f"Total unique timestamps: {n_times}")
        print(f"Total samples: {len(self.df)}")
        for name, data in splits.items():
            n_unique = data['timestamp'].nunique()
            print(f"{name.upper():5s}: {len(data):6,} samples ({n_unique} timestamps)")
            print(f"       Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
            
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
            y_true: Ground truth values (n_samples,)
            y_pred: Predicted values (n_samples,)
            y_std: Prediction uncertainty (n_samples,) or None
            
        Returns:
            EvaluationMetrics object
        """
        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if y_std is not None:
            y_std = y_std[mask]
        
        if len(y_true) == 0:
            raise ValueError("No valid samples after removing NaNs")
        
        # Basic regression metrics
        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))
        
        # MAPE (avoid division by zero)
        epsilon = 1e-8
        mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100)
        
        # Probabilistic metrics (if available)
        crps = None
        coverage_80 = None
        
        if y_std is not None:
            # CRPS (Continuous Ranked Probability Score)
            crps = self._calculate_crps(y_true, y_pred, y_std)
            
            # Coverage (80% confidence interval)
            coverage_80 = self._calculate_coverage(y_true, y_pred, y_std, z=1.28)
        
        return EvaluationMetrics(
            mae=mae,
            rmse=rmse,
            r2=r2,
            mape=mape,
            crps=crps,
            coverage_80=coverage_80
        )
    
    def _calculate_crps(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_std: np.ndarray
    ) -> float:
        """
        Calculate Continuous Ranked Probability Score for Gaussian predictions.
        
        CRPS measures the quality of probabilistic forecasts.
        Lower is better (0 = perfect).
        """
        try:
            from scipy.stats import norm
        except ImportError:
            print("WARNING: scipy not available, skipping CRPS calculation")
            return None
        
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
        """
        Calculate coverage of confidence interval.
        
        Measures what fraction of ground truth values fall within
        the predicted confidence interval.
        
        For 80% CI, ideal coverage is 0.80.
        """
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
            model: Model wrapper implementing ModelWrapper interface
            model_name: Name for identification
            device: Device to run on ('cuda' or 'cpu')
            
        Returns:
            ModelComparison object with results
        """
        import time
        
        results = {}
        inference_times = []
        
        for split_name, split_data in self.splits.items():
            print(f"\nEvaluating {model_name} on {split_name} set...")
            
            # Time inference
            start_time = time.time()
            y_pred, y_std = model.predict(split_data, device=device)
            elapsed = time.time() - start_time
            inference_times.append(elapsed / len(split_data) * 1000)  # ms per sample
            
            y_true = split_data['speed'].values
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_true, y_pred, y_std)
            results[split_name] = metrics
            
            print(f"  {metrics}")
        
        # Count parameters
        try:
            n_params = sum(p.numel() for p in model.parameters())
        except:
            n_params = 0
            print(f"WARNING: Could not count parameters for {model_name}")
        
        avg_inference_time = np.mean(inference_times)
        
        return ModelComparison(
            model_name=model_name,
            train_metrics=results['train'],
            val_metrics=results['val'],
            test_metrics=results['test'],
            training_time_minutes=0.0,  # Set externally after training
            n_parameters=n_params,
            inference_time_ms=avg_inference_time
        )
    
    def compare_models(
        self,
        model_comparisons: List[ModelComparison],
        save_path: Optional[Path] = None
    ) -> pd.DataFrame:
        """
        Create comparison table for multiple models.
        
        Args:
            model_comparisons: List of ModelComparison objects
            save_path: Optional path to save CSV
            
        Returns:
            DataFrame with comparison results
        """
        rows = []
        
        for comp in model_comparisons:
            row = {
                'Model': comp.model_name,
                'Parameters': f"{comp.n_parameters/1e6:.2f}M",
                'Train MAE': f"{comp.train_metrics.mae:.4f}",
                'Val MAE': f"{comp.val_metrics.mae:.4f}",
                'Test MAE': f"{comp.test_metrics.mae:.4f}",
                'Train R²': f"{comp.train_metrics.r2:.4f}",
                'Val R²': f"{comp.val_metrics.r2:.4f}",
                'Test R²': f"{comp.test_metrics.r2:.4f}",
                'Training Time': f"{comp.training_time_minutes:.1f} min",
                'Inference': f"{comp.inference_time_ms:.2f} ms"
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by validation MAE (lower is better)
        # Sort by numeric value extracted from string
        df['_val_mae_numeric'] = df['Val MAE'].astype(float)
        df = df.sort_values('_val_mae_numeric').drop('_val_mae_numeric', axis=1)
        df = df.reset_index(drop=True)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_path, index=False)
            print(f"\n[OK] Comparison table saved to {save_path}")
        
        return df
    
    def statistical_significance_test(
        self,
        model1_predictions: np.ndarray,
        model2_predictions: np.ndarray,
        y_true: np.ndarray,
        model1_name: str = "Model 1",
        model2_name: str = "Model 2"
    ) -> Dict[str, any]:
        """
        Test if model1 is significantly better than model2.
        
        Uses paired t-test on absolute errors.
        
        Args:
            model1_predictions: Predictions from model 1
            model2_predictions: Predictions from model 2
            y_true: Ground truth values
            model1_name: Name for model 1
            model2_name: Name for model 2
            
        Returns:
            Dictionary with test results
        """
        try:
            from scipy.stats import ttest_rel
        except ImportError:
            print("WARNING: scipy not available, skipping significance test")
            return None
        
        errors1 = np.abs(y_true - model1_predictions)
        errors2 = np.abs(y_true - model2_predictions)
        
        statistic, p_value = ttest_rel(errors1, errors2)
        
        # Negative statistic means errors1 < errors2 (model1 better)
        model1_better = statistic < 0
        significant = p_value < 0.05
        
        result = {
            'model1_name': model1_name,
            'model2_name': model2_name,
            't_statistic': float(statistic),
            'p_value': float(p_value),
            'significant': significant,
            'model1_better': model1_better,
            'interpretation': self._interpret_significance_test(
                model1_name, model2_name, model1_better, significant, p_value
            )
        }
        
        return result
    
    def _interpret_significance_test(
        self,
        model1_name: str,
        model2_name: str,
        model1_better: bool,
        significant: bool,
        p_value: float
    ) -> str:
        """Generate human-readable interpretation of significance test."""
        if significant:
            if model1_better:
                return f"{model1_name} is significantly better than {model2_name} (p={p_value:.4f})"
            else:
                return f"{model2_name} is significantly better than {model1_name} (p={p_value:.4f})"
        else:
            return f"No significant difference between {model1_name} and {model2_name} (p={p_value:.4f})"
    
    def save_results(
        self,
        model_comparisons: List[ModelComparison],
        output_dir: Path
    ):
        """
        Save all comparison results to directory.
        
        Args:
            model_comparisons: List of ModelComparison objects
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comparison table
        df = self.compare_models(model_comparisons)
        df.to_csv(output_dir / "model_comparison.csv", index=False)
        
        # Save detailed results JSON
        results_json = {
            'models': [comp.to_dict() for comp in model_comparisons],
            'dataset': {
                'path': str(self.dataset_path),
                'n_samples': len(self.df),
                'date_range': [
                    str(self.df['timestamp'].min()),
                    str(self.df['timestamp'].max())
                ],
                'split_ratios': {
                    'train': self.train_ratio,
                    'val': self.val_ratio,
                    'test': self.test_ratio
                }
            }
        }
        
        with open(output_dir / "detailed_results.json", 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\n[OK] Results saved to {output_dir}/")
