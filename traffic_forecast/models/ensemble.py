"""
Ensemble model combining multiple predictors for improved accuracy.
Implements stacking, voting, and weighted averaging strategies.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


class EnsemblePredictor:
    """
    Ensemble predictor combining multiple trained models.
    """

    STRATEGIES = ['voting', 'weighted', 'stacking']

    def __init__(self, models: Dict[str, Any], strategy: str = 'weighted'):
    """
 Initialize ensemble predictor.

 Args:
 models: Dictionary of {name: model} pairs
 strategy: Ensemble strategy ('voting', 'weighted', 'stacking')
 """
    if strategy not in self.STRATEGIES:
    raise ValueError(f"Strategy must be one of {self.STRATEGIES}")

    self.models = models
    self.strategy = strategy
    self.weights = None
    self.meta_model = None

    def fit_weights(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series
    ):
    """
 Fit optimal weights based on validation performance.
 Uses inverse RMSE as weights.
 """
    logger.info("Fitting ensemble weights...")

    rmses = {}
    for name, model in self.models.items():
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    rmses[name] = rmse
    logger.info(f"{name} - Validation RMSE: {rmse:.4f}")

    # Calculate weights as inverse RMSE (normalized)
    inv_rmses = {name: 1 / rmse for name, rmse in rmses.items()}
    total_inv = sum(inv_rmses.values())
    self.weights = {name: inv / total_inv for name, inv in inv_rmses.items()}

    logger.info("Ensemble weights:")
    for name, weight in self.weights.items():
    logger.info(f" {name}: {weight:.4f}")

    def fit_stacking(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        meta_model=None
    ):
    """
 Fit stacking ensemble with meta-learner.

 Args:
 meta_model: Meta-learner model (default: Ridge regression)
 """
    from sklearn.linear_model import Ridge

    logger.info("Fitting stacking ensemble...")

    # Generate meta-features from base models
    meta_features_train = []
    meta_features_val = []

    for name, model in self.models.items():
        # Train predictions
    y_pred_train = model.predict(X_train)
    meta_features_train.append(y_pred_train)

    # Validation predictions
    y_pred_val = model.predict(X_val)
    meta_features_val.append(y_pred_val)

    # Stack meta-features
    X_meta_train = np.column_stack(meta_features_train)
    X_meta_val = np.column_stack(meta_features_val)

    # Train meta-model
    if meta_model is None:
    meta_model = Ridge(alpha=1.0)

    self.meta_model = meta_model
    self.meta_model.fit(X_meta_train, y_train)

    # Evaluate
    y_pred_val = self.meta_model.predict(X_meta_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

    logger.info(f"Stacking ensemble - Validation RMSE: {rmse:.4f}")

    # Store meta-model coefficients as interpretable weights
    if hasattr(self.meta_model, 'coef_'):
    coefs = self.meta_model.coef_
    total = np.abs(coefs).sum()
    self.weights = {
        name: abs(coef) / total
        for name, coef in zip(self.models.keys(), coefs)
    }
    logger.info("Meta-model weights (normalized):")
    for name, weight in self.weights.items():
    logger.info(f" {name}: {weight:.4f}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
    """
 Make ensemble predictions.

 Args:
 X: Features

 Returns:
 Ensemble predictions
 """
    if self.strategy == 'voting':
    return self._predict_voting(X)
    elif self.strategy == 'weighted':
    return self._predict_weighted(X)
    elif self.strategy == 'stacking':
    return self._predict_stacking(X)

    def _predict_voting(self, X: pd.DataFrame) -> np.ndarray:
    """Simple averaging of predictions."""
    predictions = []
    for model in self.models.values():
    predictions.append(model.predict(X))

    return np.mean(predictions, axis=0)

    def _predict_weighted(self, X: pd.DataFrame) -> np.ndarray:
    """Weighted averaging based on validation performance."""
    if self.weights is None:
    raise ValueError("Weights not fitted. Call fit_weights() first.")

    ensemble_pred = np.zeros(len(X))
    for name, model in self.models.items():
    pred = model.predict(X)
    ensemble_pred += self.weights[name] * pred

    return ensemble_pred

    def _predict_stacking(self, X: pd.DataFrame) -> np.ndarray:
    """Stacking with meta-learner."""
    if self.meta_model is None:
    raise ValueError("Meta-model not fitted. Call fit_stacking() first.")

    # Generate meta-features
    meta_features = []
    for model in self.models.values():
    meta_features.append(model.predict(X))

    X_meta = np.column_stack(meta_features)

    return self.meta_model.predict(X_meta)

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
    """
 Evaluate ensemble on test set.

 Returns:
 Dictionary of metrics
 """
    y_pred = self.predict(X_test)

    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    }

    logger.info(f"Ensemble - Test RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")

    return metrics

    def save(self, output_dir: Path, name: str = "ensemble"):
    """Save ensemble model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config = {
        'strategy': self.strategy,
        'weights': self.weights,
        'model_names': list(self.models.keys())
    }

    config_path = output_dir / f"{name}_config.pkl"
    joblib.dump(config, config_path)

    # Save meta-model if stacking
    if self.meta_model is not None:
    meta_path = output_dir / f"{name}_meta_model.pkl"
    joblib.dump(self.meta_model, meta_path)

    logger.info(f"Saved ensemble config to {config_path}")

    return config_path


def create_ensemble_from_results(
    results: Dict[str, Dict],
    strategy: str = 'weighted',
    top_k: Optional[int] = None
) -> EnsemblePredictor:
    """
    Create ensemble from training results.

    Args:
    results: Dictionary of training results
    strategy: Ensemble strategy
    top_k: Only use top-k models (by RMSE)

    Returns:
    EnsemblePredictor instance
    """
    # Sort models by test RMSE
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1]['test_metrics']['rmse']
    )

    # Select top-k if specified
    if top_k is not None:
    sorted_models = sorted_models[:top_k]

    # Extract models
    models = {name: res['model'] for name, res in sorted_models}

    logger.info(f"Creating {strategy} ensemble with {len(models)} models:")
    for name, res in sorted_models:
    logger.info(f" {name} - RMSE: {res['test_metrics']['rmse']:.4f}")

    return EnsemblePredictor(models, strategy=strategy)


def compare_ensembles(
    results: Dict[str, Dict],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Dict]:
    """
    Compare different ensemble strategies.

    Returns:
    Dictionary of {strategy: metrics}
    """
    ensemble_results = {}

    for strategy in EnsemblePredictor.STRATEGIES:
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {strategy.upper()} ensemble")
    logger.info(f"{'='*60}")

    try:
        # Create ensemble
    ensemble = create_ensemble_from_results(results, strategy=strategy)

    # Fit if needed
    if strategy == 'weighted':
    ensemble.fit_weights(X_val, y_val)
    elif strategy == 'stacking':
        # Extract train data from results (if available)
        # For now, use validation as proxy
    ensemble.fit_stacking(X_val, y_val, X_val, y_val)

    # Evaluate
    metrics = ensemble.evaluate(X_test, y_test)

    ensemble_results[strategy] = {
        'ensemble': ensemble,
        'metrics': metrics
    }

    except Exception as e:
    logger.error(f"Error with {strategy} ensemble: {e}")
    continue

    # Find best ensemble
    best_strategy = min(
        ensemble_results.keys(),
        key=lambda s: ensemble_results[s]['metrics']['rmse']
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"ENSEMBLE COMPARISON RESULTS")
    logger.info(f"{'='*60}")
    for strategy, res in ensemble_results.items():
    metrics = res['metrics']
    marker = " " if strategy == best_strategy else ""
    logger.info(f"{strategy.upper()}{marker}:")
    logger.info(f" RMSE: {metrics['rmse']:.4f}")
    logger.info(f" MAE: {metrics['mae']:.4f}")
    logger.info(f" R²: {metrics['r2']:.4f}")

    return ensemble_results


def main():
    """Main script for ensemble model creation."""
    import argparse
    import json

    parser = argparse.ArgumentParser(description='Create ensemble model')
    parser.add_argument('--results', type=str, default='models/training_results.json',
                        help='Path to training results JSON')
    parser.add_argument('--data', type=str, default='data/processed/val.parquet',
                        help='Path to validation data')
    parser.add_argument('--strategy', type=str, default='weighted',
                        choices=EnsemblePredictor.STRATEGIES,
                        help='Ensemble strategy')
    parser.add_argument('--output', type=str, default='models',
                        help='Output directory')
    args = parser.parse_args()

    from traffic_forecast import PROJECT_ROOT

    # Load training results
    results_path = PROJECT_ROOT / args.results
    with open(results_path) as f:
    results_json = json.load(f)

    # Note: This is simplified - in practice, you'd need to load the actual models
    logger.info("Note: Load actual trained models to create functional ensemble")
    logger.info(f"Results loaded from {results_path}")

    # For demonstration
    logger.info(f"\nAvailable models:")
    for name, res in results_json.items():
    logger.info(f" {name}: RMSE={res['test_metrics']['rmse']:.4f}")


if __name__ == '__main__':
    main()
