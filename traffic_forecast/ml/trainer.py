"""
Model training module for ML pipeline.
Supports multiple algorithms with hyperparameter tuning and cross-validation.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
import warnings

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import cross_val_score, GridSearchCV

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    warnings.warn("XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    warnings.warn("LightGBM not installed. Install with: pip install lightgbm")

from traffic_forecast import PROJECT_ROOT


class ModelTrainer:
    """Train and evaluate ML models for traffic prediction."""

    MODELS = {
        'random_forest': RandomForestRegressor,
        'gradient_boosting': GradientBoostingRegressor,
        'ridge': Ridge,
        'lasso': Lasso,
    }

    if HAS_XGBOOST:
        MODELS['xgboost'] = xgb.XGBRegressor

    if HAS_LIGHTGBM:
        MODELS['lightgbm'] = lgb.LGBMRegressor

    DEFAULT_PARAMS = {
        'random_forest': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        },
        'gradient_boosting': {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42
        },
        'xgboost': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1
        },
        'lightgbm': {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        },
        'ridge': {
            'alpha': 1.0,
            'random_state': 42
        },
        'lasso': {
            'alpha': 1.0,
            'random_state': 42
        }
    }

    def __init__(
        self,
        model_type: str = 'random_forest',
        params: Optional[Dict] = None,
        model_dir: Optional[Path] = None
    ):
        """
        Initialize model trainer.

        Args:
            model_type: Type of model to train
            params: Model hyperparameters (uses defaults if None)
            model_dir: Directory to save models (defaults to PROJECT_ROOT/models)
        """
        if model_type not in self.MODELS:
            available = ', '.join(self.MODELS.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available}")

        self.model_type = model_type
        self.params = params or self.DEFAULT_PARAMS.get(model_type, {})
        self.model_dir = model_dir or PROJECT_ROOT / 'models'
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.feature_names = None
        self.training_metrics = {}
        self.validation_metrics = {}
        self.test_metrics = {}

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ):
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        """
        self.feature_names = list(X_train.columns)

        # Create model
        model_class = self.MODELS[self.model_type]
        self.model = model_class(**self.params)

        # Train
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)

        # Evaluate on training data
        y_train_pred = self.model.predict(X_train)
        self.training_metrics = self._calculate_metrics(y_train, y_train_pred, 'train')

        # Evaluate on validation data if provided
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            self.validation_metrics = self._calculate_metrics(y_val, y_val_pred, 'validation')

        print(f"Training complete. Train R²: {self.training_metrics['r2']:.4f}")
        if self.validation_metrics:
            print(f"Validation R²: {self.validation_metrics['r2']:.4f}")

        return self

    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        set_name: str = 'test'
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            X_test: Test features
            y_test: Test target
            set_name: Name of evaluation set

        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        y_pred = self.model.predict(X_test)
        metrics = self._calculate_metrics(y_test, y_pred, set_name)

        if set_name == 'test':
            self.test_metrics = metrics

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features DataFrame

        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict(X)

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = 'r2'
    ) -> Dict[str, Any]:
        """
        Perform cross-validation.

        Args:
            X: Features
            y: Target
            cv: Number of folds
            scoring: Scoring metric

        Returns:
            Dictionary with CV results
        """
        if self.model is None:
            model_class = self.MODELS[self.model_type]
            model = model_class(**self.params)
        else:
            model = self.model

        print(f"Running {cv}-fold cross-validation...")
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

        results = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist(),
            'cv_folds': cv,
            'scoring': scoring
        }

        print(f"CV {scoring}: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")

        return results

    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Dict[str, List],
        cv: int = 3,
        scoring: str = 'r2'
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using grid search.

        Args:
            X_train: Training features
            y_train: Training target
            param_grid: Grid of parameters to search
            cv: Number of CV folds
            scoring: Scoring metric

        Returns:
            Dictionary with best parameters and scores
        """
        model_class = self.MODELS[self.model_type]
        base_model = model_class()

        print(f"Tuning hyperparameters for {self.model_type}...")
        print(f"Parameter grid: {param_grid}")

        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        # Update model with best parameters
        self.params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        self.feature_names = list(X_train.columns)

        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': {
                'mean_test_score': grid_search.cv_results_['mean_test_score'].tolist(),
                'params': [str(p) for p in grid_search.cv_results_['params']]
            }
        }

        print(f"Best parameters: {results['best_params']}")
        print(f"Best CV score: {results['best_score']:.4f}")

        return results

    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importance scores.

        Args:
            top_n: Return only top N features (None = all)

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if not hasattr(self.model, 'feature_importances_'):
            warnings.warn(f"{self.model_type} does not support feature_importances_")
            return pd.DataFrame()

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        if top_n:
            importance_df = importance_df.head(top_n)

        return importance_df

    def save_model(self, name: Optional[str] = None) -> Path:
        """
        Save trained model to disk.

        Args:
            name: Model filename (auto-generated if None)

        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            name = f"{self.model_type}_{timestamp}.joblib"

        model_path = self.model_dir / name

        # Save model and metadata
        save_data = {
            'model': self.model,
            'model_type': self.model_type,
            'params': self.params,
            'feature_names': self.feature_names,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'test_metrics': self.test_metrics,
            'timestamp': datetime.now().isoformat()
        }

        joblib.dump(save_data, model_path)
        print(f"Model saved to: {model_path}")

        # Save metrics as JSON for easy reading
        metrics_path = model_path.with_suffix('.json')
        with metrics_path.open('w') as f:
            json.dump({
                'model_type': self.model_type,
                'params': self.params,
                'training_metrics': self.training_metrics,
                'validation_metrics': self.validation_metrics,
                'test_metrics': self.test_metrics,
                'feature_count': len(self.feature_names) if self.feature_names else 0
            }, f, indent=2)

        return model_path

    @classmethod
    def load_model(cls, model_path: Path) -> 'ModelTrainer':
        """
        Load trained model from disk.

        Args:
            model_path: Path to saved model

        Returns:
            ModelTrainer instance with loaded model
        """
        save_data = joblib.load(model_path)

        trainer = cls(
            model_type=save_data['model_type'],
            params=save_data['params']
        )

        trainer.model = save_data['model']
        trainer.feature_names = save_data['feature_names']
        trainer.training_metrics = save_data.get('training_metrics', {})
        trainer.validation_metrics = save_data.get('validation_metrics', {})
        trainer.test_metrics = save_data.get('test_metrics', {})

        print(f"Model loaded from: {model_path}")

        return trainer

    @staticmethod
    def _calculate_metrics(y_true: pd.Series, y_pred: np.ndarray, set_name: str = '') -> Dict[str, float]:
        """Calculate regression metrics."""
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
        }

        # MAPE (handle division by zero)
        try:
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        except Exception:
            metrics['mape'] = float('inf')

        if set_name:
            print(f"\n{set_name.capitalize()} Metrics:")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  MAE:  {metrics['mae']:.4f}")
            print(f"  R²:   {metrics['r2']:.4f}")
            if metrics['mape'] != float('inf'):
                print(f"  MAPE: {metrics['mape']:.4f}")

        return metrics


def compare_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compare multiple models on the same data.

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        models: List of model types to compare (None = all available)

    Returns:
        DataFrame with comparison results
    """
    if models is None:
        models = list(ModelTrainer.MODELS.keys())

    results = []

    for model_type in models:
        print(f"\n{'='*60}")
        print(f"Training {model_type}...")
        print(f"{'='*60}")

        try:
            trainer = ModelTrainer(model_type=model_type)
            trainer.train(X_train, y_train)
            test_metrics = trainer.evaluate(X_test, y_test)

            results.append({
                'model': model_type,
                'rmse': test_metrics['rmse'],
                'mae': test_metrics['mae'],
                'r2': test_metrics['r2'],
                'mape': test_metrics.get('mape', float('inf'))
            })
        except Exception as e:
            print(f"Error training {model_type}: {e}")

    comparison_df = pd.DataFrame(results).sort_values('r2', ascending=False)

    print(f"\n{'='*60}")
    print("Model Comparison Results:")
    print(f"{'='*60}")
    print(comparison_df.to_string(index=False))

    return comparison_df
