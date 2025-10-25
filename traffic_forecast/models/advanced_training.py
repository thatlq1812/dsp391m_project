"""
Advanced ML training module with MLflow tracking, hyperparameter tuning,
and model comparison for traffic speed forecasting.
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logging.warning("XGBoost not installed. Install with: pip install xgboost")

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    logging.warning("MLflow not installed. Install with: pip install mlflow")

from traffic_forecast import PROJECT_ROOT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrafficSpeedPredictor:
    """
    Advanced traffic speed prediction with multiple models and MLflow tracking.
    """
    
    MODELS = {
        'linear_regression': LinearRegression,
        'ridge': Ridge,
        'lasso': Lasso,
        'random_forest': RandomForestRegressor,
        'gradient_boosting': GradientBoostingRegressor,
    }
    
    if HAS_XGBOOST:
        MODELS['xgboost'] = xgb.XGBRegressor
    
    def __init__(self, experiment_name: str = "traffic_speed_prediction"):
        """
        Initialize predictor with MLflow experiment.
        
        Args:
            experiment_name: Name for MLflow experiment
        """
        self.experiment_name = experiment_name
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        
        if HAS_MLFLOW:
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment: {experiment_name}")
    
    def load_data(
        self,
        data_path: Path,
        target_col: str = 'avg_speed_kmh',
        feature_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Load and split data.
        
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info(f"Loading data from {data_path}")
        
        # Load data
        if data_path.suffix == '.parquet':
            df = pd.read_parquet(data_path)
        elif data_path.suffix == '.csv':
            df = pd.read_csv(data_path)
        elif data_path.suffix == '.json':
            df = pd.read_json(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
        
        logger.info(f"Loaded {len(df)} records")
        
        # Select features
        if feature_cols is None:
            # Auto-detect numeric features
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in feature_cols if c != target_col]
        
        logger.info(f"Using {len(feature_cols)} features: {feature_cols}")
        
        # Prepare X and y
        X = df[feature_cols]
        y = df[target_col]
        
        # Handle missing values
        X = X.fillna(X.mean())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )
        
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_model(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        params: Optional[Dict[str, Any]] = None,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Train a single model with cross-validation.
        
        Returns:
            Dictionary with model and metrics
        """
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODELS.keys())}")
        
        logger.info(f"Training {model_name}...")
        
        # Initialize model
        model_class = self.MODELS[model_name]
        if params is None:
            params = self._get_default_params(model_name)
        
        model = model_class(**params)
        
        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        cv_rmse = np.sqrt(-cv_scores)
        
        # Train on full training set
        model.fit(X_train, y_train)
        
        # Store model
        self.models[model_name] = model
        
        metrics = {
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std(),
            'cv_scores': cv_scores.tolist()
        }
        
        logger.info(f"{model_name} - CV RMSE: {metrics['cv_rmse_mean']:.4f} ± {metrics['cv_rmse_std']:.4f}")
        
        return {'model': model, 'metrics': metrics, 'params': params}
    
    def hyperparameter_tuning(
        self,
        model_name: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        param_grid: Optional[Dict[str, List]] = None,
        cv_folds: int = 5
    ) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning with GridSearchCV.
        
        Returns:
            Best model and parameters
        """
        logger.info(f"Hyperparameter tuning for {model_name}...")
        
        if param_grid is None:
            param_grid = self._get_param_grid(model_name)
        
        model_class = self.MODELS[model_name]
        base_model = model_class()
        
        # GridSearchCV
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = np.sqrt(-grid_search.best_score_)
        
        logger.info(f"Best params: {best_params}")
        logger.info(f"Best CV RMSE: {best_score:.4f}")
        
        # Store best model
        self.models[f"{model_name}_tuned"] = best_model
        
        return {
            'model': best_model,
            'params': best_params,
            'cv_rmse': best_score,
            'grid_search': grid_search
        }
    
    def evaluate_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: str = "model"
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Returns:
            Dictionary of metrics
        """
        y_pred = model.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        }
        
        logger.info(f"{model_name} - Test RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        tune_hyperparams: bool = False
    ) -> Dict[str, Dict]:
        """
        Train and evaluate all available models.
        
        Returns:
            Dictionary with results for all models
        """
        results = {}
        
        for model_name in self.MODELS.keys():
            try:
                with mlflow.start_run(run_name=model_name) if HAS_MLFLOW else nullcontext():
                    # Train model
                    if tune_hyperparams:
                        train_result = self.hyperparameter_tuning(model_name, X_train, y_train)
                        model = train_result['model']
                        params = train_result['params']
                    else:
                        train_result = self.train_model(model_name, X_train, y_train)
                        model = train_result['model']
                        params = train_result['params']
                    
                    # Evaluate
                    test_metrics = self.evaluate_model(model, X_test, y_test, model_name)
                    
                    # Log to MLflow
                    if HAS_MLFLOW:
                        mlflow.log_params(params)
                        mlflow.log_metrics(test_metrics)
                        mlflow.log_metrics({
                            'cv_rmse_mean': train_result['metrics']['cv_rmse_mean'],
                            'cv_rmse_std': train_result['metrics']['cv_rmse_std']
                        })
                        mlflow.sklearn.log_model(model, model_name)
                    
                    # Store results
                    results[model_name] = {
                        'model': model,
                        'params': params,
                        'train_metrics': train_result['metrics'],
                        'test_metrics': test_metrics
                    }
                    
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        # Find best model
        best_model_name = min(results.keys(), key=lambda k: results[k]['test_metrics']['rmse'])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        logger.info(f"\n🏆 Best model: {best_model_name} (RMSE: {results[best_model_name]['test_metrics']['rmse']:.4f})")
        
        return results
    
    def save_model(self, model_name: str, output_dir: Path):
        """Save trained model to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model = self.models.get(model_name)
        if model is None:
            raise ValueError(f"Model {model_name} not found")
        
        # Save model
        model_path = output_dir / f"{model_name}.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Saved model to {model_path}")
        
        return model_path
    
    def save_best_model(self, output_dir: Path):
        """Save the best performing model."""
        if self.best_model is None:
            raise ValueError("No models trained yet")
        
        return self.save_model(self.best_model_name, output_dir)
    
    @staticmethod
    def _get_default_params(model_name: str) -> Dict[str, Any]:
        """Get default parameters for a model."""
        defaults = {
            'linear_regression': {},
            'ridge': {'alpha': 1.0},
            'lasso': {'alpha': 1.0},
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'random_state': 42,
                'n_jobs': -1
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42
            },
        }
        
        if HAS_XGBOOST:
            defaults['xgboost'] = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'random_state': 42,
                'n_jobs': -1
            }
        
        return defaults.get(model_name, {})
    
    @staticmethod
    def _get_param_grid(model_name: str) -> Dict[str, List]:
        """Get parameter grid for hyperparameter tuning."""
        grids = {
            'ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'lasso': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
        
        if HAS_XGBOOST:
            grids['xgboost'] = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
        
        return grids.get(model_name, {})


# Null context manager for when MLflow is not available
class nullcontext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def main():
    """Main training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train traffic speed prediction models')
    parser.add_argument('--data', type=str, default='data/processed/train.parquet',
                       help='Path to training data')
    parser.add_argument('--output', type=str, default='models',
                       help='Output directory for models')
    parser.add_argument('--tune', action='store_true',
                       help='Enable hyperparameter tuning')
    parser.add_argument('--target', type=str, default='avg_speed_kmh',
                       help='Target column name')
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = TrafficSpeedPredictor()
    
    # Load data
    data_path = PROJECT_ROOT / args.data
    X_train, X_test, y_train, y_test = predictor.load_data(data_path, target_col=args.target)
    
    # Train all models
    results = predictor.train_all_models(
        X_train, X_test, y_train, y_test,
        tune_hyperparams=args.tune
    )
    
    # Save best model
    output_dir = PROJECT_ROOT / args.output
    predictor.save_best_model(output_dir)
    
    # Save results
    results_path = output_dir / 'training_results.json'
    results_json = {
        name: {
            'params': res['params'],
            'train_metrics': res['train_metrics'],
            'test_metrics': res['test_metrics']
        }
        for name, res in results.items()
    }
    
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    logger.info(f"✓ Training complete. Results saved to {results_path}")


if __name__ == '__main__':
    main()
