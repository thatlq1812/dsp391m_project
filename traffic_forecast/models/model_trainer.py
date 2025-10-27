"""
Maintainer Profile

Name: THAT Le Quang
- Role: AI & DS Major Student
- GitHub: [thatlq1812]
- Primary email: fxlqthat@gmail.com
- Academic email: thatlqse183256@fpt.edu.com
- Alternate email: thatlq1812@gmail.com
- Phone (VN): +84 33 863 6369 / +84 39 730 6450

---

Model training orchestrator with auto hyperparameter tuning and model comparison.
"""

import json
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from traffic_forecast import PROJECT_ROOT

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("XGBoost not available. Install: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger.warning("LightGBM not available. Install: pip install lightgbm")

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False
    logger.warning("MLflow not available. Install: pip install mlflow")


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    model_class: Any
    param_grid: Dict[str, List[Any]]
    search_type: str = "grid"  # grid or random
    n_iter: int = 20  # for random search
    cv_folds: int = 3


@dataclass
class TrainingResult:
    """Results from training a single model."""
    model_name: str
    best_model: Any
    best_params: Dict[str, Any]
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    training_time: float
    feature_importance: Optional[pd.DataFrame] = None


class ModelTrainer:
    """Train and compare multiple models."""
    
    def __init__(
        self,
        output_dir: Path = PROJECT_ROOT / "models" / "experiments",
        experiment_name: str = "traffic_forecast",
        use_mlflow: bool = False
    ):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.experiment_name = experiment_name
        self.use_mlflow = use_mlflow and HAS_MLFLOW
        
        if self.use_mlflow:
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflow experiment: {experiment_name}")
        
        self.results: List[TrainingResult] = []
        self.best_model: Optional[Any] = None
        self.best_model_name: Optional[str] = None
    
    def get_model_configs(self) -> List[ModelConfig]:
        """Get all model configurations."""
        configs = []
        
        # Linear models
        configs.append(ModelConfig(
            name="ridge",
            model_class=Ridge,
            param_grid={
                'alpha': [0.1, 1.0, 10.0, 100.0],
                'solver': ['auto', 'svd', 'lsqr']
            }
        ))
        
        configs.append(ModelConfig(
            name="lasso",
            model_class=Lasso,
            param_grid={
                'alpha': [0.1, 1.0, 10.0, 100.0],
                'max_iter': [1000, 5000]
            }
        ))
        
        configs.append(ModelConfig(
            name="elastic_net",
            model_class=ElasticNet,
            param_grid={
                'alpha': [0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.5, 0.9],
                'max_iter': [1000, 5000]
            }
        ))
        
        # Tree-based models
        configs.append(ModelConfig(
            name="random_forest",
            model_class=RandomForestRegressor,
            param_grid={
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            search_type="random",
            n_iter=20
        ))
        
        configs.append(ModelConfig(
            name="extra_trees",
            model_class=ExtraTreesRegressor,
            param_grid={
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10]
            },
            search_type="random",
            n_iter=15
        ))
        
        configs.append(ModelConfig(
            name="gradient_boosting",
            model_class=GradientBoostingRegressor,
            param_grid={
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            },
            search_type="random",
            n_iter=20
        ))
        
        # XGBoost
        if HAS_XGBOOST:
            configs.append(ModelConfig(
                name="xgboost",
                model_class=xgb.XGBRegressor,
                param_grid={
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                },
                search_type="random",
                n_iter=20
            ))
        
        # LightGBM
        if HAS_LIGHTGBM:
            configs.append(ModelConfig(
                name="lightgbm",
                model_class=lgb.LGBMRegressor,
                param_grid={
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'num_leaves': [31, 63, 127],
                    'subsample': [0.8, 1.0]
                },
                search_type="random",
                n_iter=20
            ))
        
        return configs
    
    def train_single_model(
        self,
        config: ModelConfig,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> TrainingResult:
        """Train a single model with hyperparameter tuning."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Training: {config.name}")
        logger.info(f"{'='*60}")
        
        start_time = datetime.now()
        
        # Start MLflow run
        if self.use_mlflow:
            mlflow.start_run(run_name=config.name)
            mlflow.log_param("model_type", config.name)
        
        try:
            # Hyperparameter search
            base_model = config.model_class(random_state=42)
            
            if config.search_type == "grid":
                search = GridSearchCV(
                    base_model,
                    config.param_grid,
                    cv=config.cv_folds,
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1,
                    verbose=1
                )
            else:  # random
                search = RandomizedSearchCV(
                    base_model,
                    config.param_grid,
                    n_iter=config.n_iter,
                    cv=config.cv_folds,
                    scoring='neg_mean_absolute_error',
                    n_jobs=-1,
                    random_state=42,
                    verbose=1
                )
            
            # Fit
            logger.info("Running hyperparameter search...")
            search.fit(X_train, y_train)
            
            best_model = search.best_estimator_
            best_params = search.best_params_
            
            logger.info(f"Best params: {best_params}")
            
            # Evaluate on all sets
            train_metrics = self._evaluate(best_model, X_train, y_train, "Train")
            val_metrics = self._evaluate(best_model, X_val, y_val, "Val")
            test_metrics = self._evaluate(best_model, X_test, y_test, "Test")
            
            # Feature importance
            feature_importance = None
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=False)
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Log to MLflow
            if self.use_mlflow:
                mlflow.log_params(best_params)
                mlflow.log_metrics({
                    'train_mae': train_metrics['mae'],
                    'train_rmse': train_metrics['rmse'],
                    'train_r2': train_metrics['r2'],
                    'val_mae': val_metrics['mae'],
                    'val_rmse': val_metrics['rmse'],
                    'val_r2': val_metrics['r2'],
                    'test_mae': test_metrics['mae'],
                    'test_rmse': test_metrics['rmse'],
                    'test_r2': test_metrics['r2'],
                    'training_time': training_time
                })
                mlflow.sklearn.log_model(best_model, config.name)
            
            result = TrainingResult(
                model_name=config.name,
                best_model=best_model,
                best_params=best_params,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                training_time=training_time,
                feature_importance=feature_importance
            )
            
            return result
        
        finally:
            if self.use_mlflow:
                mlflow.end_run()
    
    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_names: Optional[List[str]] = None
    ) -> List[TrainingResult]:
        """Train all configured models."""
        logger.info("\n" + "="*80)
        logger.info("STARTING MODEL TRAINING")
        logger.info("="*80)
        
        configs = self.get_model_configs()
        
        # Filter by model names if specified
        if model_names:
            configs = [c for c in configs if c.name in model_names]
        
        logger.info(f"Training {len(configs)} models: {[c.name for c in configs]}")
        
        results = []
        for config in configs:
            try:
                result = self.train_single_model(
                    config, X_train, y_train, X_val, y_val, X_test, y_test
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error training {config.name}: {e}")
                continue
        
        self.results = results
        
        # Find best model
        if results:
            best_result = min(results, key=lambda r: r.val_metrics['mae'])
            self.best_model = best_result.best_model
            self.best_model_name = best_result.model_name
            
            logger.info(f"\nBest model: {self.best_model_name}")
            logger.info(f"Val MAE: {best_result.val_metrics['mae']:.2f}")
        
        return results
    
    def compare_models(self) -> pd.DataFrame:
        """Create comparison table of all models."""
        if not self.results:
            logger.warning("No results to compare")
            return pd.DataFrame()
        
        comparison = []
        for result in self.results:
            comparison.append({
                'model': result.model_name,
                'train_mae': result.train_metrics['mae'],
                'train_rmse': result.train_metrics['rmse'],
                'train_r2': result.train_metrics['r2'],
                'val_mae': result.val_metrics['mae'],
                'val_rmse': result.val_metrics['rmse'],
                'val_r2': result.val_metrics['r2'],
                'test_mae': result.test_metrics['mae'],
                'test_rmse': result.test_metrics['rmse'],
                'test_r2': result.test_metrics['r2'],
                'training_time': result.training_time
            })
        
        df = pd.DataFrame(comparison)
        df = df.sort_values('val_mae')
        
        logger.info("\n" + "="*80)
        logger.info("MODEL COMPARISON (sorted by validation MAE)")
        logger.info("="*80)
        print(df.to_string(index=False))
        
        return df
    
    def save_results(self, timestamp: Optional[str] = None):
        """Save all training results."""
        if not self.results:
            logger.warning("No results to save")
            return
        
        timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save comparison table
        df_comparison = self.compare_models()
        comparison_path = self.output_dir / f"comparison_{timestamp}.csv"
        df_comparison.to_csv(comparison_path, index=False)
        logger.info(f"Saved comparison: {comparison_path}")
        
        # Save best model
        if self.best_model:
            model_path = self.output_dir / f"best_model_{timestamp}.pkl"
            joblib.dump(self.best_model, model_path)
            logger.info(f"Saved best model ({self.best_model_name}): {model_path}")
            
            # Save model metadata
            best_result = next(r for r in self.results if r.model_name == self.best_model_name)
            metadata = {
                'timestamp': timestamp,
                'model_name': self.best_model_name,
                'best_params': best_result.best_params,
                'metrics': {
                    'train': best_result.train_metrics,
                    'val': best_result.val_metrics,
                    'test': best_result.test_metrics
                },
                'training_time': best_result.training_time
            }
            
            metadata_path = self.output_dir / f"best_model_metadata_{timestamp}.json"
            with metadata_path.open('w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            # Save feature importance if available
            if best_result.feature_importance is not None:
                fi_path = self.output_dir / f"feature_importance_{timestamp}.csv"
                best_result.feature_importance.to_csv(fi_path, index=False)
                logger.info(f"Saved feature importance: {fi_path}")
    
    def _evaluate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str
    ) -> Dict[str, float]:
        """Evaluate model and return metrics."""
        y_pred = model.predict(X)
        
        metrics = {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'r2': r2_score(y, y_pred),
            'mape': mean_absolute_percentage_error(y, y_pred) * 100
        }
        
        logger.info(f"{dataset_name} - MAE: {metrics['mae']:.2f}, "
                   f"RMSE: {metrics['rmse']:.2f}, "
                   f"R2: {metrics['r2']:.4f}, "
                   f"MAPE: {metrics['mape']:.2f}%")
        
        return metrics


def load_latest_data(data_dir: Path = PROJECT_ROOT / "data" / "ml_ready") -> Tuple:
    """Load the most recent train/val/test datasets."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Find latest datasets
    train_files = sorted(data_dir.glob("train_*.parquet"))
    val_files = sorted(data_dir.glob("val_*.parquet"))
    test_files = sorted(data_dir.glob("test_*.parquet"))
    
    if not train_files:
        raise FileNotFoundError("No training data found. Run ML pipeline first.")
    
    train_path = train_files[-1]
    val_path = val_files[-1]
    test_path = test_files[-1]
    
    logger.info(f"Loading train data: {train_path.name}")
    logger.info(f"Loading val data: {val_path.name}")
    logger.info(f"Loading test data: {test_path.name}")
    
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)
    df_test = pd.read_parquet(test_path)
    
    # Separate features and target
    target_col = 'duration_in_traffic_s'
    
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    
    X_val = df_val.drop(columns=[target_col])
    y_val = df_val[target_col]
    
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_models_cli():
    """CLI entry point for model training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train traffic forecast models')
    parser.add_argument('--models', nargs='+', help='Specific models to train')
    parser.add_argument('--use-mlflow', action='store_true', help='Enable MLflow tracking')
    args = parser.parse_args()
    
    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_latest_data()
    
    # Train models
    trainer = ModelTrainer(use_mlflow=args.use_mlflow)
    results = trainer.train_all_models(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        model_names=args.models
    )
    
    # Save results
    trainer.save_results()
    
    logger.info("\n" + "="*80)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*80)


if __name__ == "__main__":
    train_models_cli()
