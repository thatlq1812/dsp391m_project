"""Production-ready Deep Learning model trainer for traffic forecasting.

This module provides a unified interface for training and deploying 
LSTM and ASTGCN models for traffic speed prediction.
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    warnings.warn("TensorFlow not installed. Install with: pip install tensorflow")

# Import DL models from traffic_forecast.models
try:
    from traffic_forecast.models.lstm_traffic import LSTMTrafficPredictor
    HAS_LSTM = HAS_TENSORFLOW
except (ImportError, IndentationError, SyntaxError) as e:
    HAS_LSTM = False
    warnings.warn(f"LSTM model not available: {e}")

try:
    from traffic_forecast.models.graph import ASTGCNConfig, ASTGCNTrafficModel
    HAS_ASTGCN = HAS_TENSORFLOW
except (ImportError, IndentationError, SyntaxError) as e:
    HAS_ASTGCN = False
    warnings.warn(f"ASTGCN model not available: {e}")

from traffic_forecast import PROJECT_ROOT

logger = logging.getLogger(__name__)


class DLModelTrainer:
    """
    Deep Learning model trainer compatible with ML pipeline.
    Wraps LSTM and ASTGCN models for seamless integration.
    """

    MODELS = {}

    if HAS_LSTM:
        MODELS['lstm'] = LSTMTrafficPredictor

    if HAS_ASTGCN:
        MODELS['astgcn'] = ASTGCNTrafficModel

    DEFAULT_PARAMS = {
        'lstm': {
            'sequence_length': 12,  # 12 timesteps
            'forecast_horizons': [5, 15, 30, 60],
            'lstm_units': [128, 64],
            'dropout_rate': 0.2,
            'learning_rate': 0.001
        },
        'astgcn': {
            'num_nodes': 64,  # Will be updated based on data
            'input_dim': 1,
            'horizon': 12,
            'attention_units': 32,
            'cheb_order': 3,
            'spatial_filters': 64,
            'temporal_filters': 64,
            'dropout_rate': 0.3,
            'learning_rate': 0.001
        }
    }

    def __init__(
        self,
        model_type: str = 'lstm',
        params: Optional[Dict] = None,
        model_dir: Optional[Path] = None
    ):
        """
        Initialize DL model trainer.

        Args:
            model_type: Type of DL model ('lstm' or 'astgcn')
            params: Model hyperparameters
            model_dir: Directory to save models
        """
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required for deep learning models. Install with: pip install tensorflow")

        if model_type not in self.MODELS:
            available = ', '.join(self.MODELS.keys())
            raise ValueError(f"Unknown DL model type: {model_type}. Available: {available}")

        self.model_type = model_type
        
        # Get default params and merge with provided params
        default_params = self.DEFAULT_PARAMS.get(model_type, {}).copy()
        if params:
            # Remove training params and config-specific params that are not model params
            non_model_params = {
                'name', 'enabled', 'epochs', 'batch_size', 'validation_split',
                'num_of_vertices', 'num_of_features', 'points_per_hour', 'num_for_predict',
                'early_stopping_patience', 'reduce_lr_patience'
            }
            model_params = {k: v for k, v in params.items() if k not in non_model_params}
            
            # Map config params to model params
            if model_type == 'lstm' and 'hidden_units' in model_params:
                model_params['lstm_units'] = model_params.pop('hidden_units')
            
            default_params.update(model_params)
        
        self.params = default_params
        self.model_dir = model_dir or Path('models/saved')
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.feature_names = None
        self.training_history = None
        self.test_metrics = {}

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1
    ):
        """
        Train the deep learning model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
        """
        # Handle both DataFrame and ndarray
        if hasattr(X_train, 'columns'):
            self.feature_names = list(X_train.columns)
        else:
            # If it's a numpy array, generate generic feature names
            n_features = X_train.shape[1] if len(X_train.shape) > 1 else 1
            self.feature_names = [f'feature_{i}' for i in range(n_features)]

        logger.info(f"Training {self.model_type.upper()} model...")
        logger.info(f"Training samples: {len(X_train)}, Features: {len(self.feature_names)}")

        if self.model_type == 'lstm':
            # LSTM expects tabular data (2D) - it creates sequences internally
            self.model = LSTMTrafficPredictor(**self.params)
            self.training_history = self.model.fit(
                X_train, y_train,
                X_val, y_val,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose
            )
            logger.info("LSTM training complete!")

        elif self.model_type == 'astgcn':
            # ASTGCN requires graph structure and adjacency matrix
            if 'adjacency' not in self.params or self.params['adjacency'] is None:
                # Create a simple identity matrix as default adjacency
                # In production, this should be built from actual road network topology
                import numpy as np
                num_nodes = X_train.shape[1] if len(X_train.shape) > 2 else 1
                logger.warning(f"No adjacency matrix provided. Creating identity matrix with {num_nodes} nodes.")
                logger.warning("For better ASTGCN performance, provide real road network adjacency matrix.")
                self.params['adjacency'] = np.eye(num_nodes)

            # Import component config
            from traffic_forecast.models.graph.astgcn_traffic import ASTGCNComponentConfig
            from collections import OrderedDict

            # Create temporal component configurations
            component_configs = OrderedDict([
                ("recent", ASTGCNComponentConfig(
                    window=self.params.get('recent_window', 12), 
                    blocks=self.params.get('recent_blocks', 2)
                )),
                ("daily", ASTGCNComponentConfig(
                    window=self.params.get('daily_window', 12 * 24), 
                    blocks=self.params.get('daily_blocks', 1)
                )),
            ])

            # Create ASTGCN configuration
            config = ASTGCNConfig(
                num_nodes=self.params['num_nodes'],
                input_dim=self.params.get('input_dim', 1),
                horizon=self.params.get('horizon', 12),
                component_configs=component_configs,
                attention_units=self.params.get('attention_units', 32),
                cheb_order=self.params.get('cheb_order', 3),
                spatial_filters=self.params.get('spatial_filters', 64),
                temporal_filters=self.params.get('temporal_filters', 64),
                dropout_rate=self.params.get('dropout_rate', 0.3),
                learning_rate=self.params.get('learning_rate', 0.001)
            )

            # Create model
            self.model = ASTGCNTrafficModel(self.params['adjacency'], config)
            
            # Prepare graph-structured data
            # Note: X_train should already be formatted for ASTGCN
            # Expected format: dict with keys matching component names
            self.training_history = self.model.fit(
                X_train, y_train,
                X_val, y_val,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose
            )
            logger.info("ASTGCN training complete!")

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
            X_test: Test features (tabular for LSTM, graph-structured for ASTGCN)
            y_test: Test target
            set_name: Name of evaluation set

        Returns:
            Dictionary of metrics (mae, rmse, r2, mape)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        logger.info(f"Evaluating {self.model_type.upper()} on {set_name} set...")

        metrics = self.model.evaluate(X_test, y_test)
        self.test_metrics = metrics

        logger.info(f"{set_name.capitalize()} metrics:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name.upper()}: {value:.4f}")

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features (tabular for LSTM, graph-structured for ASTGCN)

        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict(X)

    def save_model(self, filename: Optional[str] = None) -> Path:
        """
        Save trained model to disk.

        Args:
            filename: Custom filename (auto-generated if None)

        Returns:
            Path to saved model directory
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.model_type}_{timestamp}"

        model_path = self.model_dir / filename
        model_path.mkdir(parents=True, exist_ok=True)

        # Save model using its native save method
        self.model.save(model_path)

        # Save additional metadata
        import json
        metadata = {
            'model_type': self.model_type,
            'params': self.params if 'adjacency' not in self.params else {
                k: v for k, v in self.params.items() if k != 'adjacency'
            },
            'feature_names': self.feature_names,
            'test_metrics': self.test_metrics,
            'timestamp': timestamp if filename is None else filename
        }
        
        with open(model_path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to {model_path}")

        return model_path

    @classmethod
    def load_model(cls, model_path: Path, model_type: str = None):
        """
        Load a saved model from disk.

        Args:
            model_path: Path to saved model directory
            model_type: Type of model ('lstm' or 'astgcn'). Auto-detected if None.

        Returns:
            DLModelTrainer instance with loaded model
        """
        model_path = Path(model_path)
        
        # Try to load metadata
        metadata_path = model_path / 'metadata.json'
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            if model_type is None:
                model_type = metadata.get('model_type', 'lstm')
        
        if model_type is None:
            model_type = 'lstm'  # Default

        trainer = cls(model_type=model_type)

        if model_type == 'lstm':
            trainer.model = LSTMTrafficPredictor.load(model_path)
        elif model_type == 'astgcn':
            trainer.model = ASTGCNTrafficModel.load(model_path)

        logger.info(f"Loaded {model_type.upper()} model from {model_path}")

        # Restore metadata if available
        if metadata_path.exists():
            trainer.feature_names = metadata.get('feature_names')
            trainer.test_metrics = metadata.get('test_metrics', {})

        return trainer

    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importance for deep learning models.
        
        Note: Deep learning models don't have traditional feature importance.
        This method returns attention weights for LSTM or empty for ASTGCN.

        Returns:
            DataFrame with feature importance information
        """
        logger.warning("Feature importance is not directly available for deep learning models.")
        logger.info("Consider using SHAP values or integrated gradients for interpretation.")
        
        return pd.DataFrame({
            'feature': ['N/A'],
            'importance': [0.0],
            'note': ['Use SHAP or integrated gradients for DL model interpretation']
        })


def compare_dl_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    models: list = None,
    epochs: int = 50,
    batch_size: int = 32,
    verbose: int = 0
) -> pd.DataFrame:
    """
    Compare multiple deep learning models on the same dataset.

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        X_val: Validation features (optional)
        y_val: Validation target (optional)
        models: List of model names (default: all available)
        epochs: Training epochs
        batch_size: Batch size
        verbose: Verbosity level for training

    Returns:
        DataFrame with comparison results sorted by R² score
    """
    if models is None:
        models = list(DLModelTrainer.MODELS.keys())

    results = []

    for model_name in models:
        if model_name not in DLModelTrainer.MODELS:
            logger.warning(f"Model {model_name} not available, skipping...")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_name.upper()}")
        logger.info(f"{'='*60}")

        try:
            trainer = DLModelTrainer(model_type=model_name)
            trainer.train(
                X_train, y_train, 
                X_val, y_val,
                epochs=epochs, 
                batch_size=batch_size, 
                verbose=verbose
            )

            metrics = trainer.evaluate(X_test, y_test)

            results.append({
                'model': model_name,
                'mae': metrics.get('mae', np.nan),
                'rmse': metrics.get('rmse', np.nan),
                'r2': metrics.get('r2', np.nan),
                'mape': metrics.get('mape', np.nan),
                'status': 'success'
            })

            logger.info(
                f"{model_name}: R²={metrics.get('r2', 0):.4f}, "
                f"RMSE={metrics.get('rmse', 0):.4f}, "
                f"MAE={metrics.get('mae', 0):.4f}"
            )

        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            results.append({
                'model': model_name,
                'mae': np.nan,
                'rmse': np.nan,
                'r2': np.nan,
                'mape': np.nan,
                'status': f'error: {str(e)}'
            })

    df_results = pd.DataFrame(results)
    
    # Sort by R² score (descending)
    if 'r2' in df_results.columns and not df_results['r2'].isna().all():
        df_results = df_results.sort_values('r2', ascending=False)
    
    return df_results

    return pd.DataFrame(results)
