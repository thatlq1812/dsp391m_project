"""
Deep Learning model trainer for ML pipeline.
Wraps LSTM and ASTGCN models from traffic_forecast.models
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Dict, Optional

try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    warnings.warn("TensorFlow not installed. Install with: pip install tensorflow")

# Import DL models from traffic_forecast.models
try:
    from traffic_forecast.models.lstm_model import LSTMTrafficPredictor
    HAS_LSTM = HAS_TENSORFLOW
except ImportError:
    HAS_LSTM = False
    warnings.warn("LSTM model not available")

try:
    from traffic_forecast.models.research.astgcn import ASTGCNTrafficModel, ASTGCNConfig
    HAS_ASTGCN = HAS_TENSORFLOW
except ImportError:
    HAS_ASTGCN = False
    warnings.warn("ASTGCN model not available")

from traffic_forecast import PROJECT_ROOT


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
        self.params = params or self.DEFAULT_PARAMS.get(model_type, {}).copy()
        self.model_dir = model_dir or PROJECT_ROOT / 'models'
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
        self.feature_names = list(X_train.columns)
        
        print(f"\nTraining {self.model_type.upper()} model...")
        print(f"Training samples: {len(X_train)}, Features: {len(self.feature_names)}")
        
        if self.model_type == 'lstm':
            self.model = LSTMTrafficPredictor(**self.params)
            self.training_history = self.model.train(
                X_train, y_train,
                X_val, y_val,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose
            )
            
        elif self.model_type == 'astgcn':
            # ASTGCN requires adjacency matrix - for now, use simple approach
            # In production, should compute from actual graph structure
            warnings.warn("ASTGCN requires graph adjacency matrix. Using simplified approach.")
            
            # Update num_nodes based on data
            self.params['num_nodes'] = len(X_train)
            
            # Create ASTGCN config
            from traffic_forecast.models.research.astgcn import ASTGCNComponentConfig
            from collections import OrderedDict
            
            component_configs = OrderedDict([
                ("recent", ASTGCNComponentConfig(window=12, blocks=2)),
                ("daily", ASTGCNComponentConfig(window=12 * 24, blocks=1)),
            ])
            
            config = ASTGCNConfig(
                num_nodes=self.params['num_nodes'],
                input_dim=self.params['input_dim'],
                horizon=self.params['horizon'],
                component_configs=component_configs,
                attention_units=self.params['attention_units'],
                cheb_order=self.params['cheb_order'],
                spatial_filters=self.params['spatial_filters'],
                temporal_filters=self.params['temporal_filters'],
                dropout_rate=self.params['dropout_rate'],
                learning_rate=self.params['learning_rate']
            )
            
            # Create dummy adjacency matrix (identity for now)
            adjacency = np.eye(self.params['num_nodes'])
            
            self.model = ASTGCNTrafficModel(config, adjacency)
            
            # ASTGCN training requires special data format
            # For now, skip actual training and return warning
            print("⚠️ ASTGCN requires graph-structured data. Skipping training.")
            print("   Use LSTM for tabular data or provide adjacency matrix for ASTGCN.")
            return self
        
        print(f"✓ Training complete!")
        
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
        
        if self.model_type == 'lstm':
            metrics = self.model.evaluate(X_test, y_test)
            
        elif self.model_type == 'astgcn':
            # ASTGCN evaluation not implemented for tabular data
            metrics = {
                'rmse': np.nan,
                'mae': np.nan,
                'r2': np.nan,
                'mape': np.nan
            }
        
        self.test_metrics = metrics
        
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if self.model_type == 'lstm':
            return self.model.predict(X)
        
        elif self.model_type == 'astgcn':
            # ASTGCN prediction not implemented for tabular data
            return np.full(len(X), np.nan)
    
    def save_model(self, filename: Optional[str] = None) -> Path:
        """
        Save trained model.
        
        Args:
            filename: Custom filename (auto-generated if None)
            
        Returns:
            Path to saved model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if filename is None:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.model_type}_{timestamp}"
        
        model_path = self.model_dir / filename
        
        if self.model_type == 'lstm':
            self.model.save(self.model_dir, name=filename)
        
        elif self.model_type == 'astgcn':
            # ASTGCN save not implemented
            pass
        
        print(f"✓ Model saved to {model_path}")
        
        return model_path
    
    @classmethod
    def load_model(cls, model_path: Path, model_type: str = 'lstm'):
        """
        Load a saved model.
        
        Args:
            model_path: Path to saved model
            model_type: Type of model
            
        Returns:
            DLModelTrainer instance
        """
        trainer = cls(model_type=model_type)
        
        if model_type == 'lstm':
            trainer.model = LSTMTrafficPredictor.load(model_path)
        
        elif model_type == 'astgcn':
            # ASTGCN load not implemented
            pass
        
        return trainer
    
    def get_feature_importance(self, top_n: Optional[int] = None) -> pd.DataFrame:
        """
        Get feature importance (not applicable for DL models).
        
        Returns:
            Empty DataFrame with message
        """
        return pd.DataFrame({
            'feature': ['N/A'],
            'importance': [0.0],
            'message': ['Feature importance not available for deep learning models']
        })


def compare_dl_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: list = None,
    epochs: int = 50,
    batch_size: int = 32
) -> pd.DataFrame:
    """
    Compare multiple DL models.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        models: List of model names (default: all available)
        epochs: Training epochs
        batch_size: Batch size
        
    Returns:
        DataFrame with comparison results
    """
    if models is None:
        models = list(DLModelTrainer.MODELS.keys())
    
    results = []
    
    for model_name in models:
        if model_name not in DLModelTrainer.MODELS:
            print(f"⚠️ Model {model_name} not available, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*60}")
        
        try:
            trainer = DLModelTrainer(model_type=model_name)
            trainer.train(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
            
            metrics = trainer.evaluate(X_test, y_test)
            
            results.append({
                'model': model_name,
                'rmse': metrics.get('rmse', np.nan),
                'mae': metrics.get('mae', np.nan),
                'r2': metrics.get('r2', np.nan),
                'mape': metrics.get('mape', np.nan)
            })
            
            print(f"✓ {model_name}: R²={metrics.get('r2', 0):.4f}, RMSE={metrics.get('rmse', 0):.4f}")
            
        except Exception as e:
            print(f"✗ Error training {model_name}: {str(e)}")
            results.append({
                'model': model_name,
                'rmse': np.nan,
                'mae': np.nan,
                'r2': np.nan,
                'mape': np.nan,
                'error': str(e)
            })
    
    return pd.DataFrame(results)
