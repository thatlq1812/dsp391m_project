"""
LSTM model wrapper for unified evaluation framework.

Wraps the TensorFlow-based LSTM model to work with PyTorch-based
evaluation framework.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from traffic_forecast.evaluation.model_wrapper import ModelWrapper
from traffic_forecast.models.lstm_traffic import LSTMTrafficPredictor


class LSTMWrapper(ModelWrapper):
    """
    Wrapper for LSTM baseline model.
    
    This is a temporal-only baseline that:
    - Ignores spatial information (treats each edge independently)
    - No weather integration
    - Simple LSTM architecture
    
    Purpose: Establish performance floor to show value of spatial modeling.
    """
    
    def __init__(
        self,
        sequence_length: int = 12,
        lstm_units: list = None,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Initialize LSTM wrapper.
        
        Args:
            sequence_length: Number of past timesteps (default 12 = 2 hours at 10min)
            lstm_units: LSTM layer sizes (default [128, 64])
            dropout_rate: Dropout for regularization
            learning_rate: Adam learning rate
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units or [128, 64]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        # Initialize model
        self.model = LSTMTrafficPredictor(
            sequence_length=sequence_length,
            lstm_units=self.lstm_units,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )
        
        self._trained = False
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint directory."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.model = LSTMTrafficPredictor.load(checkpoint_path)
        self._trained = True
        
        print(f"[OK] LSTM model loaded from {checkpoint_path}")
    
    def predict(
        self,
        data: pd.DataFrame,
        device: str = 'cuda'
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions on data.
        
        Args:
            data: DataFrame with 'speed_kmh' column and other features
            device: Ignored (TensorFlow handles device placement)
            
        Returns:
            Tuple of (predictions, None) - LSTM is deterministic, no uncertainty
        """
        if not self._trained:
            raise ValueError("Model not trained yet. Call fit() or load_checkpoint() first.")
        
        # Prepare features
        # Handle both 'speed' and 'speed_kmh' column names
        speed_col = 'speed' if 'speed' in data.columns else 'speed_kmh'
        if speed_col not in data.columns:
            raise ValueError(f"Speed column not found. Available: {data.columns.tolist()}")
        
        # For baseline LSTM, we only use speed as feature (temporal only)
        feature_cols = [speed_col]
        
        # Add time features if available
        if 'timestamp' in data.columns:
            if 'hour_sin' not in data.columns:
                data = data.copy()
                data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
                data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek
                data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
                data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)
            feature_cols.extend(['hour_sin', 'hour_cos', 'day_of_week'])
        
        # Handle missing columns
        available_features = [col for col in feature_cols if col in data.columns]
        if not available_features:
            raise ValueError(f"No valid features found in data. Available: {data.columns.tolist()}")
        
        X = data[available_features]
        y_dummy = data[speed_col]  # Needed for sequence creation
        
        # Make predictions
        try:
            predictions = self.model.predict(X)
            
            # LSTM returns predictions for sequences (shorter than input)
            # Pad with NaN to match input length
            n_missing = len(data) - len(predictions)
            predictions_full = np.concatenate([
                np.full(n_missing, np.nan),
                predictions
            ])
            
            return predictions_full, None  # No uncertainty for deterministic model
            
        except Exception as e:
            print(f"[!] Prediction error: {e}")
            # Return zeros if prediction fails
            return np.zeros(len(data)), None
    
    def parameters(self):
        """
        Return model parameters for counting.
        
        Note: TensorFlow model, return trainable parameters count.
        """
        if self.model.model is not None:
            return [p for p in self.model.model.trainable_weights]
        return []
    
    @property
    def model_name(self) -> str:
        """Return human-readable model name."""
        return "LSTM-Baseline"
    
    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1
    ):
        """
        Train the LSTM model.
        
        Args:
            train_data: Training data with 'speed' or 'speed_kmh' column
            val_data: Validation data
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        # Handle both column names (evaluator standardizes to 'speed')
        speed_col = 'speed' if 'speed' in train_data.columns else 'speed_kmh'
        if speed_col not in train_data.columns:
            raise ValueError(f"Speed column not found. Available: {train_data.columns.tolist()}")
        
        # Prepare features
        feature_cols = [speed_col]
        
        # Add temporal features if timestamp is available
        if 'timestamp' in train_data.columns:
            for df in [train_data, val_data]:
                if 'hour_sin' not in df.columns:
                    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
                    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
                    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
                    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            
            feature_cols.extend(['hour_sin', 'hour_cos', 'day_of_week'])
        
        # Extract features and target
        X_train = train_data[feature_cols]
        y_train = train_data[speed_col]
        
        X_val = val_data[feature_cols]
        y_val = val_data[speed_col]
        
        # Train model
        print(f"\n[LSTM Baseline] Training on {len(train_data)} samples...")
        print(f"Features: {feature_cols}")
        print(f"Sequence length: {self.sequence_length}")
        print(f"LSTM units: {self.lstm_units}")
        
        history = self.model.fit(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        self._trained = True
        
        # Print final metrics
        final_val_loss = history.history['val_loss'][-1]
        final_val_mae = history.history['val_mae'][-1]
        
        print(f"\n[LSTM Baseline] Training complete!")
        print(f"Final val_loss: {final_val_loss:.4f}")
        print(f"Final val_mae: {final_val_mae:.4f}")
        
        return history
    
    def save(self, save_dir: Path):
        """Save model to directory."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        self.model.save(save_dir)
        
        # Save config
        import json
        config = {
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'model_name': self.model_name
        }
        
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[OK] LSTM model saved to {save_dir}")
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path) -> 'LSTMWrapper':
        """Load model from checkpoint directory."""
        checkpoint_path = Path(checkpoint_path)
        
        # Load config
        import json
        with open(checkpoint_path / 'config.json', 'r') as f:
            config = json.load(f)
        
        # Create wrapper
        wrapper = cls(
            sequence_length=config['sequence_length'],
            lstm_units=config['lstm_units'],
            dropout_rate=config['dropout_rate'],
            learning_rate=config['learning_rate']
        )
        
        # Load model
        wrapper.load_checkpoint(checkpoint_path)
        
        return wrapper
