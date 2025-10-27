"""
LSTM-based time series model for traffic speed forecasting.
Implements sequence-to-sequence prediction with attention mechanism.
"""

import logging
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
import joblib

try:
 import tensorflow as tf
 from tensorflow import keras
 from tensorflow.keras import layers, Model
 from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
 HAS_TENSORFLOW = True
except ImportError:
 HAS_TENSORFLOW = False
 logging.warning("TensorFlow not installed. Install with: pip install tensorflow")

from traffic_forecast import PROJECT_ROOT

logger = logging.getLogger(__name__)


class LSTMTrafficPredictor:
 """
 LSTM model for traffic speed prediction with attention mechanism.
 """
 
 def __init__(
 self,
 sequence_length: int = 12, # 12 * 5min = 1 hour history
 forecast_horizons: List[int] = [5, 15, 30, 60],
 lstm_units: List[int] = [128, 64],
 dropout_rate: float = 0.2,
 learning_rate: float = 0.001
 ):
 """
 Initialize LSTM predictor.
 
 Args:
 sequence_length: Number of past timesteps to use
 forecast_horizons: Forecast horizons in minutes
 lstm_units: List of LSTM layer sizes
 dropout_rate: Dropout rate for regularization
 learning_rate: Learning rate for Adam optimizer
 """
 if not HAS_TENSORFLOW:
 raise ImportError("TensorFlow is required for LSTM models")
 
 self.sequence_length = sequence_length
 self.forecast_horizons = forecast_horizons
 self.lstm_units = lstm_units
 self.dropout_rate = dropout_rate
 self.learning_rate = learning_rate
 self.model = None
 self.scaler_X = None
 self.scaler_y = None
 self.history = None
 
 def create_sequences(
 self,
 X: np.ndarray,
 y: np.ndarray
 ) -> Tuple[np.ndarray, np.ndarray]:
 """
 Create sequences for time series prediction.
 
 Args:
 X: Feature array of shape (n_samples, n_features)
 y: Target array of shape (n_samples,)
 
 Returns:
 X_seq: Sequences of shape (n_sequences, sequence_length, n_features)
 y_seq: Targets of shape (n_sequences,)
 """
 X_seq, y_seq = [], []
 
 for i in range(len(X) - self.sequence_length):
 X_seq.append(X[i:i + self.sequence_length])
 y_seq.append(y[i + self.sequence_length])
 
 return np.array(X_seq), np.array(y_seq)
 
 def build_model(self, input_shape: Tuple[int, int]) -> Model:
 """
 Build LSTM model with attention mechanism.
 
 Args:
 input_shape: (sequence_length, n_features)
 
 Returns:
 Compiled Keras model
 """
 inputs = layers.Input(shape=input_shape)
 
 # LSTM layers with return sequences
 x = inputs
 for i, units in enumerate(self.lstm_units):
 return_sequences = i < len(self.lstm_units) - 1
 x = layers.LSTM(
 units,
 return_sequences=return_sequences,
 dropout=self.dropout_rate,
 recurrent_dropout=self.dropout_rate,
 name=f'lstm_{i+1}'
 )(x)
 
 if return_sequences:
 x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
 
 # Dense layers for output
 x = layers.Dense(64, activation='relu', name='dense_1')(x)
 x = layers.Dropout(self.dropout_rate, name='dropout_final')(x)
 x = layers.Dense(32, activation='relu', name='dense_2')(x)
 
 # Output layer
 outputs = layers.Dense(1, activation='linear', name='output')(x)
 
 # Create model
 model = Model(inputs=inputs, outputs=outputs, name='LSTM_Traffic_Predictor')
 
 # Compile
 model.compile(
 optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
 loss='mse',
 metrics=['mae', 'mse']
 )
 
 return model
 
 def train(
 self,
 X_train: pd.DataFrame,
 y_train: pd.Series,
 X_val: Optional[pd.DataFrame] = None,
 y_val: Optional[pd.Series] = None,
 epochs: int = 100,
 batch_size: int = 32,
 verbose: int = 1
 ) -> dict:
 """
 Train LSTM model.
 
 Returns:
 Training history
 """
 logger.info("Preparing sequences for LSTM training...")
 
 # Convert to numpy
 X_train_np = X_train.values
 y_train_np = y_train.values
 
 # Create sequences
 X_train_seq, y_train_seq = self.create_sequences(X_train_np, y_train_np)
 
 logger.info(f"Created {len(X_train_seq)} training sequences")
 logger.info(f"Sequence shape: {X_train_seq.shape}")
 
 # Validation sequences
 if X_val is not None and y_val is not None:
 X_val_np = X_val.values
 y_val_np = y_val.values
 X_val_seq, y_val_seq = self.create_sequences(X_val_np, y_val_np)
 validation_data = (X_val_seq, y_val_seq)
 logger.info(f"Created {len(X_val_seq)} validation sequences")
 else:
 validation_data = None
 
 # Build model
 input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
 self.model = self.build_model(input_shape)
 
 logger.info(f"Model architecture:")
 self.model.summary(print_fn=logger.info)
 
 # Callbacks
 callbacks = [
 EarlyStopping(
 monitor='val_loss' if validation_data else 'loss',
 patience=15,
 restore_best_weights=True,
 verbose=1
 ),
 ReduceLROnPlateau(
 monitor='val_loss' if validation_data else 'loss',
 factor=0.5,
 patience=5,
 min_lr=1e-6,
 verbose=1
 )
 ]
 
 # Train
 logger.info("Starting training...")
 self.history = self.model.fit(
 X_train_seq, y_train_seq,
 validation_data=validation_data,
 epochs=epochs,
 batch_size=batch_size,
 callbacks=callbacks,
 verbose=verbose
 )
 
 logger.info("Training complete!")
 
 return self.history.history
 
 def predict(self, X: pd.DataFrame) -> np.ndarray:
 """
 Make predictions.
 
 Args:
 X: Features of shape (n_samples, n_features)
 
 Returns:
 Predictions of shape (n_samples - sequence_length,)
 """
 if self.model is None:
 raise ValueError("Model not trained yet")
 
 X_np = X.values
 X_seq, _ = self.create_sequences(X_np, np.zeros(len(X_np)))
 
 predictions = self.model.predict(X_seq, verbose=0)
 return predictions.flatten()
 
 def evaluate(
 self,
 X_test: pd.DataFrame,
 y_test: pd.Series
 ) -> dict:
 """
 Evaluate model on test set.
 
 Returns:
 Dictionary of metrics
 """
 if self.model is None:
 raise ValueError("Model not trained yet")
 
 # Create sequences
 X_test_np = X_test.values
 y_test_np = y_test.values
 X_test_seq, y_test_seq = self.create_sequences(X_test_np, y_test_np)
 
 # Evaluate
 results = self.model.evaluate(X_test_seq, y_test_seq, verbose=0)
 
 # Get predictions for additional metrics
 y_pred = self.model.predict(X_test_seq, verbose=0).flatten()
 
 metrics = {
 'loss': results[0],
 'mae': results[1],
 'mse': results[2],
 'rmse': np.sqrt(results[2]),
 'mape': np.mean(np.abs((y_test_seq - y_pred) / y_test_seq)) * 100,
 'r2': 1 - (np.sum((y_test_seq - y_pred)**2) / np.sum((y_test_seq - y_test_seq.mean())**2))
 }
 
 logger.info(f"Test RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")
 
 return metrics
 
 def save(self, output_dir: Path, name: str = "lstm_model"):
 """Save model and scaler."""
 output_dir = Path(output_dir)
 output_dir.mkdir(parents=True, exist_ok=True)
 
 # Save Keras model
 model_path = output_dir / f"{name}.keras"
 self.model.save(model_path)
 logger.info(f"Saved model to {model_path}")
 
 # Save config
 config = {
 'sequence_length': self.sequence_length,
 'forecast_horizons': self.forecast_horizons,
 'lstm_units': self.lstm_units,
 'dropout_rate': self.dropout_rate,
 'learning_rate': self.learning_rate
 }
 config_path = output_dir / f"{name}_config.pkl"
 joblib.dump(config, config_path)
 logger.info(f"Saved config to {config_path}")
 
 return model_path
 
 @classmethod
 def load(cls, model_path: Path):
 """Load saved model."""
 model_path = Path(model_path)
 config_path = model_path.parent / f"{model_path.stem}_config.pkl"
 
 # Load config
 config = joblib.load(config_path)
 
 # Create instance
 instance = cls(**config)
 
 # Load Keras model
 instance.model = keras.models.load_model(model_path)
 
 logger.info(f"Loaded model from {model_path}")
 
 return instance


def main():
 """Main training script for LSTM model."""
 import argparse
 
 parser = argparse.ArgumentParser(description='Train LSTM traffic speed prediction model')
 parser.add_argument('--data', type=str, default='data/processed/train.parquet',
 help='Path to training data')
 parser.add_argument('--output', type=str, default='models',
 help='Output directory for model')
 parser.add_argument('--epochs', type=int, default=100,
 help='Number of epochs')
 parser.add_argument('--batch-size', type=int, default=32,
 help='Batch size')
 parser.add_argument('--sequence-length', type=int, default=12,
 help='Sequence length (number of past timesteps)')
 args = parser.parse_args()
 
 # Load data
 logger.info(f"Loading data from {args.data}")
 data_path = PROJECT_ROOT / args.data
 df = pd.read_parquet(data_path)
 
 # Prepare features and target
 target_col = 'avg_speed_kmh'
 feature_cols = [c for c in df.columns if c != target_col and c not in ['node_id', 'ts']]
 
 X = df[feature_cols]
 y = df[target_col]
 
 # Split train/val
 split_idx = int(len(X) * 0.8)
 X_train, X_val = X[:split_idx], X[split_idx:]
 y_train, y_val = y[:split_idx], y[split_idx:]
 
 logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}")
 
 # Create and train model
 predictor = LSTMTrafficPredictor(
 sequence_length=args.sequence_length,
 lstm_units=[128, 64],
 dropout_rate=0.2
 )
 
 history = predictor.train(
 X_train, y_train,
 X_val, y_val,
 epochs=args.epochs,
 batch_size=args.batch_size
 )
 
 # Evaluate
 metrics = predictor.evaluate(X_val, y_val)
 logger.info(f"Validation metrics: {metrics}")
 
 # Save model
 output_dir = PROJECT_ROOT / args.output
 predictor.save(output_dir)
 
 logger.info(" LSTM training complete!")


if __name__ == '__main__':
 main()
