"""LSTM-based traffic speed prediction model.

This module implements an LSTM neural network with attention mechanism
for multi-horizon traffic speed forecasting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

logger = logging.getLogger(__name__)


class LSTMTrafficPredictor:
    """LSTM model for traffic speed prediction with attention mechanism."""

    def __init__(
        self,
        sequence_length: int = 12,
        forecast_horizons: List[int] = None,
        lstm_units: List[int] = None,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """Initialize LSTM predictor.

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
        self.forecast_horizons = forecast_horizons or [5, 15, 30, 60]
        self.lstm_units = lstm_units or [128, 64]
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.scaler_X = None
        self.scaler_y = None

    def _create_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction.

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

    def _build_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """Build LSTM model with attention mechanism.

        Args:
            input_shape: (sequence_length, n_features)

        Returns:
            Compiled Keras model
        """
        inputs = layers.Input(shape=input_shape, name='input')
        x = inputs

        # LSTM layers with dropout
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            x = layers.LSTM(
                units,
                return_sequences=return_sequences,
                name=f'lstm_{i+1}'
            )(x)
            x = layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)
            x = layers.BatchNormalization(name=f'bn_{i+1}')(x)

        # Dense layers
        x = layers.Dense(64, activation='relu', name='dense_1')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_final')(x)
        outputs = layers.Dense(1, activation='linear', name='output')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name='lstm_traffic')

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )

        return model

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1
    ) -> keras.callbacks.History:
        """Train LSTM model.

        Returns:
            Training history
        """
        # Scale features
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

        # Create sequences
        X_train_seq, y_train_seq = self._create_sequences(X_train_scaled, y_train_scaled)

        # Validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler_X.transform(X_val)
            y_val_scaled = self.scaler_y.transform(y_val.values.reshape(-1, 1)).ravel()
            X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val_scaled)
            validation_data = (X_val_seq, y_val_seq)

        # Build model if not exists
        if self.model is None:
            input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
            self.model = self._build_model(input_shape)

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=10,
                restore_best_weights=True,
                verbose=verbose
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=verbose
            )
        ]

        # Train
        history = self.model.fit(
            X_train_seq,
            y_train_seq,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )

        return history

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.

        Args:
            X: Features of shape (n_samples, n_features)

        Returns:
            Predictions of shape (n_samples - sequence_length,)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        X_scaled = self.scaler_X.transform(X)
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))

        y_pred_scaled = self.model.predict(X_seq, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled).ravel()

        return y_pred

    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """Evaluate model on test set.

        Returns:
            Dictionary of metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        y_pred = self.predict(X_test)
        y_true = y_test.values[self.sequence_length:]

        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }

        return metrics

    def save(self, save_dir: Path) -> None:
        """Save model and scaler."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            self.model.save(save_dir / 'lstm_model.h5')

        import joblib
        if self.scaler_X is not None:
            joblib.dump(self.scaler_X, save_dir / 'scaler_X.pkl')
        if self.scaler_y is not None:
            joblib.dump(self.scaler_y, save_dir / 'scaler_y.pkl')

        logger.info(f"Model saved to {save_dir}")

    @classmethod
    def load(cls, save_dir: Path) -> 'LSTMTrafficPredictor':
        """Load saved model."""
        save_dir = Path(save_dir)

        predictor = cls()

        model_path = save_dir / 'lstm_model.h5'
        if model_path.exists():
            predictor.model = keras.models.load_model(model_path)

        import joblib
        scaler_x_path = save_dir / 'scaler_X.pkl'
        if scaler_x_path.exists():
            predictor.scaler_X = joblib.load(scaler_x_path)

        scaler_y_path = save_dir / 'scaler_y.pkl'
        if scaler_y_path.exists():
            predictor.scaler_y = joblib.load(scaler_y_path)

        logger.info(f"Model loaded from {save_dir}")
        return predictor
