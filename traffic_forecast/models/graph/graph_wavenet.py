"""
Graph WaveNet model for traffic forecasting.

Implements adaptive graph convolution with dilated causal convolutions.
Unlike traditional GCN, this learns the adjacency matrix from data.

Reference: Graph WaveNet for Deep Spatial-Temporal Graph Modeling
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import register_keras_serializable
import numpy as np
from pathlib import Path
from typing import Optional
import joblib
from sklearn.preprocessing import StandardScaler


@register_keras_serializable(package="traffic_forecast")
class GraphWaveNetLayer(layers.Layer):
    """
    Graph WaveNet layer with adaptive adjacency learning.
    
    Combines:
    - Dilated causal convolution for temporal modeling
    - Adaptive graph convolution for spatial modeling
    - Skip connections for gradient flow
    """
    
    def __init__(
        self,
        num_nodes: int,
        hidden_channels: int,
        kernel_size: int = 2,
        dilation: int = 1,
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout_rate = dropout
        
        # Temporal convolution (dilated causal)
        self.temporal_conv = layers.Conv1D(
            filters=hidden_channels,
            kernel_size=kernel_size,
            dilation_rate=dilation,
            padding='causal',
            activation=None
        )
        
        # Graph convolution weights
        self.graph_conv = layers.Dense(hidden_channels, activation=None)
        
        # Gate mechanism
        self.gate_conv = layers.Conv1D(
            filters=hidden_channels,
            kernel_size=kernel_size,
            dilation_rate=dilation,
            padding='causal',
            activation='sigmoid'
        )
        
        # Residual connection
        self.residual = layers.Dense(hidden_channels)
        
        # Skip connection
        self.skip = layers.Dense(hidden_channels)
        
        self.dropout = layers.Dropout(dropout)
        self.layer_norm = layers.LayerNormalization()
        
    def call(self, inputs, adjacency=None, training=False):
        """
        Forward pass.
        
        Args:
            inputs: (batch, timesteps, num_nodes, features)
            adjacency: (num_nodes, num_nodes) adjacency matrix
            training: Whether in training mode
            
        Returns:
            (output, skip_out) tuple
        """
        batch, timesteps, nodes, features = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[3]
        
        # Reshape for temporal convolution: (batch * nodes, timesteps, features)
        x = tf.reshape(inputs, (-1, timesteps, features))
        
        # Temporal convolution
        temporal_out = self.temporal_conv(x)
        gate = self.gate_conv(x)
        x = temporal_out * gate  # Gated activation
        
        # Reshape back: (batch, timesteps, nodes, hidden)
        x = tf.reshape(x, (batch, timesteps, nodes, self.hidden_channels))
        
        # Graph convolution (if adjacency provided)
        if adjacency is not None:
            # Apply graph convolution across spatial dimension
            # For each time and feature, aggregate over nodes: x @ A
            # x shape: (batch, timesteps, nodes, hidden)
            # adjacency: (nodes, nodes)
            # Result: (batch, timesteps, nodes, hidden)
            x_graph = tf.einsum('btif,ij->btjf', x, adjacency)
            x = self.graph_conv(x_graph)
        else:
            x = self.graph_conv(x)
        
        # Residual connection
        residual_out = self.residual(inputs)
        x = x + residual_out
        
        # Skip connection
        skip_out = self.skip(x)
        
        x = self.layer_norm(x)
        x = self.dropout(x, training=training)
        
        return x, skip_out

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_nodes': self.num_nodes,
            'hidden_channels': self.hidden_channels,
            'kernel_size': self.kernel_size,
            'dilation': self.dilation,
            'dropout': self.dropout_rate,
        })
        return config


@register_keras_serializable(package="traffic_forecast")
class GraphWaveNet(keras.Model):
    """
    Graph WaveNet model for traffic speed prediction.
    
    Architecture:
    - Input: Historical speed sequences
    - Multiple Graph WaveNet layers with increasing dilation
    - Skip connections aggregation
    - Output: Future speed prediction
    """
    
    def __init__(
        self,
        num_nodes: int,
        sequence_length: int,
        num_layers: int = 4,
        hidden_channels: int = 32,
        kernel_size: int = 2,
        dropout: float = 0.2,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout
        
        # Input projection
        self.input_proj = layers.Dense(hidden_channels)
        
        # Learnable adjacency matrix (adaptive graph)
        self.source_embeddings = self.add_weight(
            name='source_embeddings',
            shape=(num_nodes, 10),
            initializer='glorot_uniform',
            trainable=True
        )
        self.target_embeddings = self.add_weight(
            name='target_embeddings',
            shape=(10, num_nodes),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Graph WaveNet layers with exponential dilation
        self.gwn_layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            self.gwn_layers.append(
                GraphWaveNetLayer(
                    num_nodes=num_nodes,
                    hidden_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout
                )
            )
        
        # Output projection
        self.output_proj = layers.Dense(1, activation=None)
        
    def compute_adaptive_adjacency(self):
        """Compute adaptive adjacency matrix from learned embeddings."""
        # A = softmax(E1 @ E2)
        adjacency = tf.matmul(self.source_embeddings, self.target_embeddings)
        adjacency = tf.nn.softmax(adjacency, axis=-1)
        return adjacency
    
    def call(self, inputs, training=False):
        """
        Forward pass.
        
        Args:
            inputs: (batch, timesteps, num_nodes, 1) speed sequences
            training: Whether in training mode
            
        Returns:
            predictions: (batch, num_nodes, 1) predicted speeds
        """
        # Input projection
        x = self.input_proj(inputs)
        
        # Compute adaptive adjacency
        adjacency = self.compute_adaptive_adjacency()
        
        # Apply Graph WaveNet layers
        skip_connections = []
        for layer in self.gwn_layers:
            x, skip = layer(x, adjacency=adjacency, training=training)
            skip_connections.append(skip)
        
        # Aggregate skip connections
        skip_sum = tf.add_n(skip_connections)
        
        # Take last timestep
        x = skip_sum[:, -1, :, :]  # (batch, num_nodes, hidden)
        
        # Output projection
        output = self.output_proj(x)  # (batch, num_nodes, 1)
        
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_nodes': self.num_nodes,
            'sequence_length': self.sequence_length,
            'num_layers': self.num_layers,
            'hidden_channels': self.hidden_channels,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class GraphWaveNetTrafficPredictor:
    """
    Traffic speed predictor using Graph WaveNet.
    
    Handles training, prediction, and model persistence.
    """
    
    def __init__(
        self,
        num_nodes: int,
        sequence_length: int = 12,
        num_layers: int = 4,
        hidden_channels: int = 32,
        kernel_size: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001
    ):
        self.num_nodes = num_nodes
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        
        # Build model
        self.model = GraphWaveNet(
            num_nodes=num_nodes,
            sequence_length=sequence_length,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        # Initialize scalers for data normalization
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        # Legacy attributes (kept for backward compatibility)
        self.scaler_mean = None
        self.scaler_std = None
        
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 16,
        verbose: int = 1
    ):
        """
        Train the model with proper data normalization.
        
        Args:
            X_train: (N, seq_len, num_nodes, 1) training sequences
            y_train: (N, num_nodes, 1) training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
        """
        # Normalize training data
        n_samples, seq_len, n_nodes, n_features = X_train.shape
        
        # Reshape X for scaling: (N * seq_len * num_nodes, 1) -> scale -> reshape back
        X_train_2d = X_train.reshape(-1, n_features)
        X_train_scaled = self.scaler_X.fit_transform(X_train_2d)
        X_train_scaled = X_train_scaled.reshape(n_samples, seq_len, n_nodes, n_features)
        
        # Reshape y for scaling: (N * num_nodes, 1) -> scale -> reshape back
        y_train_2d = y_train.reshape(-1, 1)
        y_train_scaled = self.scaler_y.fit_transform(y_train_2d)
        y_train_scaled = y_train_scaled.reshape(y_train.shape)
        
        # Store legacy params for backward compatibility
        self.scaler_mean = np.mean(y_train)
        self.scaler_std = np.std(y_train)
        
        # Normalize validation data if provided
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_2d = X_val.reshape(-1, n_features)
            X_val_scaled = self.scaler_X.transform(X_val_2d)
            X_val_scaled = X_val_scaled.reshape(X_val.shape)
            
            y_val_2d = y_val.reshape(-1, 1)
            y_val_scaled = self.scaler_y.transform(y_val_2d)
            y_val_scaled = y_val_scaled.reshape(y_val.shape)
            
            validation_data = (X_val_scaled, y_val_scaled)
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if validation_data is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        # Train on normalized data
        history = self.model.fit(
            X_train_scaled, y_train_scaled,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with proper denormalization.
        
        Args:
            X: (N, seq_len, num_nodes, 1) input sequences
            
        Returns:
            predictions: (N, num_nodes, 1) predicted speeds in original scale
        """
        # Normalize input
        n_samples, seq_len, n_nodes, n_features = X.shape
        X_2d = X.reshape(-1, n_features)
        X_scaled = self.scaler_X.transform(X_2d)
        X_scaled = X_scaled.reshape(n_samples, seq_len, n_nodes, n_features)
        
        # Predict on normalized data
        y_pred_scaled = self.model.predict(X_scaled, verbose=0)
        
        # Denormalize output
        y_pred_2d = y_pred_scaled.reshape(-1, 1)
        y_pred = self.scaler_y.inverse_transform(y_pred_2d)
        y_pred = y_pred.reshape(y_pred_scaled.shape)
        
        return y_pred
    
    def save(self, save_dir: Path):
        """Save model and scalers to directory."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(save_dir / 'graphwavenet_model.keras')
        
        # Save scalers using joblib
        if self.scaler_X is not None:
            joblib.dump(self.scaler_X, save_dir / 'scaler_X.pkl')
        if self.scaler_y is not None:
            joblib.dump(self.scaler_y, save_dir / 'scaler_y.pkl')
        
        # Save legacy scaler params for backward compatibility
        np.savez(
            save_dir / 'scaler.npz',
            mean=self.scaler_mean,
            std=self.scaler_std
        )
        
        print(f"Model saved to {save_dir}")
    
    @classmethod
    def load(cls, load_dir: Path):
        """Load model and scalers from directory."""
        load_dir = Path(load_dir)
        
        # Load model
        model = keras.models.load_model(
            load_dir / 'graphwavenet_model.keras',
            custom_objects={
                'GraphWaveNet': GraphWaveNet,
                'GraphWaveNetLayer': GraphWaveNetLayer,
            }
        )
        
        # Create predictor instance
        predictor = cls.__new__(cls)
        predictor.model = model
        
        # Load scalers
        scaler_x_path = load_dir / 'scaler_X.pkl'
        scaler_y_path = load_dir / 'scaler_y.pkl'
        
        if scaler_x_path.exists():
            predictor.scaler_X = joblib.load(scaler_x_path)
        else:
            predictor.scaler_X = StandardScaler()
            
        if scaler_y_path.exists():
            predictor.scaler_y = joblib.load(scaler_y_path)
        else:
            predictor.scaler_y = StandardScaler()
        
        # Load legacy scaler params for backward compatibility
        scaler_npz_path = load_dir / 'scaler.npz'
        if scaler_npz_path.exists():
            scaler = np.load(scaler_npz_path)
            predictor.scaler_mean = scaler['mean']
            predictor.scaler_std = scaler['std']
        else:
            predictor.scaler_mean = None
            predictor.scaler_std = None
        
        return predictor
