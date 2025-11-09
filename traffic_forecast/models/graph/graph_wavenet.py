"""
Graph WaveNet model for traffic forecasting.

Implements adaptive graph convolution with dilated causal convolutions.
Unlike traditional GCN, this learns the adjacency matrix from data.

Reference: Graph WaveNet for Deep Spatial-Temporal Graph Modeling
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from pathlib import Path
from typing import Optional


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
        Train the model.
        
        Args:
            X_train: (N, seq_len, num_nodes, 1) training sequences
            y_train: (N, num_nodes, 1) training targets
            X_val: Validation sequences
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
        """
        # Store normalization params
        self.scaler_mean = np.mean(y_train)
        self.scaler_std = np.std(y_train)
        
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6
            )
        ]
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        return history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X, verbose=0)
    
    def save(self, save_dir: Path):
        """Save model to directory."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(save_dir / 'graphwavenet_model.keras')
        
        # Save scaler params
        np.savez(
            save_dir / 'scaler.npz',
            mean=self.scaler_mean,
            std=self.scaler_std
        )
        
        print(f"Model saved to {save_dir}")
    
    @classmethod
    def load(cls, load_dir: Path):
        """Load model from directory."""
        load_dir = Path(load_dir)
        
        # Load model
        model = keras.models.load_model(load_dir / 'graphwavenet_model.keras')
        
        # Load scaler
        scaler = np.load(load_dir / 'scaler.npz')
        
        # Create predictor instance
        predictor = cls.__new__(cls)
        predictor.model = model
        predictor.scaler_mean = scaler['mean']
        predictor.scaler_std = scaler['std']
        
        return predictor
