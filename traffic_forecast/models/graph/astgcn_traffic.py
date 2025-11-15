"""Attention-based Spatial-Temporal Graph Convolutional Network (ASTGCN).

This module implements the ASTGCN architecture for traffic forecasting
as described in "Attention Based Spatial-Temporal Graph Convolutional Networks
for Traffic Flow Forecasting" (Guo et al., AAAI 2019).
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Union

import joblib
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

logger = logging.getLogger(__name__)


def compute_scaled_laplacian(adjacency: np.ndarray) -> np.ndarray:
    """Compute the symmetric scaled Laplacian for Chebyshev polynomials.

    Args:
        adjacency: Adjacency matrix of shape (n_nodes, n_nodes)

    Returns:
        Scaled Laplacian matrix
    """
    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency matrix must be square")

    adj = adjacency.astype(np.float32)
    degree = np.sum(adj, axis=1)

    with np.errstate(divide='ignore'):
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0

    d_mat_inv_sqrt = np.diag(degree_inv_sqrt)
    laplacian = np.eye(adj.shape[0], dtype=np.float32) - d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    lambda_max = np.linalg.eigvals(laplacian).real.max()
    if lambda_max <= 0:
        return laplacian

    return (2.0 / lambda_max) * laplacian - np.eye(adj.shape[0], dtype=np.float32)


def generate_chebyshev_polynomials(laplacian: np.ndarray, order: int) -> List[np.ndarray]:
    """Generate Chebyshev polynomials up to given order.

    Args:
        laplacian: Scaled Laplacian matrix
        order: Maximum order of polynomials

    Returns:
        List of Chebyshev polynomial matrices
    """
    if order < 1:
        raise ValueError("Order must be at least 1")

    polynomials = [
        np.eye(laplacian.shape[0], dtype=np.float32),
        laplacian.copy()
    ]

    for _ in range(2, order):
        new_poly = 2 * laplacian @ polynomials[-1] - polynomials[-2]
        polynomials.append(new_poly)

    return polynomials[:order]


@dataclass
class ASTGCNComponentConfig:
    """Configuration for a single temporal component (recent/daily/weekly)."""

    window: int
    blocks: int = 2
    temporal_strides: int = 1


@dataclass
class ASTGCNConfig:
    """Complete configuration for ASTGCN model."""

    num_nodes: int
    input_dim: int = 1
    horizon: int = 12
    component_configs: OrderedDict[str, ASTGCNComponentConfig] = field(
        default_factory=lambda: OrderedDict([
            ('recent', ASTGCNComponentConfig(window=12, blocks=2)),
            ('daily', ASTGCNComponentConfig(window=288, blocks=1)),
            ('weekly', ASTGCNComponentConfig(window=2016, blocks=1)),
        ])
    )
    attention_units: int = 32
    cheb_order: int = 3
    spatial_filters: int = 64
    temporal_filters: int = 64
    temporal_kernel: int = 3
    dropout_rate: float = 0.3
    learning_rate: float = 0.001

    def to_dict(self) -> Dict[str, object]:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> 'ASTGCNConfig':
        """Create config from dictionary."""
        component_data = data.get('component_configs', {})
        components = OrderedDict()

        for name, cfg in component_data.items():
            components[name] = ASTGCNComponentConfig(**cfg)

        config_dict = dict(data)
        config_dict['component_configs'] = components

        return cls(**config_dict)


class TemporalAttention(layers.Layer):
    """Temporal attention mechanism."""

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.W = self.add_weight(
            name='temporal_W',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='temporal_b',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.V = self.add_weight(
            name='temporal_V',
            shape=(self.units, 1),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        num_nodes = tf.shape(inputs)[2]

        inputs_reshaped = tf.reshape(inputs, [-1, inputs.shape[-1]])
        scores = tf.matmul(tf.nn.tanh(tf.matmul(inputs_reshaped, self.W) + self.b), self.V)
        scores = tf.reshape(scores, [batch_size, time_steps, num_nodes])

        attention_weights = tf.nn.softmax(scores, axis=1)
        return attention_weights


class SpatialAttention(layers.Layer):
    """Spatial attention mechanism that produces node-to-node weights."""

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.query_dense = layers.Dense(units, use_bias=False)
        self.key_dense = layers.Dense(units, use_bias=False)

    def call(self, inputs):
        shape = tf.shape(inputs)
        batch_size = shape[0]
        time_steps = shape[1]
        num_nodes = shape[2]

        # Merge batch and time so we can compute attention per timestep
        bt = batch_size * time_steps
        features = shape[3]
        x = tf.reshape(inputs, (bt, num_nodes, features))

        queries = self.query_dense(x)  # (bt, nodes, units)
        keys = self.key_dense(x)       # (bt, nodes, units)

        scale = tf.math.sqrt(tf.cast(self.units, tf.float32))
        logits = tf.matmul(queries, keys, transpose_b=True) / scale  # (bt, nodes, nodes)
        attention = tf.nn.softmax(logits, axis=-1)

        attention = tf.reshape(attention, (batch_size, time_steps, num_nodes, num_nodes))
        return attention


class ChebGraphConv(layers.Layer):
    """Chebyshev graph convolution layer."""

    def __init__(
        self,
        cheb_polynomials: List[np.ndarray],
        filters: int,
        dropout_rate: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.cheb_polynomials = [tf.constant(p, dtype=tf.float32) for p in cheb_polynomials]
        self.filters = filters
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        self.kernels = []

        for i in range(len(self.cheb_polynomials)):
            kernel = self.add_weight(
                name=f'cheb_kernel_{i}',
                shape=(feature_dim, self.filters),
                initializer='glorot_uniform',
                trainable=True
            )
            self.kernels.append(kernel)

        self.dropout = layers.Dropout(self.dropout_rate)

    def call(self, inputs, spatial_attention, training=None):
        batch_size = tf.shape(inputs)[0]
        time_steps = tf.shape(inputs)[1]
        num_nodes = tf.shape(inputs)[2]

        outputs = []
        for poly, kernel in zip(self.cheb_polynomials, self.kernels):
            poly_expanded = tf.reshape(poly, (1, 1, num_nodes, num_nodes))
            weighted_poly = poly_expanded * spatial_attention

            x_transformed = tf.matmul(inputs, kernel)
            x_graph = tf.matmul(weighted_poly, x_transformed)
            outputs.append(x_graph)

        output = tf.add_n(outputs)
        output = self.dropout(output, training=training)

        return output


class ASTGCNBlock(layers.Layer):
    """ASTGCN block with spatial-temporal attention and graph convolution."""

    def __init__(
        self,
        cheb_polynomials: List[np.ndarray],
        spatial_filters: int,
        temporal_filters: int,
        temporal_kernel: int,
        attention_units: int,
        dropout_rate: float,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.temporal_attention = TemporalAttention(attention_units)
        self.spatial_attention = SpatialAttention(attention_units)
        self.cheb_conv = ChebGraphConv(cheb_polynomials, spatial_filters, dropout_rate)

        self.temporal_conv = layers.Conv2D(
            filters=temporal_filters,
            kernel_size=(temporal_kernel, 1),
            padding='same',
            activation='relu'
        )
        self.batch_norm = layers.BatchNormalization()
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        temporal_att = self.temporal_attention(inputs)
        spatial_att = self.spatial_attention(inputs)

        # Expand attention weights to match input dimensions
        temporal_att = tf.expand_dims(temporal_att, axis=-1)  # [batch, time, nodes, 1]
        x = inputs * temporal_att
        x = self.cheb_conv(x, spatial_att, training=training)

        x = self.temporal_conv(x)
        x = self.batch_norm(x, training=training)
        x = self.dropout(x, training=training)

        return x


class ComponentFusion(layers.Layer):
    """Learnable fusion of multiple temporal components."""

    def __init__(self, component_names: Sequence[str], **kwargs):
        super().__init__(**kwargs)
        self.component_names = list(component_names)

    def build(self, input_shape):
        num_components = len(self.component_names)
        self.fusion_weights = self.add_weight(
            name='fusion_weights',
            shape=(num_components,),
            initializer='ones',
            trainable=True
        )

    def call(self, inputs):
        weights = tf.nn.softmax(self.fusion_weights)
        weighted_sum = sum(w * x for w, x in zip(weights, inputs))
        return weighted_sum


class ASTGCNTrafficModel:
    """ASTGCN model for traffic forecasting."""

    def __init__(self, adjacency: np.ndarray, config: ASTGCNConfig):
        """Initialize ASTGCN model.

        Args:
            adjacency: Adjacency matrix of shape (num_nodes, num_nodes)
            config: ASTGCN configuration
        """
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow is required for ASTGCN")

        if adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("Adjacency matrix must be square")

        if adjacency.shape[0] != config.num_nodes:
            raise ValueError("Adjacency shape does not match config")

        self.adjacency = adjacency
        self.config = config
        self.model = None
        self.component_order = list(config.component_configs.keys())

        laplacian = compute_scaled_laplacian(adjacency)
        self.cheb_polynomials = generate_chebyshev_polynomials(
            laplacian, config.cheb_order
        )

    def _build_component_path(self, name: str, component_cfg: ASTGCNComponentConfig):
        """Build processing path for one temporal component."""
        input_layer = layers.Input(
            shape=(component_cfg.window, self.config.num_nodes, self.config.input_dim),
            name=f'{name}_input'
        )

        x = input_layer
        for block_idx in range(component_cfg.blocks):
            x = ASTGCNBlock(
                cheb_polynomials=self.cheb_polynomials,
                spatial_filters=self.config.spatial_filters,
                temporal_filters=self.config.temporal_filters,
                temporal_kernel=self.config.temporal_kernel,
                attention_units=self.config.attention_units,
                dropout_rate=self.config.dropout_rate,
                name=f'{name}_block_{block_idx}'
            )(x)

    x = layers.GlobalAveragePooling2D(name=f'{name}_pool')(x)

        return input_layer, x

    def build_model(self) -> keras.Model:
        """Build complete ASTGCN model."""
        component_inputs = []
        component_outputs = []

        for name, comp_cfg in self.config.component_configs.items():
            inp, out = self._build_component_path(name, comp_cfg)
            component_inputs.append(inp)
            component_outputs.append(out)

        fused = ComponentFusion(
            self.component_order,
            name='fusion'
        )(component_outputs)

        output = layers.Dense(
            self.config.horizon * self.config.num_nodes,
            activation='linear',
            name='output'
        )(fused)

        output = layers.Reshape(
            (self.config.horizon, self.config.num_nodes),
            name='reshape'
        )(output)

        model = keras.Model(
            inputs=component_inputs,
            outputs=output,
            name='astgcn_traffic'
        )

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss='mse',
            metrics=['mae', 'mse']
        )

        self.model = model
        return model

    def get_model(self) -> keras.Model:
        """Get or build the Keras model."""
        if self.model is None:
            self.build_model()
        return self.model

    def fit(self, X_train, y_train, X_val=None, y_val=None, **fit_kwargs):
        """Train the model."""
        model = self.get_model()

        validation_data = None
        if X_val is not None and y_val is not None:
            if isinstance(X_val, dict):
                X_val_list = [X_val[name] for name in self.component_order]
            else:
                X_val_list = X_val
            validation_data = (X_val_list, y_val)

        if isinstance(X_train, dict):
            X_train = [X_train[name] for name in self.component_order]

        history = model.fit(
            X_train,
            y_train,
            validation_data=validation_data,
            **fit_kwargs
        )

        return history

    def predict(self, X):
        """Make predictions."""
        model = self.get_model()

        if isinstance(X, dict):
            X = [X[name] for name in self.component_order]

        return model.predict(X)

    def save(self, save_dir: Path) -> None:
        """Save model and config."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            self.model.save(save_dir / 'astgcn_model.h5')

        joblib.dump(self.config, save_dir / 'config.pkl')
        np.save(save_dir / 'adjacency.npy', self.adjacency)

        logger.info(f"ASTGCN model saved to {save_dir}")

    @classmethod
    def load(cls, save_dir: Path) -> 'ASTGCNTrafficModel':
        """Load saved model."""
        save_dir = Path(save_dir)

        config = joblib.load(save_dir / 'config.pkl')
        adjacency = np.load(save_dir / 'adjacency.npy')

        instance = cls(adjacency, config)

        model_path = save_dir / 'astgcn_model.h5'
        if model_path.exists():
            instance.model = keras.models.load_model(model_path)

        logger.info(f"ASTGCN model loaded from {save_dir}")
        return instance
