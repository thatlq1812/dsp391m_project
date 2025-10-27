"""Attention-based Spatial-Temporal Graph Convolutional Network (ASTGCN).

This module implements the full ASTGCN architecture described in
"Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic
Flow Forecasting" (Zheng et al., AAAI 2019). It exposes the three-component
design (recent, daily, weekly) with spatial-temporal attention, Chebyshev graph
convolutions, temporal convolutions, and a learnable fusion head suitable for
research experimentation inside this project.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import joblib
import numpy as np

try:
 import tensorflow as tf
 from tensorflow import keras
 from tensorflow.keras import layers

 HAS_TENSORFLOW = True
except ImportError: # pragma: no cover - TensorFlow is optional in CI
 HAS_TENSORFLOW = False

logger = logging.getLogger(__name__)


def _scaled_laplacian(adjacency: np.ndarray) -> np.ndarray:
 """Compute the symmetric scaled Laplacian used by Chebyshev polynomials."""
 if adjacency.shape[0] != adjacency.shape[1]:
 raise ValueError("Adjacency matrix must be square")

 adj = adjacency.astype(np.float32)
 degree = np.sum(adj, axis=1)
 with np.errstate(divide="ignore"):
 degree_inv_sqrt = np.power(degree, -0.5)
 degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0

 d_mat_inv_sqrt = np.diag(degree_inv_sqrt)
 laplacian = np.eye(adj.shape[0], dtype=np.float32) - d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

 lambda_max = np.linalg.eigvals(laplacian).real.max().astype(np.float32)
 if lambda_max <= 0:
 return laplacian

 return (2.0 / lambda_max) * laplacian - np.eye(adj.shape[0], dtype=np.float32)


def _chebyshev_polynomials(laplacian: np.ndarray, order: int) -> Iterable[np.ndarray]:
 """Generate Chebyshev polynomials up to the given order."""
 if order < 1:
 raise ValueError("Chebyshev order must be positive")

 t_k = [np.eye(laplacian.shape[0], dtype=np.float32), laplacian.copy()]
 for _ in range(2, order):
 t_k.append(2 * laplacian @ t_k[-1] - t_k[-2])
 return t_k[:order]


@dataclass
class ASTGCNComponentConfig:
 """Configuration for a single ASTGCN temporal component."""

 window: int
 blocks: int = 2
 temporal_strides: int = 1


@dataclass
class ASTGCNConfig:
 """High-level configuration for ASTGCN."""

 num_nodes: int
 input_dim: int = 1
 horizon: int = 12
 component_configs: "OrderedDict[str, ASTGCNComponentConfig]" = field(
 default_factory=lambda: OrderedDict(
 [
 ("recent", ASTGCNComponentConfig(window=12, blocks=2)),
 ("daily", ASTGCNComponentConfig(window=12 * 24, blocks=1)),
 ("weekly", ASTGCNComponentConfig(window=12 * 24 * 7, blocks=1)),
 ]
 )
 )
 attention_units: int = 32
 cheb_order: int = 3
 spatial_filters: int = 64
 temporal_filters: int = 64
 temporal_kernel: int = 3
 dropout_rate: float = 0.3
 learning_rate: float = 1e-3

 def to_serializable(self) -> Dict[str, object]:
 payload = asdict(self)
 payload["component_configs"] = {
 name: asdict(cfg) for name, cfg in self.component_configs.items()
 }
 return payload

 @classmethod
 def from_dict(cls, payload: Mapping[str, object]) -> "ASTGCNConfig":
 data = dict(payload)
 raw_components = data.get("component_configs", {})
 ordered = OrderedDict(
 (name, ASTGCNComponentConfig(**cfg)) for name, cfg in raw_components.items()
 )
 data["component_configs"] = ordered
 return cls(**data)


class TemporalAttention(layers.Layer):
 """Temporal attention using learned queries and keys."""

 def __init__(self, units: int, **kwargs):
 super().__init__(**kwargs)
 self.units = units
 self.query_dense = layers.Dense(units, use_bias=False)
 self.key_dense = layers.Dense(units, use_bias=False)

 def call(self, inputs): # type: ignore[override]
 # inputs: (batch, time, nodes, features)
 batch_size = tf.shape(inputs)[0]
 time_steps = tf.shape(inputs)[1]

 x = tf.reshape(inputs, (batch_size, time_steps, -1))
 queries = self.query_dense(x)
 keys = self.key_dense(x)

 scale = tf.math.sqrt(tf.cast(self.units, tf.float32))
 scores = tf.matmul(queries, keys, transpose_b=True) / scale
 attention = tf.nn.softmax(scores, axis=-1)
 return attention


class SpatialAttention(layers.Layer):
 """Spatial attention across nodes by aggregating temporal signals."""

 def __init__(self, units: int, **kwargs):
 super().__init__(**kwargs)
 self.units = units
 self.query_dense = layers.Dense(units, use_bias=False)
 self.key_dense = layers.Dense(units, use_bias=False)

 def call(self, inputs): # type: ignore[override]
 # inputs: (batch, time, nodes, features)
 batch_size = tf.shape(inputs)[0]
 num_nodes = tf.shape(inputs)[2]

 x = tf.transpose(inputs, perm=[0, 2, 1, 3]) # (batch, nodes, time, features)
 x = tf.reshape(x, (batch_size, num_nodes, -1))

 queries = self.query_dense(x)
 keys = self.key_dense(x)
 scale = tf.math.sqrt(tf.cast(self.units, tf.float32))
 scores = tf.matmul(queries, keys, transpose_b=True) / scale
 attention = tf.nn.softmax(scores, axis=-1)
 return attention


class ChebyshevGraphConv(layers.Layer):
 """Graph convolution with Chebyshev polynomials and spatial attention."""

 def __init__(self, supports: Iterable[np.ndarray], filters: int, dropout_rate: float, **kwargs):
 super().__init__(**kwargs)
 self.supports = [tf.constant(s, dtype=tf.float32) for s in supports]
 self.filters = filters
 self.dropout = layers.Dropout(dropout_rate)

 def build(self, input_shape): # type: ignore[override]
 feature_dim = int(input_shape[-1])
 self.theta = self.add_weight(
 shape=(len(self.supports), feature_dim, self.filters),
 initializer="glorot_uniform",
 trainable=True,
 name="theta"
 )

 def call(self, inputs, spatial_attention, training=None): # type: ignore[override]
 # inputs: (batch, time, nodes, features)
 batch_size = tf.shape(inputs)[0]
 num_nodes = tf.shape(inputs)[2]

 outputs = 0.0
 for k, support in enumerate(self.supports):
 support_expanded = tf.expand_dims(support, axis=0) # (1, nodes, nodes)
 if spatial_attention is not None:
 support_expanded = support_expanded * spatial_attention

 x_gconv = tf.einsum("bmn,btnf->btmf", support_expanded, inputs)
 x_transformed = tf.einsum("btmf,fd->btmd", x_gconv, self.theta[k])
 outputs += x_transformed

 outputs = tf.nn.relu(outputs)
 outputs = self.dropout(outputs, training=training)
 return outputs


class ASTGCNBlock(layers.Layer):
 """Single ASTGCN block combining attention, graph convolution, and temporal conv."""

 def __init__(
 self,
 supports: Iterable[np.ndarray],
 attention_units: int,
 spatial_filters: int,
 temporal_filters: int,
 temporal_kernel: int,
 temporal_strides: int,
 dropout_rate: float,
 **kwargs,
 ):
 super().__init__(**kwargs)
 self.temporal_attention = TemporalAttention(attention_units)
 self.spatial_attention = SpatialAttention(attention_units)
 self.graph_conv = ChebyshevGraphConv(supports, spatial_filters, dropout_rate)
 self.time_conv = layers.Conv2D(
 filters=temporal_filters,
 kernel_size=(temporal_kernel, 1),
 strides=(temporal_strides, 1),
 padding="same",
 )
 self.residual_conv = layers.Conv2D(
 filters=temporal_filters,
 kernel_size=(1, 1),
 strides=(temporal_strides, 1),
 padding="same",
 )
 self.batch_norm = layers.BatchNormalization()
 self.dropout = layers.Dropout(dropout_rate)

 def call(self, inputs, training=None): # type: ignore[override]
 temporal_attention = self.temporal_attention(inputs)

 x = tf.reshape(inputs, (tf.shape(inputs)[0], tf.shape(inputs)[1], -1))
 x = tf.matmul(temporal_attention, x)
 x = tf.reshape(x, tf.shape(inputs))

 spatial_attention = self.spatial_attention(x)
 x = self.graph_conv(x, spatial_attention, training=training)

 x = self.time_conv(x)
 residual = self.residual_conv(inputs)

 x = self.batch_norm(x + residual, training=training)
 x = tf.nn.relu(x)
 x = self.dropout(x, training=training)
 return x


class ComponentFusion(layers.Layer):
 """Learnable fusion across temporal components with softmax weights."""

 def __init__(self, component_names: Sequence[str], **kwargs):
 super().__init__(**kwargs)
 self.component_names = list(component_names)

 def build(self, input_shape): # type: ignore[override]
 self.logits = self.add_weight(
 shape=(len(self.component_names),),
 initializer="zeros",
 trainable=True,
 name="component_logits",
 )

 def call(self, inputs): # type: ignore[override]
 weights = tf.nn.softmax(self.logits)
 stacked = tf.stack(inputs, axis=-1)
 fused = tf.tensordot(stacked, weights, axes=[-1, 0])
 return fused


class ASTGCNTrafficModel:
 """High-level wrapper for training and inference with ASTGCN."""

 def __init__(self, adjacency: np.ndarray, config: ASTGCNConfig):
 if not HAS_TENSORFLOW:
 raise ImportError("TensorFlow is required to instantiate ASTGCN models")

 if adjacency.shape[0] != adjacency.shape[1]:
 raise ValueError("Adjacency matrix must be square")
 if adjacency.shape[0] != config.num_nodes:
 raise ValueError("Adjacency shape does not match number of nodes in config")

 self.adjacency = adjacency.astype(np.float32)
 self.config = config
 self.component_order = list(config.component_configs.keys())
 self.model: Optional[keras.Model] = None
 self.history: Optional[keras.callbacks.History] = None

 laplacian = _scaled_laplacian(self.adjacency)
 self.supports = list(_chebyshev_polynomials(laplacian, config.cheb_order))

 # ---- model building -----------------------------------------------------------------
 def _build_component(self, input_tensor: tf.Tensor, name: str, component_cfg: ASTGCNComponentConfig) -> tf.Tensor:
 cfg = self.config
 x = input_tensor
 for block_idx in range(component_cfg.blocks):
 x = ASTGCNBlock(
 supports=self.supports,
 attention_units=cfg.attention_units,
 spatial_filters=cfg.spatial_filters,
 temporal_filters=cfg.temporal_filters,
 temporal_kernel=cfg.temporal_kernel,
 temporal_strides=component_cfg.temporal_strides,
 dropout_rate=cfg.dropout_rate,
 name=f"{name}_block_{block_idx + 1}",
 )(x)

 x = layers.Conv2D(
 filters=cfg.horizon,
 kernel_size=(1, 1),
 activation="relu",
 name=f"{name}_forecast_head",
 )(x)
 x = layers.Lambda(lambda t: t[:, -1], name=f"{name}_collapse_time")(x)
 x = layers.Permute((2, 1), name=f"{name}_nodes_first")(x)
 return x

 def _build_model(self) -> keras.Model:
 cfg = self.config
 component_inputs: "OrderedDict[str, tf.Tensor]" = OrderedDict()
 component_outputs: List[tf.Tensor] = []

 for name, component_cfg in cfg.component_configs.items():
 input_layer = layers.Input(
 shape=(component_cfg.window, cfg.num_nodes, cfg.input_dim),
 name=f"{name}_input",
 )
 component_inputs[name] = input_layer
 component_outputs.append(self._build_component(input_layer, name, component_cfg))

 fusion = ComponentFusion(self.component_order, name="component_fusion")(component_outputs)
 outputs = layers.Permute((2, 1), name="forecast_horizon_first")(fusion)

 model = keras.Model(inputs=list(component_inputs.values()), outputs=outputs, name="ASTGCN")
 model.compile(
 optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
 loss="mse",
 metrics=["mae", keras.metrics.RootMeanSquaredError(name="rmse")],
 )
 return model

 # ---- data utilities -----------------------------------------------------------------
 @staticmethod
 def prepare_inputs_from_array(
 graph_signals: np.ndarray,
 config: ASTGCNConfig,
 target_index: int = 0,
 ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
 """Build component tensors and targets from a 3-D graph signal array.

 Args:
 graph_signals: Array with shape (timesteps, num_nodes, input_dim).
 config: ASTGCN configuration describing windows and horizon.
 target_index: Feature index to forecast.

 Returns:
 Tuple of (component_inputs, targets) where component tensors have
 shape (samples, window, num_nodes, input_dim) and targets have shape
 (samples, horizon, num_nodes).
 """
 if graph_signals.ndim != 3:
 raise ValueError("graph_signals must have shape (timesteps, num_nodes, input_dim)")
 if graph_signals.shape[1] != config.num_nodes:
 raise ValueError("graph_signals num_nodes dimension does not match config")
 if graph_signals.shape[2] != config.input_dim:
 raise ValueError("graph_signals feature dimension does not match config")
 if not 0 <= target_index < config.input_dim:
 raise ValueError("target_index is out of bounds for graph_signals")

 horizon = config.horizon
 component_windows = {name: cfg.window for name, cfg in config.component_configs.items()}
 max_window = max(component_windows.values())

 total_steps = graph_signals.shape[0]
 sample_count = total_steps - max_window - horizon + 1
 if sample_count <= 0:
 raise ValueError("Not enough timesteps to generate samples for ASTGCN")

 component_buffers: Dict[str, List[np.ndarray]] = {name: [] for name in component_windows}
 targets: List[np.ndarray] = []

 for idx in range(max_window, total_steps - horizon + 1):
 for name, window in component_windows.items():
 start = idx - window
 end = idx
 component_buffers[name].append(graph_signals[start:end])
 target_slice = graph_signals[idx : idx + horizon, :, target_index]
 targets.append(target_slice)

 component_arrays = {
 name: np.stack(buffers).astype(np.float32)
 for name, buffers in component_buffers.items()
 }
 target_array = np.stack(targets).astype(np.float32)
 return component_arrays, target_array

 # ---- helpers ------------------------------------------------------------------------
 def _ensure_model(self) -> keras.Model:
 if self.model is None:
 self.model = self._build_model()
 return self.model

 def _order_inputs(self, inputs: Union[Mapping[str, np.ndarray], Sequence[np.ndarray]]) -> List[np.ndarray]:
 if isinstance(inputs, Mapping):
 ordered = [inputs[name] for name in self.component_order]
 else:
 ordered = list(inputs)
 if len(ordered) != len(self.component_order):
 raise ValueError("Input count does not match number of ASTGCN components")
 return ordered

 def _validate_component_shapes(self, ordered_inputs: Sequence[np.ndarray]):
 cfg = self.config
 for (name, component_cfg), array in zip(cfg.component_configs.items(), ordered_inputs):
 if array.ndim != 4:
 raise ValueError(f"Component '{name}' input must be 4-D")
 if array.shape[1] != component_cfg.window:
 raise ValueError(f"Component '{name}' window mismatch: {array.shape[1]} vs {component_cfg.window}")
 if array.shape[2] != cfg.num_nodes:
 raise ValueError(f"Component '{name}' num_nodes mismatch")
 if array.shape[3] != cfg.input_dim:
 raise ValueError(f"Component '{name}' feature dimension mismatch")

 # ---- public API ---------------------------------------------------------------------
 def build(self) -> keras.Model:
 return self._ensure_model()

 def train(
 self,
 x_train: Union[Mapping[str, np.ndarray], Sequence[np.ndarray]],
 y_train: np.ndarray,
 x_val: Optional[Union[Mapping[str, np.ndarray], Sequence[np.ndarray]]] = None,
 y_val: Optional[np.ndarray] = None,
 epochs: int = 50,
 batch_size: int = 16,
 verbose: int = 1,
 ) -> Dict[str, Iterable[float]]:
 if y_train.ndim != 3:
 raise ValueError("y_train must have shape (samples, horizon, num_nodes)")
 if y_train.shape[1] != self.config.horizon or y_train.shape[2] != self.config.num_nodes:
 raise ValueError("y_train shape does not match config")

 train_inputs = self._order_inputs(x_train)
 self._validate_component_shapes(train_inputs)

 validation_data = None
 if x_val is not None and y_val is not None:
 val_inputs = self._order_inputs(x_val)
 self._validate_component_shapes(val_inputs)
 if y_val.ndim != 3:
 raise ValueError("y_val must have shape (samples, horizon, num_nodes)")
 val_inputs = [arr.astype(np.float32) for arr in val_inputs]
 validation_data = (val_inputs, y_val.astype(np.float32))

 model = self._ensure_model()
 history = model.fit(
 x=[arr.astype(np.float32) for arr in train_inputs],
 y=y_train.astype(np.float32),
 epochs=epochs,
 batch_size=batch_size,
 validation_data=validation_data,
 shuffle=True,
 verbose=verbose,
 callbacks=[
 keras.callbacks.ReduceLROnPlateau(
 monitor="val_loss" if validation_data else "loss",
 factor=0.5,
 patience=5,
 min_lr=1e-5,
 verbose=1,
 ),
 keras.callbacks.EarlyStopping(
 monitor="val_loss" if validation_data else "loss",
 patience=10,
 restore_best_weights=True,
 verbose=1,
 ),
 ],
 )
 self.history = history
 return history.history

 def predict(self, x: Union[Mapping[str, np.ndarray], Sequence[np.ndarray]]) -> np.ndarray:
 model = self._ensure_model()
 ordered = self._order_inputs(x)
 self._validate_component_shapes(ordered)
 return model.predict([arr.astype(np.float32) for arr in ordered], verbose=0)

 def evaluate(
 self,
 x: Union[Mapping[str, np.ndarray], Sequence[np.ndarray]],
 y: np.ndarray,
 ) -> Dict[str, float]:
 if y.ndim != 3:
 raise ValueError("y must have shape (samples, horizon, num_nodes)")
 if y.shape[1] != self.config.horizon or y.shape[2] != self.config.num_nodes:
 raise ValueError("y shape does not match config")

 model = self._ensure_model()
 ordered = self._order_inputs(x)
 self._validate_component_shapes(ordered)
 results = model.evaluate([arr.astype(np.float32) for arr in ordered], y.astype(np.float32), verbose=0)
 return dict(zip(model.metrics_names, results))

 def save(self, directory: Path, name: str = "astgcn") -> Path:
 if self.model is None:
 raise ValueError("Model must be trained before saving")

 directory = Path(directory)
 directory.mkdir(parents=True, exist_ok=True)

 model_path = directory / f"{name}.keras"
 self.model.save(model_path)

 config_path = directory / f"{name}_config.pkl"
 joblib.dump(self.config.to_serializable(), config_path)

 adjacency_path = directory / f"{name}_adjacency.npy"
 np.save(adjacency_path, self.adjacency)

 logger.info("Saved ASTGCN artifacts to %s", directory)
 return model_path

 @classmethod
 def load(cls, directory: Path, name: str = "astgcn") -> "ASTGCNTrafficModel":
 directory = Path(directory)
 config_path = directory / f"{name}_config.pkl"
 adjacency_path = directory / f"{name}_adjacency.npy"
 model_path = directory / f"{name}.keras"

 if not config_path.exists() or not adjacency_path.exists() or not model_path.exists():
 raise FileNotFoundError("Missing ASTGCN artifacts; cannot load model")

 config = ASTGCNConfig.from_dict(joblib.load(config_path))
 adjacency = np.load(adjacency_path)

 instance = cls(adjacency=adjacency, config=config)
 instance.model = keras.models.load_model(model_path, compile=True)
 return instance


__all__ = ["ASTGCNComponentConfig", "ASTGCNConfig", "ASTGCNTrafficModel"]
