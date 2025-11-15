"""
GraphWaveNet model wrapper for unified evaluation framework.

Wraps the TensorFlow-based GraphWaveNet model to work with evaluation framework.
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
from traffic_forecast.models.graph.graph_wavenet import GraphWaveNetTrafficPredictor


class GraphWaveNetWrapper(ModelWrapper):
    """
    Wrapper for GraphWaveNet baseline model.
    
    This model:
    - Learns adjacency matrix from data (adaptive graph)
    - Uses dilated causal convolutions for temporal modeling
    - Skip connections for gradient flow
    - Edge-level prediction (like LSTM)
    
    Purpose: Show value of learnable graph structure.
    """
    
    def __init__(
        self,
    sequence_length: int = 12,
    num_layers: int = 4,
    hidden_channels: int = 32,
    kernel_size: int = 2,
    dropout_rate: float = 0.2,
    learning_rate: float = 0.001,
    max_interp_gap: int = 3,
    imputation_noise: float = 0.3,
        seed: int = 42,
    ):
        """
        Initialize GraphWaveNet wrapper.
        
        Args:
            sequence_length: Number of past timesteps
            num_layers: Number of Graph WaveNet layers
            hidden_channels: Hidden channels in each layer
            kernel_size: Temporal convolution kernel size
            dropout_rate: Dropout for regularization
            learning_rate: Adam learning rate
            max_interp_gap: Max consecutive missing timestamps to interpolate
            imputation_noise: Noise ratio applied to imputed values
            seed: Random seed for data prep and training reproducibility
        """
        self.sequence_length = sequence_length
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None  # Will be built after seeing data
        self.num_nodes = None  # Number of unique edges
        self.edge_to_idx = None  # Edge ID mapping
        self._trained = False

        # Data preparation parameters
        self.max_interp_gap = max(0, max_interp_gap)
        self.imputation_noise = max(0.0, imputation_noise)
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        # Cached metadata for consistent preprocessing across splits
        self.edge_order = None
        self.edge_means = None
        self.edge_stds = None
        self.global_speed_mean = None
        
    def _create_edge_id(self, row):
        """Create edge ID from node pair."""
        return f"{row['node_a_id']}_{row['node_b_id']}"
    
    def _prepare_sequences(
        self,
        data: pd.DataFrame,
        speed_col: str,
        add_noise: bool = False,
        return_metadata: bool = False
    ):
        """
        Prepare sequences for GraphWaveNet.
        
        Unlike GCN, GraphWaveNet treats each edge as a node in the graph
        and learns relationships between edges adaptively.
        
        Returns:
            X: (num_sequences, seq_len, num_edges, 1)
            y: (num_sequences, num_edges, 1)
        """
        # Create edge IDs and pivot to timestamp x edge matrix
        data = data.copy()
        data['edge_id'] = data.apply(self._create_edge_id, axis=1)
        pivot = (
            data.pivot_table(
                index='timestamp',
                columns='edge_id',
                values=speed_col,
                aggfunc='mean'
            )
            .sort_index()
        )

        if pivot.empty:
            raise ValueError("Dataset slice is empty. Cannot prepare GraphWaveNet sequences.")

        pivot = self._ensure_edge_order(pivot, data, speed_col)
        filled, imputed_mask = self._fill_missing_values(pivot)

        if add_noise and self.imputation_noise > 0 and self.edge_stds is not None:
            std_vector = np.array([self.edge_stds.get(edge, 1.0) for edge in self.edge_order])
            noise = self._rng.normal(
                loc=0.0,
                scale=std_vector * self.imputation_noise,
                size=filled.shape
            )
            filled = filled + noise * imputed_mask

        speed_matrix = filled.to_numpy()
        timestamps = filled.index.to_list()
        num_edges = speed_matrix.shape[1]
        
        if np.isnan(speed_matrix).any():
            raise ValueError("Speed matrix still contains NaNs after imputation")
        
        # Create sequences
        X_list = []
        y_list = []
        
        for i in range(len(timestamps) - self.sequence_length):
            X_seq = speed_matrix[i:i + self.sequence_length]  # (seq_len, num_edges)
            y_target = speed_matrix[i + self.sequence_length]  # (num_edges,)
            
            X_list.append(X_seq)
            y_list.append(y_target)
        
        if len(X_list) == 0:
            raise ValueError(f"No sequences created. Need at least {self.sequence_length+1} timestamps.")
        
        X = np.array(X_list)  # (num_seq, seq_len, num_edges)
        y = np.array(y_list)  # (num_seq, num_edges)
        
        # Add feature dimension
        X = X[..., np.newaxis]  # (num_seq, seq_len, num_edges, 1)
        y = y[..., np.newaxis]  # (num_seq, num_edges, 1)
        
        print(f"[GraphWaveNet] Prepared sequences: X={X.shape}, y={y.shape}")
        
        if return_metadata:
            metadata = {
                'timestamps': filled.index.to_numpy()
            }
            return X, y, metadata
        
        return X, y

    def _ensure_edge_order(self, pivot: pd.DataFrame, data: pd.DataFrame, speed_col: str) -> pd.DataFrame:
        """
        Ensure consistent edge ordering and cache statistics.
        
        Note: Statistics are calculated from the entire training split.
        This is acceptable for offline baseline model comparison as statistics
        are applied consistently across all splits. For real-time deployment,
        consider using per-sequence or rolling window statistics.
        """
        if self.edge_order is None:
            self.edge_order = sorted(pivot.columns.tolist())
            self.edge_to_idx = {edge: idx for idx, edge in enumerate(self.edge_order)}
            self.num_nodes = len(self.edge_order)
            print(f"[GraphWaveNet] Initialized with {self.num_nodes} edges as nodes")
            
            self.global_speed_mean = float(data[speed_col].mean())
            means = pivot.mean(skipna=True).reindex(self.edge_order)
            global_mean = self.global_speed_mean if not np.isnan(self.global_speed_mean) else 0.0
            self.edge_means = means.fillna(global_mean)
            stds = pivot.std(skipna=True).reindex(self.edge_order)
            global_std = float(data[speed_col].std())
            if np.isnan(global_std) or global_std == 0:
                global_std = 1.0
            self.edge_stds = stds.fillna(global_std).replace(0, global_std)
        
        pivot = pivot.reindex(columns=self.edge_order)
        return pivot

    def _fill_missing_values(self, pivot: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Fill missing edge values with limited interpolation and cached stats.
        
        Methodology:
        1. Time-based interpolation (limited to max_interp_gap)
        2. Forward fill (limited to max_interp_gap)
        3. Fallback to edge means (calculated from training split)
        
        Note: Edge means are global statistics from training data, which is
        acceptable for baseline model comparison. This ensures consistent
        imputation across train/val/test splits.
        """
        pivot = pivot.sort_index()
        missing_mask = pivot.isna().to_numpy(dtype=float)
        filled = pivot.copy()

        if len(filled.index) > 1:
            filled = filled.interpolate(method='time', limit=self.max_interp_gap, limit_direction='forward')
        filled = filled.ffill(limit=self.max_interp_gap)
        
        if self.edge_means is not None:
            filled = filled.fillna(self.edge_means)
        if self.global_speed_mean is not None:
            filled = filled.fillna(self.global_speed_mean)
        
        filled = filled.fillna(0.0)
        return filled, missing_mask
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint directory."""
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.model = GraphWaveNetTrafficPredictor.load(checkpoint_path)
        self._trained = True
        
        print(f"[OK] GraphWaveNet model loaded from {checkpoint_path}")
    
    def predict(
        self,
        data: pd.DataFrame,
        device: str = 'cuda'
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Make predictions on data.
        
        Args:
            data: DataFrame with 'speed' or 'speed_kmh' column
            device: Ignored (TensorFlow handles device placement)
            
        Returns:
            Tuple of (predictions, None) - deterministic predictions
        """
        if not self._trained:
            raise ValueError("Model not trained yet. Call fit() or load_checkpoint() first.")
        
        # Determine speed column
        speed_col = 'speed' if 'speed' in data.columns else 'speed_kmh'
        if speed_col not in data.columns:
            raise ValueError(f"Speed column not found. Available: {data.columns.tolist()}")
        
        try:
            # Prepare sequences (capture timestamp metadata for alignment)
            X, _, metadata = self._prepare_sequences(
                data,
                speed_col,
                add_noise=False,
                return_metadata=True
            )
            
            # Make predictions
            predictions = self.model.predict(X)  # (num_seq, num_edges, 1)
            pred_matrix = np.squeeze(predictions, axis=-1)
            
            # Map predictions back to timestamps/edges
            timestamps = metadata['timestamps']
            target_timestamps = timestamps[self.sequence_length:]
            pred_df = pd.DataFrame(
                pred_matrix,
                index=pd.Index(target_timestamps, name='timestamp'),
                columns=self.edge_order
            )
            pred_long = (
                pred_df.stack()
                .rename('pred_speed')
                .reset_index()
                .rename(columns={'level_1': 'edge_id'})
            )
            
            # Align with original data order
            aligned = data.copy()
            aligned['_row_id'] = np.arange(len(aligned))
            aligned['edge_id'] = aligned.apply(self._create_edge_id, axis=1)
            merged = aligned.merge(
                pred_long,
                on=['timestamp', 'edge_id'],
                how='left'
            ).sort_values('_row_id')
            
            predictions_full = merged['pred_speed'].to_numpy()
            
            return predictions_full, None
            
        except Exception as e:
            print(f"[!] GraphWaveNet prediction error: {e}")
            return np.zeros(len(data)), None
    
    def parameters(self):
        """Return model parameters for counting."""
        if self.model and self.model.model:
            return [p for p in self.model.model.trainable_weights]
        return []
    
    @property
    def model_name(self) -> str:
        """Return human-readable model name."""
        return "GraphWaveNet"
    
    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        epochs: int = 100,
        batch_size: int = 16,
        verbose: int = 1
    ):
        """
        Train the GraphWaveNet model.
        
        Args:
            train_data: Training data with 'speed' or 'speed_kmh' column
            val_data: Validation data
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        # Determine speed column
        speed_col = 'speed' if 'speed' in train_data.columns else 'speed_kmh'
        if speed_col not in train_data.columns:
            raise ValueError(f"Speed column not found. Available: {train_data.columns.tolist()}")
        
        print(f"\n[GraphWaveNet Wrapper] Preparing training sequences...")
        
        # Prepare train sequences
        X_train, y_train = self._prepare_sequences(train_data, speed_col, add_noise=True)
        
        # Prepare validation sequences
        X_val, y_val = self._prepare_sequences(val_data, speed_col, add_noise=False)
        
        # Build model now that we know num_nodes
        if self.model is None:
            print(f"\n[GraphWaveNet Wrapper] Building model for {self.num_nodes} edges...")
            self.model = GraphWaveNetTrafficPredictor(
                num_nodes=self.num_nodes,
                sequence_length=self.sequence_length,
                num_layers=self.num_layers,
                hidden_channels=self.hidden_channels,
                kernel_size=self.kernel_size,
                dropout=self.dropout_rate,
                learning_rate=self.learning_rate
            )
            print(f"GraphWaveNet model built successfully")
            print(f"Architecture: {self.num_layers} layers, {self.hidden_channels} channels")
            
            # Count parameters
            total_params = sum([np.prod(p.shape) for p in self.model.model.trainable_weights])
            print(f"GraphWaveNet wrapper initialized with {total_params:,} parameters")
        
        # Train model
        print(f"\n[GraphWaveNet Wrapper] Training on {len(X_train)} samples...")
        print(f"Input shape: {X_train.shape}")
        print(f"Output shape: {y_train.shape}")
        
        history = self.model.fit(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )
        
        self._trained = True
        
        print(f"\n[GraphWaveNet Wrapper] Training complete")
        
        return history
    
    def save(self, save_dir: Path):
        """Save model to directory."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save(save_dir)
        
        # Save config and edge mapping
        import json
        config = {
            'sequence_length': self.sequence_length,
            'num_layers': self.num_layers,
            'hidden_channels': self.hidden_channels,
            'kernel_size': self.kernel_size,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'model_name': self.model_name,
            'num_nodes': self.num_nodes,
            'edge_to_idx': self.edge_to_idx,
            'edge_order': self.edge_order,
            'edge_means': self.edge_means.to_dict() if self.edge_means is not None else None,
            'edge_stds': self.edge_stds.to_dict() if self.edge_stds is not None else None,
            'global_speed_mean': self.global_speed_mean,
            'max_interp_gap': self.max_interp_gap,
            'imputation_noise': self.imputation_noise,
            'seed': self.seed,
        }
        
        with open(save_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"[OK] GraphWaveNet model saved to {save_dir}")
    
    @classmethod
    def from_checkpoint(cls, checkpoint_path: Path) -> 'GraphWaveNetWrapper':
        """Load model from checkpoint directory."""
        checkpoint_path = Path(checkpoint_path)
        
        # Load config
        import json
        with open(checkpoint_path / 'config.json', 'r') as f:
            config = json.load(f)
        
        # Create wrapper
        wrapper = cls(
            sequence_length=config['sequence_length'],
            num_layers=config['num_layers'],
            hidden_channels=config['hidden_channels'],
            kernel_size=config['kernel_size'],
            dropout_rate=config['dropout_rate'],
            learning_rate=config['learning_rate'],
            max_interp_gap=config.get('max_interp_gap', 6),
            imputation_noise=config.get('imputation_noise', 0.05),
            seed=config.get('seed', 42),
        )

        wrapper.num_nodes = config.get('num_nodes')
        wrapper.edge_to_idx = config.get('edge_to_idx')
        wrapper.edge_order = config.get('edge_order')
        if wrapper.edge_order and wrapper.edge_to_idx is None:
            wrapper.edge_to_idx = {edge: idx for idx, edge in enumerate(wrapper.edge_order)}
        if config.get('edge_means'):
            wrapper.edge_means = pd.Series(config['edge_means']).reindex(wrapper.edge_order)
        if config.get('edge_stds'):
            wrapper.edge_stds = pd.Series(config['edge_stds']).reindex(wrapper.edge_order)
        wrapper.global_speed_mean = config.get('global_speed_mean')
        wrapper._rng = np.random.default_rng(wrapper.seed)

        # Load model
        wrapper.load_checkpoint(checkpoint_path)
        
        return wrapper
