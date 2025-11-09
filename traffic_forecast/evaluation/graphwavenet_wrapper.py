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
        learning_rate: float = 0.001
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
        
    def _create_edge_id(self, row):
        """Create edge ID from node pair."""
        return f"{row['node_a_id']}_{row['node_b_id']}"
    
    def _prepare_sequences(self, data: pd.DataFrame, speed_col: str):
        """
        Prepare sequences for GraphWaveNet.
        
        Unlike GCN, GraphWaveNet treats each edge as a node in the graph
        and learns relationships between edges adaptively.
        
        Returns:
            X: (num_sequences, seq_len, num_edges, 1)
            y: (num_sequences, num_edges, 1)
        """
        # Create edge IDs
        data = data.copy()
        data['edge_id'] = data.apply(self._create_edge_id, axis=1)
        
        # Get unique edges
        unique_edges = sorted(data['edge_id'].unique())
        num_edges = len(unique_edges)
        
        if self.edge_to_idx is None:
            self.edge_to_idx = {edge: idx for idx, edge in enumerate(unique_edges)}
            self.num_nodes = num_edges
            print(f"[GraphWaveNet] Initialized with {num_edges} edges as nodes")
        
        # Get unique timestamps
        timestamps = sorted(data['timestamp'].unique())
        
        # Build speed matrix: (num_timestamps, num_edges)
        speed_matrix = np.zeros((len(timestamps), num_edges))
        for t_idx, ts in enumerate(timestamps):
            ts_data = data[data['timestamp'] == ts]
            for _, row in ts_data.iterrows():
                edge_id = row['edge_id']
                if edge_id in self.edge_to_idx:
                    edge_idx = self.edge_to_idx[edge_id]
                    speed_matrix[t_idx, edge_idx] = row[speed_col]
        
        # Create sequences
        X_list = []
        y_list = []
        
        for i in range(len(timestamps) - self.sequence_length):
            X_seq = speed_matrix[i:i+self.sequence_length]  # (seq_len, num_edges)
            y_target = speed_matrix[i+self.sequence_length]  # (num_edges,)
            
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
        
        return X, y
    
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
            # Prepare sequences
            X, _ = self._prepare_sequences(data, speed_col)
            
            # Make predictions
            predictions = self.model.predict(X)  # (num_seq, num_edges, 1)
            
            # Flatten back to match original data structure
            # Note: predictions are shorter than input due to sequence creation
            predictions_flat = predictions.reshape(-1)
            
            # Pad to match input length
            n_missing = len(data) - len(predictions_flat)
            predictions_full = np.concatenate([
                np.full(n_missing, np.nan),
                predictions_flat
            ])
            
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
        X_train, y_train = self._prepare_sequences(train_data, speed_col)
        
        # Prepare validation sequences
        X_val, y_val = self._prepare_sequences(val_data, speed_col)
        
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
            'edge_to_idx': self.edge_to_idx
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
            learning_rate=config['learning_rate']
        )
        
        wrapper.num_nodes = config.get('num_nodes')
        wrapper.edge_to_idx = config.get('edge_to_idx')
        
        # Load model
        wrapper.load_checkpoint(checkpoint_path)
        
        return wrapper
