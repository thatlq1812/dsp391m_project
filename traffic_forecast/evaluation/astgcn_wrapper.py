"""ASTGCN wrapper for unified evaluation framework.

This wrapper adapts ASTGCN model to work with the unified ModelWrapper interface.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tensorflow import keras
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False

from traffic_forecast.evaluation.model_wrapper import ModelWrapper
from traffic_forecast.models.graph.astgcn_traffic import ASTGCNConfig, ASTGCNTrafficModel


class ASTGCNWrapper(ModelWrapper):
    """Wrapper for ASTGCN spatial-temporal baseline model."""
    
    def __init__(
        self,
        num_nodes: int = 144,
        window: int = 12,
        horizon: int = 1,
        cheb_order: int = 3,
        spatial_filters: int = 64,
        temporal_filters: int = 64,
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """Initialize ASTGCN wrapper.
        
        Args:
            num_nodes: Number of nodes in graph (edges in our case)
            window: Input sequence length
            horizon: Forecast horizon (1 for next timestep)
            cheb_order: Order of Chebyshev polynomials
            spatial_filters: Number of spatial convolution filters
            temporal_filters: Number of temporal convolution filters
            dropout_rate: Dropout rate
            learning_rate: Learning rate
        """
        from collections import OrderedDict
        from traffic_forecast.models.graph.astgcn_traffic import ASTGCNComponentConfig
        
        # Simplified config - only recent component for baseline
        component_configs = OrderedDict([
            ('recent', ASTGCNComponentConfig(window=window, blocks=2))
        ])
        
        self.config = ASTGCNConfig(
            num_nodes=num_nodes,
            input_dim=1,  # Only speed
            horizon=horizon,
            component_configs=component_configs,
            cheb_order=cheb_order,
            spatial_filters=spatial_filters,
            temporal_filters=temporal_filters,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate
        )
        
        self.model = None
        self._trained = False
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint directory
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Load model
        model_path = checkpoint_path / "astgcn_model"
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load adjacency matrix (required for ASTGCN)
        adj_path = checkpoint_path / "adjacency_matrix.npy"
        if not adj_path.exists():
            raise FileNotFoundError(f"Adjacency matrix not found at {adj_path}")
        
        adjacency = np.load(adj_path)
        self.model = ASTGCNTrafficModel(adjacency, self.config)
        self.model.model = keras.models.load_model(str(model_path))
        self._trained = True
        
    def predict(
        self,
        data: pd.DataFrame,
        adjacency_matrix: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Make predictions on data.
        
        Args:
            data: DataFrame with columns ['timestamp', 'node_a_id', 'node_b_id', 'speed']
            adjacency_matrix: Adjacency matrix (n_nodes, n_nodes)
            
        Returns:
            Tuple of (predictions, None) - ASTGCN is deterministic
        """
        if not self._trained:
            raise ValueError("Model not trained. Call fit() or load_checkpoint() first.")
        
        if adjacency_matrix is None:
            raise ValueError("ASTGCN requires adjacency matrix for spatial modeling")
        
        # Handle both column names
        speed_col = 'speed' if 'speed' in data.columns else 'speed_kmh'
        if speed_col not in data.columns:
            raise ValueError(f"Speed column not found. Available: {data.columns.tolist()}")
        
        # Prepare data in ASTGCN format
        # Group by timestamp to get (n_timestamps, n_nodes) matrix
        timestamps = sorted(data['timestamp'].unique())
        
        # For each edge, get its speed sequence
        # ASTGCN expects (n_samples, n_timestamps, n_nodes, n_features)
        predictions = []
        
        for i in range(len(timestamps) - self.config.component_configs['recent'].window):
            window_timestamps = timestamps[i:i + self.config.component_configs['recent'].window]
            window_data = data[data['timestamp'].isin(window_timestamps)]
            
            # Create (window, n_nodes) speed matrix
            speed_matrix = []
            for ts in window_timestamps:
                ts_data = window_data[window_data['timestamp'] == ts].sort_values('node_a_id')
                if len(ts_data) == self.config.num_nodes:
                    speed_matrix.append(ts_data[speed_col].values)
                else:
                    # Pad with mean if missing nodes
                    speed_vec = np.full(self.config.num_nodes, ts_data[speed_col].mean())
                    speed_matrix.append(speed_vec)
            
            speed_matrix = np.array(speed_matrix)  # (window, n_nodes)
            
            # Reshape for ASTGCN: (1, n_nodes, window, 1)
            X = speed_matrix.T[np.newaxis, :, :, np.newaxis]  # (1, n_nodes, window, 1)
            
            try:
                # ASTGCN expects list of inputs (one per component)
                X_list = [X]
                pred = self.model.predict(X_list)  # (1, horizon, n_nodes)
                predictions.append(pred[0, 0, :])  # (n_nodes,) - first timestep
            except Exception as e:
                print(f"[!] Prediction error at timestep {i}: {e}")
                predictions.append(np.full(self.config.num_nodes, np.nan))
        
        # Flatten predictions to match input data shape
        # Pad beginning with NaN (can't predict first `window` timesteps)
        n_missing = len(data) - len(predictions) * self.config.num_nodes
        predictions_flat = np.concatenate([np.full(n_missing, np.nan)] + 
                                         [pred for pred in predictions])
        
        return predictions_flat[:len(data)], None
    
    def fit(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        adjacency_matrix: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1
    ):
        """Train ASTGCN model.
        
        Args:
            train_data: Training data
            val_data: Validation data
            adjacency_matrix: Graph adjacency matrix
            epochs: Number of epochs
            batch_size: Batch size
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        # Handle both column names
        speed_col = 'speed' if 'speed' in train_data.columns else 'speed_kmh'
        
        # Prepare data for ASTGCN
        # Convert to (n_samples, n_nodes, window, n_features) format
        def prepare_sequences(data):
            timestamps = sorted(data['timestamp'].unique())
            X_list, y_list = [], []
            
            window = self.config.component_configs['recent'].window
            
            for i in range(len(timestamps) - window):
                # Input window
                window_ts = timestamps[i:i + window]
                window_data = data[data['timestamp'].isin(window_ts)]
                
                speed_matrix = []
                for ts in window_ts:
                    ts_data = window_data[window_data['timestamp'] == ts].sort_values('node_a_id')
                    if len(ts_data) == self.config.num_nodes:
                        speed_matrix.append(ts_data[speed_col].values)
                    else:
                        speed_matrix.append(np.full(self.config.num_nodes, 
                                                    ts_data[speed_col].mean()))
                
                # Target (next timestep)
                target_ts = timestamps[i + window]
                target_data = data[data['timestamp'] == target_ts].sort_values('node_a_id')
                if len(target_data) == self.config.num_nodes:
                    target = target_data[speed_col].values
                else:
                    target = np.full(self.config.num_nodes, target_data[speed_col].mean())
                
                X_list.append(np.array(speed_matrix).T[:, :, np.newaxis])  # (n_nodes, window, 1)
                y_list.append(target[:, np.newaxis])  # (n_nodes, 1)
            
            return np.array(X_list), np.array(y_list)
        
        print(f"\n[ASTGCN Baseline] Preparing sequences...")
        X_train, y_train = prepare_sequences(train_data)
        X_val, y_val = prepare_sequences(val_data)
        
        print(f"[ASTGCN Baseline] Training on {len(X_train)} samples...")
        print(f"Input shape: {X_train.shape}")
        print(f"Output shape: {y_train.shape}")
        print(f"Adjacency matrix shape: {adjacency_matrix.shape}")
        
        # Initialize model
        self.model = ASTGCNTrafficModel(adjacency_matrix, self.config)
        
        # Train
        # ASTGCN expects list of inputs (one per component)
        # We only have 'recent' component
        X_train_list = [X_train]
        X_val_list = [X_val] if X_val is not None else None
        
        # Build model first to ensure it exists
        self.model.build_model()
        
        # Prepare callbacks with early stopping
        if HAS_TENSORFLOW:
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
        else:
            callbacks = []
        
        history = self.model.fit(
            X_train_list, y_train,
            X_val_list, y_val,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self._trained = True
        
        print(f"\n[ASTGCN Baseline] Training complete!")
        
        return history
    
    def parameters(self):
        """Return model parameters."""
        if self.model and self.model.model:
            return [p for p in self.model.model.trainable_weights]
        return []
    
    @property
    def model_name(self) -> str:
        """Return model name."""
        return "ASTGCN-Baseline"
    
    def save(self, save_path: str):
        """Save model to directory.
        
        Args:
            save_path: Directory path to save model
        """
        if not self._trained:
            raise ValueError("Cannot save untrained model")
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save Keras model
        model_path = save_path / "astgcn_model"
        self.model.model.save(str(model_path))
        
        # Save adjacency matrix
        adj_path = save_path / "adjacency_matrix.npy"
        np.save(adj_path, self.model.adjacency)
        
        # Save config
        import json
        config_path = save_path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2, default=str)
