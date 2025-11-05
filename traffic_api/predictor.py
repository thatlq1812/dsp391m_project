"""STMGT Model Predictor - Real-time Inference Wrapper."""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

# Import STMGT directly without going through models.__init__ (avoids TensorFlow import)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from traffic_forecast.models.stmgt.model import STMGT
from traffic_forecast.models.stmgt.inference import mixture_to_moments


class STMGTPredictor:
    """STMGT model inference wrapper for API."""

    def __init__(
        self,
        checkpoint_path: Path,
        data_path: Optional[Path] = None,
        device: str = "cuda",
    ):
        """
        Initialize predictor with model checkpoint and data.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pt file)
            data_path: Path to training data parquet (to get exact node/edge structure)
            device: Device to run inference on
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Load config from checkpoint directory
        config_path = checkpoint_path.parent / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print(f"Loaded config: {self.config['model']['num_nodes']} nodes")
        else:
            self.config = None
        
        # Auto-detect data path if not provided
        if data_path is None:
            if self.config and 'metadata' in self.config:
                data_path = Path(self.config['metadata']['data_path'])
            else:
                # Try common paths
                data_path = checkpoint_path.parents[1] / "data" / "processed" / "all_runs_extreme_augmented.parquet"
        
        self.data_path = data_path
        
        # Load model
        print(f"Loading model from {checkpoint_path}...")
        self.model = self._load_model()
        self.model.eval()
        print(f"Model loaded on {self.device}")
        
        # Load graph structure from actual training data
        print(f"Loading graph structure from {self.data_path}...")
        self._load_graph_structure()
        
        # Initialize historical data cache
        self._init_historical_data()
    
    def _load_graph_structure(self):
        """Load EXACT graph structure used in training."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        import pandas as pd
        df = pd.read_parquet(self.data_path)
        
        # Get unique nodes from data (EXACT same as training)
        unique_node_ids = set()
        unique_node_ids.update(df['node_a_id'].unique())
        unique_node_ids.update(df['node_b_id'].unique())
        
        self.node_ids = sorted(list(unique_node_ids))
        self.node_to_idx = {node: idx for idx, node in enumerate(self.node_ids)}
        self.num_nodes = len(self.node_ids)
        
        # Create edge_index from actual traffic edges
        edge_set = set()
        for _, row in df[['node_a_id', 'node_b_id']].drop_duplicates().iterrows():
            node_a = row['node_a_id']
            node_b = row['node_b_id']
            if node_a in self.node_to_idx and node_b in self.node_to_idx:
                edge_set.add((self.node_to_idx[node_a], self.node_to_idx[node_b]))
        
        edge_list = list(edge_set)
        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t().to(self.device)
        
        print(f"Graph: {self.num_nodes} nodes, {self.edge_index.size(1)} edges")
        
        # Load node metadata from topology for display purposes
        topology_path = self.checkpoint_path.parents[1] / "cache" / "overpass_topology.json"
        self.node_metadata = {}
        
        if topology_path.exists():
            with open(topology_path, 'r', encoding='utf-8') as f:
                topology = json.load(f)
                for node in topology['nodes']:
                    self.node_metadata[node['node_id']] = node
        
        # Verify node count matches config
        if self.config and self.config['model']['num_nodes'] != self.num_nodes:
            print(f"WARNING: Model expects {self.config['model']['num_nodes']} nodes but data has {self.num_nodes}")
            print(f"This may cause incorrect predictions!")
    
    def _load_model(self) -> STMGT:
        """Load STMGT model from checkpoint."""
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Extract model config (from checkpoint or use defaults)
        model_config = {
            'num_nodes': 62,  # Will update based on topology
            'in_dim': 1,
            'hidden_dim': 96,
            'num_blocks': 3,
            'num_heads': 4,
            'dropout': 0.2,
            'drop_edge_rate': 0.05,
            'mixture_components': 3,
            'seq_len': 12,
            'pred_len': 12,
        }
        
        # Create model
        model = STMGT(**model_config)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        return model
    
    def _init_historical_data(self):
        """Initialize historical data cache from actual parquet data."""
        print("Loading historical data from parquet...")
        
        try:
            import pandas as pd
            df = pd.read_parquet(self.data_path)
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Get latest 12 timesteps (for each unique run)
            latest_run = df['run_id'].max()
            df_latest = df[df['run_id'] == latest_run]
            
            # Initialize arrays
            self.historical_speeds = np.zeros((self.num_nodes, 12))
            self.historical_weather = np.zeros((self.num_nodes, 12, 3))
            
            # Build speed matrix (node × timestep)
            # Group by edge and get speed sequence
            for edge_key, edge_df in df_latest.groupby(['node_a_id', 'node_b_id']):
                node_a_id, node_b_id = edge_key
                
                # Get node index (use node_a as representative)
                if node_a_id in self.node_to_idx:
                    node_idx = self.node_to_idx[node_a_id]
                    
                    # Get speed values
                    speeds = edge_df['speed_kmh'].values[:12]
                    if len(speeds) > 0:
                        # Pad if needed
                        if len(speeds) < 12:
                            speeds = np.pad(speeds, (12 - len(speeds), 0), 
                                          mode='edge')  # Repeat edge values
                        self.historical_speeds[node_idx] = speeds[:12]
                    
                    # Get weather if available
                    if 'temperature' in edge_df.columns:
                        temps = edge_df['temperature'].values[:12]
                        if len(temps) < 12:
                            temps = np.pad(temps, (12 - len(temps), 0), mode='edge')
                        self.historical_weather[node_idx, :, 0] = temps[:12]
                    
                    if 'precipitation' in edge_df.columns:
                        precip = edge_df['precipitation'].values[:12]
                        if len(precip) < 12:
                            precip = np.pad(precip, (12 - len(precip), 0), mode='edge')
                        self.historical_weather[node_idx, :, 1] = precip[:12]
                    
                    if 'wind_speed' in edge_df.columns:
                        wind = edge_df['wind_speed'].values[:12]
                        if len(wind) < 12:
                            wind = np.pad(wind, (12 - len(wind), 0), mode='edge')
                        self.historical_weather[node_idx, :, 2] = wind[:12]
            
            # Fill any missing nodes with mean values
            for i in range(self.num_nodes):
                if self.historical_speeds[i].sum() == 0:
                    self.historical_speeds[i] = self.historical_speeds.mean(axis=0)
                if self.historical_weather[i].sum() == 0:
                    self.historical_weather[i] = self.historical_weather.mean(axis=0)
            
            print(f"✓ Loaded real data from {self.data_path.name}")
            print(f"  Latest run: {latest_run}")
            print(f"  Speed range: {self.historical_speeds.min():.1f} - {self.historical_speeds.max():.1f} km/h")
            print(f"  Mean speed: {self.historical_speeds.mean():.1f} km/h")
            
        except Exception as e:
            print(f"Warning: Failed to load real data: {e}")
            print("Using fallback synthetic data...")
            import traceback
            traceback.print_exc()
            
            # Fallback: Use realistic synthetic data
            base_speed = np.random.uniform(30, 50, (self.num_nodes, 1))
            noise = np.random.normal(0, 2, (self.num_nodes, 12))  # Small noise ±2 km/h
            self.historical_speeds = np.clip(base_speed + noise, 15, 80)
            
            self.historical_weather = np.zeros((self.num_nodes, 12, 3))
            self.historical_weather[:, :, 0] = 28.0  # Temp
            self.historical_weather[:, :, 1] = 0.0   # Rain
            self.historical_weather[:, :, 2] = 5.0   # Wind
    
    def _prepare_inputs(
        self,
        timestamp: datetime,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Prepare model inputs from historical data."""
        # Traffic history (last 12 timesteps)
        x_traffic = torch.tensor(
            self.historical_speeds,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0).unsqueeze(-1)  # (1, N, T, 1)
        
        # Weather forecast - need to average over nodes for cross-attention
        # Model expects (B, N, 3) not (B, N, T, 3)
        weather_avg = self.historical_weather.mean(axis=1)  # Average over time
        x_weather = torch.tensor(
            weather_avg,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)  # (1, N, 3)
        
        # Temporal features
        temporal_features = {
            'hour': torch.tensor([timestamp.hour], dtype=torch.long, device=self.device),
            'dow': torch.tensor([timestamp.weekday()], dtype=torch.long, device=self.device),
            'is_weekend': torch.tensor([1 if timestamp.weekday() >= 5 else 0], dtype=torch.long, device=self.device),
        }
        
        return x_traffic, x_weather, temporal_features
    
    @torch.no_grad()
    def predict(
        self,
        timestamp: Optional[datetime] = None,
        node_ids: Optional[list[str]] = None,
        horizons: Optional[list[int]] = None,
    ) -> dict:
        """
        Generate traffic predictions.
        
        Args:
            timestamp: Forecast timestamp (defaults to now)
            node_ids: Specific nodes to predict (defaults to all)
            horizons: Forecast horizons in timesteps (defaults to [1,2,3,6,9,12])
        
        Returns:
            Dictionary with predictions per node
        """
        start_time = time.time()
        
        if timestamp is None:
            timestamp = datetime.now()
        
        if horizons is None:
            horizons = [1, 2, 3, 6, 9, 12]
        
        # Prepare inputs
        x_traffic, x_weather, temporal_features = self._prepare_inputs(timestamp)
        
        # Forward pass
        pred_params = self.model(
            x_traffic,
            self.edge_index,
            x_weather,
            temporal_features,
        )
        
        # Convert mixture to moments
        pred_mean, pred_std = mixture_to_moments(pred_params)
        
        # Move to CPU and convert to numpy
        pred_mean = pred_mean.squeeze(0).cpu().numpy()  # (N, T)
        pred_std = pred_std.squeeze(0).cpu().numpy()    # (N, T)
        
        # Clip to valid range
        pred_mean = np.clip(pred_mean, 0, 100)
        
        # Build response
        predictions = []
        selected_nodes = node_ids if node_ids else self.node_ids
        
        for node_id in selected_nodes:
            if node_id not in self.node_to_idx:
                continue
            
            node = self.get_node(node_id)
            if node is None:
                continue
            
            node_idx = self.node_to_idx[node_id]
            
            forecasts = []
            for h in horizons:
                if h > 12 or h < 1:
                    continue
                
                mean = float(pred_mean[node_idx, h - 1])
                std = float(pred_std[node_idx, h - 1])
                
                # 80% confidence interval
                lower = max(0, mean - 1.28 * std)
                upper = min(100, mean + 1.28 * std)
                
                forecasts.append({
                    'horizon': h,
                    'horizon_minutes': h * 15,
                    'mean': round(mean, 2),
                    'std': round(std, 2),
                    'lower_80': round(lower, 2),
                    'upper_80': round(upper, 2),
                })
            
            predictions.append({
                'node_id': node_id,
                'lat': node['lat'],
                'lon': node['lon'],
                'forecasts': forecasts,
                'current_speed': round(float(self.historical_speeds[node_idx, -1]), 2),
            })
        
        inference_time = (time.time() - start_time) * 1000  # ms
        
        return {
            'timestamp': timestamp,
            'forecast_time': timestamp + timedelta(minutes=15),
            'nodes': predictions,
            'model_version': 'stmgt_v2',
            'inference_time_ms': round(inference_time, 2),
        }
    
    def get_nodes(self) -> list[dict]:
        """Get all node information."""
        nodes = []
        for node_id in self.node_ids:
            # Get metadata if available
            if node_id in self.node_metadata:
                nodes.append(self.node_metadata[node_id])
            else:
                # Create minimal node info
                nodes.append({
                    'node_id': node_id,
                    'lat': 0.0,  # Unknown
                    'lon': 0.0,  # Unknown
                    'degree': 0,
                    'importance_score': 0.0,
                    'road_type': 'unknown',
                    'street_names': [],
                    'intersection_name': node_id,
                    'is_major_intersection': False
                })
        return nodes
    
    def get_node(self, node_id: str) -> Optional[dict]:
        """Get specific node information."""
        if node_id in self.node_metadata:
            return self.node_metadata[node_id]
        elif node_id in self.node_to_idx:
            return {
                'node_id': node_id,
                'lat': 0.0,
                'lon': 0.0,
                'degree': 0,
                'importance_score': 0.0,
                'road_type': 'unknown',
                'street_names': [],
                'intersection_name': node_id,
                'is_major_intersection': False
            }
        return None
