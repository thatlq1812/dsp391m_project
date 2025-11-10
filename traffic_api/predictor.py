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
import networkx as nx

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
        
        # Load config from checkpoint directory (try multiple names)
        config_paths = [
            checkpoint_path.parent / "config.json",
            checkpoint_path.parent / "stmgt_config.json",
            checkpoint_path.with_suffix('.json'),  # same name as checkpoint
        ]
        
        self.config = None
        for config_path in config_paths:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
                print(f"Loaded config from {config_path.name}: {self.config['model']['num_nodes']} nodes, {self.config['model']['mixture_components']}K mixtures")
                break
        
        if self.config is None:
            print("Config file not found, will detect from checkpoint")
        
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
        # Try multiple possible locations
        possible_topology_paths = [
            Path("cache/overpass_topology.json"),  # Project root
            self.checkpoint_path.parents[2] / "cache" / "overpass_topology.json",  # From checkpoint
        ]
        
        self.node_metadata = {}
        topology_loaded = False
        
        for topology_path in possible_topology_paths:
            if topology_path.exists():
                try:
                    with open(topology_path, 'r', encoding='utf-8') as f:
                        topology = json.load(f)
                        for node in topology.get('nodes', []):
                            node_id = node.get('node_id')
                            if node_id:
                                self.node_metadata[node_id] = node
                    print(f"✓ Loaded {len(self.node_metadata)} node metadata from {topology_path}")
                    topology_loaded = True
                    break
                except Exception as e:
                    print(f"Failed to load topology from {topology_path}: {e}")
        
        if not topology_loaded:
            print("WARNING: Node metadata not loaded. Map will show coordinates as (0,0)")

        
        # Verify node count matches config
        if self.config and self.config['model']['num_nodes'] != self.num_nodes:
            print(f"WARNING: Model expects {self.config['model']['num_nodes']} nodes but data has {self.num_nodes}")
            print(f"This may cause incorrect predictions!")
    
    def _load_model(self) -> STMGT:
        """Load STMGT model from checkpoint."""
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        # Priority 1: Use config.json from checkpoint directory (loaded in __init__)
        if self.config and 'model' in self.config:
            print("Using model config from config.json")
            saved_config = self.config['model']
            model_config = {
                'num_nodes': saved_config.get('num_nodes', 62),
                'in_dim': saved_config.get('in_dim', 1),
                'hidden_dim': saved_config.get('hidden_dim', 96),
                'num_blocks': saved_config.get('num_blocks', 3),
                'num_heads': saved_config.get('num_heads', 4),
                'dropout': saved_config.get('dropout', 0.2),
                'drop_edge_rate': saved_config.get('drop_edge_rate', 0.05),
                'mixture_components': saved_config.get('mixture_components', 3),
                'seq_len': saved_config.get('seq_len', 12),
                'pred_len': saved_config.get('pred_len', 12),
            }
        # Priority 2: Extract model config from checkpoint if embedded
        elif 'config' in checkpoint and 'model' in checkpoint['config']:
            print("Using model config from checkpoint")
            saved_config = checkpoint['config']['model']
            model_config = {
                'num_nodes': saved_config.get('num_nodes', 62),
                'in_dim': saved_config.get('in_dim', 1),
                'hidden_dim': saved_config.get('hidden_dim', 96),
                'num_blocks': saved_config.get('num_blocks', 3),
                'num_heads': saved_config.get('num_heads', 4),
                'dropout': saved_config.get('dropout', 0.2),
                'drop_edge_rate': saved_config.get('drop_edge_rate', 0.05),
                'mixture_components': saved_config.get('mixture_components', 3),
                'seq_len': saved_config.get('seq_len', 12),
                'pred_len': saved_config.get('pred_len', 12),
            }
        else:
            # Detect config from checkpoint state_dict shapes
            print("Config not found in checkpoint, detecting from state_dict...")
            state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
            
            # Detect num_blocks from keys
            num_blocks = 0
            for key in state_dict.keys():
                if 'st_blocks.' in key:
                    block_idx = int(key.split('st_blocks.')[1].split('.')[0])
                    num_blocks = max(num_blocks, block_idx + 1)
            
            # Detect num_heads from attention shape
            # att shape: [1, num_heads, hidden_dim]
            if 'st_blocks.0.gat.att' in state_dict:
                num_heads = state_dict['st_blocks.0.gat.att'].shape[1]
            else:
                num_heads = 4
            
            # Detect pred_len and mixture_components from output head
            # mu_head output: [mixture_components * pred_len, hidden_dim]
            # Standard configs: K=3 mixtures, pred_len in {8, 12}
            if 'output_head.mu_head.weight' in state_dict:
                out_dim = state_dict['output_head.mu_head.weight'].shape[0]
                # Try K=3 first (most common)
                for K in [3, 2, 4]:
                    if out_dim % K == 0:
                        pred_len = out_dim // K
                        if pred_len in [6, 8, 12, 16]:  # Reasonable horizons
                            mixture_components = K
                            break
                else:
                    # Fallback
                    mixture_components = 3
                    pred_len = out_dim // mixture_components
            else:
                mixture_components = 3
                pred_len = 12
            
            # Detect hidden_dim from traffic_encoder output size
            # traffic_encoder.weight: [hidden_dim, 1]
            hidden_dim = 96  # default
            if 'traffic_encoder.weight' in state_dict:
                hidden_dim = state_dict['traffic_encoder.weight'].shape[0]
            
            print(f"Detected: hidden_dim={hidden_dim}, num_blocks={num_blocks}, num_heads={num_heads}, mixture_components={mixture_components}, pred_len={pred_len}")
            
            model_config = {
                'num_nodes': 62,
                'in_dim': 1,
                'hidden_dim': hidden_dim,
                'num_blocks': num_blocks,
                'num_heads': num_heads,
                'dropout': 0.2,
                'drop_edge_rate': 0.05,
                'mixture_components': mixture_components,
                'seq_len': 12,
                'pred_len': pred_len,
            }
        
        print(f"Model config: {model_config}")
        
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
            
            # Sort by run_id to get chronological order
            df = df.sort_values(['run_id', 'timestamp'])
            
            # Get latest 12 runs (each run = 1 timestep)
            unique_runs = df['run_id'].unique()
            latest_runs = unique_runs[-12:]  # Last 12 runs
            df_latest = df[df['run_id'].isin(latest_runs)]
            
            print(f"  Using {len(latest_runs)} most recent runs for historical data")
            
            # Initialize arrays
            self.historical_speeds = np.zeros((self.num_nodes, 12))
            self.historical_weather = np.zeros((self.num_nodes, 12, 3))
            
            # Build speed matrix: (node × timestep)
            # Each timestep = 1 run, aggregate edges to get node-level speed
            for t_idx, run_id in enumerate(latest_runs):
                run_df = df_latest[df_latest['run_id'] == run_id]
                
                # For each edge in this run, map to node
                for _, row in run_df.iterrows():
                    node_a_id = row['node_a_id']
                    node_b_id = row['node_b_id']
                    speed = row['speed_kmh']
                    
                    # Use node_a as representative (edge start point)
                    if node_a_id in self.node_to_idx:
                        node_idx = self.node_to_idx[node_a_id]
                        self.historical_speeds[node_idx, t_idx] = speed
                    
                    # Weather data (global, same for all nodes)
                    if 'temperature_c' in row:
                        self.historical_weather[node_idx, t_idx, 0] = row.get('temperature_c', 0)
                    if 'wind_speed_kmh' in row:
                        self.historical_weather[node_idx, t_idx, 1] = row.get('wind_speed_kmh', 0)
                    if 'precipitation_mm' in row:
                        self.historical_weather[node_idx, t_idx, 2] = row.get('precipitation_mm', 0)
            
            # Fill any missing nodes with mean values
            for i in range(self.num_nodes):
                if self.historical_speeds[i].sum() == 0:
                    self.historical_speeds[i] = self.historical_speeds.mean(axis=0)
                if self.historical_weather[i].sum() == 0:
                    self.historical_weather[i] = self.historical_weather.mean(axis=0)
            
            print(f"✓ Loaded real data from {self.data_path.name}")
            print(f"  Run range: {latest_runs[0]} to {latest_runs[-1]}")
            print(f"  Speed range: {self.historical_speeds.min():.1f} - {self.historical_speeds.max():.1f} km/h")
            print(f"  Mean speed: {self.historical_speeds.mean():.1f} km/h")
            print(f"  Speed variance per node: {self.historical_speeds.std(axis=1).mean():.2f} km/h")
            
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
        
        # Denormalize predictions to km/h (model outputs normalized values)
        pred_mean_denorm = self.model.speed_normalizer.denormalize(pred_mean.unsqueeze(-1)).squeeze(-1)
        pred_std_denorm = pred_std * self.model.speed_normalizer.std  # Scale std by normalizer std
        
        # Move to CPU and convert to numpy
        pred_mean = pred_mean_denorm.squeeze(0).cpu().numpy()  # (N, T)
        pred_std = pred_std_denorm.squeeze(0).cpu().numpy()    # (N, T)
        
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
    
    def get_current_traffic(self) -> list[dict]:
        """Get current traffic for all edges."""
        # Load latest data
        df = pd.read_parquet(self.data_path)
        
        # Get most recent timestamp
        latest_time = df['timestamp'].max()
        current_data = df[df['timestamp'] == latest_time].copy()
        
        # Create edge list with coordinates
        edges = []
        for _, row in current_data.iterrows():
            node_a = row['node_a_id']
            node_b = row['node_b_id']
            
            # Get node coordinates (if available)
            node_a_info = self.node_metadata.get(node_a, {})
            node_b_info = self.node_metadata.get(node_b, {})
            
            edges.append({
                'edge_id': f"{node_a}_{node_b}",
                'node_a_id': node_a,
                'node_b_id': node_b,
                'speed_kmh': float(row.get('speed', row.get('speed_kmh', 0))),
                'timestamp': row['timestamp'],
                'lat_a': float(node_a_info.get('lat', 0)),
                'lon_a': float(node_a_info.get('lon', 0)),
                'lat_b': float(node_b_info.get('lat', 0)),
                'lon_b': float(node_b_info.get('lon', 0))
            })
        
        return edges
    
    def predict_edge(self, edge_id: str, horizon: int = 12) -> dict:
        """
        Predict speed for specific edge using node-level predictions.
        
        Edge speed is derived from source node (node_a) predictions,
        representing traffic leaving that node.
        
        Args:
            edge_id: Format "node_a_id_node_b_id"
            horizon: Forecast horizon in timesteps (1-12)
        
        Returns:
            Dictionary with edge-specific predictions
        """
        # Parse edge_id
        parts = edge_id.split('_')
        if len(parts) != 2:
            raise ValueError(f"Invalid edge_id format: {edge_id}. Expected 'node_a_node_b'")
        
        node_a_id, node_b_id = parts
        
        if node_a_id not in self.node_to_idx or node_b_id not in self.node_to_idx:
            raise ValueError(f"Unknown nodes in edge: {edge_id}")
        
        # Get node-level predictions (use node_a as representative)
        node_predictions = self.predict(
            timestamp=datetime.now(),
            node_ids=[node_a_id],
            horizons=[horizon]
        )
        
        if not node_predictions['nodes']:
            raise ValueError(f"Failed to generate prediction for node {node_a_id}")
        
        node_pred = node_predictions['nodes'][0]
        forecast = node_pred['forecasts'][0] if node_pred['forecasts'] else None
        
        if not forecast:
            raise ValueError(f"No forecast available for horizon {horizon}")
        
        # Load current speed from data
        try:
            df = pd.read_parquet(self.data_path)
            edge_data = df[
                (df['node_a_id'] == node_a_id) & 
                (df['node_b_id'] == node_b_id)
            ].tail(1)
            
            current_speed = float(edge_data.iloc[0]['speed_kmh']) if not edge_data.empty else None
        except Exception as e:
            print(f"Warning: Could not load current speed for edge {edge_id}: {e}")
            current_speed = None
        
        return {
            'edge_id': edge_id,
            'node_a_id': node_a_id,
            'node_b_id': node_b_id,
            'horizon': horizon,
            'horizon_minutes': horizon * 15,
            'predicted_speed_kmh': forecast['mean'],
            'uncertainty_std': forecast['std'],
            'confidence_80_lower': forecast['lower_80'],
            'confidence_80_upper': forecast['upper_80'],
            'current_speed_kmh': current_speed,
            'timestamp': datetime.now(),
            'model_version': 'stmgt_v3'
        }
    
    def plan_routes(
        self,
        start_node_id: str,
        end_node_id: str,
        departure_time: datetime
    ) -> list[dict]:
        """
        Plan 3 route options from start to end.
        
        Returns:
            List of 3 routes: fastest, shortest, balanced
        """
        if start_node_id not in self.node_to_idx:
            raise ValueError(f"Unknown start node: {start_node_id}")
        if end_node_id not in self.node_to_idx:
            raise ValueError(f"Unknown end node: {end_node_id}")
        
        # Load edge data
        df = pd.read_parquet(self.data_path)
        
        # Build graph from edges
        import networkx as nx
        G = nx.DiGraph()
        
        for _, row in df[['node_a_id', 'node_b_id', 'speed_kmh']].drop_duplicates().iterrows():
            node_a = row['node_a_id']
            node_b = row['node_b_id']
            speed = float(row.get('speed_kmh', row.get('speed', 30)))
            
            # Weight = 1/speed (inverse for shortest path = fastest)
            weight = 1.0 / max(speed, 1.0)
            G.add_edge(node_a, node_b, weight=weight, speed=speed)
        
        # Find 3 routes
        routes = []
        
        try:
            # 1. Fastest route (based on predicted speeds)
            fastest_path = nx.shortest_path(G, start_node_id, end_node_id, weight='weight')
            fastest_route = self._path_to_route(fastest_path, G, 'fastest')
            routes.append(fastest_route)
            
            # 2. Shortest route (fewest hops)
            shortest_path = nx.shortest_path(G, start_node_id, end_node_id)
            shortest_route = self._path_to_route(shortest_path, G, 'shortest')
            routes.append(shortest_route)
            
            # 3. Balanced route (compromise)
            # Use weighted average of distance and time
            balanced_path = fastest_path  # For now, same as fastest
            balanced_route = self._path_to_route(balanced_path, G, 'balanced')
            routes.append(balanced_route)
            
        except nx.NetworkXNoPath:
            raise ValueError(f"No path found from {start_node_id} to {end_node_id}")
        
        return routes
    
    def _path_to_route(self, path: list[str], G: 'nx.DiGraph', route_type: str) -> dict:
        """Convert networkx path to route format."""
        segments = []
        total_distance = 0
        total_time = 0
        total_uncertainty = 0
        
        for i in range(len(path) - 1):
            node_a = path[i]
            node_b = path[i + 1]
            
            edge_data = G.get_edge_data(node_a, node_b)
            speed = edge_data.get('speed', 30.0)
            
            # Assume 1 km per segment (placeholder)
            distance = 1.0
            travel_time = (distance / speed) * 60  # minutes
            uncertainty = 0.15 * travel_time  # 15% uncertainty
            
            segments.append({
                'edge_id': f"{node_a}_{node_b}",
                'node_a_id': node_a,
                'node_b_id': node_b,
                'distance_km': distance,
                'predicted_speed_kmh': speed,
                'predicted_travel_time_min': travel_time,
                'uncertainty_std': uncertainty
            })
            
            total_distance += distance
            total_time += travel_time
            total_uncertainty += uncertainty ** 2
        
        total_uncertainty = np.sqrt(total_uncertainty)
        
        return {
            'route_type': route_type,
            'segments': segments,
            'total_distance_km': total_distance,
            'expected_travel_time_min': total_time,
            'travel_time_uncertainty_min': total_uncertainty,
            'confidence_level': 0.8  # 80% confidence
        }
