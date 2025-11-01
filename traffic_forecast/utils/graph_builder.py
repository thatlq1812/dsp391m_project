"""
Graph Builder - Build adjacency matrix from road network topology

Author: thatlq1812
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build graph structure and adjacency matrix from road network topology."""
    
    def __init__(self, topology_path: Optional[Path] = None):
        """
        Initialize graph builder.
        
        Args:
            topology_path: Path to topology JSON file (default: cache/overpass_topology.json)
        """
        self.topology_path = topology_path
        if self.topology_path is None:
            from traffic_forecast import PROJECT_ROOT
            self.topology_path = PROJECT_ROOT / 'cache' / 'overpass_topology.json'
        
        self.nodes = []
        self.edges = []
        self.node_id_to_idx = {}
        self.adjacency_matrix = None
        
    def load_topology(self) -> Dict:
        """Load topology from cache file."""
        if not self.topology_path.exists():
            raise FileNotFoundError(f"Topology file not found: {self.topology_path}")
        
        with open(self.topology_path, 'r') as f:
            topology = json.load(f)
        
        self.nodes = topology.get('nodes', [])
        self.edges = topology.get('edges', [])
        
        logger.info(f"Loaded topology: {len(self.nodes)} nodes, {len(self.edges)} edges")
        
        return topology
    
    def build_node_mapping(self) -> Dict[str, int]:
        """
        Build mapping from node_id to index.
        
        Returns:
            Dictionary mapping node_id to integer index
        """
        self.node_id_to_idx = {
            node['node_id']: idx 
            for idx, node in enumerate(self.nodes)
        }
        
        logger.info(f"Built node mapping: {len(self.node_id_to_idx)} nodes")
        
        return self.node_id_to_idx
    
    def build_adjacency_matrix(
        self, 
        method: str = 'distance',
        normalize: bool = True,
        self_loops: bool = True
    ) -> np.ndarray:
        """
        Build adjacency matrix from road network.
        
        Args:
            method: Method to compute edge weights
                - 'binary': 1 if connected, 0 otherwise
                - 'distance': Inverse distance weighting
                - 'speed': Based on typical speed (from edges)
            normalize: Whether to normalize adjacency matrix
            self_loops: Whether to add self-loops (identity)
            
        Returns:
            Adjacency matrix (num_nodes, num_nodes)
        """
        num_nodes = len(self.nodes)
        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        
        # Load topology if not loaded
        if not self.nodes:
            self.load_topology()
        
        # Build node mapping if not built
        if not self.node_id_to_idx:
            self.build_node_mapping()
        
        # Build adjacency from topology edges
        edge_count = 0
        for edge in self.edges:
            node_a = edge.get('source')
            node_b = edge.get('target')
            
            if node_a not in self.node_id_to_idx or node_b not in self.node_id_to_idx:
                continue
            
            idx_a = self.node_id_to_idx[node_a]
            idx_b = self.node_id_to_idx[node_b]
            
            # Compute weight based on method
            if method == 'binary':
                weight = 1.0
            elif method == 'distance':
                # Inverse distance - closer nodes have higher weight
                distance = edge.get('distance', 1.0)
                weight = 1.0 / (distance + 1e-6)
            elif method == 'speed':
                # Based on typical travel speed
                speed = edge.get('typical_speed', 30.0)
                weight = speed / 50.0  # Normalize by typical urban speed
            else:
                weight = 1.0
            
            # Undirected graph - add both directions
            adj[idx_a, idx_b] = weight
            adj[idx_b, idx_a] = weight
            edge_count += 1
        
        logger.info(f"Built adjacency matrix from {edge_count} edges")
        
        # Add self-loops
        if self_loops:
            np.fill_diagonal(adj, 1.0)
            logger.info("Added self-loops to adjacency matrix")
        
        # Normalize
        if normalize:
            adj = self._normalize_adjacency(adj)
            logger.info("Normalized adjacency matrix")
        
        self.adjacency_matrix = adj
        
        return adj
    
    def _normalize_adjacency(self, adj: np.ndarray) -> np.ndarray:
        """
        Normalize adjacency matrix using symmetric normalization.
        A_norm = D^(-1/2) * A * D^(-1/2)
        
        Args:
            adj: Adjacency matrix
            
        Returns:
            Normalized adjacency matrix
        """
        # Compute degree matrix
        degree = np.sum(adj, axis=1)
        
        # Avoid division by zero
        degree[degree == 0] = 1.0
        
        # D^(-1/2)
        d_inv_sqrt = np.power(degree, -0.5)
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        
        # Normalize: D^(-1/2) * A * D^(-1/2)
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        adj_normalized = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        
        return adj_normalized
    
    def build_from_traffic_data(
        self,
        traffic_edges: List[Dict],
        threshold_distance: float = 5.0,
        method: str = 'distance'
    ) -> np.ndarray:
        """
        Build adjacency matrix from traffic edge data.
        
        This is useful when topology edges are limited but we have traffic data.
        
        Args:
            traffic_edges: List of traffic edge dictionaries
            threshold_distance: Maximum distance (km) to consider as connected
            method: Method to compute weights
            
        Returns:
            Adjacency matrix
        """
        # Extract unique nodes from traffic edges
        node_ids = set()
        for edge in traffic_edges:
            node_ids.add(edge.get('node_a_id'))
            node_ids.add(edge.get('node_b_id'))
        
        node_ids = sorted(list(node_ids))
        num_nodes = len(node_ids)
        
        # Build node mapping
        self.node_id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
        
        # Initialize adjacency matrix
        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        
        # Fill adjacency from traffic edges
        for edge in traffic_edges:
            node_a = edge.get('node_a_id')
            node_b = edge.get('node_b_id')
            distance = edge.get('distance_km', 0)
            
            if distance > threshold_distance:
                continue
            
            idx_a = self.node_id_to_idx[node_a]
            idx_b = self.node_id_to_idx[node_b]
            
            # Compute weight
            if method == 'binary':
                weight = 1.0
            elif method == 'distance':
                weight = 1.0 / (distance + 1e-6)
            elif method == 'speed':
                speed = edge.get('speed_kmh', 30.0)
                weight = speed / 50.0
            else:
                weight = 1.0
            
            adj[idx_a, idx_b] = weight
            adj[idx_b, idx_a] = weight
        
        # Add self-loops and normalize
        np.fill_diagonal(adj, 1.0)
        adj = self._normalize_adjacency(adj)
        
        self.adjacency_matrix = adj
        
        logger.info(f"Built adjacency matrix from traffic data: {num_nodes} nodes")
        
        return adj
    
    def get_node_features(self) -> pd.DataFrame:
        """
        Extract node features for model input.
        
        Returns:
            DataFrame with node features
        """
        if not self.nodes:
            self.load_topology()
        
        features = []
        for node in self.nodes:
            features.append({
                'node_id': node['node_id'],
                'lat': node.get('lat', 0),
                'lon': node.get('lon', 0),
                'degree': node.get('degree', 0),
                'importance_score': node.get('importance_score', 0),
                'is_major_intersection': int(node.get('is_major_intersection', False))
            })
        
        df = pd.DataFrame(features)
        
        # Set node_id as index
        df = df.set_index('node_id')
        
        return df
    
    def save_adjacency(self, output_path: Path):
        """Save adjacency matrix to file."""
        if self.adjacency_matrix is None:
            raise ValueError("Adjacency matrix not built. Call build_adjacency_matrix() first.")
        
        np.save(output_path, self.adjacency_matrix)
        logger.info(f"Saved adjacency matrix to {output_path}")
    
    def load_adjacency(self, input_path: Path) -> np.ndarray:
        """Load adjacency matrix from file."""
        self.adjacency_matrix = np.load(input_path)
        logger.info(f"Loaded adjacency matrix from {input_path}: shape {self.adjacency_matrix.shape}")
        return self.adjacency_matrix


def build_adjacency_from_runs(
    runs_dir: Path,
    output_path: Optional[Path] = None,
    method: str = 'distance'
) -> np.ndarray:
    """
    Build adjacency matrix from all traffic edge data in runs.
    
    Args:
        runs_dir: Directory containing run folders
        output_path: Path to save adjacency matrix (optional)
        method: Method to compute weights
        
    Returns:
        Adjacency matrix
    """
    # Collect all traffic edges from all runs
    all_edges = []
    
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        
        traffic_file = run_dir / 'traffic_edges.json'
        if not traffic_file.exists():
            continue
        
        with open(traffic_file, 'r') as f:
            edges = json.load(f)
            all_edges.extend(edges)
    
    logger.info(f"Collected {len(all_edges)} traffic edges from {runs_dir}")
    
    # Build graph
    builder = GraphBuilder()
    adj = builder.build_from_traffic_data(all_edges, method=method)
    
    # Save if requested
    if output_path:
        builder.save_adjacency(output_path)
    
    return adj


if __name__ == '__main__':
    """Test graph builder"""
    from traffic_forecast import PROJECT_ROOT
    
    logging.basicConfig(level=logging.INFO)
    
    # Build from topology
    builder = GraphBuilder()
    builder.load_topology()
    adj = builder.build_adjacency_matrix(method='distance')
    
    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"Non-zero entries: {np.count_nonzero(adj)}")
    print(f"Density: {np.count_nonzero(adj) / adj.size:.4f}")
    
    # Save
    output_path = PROJECT_ROOT / 'cache' / 'adjacency_matrix.npy'
    builder.save_adjacency(output_path)
    
    print(f"\nAdjacency matrix saved to {output_path}")
