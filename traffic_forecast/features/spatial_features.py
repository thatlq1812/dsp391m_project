"""
Spatial features: neighbor aggregation, congestion propagation.

These features capture spatial patterns:
- Traffic congestion spreads through network
- Upstream nodes affect downstream nodes
- Neighbor node speeds correlate with target node
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def build_adjacency_graph(nodes_data: List[dict]) -> Dict[int, Set[int]]:
    """
    Build adjacency graph from nodes data.

    Each node has connected_ways which contain adjacent nodes.

    Args:
    nodes_data: List of node dictionaries with connected_ways

    Returns:
    Dict mapping node_id -> set of adjacent node_ids

    Example:
    {
    123: {456, 789}, # Node 123 connects to 456 and 789
    456: {123, 999},
    ...
    }
    """
    graph = {}

    for node in nodes_data:
    node_id = node['node_id']

    # Get connected nodes from ways
    neighbors = set()
    for way in node.get('connected_ways', []):
        # Each way contains list of node IDs
    way_nodes = way.get('nodes', [])
    if node_id in way_nodes:
        # Add all other nodes in this way as neighbors
    neighbors.update([n for n in way_nodes if n != node_id])

    graph[node_id] = neighbors

    logger.info(f"Built graph with {len(graph)} nodes")
    return graph


def get_k_hop_neighbors(
    graph: Dict[int, Set[int]],
    node_id: int,
    k: int = 1
) -> Set[int]:
    """
    Get k-hop neighbors of a node.

    Args:
    graph: Adjacency graph
    node_id: Target node ID
    k: Number of hops (default=1 for direct neighbors)

    Returns:
    Set of node IDs within k hops

    Example (k=1):
    Node A connects to B, C
    Returns: {B, C}

    Example (k=2):
    Node A connects to B, C
    Node B connects to D
    Returns: {B, C, D}
    """
    if node_id not in graph:
    return set()

    neighbors = set()
    current_level = {node_id}

    for _ in range(k):
    next_level = set()
    for node in current_level:
    if node in graph:
    next_level.update(graph[node])
    neighbors.update(next_level)
    current_level = next_level

    # Remove the node itself
    neighbors.discard(node_id)

    return neighbors


def add_neighbor_features(
    df: pd.DataFrame,
    graph: Dict[int, Set[int]],
    value_column: str = 'avg_speed_kmh',
    neighbor_hops: int = 1,
    prefix: str = 'neighbor'
) -> pd.DataFrame:
    """
    Add neighbor aggregation features.

    For each node at each timestamp, aggregate values from neighbors:
    - Mean, min, max, std of neighbor speeds
    - Speed difference (self - neighbor_mean)
    - Congestion propagation indicators

    Args:
    df: DataFrame with traffic data (must have node_id, ts, value_column)
    graph: Adjacency graph from build_adjacency_graph()
    value_column: Column to aggregate (e.g., 'avg_speed_kmh')
    neighbor_hops: Number of hops for neighbors (1=direct, 2=second-degree)
    prefix: Prefix for new columns

    Returns:
    DataFrame with neighbor features
    """
    df = df.copy()
    df['ts'] = pd.to_datetime(df['ts'])

    logger.info(f"Adding {neighbor_hops}-hop neighbor features...")

    # Initialize columns
    df[f'{prefix}_avg_{value_column}'] = np.nan
    df[f'{prefix}_min_{value_column}'] = np.nan
    df[f'{prefix}_max_{value_column}'] = np.nan
    df[f'{prefix}_std_{value_column}'] = np.nan
    df[f'{prefix}_count'] = 0

    # Group by timestamp for efficient lookup
    grouped = df.groupby('ts')

    for ts, group in grouped:
        # Create lookup dict: node_id -> value
    values_at_ts = dict(zip(group['node_id'], group[value_column]))

    # For each node at this timestamp
    for idx, row in group.iterrows():
    node_id = row['node_id']

    # Get neighbors
    neighbors = get_k_hop_neighbors(graph, node_id, k=neighbor_hops)

    # Get neighbor values at this timestamp
    neighbor_values = [
        values_at_ts[nid] for nid in neighbors
        if nid in values_at_ts and not pd.isna(values_at_ts[nid])
    ]

    if neighbor_values:
    df.loc[idx, f'{prefix}_avg_{value_column}'] = np.mean(neighbor_values)
    df.loc[idx, f'{prefix}_min_{value_column}'] = np.min(neighbor_values)
    df.loc[idx, f'{prefix}_max_{value_column}'] = np.max(neighbor_values)
    df.loc[idx, f'{prefix}_std_{value_column}'] = np.std(neighbor_values)
    df.loc[idx, f'{prefix}_count'] = len(neighbor_values)

    # Add derived features
    df[f'{prefix}_speed_diff'] = (
        df[value_column] - df[f'{prefix}_avg_{value_column}']
    )

    # Congestion indicator: if node is slower than neighbors
    df[f'{prefix}_is_bottleneck'] = (
        df[f'{prefix}_speed_diff'] < -5  # 5 km/h slower
    )

    logger.info(f"Created {6 + 1} neighbor features")  # 5 agg + 1 diff + 1 flag

    return df


def add_congestion_propagation(
    df: pd.DataFrame,
    graph: Dict[int, Set[int]],
    congestion_column: str = 'congestion_level'
) -> pd.DataFrame:
    """
    Add congestion propagation features.

    Congestion tends to spread through the network:
    - If upstream node is congested, downstream likely congested
    - Count of congested neighbors
    - Fraction of neighbors that are congested

    Args:
    df: DataFrame with traffic data
    graph: Adjacency graph
    congestion_column: Column with congestion level (0-3)

    Returns:
    DataFrame with congestion propagation features
    """
    df = df.copy()
    df['ts'] = pd.to_datetime(df['ts'])

    logger.info("Adding congestion propagation features...")

    # Initialize
    df['neighbor_congested_count'] = 0
    df['neighbor_congested_fraction'] = 0.0

    # Group by timestamp
    grouped = df.groupby('ts')

    for ts, group in grouped:
        # Lookup: node_id -> congestion_level
    congestion_at_ts = dict(zip(group['node_id'], group[congestion_column]))

    for idx, row in group.iterrows():
    node_id = row['node_id']
    neighbors = get_k_hop_neighbors(graph, node_id, k=1)

    # Count congested neighbors (congestion_level >= 2)
    neighbor_congestion = [
        congestion_at_ts[nid] for nid in neighbors
        if nid in congestion_at_ts
    ]

    if neighbor_congestion:
    congested = sum(c >= 2 for c in neighbor_congestion)
    df.loc[idx, 'neighbor_congested_count'] = congested
    df.loc[idx, 'neighbor_congested_fraction'] = (
        congested / len(neighbor_congestion)
    )

    logger.info("Created 2 congestion propagation features")

    return df


def add_spatial_features(
    df: pd.DataFrame,
    nodes_data: List[dict],
    config: dict = None
) -> pd.DataFrame:
    """
    Add all spatial features at once.

    Args:
    df: DataFrame with traffic data
    nodes_data: List of node dicts (for building graph)
    config: Configuration from project_config.yaml

    Returns:
    DataFrame with all spatial features
    """
    if config is None:
    config = {
        'enabled': True,
        'neighbor_hops': 1
    }

    if not config.get('enabled', True):
    logger.info("Spatial features disabled in config")
    return df

    logger.info("Creating all spatial features...")

    # Build graph
    graph = build_adjacency_graph(nodes_data)

    # Add neighbor features
    neighbor_hops = config.get('neighbor_hops', 1)
    df = add_neighbor_features(
        df, graph,
        value_column='avg_speed_kmh',
        neighbor_hops=neighbor_hops,
        prefix='neighbor'
    )

    # Add congestion propagation
    if 'congestion_level' in df.columns:
    df = add_congestion_propagation(df, graph)

    total_features = 7 + 2  # neighbor + congestion
    logger.info(f"Created total {total_features} spatial features")

    return df
