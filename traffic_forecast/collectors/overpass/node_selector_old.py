"""
Advanced node selection for traffic forecasting.
Selects only major intersections instead of dense spacing along all roads.
"""

import math
from typing import Dict, List, Tuple, Set
from collections import defaultdict


class NodeSelector:
    """
    Intelligent node selector that identifies major intersections
    based on road hierarchy and connectivity.
    Academic v4.0: Supports max_nodes limit and road_type filtering
    """

    # Road importance weights (higher = more important)
    ROAD_WEIGHTS = {
        'motorway': 10,
        'trunk': 9,
        'primary': 8,
        'secondary': 7,
        'tertiary': 5,
        'residential': 2,
        'unclassified': 1
    }

    def __init__(
        self,
        min_degree: int = 3,
        min_importance_score: float = 15.0,
        max_nodes: int = None,
        road_type_filter: List[str] = None
    ):
    """
 Args:
 min_degree: Minimum number of connecting roads for an intersection
 min_importance_score: Minimum importance score to be selected
 max_nodes: Maximum number of nodes to select (top N by importance)
 road_type_filter: List of allowed road types (e.g., ['motorway', 'trunk', 'primary'])
 """
    self.min_degree = min_degree
    self.min_importance_score = min_importance_score
    self.max_nodes = max_nodes
    self.road_type_filter = set(road_type_filter) if road_type_filter else None

    def extract_major_intersections(self, osm_data: dict) -> Tuple[List[dict], List[dict]]:
    """
 Extract only major intersections from OSM data.

 Returns:
 Tuple of (nodes, edges) where nodes are major intersections only
 """
    # Step 1: Build graph structure from ways
    node_connections = defaultdict(list)  # node_key -> [(way_id, road_type)]
    node_coords = {}  # node_key -> (lat, lon)
    way_nodes = {}  # way_id -> list of node_keys
    way_metadata = {}  # way_id -> tags

    for element in osm_data.get('elements', []):
    if element['type'] != 'way':
    continue

    way_id = element.get('id')
    tags = element.get('tags', {})
    road_type = tags.get('highway', 'unknown')
    geoms = element.get('geometry', [])

    if not geoms or road_type == 'unknown':
    continue

    way_metadata[way_id] = tags
    nodes_in_way = []

    # Extract actual intersection points (nodes where ways meet)
    for geom in geoms:
    lat, lon = geom['lat'], geom['lon']
    key = (round(lat, 6), round(lon, 6))
    node_coords[key] = (lat, lon)
    nodes_in_way.append(key)
    node_connections[key].append((way_id, road_type))

    way_nodes[way_id] = nodes_in_way

    # Step 2: Identify major intersections
    major_intersections = []
    seen_nodes = set()

    for node_key, connections in node_connections.items():
    degree = len(set(way_id for way_id, _ in connections))

    # Skip if not enough connections
    if degree < self.min_degree:
    continue

    # Calculate importance score based on connected road types
    importance_score = self._calculate_importance_score(connections)

    # Skip if not important enough
    if importance_score < self.min_importance_score:
    continue

    # Road type filtering (for academic v4.0 cost optimization)
    if self.road_type_filter:
        # Get primary road type for this intersection
    connected_types = set(road_type for _, road_type in connections)
    # Check if ANY of the connected roads match the filter
    if not connected_types.intersection(self.road_type_filter):
    continue

    lat, lon = node_coords[node_key]
    node_id = f"node-{lat:.6f}-{lon:.6f}"

    # Get road types of connected ways
    road_types = list(set(road_type for _, road_type in connections))
    primary_road_type = self._get_primary_road_type(road_types)

    # Extract street names from connected ways
    way_ids = list(set(way_id for way_id, _ in connections))
    street_names = []
    for way_id in way_ids:
    if way_id in way_metadata:
    tags = way_metadata[way_id]
    name = tags.get('name') or tags.get('ref') or tags.get('int_name')
    if name and name not in street_names:
    street_names.append(name)

    # Create intersection description
    if street_names:
    intersection_desc = " ∩ ".join(street_names[:3])  # Limit to 3 street names
    else:
    intersection_desc = f"{primary_road_type.title()} Intersection"

    major_intersections.append({
        'node_id': node_id,
        'lat': lat,
        'lon': lon,
        'degree': degree,
        'importance_score': importance_score,
        'road_type': primary_road_type,
        'connected_road_types': road_types,
        'street_names': street_names,
        'intersection_name': intersection_desc,
        'way_ids': way_ids,
        'is_major_intersection': True
    })

    seen_nodes.add(node_key)

    # Step 2.5: Apply max_nodes limit (academic v4.0 cost optimization)
    if self.max_nodes and len(major_intersections) > self.max_nodes:
        # Sort by importance score descending and take top N
    major_intersections.sort(key=lambda n: n['importance_score'], reverse=True)
    major_intersections = major_intersections[:self.max_nodes]
    print(f"NodeSelector: Limited to top {self.max_nodes} nodes by importance "
          f"(score range: {major_intersections[-1]['importance_score']:.1f} - "
          f"{major_intersections[0]['importance_score']:.1f})")

    # Step 3: Build edges between major intersections
    edges = self._build_edges(major_intersections, way_nodes, way_metadata, node_coords)

    return major_intersections, edges

    def _calculate_importance_score(self, connections: List[Tuple[str, str]]) -> float:
    """
 Calculate importance score based on connected road types.
 Higher score = more important intersection.
 """
    score = 0.0
    unique_road_types = set(road_type for _, road_type in connections)

    for road_type in unique_road_types:
    weight = self.ROAD_WEIGHTS.get(road_type, 1)
    score += weight

    # Bonus for diversity of road types
    diversity_bonus = len(unique_road_types) * 2
    score += diversity_bonus

    return score

    def _get_primary_road_type(self, road_types: List[str]) -> str:
    """Get the most important road type from a list."""
    if not road_types:
    return 'unknown'

    # Sort by weight descending
    sorted_types = sorted(
        road_types,
        key=lambda rt: self.ROAD_WEIGHTS.get(rt, 0),
        reverse=True
    )
    return sorted_types[0]

    def _build_edges(
        self,
        nodes: List[dict],
        way_nodes: Dict,
        way_metadata: Dict,
        node_coords: Dict
    ) -> List[dict]:
    """
 Build edges between major intersections.
 Only create edges if intersections are on the same way.
 """
    edges = []
    node_keys = {(n['lat'], n['lon']): n for n in nodes}
    edge_set = set()

    for way_id, node_sequence in way_nodes.items():
        # Find major intersections in this way
    major_nodes_in_way = [
        nk for nk in node_sequence
        if (round(nk[0], 6), round(nk[1], 6)) in node_keys
    ]

    # Create edges between consecutive major intersections
    for i in range(len(major_nodes_in_way) - 1):
    nk1 = (round(major_nodes_in_way[i][0], 6), round(major_nodes_in_way[i][1], 6))
    nk2 = (round(major_nodes_in_way[i + 1][0], 6), round(major_nodes_in_way[i + 1][1], 6))

    n1 = node_keys.get(nk1)
    n2 = node_keys.get(nk2)

    if not n1 or not n2:
    continue

    # Create unique edge key
    edge_key = tuple(sorted([n1['node_id'], n2['node_id']]))
    if edge_key in edge_set:
    continue

    edge_set.add(edge_key)

    # Calculate distance
    distance_km = self._haversine_km(
        n1['lat'], n1['lon'],
        n2['lat'], n2['lon']
    )

    tags = way_metadata.get(way_id, {})

    edges.append({
        'u': n1['node_id'],
        'v': n2['node_id'],
        'distance_m': distance_km * 1000.0,
        'way_id': way_id,
        'road_type': tags.get('highway', 'unknown'),
        'lanes': tags.get('lanes'),
        'maxspeed': tags.get('maxspeed'),
        'name': tags.get('name')
    })

    return edges

    @staticmethod
    def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate haversine distance in kilometers."""
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def get_statistics(self, nodes: List[dict], edges: List[dict]) -> Dict:
    """Get statistics about selected nodes and edges."""
    if not nodes:
    return {
        'total_nodes': 0,
        'total_edges': 0,
        'avg_degree': 0,
        'avg_importance': 0,
        'road_type_distribution': {}
    }

    total_nodes = len(nodes)
    total_edges = len(edges)
    avg_degree = sum(n['degree'] for n in nodes) / total_nodes
    avg_importance = sum(n['importance_score'] for n in nodes) / total_nodes

    # Road type distribution
    road_type_dist = defaultdict(int)
    for node in nodes:
    road_type_dist[node['road_type']] += 1

    return {
        'total_nodes': total_nodes,
        'total_edges': total_edges,
        'avg_degree': round(avg_degree, 2),
        'avg_importance': round(avg_importance, 2),
        'road_type_distribution': dict(road_type_dist),
        'min_degree': min(n['degree'] for n in nodes),
        'max_degree': max(n['degree'] for n in nodes),
        'min_importance': min(n['importance_score'] for n in nodes),
        'max_importance': max(n['importance_score'] for n in nodes)
    }
