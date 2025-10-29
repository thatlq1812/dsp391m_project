"""
Advanced node selection v5.0 with minimum distance filtering
Selects major intersections with distance constraints
"""

import math
from typing import Dict, List, Tuple, Set
from collections import defaultdict


class NodeSelector:
    """
    Intelligent node selector for major intersections
    v5.0: Added min_distance_meters to avoid clustered nodes
    """

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
        min_distance_meters: float = 200,
        road_type_filter: List[str] = None
    ):
        """
        Args:
            min_degree: Minimum number of connecting roads
            min_importance_score: Minimum importance score
            max_nodes: Maximum number of nodes (top N by importance)
            min_distance_meters: Minimum distance between nodes (default 200m)
            road_type_filter: Allowed road types
        """
        self.min_degree = min_degree
        self.min_importance_score = min_importance_score
        self.max_nodes = max_nodes
        self.min_distance_meters = min_distance_meters
        self.road_type_filter = set(road_type_filter) if road_type_filter else None

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in meters"""
        R = 6371000  # Earth radius in meters
        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        delta_phi = math.radians(lat2 - lat1)
        delta_lambda = math.radians(lon2 - lon1)

        a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    def _filter_by_distance(self, nodes: List[dict]) -> List[dict]:
        """
        Filter nodes to maintain minimum distance between them
        Keeps higher importance nodes when there's overlap
        """
        if not self.min_distance_meters or not nodes:
            return nodes

        # Sort by importance score descending
        sorted_nodes = sorted(nodes, key=lambda n: n['importance_score'], reverse=True)
        
        filtered_nodes = []
        for node in sorted_nodes:
            # Check distance to all already selected nodes
            too_close = False
            for selected_node in filtered_nodes:
                distance = self._haversine_distance(
                    node['lat'], node['lon'],
                    selected_node['lat'], selected_node['lon']
                )
                if distance < self.min_distance_meters:
                    too_close = True
                    break
            
            if not too_close:
                filtered_nodes.append(node)
        
        return filtered_nodes

    def extract_major_intersections(self, osm_data: dict) -> Tuple[List[dict], List[dict]]:
        """
        Extract major intersections from OSM data with distance filtering
        
        Returns:
            Tuple of (nodes, edges)
        """
        # Build graph structure from ways
        node_connections = defaultdict(list)
        node_coords = {}
        way_nodes = {}
        way_metadata = {}

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

            for geom in geoms:
                lat, lon = geom['lat'], geom['lon']
                key = (round(lat, 6), round(lon, 6))
                node_coords[key] = (lat, lon)
                nodes_in_way.append(key)
                node_connections[key].append((way_id, road_type))

            way_nodes[way_id] = nodes_in_way

        # Identify major intersections
        major_intersections = []

        for node_key, connections in node_connections.items():
            degree = len(set(way_id for way_id, _ in connections))

            if degree < self.min_degree:
                continue

            importance_score = self._calculate_importance_score(connections)

            if importance_score < self.min_importance_score:
                continue

            # Road type filtering
            if self.road_type_filter:
                connected_types = set(road_type for _, road_type in connections)
                if not connected_types.intersection(self.road_type_filter):
                    continue

            lat, lon = node_coords[node_key]
            node_id = f"node-{lat:.6f}-{lon:.6f}"

            road_types = list(set(road_type for _, road_type in connections))
            primary_road_type = self._get_primary_road_type(road_types)

            # Extract street names
            way_ids = list(set(way_id for way_id, _ in connections))
            street_names = []
            for way_id in way_ids:
                if way_id in way_metadata:
                    tags = way_metadata[way_id]
                    name = tags.get('name') or tags.get('ref') or tags.get('int_name')
                    if name and name not in street_names:
                        street_names.append(name)

            if street_names:
                intersection_desc = " - ".join(street_names[:3])
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

        # NEW: Filter by minimum distance
        if self.min_distance_meters:
            before_count = len(major_intersections)
            major_intersections = self._filter_by_distance(major_intersections)
            after_count = len(major_intersections)
            if before_count != after_count:
                print(f"Distance filtering: {before_count} -> {after_count} nodes (removed {before_count - after_count} clustered nodes)")

        # Apply max_nodes limit
        if self.max_nodes and len(major_intersections) > self.max_nodes:
            major_intersections.sort(key=lambda n: n['importance_score'], reverse=True)
            major_intersections = major_intersections[:self.max_nodes]
            print(f"Limited to top {self.max_nodes} nodes by importance "
                  f"(score range: {major_intersections[-1]['importance_score']:.1f} - "
                  f"{major_intersections[0]['importance_score']:.1f})")

        # Build edges
        edges = self._build_edges(major_intersections, way_nodes, way_metadata, node_coords)

        return major_intersections, edges

    def _calculate_importance_score(self, connections: List[Tuple[str, str]]) -> float:
        """Calculate importance score based on connected road types"""
        score = 0.0
        unique_road_types = set(road_type for _, road_type in connections)

        for road_type in unique_road_types:
            weight = self.ROAD_WEIGHTS.get(road_type, 1)
            score += weight

        diversity_bonus = len(unique_road_types) * 2
        score += diversity_bonus

        return score

    def _get_primary_road_type(self, road_types: List[str]) -> str:
        """Get the most important road type from a list"""
        if not road_types:
            return 'unknown'

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
        Build edges between major intersections using nearest neighbors approach
        
        This creates a well-connected graph by connecting each node to its k nearest
        neighbors within a maximum distance. This is more robust than relying solely
        on way sequences, especially when nodes are from different road segments.
        """
        edges = []
        seen_pairs = set()
        
        # Configuration for edge creation
        k_neighbors = 8  # Connect each node to 8 nearest neighbors
        max_distance_m = 1000  # Maximum edge length (1km)
        
        print(f"Building edges: k_neighbors={k_neighbors}, max_distance={max_distance_m}m")
        
        # For each node, find and connect to nearest neighbors
        for i, node_a in enumerate(nodes):
            # Find all nodes within max distance
            neighbors = []
            for j, node_b in enumerate(nodes):
                if i == j:
                    continue
                
                dist_m = self._haversine_distance(
                    node_a['lat'], node_a['lon'],
                    node_b['lat'], node_b['lon']
                )
                
                if dist_m <= max_distance_m:
                    neighbors.append((dist_m, node_b))
            
            # Sort by distance and take k nearest
            neighbors.sort(key=lambda x: x[0])
            neighbors = neighbors[:k_neighbors]
            
            # Create edges to nearest neighbors
            for dist_m, node_b in neighbors:
                # Avoid duplicate edges (undirected graph)
                pair = tuple(sorted([node_a['node_id'], node_b['node_id']]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                
                # Try to determine road information
                # First, check if nodes share common street names
                street_names_a = set(node_a.get('street_names', []))
                street_names_b = set(node_b.get('street_names', []))
                common_names = street_names_a.intersection(street_names_b)
                
                if common_names:
                    # Use common street name
                    road_name = list(common_names)[0]
                    # Try to find the way_id for this road
                    way_id = None
                    for wid in node_a.get('way_ids', []):
                        if wid in way_metadata:
                            tags = way_metadata[wid]
                            if tags.get('name') == road_name or tags.get('ref') == road_name:
                                way_id = wid
                                road_type = tags.get('highway', 'unknown')
                                break
                    if not way_id:
                        # Fallback to first way
                        way_id = node_a.get('way_ids', [None])[0]
                        road_type = node_a.get('road_type', 'unknown')
                else:
                    # No common street, use node_a's primary street
                    if street_names_a:
                        road_name = list(street_names_a)[0]
                    elif street_names_b:
                        road_name = list(street_names_b)[0]
                    else:
                        road_name = f"Connecting road"
                    
                    way_id = node_a.get('way_ids', [None])[0]
                    road_type = node_a.get('road_type', 'unknown')
                
                edges.append({
                    'edge_id': f"{node_a['node_id']}-{node_b['node_id']}",
                    'start_node_id': node_a['node_id'],
                    'end_node_id': node_b['node_id'],
                    'distance_m': dist_m,
                    'road_type': road_type,
                    'road_name': road_name,
                    'way_id': way_id
                })
        
        print(f"Built {len(edges)} edges from {len(nodes)} nodes (avg {len(edges)*2/len(nodes):.1f} edges/node)")
        return edges

    def get_statistics(self, nodes: List[dict], edges: List[dict]) -> dict:
        """Calculate statistics for selected nodes and edges"""
        if not nodes:
            return {
                'total_nodes': 0,
                'total_edges': 0,
                'avg_degree': 0,
                'avg_importance': 0,
                'road_type_distribution': {}
            }

        degrees = [n['degree'] for n in nodes]
        importances = [n['importance_score'] for n in nodes]
        road_types = [n['road_type'] for n in nodes]

        from collections import Counter
        road_type_dist = Counter(road_types)

        return {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'avg_degree': sum(degrees) / len(degrees),
            'avg_importance': sum(importances) / len(importances),
            'min_degree': min(degrees) if degrees else 0,
            'max_degree': max(degrees) if degrees else 0,
            'min_importance': min(importances) if importances else 0,
            'max_importance': max(importances) if importances else 0,
            'road_type_distribution': dict(road_type_dist),
            'min_distance_meters': self.min_distance_meters
        }
