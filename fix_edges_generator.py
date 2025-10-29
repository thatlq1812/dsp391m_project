"""
Quick fix to generate edges from nodes using nearest neighbors
"""
import json
import math
from pathlib import Path

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in meters"""
    R = 6371000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def generate_edges_from_nodes(nodes, k_neighbors=8, max_distance_m=1000):
    """
    Generate edges by connecting each node to its k nearest neighbors
    
    Args:
        nodes: List of node dicts with lat, lon, node_id
        k_neighbors: Number of neighbors to connect to
        max_distance_m: Maximum distance for edges (meters)
    
    Returns:
        List of edge dicts
    """
    edges = []
    seen_pairs = set()
    
    print(f"Generating edges for {len(nodes)} nodes...")
    print(f"  k_neighbors: {k_neighbors}")
    print(f"  max_distance: {max_distance_m}m")
    
    for i, node_a in enumerate(nodes):
        # Find all neighbors within max distance
        neighbors = []
        for j, node_b in enumerate(nodes):
            if i == j:
                continue
                
            dist_m = haversine(node_a['lat'], node_a['lon'], node_b['lat'], node_b['lon'])
            
            if dist_m <= max_distance_m:
                neighbors.append((dist_m, j, node_b))
        
        # Sort by distance and take k nearest
        neighbors.sort(key=lambda x: x[0])
        neighbors = neighbors[:k_neighbors]
        
        # Create edges
        for dist_m, j, node_b in neighbors:
            # Avoid duplicates
            pair = tuple(sorted([node_a['node_id'], node_b['node_id']]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            
            # Determine road type and name from node metadata
            road_type = node_a.get('road_type', 'unknown')
            
            # Try to find a common street name
            street_names_a = set(node_a.get('street_names', []))
            street_names_b = set(node_b.get('street_names', []))
            common_names = street_names_a.intersection(street_names_b)
            
            if common_names:
                road_name = list(common_names)[0]
            elif street_names_a:
                road_name = list(street_names_a)[0]
            elif street_names_b:
                road_name = list(street_names_b)[0]
            else:
                road_name = f"Connecting road {node_a['node_id'][:15]}"
            
            edge = {
                'edge_id': f"{node_a['node_id']}-{node_b['node_id']}",
                'start_node_id': node_a['node_id'],
                'end_node_id': node_b['node_id'],
                'distance_m': dist_m,
                'road_type': road_type,
                'road_name': road_name,
                'way_id': node_a.get('way_ids', [None])[0]  # Use first way_id if available
            }
            edges.append(edge)
    
    print(f"Generated {len(edges)} edges")
    return edges


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python fix_edges_generator.py <path_to_nodes.json>")
        sys.exit(1)
    
    nodes_file = Path(sys.argv[1])
    edges_file = nodes_file.parent / "edges.json"
    
    print(f"Loading nodes from: {nodes_file}")
    with open(nodes_file, 'r', encoding='utf-8') as f:
        nodes = json.load(f)
    
    # Generate edges
    edges = generate_edges_from_nodes(nodes, k_neighbors=8, max_distance_m=1000)
    
    # Save edges
    print(f"Saving edges to: {edges_file}")
    with open(edges_file, 'w', encoding='utf-8') as f:
        json.dump(edges, f, indent=2)
    
    print("Done!")
    print(f"  Nodes: {len(nodes)}")
    print(f"  Edges: {len(edges)}")
    print(f"  Avg edges per node: {len(edges) * 2 / len(nodes):.1f}")
