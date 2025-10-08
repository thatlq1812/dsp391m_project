"""
Google Directions collector for traffic speeds.
Mock implementation since API key not provided.
"""

import json
import os
import random
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
import yaml

from collectors.area_utils import load_area_config

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km."""
    R = 6371  # Earth radius in km
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def load_nodes():
    with open('data/nodes.json', 'r') as f:
        return json.load(f)

def find_nearest_neighbors(nodes, config):
    """Find k nearest neighbors within radius for each node."""
    collector_config = config['collectors']['google']
    edges = []
    limit_nodes = collector_config.get('limit_nodes', len(nodes))
    for i, node_a in enumerate(nodes[:limit_nodes]):
        neighbors = []
        for j, node_b in enumerate(nodes):
            if i == j: continue
            dist = haversine(node_a['lat'], node_a['lon'], node_b['lat'], node_b['lon'])
            if dist <= collector_config['radius_km']:
                neighbors.append((dist, node_b))
        neighbors.sort(key=lambda x: x[0])
        for dist, node_b in neighbors[:collector_config['k_neighbors']]:
            edges.append((node_a, node_b, dist))
    return edges

def mock_directions_api(origin, dest, config):
    """Mock Google Directions API response."""
    collector_config = config['collectors']['google']
    dist_km = haversine(origin['lat'], origin['lon'], dest['lat'], dest['lon'])
    # Mock traffic: slower during peak hours
    base_speed = collector_config['base_speed_kmh']
    traffic_factor = random.uniform(collector_config['traffic_factor_min'], collector_config['traffic_factor_max'])
    speed = base_speed * traffic_factor
    duration_sec = (dist_km / speed) * 3600
    return {
        'distance_km': dist_km,
        'duration_sec': duration_sec,
        'speed_kmh': speed
    }

def run_google_collector():
    config = yaml.safe_load(open('configs/project_config.yaml', 'r'))
    collector_config = config['collectors']['google']

    # resolve area and filter nodes
    area_cfg = load_area_config('google')

    nodes = load_nodes()
    nodes = [n for n in nodes if (area_cfg['bbox'][0] <= n['lat'] <= area_cfg['bbox'][2] and area_cfg['bbox'][1] <= n['lon'] <= area_cfg['bbox'][3])]
    edges = find_nearest_neighbors(nodes, config)
    
    traffic_data = []
    for node_a, node_b, dist in edges:
        directions = mock_directions_api(node_a, node_b, config)
        traffic_data.append({
            'node_a_id': node_a['node_id'],
            'node_b_id': node_b['node_id'],
            'distance_km': directions['distance_km'],
            'duration_sec': directions['duration_sec'],
            'speed_kmh': directions['speed_kmh'],
            'timestamp': datetime.now().isoformat()
        })
    
    os.makedirs('data', exist_ok=True)
    output_file = collector_config.get('output', 'data/traffic_edges.json')
    with open(output_file, 'w') as f:
        json.dump(traffic_data, f, indent=2)
    
    print(f"Collected traffic for {len(traffic_data)} edges.")

if __name__ == "__main__":
    run_google_collector()