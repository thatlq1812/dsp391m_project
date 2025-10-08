"""
Normalize pipeline: Aggregate traffic edges to per-node speeds.
"""

import json
import os
from collections import defaultdict
from datetime import datetime

def load_traffic_edges():
    with open('data/traffic_edges.json', 'r') as f:
        return json.load(f)

def aggregate_to_nodes(edges):
    """Aggregate edge speeds to node-level averages."""
    node_speeds = defaultdict(list)
    for edge in edges:
        node_speeds[edge['node_a_id']].append(edge['speed_kmh'])
        node_speeds[edge['node_b_id']].append(edge['speed_kmh'])
    
    snapshots = []
    for node_id, speeds in node_speeds.items():
        avg_speed = sum(speeds) / len(speeds)
        snapshots.append({
            'node_id': node_id,
            'timestamp': datetime.now().isoformat(),
            'avg_speed_kmh': avg_speed,
            'vehicle_count': len(speeds) * 10  # Mock
        })
    return snapshots

def run_normalize():
    edges = load_traffic_edges()
    snapshots = aggregate_to_nodes(edges)
    
    with open('data/traffic_snapshot_normalized.json', 'w') as f:
        json.dump(snapshots, f, indent=2)
    
    print(f"Normalized to {len(snapshots)} node snapshots.")

if __name__ == "__main__":
    run_normalize()