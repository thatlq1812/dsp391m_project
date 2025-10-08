#!/usr/bin/env python3
"""
Check number of edges generated
"""

import sys
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[0]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import os
import yaml
from collectors.google.collector import load_nodes, find_nearest_neighbors

config = yaml.safe_load(open('configs/project_config.yaml', 'r'))
nodes = load_nodes()
print(f'Loaded {len(nodes)} nodes')

edges = find_nearest_neighbors(nodes, config)
print(f'Generated {len(edges)} edges')

# Sample first few edges
for i, (node_a, node_b, dist) in enumerate(edges[:5]):
    print(f'Edge {i}: {node_a["node_id"]} -> {node_b["node_id"]} ({dist:.2f}km)')