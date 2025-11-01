#!/usr/bin/env python3
"""
Quick utility to inspect generated edges.
"""

import yaml

from traffic_forecast import PROJECT_ROOT
from traffic_forecast.collectors.google.collector import load_nodes, find_nearest_neighbors


def main() -> None:
 config_path = PROJECT_ROOT / "configs" / "project_config.yaml"
 with config_path.open(encoding="utf-8") as fh:
  config = yaml.safe_load(fh) or {}
 nodes = load_nodes()
 print(f"Loaded {len(nodes)} nodes")

 edges = find_nearest_neighbors(nodes, config)
 print(f"Generated {len(edges)} edges")

 for idx, (node_a, node_b, dist) in enumerate(edges[:5]):
  print(f"Edge {idx}: {node_a['node_id']} -> {node_b['node_id']} ({dist:.2f}km)")


if __name__ == "__main__":
 main()
