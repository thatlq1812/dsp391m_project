#!/usr/bin/env python3
"""
Test Google collector with limited edges.
"""

import json
import os
import yaml

from traffic_forecast import PROJECT_ROOT
from traffic_forecast.collectors.google.collector import (
    load_nodes,
    find_nearest_neighbors,
    mock_directions_api,
    real_directions_api,
    get_rate_limiter,
)


def test_limited_collection():
    config_path = PROJECT_ROOT / "configs" / "project_config.yaml"
    with config_path.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    nodes = load_nodes()
    print(f"Loaded {len(nodes)} nodes")

    edges = find_nearest_neighbors(nodes, config)
    print(f"Generated {len(edges)} edges")

    test_edges = edges[:10]
    print(f"Testing with {len(test_edges)} edges")

    traffic_data = []
    collector_config = config["collectors"].get("google_directions") or config["collectors"]["google"]
    api_key = os.getenv(collector_config.get("api_key_env", "GOOGLE_MAPS_API_KEY"))
    use_real_api = bool(api_key and len(api_key) > 10)

    rate_limiter = get_rate_limiter(config)

    print(f"Using {'REAL' if use_real_api else 'MOCK'} Google Directions API")
    if use_real_api:
        print(f"Rate limiting: {rate_limiter.requests_per_minute} requests/minute")

    for i, (node_a, node_b, dist) in enumerate(test_edges):
        print(f"Processing edge {i+1}/10: {node_a['node_id']} -> {node_b['node_id']} ({dist:.2f}km)")

        if use_real_api:
            directions = real_directions_api(node_a, node_b, api_key, rate_limiter)
            if directions is None:
                directions = mock_directions_api(node_a, node_b, config)
                api_type = "mock_fallback"
            else:
                api_type = "real"
        else:
            directions = mock_directions_api(node_a, node_b, config)
            api_type = "mock"

        traffic_data.append(
            {
                "node_a_id": node_a["node_id"],
                "node_b_id": node_b["node_id"],
                "distance_km": directions["distance_km"],
                "duration_sec": directions["duration_sec"],
                "speed_kmh": directions["speed_kmh"],
                "timestamp": "2025-10-09T00:00:00",
                "api_type": api_type,
            }
        )

        print(f"  Result: {directions['speed_kmh']:.1f}km/h ({api_type})")

    output_file = PROJECT_ROOT / "test_traffic_edges.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(traffic_data, f, indent=2)

    print(f"Saved {len(traffic_data)} test results to {output_file}")


if __name__ == "__main__":
    test_limited_collection()
