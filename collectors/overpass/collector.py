"""
Overpass collector for OSM topology.
"""

import requests
import json
import os
from collectors.area_utils import load_area_config

OVERPASS_URL = os.getenv('OVERPASS_URL', 'https://overpass-api.de/api/interpreter')


def query_overpass(bbox):
    # Overpass expects south,west,north,east i.e. min_lat,min_lon,max_lat,max_lon
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    query = f"""
[out:json][timeout:120];
(
  way["highway"~"motorway|trunk|primary|secondary|tertiary|residential"]
    ({bbox_str});
);
out geom;
"""
    response = requests.post(OVERPASS_URL, data={'data': query})
    return response.json()


def extract_nodes(data):
    nodes = []
    for element in data.get('elements', []):
        if element['type'] == 'way':
            for geom in element.get('geometry', []):
                nodes.append({
                    'node_id': f"node-{geom['lat']:.5f}-{geom['lon']:.5f}",
                    'lat': geom['lat'],
                    'lon': geom['lon'],
                    'road_type': element['tags'].get('highway', 'unknown'),
                    'lane_count': element['tags'].get('lanes'),
                    'speed_limit': element['tags'].get('maxspeed')
                })
    return nodes


def run_overpass_collector():
    area_cfg = load_area_config('overpass')
    data = query_overpass(area_cfg['bbox'])
    nodes = extract_nodes(data)

    os.makedirs('data', exist_ok=True)
    with open('data/nodes.json', 'w') as f:
        json.dump(nodes, f, indent=2)

    print(f"Collected {len(nodes)} nodes.")


if __name__ == "__main__":
    run_overpass_collector()