"""
Overpass collector for OSM topology.
"""

import requests
import json
import os

OVERPASS_URL = os.getenv('OVERPASS_URL', 'https://overpass-api.de/api/interpreter')

def query_overpass(bbox):
    query = f"""
[out:json][timeout:120];
(
  way["highway"~"motorway|trunk|primary|secondary|tertiary|residential"]
    ({bbox});
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
    bbox = "10.67,106.60,10.90,106.84"  # HCMC
    data = query_overpass(bbox)
    nodes = extract_nodes(data)
    
    os.makedirs('data', exist_ok=True)
    with open('data/nodes.json', 'w') as f:
        json.dump(nodes, f, indent=2)
    
    print(f"Collected {len(nodes)} nodes.")

if __name__ == "__main__":
    run_overpass_collector()