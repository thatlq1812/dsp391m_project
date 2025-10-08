"""
Overpass collector for OSM topology.
"""

import requests
import json
import os
from collectors.area_utils import load_area_config
import argparse

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
    parser = argparse.ArgumentParser(description='Overpass OSM collector')
    parser.add_argument('--mode', choices=['bbox', 'point_radius', 'circle'], help='Area selection mode')
    parser.add_argument('--bbox', help='bbox as min_lat,min_lon,max_lat,max_lon')
    parser.add_argument('--center', help='center as lon,lat')
    parser.add_argument('--radius', type=float, help='radius in meters')
    args = parser.parse_args()

    cli_area = {}
    if args.mode:
        cli_area['mode'] = args.mode
    if args.bbox:
        cli_area['bbox'] = list(map(float, args.bbox.split(',')))
    if args.center:
        cli_area['center'] = list(map(float, args.center.split(',')))
    if args.radius:
        cli_area['radius_m'] = args.radius

    area_cfg = load_area_config('overpass', cli_area=cli_area)
    data = query_overpass(area_cfg['bbox'])
    nodes = extract_nodes(data)

    os.makedirs('data', exist_ok=True)
    with open('data/nodes.json', 'w') as f:
        json.dump(nodes, f, indent=2)

    print(f"Collected {len(nodes)} nodes.")


if __name__ == "__main__":
    run_overpass_collector()