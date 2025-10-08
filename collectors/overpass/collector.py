"""
Overpass collector for OSM topology.
"""

import sys
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests
import json
import os
from dotenv import load_dotenv
from collectors.area_utils import load_area_config
from collectors.cache_utils import get_or_create_cache
import argparse
import math
import yaml

load_dotenv()
OVERPASS_URL = os.getenv('OVERPASS_URL', 'https://overpass.kumi.systems/api/interpreter')


def query_overpass(bbox):
    # Overpass expects south,west,north,east i.e. min_lat,min_lon,max_lat,max_lon
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
    query = f"""
[out:json][timeout:120];
(
  way["highway"~"motorway|trunk|primary"]
    ({bbox_str});
);
out geom;
"""
    print(f"Querying Overpass API for bbox: {bbox_str}")
    response = requests.post(OVERPASS_URL, data={'data': query}, timeout=300)  # 5 minute timeout
    response.raise_for_status()  # Raise exception for bad status codes
    print(f"Overpass API response received, status: {response.status_code}")
    return response.json()


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1-a))


def interp_point(lat1, lon1, lat2, lon2, frac):
    # simple linear interpolation in lat/lon (acceptable for short segments)
    return lat1 + (lat2 - lat1) * frac, lon1 + (lon2 - lon1) * frac


def extract_nodes_and_edges(data, spacing_m=50):
    nodes = []
    seen = {}
    way_sequences = []  # list of node keys per way in order

    for element in data.get('elements', []):
        if element['type'] != 'way':
            continue
        geoms = element.get('geometry', [])
        seq = []
        for i in range(len(geoms)-1):
            a = geoms[i]
            b = geoms[i+1]
            lat1, lon1 = a['lat'], a['lon']
            lat2, lon2 = b['lat'], b['lon']
            seg_km = haversine_km(lat1, lon1, lat2, lon2)
            seg_m = seg_km * 1000.0
            if seg_m <= 0:
                continue
            n_steps = max(1, int(math.floor(seg_m / float(spacing_m))))
            for step in range(n_steps+1):
                frac = min(1.0, float(step) / float(n_steps))
                lat, lon = interp_point(lat1, lon1, lat2, lon2, frac)
                key = (round(lat, 6), round(lon, 6))
                if key not in seen:
                    node_id = f"node-{key[0]:.6f}-{key[1]:.6f}"
                    seen[key] = node_id
                    nodes.append({
                        'node_id': node_id,
                        'lat': lat,
                        'lon': lon,
                        'road_type': element.get('tags', {}).get('highway', 'unknown'),
                        'lane_count': element.get('tags', {}).get('lanes'),
                        'speed_limit': element.get('tags', {}).get('maxspeed')
                    })
                seq.append(seen[key])
        if seq:
            # compress consecutive duplicates
            compressed = [seq[0]]
            for nid in seq[1:]:
                if nid != compressed[-1]:
                    compressed.append(nid)
            way_sequences.append({'way_id': element.get('id'), 'nodes': compressed, 'tags': element.get('tags', {})})

    # build edges by linking consecutive nodes on each way
    edges = []
    edge_set = set()
    for way in way_sequences:
        nodes_seq = way['nodes']
        for i in range(len(nodes_seq)-1):
            a = nodes_seq[i]
            b = nodes_seq[i+1]
            key = tuple(sorted((a, b)))
            if key in edge_set:
                continue
            edge_set.add(key)
            # lookup coords
            # find lat/lon from seen mapping
            # reverse seen: key->node_id mapping stored in seen; build id->coords dict
            # build id_coords once
    id_coords = {v: k for k, v in seen.items()}
    for way in way_sequences:
        nodes_seq = way['nodes']
        for i in range(len(nodes_seq)-1):
            a = nodes_seq[i]
            b = nodes_seq[i+1]
            key = tuple(sorted((a, b)))
            if any(e.get('u') == key[0] and e.get('v') == key[1] for e in edges):
                continue
            # get coords
            ka = id_coords.get(a)
            kb = id_coords.get(b)
            if not ka or not kb:
                continue
            lat1, lon1 = ka
            lat2, lon2 = kb
            dist_km = haversine_km(lat1, lon1, lat2, lon2)
            edges.append({'u': a, 'v': b, 'distance_m': dist_km * 1000.0, 'way_id': way.get('way_id'), 'tags': way.get('tags', {})})

    return nodes, edges, way_sequences


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

    # Load config for cache settings
    cfg = yaml.safe_load(open('configs/project_config.yaml', 'r'))
    collector_cfg = cfg['collectors']['overpass']
    cache_enabled = False  # Force disable cache for debugging
    cache_expiry_hours = collector_cfg.get('cache_expiry_hours', 24)
    cache_dir = cfg['data'].get('cache_dir', './cache')

    # Define fetch function
    def fetch_overpass_data():
        return query_overpass(area_cfg['bbox'])

    # Get data from cache or fetch fresh
    if cache_enabled:
        cache_params = {
            'bbox': area_cfg['bbox'],
            'spacing_m': cfg.get('globals', {}).get('node_spacing_m', 50)
        }
        
        def fetch_processed_data():
            raw_data = query_overpass(area_cfg['bbox'])
            spacing = cfg.get('globals', {}).get('node_spacing_m', 50)
            nodes, edges, way_sequences = extract_nodes_and_edges(raw_data, spacing_m=spacing)
            return {'nodes': nodes, 'edges': edges, 'way_sequences': way_sequences}
        
        cached_result = get_or_create_cache(
            collector_name='overpass',
            params=cache_params,
            cache_dir=cache_dir,
            expiry_hours=cache_expiry_hours,
            fetch_func=fetch_processed_data
        )
        nodes, edges, way_sequences = cached_result['nodes'], cached_result['edges'], cached_result['way_sequences']
    else:
        data = query_overpass(area_cfg['bbox'])
        spacing = cfg.get('globals', {}).get('node_spacing_m', 50)
        nodes, edges, way_sequences = extract_nodes_and_edges(data, spacing_m=spacing)

    # Prefer writing into RUN_DIR if provided (per-run collectors folder)
    run_dir = os.getenv('RUN_DIR')
    if run_dir:
        out_dir = os.path.join(run_dir, 'collectors', 'overpass')
    else:
        out_dir = os.path.join('data')
    os.makedirs(out_dir, exist_ok=True)
    out_path_nodes = os.path.join(out_dir, 'nodes.json')
    with open(out_path_nodes, 'w', encoding='utf-8') as f:
        json.dump(nodes, f, indent=2)
    out_path_edges = os.path.join(out_dir, 'edges.json')
    with open(out_path_edges, 'w', encoding='utf-8') as f:
        json.dump(edges, f, indent=2)
    out_path_ways = os.path.join(out_dir, 'way_sequences.json')
    with open(out_path_ways, 'w', encoding='utf-8') as f:
        json.dump(way_sequences, f, indent=2)

    print(f"Collected {len(nodes)} nodes, {len(edges)} edges, {len(way_sequences)} ways. Saved to {out_path_nodes}, {out_path_edges}, {out_path_ways}")


if __name__ == "__main__":
    run_overpass_collector()