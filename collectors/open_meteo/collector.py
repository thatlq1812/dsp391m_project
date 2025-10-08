"""
Open-Meteo collector for weather data.
"""

import sys
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import requests
import json
import os
from datetime import datetime
from collections import defaultdict
import yaml

from collectors.area_utils import load_area_config
import argparse

OPENMETEO_BASE_URL = os.getenv('OPENMETEO_BASE_URL', 'https://api.open-meteo.com/v1/forecast')


def load_nodes():
    with open('data/nodes.json', 'r') as f:
        return json.load(f)


def filter_nodes_by_bbox(nodes, bbox):
    # bbox: [min_lat, min_lon, max_lat, max_lon]
    min_lat, min_lon, max_lat, max_lon = bbox
    return [n for n in nodes if (min_lat <= n['lat'] <= max_lat and min_lon <= n['lon'] <= max_lon)]


def group_nodes_by_grid(nodes, grid_size=0.05):
    """Group nodes by lat/lon grid of grid_size degrees."""
    grid = defaultdict(list)
    for node in nodes:
        lat_grid = round(node['lat'] / grid_size) * grid_size
        lon_grid = round(node['lon'] / grid_size) * grid_size
        grid[(lat_grid, lon_grid)].append(node)
    return grid


def fetch_weather(lat, lon):
    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': 'temperature_2m,precipitation,wind_speed_10m',
        'current_weather': True,
        'timezone': 'Asia/Ho_Chi_Minh'
    }
    response = requests.get(OPENMETEO_BASE_URL, params=params)
    try:
        data = response.json()
    except Exception:
        data = {'error': 'invalid_json', 'status_code': response.status_code, 'text': response.text}

    # Optional debug: save raw response for inspection
    if os.getenv('OPENMETEO_DEBUG'):
        os.makedirs('data/open_meteo_raw', exist_ok=True)
        fname = f"data/open_meteo_raw/{lat}_{lon}.json"
        try:
            with open(fname, 'w', encoding='utf-8') as fh:
                json.dump({'url': response.url, 'status': response.status_code, 'payload': data}, fh, indent=2)
        except Exception:
            pass

    return data


def project_weather_to_nodes(grid, weather_data):
    """Project grid weather to all nodes in that grid."""
    snapshots = []
    # Open-Meteo may return `current_weather` and/or `hourly`.
    current = weather_data.get('current_weather') or {}
    hourly = weather_data.get('hourly', {}) or {}

    # Determine temperature
    temp = None
    if current:
        temp = current.get('temperature') or current.get('temperature_2m')

    # precipitation doesn't exist in current_weather -> use hourly last value
    precip = None
    # Determine wind: prefer current_weather.windspeed (km/h per API), otherwise use hourly wind (m/s -> km/h)
    wind = None
    if current:
        wind = current.get('windspeed') or current.get('wind_speed_10m')

    # If any of values missing, fallback to last hourly value
    try:
        if temp is None and 'temperature_2m' in hourly:
            arr = hourly.get('temperature_2m')
            if isinstance(arr, list) and len(arr) > 0:
                temp = arr[-1]
        if precip is None and 'precipitation' in hourly:
            arr = hourly.get('precipitation')
            if isinstance(arr, list) and len(arr) > 0:
                precip = arr[-1]
        if wind is None and 'wind_speed_10m' in hourly:
            arr = hourly.get('wind_speed_10m')
            if isinstance(arr, list) and len(arr) > 0:
                # hourly wind is provided in m/s; convert to km/h
                wind = float(arr[-1]) * 3.6
    except Exception:
        # be robust to unexpected payloads
        pass

    for grid_key, nodes in grid.items():
        lat, lon = grid_key
        for node in nodes:
            snapshots.append({
                'node_id': node['node_id'],
                'lat': node['lat'],
                'lon': node['lon'],
                'timestamp': datetime.now().isoformat(),
                'temperature_c': temp,
                'precipitation_mm': precip,
                'wind_speed_kmh': wind
            })
    return snapshots


def run_open_meteo_collector():
    parser = argparse.ArgumentParser(description='Open-Meteo collector')
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

    # load config area (CLI overrides applied)
    area_cfg = load_area_config('open_meteo', cli_area=cli_area)
    nodes = load_nodes()

    # filter nodes by computed bbox
    nodes = filter_nodes_by_bbox(nodes, area_cfg['bbox'])
    grid = group_nodes_by_grid(nodes, grid_size=float(os.getenv('OPENMETEO_GRID_SIZE', 0.05)))

    all_snapshots = []
    for (lat, lon), _ in grid.items():
        weather = fetch_weather(lat, lon)
        snapshots = project_weather_to_nodes({(lat, lon): grid[(lat, lon)]}, weather)
        all_snapshots.extend(snapshots)

    os.makedirs('data', exist_ok=True)
    with open('data/weather_snapshot.json', 'w') as f:
        json.dump(all_snapshots, f, indent=2)

    print(f"Collected weather for {len(all_snapshots)} nodes.")


if __name__ == "__main__":
    run_open_meteo_collector()