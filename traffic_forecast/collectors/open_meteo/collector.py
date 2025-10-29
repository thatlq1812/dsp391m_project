"""
Open-Meteo collector for weather data.
"""

import requests
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from collections import defaultdict
import yaml
from traffic_forecast import PROJECT_ROOT
from traffic_forecast.collectors.area_utils import load_area_config
from traffic_forecast.collectors.cache_utils import get_or_create_cache
import argparse

load_dotenv()

OPENMETEO_BASE_URL = os.getenv('OPENMETEO_BASE_URL', 'https://api.open-meteo.com/v1/forecast')


def load_nodes():
    """Load nodes from RUN_DIR or fallback locations."""
    # Try RUN_DIR first (when run from collect_once.py)
    run_dir = os.getenv('RUN_DIR')
    if run_dir:
        # New structure: nodes.json directly in run directory
        nodes_path = Path(run_dir) / 'nodes.json'
        if nodes_path.exists():
            with nodes_path.open('r', encoding='utf-8') as f:
                return json.load(f)
        
        # Legacy structure: collectors/overpass/nodes.json
        nodes_path = Path(run_dir) / 'collectors' / 'overpass' / 'nodes.json'
        if nodes_path.exists():
            with nodes_path.open('r', encoding='utf-8') as f:
                return json.load(f)

    # Fallback to global nodes.json
    nodes_path = PROJECT_ROOT / 'data' / 'nodes.json'
    if nodes_path.exists():
        with nodes_path.open(encoding='utf-8') as f:
            return json.load(f)

    raise FileNotFoundError("Could not find nodes.json in RUN_DIR or data/")


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
    response = requests.get(OPENMETEO_BASE_URL, params=params, timeout=30)
    try:
        data = response.json()
    except Exception:
        data = {'error': 'invalid_json', 'status_code': response.status_code, 'text': response.text}

    # Save raw response when OPENMETEO_DEBUG env var is set (for troubleshooting)
    if os.getenv('OPENMETEO_DEBUG'):
        raw_dir = PROJECT_ROOT / 'data' / 'open_meteo_raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        fname = raw_dir / f"{lat}_{lon}.json"
        try:
            with fname.open('w', encoding='utf-8') as fh:
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

    # Load config for cache settings
    config_path = PROJECT_ROOT / "configs" / "project_config.yaml"
    with config_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    collector_cfg = cfg['collectors']['open_meteo']
    cache_enabled = collector_cfg.get('cache_enabled', False)
    cache_expiry_hours = collector_cfg.get('cache_expiry_hours', 1)
    cache_dir = cfg['data'].get('cache_dir', './cache')

    all_snapshots = []
    for (lat, lon), _ in grid.items():
        # Define fetch function for this grid point
        def fetch_weather_for_point():
            return fetch_weather(lat, lon)

        # Get weather data from cache or fetch fresh
        if cache_enabled:
            cache_params = {
                'lat': lat,
                'lon': lon,
                'grid_size': float(os.getenv('OPENMETEO_GRID_SIZE', 0.05))
            }
            weather = get_or_create_cache(
                collector_name='open_meteo',
                params=cache_params,
                cache_dir=cache_dir,
                expiry_hours=cache_expiry_hours,
                fetch_func=fetch_weather_for_point
            )
        else:
            weather = fetch_weather_for_point()

        snapshots = project_weather_to_nodes({(lat, lon): grid[(lat, lon)]}, weather)
        all_snapshots.extend(snapshots)

    # Save - use RUN_DIR if set, otherwise create timestamped directory
    run_dir = os.getenv('RUN_DIR')
    if run_dir:
        # Use provided run directory (from orchestrator)
        out_dir = run_dir
    else:
        # Create timestamped run directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.join('data', 'runs', f'run_{timestamp}')
    
    os.makedirs(out_dir, exist_ok=True)
    
    out_path = os.path.join(out_dir, 'weather_snapshot.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_snapshots, f, indent=2)

    print(f"Collected weather for {len(all_snapshots)} nodes. Saved to {out_path}")


if __name__ == "__main__":
    run_open_meteo_collector()
