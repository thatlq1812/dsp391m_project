"""
Open-Meteo collector for weather data.
"""

import requests
import json
import os
from datetime import datetime
from collections import defaultdict

OPENMETEO_BASE_URL = os.getenv('OPENMETEO_BASE_URL', 'https://api.open-meteo.com/v1/forecast')

def load_nodes():
    with open('data/nodes.json', 'r') as f:
        return json.load(f)

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
        'current': 'temperature_2m,precipitation,wind_speed_10m',
        'timezone': 'Asia/Ho_Chi_Minh'
    }
    response = requests.get(OPENMETEO_BASE_URL, params=params)
    return response.json()

def project_weather_to_nodes(grid, weather_data):
    """Project grid weather to all nodes in that grid."""
    snapshots = []
    current = weather_data.get('current', {})
    hourly = weather_data.get('hourly', {})
    
    # Use current for snapshot
    temp = current.get('temperature_2m')
    precip = current.get('precipitation')
    wind = current.get('wind_speed_10m')
    
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
    nodes = load_nodes()
    grid = group_nodes_by_grid(nodes)
    
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