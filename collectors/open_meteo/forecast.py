"""
Open-Meteo forecast collector: Get hourly weather forecasts and interpolate to horizons.
"""

import requests
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict
import yaml

def load_config():
    with open('configs/project_config.yaml', 'r') as f:
        return yaml.safe_load(f)

def load_nodes():
    with open('data/nodes.json', 'r') as f:
        return json.load(f)

def group_nodes_by_grid(nodes, grid_size=0.05):
    grid = defaultdict(list)
    for node in nodes:
        lat_grid = round(node['lat'] / grid_size) * grid_size
        lon_grid = round(node['lon'] / grid_size) * grid_size
        grid[(lat_grid, lon_grid)].append(node)
    return grid

def fetch_forecast(lat, lon, config):
    collector_config = config['collectors']['open_meteo']
    params = {
        'latitude': lat,
        'longitude': lon,
        'hourly': collector_config['hourly_params'],
        'forecast_days': collector_config['forecast_days'],
        'timezone': collector_config['timezone']
    }
    response = requests.get(collector_config['forecast_url'], params=params)
    return response.json()

def interpolate_forecast(hourly_data, horizons_min):
    """Interpolate hourly data to specific horizons."""
    times = hourly_data['time']
    temps = hourly_data['temperature_2m']
    rains = hourly_data['precipitation']
    winds = hourly_data['wind_speed_10m']
    
    now = datetime.now()
    forecasts = {}
    for h in horizons_min:
        target_time = now + timedelta(minutes=h)
        # Simple linear interpolation (in practice, use scipy.interpolate)
        # For demo, take nearest hour
        target_hour = target_time.replace(minute=0, second=0, microsecond=0)
        if target_hour.isoformat() in times:
            idx = times.index(target_hour.isoformat())
            forecasts[f't{h}_c'] = temps[idx]
            forecasts[f'r{h}_mm'] = rains[idx]
            forecasts[f'w{h}_kmh'] = winds[idx]
        else:
            forecasts[f't{h}_c'] = None
            forecasts[f'r{h}_mm'] = None
            forecasts[f'w{h}_kmh'] = None
    return forecasts

def run_forecast_collector():
    config = load_config()
    collector_config = config['collectors']['open_meteo']
    
    nodes = load_nodes()
    grid = group_nodes_by_grid(nodes, collector_config['grid_size'])
    horizons = collector_config['horizons_min']
    
    all_forecasts = []
    for (lat, lon), _ in grid.items():
        data = fetch_forecast(lat, lon, config)
        if 'hourly' in data:
            forecasts = interpolate_forecast(data['hourly'], horizons)
            for node in grid[(lat, lon)]:
                row = {
                    'ts': datetime.now().isoformat(),
                    'node_id': node['node_id'],
                    'forecast_temp_t5_c': forecasts.get('t5_c'),
                    'forecast_temp_t15_c': forecasts.get('t15_c'),
                    'forecast_temp_t30_c': forecasts.get('t30_c'),
                    'forecast_temp_t60_c': forecasts.get('t60_c'),
                    'forecast_rain_t5_mm': forecasts.get('r5_mm'),
                    'forecast_rain_t15_mm': forecasts.get('r15_mm'),
                    'forecast_rain_t30_mm': forecasts.get('r30_mm'),
                    'forecast_rain_t60_mm': forecasts.get('r60_mm'),
                    'forecast_wind_t5_kmh': forecasts.get('w5_kmh'),
                    'forecast_wind_t15_kmh': forecasts.get('w15_kmh'),
                    'forecast_wind_t30_kmh': forecasts.get('w30_kmh'),
                    'forecast_wind_t60_kmh': forecasts.get('w60_kmh')
                }
                all_forecasts.append(row)
    
    os.makedirs('data', exist_ok=True)
    with open(collector_config['output_file'], 'w') as f:
        json.dump(all_forecasts, f, indent=2)
    
    print(f"Collected forecasts for {len(all_forecasts)} nodes.")

if __name__ == "__main__":
    run_forecast_collector()