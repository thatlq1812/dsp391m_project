"""
Build features v2: Join traffic, weather, forecast with forward-fill.
"""

import json
from datetime import datetime

from traffic_forecast import PROJECT_ROOT


def load_traffic():
    path = PROJECT_ROOT / 'data' / 'traffic_snapshot_normalized.json'
    if path.exists():
    with path.open('r', encoding='utf-8') as f:
    return json.load(f)
    return []


def load_weather():
    path = PROJECT_ROOT / 'data' / 'weather_snapshot.json'
    if path.exists():
    with path.open('r', encoding='utf-8') as f:
    return json.load(f)
    return []


def load_forecast():
    path = PROJECT_ROOT / 'data' / 'weather_forecast.json'
    if path.exists():
    with path.open('r', encoding='utf-8') as f:
    return json.load(f)
    return []


def load_nodes():
    path = PROJECT_ROOT / 'data' / 'nodes.json'
    with path.open('r', encoding='utf-8') as f:
    return json.load(f)


def forward_fill(data, key, window_minutes=30):
    """Simple forward-fill within window."""
    sorted_data = sorted(data, key=lambda x: x['timestamp'])
    filled = {}
    last_value = None
    last_time = None
    for item in sorted_data:
    ts = datetime.fromisoformat(item['timestamp'])
    if last_value and (ts - last_time).total_seconds() / 60 <= window_minutes:
    filled[item[key]] = last_value
    else:
    filled[item[key]] = item
    last_value = item
    last_time = ts
    return filled


def build_features_v2():
    nodes = load_nodes()
    traffic = {t['node_id']: t for t in load_traffic()}
    weather = {w['node_id']: w for w in load_weather()}
    forecast = {f['node_id']: f for f in load_forecast()}

    features = []
    for node in nodes:
    node_id = node['node_id']
    ts = datetime.now().isoformat()

    # Traffic
    t_data = traffic.get(node_id, {})
    avg_speed = t_data.get('avg_speed_kmh')
    congestion = 1 if avg_speed and avg_speed < 20 else 0  # Simple

    # Weather (current)
    w_data = weather.get(node_id, {})
    temp = w_data.get('temperature_c')
    rain = w_data.get('precipitation_mm')
    wind = w_data.get('wind_speed_kmh')

    # Forecast
    f_data = forecast.get(node_id, {})
    forecast_temp_t5 = f_data.get('forecast_temp_t5_c')
    forecast_temp_t15 = f_data.get('forecast_temp_t15_c')
    forecast_temp_t30 = f_data.get('forecast_temp_t30_c')
    forecast_temp_t60 = f_data.get('forecast_temp_t60_c')
    forecast_rain_t5 = f_data.get('forecast_rain_t5_mm')
    forecast_rain_t15 = f_data.get('forecast_rain_t15_mm')
    forecast_rain_t30 = f_data.get('forecast_rain_t30_mm')
    forecast_rain_t60 = f_data.get('forecast_rain_t60_mm')
    forecast_wind_t5 = f_data.get('forecast_wind_t5_kmh')
    forecast_wind_t15 = f_data.get('forecast_wind_t15_kmh')
    forecast_wind_t30 = f_data.get('forecast_wind_t30_kmh')
    forecast_wind_t60 = f_data.get('forecast_wind_t60_kmh')

    # Feature vector (basic)
    feature_vector = [
        float(node.get('lane_count') or 2) / 10,
        float(node.get('speed_limit') or 50) / 100,
        temp / 50 if temp else 0,
        rain / 10 if rain else 0,
        wind / 20 if wind else 0,
        forecast_temp_t5 / 50 if forecast_temp_t5 else 0,
        forecast_rain_t5 / 10 if forecast_rain_t5 else 0,
        forecast_wind_t5 / 20 if forecast_wind_t5 else 0
    ]

    row = {
        'node_id': node_id,
        'lat': node['lat'],
        'lon': node['lon'],
        'ts': ts,
        'avg_speed_kmh': avg_speed,
        'congestion_level': congestion,
        'temperature_c': temp,
        'rain_mm': rain,
        'wind_speed_kmh': wind,
        'forecast_temp_t5_c': forecast_temp_t5,
        'forecast_temp_t15_c': forecast_temp_t15,
        'forecast_temp_t30_c': forecast_temp_t30,
        'forecast_temp_t60_c': forecast_temp_t60,
        'forecast_rain_t5_mm': forecast_rain_t5,
        'forecast_rain_t15_mm': forecast_rain_t15,
        'forecast_rain_t30_mm': forecast_rain_t30,
        'forecast_rain_t60_mm': forecast_rain_t60,
        'forecast_wind_t5_kmh': forecast_wind_t5,
        'forecast_wind_t15_kmh': forecast_wind_t15,
        'forecast_wind_t30_kmh': forecast_wind_t30,
        'forecast_wind_t60_kmh': forecast_wind_t60,
        'feature_vector': feature_vector,
        'fetch_time': ts
    }
    features.append(row)

    output_path = PROJECT_ROOT / 'data' / 'features_nodes_v2.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
    json.dump(features, f, indent=2)

    print(f"Built features v2 for {len(features)} nodes.")


if __name__ == "__main__":
    build_features_v2()
