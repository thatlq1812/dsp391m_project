#!/usr/bin/env python3
"""
Debug script for Google Directions API
"""

import sys
import pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json
import os
import time
from dotenv import load_dotenv
import yaml

load_dotenv()

def test_api_key():
    """Test if API key is loaded."""
    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    print(f"API Key loaded: {bool(api_key)}")
    print(f"API Key length: {len(api_key) if api_key else 0}")
    return api_key

def test_config():
    """Test config loading."""
    config = yaml.safe_load(open('configs/project_config.yaml', 'r'))
    collector_config = config['collectors'].get('google_directions') or config['collectors']['google']
    print(f"Collector config: {collector_config.get('enabled', False)}")
    print(f"Rate limit: {collector_config.get('rate_limit_requests_per_minute', 'Not set')}")
    return config

def test_single_request():
    """Test a single Google Directions API request."""
    import requests

    api_key = os.getenv('GOOGLE_MAPS_API_KEY')
    if not api_key:
        print("No API key found")
        return

    # Test coordinates (near Ben Thanh market)
    origin = {'lat': 10.772465, 'lon': 106.697794}
    dest = {'lat': 10.775, 'lon': 106.700}

    base_url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        'origin': f"{origin['lat']},{origin['lon']}",
        'destination': f"{dest['lat']},{dest['lon']}",
        'key': api_key,
        'mode': 'driving',
        'departure_time': 'now',
        'traffic_model': 'best_guess'
    }

    print("Making test request...")
    start_time = time.time()
    try:
        response = requests.get(base_url, params=params, timeout=10)
        end_time = time.time()
        print(f"Response time: {end_time - start_time:.2f}s")
        print(f"Status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"API Status: {data.get('status')}")
            if data.get('status') == 'OK' and data.get('routes'):
                route = data['routes'][0]
                leg = route['legs'][0]
                distance_km = leg['distance']['value'] / 1000
                duration_sec = leg['duration_in_traffic']['value']
                speed_kmh = (distance_km / (duration_sec / 3600)) if duration_sec > 0 else 30
                print(f"Distance: {distance_km:.2f}km")
                print(f"Duration: {duration_sec}s")
                print(f"Speed: {speed_kmh:.1f}km/h")
            else:
                print(f"API Error: {data}")
        else:
            print(f"HTTP Error: {response.text}")

    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    print("=== Google Directions API Debug ===")
    test_api_key()
    print()
    test_config()
    print()
    test_single_request()