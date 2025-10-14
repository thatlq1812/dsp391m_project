#!/usr/bin/env python3
"""
Debug utilities for the Google Directions API configuration.
"""

import os
import time

import requests
from dotenv import load_dotenv
import yaml

from traffic_forecast import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "configs" / "project_config.yaml"

load_dotenv()


def test_api_key() -> str | None:
    """Display a brief summary of the Google Maps API key."""
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    print(f"API Key loaded: {bool(api_key)}")
    print(f"API Key length: {len(api_key) if api_key else 0}")
    return api_key


def test_config() -> dict:
    """Load the project config and print collector settings."""
    with CONFIG_PATH.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    collector_config = config.get("collectors", {}).get("google_directions") or config.get("collectors", {}).get("google", {})
    print(f"Collector enabled: {collector_config.get('enabled', False)}")
    print(f"Rate limit (requests/min): {collector_config.get('rate_limit_requests_per_minute', 'Not set')}")
    return config


def test_single_request() -> None:
    """Perform a simple request against Google Directions."""
    api_key = os.getenv("GOOGLE_MAPS_API_KEY")
    if not api_key:
        print("No API key found; skipping live request.")
        return

    origin = {"lat": 10.772465, "lon": 106.697794}
    dest = {"lat": 10.775, "lon": 106.700}

    base_url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{origin['lat']},{origin['lon']}",
        "destination": f"{dest['lat']},{dest['lon']}",
        "key": api_key,
        "mode": "driving",
        "departure_time": "now",
        "traffic_model": "best_guess",
    }

    print("Making test request...")
    start_time = time.time()
    try:
        response = requests.get(base_url, params=params, timeout=10)
        elapsed = time.time() - start_time
        print(f"Response time: {elapsed:.2f}s")
        print(f"Status code: {response.status_code}")

        if response.status_code == 200:
            data = response.json()
            print(f"API Status: {data.get('status')}")
            if data.get("status") == "OK" and data.get("routes"):
                leg = data["routes"][0]["legs"][0]
                distance_km = leg["distance"]["value"] / 1000
                duration_sec = leg["duration_in_traffic"]["value"]
                speed_kmh = (distance_km / (duration_sec / 3600)) if duration_sec > 0 else 0
                print(f"Distance: {distance_km:.2f} km")
                print(f"Duration: {duration_sec} s")
                print(f"Speed: {speed_kmh:.1f} km/h")
            else:
                print(f"API Error payload: {data}")
        else:
            print(f"HTTP Error: {response.text}")
    except Exception as exc:
        print(f"Request failed: {exc}")


def main() -> None:
    print("=== Google Directions API Debug ===")
    test_api_key()
    print()
    test_config()
    print()
    test_single_request()


if __name__ == "__main__":
    main()
