"""
Continuous Traffic Data Collector for VM

This script collects traffic data from Google Directions API every 15 minutes.
Appends to monthly Parquet files for demo and analysis.

Usage:
    # Manual run
    python traffic_collector.py
    
    # Cron job (every 15 minutes)
    */15 * * * * cd /opt/traffic_data && python3 traffic_collector.py >> collector.log 2>&1

Author: THAT Le Quang
Date: November 2025
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY')

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = Path('/opt/traffic_data') if Path('/opt/traffic_data').exists() else PROJECT_ROOT / 'data' / 'demo'
TOPOLOGY_FILE = PROJECT_ROOT / 'cache' / 'overpass_topology.json'

# HCMC center coordinates
HCMC_LAT = 10.762622
HCMC_LON = 106.660172


def load_edges() -> List[Dict]:
    """Load edges from topology file."""
    if not TOPOLOGY_FILE.exists():
        logger.error(f"Topology file not found: {TOPOLOGY_FILE}")
        raise FileNotFoundError(f"Run scripts/data/01_collection/build_topology.py first")
    
    with open(TOPOLOGY_FILE, 'r', encoding='utf-8') as f:
        topology = json.load(f)
    
    edges = topology.get('edges', [])
    logger.info(f"Loaded {len(edges)} edges from topology")
    return edges


def fetch_traffic_for_edge(edge: Dict) -> Optional[Dict]:
    """
    Fetch current traffic data for an edge using Google Directions API.
    
    Args:
        edge: Edge dict with node_a_id, node_b_id, lat_a, lon_a, lat_b, lon_b
        
    Returns:
        Dict with traffic data or None if failed
    """
    url = "https://maps.googleapis.com/maps/api/directions/json"
    
    params = {
        'origin': f"{edge['lat_a']},{edge['lon_a']}",
        'destination': f"{edge['lat_b']},{edge['lon_b']}",
        'mode': 'driving',
        'departure_time': 'now',  # Get current traffic
        'key': GOOGLE_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data['status'] == 'OK' and data['routes']:
            route = data['routes'][0]['legs'][0]
            
            # Extract traffic data
            distance_m = route['distance']['value']
            duration_sec = route['duration']['value']
            duration_in_traffic_sec = route.get('duration_in_traffic', {}).get('value', duration_sec)
            
            # Calculate speeds
            distance_km = distance_m / 1000
            duration_hours = duration_sec / 3600
            duration_traffic_hours = duration_in_traffic_sec / 3600
            
            speed_normal = distance_km / duration_hours if duration_hours > 0 else 0
            speed_current = distance_km / duration_traffic_hours if duration_traffic_hours > 0 else 0
            
            return {
                'edge_id': edge['edge_id'],
                'node_a_id': edge['node_a_id'],
                'node_b_id': edge['node_b_id'],
                'lat_a': edge['lat_a'],
                'lon_a': edge['lon_a'],
                'lat_b': edge['lat_b'],
                'lon_b': edge['lon_b'],
                'distance_km': distance_km,
                'duration_sec': duration_sec,
                'duration_in_traffic_sec': duration_in_traffic_sec,
                'speed_kmh': speed_current,  # Current speed with traffic
                'speed_normal_kmh': speed_normal,  # Speed without traffic
                'google_predicted_speed': speed_current  # Google's prediction
            }
        elif data['status'] == 'OVER_QUERY_LIMIT':
            logger.warning(f"Rate limit hit for edge {edge['edge_id']}")
            return None
        else:
            logger.warning(f"API returned status {data['status']} for edge {edge['edge_id']}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed for edge {edge['edge_id']}: {e}")
        return None


def fetch_weather() -> Dict:
    """Fetch current weather for HCMC from OpenWeatherMap API."""
    if not OPENWEATHER_API_KEY:
        logger.warning("No OpenWeatherMap API key, using default values")
        return {
            'temperature_c': 28.0,
            'humidity_percent': 70.0,
            'wind_speed_kmh': 10.0,
            'pressure_hpa': 1013.0,
            'weather_condition': 'Clear'
        }
    
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        'lat': HCMC_LAT,
        'lon': HCMC_LON,
        'appid': OPENWEATHER_API_KEY,
        'units': 'metric'
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        return {
            'temperature_c': data['main']['temp'],
            'humidity_percent': data['main']['humidity'],
            'pressure_hpa': data['main']['pressure'],
            'wind_speed_kmh': data['wind']['speed'] * 3.6,  # m/s to km/h
            'weather_condition': data['weather'][0]['main']
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Weather fetch failed: {e}")
        # Return default values
        return {
            'temperature_c': 28.0,
            'humidity_percent': 70.0,
            'wind_speed_kmh': 10.0,
            'pressure_hpa': 1013.0,
            'weather_condition': 'Unknown'
        }


def collect_traffic():
    """Main collection function."""
    timestamp = datetime.now()
    logger.info(f"=" * 80)
    logger.info(f"Starting traffic collection at {timestamp}")
    logger.info(f"=" * 80)
    
    # Load edges
    try:
        edges = load_edges()
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    # Fetch weather
    logger.info("Fetching weather data...")
    weather = fetch_weather()
    logger.info(f"Weather: {weather['temperature_c']:.1f}°C, {weather['weather_condition']}")
    
    # Collect traffic for each edge
    logger.info(f"Collecting traffic data for {len(edges)} edges...")
    traffic_data = []
    failed_count = 0
    
    for i, edge in enumerate(edges, 1):
        logger.info(f"[{i}/{len(edges)}] Fetching {edge['edge_id']}...")
        
        traffic = fetch_traffic_for_edge(edge)
        
        if traffic:
            # Add timestamp and weather
            traffic['timestamp'] = timestamp
            traffic.update(weather)
            traffic_data.append(traffic)
            
            logger.info(f"  ✓ Speed: {traffic['speed_kmh']:.1f} km/h")
        else:
            failed_count += 1
            logger.warning(f"  ✗ Failed")
        
        # Rate limiting - Google allows 50 req/sec
        time.sleep(0.1)  # 10 req/sec to be safe
    
    # Create DataFrame
    if not traffic_data:
        logger.error("No traffic data collected!")
        return
    
    df = pd.DataFrame(traffic_data)
    logger.info(f"\nCollected {len(df)} records ({failed_count} failed)")
    logger.info(f"Speed range: {df['speed_kmh'].min():.1f} - {df['speed_kmh'].max():.1f} km/h")
    logger.info(f"Average speed: {df['speed_kmh'].mean():.1f} km/h")
    
    # Save to monthly Parquet file
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    month_file = DATA_DIR / f"traffic_data_{timestamp:%Y%m}.parquet"
    
    if month_file.exists():
        # Append to existing file
        logger.info(f"Appending to existing file: {month_file}")
        df_existing = pd.read_parquet(month_file)
        df_combined = pd.concat([df_existing, df], ignore_index=True)
        df_combined.to_parquet(month_file, compression='snappy', index=False)
        logger.info(f"Total records in file: {len(df_combined):,}")
    else:
        # Create new file
        logger.info(f"Creating new file: {month_file}")
        df.to_parquet(month_file, compression='snappy', index=False)
    
    # Print file size
    file_size_mb = month_file.stat().st_size / (1024 * 1024)
    logger.info(f"File size: {file_size_mb:.2f} MB")
    
    logger.info(f"=" * 80)
    logger.info(f"✓ Collection completed at {datetime.now()}")
    logger.info(f"=" * 80)


if __name__ == '__main__':
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_MAPS_API_KEY not found in .env file!")
        logger.error("Set it in .env or export GOOGLE_MAPS_API_KEY=your_key")
        exit(1)
    
    try:
        collect_traffic()
    except KeyboardInterrupt:
        logger.info("\nCollection interrupted by user")
    except Exception as e:
        logger.error(f"Collection failed: {e}", exc_info=True)
        exit(1)
