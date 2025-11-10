"""
Fetch and cache route geometries from Google Maps Directions API.

This script:
1. Reads existing edges from topology data
2. Fetches route geometry for each edge using Google Maps
3. Saves geometries to cache file for fast loading in dashboard
4. Handles rate limiting and retries

Author: THAT Le Quang
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
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

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_MAPS_API_KEY not found in .env file")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
TOPOLOGY_FILE = PROJECT_ROOT / 'cache' / 'overpass_topology.json'
OUTPUT_FILE = PROJECT_ROOT / 'cache' / 'route_geometries.json'


def load_topology() -> Dict:
    """Load topology data with edges."""
    if not TOPOLOGY_FILE.exists():
        raise FileNotFoundError(f"Topology file not found: {TOPOLOGY_FILE}")
    
    with open(TOPOLOGY_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)


def fetch_directions(origin: Tuple[float, float], 
                     destination: Tuple[float, float],
                     retries: int = 3) -> Optional[List[Tuple[float, float]]]:
    """
    Fetch route geometry from Google Maps Directions API.
    
    Args:
        origin: (lat, lon) tuple
        destination: (lat, lon) tuple
        retries: Number of retry attempts
        
    Returns:
        List of (lat, lon) coordinates or None if failed
    """
    url = "https://maps.googleapis.com/maps/api/directions/json"
    
    params = {
        'origin': f"{origin[0]},{origin[1]}",
        'destination': f"{destination[0]},{destination[1]}",
        'mode': 'driving',
        'key': GOOGLE_API_KEY
    }
    
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] == 'OK' and data['routes']:
                # Extract polyline
                polyline = data['routes'][0]['overview_polyline']['points']
                # Decode polyline to coordinates
                coords = decode_polyline(polyline)
                return coords
            elif data['status'] == 'OVER_QUERY_LIMIT':
                logger.warning(f"Rate limit hit, waiting {(attempt + 1) * 2}s...")
                time.sleep((attempt + 1) * 2)
                continue
            else:
                logger.warning(f"API returned status: {data['status']}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed (attempt {attempt + 1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep((attempt + 1))
                continue
            return None
    
    return None


def decode_polyline(polyline_str: str) -> List[Tuple[float, float]]:
    """
    Decode Google Maps polyline encoding to lat/lon coordinates.
    
    Args:
        polyline_str: Encoded polyline string
        
    Returns:
        List of (lat, lon) tuples
    """
    coords = []
    index = 0
    lat = 0
    lng = 0
    
    while index < len(polyline_str):
        # Decode latitude
        result = 0
        shift = 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlat = ~(result >> 1) if (result & 1) else (result >> 1)
        lat += dlat
        
        # Decode longitude
        result = 0
        shift = 0
        while True:
            b = ord(polyline_str[index]) - 63
            index += 1
            result |= (b & 0x1f) << shift
            shift += 5
            if b < 0x20:
                break
        dlng = ~(result >> 1) if (result & 1) else (result >> 1)
        lng += dlng
        
        coords.append((lat / 1e5, lng / 1e5))
    
    return coords


def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate Euclidean distance in degrees (rough estimate)."""
    return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5


def process_edges(topology: Dict) -> Dict[str, List[Tuple[float, float]]]:
    """
    Process all edges and fetch route geometries.
    
    Args:
        topology: Topology data with edges
        
    Returns:
        Dictionary mapping edge keys to coordinate lists
    """
    geometries = {}
    edges = topology.get('edges', [])
    
    logger.info(f"Processing {len(edges)} edges...")
    
    short_edges = 0
    cached_edges = 0
    fetched_edges = 0
    failed_edges = 0
    
    for i, edge in enumerate(edges):
        lat_a = edge['lat_a']
        lon_a = edge['lon_a']
        lat_b = edge['lat_b']
        lon_b = edge['lon_b']
        
        # Create cache key
        key = f"{lat_a:.5f},{lon_a:.5f}-{lat_b:.5f},{lon_b:.5f}"
        
        # Skip if already processed (reverse direction)
        reverse_key = f"{lat_b:.5f},{lon_b:.5f}-{lat_a:.5f},{lon_a:.5f}"
        if reverse_key in geometries:
            # Use reversed coordinates
            geometries[key] = list(reversed(geometries[reverse_key]))
            cached_edges += 1
            continue
        
        # Calculate distance
        distance = calculate_distance(lat_a, lon_a, lat_b, lon_b)
        
        # For very short edges, use straight line
        if distance < 0.003:  # ~300m in degrees
            geometries[key] = [[lat_a, lon_a], [lat_b, lon_b]]
            short_edges += 1
        else:
            # Fetch from Google Maps
            coords = fetch_directions((lat_a, lon_a), (lat_b, lon_b))
            
            if coords:
                geometries[key] = coords
                fetched_edges += 1
                logger.info(f"[{i+1}/{len(edges)}] Fetched geometry for edge {key}")
            else:
                # Fallback to straight line
                geometries[key] = [[lat_a, lon_a], [lat_b, lon_b]]
                failed_edges += 1
                logger.warning(f"[{i+1}/{len(edges)}] Failed, using straight line for {key}")
        
        # Rate limiting: wait between requests
        if i % 10 == 0 and i > 0:
            logger.info(f"Progress: {i}/{len(edges)} - Short: {short_edges}, Cached: {cached_edges}, Fetched: {fetched_edges}, Failed: {failed_edges}")
            time.sleep(1)  # Longer pause every 10 requests
        else:
            time.sleep(0.1)  # Small delay between requests
    
    logger.info(f"\nProcessing complete!")
    logger.info(f"- Short edges (straight line): {short_edges}")
    logger.info(f"- Cached (reversed): {cached_edges}")
    logger.info(f"- Fetched from Google Maps: {fetched_edges}")
    logger.info(f"- Failed (fallback): {failed_edges}")
    logger.info(f"- Total geometries: {len(geometries)}")
    
    return geometries


def save_geometries(geometries: Dict[str, List[Tuple[float, float]]]) -> None:
    """Save geometries to cache file."""
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(geometries, f, indent=2)
    
    logger.info(f"Saved geometries to: {OUTPUT_FILE}")


def main():
    """Main execution."""
    try:
        logger.info("Starting route geometry fetch...")
        
        # Load topology
        topology = load_topology()
        logger.info(f"Loaded topology with {len(topology.get('edges', []))} edges")
        
        # Process edges
        geometries = process_edges(topology)
        
        # Save results
        save_geometries(geometries)
        
        logger.info("Done!")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
