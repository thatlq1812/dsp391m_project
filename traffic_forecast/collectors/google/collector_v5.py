"""
Google Directions collector v5.0 - Real API only with retry mechanism
No mock API fallback - production ready
"""

import json
import os
import time
import concurrent.futures
import threading
from dotenv import load_dotenv
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2
import yaml
from pathlib import Path
from traffic_forecast import PROJECT_ROOT
from traffic_forecast.collectors.area_utils import load_area_config
import argparse

load_dotenv()


class RateLimiter:
    """Thread-safe rate limiter for Google Maps API"""
    
    def __init__(self, requests_per_minute=2800):
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_minute / 60
        self.min_interval = 1.0 / self.requests_per_second
        self.last_request_time = 0
        self.lock = threading.Lock()
        self.request_count = 0
        self.window_start = time.time()

    def wait_if_needed(self):
        """Wait if necessary to maintain rate limit"""
        with self.lock:
            current_time = time.time()

            # Reset counter every minute
            if current_time - self.window_start >= 60:
                self.request_count = 0
                self.window_start = current_time

            # Check if exceeded per-minute limit
            if self.request_count >= self.requests_per_minute:
                sleep_time = 60 - (current_time - self.window_start)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                self.request_count = 0
                self.window_start = time.time()
                return

            # Enforce minimum interval between requests
            time_since_last = current_time - self.last_request_time
            if time_since_last < self.min_interval and self.last_request_time > 0:
                sleep_time = self.min_interval - time_since_last
                time.sleep(sleep_time)

            self.last_request_time = time.time()
            self.request_count += 1


def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km"""
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def load_nodes():
    """Load nodes from RUN_DIR or fallback to global nodes.json"""
    run_dir = os.getenv('RUN_DIR')
    if run_dir:
        nodes_path = os.path.join(run_dir, 'collectors', 'overpass', 'nodes.json')
        if os.path.exists(nodes_path):
            with open(nodes_path, 'r') as f:
                return json.load(f)

    nodes_path = PROJECT_ROOT / 'data' / 'nodes.json'
    if nodes_path.exists():
        with nodes_path.open(encoding='utf-8') as f:
            return json.load(f)

    raise FileNotFoundError("Could not find nodes.json in RUN_DIR or data/")


def real_directions_api(origin, dest, api_key, rate_limiter, config):
    """
    Real Google Directions API call with rate limiting and retry
    Returns None on failure after all retries
    """
    import requests
    
    collector_config = config['collectors']['google_directions']
    max_retries = collector_config.get('retry_on_failure', 3)
    timeout = collector_config.get('timeout_seconds', 10)
    
    for attempt in range(max_retries):
        try:
            # Apply rate limiting before request
            rate_limiter.wait_if_needed()

            base_url = "https://maps.googleapis.com/maps/api/directions/json"
            params = {
                'origin': f"{origin['lat']},{origin['lon']}",
                'destination': f"{dest['lat']},{dest['lon']}",
                'key': api_key,
                'mode': 'driving',
                'departure_time': 'now',
                'traffic_model': 'best_guess'
            }

            start_time = time.time()
            response = requests.get(base_url, params=params, timeout=timeout)
            response_time = time.time() - start_time
            response.raise_for_status()
            data = response.json()

            if data['status'] == 'OK' and data['routes']:
                route = data['routes'][0]
                leg = route['legs'][0]

                distance_km = leg['distance']['value'] / 1000
                
                # Use traffic-aware duration if available, otherwise use regular duration
                if 'duration_in_traffic' in leg:
                    duration_sec = leg['duration_in_traffic']['value']
                else:
                    duration_sec = leg['duration']['value']
                    
                speed_kmh = (distance_km / (duration_sec / 3600)) if duration_sec > 0 else 30

                # Log every 50th successful call
                if not hasattr(real_directions_api, '_call_count'):
                    real_directions_api._call_count = 0
                real_directions_api._call_count += 1

                if real_directions_api._call_count % 50 == 0:
                    print(f"API call #{real_directions_api._call_count} successful in {response_time:.2f}s: {speed_kmh:.1f} km/h")

                return {
                    'distance_km': distance_km,
                    'duration_sec': duration_sec,
                    'speed_kmh': speed_kmh,
                    'status': 'OK',
                    'has_traffic_data': 'duration_in_traffic' in leg
                }
            else:
                error_msg = data.get('status', 'Unknown')
                if attempt < max_retries - 1:
                    print(f"Google API error: {error_msg}, retry {attempt + 1}/{max_retries}")
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
                    continue
                else:
                    print(f"Google API error after {max_retries} attempts: {error_msg}")
                    return None

        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                print(f"Google API timeout, retry {attempt + 1}/{max_retries}")
                time.sleep(1 * (attempt + 1))
                continue
            else:
                print(f"Google API timeout after {max_retries} attempts")
                return None
                
        except requests.exceptions.RequestException as e:
            if attempt < max_retries - 1:
                print(f"Google API request failed: {e}, retry {attempt + 1}/{max_retries}")
                time.sleep(1 * (attempt + 1))
                continue
            else:
                print(f"Google API request failed after {max_retries} attempts: {e}")
                return None
                
        except Exception as e:
            print(f"Unexpected error in Google API: {e}")
            return None
    
    return None


def process_edge_batch(edge_batch, api_key, rate_limiter, config):
    """Process a batch of edges, returning results"""
    results = []
    failed_count = 0
    
    for node_a, node_b, dist in edge_batch:
        directions = real_directions_api(node_a, node_b, api_key, rate_limiter, config)
        
        if directions is None:
            failed_count += 1
            print(f"FAILED: Could not get traffic data for edge {node_a['node_id']} -> {node_b['node_id']}")
            continue

        result = {
            'node_a_id': node_a['node_id'],
            'node_b_id': node_b['node_id'],
            'distance_km': directions['distance_km'],
            'duration_sec': directions['duration_sec'],
            'speed_kmh': directions['speed_kmh'],
            'has_traffic_data': directions.get('has_traffic_data', False),
            'timestamp': datetime.now().isoformat(),
            'api_type': 'real'
        }
        results.append(result)
    
    if failed_count > 0:
        print(f"Batch completed with {failed_count} failed requests")
    
    return results


def find_road_segments(nodes, config):
    """Find road segments by connecting consecutive nodes on the same way"""
    run_dir = os.getenv('RUN_DIR')
    if run_dir:
        ways_path = os.path.join(run_dir, 'collectors', 'overpass', 'way_sequences.json')
    else:
        ways_path = os.path.join('data', 'way_sequences.json')

    try:
        with open(ways_path, 'r', encoding='utf-8') as f:
            way_sequences = json.load(f)
    except FileNotFoundError:
        print(f"Warning: {ways_path} not found, falling back to nearest neighbors")
        return find_nearest_neighbors(nodes, config)

    node_dict = {node['node_id']: node for node in nodes}

    # Calculate node degrees
    node_degrees = {}
    for way in way_sequences:
        nodes_seq = way['nodes']
        for node_id in nodes_seq:
            if node_id in node_dict:
                node_degrees[node_id] = node_degrees.get(node_id, 0) + 1

    # Find intersection nodes (degree > 2)
    intersection_nodes = {node_id: node_dict[node_id] for node_id, degree in node_degrees.items() if degree > 2}

    print(f"Found {len(intersection_nodes)} intersection nodes (degree > 2) out of {len(node_dict)} total nodes")

    edges = []
    processed_pairs = set()

    for way in way_sequences:
        nodes_seq = way['nodes']
        intersection_indices = []
        for i, node_id in enumerate(nodes_seq):
            if node_id in intersection_nodes:
                intersection_indices.append(i)

        # Create edges between consecutive intersections
        for i in range(len(intersection_indices) - 1):
            start_idx = intersection_indices[i]
            end_idx = intersection_indices[i + 1]

            start_node_id = nodes_seq[start_idx]
            end_node_id = nodes_seq[end_idx]

            pair_key = tuple(sorted([start_node_id, end_node_id]))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)

            if start_node_id in node_dict and end_node_id in node_dict:
                node_a = node_dict[start_node_id]
                node_b = node_dict[end_node_id]
                dist = haversine(node_a['lat'], node_a['lon'], node_b['lat'], node_b['lon'])
                edges.append((node_a, node_b, dist))

    print(f"Created {len(edges)} intersection-to-intersection edges from {len(way_sequences)} ways")
    return edges


def find_nearest_neighbors(nodes, config):
    """Find k nearest neighbors within radius for each node"""
    collector_config = config['collectors']['google_directions']
    edges = []
    limit_nodes = collector_config.get('limit_nodes', len(nodes))
    
    for i, node_a in enumerate(nodes[:limit_nodes]):
        neighbors = []
        for j, node_b in enumerate(nodes):
            if i == j:
                continue
            dist = haversine(node_a['lat'], node_a['lon'], node_b['lat'], node_b['lon'])
            if dist <= collector_config['radius_km']:
                neighbors.append((dist, node_b))
        neighbors.sort(key=lambda x: x[0])
        for dist, node_b in neighbors[:collector_config['k_neighbors']]:
            edges.append((node_a, node_b, dist))
    
    return edges


def run_google_collector():
    """Main entry point for Google Directions collector"""
    parser = argparse.ArgumentParser(description='Google Directions collector v5.0 - Real API only')
    parser.add_argument('--mode', choices=['bbox', 'point_radius', 'circle'], help='Area selection mode')
    parser.add_argument('--bbox', help='bbox as min_lat,min_lon,max_lat,max_lon')
    parser.add_argument('--center', help='center as lon,lat')
    parser.add_argument('--radius', type=float, help='radius in meters')
    parser.add_argument('--config', default='project_config_v5.yaml', help='Config file name')
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

    # Load configuration
    config_path = PROJECT_ROOT / "configs" / args.config
    with config_path.open(encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    
    collector_config = config['collectors']['google_directions']

    # Resolve area and filter nodes
    area_cfg = load_area_config('google', cli_area=cli_area)

    nodes = load_nodes()
    nodes = [n for n in nodes if (area_cfg['bbox'][0] <= n['lat'] <= area_cfg['bbox'][2]
                                  and area_cfg['bbox'][1] <= n['lon'] <= area_cfg['bbox'][3])]
    
    print(f"Loaded {len(nodes)} nodes within area")
    
    edges = find_road_segments(nodes, config)

    # Get API key
    api_key = os.getenv(collector_config.get('api_key_env', 'GOOGLE_MAPS_API_KEY'))
    if not api_key or len(api_key) < 10:
        raise ValueError(
            "GOOGLE_MAPS_API_KEY environment variable not set or invalid. "
            "Real API is required in v5.0 (no mock fallback)."
        )

    # Initialize rate limiter
    rate_limiter = RateLimiter(requests_per_minute=collector_config['rate_limit_requests_per_minute'])

    print(f"Using REAL Google Directions API (no mock fallback)")
    print(f"Rate limiting: {rate_limiter.requests_per_minute} requests/minute")

    # Test limit for development
    test_limit = int(os.getenv('GOOGLE_TEST_LIMIT', '0'))
    if test_limit > 0:
        edges = edges[:test_limit]
        print(f"TEST MODE: Processing only {test_limit} edges")

    total_edges = len(edges)
    print(f"Processing {total_edges} traffic edges with parallel processing...")

    # Process edges in parallel batches
    batch_size = 10
    traffic_data = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = []
        for i in range(0, total_edges, batch_size):
            batch = edges[i:i + batch_size]
            future = executor.submit(process_edge_batch, batch, api_key, rate_limiter, config)
            futures.append(future)

        # Collect results
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            try:
                batch_results = future.result()
                traffic_data.extend(batch_results)
                completed += len(batch_results)
                if completed % 100 == 0:
                    print(f"Processed {completed}/{total_edges} edges...")
            except Exception as e:
                print(f"Batch processing failed: {e}")

    # Save results
    run_dir = os.getenv('RUN_DIR')
    if run_dir:
        out_dir = os.path.join(run_dir, 'collectors', 'google')
    else:
        out_dir = os.path.join('data')
    os.makedirs(out_dir, exist_ok=True)
    
    output_file = os.path.join(out_dir, 'traffic_edges.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(traffic_data, f, indent=2)

    # Summary
    total_collected = len(traffic_data)
    failed_count = total_edges - total_collected
    success_rate = (total_collected / total_edges * 100) if total_edges > 0 else 0
    
    print(f"\n=== Collection Summary ===")
    print(f"Total edges: {total_edges}")
    print(f"Successful: {total_collected}")
    print(f"Failed: {failed_count}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    run_google_collector()
