"""
Overpass collector v5.0 with permanent caching
Static topology data - collect once, cache forever
"""

import requests
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
from traffic_forecast import PROJECT_ROOT
from traffic_forecast.collectors.area_utils import load_area_config
from traffic_forecast.collectors.overpass.node_selector import NodeSelector
import argparse
import yaml
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()
OVERPASS_URL = os.getenv('OVERPASS_URL', 'https://overpass.kumi.systems/api/interpreter')


def query_overpass(bbox):
    """Query Overpass API for major roads in the bounding box"""
    bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

    query = f"""
[out:json][timeout:120];
(
  way["highway"~"motorway|trunk|primary|secondary"]
  ({bbox_str});
);
out geom;
"""
    logger.info(f"Querying Overpass API for bbox: {bbox_str}")
    response = requests.post(OVERPASS_URL, data={'data': query}, timeout=300)
    response.raise_for_status()
    logger.info(f"Overpass API response received, status: {response.status_code}")
    return response.json()


def load_cache(cache_file):
    """Load cached topology data if exists and valid"""
    if not os.path.exists(cache_file):
        logger.info(f"Cache file not found: {cache_file}")
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # Check cache metadata
        cached_at = datetime.fromisoformat(cache_data['metadata']['cached_at'])
        cache_age_hours = (datetime.now() - cached_at).total_seconds() / 3600
        
        logger.info(f"Cache file found: {cache_file}")
        logger.info(f"Cached at: {cached_at}, Age: {cache_age_hours:.1f} hours")
        logger.info(f"Cached nodes: {cache_data['metadata']['total_nodes']}")
        logger.info(f"Cached edges: {cache_data['metadata']['total_edges']}")
        
        return cache_data
    
    except Exception as e:
        logger.error(f"Failed to load cache: {e}")
        return None


def save_cache(cache_file, nodes, edges, metadata):
    """Save topology data to cache"""
    cache_dir = os.path.dirname(cache_file)
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_data = {
        'metadata': {
            'cached_at': datetime.now().isoformat(),
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            **metadata
        },
        'nodes': nodes,
        'edges': edges
    }
    
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(cache_data, f, indent=2)
    
    logger.info(f"Cached topology data to: {cache_file}")


def run_overpass_collector():
    """Main collector function with permanent caching"""
    parser = argparse.ArgumentParser(description='Overpass OSM collector v5.0 - Cached topology')
    parser.add_argument('--mode', choices=['bbox', 'point_radius', 'circle'], help='Area selection mode')
    parser.add_argument('--bbox', help='bbox as min_lat,min_lon,max_lat,max_lon')
    parser.add_argument('--center', help='center as lon,lat')
    parser.add_argument('--radius', type=float, help='radius in meters')
    parser.add_argument('--force-refresh', action='store_true', help='Force refresh cache')
    parser.add_argument('--config', default='project_config.yaml', help='Config file name')
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

    area_cfg = load_area_config('overpass', cli_area=cli_area)

    # Load config
    config_path = PROJECT_ROOT / "configs" / args.config
    with config_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    
    collector_cfg = cfg['collectors']['overpass']
    cache_file = PROJECT_ROOT / collector_cfg.get('cache_file', 'cache/overpass_topology.json')
    use_cache = collector_cfg.get('use_cache', True)
    
    # Node selection config
    node_selection_cfg = cfg.get('node_selection', {})
    min_degree = node_selection_cfg.get('min_degree', 6)
    min_importance = node_selection_cfg.get('min_importance_score', 40.0)
    max_nodes = node_selection_cfg.get('max_nodes', 128)
    min_distance = node_selection_cfg.get('min_distance_meters', 200)
    road_type_filter = node_selection_cfg.get('road_type_filter', ['motorway', 'trunk', 'primary'])

    # Initialize NodeSelector
    node_selector = NodeSelector(
        min_degree=min_degree,
        min_importance_score=min_importance,
        max_nodes=max_nodes,
        min_distance_meters=min_distance,
        road_type_filter=road_type_filter
    )

    logger.info(
        f"NodeSelector: min_degree={min_degree}, min_importance={min_importance}, "
        f"max_nodes={max_nodes}, min_distance={min_distance}m, road_filter={road_type_filter}"
    )

    # Try to load from cache
    cache_data = None
    if use_cache and not args.force_refresh:
        cache_data = load_cache(cache_file)
    
    if cache_data:
        # Use cached data
        nodes = cache_data['nodes']
        edges = cache_data['edges']
        logger.info("Using cached topology data")
    else:
        # Fetch fresh data
        logger.info("Fetching fresh topology data from Overpass API")
        raw_data = query_overpass(area_cfg['bbox'])
        nodes, edges = node_selector.extract_major_intersections(raw_data)
        
        # Save to cache
        if use_cache:
            metadata = {
                'bbox': area_cfg['bbox'],
                'min_degree': min_degree,
                'min_importance': min_importance,
                'max_nodes': max_nodes,
                'min_distance': min_distance,
                'road_type_filter': road_type_filter
            }
            save_cache(cache_file, nodes, edges, metadata)

    # Get statistics
    stats = node_selector.get_statistics(nodes, edges)

    # Print statistics
    print("\n" + "=" * 60)
    print("TOPOLOGY EXTRACTION STATISTICS")
    print("=" * 60)
    print(f"Total Nodes: {stats['total_nodes']}")
    print(f"Total Edges: {stats['total_edges']}")
    print(f"Average Degree: {stats['avg_degree']:.2f}")
    print(f"Average Importance: {stats['avg_importance']:.2f}")
    print(f"Degree Range: {stats.get('min_degree', 0)} - {stats.get('max_degree', 0)}")
    print(f"Importance Range: {stats.get('min_importance', 0):.2f} - {stats.get('max_importance', 0):.2f}")
    print(f"\nRoad Type Distribution:")
    for road_type, count in sorted(stats['road_type_distribution'].items(), key=lambda x: -x[1]):
        print(f"  {road_type}: {count}")
    print("=" * 60)

    # Save to RUN_DIR if available, otherwise data/
    run_dir = os.getenv('RUN_DIR')
    if run_dir:
        out_dir = run_dir
    else:
        # Create timestamped run directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.join('data', 'runs', f'run_{timestamp}')
    
    os.makedirs(out_dir, exist_ok=True)
    
    nodes_file = os.path.join(out_dir, 'nodes.json')
    edges_file = os.path.join(out_dir, 'edges.json')
    stats_file = os.path.join(out_dir, 'statistics.json')
    
    with open(nodes_file, 'w', encoding='utf-8') as f:
        json.dump(nodes, f, indent=2)
    with open(edges_file, 'w', encoding='utf-8') as f:
        json.dump(edges, f, indent=2)
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Saved {len(nodes)} nodes to {nodes_file}")
    logger.info(f"Saved {len(edges)} edges to {edges_file}")
    logger.info(f"Saved statistics to {stats_file}")


if __name__ == "__main__":
    run_overpass_collector()
