"""
Overpass collector for OSM topology.
Updated to collect only major intersections instead of dense node spacing.
Includes data validation and quality checks.
"""

import requests
import json
import os
from dotenv import load_dotenv
from traffic_forecast import PROJECT_ROOT
from traffic_forecast.collectors.area_utils import load_area_config
from traffic_forecast.collectors.cache_utils import get_or_create_cache
from traffic_forecast.collectors.overpass.node_selector import NodeSelector
from traffic_forecast.validation.schemas import validate_nodes, validate_edges, generate_quality_report
import argparse
import math
import yaml
import logging

# Configure logging
logging.basicConfig(
 level=logging.INFO,
 format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()
OVERPASS_URL = os.getenv('OVERPASS_URL', 'https://overpass.kumi.systems/api/interpreter')


def query_overpass(bbox):
 """
 Query Overpass API for major roads in the bounding box.
 Returns ways with geometry for intersection extraction.
 """
 # Overpass expects south,west,north,east i.e. min_lat,min_lon,max_lat,max_lon
 bbox_str = f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
 
 # Query for major roads: motorway, trunk, primary, secondary
 # These are the roads where major intersections matter most
 query = f"""
[out:json][timeout:120];
(
 way["highway"~"motorway|trunk|primary|secondary"]
 ({bbox_str});
);
out geom;
"""
 print(f"Querying Overpass API for major roads in bbox: {bbox_str}")
 response = requests.post(OVERPASS_URL, data={'data': query}, timeout=300) # 5 minute timeout
 response.raise_for_status() # Raise exception for bad status codes
 print(f"Overpass API response received, status: {response.status_code}")
 return response.json()


def run_overpass_collector():
 """
 Main collector function using NodeSelector for major intersections only.
 """
 parser = argparse.ArgumentParser(description='Overpass OSM collector - Major Intersections')
 parser.add_argument('--mode', choices=['bbox', 'point_radius', 'circle'], help='Area selection mode')
 parser.add_argument('--bbox', help='bbox as min_lat,min_lon,max_lat,max_lon')
 parser.add_argument('--center', help='center as lon,lat')
 parser.add_argument('--radius', type=float, help='radius in meters')
 parser.add_argument('--min-degree', type=int, default=3, help='Minimum intersection degree (default: 3)')
 parser.add_argument('--min-importance', type=float, default=15.0, help='Minimum importance score (default: 15.0)')
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

 # Load config for cache settings
 config_path = PROJECT_ROOT / "configs" / "project_config.yaml"
 with config_path.open(encoding="utf-8") as fh:
 cfg = yaml.safe_load(fh) or {}
 collector_cfg = cfg['collectors']['overpass']
 cache_enabled = collector_cfg.get('cache_enabled', True)
 cache_expiry_hours = collector_cfg.get('cache_expiry_hours', 168) # 1 week default
 cache_dir = cfg['data'].get('cache_dir', './cache')

 # Get node selection config (academic v4.0)
 node_selection_cfg = cfg.get('node_selection', {})
 max_nodes = node_selection_cfg.get('max_nodes')
 road_type_filter = node_selection_cfg.get('road_type_filter')

 # Initialize NodeSelector with config from project_config.yaml
 node_selector = NodeSelector(
 min_degree=args.min_degree,
 min_importance_score=args.min_importance,
 max_nodes=max_nodes,
 road_type_filter=road_type_filter
 )
 
 logger.info(
 f"NodeSelector initialized: min_degree={args.min_degree}, "
 f"min_importance={args.min_importance}, max_nodes={max_nodes}, "
 f"road_filter={road_type_filter}"
 )

 # Get data from cache or fetch fresh
 if cache_enabled:
 cache_params = {
 'bbox': area_cfg['bbox'],
 'min_degree': args.min_degree,
 'min_importance': args.min_importance,
 'max_nodes': max_nodes,
 'road_type_filter': road_type_filter
 }
 
 def fetch_processed_data():
 raw_data = query_overpass(area_cfg['bbox'])
 nodes, edges = node_selector.extract_major_intersections(raw_data)
 stats = node_selector.get_statistics(nodes, edges)
 return {'nodes': nodes, 'edges': edges, 'statistics': stats}
 
 cached_result = get_or_create_cache(
 collector_name='overpass_major_intersections',
 params=cache_params,
 cache_dir=cache_dir,
 expiry_hours=cache_expiry_hours,
 fetch_func=fetch_processed_data
 )
 nodes, edges, stats = cached_result['nodes'], cached_result['edges'], cached_result['statistics']
 else:
 data = query_overpass(area_cfg['bbox'])
 nodes, edges = node_selector.extract_major_intersections(data)
 stats = node_selector.get_statistics(nodes, edges)

 # Print statistics
 print("\n" + "="*60)
 print("MAJOR INTERSECTIONS EXTRACTION STATISTICS")
 print("="*60)
 print(f"Total Nodes (Major Intersections): {stats['total_nodes']}")
 print(f"Total Edges: {stats['total_edges']}")
 print(f"Average Degree: {stats['avg_degree']}")
 print(f"Average Importance Score: {stats['avg_importance']}")
 print(f"Degree Range: {stats.get('min_degree', 'N/A')} - {stats.get('max_degree', 'N/A')}")
 print(f"Importance Range: {stats.get('min_importance', 'N/A'):.2f} - {stats.get('max_importance', 'N/A'):.2f}")
 print(f"\nRoad Type Distribution:")
 for road_type, count in sorted(stats['road_type_distribution'].items(), key=lambda x: -x[1]):
 print(f" {road_type}: {count}")
 print("="*60 + "\n")

 # Validate data quality
 logger.info("Validating data quality...")
 valid_nodes, node_errors = validate_nodes(nodes)
 valid_edges, edge_errors = validate_edges(edges)
 
 # Generate quality reports
 node_quality = generate_quality_report(
 dataset_name="traffic_nodes",
 total_records=len(nodes),
 valid_records=len(valid_nodes),
 validation_errors=node_errors[:10] # Keep first 10 errors
 )
 
 edge_quality = generate_quality_report(
 dataset_name="traffic_edges",
 total_records=len(edges),
 valid_records=len(valid_edges),
 validation_errors=edge_errors[:10]
 )
 
 print("\n" + "="*60)
 print("DATA QUALITY REPORT")
 print("="*60)
 print(f"Nodes - Valid: {node_quality.valid_records}/{node_quality.total_records} ({node_quality.validity_pct}%)")
 if node_errors:
 print(f" Errors: {len(node_errors)} (showing first 3):")
 for err in node_errors[:3]:
 print(f" - {err}")
 
 print(f"\nEdges - Valid: {edge_quality.valid_records}/{edge_quality.total_records} ({edge_quality.validity_pct}%)")
 if edge_errors:
 print(f" Errors: {len(edge_errors)} (showing first 3):")
 for err in edge_errors[:3]:
 print(f" - {err}")
 print("="*60 + "\n")

 # Use validated data (convert back to dict for JSON serialization)
 nodes_to_save = [n.dict() for n in valid_nodes] if valid_nodes else nodes
 edges_to_save = [e.dict() for e in valid_edges] if valid_edges else edges

 # Use validated data (convert back to dict for JSON serialization)
 nodes_to_save = [n.dict() for n in valid_nodes] if valid_nodes else nodes
 edges_to_save = [e.dict() for e in valid_edges] if valid_edges else edges

 # Prefer writing into RUN_DIR if provided (per-run collectors folder)
 run_dir = os.getenv('RUN_DIR')
 if run_dir:
 out_dir = os.path.join(run_dir, 'collectors', 'overpass')
 else:
 out_dir = os.path.join('data')
 os.makedirs(out_dir, exist_ok=True)
 
 # Save nodes
 out_path_nodes = os.path.join(out_dir, 'nodes.json')
 with open(out_path_nodes, 'w', encoding='utf-8') as f:
 json.dump(nodes_to_save, f, indent=2)
 
 # Save edges
 out_path_edges = os.path.join(out_dir, 'edges.json')
 with open(out_path_edges, 'w', encoding='utf-8') as f:
 json.dump(edges_to_save, f, indent=2)
 
 # Save statistics
 out_path_stats = os.path.join(out_dir, 'statistics.json')
 with open(out_path_stats, 'w', encoding='utf-8') as f:
 json.dump(stats, f, indent=2)
 
 # Save quality reports
 out_path_quality = os.path.join(out_dir, 'quality_report.json')
 quality_report = {
 'nodes': node_quality.model_dump() if hasattr(node_quality, 'model_dump') else node_quality.dict(),
 'edges': edge_quality.model_dump() if hasattr(edge_quality, 'model_dump') else edge_quality.dict()
 }
 # Convert datetime objects to string
 def json_serial(obj):
 """JSON serializer for objects not serializable by default json code"""
 from datetime import datetime, date
 if isinstance(obj, (datetime, date)):
 return obj.isoformat()
 raise TypeError(f"Type {type(obj)} not serializable")
 
 with open(out_path_quality, 'w', encoding='utf-8') as f:
 json.dump(quality_report, f, indent=2, default=json_serial)

 logger.info(f" Saved {len(nodes_to_save)} major intersections to {out_path_nodes}")
 logger.info(f" Saved {len(edges_to_save)} edges to {out_path_edges}")
 logger.info(f" Saved statistics to {out_path_stats}")
 logger.info(f" Saved quality report to {out_path_quality}")



if __name__ == "__main__":
 run_overpass_collector()
