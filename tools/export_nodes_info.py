"""
Export detailed node information including Google Maps links and street names.

This utility extracts comprehensive information about traffic nodes:
- Node ID and coordinates
- Google Maps link to the location
- Connected road names (from OSM data)
- Road types and intersection details
- Importance score and degree
"""

import json
import csv
from pathlib import Path
from typing import List, Dict, Optional
import requests
from urllib.parse import quote


def load_nodes(filepath: str = "data/nodes.json") -> List[Dict]:
 """Load nodes from JSON file."""
 with open(filepath, 'r', encoding='utf-8') as f:
 return json.load(f)


def get_google_maps_link(lat: float, lon: float) -> str:
 """Generate Google Maps link for coordinates."""
 return f"https://www.google.com/maps?q={lat},{lon}"


def get_street_names_from_osm(lat: float, lon: float, radius: int = 20) -> List[str]:
 """
 Query OSM Overpass API to get street names at intersection.
 
 Args:
 lat: Latitude
 lon: Longitude
 radius: Search radius in meters
 
 Returns:
 List of street names
 """
 overpass_url = "https://overpass-api.de/api/interpreter"
 
 # Overpass query to find ways near the point
 query = f"""
 [out:json];
 (
 way["highway"]["name"](around:{radius},{lat},{lon});
 );
 out body;
 """
 
 try:
 response = requests.get(overpass_url, params={'data': query}, timeout=30)
 if response.status_code == 200:
 data = response.json()
 street_names = []
 
 for element in data.get('elements', []):
 name = element.get('tags', {}).get('name')
 if name and name not in street_names:
 street_names.append(name)
 
 return street_names
 else:
 print(f"OSM API error: {response.status_code}")
 return []
 except Exception as e:
 print(f"Error fetching street names: {e}")
 return []


def get_road_info_from_cached_osm(lat: float, lon: float, 
 osm_cache_file: str = "cache/osm_ways.json") -> Dict:
 """
 Get road information from cached OSM data (if available).
 This is faster than querying OSM API every time.
 
 Returns:
 Dict with 'street_names' and 'road_types'
 """
 try:
 with open(osm_cache_file, 'r', encoding='utf-8') as f:
 osm_data = json.load(f)
 
 # Find ways near this coordinate
 node_key = (round(lat, 6), round(lon, 6))
 street_names = []
 road_types = []
 
 for element in osm_data.get('elements', []):
 if element['type'] != 'way':
 continue
 
 geoms = element.get('geometry', [])
 tags = element.get('tags', {})
 
 # Check if this way passes through our node
 for geom in geoms:
 way_lat = round(geom['lat'], 6)
 way_lon = round(geom['lon'], 6)
 
 if abs(way_lat - node_key[0]) < 0.0001 and abs(way_lon - node_key[1]) < 0.0001:
 name = tags.get('name', tags.get('ref', ''))
 road_type = tags.get('highway', 'unknown')
 
 if name and name not in street_names:
 street_names.append(name)
 if road_type and road_type not in road_types:
 road_types.append(road_type)
 break
 
 return {
 'street_names': street_names,
 'road_types': road_types
 }
 except FileNotFoundError:
 return {'street_names': [], 'road_types': []}


def enrich_node_info(node: Dict, use_osm_api: bool = False) -> Dict:
 """
 Enrich node with additional information.
 
 Args:
 node: Node dictionary
 use_osm_api: Whether to query OSM API for street names (slower but more accurate)
 
 Returns:
 Enriched node dictionary
 """
 lat = node['lat']
 lon = node['lon']
 
 # Add Google Maps link
 node['google_maps_link'] = get_google_maps_link(lat, lon)
 
 # Get street names
 if use_osm_api:
 street_names = get_street_names_from_osm(lat, lon)
 node['street_names'] = street_names
 node['intersection_description'] = " ∩ ".join(street_names) if street_names else "Unknown intersection"
 else:
 # Try to get from cached data first
 road_info = get_road_info_from_cached_osm(lat, lon)
 node['street_names'] = road_info['street_names']
 node['cached_road_types'] = road_info['road_types']
 node['intersection_description'] = " ∩ ".join(road_info['street_names']) if road_info['street_names'] else "Unknown intersection"
 
 return node


def export_to_csv(nodes: List[Dict], output_file: str = "data/nodes_detailed.csv"):
 """Export nodes to CSV with all details."""
 if not nodes:
 print("No nodes to export")
 return
 
 # Define CSV columns
 fieldnames = [
 'node_id',
 'lat',
 'lon',
 'google_maps_link',
 'intersection_description',
 'street_names',
 'road_type',
 'degree',
 'importance_score',
 'connected_road_types',
 'lane_count',
 'speed_limit'
 ]
 
 with open(output_file, 'w', newline='', encoding='utf-8') as f:
 writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
 writer.writeheader()
 
 for node in nodes:
 # Convert lists to strings for CSV
 if 'street_names' in node and isinstance(node['street_names'], list):
 node['street_names_str'] = ', '.join(node['street_names'])
 if 'connected_road_types' in node and isinstance(node['connected_road_types'], list):
 node['connected_road_types_str'] = ', '.join(node['connected_road_types'])
 
 writer.writerow({
 'node_id': node.get('node_id'),
 'lat': node.get('lat'),
 'lon': node.get('lon'),
 'google_maps_link': node.get('google_maps_link', ''),
 'intersection_description': node.get('intersection_description', ''),
 'street_names': node.get('street_names_str', ''),
 'road_type': node.get('road_type', ''),
 'degree': node.get('degree', ''),
 'importance_score': node.get('importance_score', ''),
 'connected_road_types': node.get('connected_road_types_str', ''),
 'lane_count': node.get('lane_count', ''),
 'speed_limit': node.get('speed_limit', '')
 })
 
 print(f"Exported {len(nodes)} nodes to {output_file}")


def export_to_json(nodes: List[Dict], output_file: str = "data/nodes_detailed.json"):
 """Export nodes to JSON with all details."""
 with open(output_file, 'w', encoding='utf-8') as f:
 json.dump(nodes, f, indent=2, ensure_ascii=False)
 
 print(f"Exported {len(nodes)} nodes to {output_file}")


def export_to_markdown(nodes: List[Dict], output_file: str = "data/NODES_INFO.md"):
 """Export nodes to a human-readable Markdown table."""
 with open(output_file, 'w', encoding='utf-8') as f:
 f.write("# Traffic Nodes Information\n\n")
 f.write(f"Total nodes: **{len(nodes)}**\n\n")
 f.write("Generated: " + str(Path(__file__).stat().st_mtime) + "\n\n")
 
 f.write("## Node List\n\n")
 f.write("| # | Node ID | Coordinates | Google Maps | Intersection | Road Type | Degree |\n")
 f.write("|---|---------|-------------|-------------|--------------|-----------|--------|\n")
 
 for i, node in enumerate(nodes, 1):
 node_id = node.get('node_id', 'N/A')
 lat = node.get('lat', 0)
 lon = node.get('lon', 0)
 coords = f"{lat:.6f}, {lon:.6f}"
 gmaps = node.get('google_maps_link', '')
 intersection = node.get('intersection_description', 'Unknown')
 road_type = node.get('road_type', 'unknown')
 degree = node.get('degree', 'N/A')
 
 # Truncate long intersection names
 if len(intersection) > 40:
 intersection = intersection[:37] + "..."
 
 gmaps_link = f"[View]({gmaps})" if gmaps else "N/A"
 
 f.write(f"| {i} | `{node_id}` | {coords} | {gmaps_link} | {intersection} | {road_type} | {degree} |\n")
 
 # Add detailed section
 f.write("\n## Detailed Information\n\n")
 for i, node in enumerate(nodes, 1):
 f.write(f"### {i}. {node.get('node_id', 'N/A')}\n\n")
 f.write(f"- **Coordinates**: {node.get('lat', 0):.6f}, {node.get('lon', 0):.6f}\n")
 f.write(f"- **Google Maps**: [{node.get('google_maps_link', '')}]({node.get('google_maps_link', '')})\n")
 f.write(f"- **Intersection**: {node.get('intersection_description', 'Unknown')}\n")
 
 if node.get('street_names'):
 f.write(f"- **Streets**:\n")
 for street in node['street_names']:
 f.write(f" - {street}\n")
 
 f.write(f"- **Road Type**: {node.get('road_type', 'unknown')}\n")
 f.write(f"- **Degree**: {node.get('degree', 'N/A')} connecting roads\n")
 
 if node.get('importance_score'):
 f.write(f"- **Importance Score**: {node.get('importance_score', 0):.1f}\n")
 
 if node.get('connected_road_types'):
 f.write(f"- **Connected Road Types**: {', '.join(node['connected_road_types'])}\n")
 
 f.write("\n---\n\n")
 
 print(f"Exported {len(nodes)} nodes to {output_file}")


def main():
 """Main function to export node information."""
 import argparse
 
 parser = argparse.ArgumentParser(description='Export detailed traffic node information')
 parser.add_argument('--input', default='data/nodes.json', help='Input nodes JSON file')
 parser.add_argument('--output-csv', default='data/nodes_detailed.csv', help='Output CSV file')
 parser.add_argument('--output-json', default='data/nodes_detailed.json', help='Output JSON file')
 parser.add_argument('--output-md', default='data/NODES_INFO.md', help='Output Markdown file')
 parser.add_argument('--use-osm-api', action='store_true', help='Query OSM API for street names (slower)')
 parser.add_argument('--limit', type=int, help='Limit number of nodes to process')
 parser.add_argument('--format', choices=['csv', 'json', 'md', 'all'], default='all', 
 help='Output format')
 
 args = parser.parse_args()
 
 print("Loading nodes...")
 nodes = load_nodes(args.input)
 
 if args.limit:
 nodes = nodes[:args.limit]
 print(f"Limited to {args.limit} nodes")
 
 print(f"Processing {len(nodes)} nodes...")
 
 # Enrich nodes with additional information
 enriched_nodes = []
 for i, node in enumerate(nodes, 1):
 print(f"Processing node {i}/{len(nodes)}: {node.get('node_id', 'N/A')}", end='\r')
 enriched = enrich_node_info(node, use_osm_api=args.use_osm_api)
 enriched_nodes.append(enriched)
 
 print() # New line after progress
 
 # Export to requested formats
 if args.format in ['csv', 'all']:
 export_to_csv(enriched_nodes, args.output_csv)
 
 if args.format in ['json', 'all']:
 export_to_json(enriched_nodes, args.output_json)
 
 if args.format in ['md', 'all']:
 export_to_markdown(enriched_nodes, args.output_md)
 
 print("\nExport complete!")
 print(f"\nSample node info:")
 if enriched_nodes:
 sample = enriched_nodes[0]
 print(f" Node ID: {sample.get('node_id')}")
 print(f" Coordinates: {sample.get('lat')}, {sample.get('lon')}")
 print(f" Google Maps: {sample.get('google_maps_link')}")
 print(f" Intersection: {sample.get('intersection_description')}")
 if sample.get('street_names'):
 print(f" Streets: {', '.join(sample['street_names'])}")


if __name__ == '__main__':
 main()
