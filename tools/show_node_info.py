"""
Quick node information viewer - displays current node data structure.

Shows what information is currently available in nodes.json and features_nodes_v2.json
"""

import json
from pathlib import Path
from typing import Dict, List


def analyze_node_structure(filepath: str = "data/nodes.json") -> Dict:
    """Analyze what fields are available in nodes.json"""
    with open(filepath, 'r', encoding='utf-8') as f:
        nodes = json.load(f)
    
    if not nodes:
        return {"error": "No nodes found"}
    
    # Analyze first node to see structure
    sample_node = nodes[0]
    all_fields = set()
    
    # Collect all unique fields across all nodes
    for node in nodes:
        all_fields.update(node.keys())
    
    return {
        "total_nodes": len(nodes),
        "available_fields": sorted(list(all_fields)),
        "sample_node": sample_node,
        "sample_count": min(5, len(nodes))
    }


def show_current_capabilities():
    """Show what node information is currently available"""
    
    print("=" * 80)
    print("CURRENT NODE DATA STRUCTURE ANALYSIS")
    print("=" * 80)
    
    # Analyze nodes.json
    print("\n1️⃣  NODES.JSON")
    print("-" * 80)
    
    try:
        info = analyze_node_structure("data/nodes.json")
        print(f"📊 Total nodes: {info['total_nodes']}")
        print(f"\n📋 Available fields:")
        for field in info['available_fields']:
            print(f"   ✓ {field}")
        
        print(f"\n📄 Sample node (first node):")
        sample = info['sample_node']
        for key, value in sample.items():
            print(f"   {key}: {value}")
    except FileNotFoundError:
        print("   ❌ File not found")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Analyze features_nodes_v2.json
    print("\n\n2️⃣  FEATURES_NODES_V2.JSON")
    print("-" * 80)
    
    try:
        info = analyze_node_structure("data/features_nodes_v2.json")
        print(f"📊 Total records: {info['total_nodes']}")
        print(f"\n📋 Available fields:")
        for field in info['available_fields']:
            print(f"   ✓ {field}")
        
        print(f"\n📄 Sample record (first record):")
        sample = info['sample_node']
        for key, value in sample.items():
            if key == 'feature_vector':
                print(f"   {key}: [array with {len(value)} elements]")
            else:
                print(f"   {key}: {value}")
    except FileNotFoundError:
        print("   ❌ File not found")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Show what we CAN extract
    print("\n\n3️⃣  WHAT CAN BE EXTRACTED")
    print("-" * 80)
    print("""
✅ CURRENTLY AVAILABLE:
   • node_id (unique identifier)
   • lat, lon (coordinates)
   • road_type (primary, secondary, tertiary, etc.)
   • lane_count (number of lanes)
   • speed_limit (speed limit if available)
   • Google Maps link (can be generated from lat/lon)
   
✅ CAN BE CALCULATED:
   • Google Maps URL: https://www.google.com/maps?q={lat},{lon}
   • Distance between nodes
   • Bounding box
   
❌ NOT CURRENTLY STORED (but CAN be queried from OSM):
   • Street names at intersection
   • Official intersection name
   • Connected way names
   • Way IDs
   
💡 TO GET STREET NAMES:
   Option 1: Query OSM Overpass API (slower, real-time)
   Option 2: Re-run collector with enhanced NodeSelector to cache street names
   Option 3: Use export_nodes_info.py tool to query and cache
""")
    
    # Show example usage
    print("\n4️⃣  EXAMPLE USAGE")
    print("-" * 80)
    print("""
# Export all node info to CSV/JSON/Markdown:
python tools/export_nodes_info.py --format all

# Export first 10 nodes only (for testing):
python tools/export_nodes_info.py --limit 10 --format md

# Query OSM API for street names (slower but complete):
python tools/export_nodes_info.py --use-osm-api --limit 5 --format md

# Quick view in Python:
import json
with open('data/nodes.json') as f:
    nodes = json.load(f)
    
for node in nodes[:5]:
    lat, lon = node['lat'], node['lon']
    gmaps = f"https://www.google.com/maps?q={lat},{lon}"
    print(f"{node['node_id']}: {gmaps}")
""")
    
    print("\n" + "=" * 80)


def generate_quick_csv():
    """Generate a quick CSV with basic info + Google Maps links"""
    import csv
    
    try:
        with open('data/nodes.json', 'r') as f:
            nodes = json.load(f)
        
        output_file = 'data/nodes_quick.csv'
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Node ID', 'Latitude', 'Longitude', 'Google Maps Link', 
                'Road Type', 'Lanes', 'Speed Limit'
            ])
            
            for node in nodes:
                lat = node.get('lat', 0)
                lon = node.get('lon', 0)
                gmaps = f"https://www.google.com/maps?q={lat},{lon}"
                
                writer.writerow([
                    node.get('node_id', ''),
                    lat,
                    lon,
                    gmaps,
                    node.get('road_type', ''),
                    node.get('lane_count', ''),
                    node.get('speed_limit', '')
                ])
        
        print(f"\n✅ Quick CSV generated: {output_file}")
        print(f"   Total nodes: {len(nodes)}")
        return output_file
    
    except Exception as e:
        print(f"\n❌ Error generating CSV: {e}")
        return None


if __name__ == '__main__':
    import sys
    
    show_current_capabilities()
    
    # Ask if user wants to generate quick CSV
    if len(sys.argv) > 1 and sys.argv[1] == '--generate-csv':
        print("\n📝 Generating quick CSV...")
        generate_quick_csv()
    else:
        print("\n💡 Tip: Run with --generate-csv to create a quick CSV file")
