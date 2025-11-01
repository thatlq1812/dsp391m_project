#!/usr/bin/env python3
"""
Merge all run directories into final dataset
Useful after 3-day collection period
"""

import json
import glob
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent

def merge_collections(runs_dir='data/runs', output_dir='data/final'):
    """Merge all run directories"""
    
    runs_path = PROJECT_ROOT / runs_dir
    output_path = PROJECT_ROOT / output_dir
    output_path.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("MERGING COLLECTION DATA")
    print("=" * 70)
    print()
    
    # Find all run directories
    run_dirs = sorted(glob.glob(str(runs_path / 'run_*')))
    
    print(f"Found {len(run_dirs)} runs")
    print()
    
    # Merge traffic data
    all_traffic = []
    traffic_by_edge = defaultdict(list)  # For time-series analysis
    
    for i, run_dir in enumerate(run_dirs, 1):
        traffic_file = Path(run_dir) / 'traffic_edges.json'
        
        if traffic_file.exists():
            with open(traffic_file) as f:
                data = json.load(f)
                all_traffic.extend(data)
                
                # Group by edge for time-series
                for record in data:
                    edge_key = f"{record['node_a_id']}-->{record['node_b_id']}"
                    traffic_by_edge[edge_key].append(record)
        
        if i % 10 == 0:
            print(f"Processed {i}/{len(run_dirs)} runs...")
    
    # Merge weather data
    all_weather = []
    weather_by_node = defaultdict(list)  # For time-series analysis
    
    for i, run_dir in enumerate(run_dirs, 1):
        weather_file = Path(run_dir) / 'weather_snapshot.json'
        
        if weather_file.exists():
            with open(weather_file) as f:
                data = json.load(f)
                all_weather.extend(data)
                
                # Group by node for time-series
                for record in data:
                    node_key = record['node_id']
                    weather_by_node[node_key].append(record)
        
        if i % 10 == 0:
            print(f"Processed {i}/{len(run_dirs)} runs...")
    
    print()
    print("=" * 70)
    print("SAVING MERGED DATA")
    print("=" * 70)
    print()
    
    # Save complete dataset
    traffic_output = output_path / 'traffic_complete.json'
    with open(traffic_output, 'w') as f:
        json.dump(all_traffic, f, indent=2)
    print(f"✓ Saved {len(all_traffic):,} traffic records to {traffic_output}")
    
    weather_output = output_path / 'weather_complete.json'
    with open(weather_output, 'w') as f:
        json.dump(all_weather, f, indent=2)
    print(f"✓ Saved {len(all_weather):,} weather records to {weather_output}")
    
    # Save time-series format (easier for ML)
    traffic_timeseries = output_path / 'traffic_timeseries.json'
    with open(traffic_timeseries, 'w') as f:
        json.dump(traffic_by_edge, f, indent=2)
    print(f"✓ Saved {len(traffic_by_edge)} edges time-series to {traffic_timeseries}")
    
    weather_timeseries = output_path / 'weather_timeseries.json'
    with open(weather_timeseries, 'w') as f:
        json.dump(weather_by_node, f, indent=2)
    print(f"✓ Saved {len(weather_by_node)} nodes weather time-series to {weather_timeseries}")
    
    # Generate statistics
    print()
    print("=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print()
    
    # Time range
    if all_traffic:
        timestamps = [r['timestamp'] for r in all_traffic]
        first_ts = min(timestamps)
        last_ts = max(timestamps)
        print(f"Time range:")
        print(f"  First: {first_ts}")
        print(f"  Last:  {last_ts}")
        print()
    
    # Traffic stats
    print(f"Traffic data:")
    print(f"  Total records: {len(all_traffic):,}")
    print(f"  Unique edges: {len(traffic_by_edge)}")
    if traffic_by_edge:
        collections_per_edge = [len(records) for records in traffic_by_edge.values()]
        print(f"  Collections per edge: {min(collections_per_edge)}-{max(collections_per_edge)} (avg: {sum(collections_per_edge)/len(collections_per_edge):.1f})")
    
    # Speed stats
    speeds = [r['speed_kmh'] for r in all_traffic if 'speed_kmh' in r]
    if speeds:
        print(f"  Speed range: {min(speeds):.1f} - {max(speeds):.1f} km/h")
        print(f"  Average speed: {sum(speeds)/len(speeds):.1f} km/h")
    
    print()
    
    # Weather stats
    print(f"Weather data:")
    print(f"  Total records: {len(all_weather):,}")
    print(f"  Unique nodes: {len(weather_by_node)}")
    if weather_by_node:
        collections_per_node = [len(records) for records in weather_by_node.values()]
        print(f"  Collections per node: {min(collections_per_node)}-{max(collections_per_node)} (avg: {sum(collections_per_node)/len(collections_per_node):.1f})")
    
    print()
    print("=" * 70)
    print("MERGE COMPLETE!")
    print("=" * 70)
    print()
    print("Output files:")
    print(f"  {traffic_output}")
    print(f"  {weather_output}")
    print(f"  {traffic_timeseries}")
    print(f"  {weather_timeseries}")
    print()
    print("Ready for ML training!")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Merge run directories')
    parser.add_argument('--input-dir', default='data/runs',
                       help='Directory with run folders')
    parser.add_argument('--output-dir', default='data/final',
                       help='Output directory for merged data')
    
    args = parser.parse_args()
    
    merge_collections(args.input_dir, args.output_dir)
