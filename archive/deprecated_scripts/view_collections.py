#!/usr/bin/env python3
"""
Quick view of collection progress
Shows how many runs, time range, etc.
"""

import json
import glob
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent

def view_collection_stats(runs_dir='data/runs'):
    """View statistics of collected runs"""
    
    runs_path = PROJECT_ROOT / runs_dir
    
    print("=" * 70)
    print("COLLECTION STATISTICS")
    print("=" * 70)
    print()
    
    # Find all run directories
    run_dirs = sorted(glob.glob(str(runs_path / 'run_*')))
    
    print(f"Total runs: {len(run_dirs)}")
    print()
    
    if not run_dirs:
        print("No runs yet!")
        return
    
    # Parse timestamps from directory names
    def extract_timestamp(dirname):
        # run_20251029_084417 -> 20251029_084417
        parts = Path(dirname).name.split('_', 1)
        if len(parts) == 2:
            return parts[1]
        return None
    
    run_times = [extract_timestamp(d) for d in run_dirs]
    run_times = [t for t in run_times if t]
    
    if run_times:
        print(f"Time range:")
        print(f"  First: {run_times[0]}")
        print(f"  Last:  {run_times[-1]}")
        print()
    
    # Load first and last for detailed stats
    print("Loading sample data...")
    
    first_traffic = Path(run_dirs[0]) / 'traffic_edges.json'
    last_traffic = Path(run_dirs[-1]) / 'traffic_edges.json'
    
    if first_traffic.exists():
        with open(first_traffic) as f:
            first_collection = json.load(f)
    else:
        first_collection = []
    
    if last_traffic.exists():
        with open(last_traffic) as f:
            last_collection = json.load(f)
    else:
        last_collection = []
    
    print()
    print("First run:")
    print(f"  Directory: {Path(run_dirs[0]).name}")
    print(f"  Records: {len(first_collection)}")
    if first_collection:
        speeds = [r['speed_kmh'] for r in first_collection if 'speed_kmh' in r]
        if speeds:
            print(f"  Speed: {min(speeds):.1f} - {max(speeds):.1f} km/h (avg: {sum(speeds)/len(speeds):.1f})")
        print(f"  Timestamp: {first_collection[0].get('timestamp', 'N/A')}")
    
    print()
    print("Last run:")
    print(f"  Directory: {Path(run_dirs[-1]).name}")
    print(f"  Records: {len(last_collection)}")
    if last_collection:
        speeds = [r['speed_kmh'] for r in last_collection if 'speed_kmh' in r]
        if speeds:
            print(f"  Speed: {min(speeds):.1f} - {max(speeds):.1f} km/h (avg: {sum(speeds)/len(speeds):.1f})")
        print(f"  Timestamp: {last_collection[0].get('timestamp', 'N/A')}")
    
    # Estimate total data points
    total_records = len(run_dirs) * len(first_collection) if first_collection else 0
    
    print()
    print("=" * 70)
    print("ESTIMATED TOTALS")
    print("=" * 70)
    print()
    print(f"Traffic data points: ~{total_records:,}")
    print(f"Weather data points: ~{len(run_dirs) * 78:,}")  # Assuming 78 nodes
    
    # File sizes
    total_size = sum(
        sum(f.stat().st_size for f in Path(d).glob('*') if f.is_file())
        for d in run_dirs
    )
    
    print()
    print(f"Storage:")
    print(f"  Total: {total_size / 1024 / 1024:.1f} MB")
    print(f"  Per run: ~{total_size / len(run_dirs) / 1024:.1f} KB")
    
    # Progress to 3-day target
    target = 54  # 3 days × 18 collections/day
    progress = len(run_dirs) / target * 100 if target > 0 else 0
    
    print()
    print("=" * 70)
    print("PROGRESS TO 3-DAY TARGET")
    print("=" * 70)
    print()
    print(f"Collections: {len(run_dirs)}/{target} ({progress:.1f}%)")
    print(f"Remaining: {max(0, target - len(run_dirs))}")
    
    if len(run_dirs) > 1 and run_times:
        # Estimate completion
        try:
            first_dt = datetime.strptime(run_times[0], '%Y%m%d_%H%M%S')
            last_dt = datetime.strptime(run_times[-1], '%Y%m%d_%H%M%S')
            elapsed = (last_dt - first_dt).total_seconds() / 3600  # hours
            
            if elapsed > 0:
                rate = len(run_dirs) / elapsed  # runs per hour
                remaining_hours = (target - len(run_dirs)) / rate
                
                print(f"Collection rate: {rate:.2f} runs/hour")
                print(f"Estimated completion: {remaining_hours:.1f} hours")
        except:
            pass
    
    print()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='View collection statistics')
    parser.add_argument('--dir', default='data/runs',
                       help='Runs directory')
    
    args = parser.parse_args()
    
    view_collection_stats(args.dir)
