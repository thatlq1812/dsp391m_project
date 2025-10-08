#!/usr/bin/env python3
"""
Real collectors runner for production.
Runs all enabled collectors and saves outputs under a run directory.
"""

import argparse
import subprocess
import sys
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def run_cmd(cmd, cwd=None):
    print(f"$ {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        if r.stdout:
            print(r.stdout)
        if r.stderr:
            print(r.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e}; stderr:\n{e.stderr}")
        return False

def run_collectors(run_dir=None):
    if run_dir is None:
        run_dir = os.getenv('RUN_DIR', 'data_runs/latest')
    
    # Run individual collectors
    collectors = [
        ['python', 'collectors/overpass/collector.py'],
        ['python', 'collectors/open_meteo/collector.py'],
        ['python', 'collectors/google/collector.py'],
    ]
    for c in collectors:
        if not run_cmd([sys.executable if c[0] == 'python' else c[0]] + c[1:]):
            print(f"Failed to run {c}")
            # Continue to next
    
    # Generate mock traffic snapshot if no real traffic
    import json
    import os
    from datetime import datetime
    import random
    
    traffic_path = os.path.join(run_dir, 'collectors', 'mock', 'traffic_snapshot_normalized.json')
    if not os.path.exists(traffic_path):
        # Load nodes
        nodes_path = os.path.join(run_dir, 'collectors', 'overpass', 'nodes.json')
        if os.path.exists(nodes_path):
            with open(nodes_path, 'r') as f:
                nodes = json.load(f)
            traffic_data = []
            for n in nodes[:1000]:  # Limit for speed
                sp = round(random.uniform(10, 55), 1)
                traffic_data.append({'node_id': n['node_id'], 'timestamp': datetime.now().isoformat(), 'avg_speed_kmh': sp, 'vehicle_count': random.randint(10, 300)})
            os.makedirs(os.path.dirname(traffic_path), exist_ok=True)
            with open(traffic_path, 'w') as f:
                json.dump(traffic_data, f, indent=2)
            print(f"Generated mock traffic for {len(traffic_data)} nodes")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', help='Run directory')
    args = parser.parse_args()
    run_collectors(args.run_dir or os.getenv('RUN_DIR'))