#!/usr/bin/env python3
"""
Copy the latest run's nodes and traffic snapshot into `data/` so
`scripts/live_dashboard.py` can serve them. Prints the command to
start the dashboard.

Usage:
  python scripts/serve_latest_run.py

This is a convenience helper for local development.
"""
import sys
import shutil

from traffic_forecast import PROJECT_ROOT

DATA_NODE = PROJECT_ROOT / 'data' / 'node'
DATA_ROOT = PROJECT_ROOT / 'data'


def fail(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)


if not DATA_NODE.exists():
    fail('No data/node directory found. Run collectors first.')

runs = [d for d in DATA_NODE.iterdir() if d.is_dir()]
if not runs:
    fail('No run directories found under data/node')

runs.sort()
latest = runs[-1]
print('Latest run:', latest.name)

collector_dirs = sorted((latest / 'collectors').glob('*'))
if not collector_dirs:
    fail('No collector output found in latest run')

nodes_src = None
traffic_src = None
for c in collector_dirs:
    n = c / 'nodes.json'
    t1 = c / 'traffic_snapshot_normalized.json'
    t2 = c / 'traffic_snapshot.json'
    t3 = c / 'traffic_snapshot.csv'
    if n.exists():
        nodes_src = n
        if t1.exists():
            traffic_src = t1
        elif t2.exists():
            traffic_src = t2
        elif t3.exists():
            traffic_src = t3
        break

if not nodes_src:
    fail('No nodes.json found in latest run collectors')


def backup_if_exists(dst):
    if dst.exists():
        bak = dst.with_suffix(dst.suffix + '.bak')
        print(f'Backing up {dst} -> {bak}')
        shutil.copy2(dst, bak)


dst_nodes = DATA_ROOT / 'nodes.json'
dst_traffic = None
if traffic_src:
    dst_traffic = DATA_ROOT / traffic_src.name

backup_if_exists(dst_nodes)
shutil.copy2(nodes_src, dst_nodes)
print('Copied nodes ->', dst_nodes)

if traffic_src:
    backup_if_exists(dst_traffic)
    shutil.copy2(traffic_src, dst_traffic)
    print('Copied traffic snapshot ->', dst_traffic)
else:
    print('No traffic snapshot found in collector output; dashboard will use default congestion values')

print('\nTo start the dashboard:')
print('  python scripts/live_dashboard.py')
print('\nThen open http://localhost:8070 in your browser.')
