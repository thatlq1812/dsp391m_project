#!/usr/bin/env python3
"""
Copy the latest run's nodes and traffic snapshot into `data/` so
`scripts/live_dashboard.py` can serve them. Prints the command to
start the dashboard.

Usage:
  python scripts/serve_latest_run.py

This is a convenience helper for local development.
"""
import os
import sys
import shutil
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_NODE = os.path.join(ROOT, 'data', 'node')
DATA_ROOT = os.path.join(ROOT, 'data')


def fail(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)


if not os.path.isdir(DATA_NODE):
    fail('No data/node directory found. Run collectors first.')

runs = [d for d in os.listdir(DATA_NODE) if os.path.isdir(os.path.join(DATA_NODE, d))]
if not runs:
    fail('No run directories found under data/node')

runs.sort()
latest = runs[-1]
print('Latest run:', latest)

collector_glob = os.path.join(DATA_NODE, latest, 'collectors', '*')
collector_dirs = sorted(glob.glob(collector_glob))
if not collector_dirs:
    fail('No collector output found in latest run')

nodes_src = None
traffic_src = None
for c in collector_dirs:
    n = os.path.join(c, 'nodes.json')
    t1 = os.path.join(c, 'traffic_snapshot_normalized.json')
    t2 = os.path.join(c, 'traffic_snapshot.json')
    t3 = os.path.join(c, 'traffic_snapshot.csv')
    if os.path.exists(n):
        nodes_src = n
        if os.path.exists(t1):
            traffic_src = t1
        elif os.path.exists(t2):
            traffic_src = t2
        elif os.path.exists(t3):
            traffic_src = t3
        break

if not nodes_src:
    fail('No nodes.json found in latest run collectors')


def backup_if_exists(dst):
    if os.path.exists(dst):
        bak = dst + '.bak'
        print(f'Backing up {dst} -> {bak}')
        shutil.copy2(dst, bak)


dst_nodes = os.path.join(DATA_ROOT, 'nodes.json')
dst_traffic = None
if traffic_src:
    dst_traffic = os.path.join(DATA_ROOT, os.path.basename(traffic_src))

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
