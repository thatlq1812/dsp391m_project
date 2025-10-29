#!/bin/bash
# Collection Statistics Viewer

PROJECT_ID="sonorous-nomad-476606-g3"
ZONE="asia-southeast1-a"
VM_NAME="traffic-forecast-collector"

echo "======================================================================"
echo "  COLLECTION STATISTICS"
echo "======================================================================"
echo ""

gcloud compute ssh $VM_NAME \
    --zone=$ZONE \
    --project=$PROJECT_ID \
    --command="
cd ~/traffic-forecast
source ~/miniconda3/bin/activate dsp

python << 'EOF'
import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

runs_dir = Path('data/runs')
runs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], reverse=True)

print('╔════════════════════════════════════════════════════════════════╗')
print('║  COLLECTION STATISTICS SUMMARY                                 ║')
print('╚════════════════════════════════════════════════════════════════╝')
print()

# Overall stats
total_runs = len(runs)
total_edges = 0
total_successful = 0
total_failed = 0

# By hour distribution
by_hour = defaultdict(int)

print(f'Total runs: {total_runs}')
print()

# Analyze recent runs
recent_runs = runs[:100]  # Last 100 runs
print('Recent 100 runs analysis:')
print()

for run_dir in recent_runs:
    traffic_file = run_dir / 'traffic_edges.json'
    if traffic_file.exists():
        try:
            with open(traffic_file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    edge_count = len(data)
                    total_edges += edge_count
                    total_successful += sum(1 for e in data if e.get('duration_in_traffic_minutes'))
                    total_failed += sum(1 for e in data if not e.get('duration_in_traffic_minutes'))
                    
            # Extract hour from run name (run_20251030_032440)
            run_name = run_dir.name
            if 'run_' in run_name:
                time_str = run_name.split('_')[2]
                hour = int(time_str[:2])
                by_hour[hour] += 1
                
        except Exception as e:
            pass

# Print summary
if recent_runs:
    avg_edges = total_edges / len(recent_runs) if recent_runs else 0
    success_rate = (total_successful / total_edges * 100) if total_edges > 0 else 0
    
    print(f'Average edges/run: {avg_edges:.1f}')
    print(f'Total edges analyzed: {total_edges:,}')
    print(f'Successful: {total_successful:,} ({success_rate:.1f}%)')
    print(f'Failed: {total_failed:,}')
    print()

# Collections by hour
print('Collections by hour (last 100 runs):')
print()
for hour in sorted(by_hour.keys()):
    bar = '█' * (by_hour[hour] // 2)
    print(f'{hour:02d}:00 | {bar} {by_hour[hour]}')

print()

# Recent runs detail
print('Latest 10 runs:')
print()
print(f'{'Run':<25} {'Edges':<10} {'Success Rate':<15}')
print('-' * 50)

for run_dir in runs[:10]:
    traffic_file = run_dir / 'traffic_edges.json'
    if traffic_file.exists():
        try:
            with open(traffic_file) as f:
                data = json.load(f)
                if isinstance(data, list):
                    edge_count = len(data)
                    successful = sum(1 for e in data if e.get('duration_in_traffic_minutes'))
                    rate = (successful / edge_count * 100) if edge_count > 0 else 0
                    print(f'{run_dir.name:<25} {edge_count:<10} {rate:>6.1f}%')
        except:
            print(f'{run_dir.name:<25} {'ERROR':<10}')
    else:
        print(f'{run_dir.name:<25} {'NO DATA':<10}')

print()
print('╚════════════════════════════════════════════════════════════════╝')

EOF
"
