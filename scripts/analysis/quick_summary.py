import json
import statistics

with open('data/traffic_edges.json', 'r') as f:
    data = json.load(f)

speeds = [d['speed_kmh'] for d in data if 'speed_kmh' in d]

print('=' * 70)
print('TRAFFIC DATA SUMMARY')
print('=' * 70)
print(f'Total Edges Collected: {len(data)}')
print(f'\nSpeed Statistics (km/h):')
print(f'   Mean:   {statistics.mean(speeds):.2f}')
print(f'   Median: {statistics.median(speeds):.2f}')
print(f'   Min:    {min(speeds):.2f}')
print(f'   Max:    {max(speeds):.2f}')
print(f'   Std:    {statistics.stdev(speeds):.2f}')

# Timestamp
timestamps = [d['timestamp'] for d in data]
print(f'\nCollection Time:')
print(f'   First: {min(timestamps)}')
print(f'   Last:  {max(timestamps)}')
print('=' * 70)
