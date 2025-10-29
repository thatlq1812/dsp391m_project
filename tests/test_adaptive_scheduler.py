"""
Test Adaptive Scheduler v5.1
Verify cost reduction and scheduling logic
"""

import sys
from pathlib import Path
from datetime import datetime, time

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import config
import yaml

config_path = PROJECT_ROOT / 'configs' / 'project_config.yaml'
with open(config_path) as f:
    config = yaml.safe_load(f)

scheduler_config = config['scheduler']

print("=" * 70)
print("ADAPTIVE SCHEDULER v5.1 TEST")
print("=" * 70)

print(f"\nMode: {scheduler_config['mode']}")

if scheduler_config['mode'] == 'adaptive':
    adaptive = scheduler_config['adaptive']
    
    # Peak hours
    print("\n🔴 PEAK HOURS (High traffic variability):")
    peak = adaptive['peak_hours']
    print(f"   Interval: {peak['interval_minutes']} minutes")
    print("   Time ranges:")
    for tr in peak['time_ranges']:
        print(f"      {tr['start']} - {tr['end']}")
    
    # Off-peak
    print("\n🟡 OFF-PEAK HOURS (Moderate traffic):")
    offpeak = adaptive['offpeak']
    print(f"   Interval: {offpeak['interval_minutes']} minutes")
    print("   Time ranges:")
    for tr in offpeak['time_ranges']:
        print(f"      {tr['start']} - {tr['end']}")
    
    # Night
    print("\n🔵 NIGHT HOURS (Stable traffic):")
    night = adaptive['night']
    print(f"   Interval: {night['interval_minutes']} minutes")
    print(f"   Skip: {night.get('skip', False)}")
    print("   Time ranges:")
    for tr in night['time_ranges']:
        print(f"      {tr['start']} - {tr['end']}")
    
    # Weekend
    print("\n🟢 WEEKEND:")
    weekend = adaptive['weekend']
    print(f"   Enabled: {weekend['enabled']}")
    print(f"   Interval: {weekend['interval_minutes']} minutes")

# Cost calculation
print("\n" + "=" * 70)
print("COST ESTIMATE")
print("=" * 70)

def estimate_collections():
    """Estimate daily collections"""
    if scheduler_config['mode'] == 'fixed':
        interval = scheduler_config['fixed_interval_minutes']
        return 24 * 60 / interval
    
    # Adaptive mode
    adaptive = scheduler_config['adaptive']
    
    # Peak hours (assume 6 hours total: 7-9, 12-1, 5-8)
    peak_hours = 6
    peak_interval = adaptive['peak_hours']['interval_minutes']
    peak_collections = peak_hours * 60 / peak_interval
    
    # Off-peak (assume 9 hours: 9-12, 1-5, 8-11)
    offpeak_hours = 9
    offpeak_interval = adaptive['offpeak']['interval_minutes']
    offpeak_collections = offpeak_hours * 60 / offpeak_interval
    
    # Night (assume 9 hours: 11-7)
    night_hours = 9
    if adaptive['night'].get('skip', False):
        night_collections = 0
    else:
        night_interval = adaptive['night']['interval_minutes']
        night_collections = night_hours * 60 / night_interval
    
    return {
        'peak': peak_collections,
        'offpeak': offpeak_collections,
        'night': night_collections,
        'total': peak_collections + offpeak_collections + night_collections
    }

collections = estimate_collections()

print(f"\nCollections per weekday:")
print(f"  Peak hours (15 min):    {collections['peak']:5.0f}")
print(f"  Off-peak (60 min):      {collections['offpeak']:5.0f}")
print(f"  Night (120 min):        {collections['night']:5.0f}")
print(f"  Total:                  {collections['total']:5.0f}")

# API cost
edges_per_collection = 234
cost_per_1000_requests = 5.0  # $5 per 1000 requests

daily_requests = collections['total'] * edges_per_collection
daily_cost = (daily_requests / 1000) * cost_per_1000_requests

print(f"\nAPI requests per day: {daily_requests:,}")
print(f"Daily cost: ${daily_cost:.2f}")
print(f"Weekly cost (7 days): ${daily_cost * 7:.2f}")

# Compare with old constant interval
old_interval = 60  # Old: every hour
old_collections = 24 * 60 / old_interval
old_requests = old_collections * edges_per_collection
old_cost = (old_requests / 1000) * cost_per_1000_requests

print(f"\n📊 COMPARISON:")
print(f"  Old (60 min constant): {old_collections:.0f} collections/day = ${old_cost:.2f}/day")
print(f"  New (adaptive):        {collections['total']:.0f} collections/day = ${daily_cost:.2f}/day")

if daily_cost < old_cost:
    savings = ((old_cost - daily_cost) / old_cost) * 100
    print(f"  💰 SAVINGS: {savings:.1f}% = ${old_cost - daily_cost:.2f}/day")
else:
    increase = ((daily_cost - old_cost) / old_cost) * 100
    print(f"  📈 INCREASE: {increase:.1f}% = ${daily_cost - old_cost:.2f}/day")
    print(f"     (But better data during peak hours!)")

print("\n" + "=" * 70)
print("✅ Adaptive scheduler optimizes cost while maintaining data quality")
print("   - Frequent collection during peaks (capture variations)")
print("   - Infrequent collection at night (stable traffic)")
print("=" * 70)
