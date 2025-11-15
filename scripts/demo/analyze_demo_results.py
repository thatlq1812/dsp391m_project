"""Check actual demo predictions vs actuals in detail"""
import pandas as pd
import numpy as np
import json

# Load demo results
metrics = json.loads(open('outputs/demo_final/metrics.json').read())

print("="*80)
print("DEMO METRICS DETAILED ANALYSIS")
print("="*80)

print("\nOVERALL METRICS:")
print(f"MAE: {metrics['overall']['mae']:.2f} km/h")
print(f"RMSE: {metrics['overall']['rmse']:.2f} km/h")
print(f"R²: {metrics['overall']['r2']:.3f}")
print(f"Samples: {metrics['overall']['num_samples']}")

print("\nBY HORIZON:")
for horizon, data in metrics['by_horizon'].items():
    print(f"Horizon {horizon}h: MAE={data['mae']:.2f}, R²={data['r2']:.3f}, n={data['num_samples']}")

print("\nBY EDGE (sorted by MAE, worst 10):")
edges = []
for edge, data in metrics['by_edge'].items():
    edges.append((edge, data['mae'], data['num_samples']))
edges.sort(key=lambda x: x[1], reverse=True)

for i, (edge, mae, n) in enumerate(edges[:10]):
    print(f"{i+1}. MAE={mae:.2f} (n={n}): {edge[:60]}...")

print("\nBY EDGE (sorted by MAE, best 10):")
for i, (edge, mae, n) in enumerate(edges[-10:]):
    print(f"{i+1}. MAE={mae:.2f} (n={n}): {edge[:60]}...")

# Load actual data to check distribution
df = pd.read_parquet('data/processed/baseline_1month.parquet')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['edge_id'] = df['node_a_id'].astype(str) + '->' + df['node_b_id'].astype(str)

demo_date = pd.to_datetime('2025-10-30')
df_demo = df[df['timestamp'].dt.date == demo_date.date()]
df_demo_window = df_demo[(df_demo['timestamp'].dt.hour >= 14) & 
                         (df_demo['timestamp'].dt.hour <= 18)]

print("\n" + "="*80)
print("ACTUAL DATA DISTRIBUTION")
print("="*80)
print(f"\nDemo window (14:00-18:00) actual speeds:")
print(f"Mean: {df_demo_window['speed_kmh'].mean():.2f} km/h")
print(f"Std: {df_demo_window['speed_kmh'].std():.2f} km/h")
print(f"Median: {df_demo_window['speed_kmh'].median():.2f} km/h")
print(f"Min: {df_demo_window['speed_kmh'].min():.2f} km/h")
print(f"Max: {df_demo_window['speed_kmh'].max():.2f} km/h")

# Check specific edges from demo
print("\nEdges in demo metrics:")
demo_edges = list(metrics['by_edge'].keys())
print(f"Total: {len(demo_edges)}")

# Check these edges in actual data
demo_edge_data = df_demo_window[df_demo_window['edge_id'].isin(demo_edges)]
print(f"\nActual data for demo edges:")
print(f"Records: {len(demo_edge_data)}")
print(f"Mean speed: {demo_edge_data['speed_kmh'].mean():.2f} km/h")
print(f"Std speed: {demo_edge_data['speed_kmh'].std():.2f} km/h")

# Sample predictions from worst edges
print("\n" + "="*80)
print("SUSPECTED ISSUE: CHECK PREDICTION VALUES")
print("="*80)
print("\nNote: If predictions are consistently ~20 km/h but actuals are ~15 km/h,")
print("that's expected (model trained on global mean, demo shows rush hour).")
print("\nBut if predictions are wildly off (like 50+ km/h), that's a bug!")

# Estimate typical prediction value
# From test: input 20 km/h → output ~26 km/h mean
# If demo inputs are ~15-18 km/h, predictions should be ~20-25 km/h
# MAE=7 suggests predictions are ~22 km/h vs actuals ~15 km/h
estimated_pred_mean = df_demo_window['speed_kmh'].mean() + metrics['overall']['mae']
print(f"\nEstimated prediction mean: {estimated_pred_mean:.2f} km/h")
print(f"Actual mean: {df_demo_window['speed_kmh'].mean():.2f} km/h")
print(f"Difference: {estimated_pred_mean - df_demo_window['speed_kmh'].mean():.2f} km/h")

if estimated_pred_mean > 30:
    print("\n❌ PREDICTIONS TOO HIGH - Possible bug!")
elif estimated_pred_mean > 22:
    print("\n⚠️  PREDICTIONS MODERATELY HIGH - Model overestimating rush hour")
else:
    print("\n✓ PREDICTIONS REASONABLE - MAE=7 might be acceptable given conditions")
