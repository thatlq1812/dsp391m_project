import pandas as pd
import numpy as np

# Load data
df = pd.read_parquet('data/processed/all_runs_combined.parquet')

print('=== DATA STATISTICS ===')
print(f'Total records: {len(df):,}')
print(f'Unique runs: {df["run_id"].nunique()}')
print(f'Date range: {df["timestamp"].min()} to {df["timestamp"].max()}')
print()

print('=== SPEED STATISTICS ===')
print(f'Mean speed: {df["speed_kmh"].mean():.2f} km/h')
print(f'Std speed: {df["speed_kmh"].std():.2f} km/h')
print(f'Min speed: {df["speed_kmh"].min():.2f} km/h')
print(f'Max speed: {df["speed_kmh"].max():.2f} km/h')
print(f'Median speed: {df["speed_kmh"].median():.2f} km/h')
print()

print('=== SPEED DISTRIBUTION (Percentiles) ===')
for p in [10, 25, 50, 75, 90, 95, 99]:
    val = df['speed_kmh'].quantile(p/100)
    print(f'{p}th percentile: {val:.2f} km/h')
print()

print('=== REALISTIC ERROR BOUNDS ===')
# If predictions have MAE=0.65 km/h, what does that mean?
print(f'If MAE = 0.65 km/h:')
print(f'  -> On average, predictions are off by 0.65 km/h')
print(f'  -> As % of mean speed ({df["speed_kmh"].mean():.1f} km/h): {(0.65/df["speed_kmh"].mean())*100:.2f}%')
print(f'  -> As % of std ({df["speed_kmh"].std():.1f} km/h): {(0.65/df["speed_kmh"].std())*100:.2f}%')
print()

# Check temporal variation - analyze speed changes between consecutive timestamps
# Group by edge (node_a -> node_b) and calculate difference within each run
df['edge_id'] = df['node_a_id'].astype(str) + '_' + df['node_b_id'].astype(str)
df_sorted = df.sort_values(['edge_id', 'run_id', 'timestamp'])

# For each edge in each run, calculate speed change from previous timestamp
speed_changes = []
for edge_id in df_sorted['edge_id'].unique():
    edge_data = df_sorted[df_sorted['edge_id'] == edge_id]
    for run_id in edge_data['run_id'].unique():
        run_data = edge_data[edge_data['run_id'] == run_id].sort_values('timestamp')
        if len(run_data) > 1:
            changes = run_data['speed_kmh'].diff().abs().dropna()
            speed_changes.extend(changes.tolist())

speed_changes = pd.Series(speed_changes)
print('=== TEMPORAL VARIABILITY (15-min changes within same edge & run) ===')
print(f'Mean 15-min speed change: {speed_changes.mean():.2f} km/h')
print(f'Median 15-min speed change: {speed_changes.median():.2f} km/h')
print(f'90th percentile change: {speed_changes.quantile(0.9):.2f} km/h')
print(f'Max 15-min change: {speed_changes.max():.2f} km/h')
print()

print('=== GRAPH STRUCTURE ===')
print(f'Unique nodes: {len(set(df["node_a_id"].unique()) | set(df["node_b_id"].unique()))}')
print(f'Unique edges: {df.groupby(["node_a_id", "node_b_id"]).ngroups}')
print()

print('=== REALITY CHECK: MAE = 0.65 km/h ===')
print('Is this realistic?')
print(f'1. Mean 15-min change is ~{speed_changes.mean():.1f} km/h')
print(f'   -> Predicting 15 min ahead with MAE=0.65 means:')
print(f'   -> Model is {speed_changes.mean()/0.65:.1f}x better than "no-change" baseline')
print()
print(f'2. Data std is {df["speed_kmh"].std():.1f} km/h')
print(f'   -> MAE=0.65 is only {(0.65/df["speed_kmh"].std())*100:.1f}% of std')
print(f'   -> This means R² ≈ {1 - (0.65/df["speed_kmh"].std())**2:.3f} (unrealistically high!)')
print()
print('CONCLUSION:')
print('MAE=0.65 km/h for 15-min forecast is HIGHLY SUSPICIOUS because:')
print('- Real traffic speeds change by ~3-5 km/h every 15 minutes')
print('- Even perfect knowledge of current speed cannot predict future perfectly')
print('- External factors (traffic lights, incidents) add irreducible uncertainty')
print()
print('REALISTIC EXPECTATIONS:')
print(f'- Naive "persistence" model (predict same as current): MAE ≈ {speed_changes.mean():.1f} km/h')
print(f'- Good GNN model: MAE ≈ 2-4 km/h (40-60% improvement over baseline)')
print(f'- SOTA model on clean data: MAE ≈ 1.5-2.5 km/h')
