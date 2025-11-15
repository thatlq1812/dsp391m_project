import pandas as pd
import numpy as np

df = pd.read_parquet('data/processed/baseline_1month.parquet')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print('Dataset columns:', df.columns.tolist())
print('Dataset shape:', df.shape)

# Create edge_id like in the script
if 'edge_id' not in df.columns:
    if {'node_a_id', 'node_b_id'}.issubset(df.columns):
        df['edge_id'] = df['node_a_id'].astype(str) + '->' + df['node_b_id'].astype(str)
        print('Created edge_id from node_a_id -> node_b_id')
    else:
        df['edge_id'] = df.index.astype(str)
        print('Created edge_id from index')

# Filter to Oct 30
demo_date = pd.to_datetime('2025-10-30')
df_day = df[(df['timestamp'].dt.date == demo_date.date()) & (df['timestamp'].dt.hour >= 14) & (df['timestamp'].dt.hour <= 18)]

print('\nDataset Info:')
print(f'Total rows: {len(df_day)}')
print(f'Unique edges: {df_day["edge_id"].nunique()}')
print(f'Time range: {df_day["timestamp"].min()} to {df_day["timestamp"].max()}')
print(f'Speed stats: mean={df_day["speed_kmh"].mean():.2f}, std={df_day["speed_kmh"].std():.2f}, min={df_day["speed_kmh"].min():.2f}, max={df_day["speed_kmh"].max():.2f}')

# Check edge distribution
edge_counts = df_day['edge_id'].value_counts()
print(f'\nTop 10 edges with most data:')
for i in range(min(10, len(edge_counts))):
    print(f'  {edge_counts.index[i]}: {edge_counts.iloc[i]} records')

# Check specific edge
sample_edge = edge_counts.index[0]
edge_data = df_day[df_day['edge_id'] == sample_edge].sort_values('timestamp')
print(f'\nSample edge {sample_edge} (first 20 rows):')
print(edge_data[['timestamp', 'speed_kmh']].head(20).to_string())

# Check for the edge in the figure title
target_edge = 'node-10.750048-106.633663->node-10.755373-106.635663'
if target_edge in df_day['edge_id'].values:
    edge_data = df_day[df_day['edge_id'] == target_edge].sort_values('timestamp')
    print(f'\n\nTarget edge from figure: {target_edge}')
    print(f'Records: {len(edge_data)}')
    print(edge_data[['timestamp', 'speed_kmh']].to_string())
