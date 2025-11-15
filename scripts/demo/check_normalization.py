"""Check model predictions vs actual data distribution"""
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path

# Load checkpoint to see embedded normalization
checkpoint = torch.load('outputs/stmgt_baseline_1month_20251115_132552/best_model.pt', 
                       map_location='cpu', weights_only=False)

print("=== MODEL NORMALIZATION ===")
if 'model_state_dict' in checkpoint:
    state = checkpoint['model_state_dict']
elif 'state_dict' in checkpoint:
    state = checkpoint['state_dict']
else:
    state = checkpoint

# Find speed normalizer parameters
for key in state.keys():
    if 'speed_normalizer' in key or 'normalizer' in key:
        print(f"{key}: {state[key]}")

if 'data_stats' in checkpoint:
    print("\nData stats from checkpoint:")
    print(checkpoint['data_stats'])

# Load actual data
df = pd.read_parquet('data/processed/baseline_1month.parquet')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print("\n=== FULL DATASET STATS ===")
print(f"Speed mean: {df['speed_kmh'].mean():.2f}")
print(f"Speed std: {df['speed_kmh'].std():.2f}")
print(f"Speed min: {df['speed_kmh'].min():.2f}")
print(f"Speed max: {df['speed_kmh'].max():.2f}")
print(f"Speed median: {df['speed_kmh'].median():.2f}")

# Oct 30 demo day
demo_date = pd.to_datetime('2025-10-30')
df_demo = df[df['timestamp'].dt.date == demo_date.date()]

print("\n=== DEMO DAY (Oct 30) STATS ===")
print(f"Speed mean: {df_demo['speed_kmh'].mean():.2f}")
print(f"Speed std: {df_demo['speed_kmh'].std():.2f}")
print(f"Speed min: {df_demo['speed_kmh'].min():.2f}")
print(f"Speed max: {df_demo['speed_kmh'].max():.2f}")

# Rush hour (16:00-18:00)
df_rush = df_demo[(df_demo['timestamp'].dt.hour >= 16) & (df_demo['timestamp'].dt.hour < 18)]
print("\n=== RUSH HOUR (16:00-18:00) STATS ===")
print(f"Speed mean: {df_rush['speed_kmh'].mean():.2f}")
print(f"Speed std: {df_rush['speed_kmh'].std():.2f}")
print(f"Records: {len(df_rush)}")
