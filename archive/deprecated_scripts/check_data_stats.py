#!/usr/bin/env python3
"""Check data statistics"""
import pandas as pd
import numpy as np

# Load combined data
df = pd.read_parquet('data/processed/all_runs_combined.parquet')

print('=== DATA STATISTICS ===')
print(f'Total records: {len(df):,}')
print(f'Speed range: {df.speed_kmh.min():.2f} - {df.speed_kmh.max():.2f} km/h')
print(f'Speed mean: {df.speed_kmh.mean():.2f} km/h')
print(f'Speed std: {df.speed_kmh.std():.2f} km/h')
print(f'Speed median: {df.speed_kmh.median():.2f} km/h')

print('\n=== SPEED DISTRIBUTION ===')
print(df.speed_kmh.describe())

print('\n=== ZERO/LOW SPEED ANALYSIS ===')
print(f'Speed = 0: {(df.speed_kmh == 0).sum():,} ({(df.speed_kmh == 0).sum() / len(df) * 100:.2f}%)')
print(f'Speed < 5: {(df.speed_kmh < 5).sum():,} ({(df.speed_kmh < 5).sum() / len(df) * 100:.2f}%)')
print(f'Speed < 10: {(df.speed_kmh < 10).sum():,} ({(df.speed_kmh < 10).sum() / len(df) * 100:.2f}%)')

print('\n=== TIME RANGE ===')
print(f'Start: {df.timestamp.min()}')
print(f'End: {df.timestamp.max()}')
print(f'Duration: {df.timestamp.max() - df.timestamp.min()}')
