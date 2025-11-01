#!/usr/bin/env python3
"""Compare original vs augmented runs"""
import pandas as pd
import numpy as np

# Load combined data
df = pd.read_parquet('data/processed/all_runs_combined.parquet')

# Identify original runs (from October 2025)
df['is_original'] = df['run_id'].str.startswith('run_202510')

print('=== ORIGINAL VS AUGMENTED COMPARISON ===')
print(f'\nTotal runs: {df.run_id.nunique():,}')
print(f'Original runs (Oct 2025): {df[df.is_original].run_id.nunique():,}')
print(f'Augmented runs (Sep 2025): {df[~df.is_original].run_id.nunique():,}')

print('\n=== SPEED STATISTICS ===')
print('\nOriginal runs:')
print(df[df.is_original].speed_kmh.describe())

print('\nAugmented runs:')
print(df[~df.is_original].speed_kmh.describe())

print('\n=== VARIANCE COMPARISON ===')
print(f'Original std: {df[df.is_original].speed_kmh.std():.4f} km/h')
print(f'Augmented std: {df[~df.is_original].speed_kmh.std():.4f} km/h')
print(f'Ratio: {df[~df.is_original].speed_kmh.std() / df[df.is_original].speed_kmh.std():.2f}x')
