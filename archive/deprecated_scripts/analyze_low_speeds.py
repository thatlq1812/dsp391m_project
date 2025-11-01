#!/usr/bin/env python3
"""Analyze low speed distribution"""
import pandas as pd

df = pd.read_parquet('data/processed/all_runs_combined.parquet')

print('=== LOW SPEED ANALYSIS ===')
for threshold in [5, 10, 15, 20]:
    count = (df.speed_kmh < threshold).sum()
    pct = count / len(df) * 100
    print(f'Speed < {threshold:2d} km/h: {count:>8,} ({pct:>5.2f}%)')

print('\n=== PERCENTILES ===')
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    val = df.speed_kmh.quantile(p/100)
    print(f'P{p:>2}: {val:>6.2f} km/h')
