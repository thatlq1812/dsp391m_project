#!/usr/bin/env python3
"""Create combined dataset with ONLY original runs"""
import pandas as pd

# Load combined data
df = pd.read_parquet('data/processed/all_runs_combined.parquet')

# Filter only original runs (Oct 2025)
df_original = df[df['run_id'].str.startswith('run_202510')].copy()

print(f'Original runs only: {df_original.run_id.nunique():,} runs, {len(df_original):,} records')

# Save
output_file = 'data/processed/original_runs_only.parquet'
df_original.to_parquet(output_file, index=False)
print(f'Saved to: {output_file}')
