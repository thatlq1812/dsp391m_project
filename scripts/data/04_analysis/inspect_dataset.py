"""Quick inspection of generated dataset."""
import pandas as pd
import json

df = pd.read_parquet('data/processed/super_dataset_prototype.parquet')
print('Dataset shape:', df.shape)
print('\nColumns:', list(df.columns))
print('\nFirst few rows:')
print(df.head())
print('\nSpeed stats:')
print(df['speed_kmh'].describe())
print('\nEvent counts:')
print(f"Incidents: {df['is_incident'].sum()}")
print(f"Construction: {df['is_construction'].sum()}")
print(f"Holidays: {df['is_holiday'].sum()}")
print('\nWeather:')
print(df['weather_condition'].value_counts())

with open('data/processed/super_dataset_metadata.json', 'r') as f:
    meta = json.load(f)
print(f"\n\nMetadata summary:")
print(f"  Incidents: {len(meta['incidents'])}")
print(f"  Construction zones: {len(meta['construction_zones'])}")
print(f"  Weather events: {len(meta['weather_events'])}")
print(f"  Special events: {len(meta['special_events'])}")
print(f"  Holidays: {len(meta['holidays'])}")
