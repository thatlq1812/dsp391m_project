"""
Verify data quality for traffic forecasting project.

This script checks the integrity and quality of processed data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    """Verify data quality."""
    print("=" * 80)
    print("DATA QUALITY VERIFICATION")
    print("=" * 80)
    
    # Load dataset
    data_path = project_root / 'data' / 'processed' / 'all_runs_gapfilled_week.parquet'
    
    if not data_path.exists():
        print(f"\n[ERROR] Dataset not found: {data_path}")
        sys.exit(1)
    
    print(f"\n[1/8] Loading dataset...")
    df = pd.read_parquet(data_path)
    print(f"✓ Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Basic info
    print(f"\n[2/8] Dataset Overview")
    print("-" * 80)
    print(f"Columns: {', '.join(df.columns.tolist())}")
    print(f"\nData types:")
    print(df.dtypes)
    
    # Check for required columns
    print(f"\n[3/8] Checking Required Columns")
    print("-" * 80)
    required_cols = ['timestamp', 'node_a_id', 'node_b_id']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"✗ Missing required columns: {missing_cols}")
        sys.exit(1)
    else:
        print(f"✓ All required columns present")
    
    # Determine speed column name
    speed_col = 'speed' if 'speed' in df.columns else 'speed_kmh'
    if speed_col not in df.columns:
        print(f"✗ No speed column found (checked 'speed' and 'speed_kmh')")
        sys.exit(1)
    
    print(f"✓ Using speed column: '{speed_col}'")
    
    # Speed statistics
    print(f"\n[4/8] Speed Statistics")
    print("-" * 80)
    print(df[speed_col].describe())
    
    print(f"\nSpeed range check:")
    min_speed = df[speed_col].min()
    max_speed = df[speed_col].max()
    print(f"  Min speed: {min_speed:.2f} km/h")
    print(f"  Max speed: {max_speed:.2f} km/h")
    
    if min_speed < 0:
        print(f"  ⚠ WARNING: Negative speeds found!")
    if max_speed > 150:
        print(f"  ⚠ WARNING: Unrealistic high speeds (>150 km/h)!")
    
    # Check for missing values
    print(f"\n[5/8] Missing Values Check")
    print("-" * 80)
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    if missing.sum() == 0:
        print("✓ No missing values")
    else:
        pct_missing = (missing.sum() / len(df)) * 100
        print(f"\n⚠ Total missing: {missing.sum():,} ({pct_missing:.2f}%)")
    
    # Timestamp analysis
    print(f"\n[6/8] Timestamp Analysis")
    print("-" * 80)
    df_sorted = df.sort_values('timestamp')
    
    print(f"Time range: {df_sorted['timestamp'].min()} to {df_sorted['timestamp'].max()}")
    print(f"Duration: {df_sorted['timestamp'].max() - df_sorted['timestamp'].min()}")
    
    # Check timestamp gaps
    print(f"\nTimestamp gaps (minutes):")
    ts_diffs = df_sorted['timestamp'].diff().dropna().dt.total_seconds() / 60
    gap_counts = ts_diffs.round().value_counts().head(10)
    
    for gap, count in gap_counts.items():
        print(f"  {int(gap):>4d} min: {count:>8,} occurrences ({count/len(ts_diffs)*100:>5.2f}%)")
    
    max_gap = ts_diffs.max()
    print(f"\nMax gap: {max_gap:.1f} minutes")
    
    if max_gap > 60:
        print(f"⚠ WARNING: Large gaps (>1 hour) detected!")
    
    # Edge statistics
    print(f"\n[7/8] Edge Statistics")
    print("-" * 80)
    
    # Create edge IDs
    df['edge_id'] = df['node_a_id'].astype(str) + '_' + df['node_b_id'].astype(str)
    n_edges = df['edge_id'].nunique()
    n_nodes = pd.concat([df['node_a_id'], df['node_b_id']]).nunique()
    
    print(f"Unique edges: {n_edges:,}")
    print(f"Unique nodes: {n_nodes:,}")
    
    # Samples per edge
    edge_counts = df['edge_id'].value_counts()
    print(f"\nSamples per edge:")
    print(f"  Mean:   {edge_counts.mean():.1f}")
    print(f"  Median: {edge_counts.median():.1f}")
    print(f"  Min:    {edge_counts.min():,}")
    print(f"  Max:    {edge_counts.max():,}")
    print(f"  Std:    {edge_counts.std():.1f}")
    
    # Check for highly unbalanced edges
    min_samples = edge_counts.min()
    max_samples = edge_counts.max()
    ratio = max_samples / min_samples if min_samples > 0 else float('inf')
    
    if ratio > 10:
        print(f"\n⚠ WARNING: Highly unbalanced edge sampling (ratio: {ratio:.1f}x)")
        print(f"  Edges with <100 samples: {(edge_counts < 100).sum()}")
        print(f"  Edges with <50 samples:  {(edge_counts < 50).sum()}")
    
    # Data distribution
    print(f"\n[8/8] Data Distribution Analysis")
    print("-" * 80)
    
    # Split by time (simulate train/val/test)
    df_sorted = df.sort_values('timestamp')
    n = len(df_sorted)
    
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_data = df_sorted.iloc[:train_end]
    val_data = df_sorted.iloc[train_end:val_end]
    test_data = df_sorted.iloc[val_end:]
    
    print(f"Split sizes (temporal):")
    print(f"  Train: {len(train_data):>8,} samples ({len(train_data)/n*100:>5.2f}%)")
    print(f"  Val:   {len(val_data):>8,} samples ({len(val_data)/n*100:>5.2f}%)")
    print(f"  Test:  {len(test_data):>8,} samples ({len(test_data)/n*100:>5.2f}%)")
    
    print(f"\nSpeed statistics by split:")
    print(f"  Train - Mean: {train_data[speed_col].mean():.2f}, Std: {train_data[speed_col].std():.2f}")
    print(f"  Val   - Mean: {val_data[speed_col].mean():.2f}, Std: {val_data[speed_col].std():.2f}")
    print(f"  Test  - Mean: {test_data[speed_col].mean():.2f}, Std: {test_data[speed_col].std():.2f}")
    
    # Check for distribution shift
    train_mean = train_data[speed_col].mean()
    val_mean = val_data[speed_col].mean()
    test_mean = test_data[speed_col].mean()
    
    val_shift = abs(val_mean - train_mean) / train_mean * 100
    test_shift = abs(test_mean - train_mean) / train_mean * 100
    
    print(f"\nDistribution shift:")
    print(f"  Val vs Train:  {val_shift:.2f}%")
    print(f"  Test vs Train: {test_shift:.2f}%")
    
    if val_shift > 10 or test_shift > 10:
        print(f"  ⚠ WARNING: Significant distribution shift detected!")
    else:
        print(f"  ✓ Distribution is consistent across splits")
    
    # Final summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    
    issues = []
    
    if min_speed < 0:
        issues.append("Negative speeds found")
    if max_speed > 150:
        issues.append("Unrealistic high speeds")
    if missing.sum() > 0:
        issues.append(f"{missing.sum():,} missing values")
    if max_gap > 60:
        issues.append(f"Large timestamp gaps ({max_gap:.1f} min)")
    if ratio > 10:
        issues.append(f"Unbalanced edge sampling (ratio: {ratio:.1f}x)")
    if val_shift > 10 or test_shift > 10:
        issues.append("Significant distribution shift")
    
    if issues:
        print(f"\n⚠ Issues found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print(f"\n✓ Data quality is GOOD")
        print(f"  - No missing values")
        print(f"  - Reasonable speed range")
        print(f"  - Consistent temporal sampling")
        print(f"  - Balanced edge representation")
        print(f"  - No distribution shift")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
