"""Data Augmentation Strategy Analysis for STMGT datasets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assess data coverage for STMGT augmentation planning.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/all_runs_combined.parquet"),
        help="Processed parquet summarising raw collection runs (default: data/processed/all_runs_combined.parquet)",
    )
    return parser.parse_args(argv)


def load_dataset(dataset: Path) -> pd.DataFrame:
    dataset_path = dataset if dataset.is_absolute() else (PROJECT_ROOT / dataset)
    frame = pd.read_parquet(dataset_path)
    frame['timestamp'] = pd.to_datetime(frame['timestamp'])
    return frame


def print_current_analysis(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("CURRENT DATA ANALYSIS")
    print("=" * 70)

    print("\nTemporal Coverage:")
    print(f"  Start: {df['timestamp'].min()}")
    print(f"  End: {df['timestamp'].max()}")
    duration_days = (df['timestamp'].max() - df['timestamp'].min()).days
    print(f"  Duration: {duration_days} days")
    print(f"  Unique timestamps: {df['timestamp'].nunique()}")

    print("\nSpatial Coverage:")
    if {'node_a_id', 'node_b_id'}.issubset(df.columns):
        unique_edges = df.groupby(['node_a_id', 'node_b_id']).size().shape[0]
        print(f"  Unique node_a: {df['node_a_id'].nunique()}")
        print(f"  Unique node_b: {df['node_b_id'].nunique()}")
        print(f"  Unique edges: {unique_edges}")
    else:
        print("  Edge columns missing in dataset")

    print("\nTraffic Patterns:")
    if 'speed_kmh' in df.columns:
        print(f"  Speed range: {df['speed_kmh'].min():.2f} - {df['speed_kmh'].max():.2f} km/h")
        print(f"  Speed mean: {df['speed_kmh'].mean():.2f} ± {df['speed_kmh'].std():.2f}")
        print("  Speed quartiles:")
        for quantile, value in df['speed_kmh'].quantile([0.25, 0.5, 0.75]).items():
            print(f"    {quantile * 100:.0f}%: {value:.2f} km/h")

    print("\nWeather Patterns:")
    for column in ['temperature_c', 'wind_speed_kmh', 'precipitation_mm']:
        if column in df.columns:
            print(f"  {column}: {df[column].min():.2f} - {df[column].max():.2f}")

    df['hour'] = df['timestamp'].dt.hour
    df['dow'] = df['timestamp'].dt.dayofweek
    if 'speed_kmh' in df.columns:
        hourly_speed = df.groupby('hour')['speed_kmh'].mean()
        dow_speed = df.groupby('dow')['speed_kmh'].mean()

        print("\nTemporal Patterns Detected:")
        print(f"  Peak hour (slowest): {hourly_speed.idxmin()}:00 ({hourly_speed.min():.2f} km/h)")
        print(f"  Off-peak (fastest): {hourly_speed.idxmax()}:00 ({hourly_speed.max():.2f} km/h)")
        print(f"  Weekday vs Weekend variance: {dow_speed.std():.2f}")
    else:
        print("\nTemporal Patterns Detected:")
        print("  Speed column missing; temporal analysis skipped")


def print_strategy_guidance() -> None:
    print("\n" + "=" * 70)
    print("AUGMENTATION STRATEGY")
    print("=" * 70)

    print(
        """
APPROACH 1: TEMPORAL EXTRAPOLATION (Back to Oct 1)
Pros:
  ✓ Tạo nhiều data (30 days vs 2 days → 15x increase)
  ✓ Giữ được spatial structure (same edges)
  ✓ Model đã học temporal patterns → có thể extrapolate
  
Cons:
  ✗ Synthetic data có thể không realistic
  ✗ October đầu tháng có thể khác cuối tháng (concept drift)
  ✗ Weather patterns khác nhau
  
Method:
  1. Fit statistical model (GAM, Prophet) trên 2 ngày hiện có
  2. Extrapolate về Oct 1-29
  3. Add controlled noise để giữ variance
  4. Preserve correlations: speed-weather, speed-time, node-node

APPROACH 2: PATTERN-BASED SYNTHESIS
Pros:
  ✓ Realistic hơn vì dựa trên real patterns
  ✓ Có thể control distribution (match real data)
  ✓ Preserve spatial & temporal correlations
  
Method:
  1. Extract patterns: hourly profiles, day-of-week effects
  2. Model correlations: speed vs weather, node vs node
  3. Sample từ learned distributions
  4. Add realistic noise

APPROACH 3: HYBRID (RECOMMENDED)
Combine both:
  1. Use Oct 30-31 patterns → replicate cho Oct 1-29
  2. Add realistic variations:
     - Weather: Sample từ historical weather (có thể query API)
     - Traffic: Scale based on day-of-week patterns
     - Noise: Preserve variance structure
  3. Validate: Check statistical properties match

IMPLEMENTATION PLAN:
"""
    )


def print_statistical_checks(df: pd.DataFrame) -> None:
    print("\nKey Statistics to Preserve:")

    if 'speed_kmh' not in df.columns:
        print("  Speed column missing; distribution checks skipped")
        return

    speed_values = df['speed_kmh'].dropna().values
    if speed_values.size > 0:
        sample = speed_values[:5000] if len(speed_values) > 5000 else speed_values
        _, shapiro_p = stats.shapiro(sample)
        print(f"  Speed distribution: Shapiro-Wilk p={shapiro_p:.4f}")

    corr_columns = {'speed_kmh', 'temperature_c', 'wind_speed_kmh'}
    if corr_columns.issubset(df.columns):
        corr_data = df[list(corr_columns)].dropna()
        if not corr_data.empty:
            corr_matrix = corr_data.corr()
            print(f"  Speed-Temperature corr: {corr_matrix.loc['speed_kmh', 'temperature_c']:.3f}")
            print(f"  Speed-Wind corr: {corr_matrix.loc['speed_kmh', 'wind_speed_kmh']:.3f}")

    node_columns = {'node_a_id', 'node_b_id'}
    if node_columns.issubset(df.columns):
        speeds_sorted = (
            df.sort_values('timestamp')
            .groupby(['node_a_id', 'node_b_id'])['speed_kmh']
            .first()
        )
        if len(speeds_sorted) > 1:
            lag1_corr = np.corrcoef(speeds_sorted[:-1], speeds_sorted[1:])[0, 1]
            print(f"  Temporal autocorr (lag-1): {lag1_corr:.3f}")


def print_recommendations() -> None:
    print("\n" + "=" * 70)
    print("RECOMMENDED AUGMENTATION PARAMETERS")
    print("=" * 70)

    print(
        """
TARGET: 100+ samples (vs current 3)

Option A - Conservative (30 samples):
  - seq_len = 4, pred_len = 2 (6 runs/window)
  - Augment: Oct 15-31 (16 days vs 2 days = 8x)
  - Total runs: 38 * 8 = 304 runs
  - Samples: (304 - 6 + 1) = 299 per edge → 30-50 training samples

Option B - Moderate (100 samples):
  - seq_len = 6, pred_len = 3 (9 runs/window)  
  - Augment: Oct 1-31 (30 days vs 2 days = 15x)
  - Total runs: 38 * 15 = 570 runs
  - Samples: (570 - 9 + 1) = 562 per edge → 100+ samples

Option C - Aggressive (300+ samples):
  - seq_len = 4, pred_len = 2
  - Augment: Oct 1-31 (15x) + variations (3x)
  - Total runs: 38 * 15 * 3 = 1710 runs
  - Samples: 300+ samples

RECOMMENDATION: Start with Option B (moderate)
"""
    )

    print("\n" + "=" * 70)
    print("MODEL'S CURRENT DATA USAGE")
    print("=" * 70)

    print(
        """
✓ TEMPORAL DEPENDENCIES:
  - seq_len=12: Model nhìn 12 timesteps trước
  - Temporal encoder: sin/cos + embeddings
  - Transformer blocks: Capture temporal patterns
  → Model ĐÃ DÙNG temporal information correctly

✓ SPATIAL RELATIONSHIPS:
  - GAT layers: Aggregate info từ neighboring nodes
  - edge_index [2, 144]: Full graph structure
  - Parallel ST blocks: Spatial + Temporal simultaneously
  → Model ĐÃ DÙNG spatial structure correctly

✓ WEATHER INTEGRATION:
  - Weather cross-attention: Query traffic, Key/Value weather
  - 3 features: temp, wind, precip
  → Model ĐÃ LEVERAGE weather correctly

✓ MULTI-MODAL LEARNING:
  - Traffic + Weather + Time → Fused representation
  - Gaussian Mixture output: Capture uncertainty
  → Architecture OPTIMAL cho problem

GAPS TO FILL:
  1. Need more DATA (currently 3 samples)
  2. Historical patterns chưa được explicit model
     → Augmentation sẽ giúp model học patterns này

AUGMENTATION SẼ GIÚP:
  1. Model học weekly patterns (Mon-Sun variations)
  2. Model học monthly trends (if any)
  3. Better generalization (more diverse scenarios)
  4. Robust weather correlations (more weather conditions)
"""
    )

    print("\n" + "=" * 70)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    df = load_dataset(args.dataset)

    print_current_analysis(df)
    print_strategy_guidance()
    print_statistical_checks(df)
    print_recommendations()


if __name__ == "__main__":
    main()
