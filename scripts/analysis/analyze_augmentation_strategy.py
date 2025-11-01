"""
Data Augmentation Strategy Analysis

Purpose:
    - Analyze current data usage
    - Design augmentation strategy
    - Implement temporal & spatial augmentation
    
Key Questions:
1. Model đang dùng temporal dependencies chưa? → CÓ (seq_len=12)
2. Model có dùng spatial relationships? → CÓ (GAT qua edges)
3. Weather correlations được leverage? → CÓ (cross-attention)
4. Có thể synthetic data không? → CÓ THỂ với constraints

Author: DSP391m Team
Date: November 1, 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Load existing data
df = pd.read_parquet('data/processed/all_runs_combined.parquet')

print("=" * 70)
print("CURRENT DATA ANALYSIS")
print("=" * 70)

# Temporal coverage
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"\nTemporal Coverage:")
print(f"  Start: {df['timestamp'].min()}")
print(f"  End: {df['timestamp'].max()}")
print(f"  Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
print(f"  Unique timestamps: {df['timestamp'].nunique()}")

# Spatial coverage
print(f"\nSpatial Coverage:")
print(f"  Unique node_a: {df['node_a_id'].nunique()}")
print(f"  Unique node_b: {df['node_b_id'].nunique()}")
print(f"  Unique edges: {df.groupby(['node_a_id', 'node_b_id']).size().shape[0]}")

# Traffic patterns
print(f"\nTraffic Patterns:")
print(f"  Speed range: {df['speed_kmh'].min():.2f} - {df['speed_kmh'].max():.2f} km/h")
print(f"  Speed mean: {df['speed_kmh'].mean():.2f} ± {df['speed_kmh'].std():.2f}")
print(f"  Speed quartiles:")
for q, val in df['speed_kmh'].quantile([0.25, 0.5, 0.75]).items():
    print(f"    {q*100:.0f}%: {val:.2f} km/h")

# Weather patterns
print(f"\nWeather Patterns:")
weather_cols = ['temperature_c', 'wind_speed_kmh', 'precipitation_mm']
for col in weather_cols:
    if col in df.columns:
        print(f"  {col}: {df[col].min():.2f} - {df[col].max():.2f}")

# Temporal patterns
df['hour'] = df['timestamp'].dt.hour
df['dow'] = df['timestamp'].dt.dayofweek
hourly_speed = df.groupby('hour')['speed_kmh'].mean()
dow_speed = df.groupby('dow')['speed_kmh'].mean()

print(f"\nTemporal Patterns Detected:")
print(f"  Peak hour (slowest): {hourly_speed.idxmin()}:00 ({hourly_speed.min():.2f} km/h)")
print(f"  Off-peak (fastest): {hourly_speed.idxmax()}:00 ({hourly_speed.max():.2f} km/h)")
print(f"  Weekday vs Weekend variance: {dow_speed.std():.2f}")

print("\n" + "=" * 70)
print("AUGMENTATION STRATEGY")
print("=" * 70)

print("""
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
""")

# Statistical analysis for augmentation
print("\nKey Statistics to Preserve:")

# 1. Speed distribution
from scipy import stats
speed_dist = df['speed_kmh'].values
shapiro_stat, shapiro_p = stats.shapiro(speed_dist[:5000] if len(speed_dist) > 5000 else speed_dist)
print(f"  Speed distribution: Shapiro-Wilk p={shapiro_p:.4f}")

# 2. Correlations
corr_data = df[['speed_kmh', 'temperature_c', 'wind_speed_kmh']].dropna()
if len(corr_data) > 0:
    corr_matrix = corr_data.corr()
    print(f"  Speed-Temperature corr: {corr_matrix.loc['speed_kmh', 'temperature_c']:.3f}")
    print(f"  Speed-Wind corr: {corr_matrix.loc['speed_kmh', 'wind_speed_kmh']:.3f}")

# 3. Temporal autocorrelation
speeds_sorted = df.sort_values('timestamp').groupby(['node_a_id', 'node_b_id'])['speed_kmh'].first()
if len(speeds_sorted) > 1:
    lag1_corr = np.corrcoef(speeds_sorted[:-1], speeds_sorted[1:])[0, 1]
    print(f"  Temporal autocorr (lag-1): {lag1_corr:.3f}")

print("\n" + "=" * 70)
print("RECOMMENDED AUGMENTATION PARAMETERS")
print("=" * 70)

print("""
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
""")

print("\n" + "=" * 70)
print("MODEL'S CURRENT DATA USAGE")
print("=" * 70)

print("""
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
""")

print("\n" + "=" * 70)
