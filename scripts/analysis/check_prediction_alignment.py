"""
Verify GraphWaveNet prediction alignment to ensure ground truth matching.

This script checks:
1. Whether predictions are aligned to correct timestamps
2. If sequence creation introduces any unintended correlations
3. Compare sequence-based predictions vs direct predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from traffic_forecast.evaluation.graphwavenet_wrapper import GraphWaveNetWrapper


def main():
    print("=" * 80)
    print("GRAPHWAVENET PREDICTION ALIGNMENT CHECK")
    print("=" * 80)
    
    # Load data
    data_path = project_root / "data" / "processed" / "all_runs_gapfilled_week.parquet"
    print(f"Loading dataset from {data_path}...")
    df = pd.read_parquet(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['timestamp', 'node_a_id', 'node_b_id']).reset_index(drop=True)
    
    print(f"[OK] Dataset validated: {len(df):,} samples")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Speed range: [{df['speed_kmh'].min():.2f}, {df['speed_kmh'].max():.2f}] km/h")
    
    # Temporal split
    unique_timestamps = sorted(df['timestamp'].unique())
    n = len(unique_timestamps)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_ts = unique_timestamps[:train_end]
    test_ts = unique_timestamps[val_end:]
    
    train_df = df[df['timestamp'].isin(train_ts)].copy()
    test_df = df[df['timestamp'].isin(test_ts)].copy()
    
    print(f"\n=== Data Splits ===")
    print(f"TRAIN: {len(train_df):,} samples ({len(train_ts)} timestamps)")
    print(f"TEST : {len(test_df):,} samples ({len(test_ts)} timestamps)")
    
    # Initialize model
    model_path = project_root / "outputs" / "graphwavenet_baseline_fixed" / "run_20251114_220637"
    wrapper = GraphWaveNetWrapper(sequence_length=12)
    wrapper.load_checkpoint(str(model_path))
    print(f"[OK] Model loaded from {model_path.relative_to(project_root)}")
    
    # Get predictions with row alignment info
    print("\n=== Getting Predictions ===")
    predictions, _ = wrapper.predict(test_df, 'cpu')
    
    # Check alignment
    print(f"\nTest data: {len(test_df)} rows")
    print(f"Predictions: {len(predictions)} values")
    print(f"Non-null predictions: {np.sum(~np.isnan(predictions)) if predictions is not None else 0}")
    
    # Add predictions to test data
    test_df['prediction'] = predictions
    merged = test_df[~np.isnan(test_df['prediction'])].copy()
    
    print(f"\n=== Alignment Verification ===")
    print(f"Merged rows: {len(merged)}")
    
    # Calculate errors
    merged['error'] = merged['prediction'] - merged['speed_kmh']
    merged['abs_error'] = np.abs(merged['error'])
    
    # Group by timestamp to check if certain timestamps have systematic errors
    ts_errors = merged.groupby('timestamp').agg({
        'error': ['mean', 'std', 'min', 'max'],
        'abs_error': 'mean',
        'prediction': 'mean',
        'speed_kmh': 'mean'
    }).reset_index()
    
    ts_errors.columns = ['timestamp', 'mean_error', 'std_error', 'min_error', 'max_error', 
                         'mae', 'mean_pred', 'mean_true']
    
    print(f"\n=== Error Statistics by Timestamp ===")
    print(f"Number of predicted timestamps: {len(ts_errors)}")
    print(f"Overall mean error: {ts_errors['mean_error'].mean():.6f} km/h")
    print(f"Overall MAE: {ts_errors['mae'].mean():.6f} km/h")
    
    # Check if first 12 timestamps are excluded (they should be)
    test_timestamps_sorted = sorted(test_df['timestamp'].unique())
    predicted_timestamps = sorted(merged['timestamp'].unique())
    
    print(f"\n=== Sequence Offset Check ===")
    print(f"Test timestamps: {len(test_timestamps_sorted)}")
    print(f"Predicted timestamps: {len(predicted_timestamps)}")
    print(f"First test timestamp: {test_timestamps_sorted[0]}")
    print(f"First predicted timestamp: {predicted_timestamps[0]}")
    print(f"Expected first prediction: {test_timestamps_sorted[12] if len(test_timestamps_sorted) > 12 else 'N/A'}")
    
    if len(test_timestamps_sorted) > 12:
        expected_first = test_timestamps_sorted[12]
        actual_first = predicted_timestamps[0]
        if expected_first == actual_first:
            print("[OK] Predictions start at correct timestamp (after 12-step sequence)")
        else:
            print(f"[WARNING] Prediction alignment mismatch!")
            print(f"  Expected: {expected_first}")
            print(f"  Actual: {actual_first}")
    
    # Check if predictions are just lagged copies
    print(f"\n=== Lag Correlation Check ===")
    # For each timestamp, check correlation with previous timestamps
    if len(test_timestamps_sorted) > 13:
        sample_ts = predicted_timestamps[0]  # First predicted timestamp
        sample_data = merged[merged['timestamp'] == sample_ts]['prediction'].values
        
        # Get ground truth from 12 steps before
        lag_ts = test_timestamps_sorted[0]  # 12 steps before first prediction
        lag_data = test_df[test_df['timestamp'] == lag_ts]['speed_kmh'].values
        
        # They should have same number of edges
        if len(sample_data) == len(lag_data):
            corr = np.corrcoef(sample_data, lag_data)[0, 1]
            print(f"Correlation between prediction[t] and ground_truth[t-12]: {corr:.6f}")
            if corr > 0.99:
                print("[WARNING] Very high correlation with lagged input!")
        
        # Check correlation with same timestamp
        same_ts_data = test_df[test_df['timestamp'] == sample_ts]['speed_kmh'].values
        if len(sample_data) == len(same_ts_data):
            corr_same = np.corrcoef(sample_data, same_ts_data)[0, 1]
            print(f"Correlation between prediction[t] and ground_truth[t]: {corr_same:.6f}")
    
    # Sample some predictions
    print(f"\n=== Sample Predictions ===")
    sample = merged.sample(min(10, len(merged)), random_state=42)
    for _, row in sample.iterrows():
        print(f"Timestamp: {row['timestamp']}, Edge: {row['node_a_id']}->{row['node_b_id']}")
        print(f"  True: {row['speed_kmh']:.2f}, Pred: {row['prediction']:.2f}, Error: {row['error']:.2f} km/h")
    
    print("\n" + "=" * 80)
    print("ALIGNMENT CHECK COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
