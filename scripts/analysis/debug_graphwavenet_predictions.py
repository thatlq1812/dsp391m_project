"""
Debug GraphWaveNet predictions to understand why MAE is so low.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from traffic_forecast.evaluation.graphwavenet_wrapper import GraphWaveNetWrapper
from traffic_forecast.evaluation.unified_evaluator import UnifiedEvaluator


def main():
    print("=" * 80)
    print("GRAPHWAVENET PREDICTION DEBUG")
    print("=" * 80)
    
    # Load dataset
    dataset_path = Path('data/processed/all_runs_gapfilled_week.parquet')
    
    # Create evaluator
    evaluator = UnifiedEvaluator(
        dataset_path=dataset_path,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    test_data = evaluator.splits['test']
    print(f"\n[1] Test data original: {len(test_data)} rows")
    print(f"Unique timestamps: {test_data['timestamp'].nunique()}")
    print(f"Unique edges: {test_data.groupby(['node_a_id', 'node_b_id']).ngroups}")
    
    # Load trained model
    model_path = Path('outputs/graphwavenet_baseline_fixed/run_20251114_220637')
    
    if not model_path.exists():
        print(f"\n[!] Model not found: {model_path}")
        return
    
    print(f"\n[2] Loading model from: {model_path}")
    model = GraphWaveNetWrapper.from_checkpoint(model_path)
    
    # Make predictions
    print(f"\n[3] Making predictions...")
    predictions, _ = model.predict(test_data, device='cpu')
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Predictions range: [{np.nanmin(predictions):.2f}, {np.nanmax(predictions):.2f}]")
    print(f"Non-NaN predictions: {(~np.isnan(predictions)).sum()}")
    print(f"NaN predictions: {np.isnan(predictions).sum()}")
    
    # Check ground truth
    speed_col = 'speed' if 'speed' in test_data.columns else 'speed_kmh'
    y_true = test_data[speed_col].values
    
    print(f"\nGround truth shape: {y_true.shape}")
    print(f"Ground truth range: [{y_true.min():.2f}, {y_true.max():.2f}]")
    
    # Calculate metrics on non-NaN predictions
    valid_mask = ~np.isnan(predictions)
    valid_preds = predictions[valid_mask]
    valid_true = y_true[valid_mask]
    
    print(f"\n[4] Valid predictions for evaluation: {len(valid_preds)}")
    print(f"Percentage of data evaluated: {len(valid_preds)/len(test_data)*100:.2f}%")
    
    mae = np.mean(np.abs(valid_preds - valid_true))
    rmse = np.sqrt(np.mean((valid_preds - valid_true)**2))
    
    print(f"\nMetrics on valid predictions:")
    print(f"  MAE:  {mae:.4f} km/h")
    print(f"  RMSE: {rmse:.4f} km/h")
    
    # Check which timestamps are predicted
    test_data_copy = test_data.copy()
    test_data_copy['predictions'] = predictions
    test_data_copy['has_pred'] = ~np.isnan(predictions)
    
    pred_summary = test_data_copy.groupby('timestamp')['has_pred'].agg(['sum', 'count'])
    print(f"\n[5] Predictions by timestamp:")
    print(f"Timestamps with predictions: {(pred_summary['sum'] > 0).sum()}")
    print(f"Total timestamps: {len(pred_summary)}")
    
    # Check first and last predicted timestamps
    predicted_timestamps = test_data_copy[test_data_copy['has_pred']]['timestamp'].unique()
    if len(predicted_timestamps) > 0:
        print(f"\nFirst predicted timestamp: {predicted_timestamps.min()}")
        print(f"Last predicted timestamp: {predicted_timestamps.max()}")
        print(f"Number of predicted timestamps: {len(predicted_timestamps)}")
    
    # Analyze errors by timestamp
    print(f"\n[6] Error analysis by timestamp:")
    test_data_copy['error'] = np.abs(predictions - y_true)
    timestamp_errors = test_data_copy[test_data_copy['has_pred']].groupby('timestamp')['error'].agg(['mean', 'min', 'max', 'count'])
    
    print("\nTop 5 timestamps with lowest error:")
    print(timestamp_errors.nsmallest(5, 'mean'))
    
    print("\nTop 5 timestamps with highest error:")
    print(timestamp_errors.nlargest(5, 'mean'))
    
    # Check if predictions are just repeating training data
    print(f"\n[7] Checking for data leakage signs:")
    print(f"Predictions std dev: {np.std(valid_preds):.4f}")
    print(f"Ground truth std dev: {np.std(valid_true):.4f}")
    print(f"Correlation: {np.corrcoef(valid_preds, valid_true)[0, 1]:.6f}")
    
    # Check if predictions are constant
    unique_preds = np.unique(np.round(valid_preds, 2))
    print(f"\nUnique prediction values (rounded to 2 decimals): {len(unique_preds)}")
    if len(unique_preds) < 10:
        print(f"Values: {unique_preds}")
        print("⚠️ WARNING: Very few unique predictions - model may not be learning properly")
    
    # Sample some predictions
    print(f"\n[8] Sample predictions vs ground truth:")
    sample_indices = np.random.choice(len(valid_preds), min(10, len(valid_preds)), replace=False)
    for idx in sample_indices:
        pred = valid_preds[idx]
        true = valid_true[idx]
        error = abs(pred - true)
        print(f"  Pred: {pred:6.2f} | True: {true:6.2f} | Error: {error:6.2f}")
    
    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
