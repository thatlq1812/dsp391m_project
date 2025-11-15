"""
Check if GraphWaveNet predictions have systematic offset.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from traffic_forecast.evaluation.graphwavenet_wrapper import GraphWaveNetWrapper
from traffic_forecast.evaluation.unified_evaluator import UnifiedEvaluator


def main():
    print("=" * 80)
    print("GRAPHWAVENET PREDICTION OFFSET ANALYSIS")
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
    
    # Load trained model
    model_path = Path('outputs/graphwavenet_baseline_fixed/run_20251114_220637')
    model = GraphWaveNetWrapper.from_checkpoint(model_path)
    
    # Make predictions
    predictions, _ = model.predict(test_data, device='cpu')
    
    speed_col = 'speed' if 'speed' in test_data.columns else 'speed_kmh'
    y_true = test_data[speed_col].values
    
    # Filter valid predictions
    valid_mask = ~np.isnan(predictions)
    valid_preds = predictions[valid_mask]
    valid_true = y_true[valid_mask]
    
    # Calculate error statistics
    errors = valid_preds - valid_true
    
    print(f"\n[1] Error Distribution:")
    print(f"Mean error (bias): {np.mean(errors):.6f} km/h")
    print(f"Median error: {np.median(errors):.6f} km/h")
    print(f"Std dev: {np.std(errors):.6f} km/h")
    print(f"Min error: {np.min(errors):.6f} km/h")
    print(f"Max error: {np.max(errors):.6f} km/h")
    
    print(f"\n[2] Percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"  {p:2d}th: {np.percentile(errors, p):7.4f} km/h")
    
    # Check if errors are symmetric
    print(f"\n[3] Error Symmetry:")
    print(f"Positive errors (over-prediction): {(errors > 0).sum()} ({(errors > 0).mean()*100:.2f}%)")
    print(f"Negative errors (under-prediction): {(errors < 0).sum()} ({(errors < 0).mean()*100:.2f}%)")
    
    # Calculate MAE and check components
    mae = np.mean(np.abs(errors))
    print(f"\n[4] MAE Breakdown:")
    print(f"Overall MAE: {mae:.6f} km/h")
    print(f"MAE if errors were random: ~{np.std(errors):.6f} km/h")
    print(f"Actual vs expected ratio: {mae / np.std(errors):.4f}")
    
    # Check if there's a systematic offset
    print(f"\n[5] Systematic Offset Check:")
    print(f"Mean(predictions): {np.mean(valid_preds):.6f} km/h")
    print(f"Mean(ground truth): {np.mean(valid_true):.6f} km/h")
    print(f"Difference: {np.mean(valid_preds) - np.mean(valid_true):.6f} km/h")
    
    # Check correlation
    corr = np.corrcoef(valid_preds, valid_true)[0, 1]
    r2 = corr ** 2
    print(f"\n[6] Correlation Analysis:")
    print(f"Pearson correlation: {corr:.8f}")
    print(f"R²: {r2:.8f}")
    print(f"Explained variance: {r2*100:.6f}%")
    
    # Linear fit to check if there's a linear offset
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(valid_true, valid_preds)
    
    print(f"\n[7] Linear Regression (pred = slope * true + intercept):")
    print(f"Slope: {slope:.8f} (perfect = 1.0)")
    print(f"Intercept: {intercept:.8f} km/h (perfect = 0.0)")
    print(f"R: {r_value:.8f}")
    print(f"P-value: {p_value:.2e}")
    
    # Check if predictions are just shifted ground truth
    if abs(slope - 1.0) < 0.001:
        print(f"\n⚠️ WARNING: Slope ≈ 1.0, predictions may be just offset ground truth!")
        print(f"   This suggests the model is learning to simply copy input with small offset.")
    
    # Create scatter plot
    print(f"\n[8] Creating scatter plot...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot
    ax = axes[0]
    ax.scatter(valid_true, valid_preds, alpha=0.1, s=1)
    ax.plot([valid_true.min(), valid_true.max()], 
            [valid_true.min(), valid_true.max()], 
            'r--', linewidth=2, label='Perfect prediction')
    ax.plot(valid_true, slope * valid_true + intercept, 
            'g-', linewidth=2, label=f'Fitted line (slope={slope:.4f})')
    ax.set_xlabel('Ground Truth (km/h)')
    ax.set_ylabel('Predictions (km/h)')
    ax.set_title(f'GraphWaveNet Predictions vs Ground Truth\nCorr={corr:.6f}, MAE={mae:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Error distribution
    ax = axes[1]
    ax.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero error')
    ax.axvline(np.mean(errors), color='g', linestyle='--', linewidth=2, 
               label=f'Mean error={np.mean(errors):.4f}')
    ax.set_xlabel('Prediction Error (km/h)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = Path('outputs/graphwavenet_baseline_fixed/prediction_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] Plot saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    
    if abs(slope - 1.0) < 0.001 and mae < 0.5:
        print("⚠️ Model appears to be memorizing/copying input with small offset!")
        print("   Possible causes:")
        print("   1. Data leakage in preprocessing")
        print("   2. Model architecture allows information shortcut")
        print("   3. Overfitting to training patterns")
        print("   4. Dataset is too easy/predictable")
    elif corr > 0.999:
        print("✓ Model learning is excellent but suspiciously perfect")
        print("  Need to verify:")
        print("  1. No data leakage")
        print("  2. Evaluation is correct")
        print("  3. Dataset difficulty")
    else:
        print("✓ Model predictions look reasonable")
    
    print("=" * 80)


if __name__ == '__main__':
    main()
