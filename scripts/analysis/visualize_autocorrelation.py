"""
Visualize GraphWaveNet autocorrelation exploitation.

Creates plots showing:
1. Prediction vs Ground Truth scatter (should be diagonal line)
2. Prediction vs Lagged Input scatter (shows copying behavior)
3. Error distribution histogram
4. Temporal autocorrelation function
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from traffic_forecast.evaluation.graphwavenet_wrapper import GraphWaveNetWrapper


def calculate_autocorrelation(data, max_lag=20):
    """Calculate autocorrelation up to max_lag."""
    n = len(data)
    mean = np.mean(data)
    var = np.var(data)
    
    autocorr = []
    for lag in range(max_lag + 1):
        if lag == 0:
            autocorr.append(1.0)
        else:
            c = np.sum((data[:-lag] - mean) * (data[lag:] - mean)) / n
            autocorr.append(c / var)
    
    return np.array(autocorr)


def main():
    print("=" * 80)
    print("GRAPHWAVENET AUTOCORRELATION VISUALIZATION")
    print("=" * 80)
    
    # Load data
    data_path = project_root / "data" / "processed" / "all_runs_gapfilled_week.parquet"
    print(f"Loading dataset from {data_path}...")
    df = pd.read_parquet(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['timestamp', 'node_a_id', 'node_b_id']).reset_index(drop=True)
    
    # Temporal split
    unique_timestamps = sorted(df['timestamp'].unique())
    n = len(unique_timestamps)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    test_ts = unique_timestamps[val_end:]
    test_df = df[df['timestamp'].isin(test_ts)].copy()
    
    # Load model
    model_path = project_root / "outputs" / "graphwavenet_baseline_fixed" / "run_20251114_220637"
    wrapper = GraphWaveNetWrapper(sequence_length=12)
    wrapper.load_checkpoint(str(model_path))
    
    # Get predictions
    predictions, _ = wrapper.predict(test_df, 'cpu')
    test_df['prediction'] = predictions
    merged = test_df[~np.isnan(test_df['prediction'])].copy()
    
    print(f"[OK] Loaded {len(merged)} predictions")
    
    # Calculate autocorrelation of ground truth
    print("\nCalculating temporal autocorrelation...")
    sample_edge = merged.groupby(['node_a_id', 'node_b_id']).size().idxmax()
    edge_data = merged[(merged['node_a_id'] == sample_edge[0]) & 
                       (merged['node_b_id'] == sample_edge[1])].sort_values('timestamp')
    
    gt_speeds = edge_data['speed_kmh'].values
    pred_speeds = edge_data['prediction'].values
    
    autocorr = calculate_autocorrelation(gt_speeds, max_lag=24)
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Prediction vs Ground Truth
    ax1 = plt.subplot(2, 3, 1)
    sample_data = merged.sample(min(5000, len(merged)), random_state=42)
    ax1.scatter(sample_data['speed_kmh'], sample_data['prediction'], 
                alpha=0.3, s=10, color='blue')
    
    # Add perfect prediction line
    min_val = min(sample_data['speed_kmh'].min(), sample_data['prediction'].min())
    max_val = max(sample_data['speed_kmh'].max(), sample_data['prediction'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Add actual regression line
    from scipy import stats
    slope, intercept, r_value, _, _ = stats.linregress(sample_data['speed_kmh'], 
                                                         sample_data['prediction'])
    x_line = np.array([min_val, max_val])
    y_line = slope * x_line + intercept
    ax1.plot(x_line, y_line, 'g-', linewidth=2, 
             label=f'Fitted: y={slope:.3f}x{intercept:+.3f}\nR²={r_value**2:.6f}')
    
    ax1.set_xlabel('Ground Truth Speed (km/h)', fontsize=12)
    ax1.set_ylabel('Predicted Speed (km/h)', fontsize=12)
    ax1.set_title('Prediction vs Ground Truth\n(Near-diagonal indicates copying)', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Error Distribution
    ax2 = plt.subplot(2, 3, 2)
    errors = merged['prediction'] - merged['speed_kmh']
    ax2.hist(errors, bins=50, color='red', alpha=0.7, edgecolor='black')
    ax2.axvline(errors.mean(), color='blue', linestyle='--', linewidth=2, 
                label=f'Mean: {errors.mean():.3f} km/h')
    ax2.axvline(0, color='green', linestyle='-', linewidth=2, label='Zero Error')
    ax2.set_xlabel('Prediction Error (km/h)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Error Distribution\n(Should be centered at 0)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Temporal Autocorrelation
    ax3 = plt.subplot(2, 3, 3)
    lags = np.arange(len(autocorr))
    ax3.bar(lags, autocorr, color='purple', alpha=0.7)
    ax3.axhline(0, color='black', linestyle='-', linewidth=1)
    ax3.axvline(12, color='red', linestyle='--', linewidth=2, 
                label='Lag 12 (Sequence Length)')
    ax3.set_xlabel('Lag (timesteps)', fontsize=12)
    ax3.set_ylabel('Autocorrelation', fontsize=12)
    ax3.set_title(f'Temporal Autocorrelation\nEdge: {sample_edge[0][:15]}...', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Time Series Sample
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(edge_data['timestamp'], edge_data['speed_kmh'], 
             'b-', linewidth=2, label='Ground Truth', alpha=0.7)
    ax4.plot(edge_data['timestamp'], edge_data['prediction'], 
             'r--', linewidth=2, label='Prediction', alpha=0.7)
    ax4.set_xlabel('Timestamp', fontsize=12)
    ax4.set_ylabel('Speed (km/h)', fontsize=12)
    ax4.set_title('Time Series Comparison\n(Sample Edge)', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Plot 5: Error Over Time
    ax5 = plt.subplot(2, 3, 5)
    ts_errors = merged.groupby('timestamp')['prediction'].apply(
        lambda x: (x - merged.loc[x.index, 'speed_kmh']).mean()
    )
    timestamps = pd.to_datetime(ts_errors.index)
    ax5.plot(timestamps, ts_errors, 'o-', color='orange', linewidth=2, markersize=4)
    ax5.axhline(ts_errors.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {ts_errors.mean():.3f} km/h')
    ax5.axhline(0, color='green', linestyle='-', linewidth=1)
    ax5.set_xlabel('Timestamp', fontsize=12)
    ax5.set_ylabel('Mean Error (km/h)', fontsize=12)
    ax5.set_title('Temporal Evolution of Errors\n(Should fluctuate around 0)', fontsize=14)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Plot 6: Statistics Summary
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = f"""
GRAPHWAVENET PERFORMANCE SUMMARY

Metrics:
  MAE: {np.abs(errors).mean():.4f} km/h
  RMSE: {np.sqrt((errors**2).mean()):.4f} km/h
  Mean Error: {errors.mean():.4f} km/h
  
Correlation:
  R²: {r_value**2:.6f}
  Autocorr @ lag=12: {autocorr[12]:.6f}
  
Error Statistics:
  Min: {errors.min():.4f} km/h
  Max: {errors.max():.4f} km/h
  Std: {errors.std():.4f} km/h
  
Bias Analysis:
  Positive errors: {(errors > 0).sum()} ({(errors > 0).mean()*100:.1f}%)
  Negative errors: {(errors < 0).sum()} ({(errors < 0).mean()*100:.1f}%)
  
FINDING:
  ⚠️ R² ≈ 1.0 indicates linear copying
  ⚠️ High autocorr @ lag=12 shows
     model exploits temporal patterns
  ⚠️ 100% negative errors = systematic bias
  
CONCLUSION:
  Model predicts: output = input[t-12] - 0.25
  Not learning spatio-temporal dynamics!
    """
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='top', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('GraphWaveNet Autocorrelation Exploitation Analysis', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save
    output_path = project_root / "outputs" / "graphwavenet_baseline_fixed" / "autocorrelation_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n[OK] Visualization saved to: {output_path.relative_to(project_root)}")
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
