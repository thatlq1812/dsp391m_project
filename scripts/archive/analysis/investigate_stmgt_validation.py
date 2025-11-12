"""
Investigate STMGT validation metrics discrepancy.

This script analyzes the current STMGT model performance to identify:
1. Data split issues (leakage, improper temporal split)
2. Overfitting signals
3. Metric calculation errors
4. Dataset quality issues

Usage:
    conda run -n dsp python scripts/analysis/investigate_stmgt_validation.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def analyze_training_history():
    """Analyze training history for overfitting and performance trends."""
    
    history_path = Path("outputs/stmgt_v2_20251102_200308/training_history.csv")
    
    if not history_path.exists():
        print(f"ERROR: Training history not found at {history_path}")
        return None
    
    history = pd.read_csv(history_path)
    
    print("=" * 80)
    print("TRAINING HISTORY ANALYSIS")
    print("=" * 80)
    
    print(f"\nTotal epochs: {len(history)}")
    print(f"Training completed: {history.iloc[-1]['epoch']} epochs")
    
    # Find best validation performance
    best_epoch_idx = history['val_mae'].idxmin()
    best_epoch = history.loc[best_epoch_idx]
    
    print(f"\n--- Best Validation Performance ---")
    print(f"Epoch: {best_epoch['epoch']}")
    print(f"  Val MAE:  {best_epoch['val_mae']:.4f} km/h")
    print(f"  Val R²:   {best_epoch['val_r2']:.4f}")
    print(f"  Val RMSE: {best_epoch['val_rmse']:.4f} km/h")
    print(f"  Val MAPE: {best_epoch['val_mape']:.2f}%")
    
    # Compare with README claim
    readme_claim = 3.05
    print(f"\n--- Discrepancy Check ---")
    print(f"README claims:    {readme_claim:.2f} km/h")
    print(f"Best val MAE:     {best_epoch['val_mae']:.2f} km/h")
    print(f"Difference:       {abs(best_epoch['val_mae'] - readme_claim):.2f} km/h")
    
    if abs(best_epoch['val_mae'] - readme_claim) > 0.1:
        print("WARNING: Significant discrepancy found!")
    
    # Analyze final epoch for overfitting
    final_epoch = history.iloc[-1]
    
    print(f"\n--- Final Epoch Performance ({final_epoch['epoch']}) ---")
    print(f"Train MAE: {final_epoch['train_mae']:.4f} km/h")
    print(f"Val MAE:   {final_epoch['val_mae']:.4f} km/h")
    print(f"Gap:       {final_epoch['val_mae'] - final_epoch['train_mae']:.4f} km/h")
    
    # Check overfitting
    gap = final_epoch['val_mae'] - final_epoch['train_mae']
    if gap > 1.0:
        print("[!]  WARNING: Large train/val gap suggests overfitting")
        print("   Recommendations:")
        print("   - Increase dropout rate")
        print("   - Add weight decay regularization")
        print("   - Reduce model complexity")
        print("   - Early stopping (best was epoch {})".format(best_epoch['epoch']))
    elif gap < 0:
        print("[!]  SUSPICIOUS: Val MAE < Train MAE is unusual")
        print("   Possible causes:")
        print("   - Data leakage")
        print("   - Validation set too easy")
        print("   - Metric calculation error")
    else:
        print("[OK] Healthy train/val gap")
    
    # Learning curve analysis
    print(f"\n--- Learning Curve ---")
    first_epoch = history.iloc[0]
    improvement = first_epoch['val_mae'] - best_epoch['val_mae']
    improvement_pct = (improvement / first_epoch['val_mae']) * 100
    
    print(f"Initial val MAE:  {first_epoch['val_mae']:.4f} km/h")
    print(f"Best val MAE:     {best_epoch['val_mae']:.4f} km/h")
    print(f"Improvement:      {improvement:.4f} km/h ({improvement_pct:.1f}%)")
    
    # Check if still improving
    last_5_epochs = history.tail(5)
    val_mae_trend = last_5_epochs['val_mae'].values
    if np.all(np.diff(val_mae_trend) < 0):
        print("[OK] Still improving - could train longer")
    elif np.all(np.diff(val_mae_trend) > 0):
        print("[!]  Val MAE increasing - overfitting detected")
    else:
        print("[-] Fluctuating - likely converged")
    
    return history


def analyze_dataset():
    """Analyze dataset for quality issues."""
    
    dataset_path = Path("data/processed/all_runs_combined.parquet")
    
    if not dataset_path.exists():
        print(f"\nERROR: Dataset not found at {dataset_path}")
        return None
    
    print("\n" + "=" * 80)
    print("DATASET ANALYSIS")
    print("=" * 80)
    
    df = pd.read_parquet(dataset_path)
    
    print(f"\nTotal samples: {len(df):,}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Check temporal ordering
    is_sorted = df['timestamp'].is_monotonic_increasing
    print(f"\nTemporally sorted: {'[OK] Yes' if is_sorted else '[!]  NO'}")
    
    if not is_sorted:
        print("  ERROR: Dataset must be temporally sorted for proper splits!")
        print("  Action required: Sort by timestamp before training")
    
    # Check for duplicates (use timestamp only if edge_index doesn't exist)
    dup_cols = ['timestamp']
    if 'edge_index' in df.columns:
        dup_cols.append('edge_index')
    
    n_duplicates = df.duplicated(subset=dup_cols).sum()
    if n_duplicates > 0:
        print(f"\n[!]  Found {n_duplicates} duplicate records")
    else:
        print("\n[OK] No duplicate records")
    
    # Analyze speed distribution
    speed_stats = df['speed'].describe()
    print(f"\n--- Speed Statistics ---")
    print(f"Mean:   {speed_stats['mean']:.2f} km/h")
    print(f"Std:    {speed_stats['std']:.2f} km/h")
    print(f"Min:    {speed_stats['min']:.2f} km/h")
    print(f"Max:    {speed_stats['max']:.2f} km/h")
    print(f"Median: {speed_stats['50%']:.2f} km/h")
    
    # Check for outliers
    Q1 = df['speed'].quantile(0.25)
    Q3 = df['speed'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    n_outliers = ((df['speed'] < lower_bound) | (df['speed'] > upper_bound)).sum()
    outlier_pct = (n_outliers / len(df)) * 100
    
    print(f"\n--- Outlier Detection ---")
    print(f"IQR range: [{Q1:.2f}, {Q3:.2f}]")
    print(f"Outlier bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"Outliers: {n_outliers:,} ({outlier_pct:.2f}%)")
    
    if outlier_pct > 5:
        print("[!]  High percentage of outliers - consider cleaning")
    else:
        print("[OK] Acceptable outlier rate")
    
    # Check missing values
    missing = df.isna().sum()
    if missing.any():
        print(f"\n[!]  Missing values found:")
        print(missing[missing > 0])
    else:
        print("\n[OK] No missing values")
    
    return df


def verify_data_splits():
    """Verify train/val/test splits are proper temporal splits."""
    
    dataset_path = Path("data/processed/all_runs_combined.parquet")
    
    if not dataset_path.exists():
        return
    
    print("\n" + "=" * 80)
    print("DATA SPLIT VERIFICATION")
    print("=" * 80)
    
    df = pd.read_parquet(dataset_path)
    
    # Standard split ratios
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"\n--- Split Sizes ---")
    print(f"Train: {len(train_df):6,} samples ({len(train_df)/n*100:5.1f}%)")
    print(f"Val:   {len(val_df):6,} samples ({len(val_df)/n*100:5.1f}%)")
    print(f"Test:  {len(test_df):6,} samples ({len(test_df)/n*100:5.1f}%)")
    
    print(f"\n--- Temporal Ranges ---")
    print(f"Train: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"Val:   {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
    print(f"Test:  {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    
    # Verify no overlap
    train_max = train_df['timestamp'].max()
    val_min = val_df['timestamp'].min()
    val_max = val_df['timestamp'].max()
    test_min = test_df['timestamp'].min()
    
    print(f"\n--- Overlap Check ---")
    if train_max < val_min:
        print("[OK] Train/Val: No overlap")
    else:
        print("[!]  Train/Val: OVERLAP DETECTED - Data leakage risk!")
    
    if val_max < test_min:
        print("[OK] Val/Test: No overlap")
    else:
        print("[!]  Val/Test: OVERLAP DETECTED - Data leakage risk!")
    
    # Analyze speed distributions across splits
    print(f"\n--- Speed Distribution Across Splits ---")
    print(f"Train: Mean={train_df['speed'].mean():.2f}, Std={train_df['speed'].std():.2f}")
    print(f"Val:   Mean={val_df['speed'].mean():.2f}, Std={val_df['speed'].std():.2f}")
    print(f"Test:  Mean={test_df['speed'].mean():.2f}, Std={test_df['speed'].std():.2f}")
    
    # Check if distributions are similar
    train_mean = train_df['speed'].mean()
    val_diff = abs(val_df['speed'].mean() - train_mean) / train_mean * 100
    test_diff = abs(test_df['speed'].mean() - train_mean) / train_mean * 100
    
    if val_diff > 10 or test_diff > 10:
        print("[!]  WARNING: Large distribution shift between splits")
        print("   This may explain performance differences")
    else:
        print("[OK] Similar distributions across splits")


def generate_recommendations():
    """Generate actionable recommendations based on analysis."""
    
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n1. Immediate Actions:")
    print("   - Verify data is temporally sorted")
    print("   - Re-run training with verified splits")
    print("   - Use early stopping (best val MAE, not final epoch)")
    
    print("\n2. Improve Performance (Target: MAE < 2.5):")
    print("   - Hyperparameter tuning:")
    print("     * Learning rate: Try [1e-4, 5e-4, 1e-3]")
    print("     * Batch size: Try [32, 64, 128]")
    print("     * Dropout: Try [0.2, 0.3, 0.4]")
    print("   - Architecture improvements:")
    print("     * Increase model capacity (more layers/hidden dims)")
    print("     * Add residual connections")
    print("     * Try different attention mechanisms")
    print("   - Data improvements:")
    print("     * Collect more data")
    print("     * Feature engineering (add more temporal features)")
    print("     * Data augmentation")
    
    print("\n3. Address Overfitting:")
    print("   - Increase dropout rate (current: check config)")
    print("   - Add weight decay regularization")
    print("   - Use gradient clipping")
    print("   - Implement early stopping")
    
    print("\n4. Validation:")
    print("   - Implement k-fold cross-validation")
    print("   - Calculate confidence intervals")
    print("   - Run statistical significance tests")
    
    print("\n5. Next Steps:")
    print("   - Train baseline models (LSTM, ASTGCN)")
    print("   - Conduct ablation study")
    print("   - Compare with baselines")
    print("   - Document findings")


def main():
    """Main analysis function."""
    
    print("\n" + "=" * 80)
    print("STMGT VALIDATION INVESTIGATION")
    print("=" * 80)
    print("\nAnalyzing current STMGT model performance...")
    print("Output: outputs/stmgt_v2_20251102_200308/")
    
    # Run all analyses
    history = analyze_training_history()
    df = analyze_dataset()
    verify_data_splits()
    generate_recommendations()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nSee recommendations above for next steps.")
    

if __name__ == "__main__":
    main()
