"""
Compare Augmentation Methods: Leaky vs Safe

Objective: Quantify the impact of data leakage in augmentation on model performance.

Experiments:
1. Baseline: No augmentation
2. Leaky: Old augmentation (uses global stats)
3. Safe: New augmentation (uses train-only stats)

Metrics:
- Training performance
- Validation performance
- Test performance (key metric)
- Generalization gap (test - train)

Expected Results:
- Leaky may show artificially good test performance (if leakage is severe)
- Safe should show better generalization
- Baseline provides reference point

Author: THAT Le Quang (thatlq1812)
Date: 2025-11-12
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data.augment_safe import SafeTrafficAugmentor, validate_no_leakage


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare augmentation methods for data leakage assessment"
    )
    parser.add_argument(
        '--dataset',
        type=Path,
        default=Path('data/processed/all_runs_combined.parquet'),
        help='Path to original dataset'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('outputs/augmentation_comparison'),
        help='Output directory for results'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training set ratio'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio'
    )
    return parser.parse_args()


def create_temporal_splits(df, train_ratio, val_ratio):
    """Create temporal splits"""
    df = df.sort_values('timestamp')
    unique_times = sorted(df['timestamp'].unique())
    
    n_train = int(len(unique_times) * train_ratio)
    n_val = int(len(unique_times) * val_ratio)
    
    train_times = unique_times[:n_train]
    val_times = unique_times[n_train:n_train+n_val]
    test_times = unique_times[n_train+n_val:]
    
    splits = {
        'train': df[df['timestamp'].isin(train_times)],
        'val': df[df['timestamp'].isin(val_times)],
        'test': df[df['timestamp'].isin(test_times)]
    }
    
    return splits


def compute_statistics(df, split_name):
    """Compute basic statistics for a dataset split"""
    stats = {
        'split': split_name,
        'n_samples': len(df),
        'n_runs': df['run_id'].nunique(),
        'n_timestamps': df['timestamp'].nunique(),
        'date_min': str(df['timestamp'].min()),
        'date_max': str(df['timestamp'].max()),
        'speed_mean': float(df['speed_kmh'].mean()),
        'speed_std': float(df['speed_kmh'].std()),
        'speed_min': float(df['speed_kmh'].min()),
        'speed_max': float(df['speed_kmh'].max())
    }
    return stats


def analyze_augmentation_quality(original_train, augmented_train):
    """
    Analyze quality of augmented data.
    
    Checks:
    1. Distribution similarity
    2. Diversity increase
    3. Temporal coverage
    """
    print("\n=== Augmentation Quality Analysis ===")
    
    orig_speed = original_train['speed_kmh']
    aug_speed = augmented_train['speed_kmh']
    
    # Distribution comparison
    print("\nSpeed Distribution:")
    print(f"  Original: mean={orig_speed.mean():.2f}, std={orig_speed.std():.2f}")
    print(f"  Augmented: mean={aug_speed.mean():.2f}, std={aug_speed.std():.2f}")
    print(f"  Mean shift: {abs(aug_speed.mean() - orig_speed.mean()):.2f} km/h")
    print(f"  Std change: {abs(aug_speed.std() - orig_speed.std()):.2f} km/h")
    
    # Diversity metrics
    orig_unique_speeds = len(orig_speed.unique())
    aug_unique_speeds = len(aug_speed.unique())
    print(f"\nDiversity:")
    print(f"  Original unique speeds: {orig_unique_speeds}")
    print(f"  Augmented unique speeds: {aug_unique_speeds}")
    print(f"  Diversity increase: {(aug_unique_speeds / orig_unique_speeds):.2f}x")
    
    # Data volume
    print(f"\nData Volume:")
    print(f"  Original: {len(original_train)} samples")
    print(f"  Augmented: {len(augmented_train)} samples")
    print(f"  Augmentation factor: {len(augmented_train) / len(original_train):.2f}x")
    
    # Temporal coverage
    orig_runs = original_train['run_id'].nunique()
    aug_runs = augmented_train['run_id'].nunique()
    print(f"\nTemporal Coverage:")
    print(f"  Original runs: {orig_runs}")
    print(f"  Augmented runs: {aug_runs}")
    print(f"  Run expansion: {(aug_runs / orig_runs):.2f}x")


def main():
    args = parse_args()
    
    print("=" * 80)
    print("AUGMENTATION METHOD COMPARISON")
    print("=" * 80)
    print("\nObjective: Assess data leakage impact on model performance")
    print("\nExperiments:")
    print("  1. Baseline: No augmentation")
    print("  2. Safe: Augmentation using train-only statistics")
    print("  3. (Reference) Leaky: Old augmentation with global stats")
    print("=" * 80)
    
    # Create output directory
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Load dataset
    print(f"\n[1/5] Loading dataset: {args.dataset}")
    if not args.dataset.exists():
        print(f"ERROR: Dataset not found: {args.dataset}")
        return 1
    
    df = pd.read_parquet(args.dataset)
    print(f"  Loaded {len(df):,} records")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Create temporal splits
    print(f"\n[2/5] Creating temporal splits")
    splits = create_temporal_splits(df, args.train_ratio, args.val_ratio)
    
    print(f"  Train: {len(splits['train']):,} samples")
    print(f"  Val:   {len(splits['val']):,} samples")
    print(f"  Test:  {len(splits['test']):,} samples")
    
    # Compute split statistics
    split_stats = {
        'train': compute_statistics(splits['train'], 'train'),
        'val': compute_statistics(splits['val'], 'val'),
        'test': compute_statistics(splits['test'], 'test')
    }
    
    # Save baseline statistics
    baseline_path = output_dir / f'baseline_stats_{timestamp}.json'
    with open(baseline_path, 'w') as f:
        json.dump(split_stats, f, indent=2)
    print(f"\n  Saved baseline statistics: {baseline_path}")
    
    # Apply safe augmentation
    print(f"\n[3/5] Applying safe augmentation (train-only stats)")
    augmentor = SafeTrafficAugmentor(splits['train'], random_seed=42)
    
    train_augmented = augmentor.augment_all(
        noise_copies=3,
        weather_scenarios=5,
        jitter_copies=2,
        include_original=True
    )
    
    # Analyze augmentation quality
    analyze_augmentation_quality(splits['train'], train_augmented)
    
    # Validate no leakage
    print(f"\n[4/5] Validating no data leakage")
    leakage_free = validate_no_leakage(
        train_augmented,
        splits['val'],
        splits['test']
    )
    
    if not leakage_free:
        print("\nWARNING: Leakage detected in augmentation!")
        print("This should not happen with SafeTrafficAugmentor.")
        return 1
    
    # Save augmented data
    print(f"\n[5/5] Saving augmented data")
    safe_aug_path = output_dir / f'train_safe_augmented_{timestamp}.parquet'
    train_augmented.to_parquet(safe_aug_path, index=False)
    print(f"  Saved: {safe_aug_path}")
    
    # Save augmentation statistics
    aug_stats = compute_statistics(train_augmented, 'train_augmented')
    aug_stats_path = output_dir / f'augmentation_stats_{timestamp}.json'
    with open(aug_stats_path, 'w') as f:
        json.dump({
            'baseline': split_stats,
            'augmented': aug_stats,
            'augmentation_factor': len(train_augmented) / len(splits['train']),
            'leakage_free': leakage_free
        }, f, indent=2)
    print(f"  Saved: {aug_stats_path}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nBaseline (no augmentation):")
    print(f"  Train: {len(splits['train']):,} samples")
    print(f"  Val:   {len(splits['val']):,} samples")
    print(f"  Test:  {len(splits['test']):,} samples")
    
    print(f"\nSafe augmentation (train-only stats):")
    print(f"  Train: {len(train_augmented):,} samples ({len(train_augmented)/len(splits['train']):.2f}x)")
    print(f"  Val:   {len(splits['val']):,} samples (unchanged)")
    print(f"  Test:  {len(splits['test']):,} samples (unchanged)")
    
    print(f"\nLeakage validation: {'PASSED ✓' if leakage_free else 'FAILED ✗'}")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Train models on three scenarios:")
    print("   a. Baseline: Use original train split")
    print("   b. Safe: Use safe augmented train split")
    print("   c. (Optional) Leaky: Use old all_runs_augmented.parquet")
    
    print("\n2. Compare test performance:")
    print("   - If Safe > Leaky: Leakage was helping (overfitting)")
    print("   - If Safe ≈ Leaky: Leakage impact is minimal")
    print("   - If Safe < Leaky: Investigate augmentation quality")
    
    print("\n3. Check generalization gap:")
    print("   - Gap = Test MAE - Train MAE")
    print("   - Leaky should have smaller gap (suspicious)")
    print("   - Safe should have realistic gap")
    
    print("\n" + "=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
