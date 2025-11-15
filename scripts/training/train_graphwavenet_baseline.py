"""
Train GraphWaveNet baseline model for comparison with STMGT.

This script trains GraphWaveNet which learns adaptive adjacency matrix
and uses dilated causal convolutions for temporal modeling.

Expected performance: MAE 3.5-4.0 km/h (between LSTM and STMGT)

Usage:
    python scripts/training/train_graphwavenet_baseline.py [--epochs 100] [--batch-size 16]
"""

import argparse
import sys
import random
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from traffic_forecast.evaluation.graphwavenet_wrapper import GraphWaveNetWrapper
from traffic_forecast.evaluation.unified_evaluator import UnifiedEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train GraphWaveNet baseline for model comparison'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='data/processed/baseline_1month.parquet',
        help='Path to dataset parquet file (default: data/processed/baseline_1month.parquet)'
    )

    parser.add_argument(
        '--dataset-preset',
        type=str,
        choices=['baseline', 'augmented'],
        default=None,
        help="Quick preset to select dataset: 'baseline' -> baseline_1month.parquet, 'augmented' -> augmented_1year.parquet. Overrides --dataset if provided."
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/graphwavenet_baseline',
        help='Output directory for model and results'
    )
    
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=12,
        help='Number of past timesteps (default: 12)'
    )
    
    parser.add_argument(
        '--num-layers',
        type=int,
        default=4,
        help='Number of Graph WaveNet layers (default: 4)'
    )
    
    parser.add_argument(
        '--hidden-channels',
        type=int,
        default=32,
        help='Hidden channels per layer (default: 32)'
    )
    
    parser.add_argument(
        '--kernel-size',
        type=int,
        default=2,
        help='Temporal convolution kernel size (default: 2)'
    )
    
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.2,
        help='Dropout rate (default: 0.2)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=16,
        help='Batch size (default: 16)'
    )
    
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training set ratio (default: 0.7)'
    )
    
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.15,
        help='Validation set ratio (default: 0.15)'
    )

    parser.add_argument(
        '--max-interp-gap',
        type=int,
        default=3,
        help='Max consecutive missing timestamps to interpolate (default: 3)'
    )

    parser.add_argument(
        '--imputation-noise',
        type=float,
        default=0.3,
        help='Noise ratio applied to imputed values (default: 0.3)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for preprocessing and training (default: 42)'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    try:
        import tensorflow as tf  # type: ignore
        tf.random.set_seed(args.seed)
    except ImportError:
        pass
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("GRAPHWAVENET BASELINE TRAINING")
    print("=" * 80)
    print(f"\nDevice: {device.type.upper()}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("\nObjective: Spatial-temporal baseline with adaptive graph learning")
    print("Expected MAE: ~3.5-4.0 km/h (between LSTM and STMGT)")
    print("\nThis baseline learns adjacency matrix from data and uses")
    print("dilated causal convolutions for temporal modeling.")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_dir / f'run_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {run_dir}")
    
    # Resolve dataset from preset if provided
    if args.dataset_preset == 'baseline':
        dataset_path = Path('data/processed/baseline_1month.parquet')
    elif args.dataset_preset == 'augmented':
        dataset_path = Path('data/processed/augmented_1year.parquet')
    else:
        dataset_path = Path(args.dataset)

    # Load dataset
    print(f"\n[1/7] Loading dataset: {dataset_path}")
    
    if not dataset_path.exists():
        print(f"[!] Dataset not found: {dataset_path}")
        print("Available files in data/processed/:")
        for f in Path('data/processed').glob('*.parquet'):
            print(f"  - {f}")
        sys.exit(1)
    
    # Create evaluator (handles data splits)
    print(f"\n[2/7] Creating data splits (temporal)")
    evaluator = UnifiedEvaluator(
        dataset_path=dataset_path,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1 - args.train_ratio - args.val_ratio,
        seed=args.seed
    )
    
    # Get splits
    train_data = evaluator.splits['train']
    val_data = evaluator.splits['val']
    test_data = evaluator.splits['test']
    
    print(f"Train: {len(train_data):,} samples")
    print(f"Val:   {len(val_data):,} samples")
    print(f"Test:  {len(test_data):,} samples")

    # Diagnose timestamp gaps (minutes)
    ts_diffs = (
        train_data['timestamp']
        .sort_values()
        .diff()
        .dropna()
        .dt.total_seconds()
        / 60
    )
    if not ts_diffs.empty:
        top_gaps = ts_diffs.round().value_counts().head(5)
        median_gap = ts_diffs.median()
        interp_limit_minutes = args.max_interp_gap * median_gap
        print("\n[INFO] Most common timestamp gaps in training set (minutes):")
        for gap, count in top_gaps.items():
            print(f"  {int(gap):>3d} min : {count:>6} occurrences")
        max_gap = ts_diffs.max()
        print(f"  Max observed gap: {max_gap:.1f} minutes")
        if interp_limit_minutes > 0 and max_gap > interp_limit_minutes:
            print(
                f"  NOTE: Some gaps exceed interpolation limit ({args.max_interp_gap} steps ≈ {interp_limit_minutes:.1f} min)."
            )
    
    # Create model
    print(f"\n[3/7] Creating GraphWaveNet model")
    print(f"Architecture:")
    print(f"  Sequence length: {args.sequence_length}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  Hidden channels: {args.hidden_channels}")
    print(f"  Kernel size: {args.kernel_size}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Note: Adjacency matrix learned from data")
    print(f"  Preprocess: max gap {args.max_interp_gap} steps, noise {args.imputation_noise}")
    
    model = GraphWaveNetWrapper(
        sequence_length=args.sequence_length,
        num_layers=args.num_layers,
        hidden_channels=args.hidden_channels,
        kernel_size=args.kernel_size,
        dropout_rate=args.dropout,
        learning_rate=args.learning_rate,
        max_interp_gap=args.max_interp_gap,
        imputation_noise=args.imputation_noise,
        seed=args.seed,
    )
    
    print("GraphWaveNet wrapper initialized (model will be built after data loading)")
    
    # Train model
    print(f"\n[4/7] Training model")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("-" * 80)
    print()
    
    history = model.fit(
        train_data=train_data,
        val_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1
    )
    
    print("\n[OK] Training complete")
    
    # Extract metrics from history (normalized scale - for reference only)
    print(f"\n[5/8] Training history summary")
    print("-" * 80)
    
    train_mae_hist = history.history['mae'][-1]
    val_mae_hist = history.history['val_mae'][-1]
    
    print("\nFinal epoch metrics (normalized scale - reference only):")
    print(f"  Train MAE: {train_mae_hist:.4f}")
    print(f"  Val   MAE: {val_mae_hist:.4f}")
    print("\nNote: Metrics below are computed on original scale (km/h)")
    
    # Evaluate on temporal splits
    print(f"\n[6/8] Evaluating on temporal splits")
    split_metrics = {}
    split_metrics_dict = {}
    for split_name, split_data in evaluator.splits.items():
        preds, _ = model.predict(split_data, device=device.type)
        metrics = evaluator.calculate_metrics(
            split_data['speed'].values,
            preds
        )
        split_metrics[split_name] = metrics
        split_metrics_dict[split_name] = metrics.to_dict()
        print(
            f"  {split_name.upper():5s} - MAE {metrics.mae:.3f} km/h | RMSE {metrics.rmse:.3f} | R² {metrics.r2:.3f}"
        )

    train_eval_mae = split_metrics['train'].mae
    val_eval_mae = split_metrics['val'].mae
    test_eval_mae = split_metrics['test'].mae

    # Save model
    print(f"\n[7/8] Saving model...")
    model.save(run_dir)
    print(f"[OK] Model saved to {run_dir}")
    
    # Save results
    print(f"\n[8/8] Saving results...")
    
    results = {
        'model': 'GraphWaveNet',
        'timestamp': timestamp,
        'config': {
            'sequence_length': args.sequence_length,
            'num_layers': args.num_layers,
            'hidden_channels': args.hidden_channels,
            'kernel_size': args.kernel_size,
            'dropout': args.dropout,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'batch_size': args.batch_size
        },
        'data_splits': {
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'test_samples': len(test_data),
            'train_ratio': args.train_ratio,
            'val_ratio': args.val_ratio
        },
        'results': split_metrics_dict,
        'final_metrics': {
            'train_mae_kmh': float(train_eval_mae),
            'val_mae_kmh': float(val_eval_mae),
            'test_mae_kmh': float(test_eval_mae)
        },
        'training_history': {
            'train_loss': [float(x) for x in history.history['loss']],
            'val_loss': [float(x) for x in history.history['val_loss']],
            'train_mae': [float(x) for x in history.history['mae']],
            'val_mae': [float(x) for x in history.history['val_mae']]
        }
    }
    
    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"[OK] Results saved to {run_dir / 'results.json'}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"\nModel: GraphWaveNet Baseline")
    
    # Count parameters
    if model.model:
        total_params = sum([np.prod(p.shape) for p in model.model.model.trainable_weights])
        print(f"Parameters: {total_params:,}")
    
    print(f"\nEvaluation performance (km/h):")
    print(f"  Train MAE:   {train_eval_mae:.4f}")
    print(f"  Val MAE:     {val_eval_mae:.4f}")
    print(f"  Test MAE:    {test_eval_mae:.4f}")
    
    # Compare to benchmarks
    print(f"\nComparison to benchmarks:")
    print(f"  LSTM baseline:  3.94 km/h")
    print(f"  GraphWaveNet:   {val_eval_mae:.2f} km/h")
    print(f"  STMGT:          3.69 km/h")
    
    if val_eval_mae < 3.94:
        improvement = ((3.94 - val_eval_mae) / 3.94) * 100
        print(f"\nGraphWaveNet improves over LSTM by {improvement:.1f}%")
    
    if val_eval_mae > 3.69:
        gap = ((val_eval_mae - 3.69) / 3.69) * 100
        print(f"STMGT improves over GraphWaveNet by {gap:.1f}%")
    
    print(f"\nOutput directory: {run_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
