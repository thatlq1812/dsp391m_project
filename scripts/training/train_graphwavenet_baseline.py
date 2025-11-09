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
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime

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
        default='data/processed/all_runs_combined.parquet',
        help='Path to dataset parquet file'
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
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    print("=" * 80)
    print("GRAPHWAVENET BASELINE TRAINING")
    print("=" * 80)
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
    
    # Load dataset
    print(f"\n[1/7] Loading dataset: {args.dataset}")
    dataset_path = Path(args.dataset)
    
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
        seed=42
    )
    
    # Get splits
    train_data = evaluator.splits['train']
    val_data = evaluator.splits['val']
    test_data = evaluator.splits['test']
    
    print(f"Train: {len(train_data):,} samples")
    print(f"Val:   {len(val_data):,} samples")
    print(f"Test:  {len(test_data):,} samples")
    
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
    
    model = GraphWaveNetWrapper(
        sequence_length=args.sequence_length,
        num_layers=args.num_layers,
        hidden_channels=args.hidden_channels,
        kernel_size=args.kernel_size,
        dropout_rate=args.dropout,
        learning_rate=args.learning_rate
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
    
    # Extract metrics from history
    print(f"\n[5/7] Extracting metrics from training history")
    print("-" * 80)
    
    train_mae = history.history['mae'][-1]
    val_mae = history.history['val_mae'][-1]
    
    # Get scaler std for denormalization
    scaler_std = model.model.scaler_std
    
    # Denormalize MAE
    train_mae_kmh = train_mae * scaler_std
    val_mae_kmh = val_mae * scaler_std
    
    print(f"\nTrain MAE (normalized): {train_mae:.4f}")
    print(f"Train MAE (km/h): {train_mae_kmh:.4f}")
    print(f"\nVal MAE (normalized): {val_mae:.4f}")
    print(f"Val MAE (km/h): {val_mae_kmh:.4f}")
    
    # Save model
    print(f"\n[6/7] Saving model...")
    model.save(run_dir)
    print(f"[OK] Model saved to {run_dir}")
    
    # Save results
    print(f"\n[7/7] Saving results...")
    
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
        'final_metrics': {
            'train_mae_normalized': float(train_mae),
            'train_mae_kmh': float(train_mae_kmh),
            'val_mae_normalized': float(val_mae),
            'val_mae_kmh': float(val_mae_kmh)
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
    
    print(f"\nPerformance:")
    print(f"  Train MAE:   {train_mae_kmh:.4f} km/h")
    print(f"  Val MAE:     {val_mae_kmh:.4f} km/h")
    
    # Compare to benchmarks
    print(f"\nComparison to benchmarks:")
    print(f"  LSTM baseline:  3.94 km/h")
    print(f"  GraphWaveNet:   {val_mae_kmh:.2f} km/h")
    print(f"  STMGT:          3.69 km/h")
    
    if val_mae_kmh < 3.94:
        improvement = ((3.94 - val_mae_kmh) / 3.94) * 100
        print(f"\nGraphWaveNet improves over LSTM by {improvement:.1f}%")
    
    if val_mae_kmh > 3.69:
        gap = ((val_mae_kmh - 3.69) / 3.69) * 100
        print(f"STMGT improves over GraphWaveNet by {gap:.1f}%")
    
    print(f"\nOutput directory: {run_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
