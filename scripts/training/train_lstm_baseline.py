"""
Train LSTM baseline model for comparison with STMGT.

This script trains a simple LSTM model (temporal only, no spatial info)
to establish a performance baseline. The goal is NOT to achieve SOTA
performance, but to show that spatial modeling (ASTGCN, STMGT) adds value.

Expected performance: MAE 4.0-5.5 km/h

Usage:
    python scripts/training/train_lstm_baseline.py [--epochs 100] [--batch-size 32]
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

from traffic_forecast.evaluation.lstm_wrapper import LSTMWrapper
from traffic_forecast.evaluation.unified_evaluator import UnifiedEvaluator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train LSTM baseline for model comparison'
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
        default='outputs/lstm_baseline',
        help='Output directory for model and results'
    )
    
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=12,
        help='Number of past timesteps (default: 12)'
    )
    
    parser.add_argument(
        '--lstm-units',
        type=int,
        nargs='+',
        default=[128, 64],
        help='LSTM layer sizes (default: 128 64)'
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
        default=32,
        help='Batch size (default: 32)'
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
    print("LSTM BASELINE TRAINING")
    print("=" * 80)
    print("\nObjective: Establish temporal-only baseline (no spatial info)")
    print("Expected MAE: 4.0-5.5 km/h (worse than STMGT)")
    print("\nThis baseline shows the value of spatial modeling in STMGT.")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_dir / f'run_{timestamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {run_dir}")
    
    # Load dataset
    print(f"\n[1/6] Loading dataset: {args.dataset}")
    dataset_path = Path(args.dataset)
    
    if not dataset_path.exists():
        print(f"[!] Dataset not found: {dataset_path}")
        print("Available files in data/processed/:")
        for f in Path('data/processed').glob('*.parquet'):
            print(f"  - {f}")
        sys.exit(1)
    
    # Create evaluator (handles data splits)
    print(f"\n[2/6] Creating data splits (temporal)")
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
    print(f"\n[3/6] Creating LSTM model")
    print(f"Architecture:")
    print(f"  Sequence length: {args.sequence_length}")
    print(f"  LSTM units: {args.lstm_units}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Learning rate: {args.learning_rate}")
    
    model = LSTMWrapper(
        sequence_length=args.sequence_length,
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout,
        learning_rate=args.learning_rate
    )
    
    # Train model
    print(f"\n[4/6] Training model")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("-" * 80)
    
    try:
        history = model.fit(
            train_data=train_data,
            val_data=val_data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=1
        )
        
        # Save training history
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(run_dir / 'training_history.csv', index=False)
        print(f"\n[OK] Training history saved to {run_dir / 'training_history.csv'}")
        
    except Exception as e:
        print(f"\n[!] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Save model
    print(f"\n[5/6] Saving model")
    model.save(run_dir)
    
    # Evaluate on all splits
    print(f"\n[6/6] Evaluating model")
    print("-" * 80)
    
    results = {}
    
    for split_name, split_data in [
        ('train', train_data),
        ('val', val_data),
        ('test', test_data)
    ]:
        print(f"\nEvaluating on {split_name} set...")
        
        try:
            # Get predictions
            y_pred, _ = model.predict(split_data)
            # Handle both column names (evaluator standardizes to 'speed')
            speed_col = 'speed' if 'speed' in split_data.columns else 'speed_kmh'
            y_true = split_data[speed_col].values
            
            # Remove NaN predictions (from sequence start)
            mask = ~np.isnan(y_pred)
            y_pred = y_pred[mask]
            y_true = y_true[mask]
            
            # Calculate metrics
            metrics = evaluator.calculate_metrics(y_true, y_pred)
            
            results[split_name] = metrics.to_dict()
            
            print(f"  MAE:  {metrics.mae:.4f} km/h")
            print(f"  RMSE: {metrics.rmse:.4f} km/h")
            print(f"  R²:   {metrics.r2:.4f}")
            print(f"  MAPE: {metrics.mape:.2f}%")
            
        except Exception as e:
            print(f"[!] Evaluation failed for {split_name}: {e}")
            results[split_name] = None
    
    # Save results
    results_json = {
        'model': 'LSTM-Baseline',
        'timestamp': timestamp,
        'config': {
            'sequence_length': args.sequence_length,
            'lstm_units': args.lstm_units,
            'dropout': args.dropout,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'batch_size': args.batch_size
        },
        'dataset': {
            'path': str(dataset_path),
            'n_train': len(train_data),
            'n_val': len(val_data),
            'n_test': len(test_data)
        },
        'results': results
    }
    
    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n[OK] Results saved to {run_dir / 'results.json'}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    
    if results['val']:
        val_mae = results['val']['mae']
        print(f"\nValidation MAE: {val_mae:.4f} km/h")
        
        # Compare with expected range
        if val_mae < 4.0:
            print("[!] WARNING: MAE better than expected (< 4.0)")
            print("    This is actually good news, but check if spatial info leaked in.")
        elif val_mae > 5.5:
            print("[!] WARNING: MAE worse than expected (> 5.5)")
            print("    Consider hyperparameter tuning or checking data quality.")
        else:
            print("[OK] MAE in expected range (4.0-5.5) for temporal-only baseline")
    
    print(f"\nModel saved to: {run_dir}")
    print("\nNext steps:")
    print("1. Compare with STMGT results")
    print("2. Train ASTGCN baseline")
    print("3. Create comparison visualizations")
    
    return run_dir


if __name__ == '__main__':
    try:
        run_dir = main()
        print(f"\n[SUCCESS] LSTM baseline training complete: {run_dir}")
    except KeyboardInterrupt:
        print("\n\n[!] Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
