"""Train ASTGCN baseline model for comparison.

This script trains ASTGCN (spatial-temporal) baseline to compare with STMGT.
ASTGCN uses graph convolutions to model spatial relationships between road segments.

Expected performance: Better than LSTM (~4.2 km/h) but may be similar to STMGT
since both use spatial information.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from traffic_forecast.evaluation.astgcn_wrapper import ASTGCNWrapper
from traffic_forecast.evaluation.unified_evaluator import UnifiedEvaluator


def load_adjacency_matrix(cache_path: str = "cache/adjacency_matrix.npy") -> np.ndarray:
    """Load adjacency matrix from cache.
    
    Args:
        cache_path: Path to cached adjacency matrix
        
    Returns:
        Adjacency matrix (n_edges, n_edges)
    """
    adj_path = Path(cache_path)
    if not adj_path.exists():
        raise FileNotFoundError(
            f"Adjacency matrix not found at {cache_path}. "
            "Please run data collection first to generate graph structure."
        )
    
    adj_matrix = np.load(adj_path)
    print(f"[OK] Loaded adjacency matrix: {adj_matrix.shape}")
    
    return adj_matrix


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train ASTGCN baseline model")
    
    # Data arguments
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/all_runs_combined.parquet",
        help="Path to training data"
    )
    parser.add_argument(
        "--adjacency-path",
        type=str,
        default="cache/adjacency_matrix.npy",
        help="Path to adjacency matrix"
    )
    
    # Model arguments
    parser.add_argument(
        "--window",
        type=int,
        default=12,
        help="Input sequence length"
    )
    parser.add_argument(
        "--cheb-order",
        type=int,
        default=3,
        help="Order of Chebyshev polynomials"
    )
    parser.add_argument(
        "--spatial-filters",
        type=int,
        default=64,
        help="Number of spatial filters"
    )
    parser.add_argument(
        "--temporal-filters",
        type=int,
        default=64,
        help="Number of temporal filters"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate"
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/astgcn_baseline",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Print header
    print("=" * 80)
    print("ASTGCN BASELINE TRAINING")
    print("=" * 80)
    print()
    print("Objective: Establish spatial-temporal baseline")
    print("Expected MAE: ~3.5-4.0 km/h (similar to STMGT)")
    print()
    print("This baseline uses graph convolutions for spatial modeling.")
    print("Comparison: LSTM (temporal) -> ASTGCN (spatial-temporal) -> STMGT (hybrid)")
    print("=" * 80)
    print()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {run_dir}")
    print()
    
    # [1/7] Load adjacency matrix
    print("[1/7] Loading adjacency matrix...")
    print()
    adjacency_matrix = load_adjacency_matrix(args.adjacency_path)
    num_nodes = adjacency_matrix.shape[0]
    print()
    
    # [2/7] Load dataset and create splits
    print(f"[2/7] Loading dataset: {args.data_path}")
    print()
    evaluator = UnifiedEvaluator(
        dataset_path=args.data_path,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    splits = evaluator.splits
    print(f"Train: {len(splits['train']):,} samples")
    print(f"Val:   {len(splits['val']):,} samples")
    print(f"Test:  {len(splits['test']):,} samples")
    print()
    
    # [3/7] Create ASTGCN model
    print("[3/7] Creating ASTGCN model")
    print("Architecture:")
    print(f"  Window length: {args.window}")
    print(f"  Num nodes: {num_nodes}")
    print(f"  Chebyshev order: {args.cheb_order}")
    print(f"  Spatial filters: {args.spatial_filters}")
    print(f"  Temporal filters: {args.temporal_filters}")
    print(f"  Dropout: {args.dropout}")
    print(f"  Learning rate: {args.learning_rate}")
    print()
    
    model = ASTGCNWrapper(
        num_nodes=num_nodes,
        window=args.window,
        cheb_order=args.cheb_order,
        spatial_filters=args.spatial_filters,
        temporal_filters=args.temporal_filters,
        dropout_rate=args.dropout,
        learning_rate=args.learning_rate
    )
    
    # [4/7] Train model
    print("[4/7] Training model")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("-" * 80)
    print()
    
    try:
        history = model.fit(
            train_data=splits['train'],
            val_data=splits['val'],
            adjacency_matrix=adjacency_matrix,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=1
        )
        
        # Save training history
        if hasattr(history, 'history'):
            import pandas as pd
            history_df = pd.DataFrame(history.history)
            history_path = run_dir / "training_history.csv"
            history_df.to_csv(history_path, index=False)
            print(f"[OK] Training history saved to {history_path}")
        
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    print()
    
    # [5/7] Save model
    print("[5/7] Saving model")
    print("-" * 80)
    try:
        model.save(str(run_dir))
        print(f"[OK] ASTGCN model saved to {run_dir}")
    except Exception as e:
        print(f"[!] Failed to save model: {e}")
    print()
    
    # [6/7] Evaluate model
    print("[6/7] Evaluating model")
    print("-" * 80)
    print()
    
    results = {}
    
    for split_name, split_data in splits.items():
        print(f"Evaluating on {split_name} set...")
        
        try:
            # Get predictions
            y_pred, _ = model.predict(split_data, adjacency_matrix)
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
            print()
            
        except Exception as e:
            print(f"[!] Evaluation failed for {split_name}: {e}")
            results[split_name] = None
            print()
    
    # [7/7] Save results
    results_json = {
        'model': 'ASTGCN-Baseline',
        'timestamp': timestamp,
        'config': {
            'num_nodes': num_nodes,
            'window': args.window,
            'cheb_order': args.cheb_order,
            'spatial_filters': args.spatial_filters,
            'temporal_filters': args.temporal_filters,
            'dropout': args.dropout,
            'learning_rate': args.learning_rate,
            'epochs': args.epochs,
            'batch_size': args.batch_size
        },
        'dataset': {
            'path': args.data_path,
            'n_train': len(splits['train']),
            'n_val': len(splits['val']),
            'n_test': len(splits['test'])
        },
        'results': results
    }
    
    results_path = run_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"[OK] Results saved to {results_path}")
    print()
    
    # Print summary
    print("=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print()
    print(f"Model saved to: {run_dir}")
    print()
    print("Next steps:")
    print("1. Compare ASTGCN with LSTM and STMGT")
    print("2. Create comparison visualizations")
    print("3. Write model comparison report")
    print()
    
    # Sanity check
    if results.get('val') and results['val'].get('mae'):
        val_mae = results['val']['mae']
        if val_mae < 2.0:
            print("[WARNING] Val MAE unusually low - check for data leakage!")
        elif val_mae > 6.0:
            print("[WARNING] Val MAE unusually high - check model implementation!")
        else:
            print(f"[SUCCESS] ASTGCN baseline training complete: {run_dir}")
    
    return run_dir


if __name__ == "__main__":
    run_dir = main()
