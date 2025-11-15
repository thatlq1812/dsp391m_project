"""
Proper Model Evaluation Script

Uses the EXACT same pipeline as training to evaluate model performance.
This is the CORRECT way to evaluate STMGT models.

Usage:
    python scripts/evaluation/evaluate_stmgt_proper.py \
        --model outputs/stmgt_baseline_1month_20251115_132552/best_model.pt \
        --data data/processed/baseline_1month.parquet \
        --output outputs/evaluation_proper
"""

import argparse
import sys
from pathlib import Path
import json
import torch
import numpy as np
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from traffic_forecast.models.stmgt.evaluate import evaluate_model
from traffic_forecast.models.stmgt.model import STMGT
from traffic_forecast.data.stmgt_dataset import create_stmgt_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate STMGT model properly')
    parser.add_argument('--model', type=Path, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=Path, required=True, help='Path to processed data')
    parser.add_argument('--output', type=Path, default=Path('outputs/evaluation_proper'), 
                       help='Output directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--save-predictions', action='store_true', 
                       help='Save all predictions to file (large!)')
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("PROPER STMGT EVALUATION")
    print("=" * 80)
    print(f"\nModel: {args.model}")
    print(f"Data: {args.data}")
    print(f"Output: {args.output}")
    
    # Load checkpoint and config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Handle both directory and file paths
    model_path = Path(args.model)
    if model_path.is_dir():
        checkpoint_file = model_path / 'best_model.pt'
        config_path = model_path / 'config.json'
    else:
        checkpoint_file = model_path
        config_path = model_path.parent / 'config.json'
    
    checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    model_config = config.get('model', {})
    
    # Create dataloaders using training pipeline
    print("\n" + "=" * 80)
    print("LOADING DATA (Training Pipeline)")
    print("=" * 80)
    
    train_loader, val_loader, test_loader, num_nodes, edge_index = create_stmgt_dataloaders(
        data_path=str(args.data),
        batch_size=args.batch_size,
        num_workers=0,
        seq_len=model_config.get('seq_len', 12),
        pred_len=model_config.get('pred_len', 12),
        pin_memory=False,
        persistent_workers=False,
        prefetch_factor=None,
    )
    
    # Get normalization from dataset
    train_dataset = train_loader.dataset
    speed_mean = train_dataset.speed_mean
    speed_std = train_dataset.speed_std
    weather_mean = train_dataset.weather_mean.tolist()
    weather_std = train_dataset.weather_std.tolist()
    
    print(f"\nDataset Info:")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {edge_index.size(1)}")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Speed normalization: mean={speed_mean:.2f}, std={speed_std:.2f}")
    
    # Create model
    print("\n" + "=" * 80)
    print("LOADING MODEL")
    print("=" * 80)
    
    model = STMGT(
        num_nodes=num_nodes,
        in_dim=1,
        hidden_dim=model_config.get('hidden_dim', 96),
        num_blocks=model_config.get('num_blocks', 3),
        num_heads=model_config.get('num_heads', 4),
        dropout=model_config.get('dropout', 0.25),
        drop_edge_rate=model_config.get('drop_edge_rate', 0.15),
        mixture_components=model_config.get('mixture_components', 5),
        seq_len=model_config.get('seq_len', 12),
        pred_len=model_config.get('pred_len', 12),
        speed_mean=speed_mean,
        speed_std=speed_std,
        weather_mean=weather_mean,
        weather_std=weather_std,
    ).to(device)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    print("[OK] Model loaded successfully")
    print(f"  Device: {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Evaluate on all splits
    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)
    
    results = {}
    
    for split_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
        print(f"\nEvaluating on {split_name} set...")
        metrics = evaluate_model(model, loader, device)
        results[split_name] = metrics
        
        print(f"\n{split_name.upper()} SET RESULTS:")
        print(f"  MAE:  {metrics['mae']:.4f} km/h")
        print(f"  RMSE: {metrics['rmse']:.4f} km/h")
        print(f"  R²:   {metrics['r2']:.4f}")
        print(f"  MAPE: {metrics['mape']:.2f}%")
        print(f"  CRPS: {metrics['crps']:.4f}")
        print(f"  Coverage (80%): {metrics['coverage_80']:.2%}")
    
    # Save results
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    output_file = args.output / 'evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'model_path': str(args.model),
            'data_path': str(args.data),
            'results': results,
            'config': {
                'num_nodes': num_nodes,
                'num_edges': edge_index.size(1),
                'batch_size': args.batch_size,
                'device': str(device),
            }
        }, f, indent=2)
    
    print(f"[OK] Results saved to {output_file}")
    
    # Compare with saved test_results.json if exists
    saved_test = args.model.parent / 'test_results.json'
    if saved_test.exists():
        saved_metrics = json.loads(saved_test.read_text())
        print("\n" + "=" * 80)
        print("VERIFICATION")
        print("=" * 80)
        print(f"\nComparing with saved test_results.json:")
        print(f"  Saved MAE: {saved_metrics['mae']:.4f}")
        print(f"  Current MAE: {results['test']['mae']:.4f}")
        print(f"  Difference: {abs(results['test']['mae'] - saved_metrics['mae']):.4f}")
        
        if abs(results['test']['mae'] - saved_metrics['mae']) < 0.01:
            print("  [OK] MATCHES! Evaluation is correct.")
        else:
            print("  ⚠ DIFFERS! Check if model/data changed.")
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nTest Set Performance: MAE={results['test']['mae']:.2f} km/h, R²={results['test']['r2']:.3f}")
    print(f"\nThis is the CORRECT way to evaluate STMGT models.")
    print(f"Any demo script showing worse performance has implementation bugs.")


if __name__ == '__main__':
    main()
