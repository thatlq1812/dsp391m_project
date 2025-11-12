"""
Compare results from all trained models

This script reads results from all model training runs and generates
a comprehensive comparison report for the final report.

Usage:
    python scripts/training/compare_models.py --comparison-dir outputs/final_comparison/run_YYYYMMDD_HHMMSS
"""

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np


def load_model_results(model_dir: Path, model_name: str) -> dict:
    """Load results from a model's output directory"""
    
    # Try different result file locations
    result_files = [
        model_dir / 'results.json',
        model_dir / 'test_results.json',
        list(model_dir.glob('run_*/results.json')),
        list(model_dir.glob('run_*/test_results.json'))
    ]
    
    for path in result_files:
        if isinstance(path, list):
            if path:
                path = path[0]  # Take first match
            else:
                continue
                
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            
            # Extract metrics
            metrics = {}
            
            # Handle different result formats
            if 'results' in data:
                # Format 1: {results: {train: {...}, val: {...}, test: {...}}}
                for split in ['train', 'val', 'test']:
                    if split in data['results'] and data['results'][split]:
                        metrics[split] = data['results'][split]
            elif 'mae' in data:
                # Format 2: Direct metrics {mae: ..., rmse: ...}
                metrics['test'] = data
            else:
                print(f"  Warning: Unknown format in {path}")
                continue
            
            return {
                'model': model_name,
                'metrics': metrics,
                'config': data.get('config', {}),
                'source': str(path)
            }
    
    print(f"  ⚠️  No results found for {model_name} in {model_dir}")
    return None


def generate_comparison_table(results: list) -> pd.DataFrame:
    """Generate comparison table from all results"""
    
    rows = []
    
    for result in results:
        if not result:
            continue
            
        model_name = result['model']
        test_metrics = result['metrics'].get('test', {})
        val_metrics = result['metrics'].get('val', {})
        
        # Use test metrics if available, otherwise val
        metrics = test_metrics if test_metrics else val_metrics
        split = 'test' if test_metrics else 'val'
        
        if metrics:
            rows.append({
                'Model': model_name,
                'MAE (km/h)': metrics.get('mae', np.nan),
                'RMSE (km/h)': metrics.get('rmse', np.nan),
                'R²': metrics.get('r2', np.nan),
                'MAPE (%)': metrics.get('mape', np.nan),
                'CRPS': metrics.get('crps', np.nan),
                'Split': split
            })
    
    df = pd.DataFrame(rows)
    
    # Sort by MAE (best first)
    if 'MAE (km/h)' in df.columns:
        df = df.sort_values('MAE (km/h)')
    
    return df


def calculate_improvements(df: pd.DataFrame) -> dict:
    """Calculate improvements over baselines"""
    
    if df.empty or 'MAE (km/h)' not in df.columns:
        return {}
    
    improvements = {}
    
    # Find STMGT and baselines
    stmgt_row = df[df['Model'].str.contains('STMGT', case=False, na=False)]
    lstm_row = df[df['Model'].str.contains('LSTM', case=False, na=False)]
    gwnet_row = df[df['Model'].str.contains('GraphWaveNet', case=False, na=False)]
    
    if stmgt_row.empty:
        return improvements
    
    stmgt_mae = stmgt_row['MAE (km/h)'].values[0]
    
    # vs LSTM
    if not lstm_row.empty:
        lstm_mae = lstm_row['MAE (km/h)'].values[0]
        improvement = ((lstm_mae - stmgt_mae) / lstm_mae) * 100
        improvements['vs_LSTM'] = {
            'mae_reduction': lstm_mae - stmgt_mae,
            'percentage': improvement
        }
    
    # vs GraphWaveNet
    if not gwnet_row.empty:
        gwnet_mae = gwnet_row['MAE (km/h)'].values[0]
        improvement = ((gwnet_mae - stmgt_mae) / gwnet_mae) * 100
        improvements['vs_GraphWaveNet'] = {
            'mae_reduction': gwnet_mae - stmgt_mae,
            'percentage': improvement
        }
    
    return improvements


def main():
    parser = argparse.ArgumentParser(description='Compare model results')
    parser.add_argument('--comparison-dir', type=str, required=True,
                       help='Directory containing all model results')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for comparison report')
    
    args = parser.parse_args()
    
    comparison_dir = Path(args.comparison_dir)
    
    if not comparison_dir.exists():
        print(f"Error: Directory not found: {comparison_dir}")
        return
    
    print("="*80)
    print("MODEL COMPARISON REPORT GENERATION")
    print("="*80)
    print(f"\nComparison directory: {comparison_dir}\n")
    
    # Load results from all models
    model_configs = [
        ('lstm_baseline', 'LSTM Baseline'),
        ('graphwavenet', 'GraphWaveNet'),
        ('astgcn', 'ASTGCN'),
        ('stmgt_v3', 'STMGT V3')
    ]
    
    results = []
    
    for dirname, model_name in model_configs:
        model_dir = comparison_dir / dirname
        print(f"Loading {model_name} from {dirname}...")
        
        result = load_model_results(model_dir, model_name)
        if result:
            results.append(result)
            print(f"  ✓ Loaded")
        else:
            print(f"  ✗ Failed")
    
    print(f"\nSuccessfully loaded {len(results)} models\n")
    
    # Generate comparison table
    print("Generating comparison table...")
    df = generate_comparison_table(results)
    
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print()
    print(df.to_string(index=False))
    print()
    
    # Calculate improvements
    improvements = calculate_improvements(df)
    
    if improvements:
        print("="*80)
        print("IMPROVEMENTS OVER BASELINES")
        print("="*80)
        print()
        
        for baseline, metrics in improvements.items():
            print(f"{baseline}:")
            print(f"  MAE reduction: {metrics['mae_reduction']:.3f} km/h")
            print(f"  Improvement: {metrics['percentage']:.1f}%")
            print()
    
    # Save report
    report = {
        'comparison_table': df.to_dict(orient='records'),
        'improvements': improvements,
        'summary': {
            'num_models': len(results),
            'best_model': df.iloc[0]['Model'] if not df.empty else None,
            'best_mae': float(df.iloc[0]['MAE (km/h)']) if not df.empty else None
        },
        'models': results
    }
    
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("="*80)
    print(f"Report saved to: {output_path}")
    print("="*80)


if __name__ == '__main__':
    main()
