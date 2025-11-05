import pandas as pd
import glob
import os
import json

runs = []
for path in sorted(glob.glob('outputs/stmgt_v2_*/training_history.csv')):
    run_dir = os.path.dirname(path)
    run_name = os.path.basename(run_dir)
    
    # Read training history
    df = pd.read_csv(path)
    if len(df) == 0:
        continue
        
    # Find best epoch
    best_idx = df['val_mae'].idxmin()
    best = df.iloc[best_idx]
    
    # Try to read config
    config_path = os.path.join(run_dir, 'config.json')
    config_info = 'N/A'
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
            model_cfg = cfg.get('model', {})
            config_info = f"h{model_cfg.get('hidden_dim', '?')}_b{model_cfg.get('num_blocks', '?')}_mix{model_cfg.get('mixture_components', '?')}"
    
    runs.append({
        'Run': run_name[-14:],  # timestamp only
        'Config': config_info,
        'Epochs': len(df),
        'Best@': int(best['epoch']),
        'Train MAE': f"{best['train_mae']:.3f}",
        'Val MAE': f"{best['val_mae']:.3f}",
        'Val RMSE': f"{best['val_rmse']:.3f}",
    })

summary = pd.DataFrame(runs)
print("=" * 100)
print("STMGT EXPERIMENTAL PROGRESSION - Previous Training Runs")
print("=" * 100)
print(summary.to_string(index=False))
print()
print(f"Total experiments conducted: {len(runs)}")
print(f"Best validation MAE achieved: {summary['Val MAE'].min()}")
