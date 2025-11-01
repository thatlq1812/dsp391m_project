"""Quick training monitor - check latest outputs"""

import json
from pathlib import Path
from datetime import datetime

outputs_dir = Path('outputs')
stmgt_dirs = sorted(outputs_dir.glob('stmgt_*'), key=lambda x: x.stat().st_mtime, reverse=True)

if not stmgt_dirs:
    print("No training outputs found")
else:
    latest = stmgt_dirs[0]
    print(f"Latest training: {latest.name}")
    print(f"Started: {datetime.fromtimestamp(latest.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check config
    config_path = latest / 'config.json'
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"\nConfig:")
        print(f"  Batch size: {config.get('batch_size', 'N/A')}")
        print(f"  Workers: {config.get('num_workers', 'N/A')}")
        print(f"  AMP: {config.get('use_amp', 'N/A')}")
        print(f"  Max epochs: {config.get('max_epochs', 'N/A')}")
    
    # Check history
    history_path = latest / 'history.json'
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        
        if history.get('val_metrics'):
            print(f"\nProgress: {len(history['val_metrics'])} epochs")
            latest_metrics = history['val_metrics'][-1]
            print(f"Latest metrics:")
            print(f"  MAE: {latest_metrics.get('mae', 'N/A'):.4f} km/h")
            print(f"  R²: {latest_metrics.get('r2', 'N/A'):.4f}")
            print(f"  MAPE: {latest_metrics.get('mape', 'N/A'):.2f}%")
    else:
        print("\nTraining in progress (no history yet)...")
