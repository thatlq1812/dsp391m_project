"""Deep diagnosis of prediction vs actual mismatch"""
import pandas as pd
import numpy as np
import torch
import json
from pathlib import Path

print("=" * 80)
print("DEEP DIAGNOSIS: WHY MAE=7?")
print("=" * 80)

# 1. Load checkpoint and check what model actually outputs
checkpoint = torch.load('outputs/stmgt_baseline_1month_20251115_132552/best_model.pt', 
                       map_location='cpu', weights_only=False)

print("\n1. MODEL NORMALIZATION CHECK")
print("-" * 40)
if 'model_state_dict' in checkpoint:
    state = checkpoint['model_state_dict']
elif 'state_dict' in checkpoint:
    state = checkpoint['state_dict']
else:
    state = checkpoint

for key in sorted(state.keys()):
    if 'normalizer' in key:
        print(f"{key}: {state[key]}")

# 2. Check actual training metrics
print("\n2. TRAINING PERFORMANCE")
print("-" * 40)
config_path = Path('outputs/stmgt_baseline_1month_20251115_132552/config.json')
if config_path.exists():
    cfg = json.loads(config_path.read_text())
    print(f"Training MAE: {cfg.get('best_metrics', {}).get('val_mae', 'N/A')}")
    print(f"Training RMSE: {cfg.get('best_metrics', {}).get('val_rmse', 'N/A')}")

# 3. Check demo data vs training data distribution
print("\n3. DATA DISTRIBUTION COMPARISON")
print("-" * 40)
df = pd.read_parquet('data/processed/baseline_1month.parquet')
df['timestamp'] = pd.to_datetime(df['timestamp'])

demo_date = pd.to_datetime('2025-10-30')
df_demo = df[df['timestamp'].dt.date == demo_date.date()]
df_demo_afternoon = df_demo[(df_demo['timestamp'].dt.hour >= 14) & 
                            (df_demo['timestamp'].dt.hour <= 18)]

print(f"Full dataset speed: mean={df['speed_kmh'].mean():.2f}, std={df['speed_kmh'].std():.2f}")
print(f"Demo day speed: mean={df_demo['speed_kmh'].mean():.2f}, std={df_demo['speed_kmh'].std():.2f}")
print(f"Demo afternoon: mean={df_demo_afternoon['speed_kmh'].mean():.2f}, std={df_demo_afternoon['speed_kmh'].std():.2f}")

# 4. Check if predictions are normalized or denormalized
print("\n4. CHECK ACTUAL PREDICTION OUTPUT")
print("-" * 40)

# Load the model properly
import sys
sys.path.insert(0, 'D:/UNI/DSP391m/project')

try:
    from traffic_forecast.models.stmgt.model import STMGT
    
    device = torch.device('cpu')
    
    # Model config from checkpoint
    config = json.loads(config_path.read_text())
    model_config = config.get('model', {})
    
    model = STMGT(
        num_nodes=62,
        in_dim=1,
        hidden_dim=96,
        num_blocks=3,
        num_heads=4,
        dropout=0.25,
        drop_edge_rate=0.15,
        mixture_components=5,
        seq_len=12,
        pred_len=12,
        speed_mean=19.054603576660156,
        speed_std=7.832137107849121,
    )
    
    # Load state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Create sample input (similar to what demo script does)
    # Use demo afternoon data
    df_demo_afternoon['edge_id'] = df_demo_afternoon['node_a_id'].astype(str) + '->' + df_demo_afternoon['node_b_id'].astype(str)
    
    # Get a time slice for prediction
    pred_time = pd.to_datetime('2025-10-30 16:00:00')
    lookback_start = pred_time - pd.Timedelta(hours=3)
    lookback_data = df_demo_afternoon[
        (df_demo_afternoon['timestamp'] >= lookback_start) & 
        (df_demo_afternoon['timestamp'] <= pred_time)
    ].copy()
    
    print(f"Lookback data: {len(lookback_data)} records")
    print(f"Speed range: {lookback_data['speed_kmh'].min():.2f} - {lookback_data['speed_kmh'].max():.2f}")
    print(f"Speed mean: {lookback_data['speed_kmh'].mean():.2f}")
    
    # Prepare input like demo script
    unique_edges = sorted(lookback_data['edge_id'].unique())[:62]  # Limit to model nodes
    print(f"Using {len(unique_edges)} edges")
    
    # Resample to 15-min
    lookback_data['ts_15min'] = lookback_data['timestamp'].dt.floor('15T')
    agg = lookback_data.groupby(['ts_15min','edge_id'])['speed_kmh'].mean().reset_index()
    pivot = agg.pivot(index='edge_id', columns='ts_15min', values='speed_kmh')
    pivot = pivot.reindex(unique_edges)
    pivot = pivot.sort_index(axis=1)
    
    seq_len = 12
    if pivot.shape[1] < seq_len:
        needed = seq_len - pivot.shape[1]
        col_mean = pivot.mean(axis=1)
        pad_cols = [pivot.columns.min() - pd.Timedelta(minutes=15*(i+1)) for i in range(needed)][::-1]
        pad_df = pd.DataFrame({c: col_mean for c in pad_cols})
        pivot = pd.concat([pad_df, pivot], axis=1)
    else:
        pivot = pivot.iloc[:, -seq_len:]
    
    speed_matrix = pivot.fillna(pivot.mean(axis=1)).to_numpy()
    
    print(f"\nInput matrix shape: {speed_matrix.shape}")
    print(f"Input speed range: {speed_matrix.min():.2f} - {speed_matrix.max():.2f}")
    print(f"Input mean: {speed_matrix.mean():.2f}, std: {speed_matrix.std():.2f}")
    
    # Create tensor
    x_traffic = torch.from_numpy(speed_matrix).float().unsqueeze(0).unsqueeze(-1)
    
    # Dummy weather and time
    x_weather = torch.randn(1, 12, 3)
    x_time = torch.zeros(1, 12, 4)
    
    # Predict
    with torch.no_grad():
        output = model(x_traffic, x_weather, x_time)
    
    print(f"\n5. MODEL OUTPUT ANALYSIS")
    print("-" * 40)
    print(f"Output keys: {output.keys() if isinstance(output, dict) else 'tensor'}")
    
    if isinstance(output, dict):
        if 'pred_mean' in output:
            pred_mean = output['pred_mean']
            print(f"pred_mean shape: {pred_mean.shape}")
            print(f"pred_mean range: {pred_mean.min():.2f} - {pred_mean.max():.2f}")
            print(f"pred_mean mean: {pred_mean.mean():.2f}")
            
            # Check 1-hour ahead (step 4)
            step_4 = pred_mean[0, :, 3, 0]
            print(f"\nPrediction at 1-hour (step 4):")
            print(f"  Range: {step_4.min():.2f} - {step_4.max():.2f}")
            print(f"  Mean: {step_4.mean():.2f}")
            
        if 'mixture_pi' in output:
            print(f"mixture_pi shape: {output['mixture_pi'].shape}")
        if 'mixture_mu' in output:
            print(f"mixture_mu shape: {output['mixture_mu'].shape}")
        if 'mixture_sigma' in output:
            print(f"mixture_sigma shape: {output['mixture_sigma'].shape}")
    
    # 6. Check actual values at target time
    print(f"\n6. ACTUAL VALUES AT TARGET TIME")
    print("-" * 40)
    target_time = pred_time + pd.Timedelta(hours=1)
    actuals = df_demo_afternoon[
        (df_demo_afternoon['timestamp'] >= target_time - pd.Timedelta(minutes=10)) &
        (df_demo_afternoon['timestamp'] <= target_time + pd.Timedelta(minutes=10))
    ]
    
    if len(actuals) > 0:
        print(f"Actuals at {target_time}:")
        print(f"  Records: {len(actuals)}")
        print(f"  Speed range: {actuals['speed_kmh'].min():.2f} - {actuals['speed_kmh'].max():.2f}")
        print(f"  Speed mean: {actuals['speed_kmh'].mean():.2f}")
        
        # Compare with predictions
        if isinstance(output, dict) and 'pred_mean' in output:
            pred_values = step_4.cpu().numpy()
            actual_values = actuals.groupby('edge_id')['speed_kmh'].mean()
            
            # Match edges
            common_edges = set(unique_edges) & set(actual_values.index)
            print(f"\n  Common edges: {len(common_edges)}")
            
            if len(common_edges) > 0:
                preds = []
                acts = []
                for edge in common_edges:
                    idx = unique_edges.index(edge)
                    preds.append(pred_values[idx])
                    acts.append(actual_values[edge])
                
                preds = np.array(preds)
                acts = np.array(acts)
                
                mae = np.mean(np.abs(preds - acts))
                print(f"\n  Direct MAE: {mae:.2f}")
                print(f"  Prediction mean: {preds.mean():.2f}")
                print(f"  Actual mean: {acts.mean():.2f}")
                print(f"  Difference: {preds.mean() - acts.mean():.2f}")
                
                # Show some examples
                print(f"\n  Sample comparisons:")
                for i in range(min(5, len(preds))):
                    print(f"    Edge {i}: Pred={preds[i]:.2f}, Actual={acts[i]:.2f}, Error={abs(preds[i]-acts[i]):.2f}")

except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
