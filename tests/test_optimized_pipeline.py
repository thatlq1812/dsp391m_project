"""Quick test of new training config and data loading speed"""
import time
import torch
from pathlib import Path
from traffic_forecast.data.stmgt_dataset import STMGTDataset
from torch.utils.data import DataLoader
from traffic_forecast.models.stmgt.model import STMGT

print("="*60)
print("Testing Optimized Training Pipeline")
print("="*60)

# 1. Test dataset loading speed
print("\n1. Dataset Creation (optimized):")
start = time.time()
train_dataset = STMGTDataset(
    data_path=Path("data/processed/all_runs_gapfilled_week.parquet"),
    seq_len=12,
    pred_len=12,
    split='train'
)
creation_time = time.time() - start
print(f"   Created in {creation_time:.2f}s")
print(f"   Samples: {len(train_dataset)}")

# 2. Test DataLoader speed
print("\n2. DataLoader Performance (batch_size=64):")
loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0,
)

times = []
for i, batch in enumerate(loader):
    start = time.time()
    _ = batch['x_traffic']
    times.append(time.time() - start)
    if i >= 4:  # Test 5 batches
        break

avg_time = sum(times) / len(times)
print(f"   Average: {avg_time:.3f}s per batch")
print(f"   Estimated epoch time: {avg_time * len(loader) / 60:.1f} minutes (data only)")

# 3. Test model with new config
print("\n3. Model Creation (hidden_dim=96, K=5):")
model = STMGT(
    num_nodes=62,
    in_dim=1,
    hidden_dim=96,  # Upgraded from 64
    num_heads=4,
    num_blocks=3,
    mixture_components=5,  # Upgraded from 3
    seq_len=12,
    pred_len=12,
    dropout=0.2,
    speed_mean=train_dataset.speed_mean,
    speed_std=train_dataset.speed_std,
    weather_mean=train_dataset.weather_mean.tolist(),
    weather_std=train_dataset.weather_std.tolist()
)

total_params = sum(p.numel() for p in model.parameters())
print(f"   Parameters: {total_params:,}")
print(f"   vs old model: 304,236 ({(total_params/304236-1)*100:+.1f}%)")

# 4. Test forward pass speed
print("\n4. Forward Pass Speed:")
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

batch = next(iter(loader))
x_traffic = batch['x_traffic'].to(device)
x_weather = batch['x_weather'].to(device)
edge_index = train_dataset.edge_index.to(device)
temporal = {
    'hour': batch['hour'].to(device),
    'dow': batch['dow'].to(device),
    'is_weekend': batch['is_weekend'].to(device)
}

# Warmup
with torch.no_grad():
    _ = model(x_traffic, edge_index, x_weather, temporal)

# Measure
torch.cuda.synchronize() if device.type == 'cuda' else None
start = time.time()
with torch.no_grad():
    for _ in range(10):
        output = model(x_traffic, edge_index, x_weather, temporal)
torch.cuda.synchronize() if device.type == 'cuda' else None
forward_time = (time.time() - start) / 10

print(f"   Forward pass: {forward_time:.3f}s per batch")
print(f"   Device: {device}")

# 5. Check normalization
print("\n5. Verify Normalized Output:")
means = output['means']
print(f"   Output mean: {means.mean():.2f} (should be ~0 for normalized)")
print(f"   Output std: {means.std():.2f} (should be ~1 for normalized)")
print(f"   Output range: {means.min():.2f} to {means.max():.2f}")

if -2 <= means.mean() <= 2 and 0.5 <= means.std() <= 2:
    print("   ✓ Output appears normalized!")
else:
    print("   ⚠️ Output may not be normalized properly")

print("\n" + "="*60)
print("Estimated Training Performance:")
print("="*60)
data_time = avg_time * len(loader)
forward_time_total = forward_time * len(loader)
backward_time_est = forward_time_total * 2  # Backward typically 2x forward
total_time = data_time + forward_time_total + backward_time_est

print(f"Data loading: {data_time/60:.1f} min per epoch")
print(f"Forward pass: {forward_time_total/60:.1f} min per epoch")
print(f"Backward (est): {backward_time_est/60:.1f} min per epoch")
print(f"Total (est): {total_time/60:.1f} min per epoch")
print(f"\n100 epochs: {total_time*100/3600:.1f} hours")
print("="*60)
