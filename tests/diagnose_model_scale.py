"""
Test if current trained model outputs normalized or denormalized predictions
"""
import torch
from traffic_forecast.models.stmgt.model import STMGT

print("="*60)
print("Analyzing Trained Model Output Scale")
print("="*60)

# Load checkpoint
checkpoint = torch.load("traffic_api/models/stmgt_production.pt", map_location="cpu", weights_only=False)

# Create model
model = STMGT(
    num_nodes=62,
    in_dim=1,
    hidden_dim=64,
    num_blocks=3,
    num_heads=4,
    dropout=0.2,
    drop_edge_rate=0.05,
    mixture_components=3,
    seq_len=12,
    pred_len=12,
    speed_mean=18.72,
    speed_std=7.03,
    weather_mean=[27.4877, 6.0848, 0.1631],
    weather_std=[1.9255, 3.4700, 0.2664]
)
model.load_state_dict(checkpoint)
model.eval()

# Create input matching training distribution
batch_size = 1
num_nodes = 62
seq_len = 12
pred_len = 12

# IMPORTANT: Create input in RAW scale (as training data provides)
x_traffic_raw = torch.randn(batch_size, num_nodes, seq_len, 1) * 7 + 19
x_weather_raw = torch.randn(batch_size, pred_len, 3)
x_weather_raw[:, :, 0] = x_weather_raw[:, :, 0] * 2 + 27
x_weather_raw[:, :, 1] = x_weather_raw[:, :, 1] * 3 + 6
x_weather_raw[:, :, 2] = x_weather_raw[:, :, 2].abs() * 0.3

edge_index = torch.stack([
    torch.tensor([i for i in range(62) for _ in range(2)]),
    torch.tensor([j for i in range(62) for j in [(i+1)%62, (i+2)%62]])
], dim=0)

temporal = {
    'hour': torch.randint(0, 24, (batch_size, seq_len)),
    'dow': torch.randint(0, 7, (batch_size, seq_len)),
    'is_weekend': torch.randint(0, 2, (batch_size, seq_len))
}

print("\n1. Input Statistics (raw scale):")
print(f"   Traffic: mean={x_traffic_raw.mean():.2f}, std={x_traffic_raw.std():.2f} km/h")
print(f"   Weather temp: mean={x_weather_raw[:,:,0].mean():.2f}, std={x_weather_raw[:,:,0].std():.2f}")

with torch.no_grad():
    # Test what model.forward() returns (should be normalized)
    output = model.forward(x_traffic_raw, edge_index, x_weather_raw, temporal)
    
    print("\n2. Model.forward() output (should be NORMALIZED):")
    print(f"   means: min={output['means'].min():.2f}, max={output['means'].max():.2f}, mean={output['means'].mean():.2f}")
    print(f"   stds: min={output['stds'].min():.2f}, max={output['stds'].max():.2f}, mean={output['stds'].mean():.2f}")
    
    # Manually denormalize
    means_denorm = model.speed_normalizer.denormalize(output['means'])
    stds_denorm = output['stds'] * model.speed_normalizer.std
    
    print("\n3. Manually denormalized:")
    print(f"   means: min={means_denorm.min():.2f}, max={means_denorm.max():.2f}, mean={means_denorm.mean():.2f} km/h")
    print(f"   stds: min={stds_denorm.min():.2f}, max={stds_denorm.max():.2f}, mean={stds_denorm.mean():.2f} km/h")
    
    # Test model.predict() (should do denormalization internally)
    pred_output = model.predict(x_traffic_raw, edge_index, x_weather_raw, temporal)
    
    print("\n4. Model.predict() output (should be DENORMALIZED):")
    print(f"   means: min={pred_output['means'].min():.2f}, max={pred_output['means'].max():.2f}, mean={pred_output['means'].mean():.2f} km/h")
    print(f"   stds: min={pred_output['stds'].min():.2f}, max={pred_output['stds'].max():.2f}, mean={pred_output['stds'].mean():.2f} km/h")

print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)

# Check if model was trained incorrectly
if output['means'].mean() > 5:
    print("❌ PROBLEM DETECTED!")
    print("   Model.forward() outputs are NOT normalized (mean > 5)")
    print("   This means model was trained to predict RAW speeds directly")
    print("   despite inputs being normalized.")
    print("\n   ROOT CAUSE:")
    print("   - Training loss compared normalized predictions with RAW targets")
    print("   - Model learned to output RAW scale despite normalized inputs")
    print("\n   FIX REQUIRED:")
    print("   - Retrain model with normalized targets in loss")
    print("   - OR: Remove normalization layers from model")
elif -2 <= output['means'].mean() <= 2:
    print("✅ Model outputs are properly NORMALIZED")
    print("   Mean ~0, predictions in normalized space")
    print("   Denormalization in predict() is correct")
else:
    print("⚠️  Uncertain - output mean is {:.2f}".format(output['means'].mean()))
    
print("="*60)
