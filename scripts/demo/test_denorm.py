"""Test denormalization issue"""
import torch
import json
from pathlib import Path
import sys
sys.path.insert(0, 'D:/UNI/DSP391m/project')

from traffic_forecast.models.stmgt.model import STMGT

# Load model
checkpoint = torch.load('outputs/stmgt_baseline_1month_20251115_132552/best_model.pt', 
                       map_location='cpu', weights_only=False)

config = json.loads(Path('outputs/stmgt_baseline_1month_20251115_132552/config.json').read_text())

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

if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
elif 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()

# Create simple test input with known values
# Input: all nodes have speed=20 km/h
test_speed = 20.0
x_traffic = torch.full((1, 62, 12, 1), test_speed)

# Simple edge index (chain)
edges = [[i, i+1] for i in range(61)]
edges += [[i+1, i] for i in range(61)]
edge_index = torch.LongTensor(edges).t()

# Dummy weather and temporal
x_weather = torch.zeros(1, 12, 3)
x_weather[0, :, 0] = 27.0  # temp
x_weather[0, :, 1] = 15.0  # humidity
x_weather[0, :, 2] = 0.0   # precip

temporal_features = {
    'hour': torch.LongTensor([16]),
    'dow': torch.LongTensor([2]),  # Wednesday
    'is_weekend': torch.LongTensor([0])
}

print("=" * 80)
print("DENORMALIZATION TEST")
print("=" * 80)

print(f"\nInput speed: {test_speed} km/h (all nodes)")
print(f"Model normalization: mean={model.speed_normalizer.mean.item():.2f}, std={model.speed_normalizer.std.item():.2f}")

# Test with denormalize=False (OLD behavior)
with torch.no_grad():
    pred_false = model.predict(x_traffic, edge_index, x_weather, temporal_features, denormalize=False)

print("\n1. WITH denormalize=FALSE (old models):")
print(f"   Keys: {pred_false.keys()}")
mu_false = pred_false['means']
print(f"   means shape: {mu_false.shape}")
print(f"   means range: {mu_false.min():.2f} - {mu_false.max():.2f}")
print(f"   means mean: {mu_false.mean():.2f}")

# Test with denormalize=True (NEW behavior)
with torch.no_grad():
    pred_true = model.predict(x_traffic, edge_index, x_weather, temporal_features, denormalize=True)

print("\n2. WITH denormalize=TRUE (new models):")
print(f"   Keys: {pred_true.keys()}")
mu_true = pred_true['means']
print(f"   means shape: {mu_true.shape}")
print(f"   means range: {mu_true.min():.2f} - {mu_true.max():.2f}")
print(f"   means mean: {mu_true.mean():.2f}")

# Calculate expected if double normalized
expected_double = test_speed * model.speed_normalizer.std.item() + model.speed_normalizer.mean.item()
print(f"\n3. ANALYSIS:")
print(f"   If model outputs normalized values and we denormalize:")
print(f"     Expected: ~{test_speed} km/h")
print(f"   If model outputs unnormalized and we denormalize again:")
print(f"     Expected: ~{expected_double:.2f} km/h")
print(f"\n   Actual with denormalize=FALSE: {mu_false.mean():.2f}")
print(f"   Actual with denormalize=TRUE: {mu_true.mean():.2f}")

# Determine which is correct
print(f"\n4. CONCLUSION:")
if abs(mu_false.mean() - test_speed) < abs(mu_true.mean() - test_speed):
    print(f"   ✓ Model outputs UNNORMALIZED values (raw km/h)")
    print(f"   ✓ Should use denormalize=FALSE in demo script")
    print(f"   ✗ Current script uses denormalize=TRUE → WRONG!")
    print(f"   ERROR MAGNITUDE: {abs(mu_true.mean() - test_speed):.2f} km/h")
else:
    print(f"   ✓ Model outputs NORMALIZED values")
    print(f"   ✓ Should use denormalize=TRUE in demo script")
    print(f"   ✓ Current script is CORRECT")
