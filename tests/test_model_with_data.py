"""Quick test: Model + Real Data"""

import torch
from traffic_forecast.models.stmgt.model import STMGT
from traffic_forecast.data.stmgt_dataset import create_stmgt_dataloaders

# Create dataloaders
print("Creating dataloaders...")
train_loader, val_loader, test_loader, num_nodes, edge_index = create_stmgt_dataloaders(
    batch_size=4,
    seq_len=12,
    pred_len=12
)

print(f"Num nodes: {num_nodes}")
print(f"Num edges: {edge_index.size(1)}")

# Create model
print("\nCreating model...")
model = STMGT(
    num_nodes=num_nodes,
    mixture_components=3,
    hidden_dim=64,
    num_heads=4,
    num_blocks=2,
    seq_len=12,
    pred_len=12
)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Get one batch
print("\nTesting forward pass...")
batch = next(iter(train_loader))

print(f"Input shapes:")
print(f"  x_traffic: {batch['x_traffic'].shape}")
print(f"  x_weather: {batch['x_weather'].shape}")
print(f"  edge_index: {batch['edge_index'].shape}")
print(f"  y_target: {batch['y_target'].shape}")

# Forward pass
with torch.no_grad():
    pred_params = model(
        batch['x_traffic'],
        batch['edge_index'],
        batch['x_weather'],
        batch['temporal_features']
    )

print(f"\nOutput shapes:")
print(f"  means: {pred_params['means'].shape}")
print(f"  stds: {pred_params['stds'].shape}")
print(f"  logits: {pred_params['logits'].shape}")

print("\nSample predictions (first node, first timestep):")
print(f"  Means (3 components): {pred_params['means'][0, 0, 0, :]}")
print(f"  Stds (3 components): {pred_params['stds'][0, 0, 0, :]}")
print(f"  Logits (3 components): {pred_params['logits'][0, 0, 0, :]}")

print("\nTest PASSED!")
