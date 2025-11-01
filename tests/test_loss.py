"""Test model + loss"""
from traffic_forecast.models.stmgt import STMGT, mixture_nll_loss
from traffic_forecast.data.stmgt_dataset import create_stmgt_dataloaders
import torch

print("Creating dataloaders...")
tl, _, _, num_nodes, ei = create_stmgt_dataloaders(batch_size=4)
batch = next(iter(tl))

print("Creating model...")
model = STMGT(
    num_nodes=num_nodes,
    mixture_components=3,
    hidden_dim=64,
    num_heads=4,
    num_blocks=2,
    seq_len=12,
    pred_len=12
)

print("Forward pass...")
pred_params = model(
    batch['x_traffic'],
    batch['edge_index'],
    batch['x_weather'],
    batch['temporal_features']
)

print("Computing loss...")
loss = mixture_nll_loss(pred_params, batch['y_target'])
print(f'Loss: {loss.item():.4f}')

print("\nBackward pass...")
loss.backward()
print("Gradients computed successfully!")

print("\nTest PASSED!")
