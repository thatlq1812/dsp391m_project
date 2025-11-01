"""
STMGT Training Script

Purpose:
    - Train STMGT model on traffic data
    - Use AdamW optimizer with early stopping
    - Apply DropEdge regularization
    - Track multiple evaluation metrics

Author: DSP391m Team
Date: October 31, 2025
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from traffic_forecast.models.stmgt import STMGT, mixture_nll_loss
from traffic_forecast.data.stmgt_dataset import create_stmgt_dataloaders


class EarlyStopping:
    """Early stopping handler"""
    
    def __init__(self, patience=15, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None
    
    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
        elif self._is_improvement(score):
            self.best_score = score
            self.best_model_state = model.state_dict().copy()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
    
    def _is_improvement(self, score):
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta


class MetricsCalculator:
    """Calculate evaluation metrics"""
    
    @staticmethod
    def mae(pred, target):
        """Mean Absolute Error"""
        return torch.mean(torch.abs(pred - target)).item()
    
    @staticmethod
    def rmse(pred, target):
        """Root Mean Squared Error"""
        return torch.sqrt(torch.mean((pred - target) ** 2)).item()
    
    @staticmethod
    def r2(pred, target):
        """R-squared score"""
        ss_res = torch.sum((target - pred) ** 2)
        ss_tot = torch.sum((target - torch.mean(target)) ** 2)
        return (1 - ss_res / ss_tot).item()
    
    @staticmethod
    def mape(pred, target, epsilon=1e-3):
        """Mean Absolute Percentage Error"""
        mask = target > epsilon
        if mask.sum() == 0:
            return 0.0
        return torch.mean(torch.abs((target[mask] - pred[mask]) / target[mask])).item() * 100
    
    @staticmethod
    def crps_gaussian(pred_mean, pred_std, target):
        """
        Continuous Ranked Probability Score for Gaussian distribution
        Lower is better
        """
        # CRPS for normal distribution
        z = (target - pred_mean) / (pred_std + 1e-6)
        phi_z = torch.distributions.Normal(0, 1).cdf(z)
        pdf_z = torch.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
        
        crps = pred_std * (z * (2 * phi_z - 1) + 2 * pdf_z - 1 / np.sqrt(np.pi))
        return torch.mean(crps).item()
    
    @staticmethod
    def coverage_80(pred_mean, pred_std, target):
        """
        Coverage at 80% confidence interval
        Should be close to 0.80
        """
        # 80% CI: mean +- 1.28 * std
        lower = pred_mean - 1.28 * pred_std
        upper = pred_mean + 1.28 * pred_std
        
        coverage = ((target >= lower) & (target <= upper)).float().mean().item()
        return coverage


def drop_edge(edge_index, p=0.2):
    """
    Randomly drop edges with probability p
    """
    if p == 0:
        return edge_index
    
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges) > p
    return edge_index[:, mask]


def train_epoch(model, loader, optimizer, device, drop_edge_p=0.2):
    """Train one epoch"""
    model.train()
    total_loss = 0
    total_samples = 0
    
    for batch in tqdm(loader, desc="Training", leave=False):
        # Move to device
        x_traffic = batch['x_traffic'].to(device)
        x_weather = batch['x_weather'].to(device)
        edge_index = batch['edge_index'].to(device)
        temporal_features = {k: v.to(device) for k, v in batch['temporal_features'].items()}
        y_target = batch['y_target'].to(device)
        
        # Apply DropEdge
        edge_index_dropped = drop_edge(edge_index, p=drop_edge_p)
        
        # Forward pass
        pred_params = model(
            x_traffic,
            edge_index_dropped,
            x_weather,
            temporal_features
        )
        
        # Calculate loss
        loss = mixture_nll_loss(pred_params, y_target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        batch_size = x_traffic.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    
    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model"""
    model.eval()
    
    all_preds = []
    all_targets = []
    all_means = []
    all_stds = []
    total_loss = 0
    total_samples = 0
    
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        # Move to device
        x_traffic = batch['x_traffic'].to(device)
        x_weather = batch['x_weather'].to(device)
        edge_index = batch['edge_index'].to(device)
        temporal_features = {k: v.to(device) for k, v in batch['temporal_features'].items()}
        y_target = batch['y_target'].to(device)
        
        # Forward pass (no DropEdge)
        pred_params = model(
            x_traffic,
            edge_index,
            x_weather,
            temporal_features
        )
        
        # Calculate loss
        loss = mixture_nll_loss(pred_params, y_target)
        
        # Get point predictions (weighted mean)
        means = pred_params['means']  # [B, N, T, K]
        stds = pred_params['stds']  # [B, N, T, K]
        weights = torch.softmax(pred_params['logits'], dim=-1)  # [B, N, T, K]
        
        pred_mean = torch.sum(means * weights, dim=-1)  # [B, N, T]
        pred_std = torch.sqrt(torch.sum((stds ** 2 + means ** 2) * weights, dim=-1) - pred_mean ** 2)
        
        # Collect predictions
        all_preds.append(pred_mean.cpu())
        all_targets.append(y_target.cpu())
        all_means.append(pred_mean.cpu())
        all_stds.append(pred_std.cpu())
        
        batch_size = x_traffic.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    
    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_means = torch.cat(all_means, dim=0)
    all_stds = torch.cat(all_stds, dim=0)
    
    # Calculate metrics
    metrics = {
        'loss': total_loss / total_samples,
        'mae': MetricsCalculator.mae(all_preds, all_targets),
        'rmse': MetricsCalculator.rmse(all_preds, all_targets),
        'r2': MetricsCalculator.r2(all_preds, all_targets),
        'mape': MetricsCalculator.mape(all_preds, all_targets),
        'crps': MetricsCalculator.crps_gaussian(all_means, all_stds, all_targets),
        'coverage_80': MetricsCalculator.coverage_80(all_means, all_stds, all_targets)
    }
    
    return metrics


def main():
    """Main training loop"""
    
    # Configuration
    config = {
        'seq_len': 12,
        'pred_len': 12,
        'batch_size': 4,
        'num_workers': 0,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'max_epochs': 100,
        'patience': 15,
        'drop_edge_p': 0.2,
        'num_components': 3,
        'hidden_dim': 64,
        'num_heads': 4,
        'num_layers': 2
    }
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path('outputs') / f'stmgt_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Creating dataloaders...")
    try:
        train_loader, val_loader, test_loader, num_nodes, edge_index = create_stmgt_dataloaders(
            data_path='data/processed/all_runs_augmented.parquet',  # Use augmented data
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            seq_len=config['seq_len'],
            pred_len=config['pred_len']
        )
    except ValueError as e:
        print(f"Error creating dataloaders: {e}")
        print("Not enough data for validation/test sets. Adjusting split ratios...")
        
        # Use smaller seq/pred or different split
        config['seq_len'] = 6
        config['pred_len'] = 6
        
        train_loader, val_loader, test_loader, num_nodes, edge_index = create_stmgt_dataloaders(
            batch_size=config['batch_size'],
            num_workers=config['num_workers'],
            seq_len=config['seq_len'],
            pred_len=config['pred_len']
        )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader) if val_loader else 0}")
    print(f"Test batches: {len(test_loader) if test_loader else 0}")
    
    # Create model
    print("Creating model...")
    model = STMGT(
        num_nodes=num_nodes,
        mixture_components=config['num_components'],
        hidden_dim=config['hidden_dim'],
        num_heads=config['num_heads'],
        num_blocks=config['num_layers'],
        seq_len=config['seq_len'],
        pred_len=config['pred_len']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config['patience'],
        mode='min'
    )
    
    # Training loop
    print("\nStarting training...")
    history = {
        'train_loss': [],
        'val_metrics': []
    }
    
    best_val_mae = float('inf')
    
    for epoch in range(config['max_epochs']):
        print(f"\nEpoch {epoch+1}/{config['max_epochs']}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            drop_edge_p=config['drop_edge_p']
        )
        history['train_loss'].append(train_loss)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate (if available)
        if val_loader and len(val_loader) > 0:
            val_metrics = evaluate(model, val_loader, device)
            history['val_metrics'].append(val_metrics)
            
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val MAE: {val_metrics['mae']:.4f} km/h")
            print(f"Val RMSE: {val_metrics['rmse']:.4f} km/h")
            print(f"Val R2: {val_metrics['r2']:.4f}")
            print(f"Val MAPE: {val_metrics['mape']:.2f}%")
            print(f"Val CRPS: {val_metrics['crps']:.4f}")
            print(f"Val Coverage@80: {val_metrics['coverage_80']:.4f}")
            
            # Update scheduler
            scheduler.step(val_metrics['loss'])
            
            # Early stopping
            early_stopping(val_metrics['mae'], model)
            
            if val_metrics['mae'] < best_val_mae:
                best_val_mae = val_metrics['mae']
                torch.save(model.state_dict(), output_dir / 'best_model.pt')
                print(f"Saved best model (MAE: {best_val_mae:.4f})")
            
            if early_stopping.early_stop:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
        else:
            # No validation, just save periodically
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), output_dir / f'model_epoch_{epoch+1}.pt')
    
    # Save final model
    torch.save(model.state_dict(), output_dir / 'final_model.pt')
    
    # Save history
    history_path = output_dir / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2, default=float)
    
    print(f"\nTraining complete. Results saved to {output_dir}")
    
    # Test evaluation (if available)
    if test_loader and len(test_loader) > 0:
        print("\nEvaluating on test set...")
        
        # Load best model if available
        best_model_path = output_dir / 'best_model.pt'
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path))
            print("Loaded best model")
        
        test_metrics = evaluate(model, test_loader, device)
        
        print("\nTest Results:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  MAE: {test_metrics['mae']:.4f} km/h")
        print(f"  RMSE: {test_metrics['rmse']:.4f} km/h")
        print(f"  R2: {test_metrics['r2']:.4f}")
        print(f"  MAPE: {test_metrics['mape']:.2f}%")
        print(f"  CRPS: {test_metrics['crps']:.4f}")
        print(f"  Coverage@80: {test_metrics['coverage_80']:.4f}")
        
        # Save test results
        with open(output_dir / 'test_results.json', 'w') as f:
            json.dump(test_metrics, f, indent=2, default=float)


if __name__ == '__main__':
    main()
