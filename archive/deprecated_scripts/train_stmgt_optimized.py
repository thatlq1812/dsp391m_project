"""
OPTIMIZED STMGT Training Script

Improvements:
- Use EXTREME augmented data (1,839 runs)
- Larger batch size (32)
- Multi-worker data loading (2 workers for Windows)
- Mixed precision training (AMP)
- Gradient accumulation
- Better progress tracking

Author: DSP391m Team
Date: November 1, 2025
"""

# Minimal top-level imports for Windows multiprocessing
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def train_epoch(model, loader, optimizer, device, scaler, drop_edge_p=0.2, accumulation_steps=1):
    """Train one epoch with mixed precision"""
    import torch
    from tqdm import tqdm
    from traffic_forecast.models.stmgt import mixture_nll_loss
    from torch.amp import autocast
    
    model.train()
    total_loss = 0
    total_samples = 0
    
    optimizer.zero_grad()
    
    def drop_edge(edge_index, p=0.2):
        """Randomly drop edges with probability p"""
        if p == 0:
            return edge_index
        num_edges = edge_index.size(1)
        mask = torch.rand(num_edges) > p
        return edge_index[:, mask]
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        x_traffic = batch['x_traffic'].to(device)
        x_weather = batch['x_weather'].to(device)
        edge_index = batch['edge_index'].to(device)
        temporal_features = {k: v.to(device) for k, v in batch['temporal_features'].items()}
        y_target = batch['y_target'].to(device)
        
        # Apply DropEdge
        edge_index_dropped = drop_edge(edge_index, p=drop_edge_p)
        
        # Mixed precision forward pass
        with autocast(device_type='cuda', enabled=scaler is not None):
            pred_params = model(
                x_traffic,
                edge_index_dropped,
                x_weather,
                temporal_features
            )
            loss = mixture_nll_loss(pred_params, y_target)
            loss = loss / accumulation_steps
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Track metrics
        batch_size = x_traffic.size(0)
        total_loss += loss.item() * batch_size * accumulation_steps
        total_samples += batch_size
        
        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
    
    return total_loss / total_samples
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
        """Continuous Ranked Probability Score for Gaussian distribution"""
        z = (target - pred_mean) / (pred_std + 1e-6)
        phi_z = torch.distributions.Normal(0, 1).cdf(z)
        pdf_z = torch.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
        crps = pred_std * (z * (2 * phi_z - 1) + 2 * pdf_z - 1 / np.sqrt(np.pi))
        return torch.mean(crps).item()
    
    @staticmethod
    def coverage_80(pred_mean, pred_std, target):
        """Coverage at 80% confidence interval"""
        lower = pred_mean - 1.28 * pred_std
        upper = pred_mean + 1.28 * pred_std
        coverage = ((target >= lower) & (target <= upper)).float().mean().item()
        return coverage


def drop_edge(edge_index, p=0.2):
    """Randomly drop edges with probability p"""
    if p == 0:
        return edge_index
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges) > p
    return edge_index[:, mask]


def train_epoch(model, loader, optimizer, device, scaler, drop_edge_p=0.2, accumulation_steps=1):
    """Train one epoch with mixed precision"""
    model.train()
    total_loss = 0
    total_samples = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        x_traffic = batch['x_traffic'].to(device)
        x_weather = batch['x_weather'].to(device)
        edge_index = batch['edge_index'].to(device)
        temporal_features = {k: v.to(device) for k, v in batch['temporal_features'].items()}
        y_target = batch['y_target'].to(device)
        
        # Apply DropEdge
        edge_index_dropped = drop_edge(edge_index, p=drop_edge_p)
        
        # Mixed precision forward pass
        with autocast(device_type='cuda', enabled=scaler is not None):
            pred_params = model(
                x_traffic,
                edge_index_dropped,
                x_weather,
                temporal_features
            )
            loss = mixture_nll_loss(pred_params, y_target)
            loss = loss / accumulation_steps
        
        # Backward with gradient scaling
        scaler.scale(loss).backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Track metrics
        batch_size = x_traffic.size(0)
        total_loss += loss.item() * batch_size * accumulation_steps
        total_samples += batch_size
        
        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})
    
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
        x_traffic = batch['x_traffic'].to(device)
        x_weather = batch['x_weather'].to(device)
        edge_index = batch['edge_index'].to(device)
        temporal_features = {k: v.to(device) for k, v in batch['temporal_features'].items()}
        y_target = batch['y_target'].to(device)
        
        # Forward pass
        pred_params = model(x_traffic, edge_index, x_weather, temporal_features)
        loss = mixture_nll_loss(pred_params, y_target)
        
        # Get predictions
        means = pred_params['means']
        stds = pred_params['stds']
        weights = torch.softmax(pred_params['logits'], dim=-1)
        
        pred_mean = torch.sum(means * weights, dim=-1)
        pred_std = torch.sqrt(torch.sum((stds ** 2 + means ** 2) * weights, dim=-1) - pred_mean ** 2)
        
        all_preds.append(pred_mean.cpu())
        all_targets.append(y_target.cpu())
        all_means.append(pred_mean.cpu())
        all_stds.append(pred_std.cpu())
        
        batch_size = x_traffic.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    
    # Concatenate
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
    
    # Configuration - OPTIMIZED
    config = {
        'seq_len': 12,
        'pred_len': 12,
        'batch_size': 64,  # Increased to 64 for better GPU utilization
        'num_workers': 0,   # Windows compatibility (use 0 to avoid multiprocessing issues)
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'max_epochs': 100,
        'patience': 20,  # Increased patience
        'drop_edge_p': 0.2,
        'num_components': 3,
        'hidden_dim': 64,
        'num_heads': 4,
        'num_layers': 2,
        'use_amp': True,  # Mixed precision
        'accumulation_steps': 1
    }
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"Data workers: {config['num_workers']}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path('outputs') / f'stmgt_extreme_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 70)
    print("CREATING DATALOADERS (EXTREME AUGMENTED DATA)")
    print("=" * 70)
    
    # Use extreme augmented data
    data_path = 'data/processed/all_runs_extreme_augmented.parquet'
    if not Path(data_path).exists():
        print(f"⚠ {data_path} not found, using regular augmented data")
        data_path = 'data/processed/all_runs_augmented.parquet'
    
    train_loader, val_loader, test_loader, num_nodes, edge_index = create_stmgt_dataloaders(
        data_path=data_path,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        seq_len=config['seq_len'],
        pred_len=config['pred_len']
    )
    
    print(f"\nDataset Statistics:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edges: {edge_index.size(1)}")
    
    # Create model
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)
    
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
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
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
    
    # Mixed precision scaler
    scaler = GradScaler() if config['use_amp'] else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config['patience'], mode='min')
    
    # Training loop
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Mixed precision: {config['use_amp']}")
    print(f"  Max epochs: {config['max_epochs']}")
    print(f"  Early stopping patience: {config['patience']}")
    print("=" * 70 + "\n")
    
    history = {
        'train_loss': [],
        'val_metrics': []
    }
    
    best_val_mae = float('inf')
    
    for epoch in range(config['max_epochs']):
        print(f"\nEpoch {epoch+1}/{config['max_epochs']}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, scaler,
            drop_edge_p=config['drop_edge_p'],
            accumulation_steps=config['accumulation_steps']
        )
        history['train_loss'].append(train_loss)
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
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
                print(f"✓ Saved best model (MAE: {best_val_mae:.4f})")
            
            if early_stopping.early_stop:
                print(f"\n🛑 Early stopping triggered at epoch {epoch+1}")
                break
        else:
            if (epoch + 1) % 10 == 0:
                torch.save(model.state_dict(), output_dir / f'model_epoch_{epoch+1}.pt')
    
    # Save final model
    torch.save(model.state_dict(), output_dir / 'final_model.pt')
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2, default=float)
    
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"Best validation MAE: {best_val_mae:.4f} km/h")
    
    # Test evaluation
    if test_loader and len(test_loader) > 0:
        print(f"\n{'='*70}")
        print("EVALUATING ON TEST SET")
        print(f"{'='*70}")
        
        # Load best model
        best_model_path = output_dir / 'best_model.pt'
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path))
            print("✓ Loaded best model")
        
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
    
    print(f"\n{'='*70}")
    print("🎉 ALL DONE!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    # Set multiprocessing start method for Windows compatibility
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set
    
    main()
