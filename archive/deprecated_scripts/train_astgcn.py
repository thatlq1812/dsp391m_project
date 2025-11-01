"""
ASTGCN Training Script
Integrated with existing data pipeline

Usage:
    python scripts/train_astgcn.py --epochs 50 --batch-size 32
    python scripts/train_astgcn.py --quick-test  # Fast test mode
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from traffic_forecast.models.graph.astgcn_pytorch import (
    create_astgcn_model,
    build_adjacency_from_edges,
    build_adjacency_from_coords,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrafficDataset(Dataset):
    """PyTorch dataset for ASTGCN training."""
    
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        """
        Args:
            X: Input sequences (S, N, F, T_in)
            Y: Target sequences (S, N, F, T_out)
        """
        self.X = torch.from_numpy(X).float()
        self.Y = torch.from_numpy(Y).float()
        
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def load_processed_data(
    data_path: Path,
    features: list = ['speed_kmh']
) -> Tuple[pd.DataFrame, list]:
    """
    Load preprocessed data from parquet.
    
    Args:
        data_path: Path to all_runs_combined.parquet
        features: List of feature columns to use
        
    Returns:
        DataFrame and list of unique nodes
    """
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    
    # Ensure timestamp column
    if 'timestamp' not in df.columns:
        raise ValueError("Data must have 'timestamp' column")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Get unique nodes (from node_a_id, node_b_id)
    nodes = sorted(set(df['node_a_id'].unique()) | set(df['node_b_id'].unique()))
    
    logger.info(f"Loaded {len(df)} records, {len(nodes)} unique nodes")
    logger.info(f"Features: {features}")
    
    return df, nodes


def build_graph_structure(df: pd.DataFrame, nodes: list) -> Tuple[np.ndarray, Dict]:
    """
    Build adjacency matrix and node info from dataframe.
    
    Args:
        df: Traffic data
        nodes: List of node IDs
        
    Returns:
        Adjacency matrix and node coordinates dict
    """
    # Extract edges
    edges = df[['node_a_id', 'node_b_id']].drop_duplicates()
    edge_list = [(row['node_a_id'], row['node_b_id']) for _, row in edges.iterrows()]
    
    # Build adjacency from edges
    A = build_adjacency_from_edges(edge_list, nodes)
    
    # Extract coordinates (if available)
    coords = {}
    for node in nodes:
        # Try to get coordinates from either node_a or node_b columns
        node_a_rows = df[df['node_a_id'] == node]
        if not node_a_rows.empty:
            row = node_a_rows.iloc[0]
            lat = row.get('node_a_lat', row.get('node_a_lat_x', 0))
            lon = row.get('node_a_lon', row.get('node_a_lon_x', 0))
        else:
            node_b_rows = df[df['node_b_id'] == node]
            if not node_b_rows.empty:
                row = node_b_rows.iloc[0]
                lat = row.get('node_b_lat', row.get('node_b_lat_x', 0))
                lon = row.get('node_b_lon', row.get('node_b_lon_x', 0))
            else:
                lat, lon = 0, 0
        
        coords[node] = (float(lat) if pd.notna(lat) else 0.0,
                       float(lon) if pd.notna(lon) else 0.0)
    
    logger.info(f"Built graph: {A.shape[0]} nodes, {int(A.sum())/2} edges")
    
    return A, coords


def create_sequences(
    df: pd.DataFrame,
    nodes: list,
    features: list,
    T_in: int,
    T_out: int
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Create input/output sequences from dataframe.
    
    Args:
        df: Traffic data
        nodes: List of node IDs
        features: Feature columns
        T_in: Input sequence length
        T_out: Output sequence length
        
    Returns:
        X (S, N, F, T_in), Y (S, N, F, T_out), normalization_params
    """
    N = len(nodes)
    F = len(features)
    node_idx = {n: i for i, n in enumerate(nodes)}
    
    # Get unique timestamps
    timestamps = sorted(df['timestamp'].unique())
    T_total = len(timestamps)
    time_idx = {t: i for i, t in enumerate(timestamps)}
    
    logger.info(f"Creating sequences: T_in={T_in}, T_out={T_out}, Total timesteps={T_total}")
    
    # Create data cube: (T, N, F)
    data_cube = np.zeros((T_total, N, F), dtype=np.float32)
    
    # Fill cube with edge data (average if multiple edges per node pair)
    for _, row in df.iterrows():
        t_idx = time_idx[row['timestamp']]
        
        # Add data for both nodes in the edge
        for node_col in ['node_a_id', 'node_b_id']:
            node = row[node_col]
            if node in node_idx:
                n_idx = node_idx[node]
                for f_idx, feat in enumerate(features):
                    if feat in row and pd.notna(row[feat]):
                        # Average if multiple values (simple aggregation)
                        data_cube[t_idx, n_idx, f_idx] = row[feat]
    
    # Normalize per feature (and save normalization params)
    normalization_params = {}
    for f_idx, feat in enumerate(features):
        feature_data = data_cube[:, :, f_idx]
        mean = feature_data.mean()
        std = feature_data.std() + 1e-6
        data_cube[:, :, f_idx] = (feature_data - mean) / std
        normalization_params[feat] = {'mean': float(mean), 'std': float(std)}
    
    # Create sliding window sequences
    X_list, Y_list = [], []
    
    for end_idx in range(T_in, T_total - T_out + 1):
        start_idx = end_idx - T_in
        
        # X: (N, F, T_in)
        X_seq = data_cube[start_idx:end_idx].transpose(1, 2, 0)  # (N, F, T_in)
        
        # Y: (N, F, T_out)
        Y_seq = data_cube[end_idx:end_idx + T_out].transpose(1, 2, 0)  # (N, F, T_out)
        
        X_list.append(X_seq)
        Y_list.append(Y_seq)
    
    X = np.stack(X_list, axis=0)  # (S, N, F, T_in)
    Y = np.stack(Y_list, axis=0)  # (S, N, F, T_out)
    
    logger.info(f"Created {len(X)} sequences: X{X.shape}, Y{Y.shape}")
    
    return X, Y, normalization_params


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for X, Y in loader:
        X = X.to(device)
        Y = Y.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(X)
        
        # Compute loss
        loss = criterion(pred, Y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * X.size(0)
    
    return total_loss / len(loader.dataset)


def eval_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    norm_params: Dict = None
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for X, Y in loader:
            X = X.to(device)
            Y = Y.to(device)
            
            pred = model(X)
            loss = criterion(pred, Y)
            
            total_loss += loss.item() * X.size(0)
            
            all_preds.append(pred.cpu().numpy())
            all_targets.append(Y.cpu().numpy())
    
    avg_loss = total_loss / len(loader.dataset)
    
    # Compute metrics
    y_pred = np.concatenate(all_preds, axis=0).flatten()
    y_true = np.concatenate(all_targets, axis=0).flatten()
    
    # Denormalize if params provided
    if norm_params and 'speed_kmh' in norm_params:
        mean = norm_params['speed_kmh']['mean']
        std = norm_params['speed_kmh']['std']
        y_pred_denorm = y_pred * std + mean
        y_true_denorm = y_true * std + mean
    else:
        y_pred_denorm = y_pred
        y_true_denorm = y_true
    
    # Filter values > 5 km/h for MAPE (avoid division by near-zero)
    mask = y_true_denorm > 5.0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true_denorm[mask] - y_pred_denorm[mask]) / y_true_denorm[mask])) * 100
    else:
        mape = 0.0
    
    metrics = {
        'mse': mean_squared_error(y_true_denorm, y_pred_denorm),
        'rmse': np.sqrt(mean_squared_error(y_true_denorm, y_pred_denorm)),
        'mae': mean_absolute_error(y_true_denorm, y_pred_denorm),
        'mape': mape,
        'r2': r2_score(y_true_denorm, y_pred_denorm)
    }
    
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description='Train ASTGCN model')
    
    # Data parameters
    parser.add_argument('--data', type=str, default='data/processed/all_runs_combined.parquet',
                       help='Path to processed data')
    parser.add_argument('--features', type=str, nargs='+', default=['speed_kmh'],
                       help='Features to use')
    
    # Model parameters
    parser.add_argument('--T-in', type=int, default=12,
                       help='Input sequence length (hours)')
    parser.add_argument('--T-out', type=int, default=3,
                       help='Output sequence length (hours)')
    parser.add_argument('--K-cheb', type=int, default=3,
                       help='Chebyshev polynomial order')
    parser.add_argument('--hidden-channels', type=int, default=64,
                       help='Hidden layer size')
    parser.add_argument('--num-blocks', type=int, default=2,
                       help='Number of ST blocks')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='L2 regularization weight decay')
    parser.add_argument('--val-split', type=float, default=0.15,
                       help='Validation split ratio')
    parser.add_argument('--test-split', type=float, default=0.15,
                       help='Test split ratio')
    parser.add_argument('--early-stopping', type=int, default=10,
                       help='Early stopping patience (epochs)')
    
    # Other
    parser.add_argument('--output-dir', type=str, default='models/saved/astgcn',
                       help='Output directory for models')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test mode (1 epoch, small data)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick_test:
        logger.info("QUICK TEST MODE")
        args.epochs = 1
        args.T_in = 4
        args.T_out = 1
    
    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df, nodes = load_processed_data(Path(args.data), args.features)
    
    # Build graph structure
    A, coords = build_graph_structure(df, nodes)
    
    # Create sequences
    X, Y, norm_params = create_sequences(df, nodes, args.features, args.T_in, args.T_out)
    logger.info(f"Normalization params: {norm_params}")
    
    # Train/val/test split - TIME-BASED to avoid data leakage
    # Sort by timestamp order (sequences are already chronological)
    # Train: first 70%, Val: next 15%, Test: last 15%
    n_samples = len(X)
    train_size = int(n_samples * (1 - args.test_split - args.val_split))
    val_size = int(n_samples * args.val_split)
    
    train_idx = np.arange(0, train_size)
    val_idx = np.arange(train_size, train_size + val_size)
    test_idx = np.arange(train_size + val_size, n_samples)
    
    logger.info(f"Time-based split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    
    # Create datasets
    train_dataset = TrafficDataset(X[train_idx], Y[train_idx])
    val_dataset = TrafficDataset(X[val_idx], Y[val_idx])
    test_dataset = TrafficDataset(X[test_idx], Y[test_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    logger.info(f"Dataset splits: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    
    # Create model
    model = create_astgcn_model(
        num_nodes=len(nodes),
        in_features=len(args.features),
        time_steps_in=args.T_in,
        time_steps_out=args.T_out,
        adjacency=A,
        K_cheb=args.K_cheb,
        hidden_channels=args.hidden_channels,
        num_blocks=args.num_blocks,
        device=device
    )
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = {'train': [], 'val': []}
    
    logger.info(f"\nStarting training for {args.epochs} epochs (early stopping: {args.early_stopping})...")
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_metrics = eval_model(model, val_loader, criterion, device, norm_params)
        
        # Track history
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        
        epoch_time = time.time() - start_time
        
        logger.info(
            f"Epoch {epoch+1}/{args.epochs} - "
            f"train_loss: {train_loss:.6f}, "
            f"val_loss: {val_loss:.6f}, "
            f"time: {epoch_time:.1f}s"
        )
        
        # Save best model and check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'nodes': nodes,
                'adjacency': A,
                'config': vars(args)
            }
            torch.save(checkpoint, output_dir / 'astgcn_best.pth')
            logger.info(f"  ✓ Saved best model (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            logger.info(f"  No improvement for {patience_counter} epochs")
            
            if patience_counter >= args.early_stopping:
                logger.info(f"\n⚠ Early stopping triggered at epoch {epoch+1}")
                logger.info(f"Best model was at epoch {best_epoch+1} with val_loss={best_val_loss:.6f}")
                break
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    test_loss, test_metrics = eval_model(model, test_loader, criterion, device, norm_params)
    
    logger.info("\n" + "="*60)
    logger.info("TEST SET EVALUATION")
    logger.info("="*60)
    for metric, value in test_metrics.items():
        logger.info(f"{metric.upper():>6}: {value:.4f}")
    logger.info("="*60)
    
    # Save final results
    results = {
        'test_metrics': test_metrics,
        'training_history': history,
        'best_val_loss': best_val_loss,
        'config': vars(args)
    }
    
    with open(output_dir / 'training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {output_dir}/")
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
