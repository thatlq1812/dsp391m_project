"""
Training monitoring and validation utilities for STMGT

Provides tools to monitor:
- Mixture component usage
- Fusion gate values (spatial vs temporal)
- Attention weights
- Training stability

Author: THAT Le Quang
Date: November 9, 2025
"""

import torch
import numpy as np
from typing import Dict, Optional


class STMGTMonitor:
    """
    Monitor STMGT training metrics for stability and diagnostics
    """
    
    def __init__(self):
        self.metrics_history = {
            'mixture_weights': [],
            'fusion_alpha': [],
            'loss_components': [],
            'gradients': []
        }
    
    def update(self, model: torch.nn.Module, loss_dict: Optional[Dict] = None):
        """
        Update monitoring metrics
        
        Args:
            model: STMGT model
            loss_dict: Dictionary with loss components (from mixture_nll_loss)
        """
        # Monitor mixture weights
        if loss_dict and 'mixture_weights' in loss_dict:
            weights = loss_dict['mixture_weights'].detach().cpu().numpy()
            self.metrics_history['mixture_weights'].append(weights)
        
        # Monitor loss components
        if loss_dict:
            self.metrics_history['loss_components'].append({
                'total': loss_dict.get('total', 0).item(),
                'nll': loss_dict.get('nll', 0).item(),
                'diversity': loss_dict.get('diversity', 0).item(),
                'entropy': loss_dict.get('entropy', 0).item()
            })
    
    def check_mixture_collapse(self, threshold=0.8):
        """
        Check if mixture has collapsed to single component
        
        Returns:
            bool: True if collapsed
        """
        if not self.metrics_history['mixture_weights']:
            return False
        
        recent_weights = self.metrics_history['mixture_weights'][-10:]
        mean_weights = np.mean(recent_weights, axis=0)
        
        return np.max(mean_weights) > threshold
    
    def check_gradient_health(self, model: torch.nn.Module):
        """
        Check for gradient issues (vanishing, exploding)
        
        Returns:
            dict with gradient statistics
        """
        total_norm = 0.0
        max_grad = 0.0
        min_grad = float('inf')
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                max_grad = max(max_grad, p.grad.abs().max().item())
                min_grad = min(min_grad, p.grad.abs().min().item())
        
        total_norm = total_norm ** 0.5
        
        return {
            'total_norm': total_norm,
            'max_grad': max_grad,
            'min_grad': min_grad,
            'is_exploding': total_norm > 10.0,
            'is_vanishing': total_norm < 1e-6
        }
    
    def get_summary(self):
        """Get summary of monitoring metrics"""
        summary = {}
        
        # Mixture weights
        if self.metrics_history['mixture_weights']:
            recent = self.metrics_history['mixture_weights'][-10:]
            mean_weights = np.mean(recent, axis=0)
            summary['mixture_weights'] = {
                'mean': mean_weights.tolist(),
                'collapsed': self.check_mixture_collapse()
            }
        
        # Loss components
        if self.metrics_history['loss_components']:
            recent = self.metrics_history['loss_components'][-10:]
            summary['loss_components'] = {
                'nll': np.mean([x['nll'] for x in recent]),
                'diversity': np.mean([x['diversity'] for x in recent]),
                'entropy': np.mean([x['entropy'] for x in recent])
            }
        
        return summary


def validate_model_output(pred_params: Dict, y_true: torch.Tensor):
    """
    Validate model output for correctness
    
    Args:
        pred_params: dict with 'means', 'stds', 'logits'
        y_true: Ground truth tensor
    
    Raises:
        AssertionError if validation fails
    """
    means = pred_params['means']
    stds = pred_params['stds']
    logits = pred_params['logits']
    
    # Check shapes match
    B, N, T, K = means.shape
    assert stds.shape == (B, N, T, K), f"Stds shape mismatch: {stds.shape}"
    assert logits.shape == (B, N, T, K), f"Logits shape mismatch: {logits.shape}"
    assert y_true.shape == (B, N, T), f"Y_true shape mismatch: {y_true.shape}"
    
    # Check no NaN or Inf
    assert not torch.isnan(means).any(), "Means contains NaN"
    assert not torch.isnan(stds).any(), "Stds contains NaN"
    assert not torch.isnan(logits).any(), "Logits contains NaN"
    assert not torch.isinf(means).any(), "Means contains Inf"
    assert not torch.isinf(stds).any(), "Stds contains Inf"
    assert not torch.isinf(logits).any(), "Logits contains Inf"
    
    # Check stds are positive
    assert (stds > 0).all(), f"Stds contains non-positive values: min={stds.min()}"
    
    # Check reasonable ranges (for normalized data)
    assert means.abs().max() < 10.0, f"Means out of range: max={means.abs().max()}"
    assert stds.max() < 10.0, f"Stds out of range: max={stds.max()}"
    
    # Compute mixture weights and check they sum to 1
    pi = torch.softmax(logits, dim=-1)
    pi_sum = pi.sum(dim=-1)
    assert torch.allclose(pi_sum, torch.ones_like(pi_sum), atol=1e-5), \
        f"Mixture weights don't sum to 1: {pi_sum.min()}, {pi_sum.max()}"


def validate_data_batch(batch: Dict):
    """
    Validate data batch for correctness
    
    Args:
        batch: Dictionary with data tensors
    
    Raises:
        AssertionError if validation fails
    """
    x_traffic = batch['x_traffic']
    x_weather = batch['x_weather']
    y_target = batch['y_target']
    temporal = batch['temporal_features']
    
    # Check shapes
    B, N, T, _ = x_traffic.shape
    assert x_weather.shape[0] == B, "Batch size mismatch"
    assert y_target.shape[0] == B, "Batch size mismatch"
    
    # Check no NaN
    assert not torch.isnan(x_traffic).any(), "Traffic data contains NaN"
    assert not torch.isnan(x_weather).any(), "Weather data contains NaN"
    assert not torch.isnan(y_target).any(), "Target data contains NaN"
    
    # Check temporal features
    hour = temporal['hour']
    dow = temporal['dow']
    is_weekend = temporal['is_weekend']
    
    assert (hour >= 0).all() and (hour <= 23).all(), f"Invalid hour: {hour.min()}, {hour.max()}"
    assert (dow >= 0).all() and (dow <= 6).all(), f"Invalid dow: {dow.min()}, {dow.max()}"
    assert (is_weekend >= 0).all() and (is_weekend <= 1).all(), "Invalid weekend flag"
    
    # Check speed is positive
    assert (x_traffic >= 0).all(), f"Negative speed: {x_traffic.min()}"
    assert (y_target >= 0).all(), f"Negative target: {y_target.min()}"


def print_training_diagnostics(
    epoch: int,
    train_loss: float,
    val_loss: float,
    monitor: STMGTMonitor,
    model: torch.nn.Module
):
    """
    Print training diagnostics
    
    Args:
        epoch: Current epoch
        train_loss: Training loss
        val_loss: Validation loss
        monitor: STMGTMonitor instance
        model: STMGT model
    """
    print(f"\n{'='*60}")
    print(f"Epoch {epoch} Diagnostics")
    print(f"{'='*60}")
    
    print(f"\nLoss:")
    print(f"  Train: {train_loss:.4f}")
    print(f"  Val:   {val_loss:.4f}")
    
    # Get summary
    summary = monitor.get_summary()
    
    # Mixture weights
    if 'mixture_weights' in summary:
        weights = summary['mixture_weights']['mean']
        collapsed = summary['mixture_weights']['collapsed']
        print(f"\nMixture Weights:")
        for i, w in enumerate(weights):
            print(f"  Component {i+1}: {w:.3f}")
        if collapsed:
            print("  ⚠️  WARNING: Mixture has collapsed!")
    
    # Loss components
    if 'loss_components' in summary:
        comps = summary['loss_components']
        print(f"\nLoss Components:")
        print(f"  NLL:       {comps['nll']:.4f}")
        print(f"  Diversity: {comps['diversity']:.4f}")
        print(f"  Entropy:   {comps['entropy']:.4f}")
    
    # Gradient health
    grad_stats = monitor.check_gradient_health(model)
    print(f"\nGradient Health:")
    print(f"  Total norm: {grad_stats['total_norm']:.4f}")
    print(f"  Max grad:   {grad_stats['max_grad']:.4f}")
    print(f"  Min grad:   {grad_stats['min_grad']:.6f}")
    if grad_stats['is_exploding']:
        print("  ⚠️  WARNING: Gradients exploding!")
    if grad_stats['is_vanishing']:
        print("  ⚠️  WARNING: Gradients vanishing!")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("STMGT Monitoring Utilities")
    print("=" * 60)
    print("✅ Mixture weight monitoring")
    print("✅ Gradient health checks")
    print("✅ Data validation")
    print("✅ Model output validation")
    print("=" * 60)
