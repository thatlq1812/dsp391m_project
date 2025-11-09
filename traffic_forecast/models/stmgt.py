"""
STMGT: Spatial-Temporal Multi-Modal Graph Transformer

Novel architecture combining:
    1. Parallel Spatial-Temporal Processing (GAT || Transformer)
    2. Weather Cross-Attention
    3. Hierarchical Temporal Encoding
    4. Gaussian Mixture Output (K=3)

Based on research findings from STMGT_RESEARCH_CONSOLIDATED.md

Author: DSP391m Team
Date: October 31, 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
import numpy as np


class Normalizer(nn.Module):
    """
    Feature normalizer with learnable affine transformation
    
    Normalizes input to zero mean and unit variance, then applies
    learnable scale and shift parameters.
    """
    
    def __init__(self, mean, std, eps=1e-8):
        super().__init__()
        
        # Register as buffers (not trainable, but moved with model)
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32))
        self.eps = eps
    
    def forward(self, x):
        """
        Args:
            x: Input tensor [..., features]
        
        Returns:
            Normalized tensor
        """
        return (x - self.mean) / (self.std + self.eps)
    
    def denormalize(self, x):
        """Inverse transformation for predictions"""
        return x * (self.std + self.eps) + self.mean


class TemporalEncoder(nn.Module):
    """
    Hierarchical temporal encoding
    - Hour-of-day: sin/cos (24h cycle)
    - Day-of-week: embeddings (7 days)
    - Weekend: binary embedding
    """
    
    def __init__(self, d_model):
        super().__init__()
        
        # Embeddings
        self.dow_embedding = nn.Embedding(7, d_model // 2)
        self.weekend_embedding = nn.Embedding(2, d_model // 4)
        
        # Projection to d_model
        self.proj = nn.Linear(d_model // 2 + d_model // 4 + 2, d_model)
    
    def forward(self, temporal_features):
        """
        Args:
            temporal_features: dict with keys 'hour', 'dow', 'is_weekend'
                hour: [B, T] (0-23)
                dow: [B, T] (0-6)
                is_weekend: [B, T] (0 or 1)
        
        Returns:
            temporal_emb: [B, T, D]
        """
        hour = temporal_features['hour']
        dow = temporal_features['dow']
        is_weekend = temporal_features['is_weekend']
        
        # Sin/cos for hour (24h cycle)
        hour_rad = hour * (2 * np.pi / 24)
        hour_sin = torch.sin(hour_rad).unsqueeze(-1)  # [B, T, 1]
        hour_cos = torch.cos(hour_rad).unsqueeze(-1)  # [B, T, 1]
        
        # Embeddings
        dow_emb = self.dow_embedding(dow)  # [B, T, D//2]
        weekend_emb = self.weekend_embedding(is_weekend)  # [B, T, D//4]
        
        # Concatenate and project
        temporal_emb = torch.cat([hour_sin, hour_cos, dow_emb, weekend_emb], dim=-1)
        temporal_emb = self.proj(temporal_emb)  # [B, T, D]
        
        return temporal_emb


class WeatherCrossAttention(nn.Module):
    """
    Cross-attention: Traffic features attend to weather features
    Query: Traffic, Key/Value: Weather
    """
    
    def __init__(self, d_model, num_heads=4, dropout=0.2):
        super().__init__()
        
        self.weather_proj = nn.Linear(3, d_model)  # 3 weather features -> d_model
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, traffic_features, weather_features):
        """
        Args:
            traffic_features: [B, N, D] - traffic embeddings
            weather_features: [B, T, 3] - weather data (temp, wind, precip)
        
        Returns:
            output: [B, N, D]
            attn_weights: [B, N, T]
        """
        # Project weather to d_model
        weather_proj = self.weather_proj(weather_features)  # [B, T, D]
        
        # Cross-attention
        attn_output, attn_weights = self.cross_attn(
            query=traffic_features,   # [B, N, D]
            key=weather_proj,          # [B, T, D]
            value=weather_proj         # [B, T, D]
        )
        
        # Residual + norm
        output = self.norm(traffic_features + self.dropout(attn_output))
        
        return output, attn_weights


class ParallelSTBlock(nn.Module):
    """
    Parallel Spatial-Temporal Block
    
    Process spatial and temporal in parallel:
    - Spatial branch: GATv2 (graph attention)
    - Temporal branch: Transformer (self-attention)
    - Fusion: Gated fusion with residual
    """
    
    def __init__(self, hidden_dim, num_heads=4, dropout=0.2, drop_edge_rate=0.2):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.drop_edge_rate = drop_edge_rate
        
        # Spatial branch (GAT)
        self.gat = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout)
        
        # Temporal branch (Transformer)
        self.temporal_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.temporal_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Gated fusion
        self.fusion_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Layer norms
        self.ln_spatial = nn.LayerNorm(hidden_dim)
        self.ln_temporal1 = nn.LayerNorm(hidden_dim)
        self.ln_temporal2 = nn.LayerNorm(hidden_dim)
        self.ln_fusion = nn.LayerNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, edge_index):
        """
        Args:
            x: [B, N, T, D] - node features over time
            edge_index: [2, E] - edge connectivity
        
        Returns:
            output: [B, N, T, D]
        """
        B, N, T, D = x.shape
        residual = x
        
        # Spatial branch: process each timestep
        x_spatial = []
        for t in range(T):
            x_t = x[:, :, t, :]  # [B, N, D]
            
            # DropEdge during training
            if self.training and self.drop_edge_rate > 0:
                mask = torch.rand(edge_index.size(1), device=edge_index.device) > self.drop_edge_rate
                edge_index_dropped = edge_index[:, mask]
            else:
                edge_index_dropped = edge_index
            
            # Flatten batch and nodes for GATv2Conv
            x_t_flat = x_t.reshape(B * N, D)
            
            # Adjust edge_index for batched graphs
            edge_index_batch = edge_index_dropped.unsqueeze(-1).repeat(1, 1, B)  # [2, E, B]
            offsets = torch.arange(B, device=edge_index.device) * N
            edge_index_batch = edge_index_batch + offsets.view(1, 1, B)  # [2, E, B]
            edge_index_batch = edge_index_batch.reshape(2, -1)  # [2, E*B]
            
            # Apply GAT
            x_t_out = self.gat(x_t_flat, edge_index_batch)  # [B*N, D]
            x_t_out = x_t_out.reshape(B, N, D)
            x_spatial.append(x_t_out)
        
        x_spatial = torch.stack(x_spatial, dim=2)  # [B, N, T, D]
        x_spatial = self.ln_spatial(x_spatial)
        
        # Temporal branch: process each node
        x_temporal = []
        for n in range(N):
            x_n = x[:, n, :, :]  # [B, T, D]
            
            # Self-attention over time
            attn_out, _ = self.temporal_attn(x_n, x_n, x_n)
            x_n = self.ln_temporal1(x_n + self.dropout(attn_out))
            
            # FFN
            ffn_out = self.temporal_ffn(x_n)
            x_n = self.ln_temporal2(x_n + self.dropout(ffn_out))
            
            x_temporal.append(x_n)
        
        x_temporal = torch.stack(x_temporal, dim=1)  # [B, N, T, D]
        
        # Gated fusion
        x_cat = torch.cat([x_spatial, x_temporal], dim=-1)  # [B, N, T, 2D]
        alpha = torch.sigmoid(self.fusion_gate(x_cat))  # [B, N, T, D]
        beta = 1 - alpha
        
        # Fuse with residual
        x_fused = alpha * x_spatial + beta * x_temporal + residual
        x_fused = self.ln_fusion(x_fused)
        
        return x_fused


class GaussianMixtureHead(nn.Module):
    """
    Gaussian Mixture output layer (K=3 components)
    Outputs: mu, sigma, pi for each component
    """
    
    def __init__(self, hidden_dim, num_components=3, pred_len=12):
        super().__init__()
        
        self.K = num_components
        self.pred_len = pred_len
        
        # Prediction layers
        self.mu_head = nn.Linear(hidden_dim, pred_len * num_components)
        self.sigma_head = nn.Linear(hidden_dim, pred_len * num_components)
        self.pi_head = nn.Linear(hidden_dim, pred_len * num_components)
    
    def forward(self, x):
        """
        Args:
            x: [B, N, D]
        
        Returns:
            dict with keys:
                'means': [B, N, T_pred, K]
                'stds': [B, N, T_pred, K]
                'logits': [B, N, T_pred, K]
        """
        B, N, D = x.shape
        
        # Predict parameters
        mu = self.mu_head(x).reshape(B, N, self.pred_len, self.K)
        log_sigma = self.sigma_head(x).reshape(B, N, self.pred_len, self.K)
        logits_pi = self.pi_head(x).reshape(B, N, self.pred_len, self.K)
        
        # Apply constraints
        sigma = torch.exp(log_sigma).clamp(min=0.1, max=10.0)  # Variance bounds
        
        return {
            'means': mu,
            'stds': sigma,
            'logits': logits_pi
        }


class STMGT(nn.Module):
    """
    Spatial-Temporal Multi-Modal Graph Transformer
    
    Novel architecture with:
    1. Parallel ST processing
    2. Weather cross-attention
    3. Hierarchical temporal encoding
    4. Gaussian mixture output (K=3)
    """
    
    def __init__(
        self,
        num_nodes=62,
        in_dim=1,  # speed only
        hidden_dim=64,
        num_blocks=3,
        num_heads=4,
        dropout=0.2,
        drop_edge_rate=0.2,
        mixture_components=3,
        seq_len=48,  # 12h @ 15min
        pred_len=12,  # 3h @ 15min
        speed_mean=18.72,
        speed_std=7.03,
        weather_mean=None,
        weather_std=None,
    ):
        super().__init__()
        
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Default weather statistics
        if weather_mean is None:
            weather_mean = [25.0, 15.0, 1.0]  # temp, wind, precip
        if weather_std is None:
            weather_std = [5.0, 10.0, 2.0]
        
        # Normalizers
        self.speed_normalizer = Normalizer(mean=speed_mean, std=speed_std)
        self.weather_normalizer = Normalizer(mean=weather_mean, std=weather_std)
        
        # Input encoders
        self.traffic_encoder = nn.Linear(in_dim, hidden_dim)
        self.weather_encoder = nn.Linear(3, hidden_dim)  # temp, wind, precip
        self.temporal_encoder = TemporalEncoder(hidden_dim)
        
        # Parallel ST blocks
        self.st_blocks = nn.ModuleList([
            ParallelSTBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                drop_edge_rate=drop_edge_rate
            )
            for _ in range(num_blocks)
        ])
        
        # Weather cross-attention
        self.weather_cross_attn = WeatherCrossAttention(hidden_dim, num_heads, dropout)
        
        # Output head
        self.output_head = GaussianMixtureHead(
            hidden_dim=hidden_dim,
            num_components=mixture_components,
            pred_len=pred_len
        )
    
    def forward(self, x_traffic, edge_index, x_weather, temporal_features):
        """
        Args:
            x_traffic: [B, N, T, 1] - traffic speed history (unnormalized)
            edge_index: [2, E] - graph edges
            x_weather: [B, T_pred, 3] - weather forecast (unnormalized)
            temporal_features: dict with hour, dow, is_weekend [B, T]
        
        Returns:
            dict with 'means', 'stds', 'logits' for mixture components (normalized)
        """
        B, N, T, _ = x_traffic.shape
        
        # Normalize inputs
        x_traffic = self.speed_normalizer(x_traffic)  # [B, N, T, 1]
        x_weather = self.weather_normalizer(x_weather)  # [B, T_pred, 3]
        
        # Encode traffic
        traffic_emb = self.traffic_encoder(x_traffic)  # [B, N, T, D]
        
        # Encode temporal
        temporal_emb = self.temporal_encoder(temporal_features)  # [B, T, D]
        
        # Add temporal to traffic
        x = traffic_emb + temporal_emb.unsqueeze(1)  # [B, N, T, D]
        
        # Parallel ST blocks
        for block in self.st_blocks:
            x = block(x, edge_index)
        
        # Aggregate over time
        x = x.mean(dim=2)  # [B, N, D]
        
        # Weather cross-attention
        x, attn_weights = self.weather_cross_attn(x, x_weather)
        
        # Gaussian mixture output
        pred_params = self.output_head(x)
        
        return pred_params
    
    def denormalize_predictions(self, pred_params):
        """
        Denormalize prediction parameters back to original scale
        
        Args:
            pred_params: dict with 'means', 'stds', 'logits' (normalized)
        
        Returns:
            dict with denormalized parameters
        """
        means_denorm = self.speed_normalizer.denormalize(pred_params['means'])
        stds_denorm = pred_params['stds'] * self.speed_normalizer.std
        
        return {
            'means': means_denorm,
            'stds': stds_denorm,
            'logits': pred_params['logits']  # unchanged
        }
    
    def predict(self, x_traffic, edge_index, x_weather, temporal_features):
        """
        Prediction with denormalization
        
        Returns predictions in original speed scale (km/h)
        """
        pred_params = self.forward(x_traffic, edge_index, x_weather, temporal_features)
        return self.denormalize_predictions(pred_params)


def mixture_nll_loss(y_pred_params, y_true, return_components=False):
    """
    Mixture Negative Log-Likelihood loss with stability tricks
    
    Args:
        y_pred_params: dict with keys 'means', 'stds', 'logits'
            means: [B, N, T_pred, K] (normalized)
            stds: [B, N, T_pred, K] (normalized)
            logits: [B, N, T_pred, K]
        y_true: [B, N, T_pred] (normalized)
        return_components: If True, return dict with loss components
    
    Returns:
        loss: scalar or dict if return_components=True
    """
    mu = y_pred_params['means']
    sigma = y_pred_params['stds']
    logits = y_pred_params['logits']
    
    # Compute mixture weights
    pi = torch.softmax(logits, dim=-1)
    
    # Expand y_true
    y_true = y_true.unsqueeze(-1)  # [B, N, T_pred, 1]
    
    # Log probability for each component (with numerical stability)
    log_prob = -0.5 * ((y_true - mu) / (sigma + 1e-8)) ** 2
    log_prob = log_prob - torch.log(sigma + 1e-8) - 0.5 * np.log(2 * np.pi)
    
    # Weight by mixture probabilities
    weighted_log_prob = log_prob + torch.log(pi + 1e-8)
    
    # Log-sum-exp for numerical stability
    nll = -torch.logsumexp(weighted_log_prob, dim=-1)  # [B, N, T_pred]
    
    # Component diversity regularization (prevent mode collapse)
    # Encourage components to have different means
    diversity_loss = -torch.std(mu, dim=-1).mean()
    
    # Entropy regularization (prevent one component dominating)
    entropy = -(pi * torch.log(pi + 1e-8)).sum(dim=-1).mean()
    entropy_reg = -entropy
    
    # Total loss with regularization
    total_loss = nll.mean() + 0.01 * diversity_loss + 0.001 * entropy_reg
    
    if return_components:
        return {
            'total': total_loss,
            'nll': nll.mean(),
            'diversity': diversity_loss,
            'entropy': entropy_reg,
            'mixture_weights': pi.mean(dim=(0, 1, 2))  # [K]
        }
    
    return total_loss


if __name__ == "__main__":
    # Test model with different scales
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Test with current scale (62 nodes)
    print("\n" + "="*60)
    print("Testing STMGT Scalability")
    print("="*60)
    
    for num_nodes in [62, 100, 200]:
        print(f"\n--- Testing with {num_nodes} nodes ---")
        
        model = STMGT(
            num_nodes=num_nodes,
            in_dim=1,
            hidden_dim=64,
            num_blocks=3,
            num_heads=4,
            dropout=0.2,
            drop_edge_rate=0.2,
            mixture_components=3,
            seq_len=48,
            pred_len=12
        ).to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Test forward pass
        B, N, T, T_pred = 4, num_nodes, 48, 12
        E = int(num_nodes * 2.3)  # Approximate edges based on graph density
        
        x_traffic = torch.randn(B, N, T, 1).to(device)
        x_weather = torch.randn(B, T_pred, 3).to(device)
        edge_index = torch.randint(0, N, (2, E)).to(device)
        temporal = {
            'hour': torch.randint(0, 24, (B, T)).to(device),
            'dow': torch.randint(0, 7, (B, T)).to(device),
            'is_weekend': torch.randint(0, 2, (B, T)).to(device)
        }
        
        mu, sigma, pi = model(x_traffic, x_weather, edge_index, temporal)
        
        print(f"✅ Forward pass successful!")
        print(f"   Output shapes: mu={mu.shape}, sigma={sigma.shape}, pi={pi.shape}")
    
    print("\n" + "="*60)
    print("STMGT is fully scalable! ✨")
    print("="*60)
    
    # Test loss
    y_true = torch.randn(B, N, T_pred).to(device)
    loss = mixture_nll_loss((mu, sigma, pi), y_true)
    
    print(f"\nLoss: {loss.item():.4f}")
    print("\nModel test successful!")
