"""
ASTGCN Implementation in PyTorch
Ported from research notebook (temp/astgcn-data-merge-1.ipynb)

Author: Team collaboration (Research → Production)
Date: 2025-10-31
"""

import math
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances


def compute_scaled_laplacian(adjacency: np.ndarray) -> np.ndarray:
    """
    Compute scaled Laplacian for Chebyshev polynomials.
    
    L_tilde = (2/lambda_max) * L - I
    where L = I - D^(-1/2) * A * D^(-1/2)
    
    Args:
        adjacency: Adjacency matrix (N, N)
        
    Returns:
        Scaled Laplacian matrix (N, N)
    """
    N = adjacency.shape[0]
    A = adjacency.astype(np.float32)
    
    # Degree matrix
    d = A.sum(axis=1)
    
    # Avoid divide by zero
    d_sqrt_inv = np.diag(1.0 / np.sqrt(np.where(d == 0, 1.0, d)))
    
    # Normalized Laplacian
    I = np.eye(N, dtype=np.float32)
    L = I - d_sqrt_inv @ A @ d_sqrt_inv
    
    # Compute lambda_max
    try:
        eigs = np.linalg.eigvals(L)
        lambda_max = np.max(eigs).real
        if lambda_max == 0:
            lambda_max = 2.0
    except Exception:
        lambda_max = 2.0
    
    # Scale
    L_tilde = (2.0 / lambda_max) * L - I
    
    return L_tilde.astype(np.float32)


def build_adjacency_from_edges(edges: List[Tuple[str, str]], nodes: List[str]) -> np.ndarray:
    """
    Build adjacency matrix from edge list.
    
    Args:
        edges: List of (node_u, node_v) tuples
        nodes: List of node IDs
        
    Returns:
        Symmetric adjacency matrix (N, N)
    """
    N = len(nodes)
    node_idx = {n: i for i, n in enumerate(nodes)}
    A = np.zeros((N, N), dtype=np.float32)
    
    for u, v in edges:
        if u in node_idx and v in node_idx:
            i, j = node_idx[u], node_idx[v]
            A[i, j] = 1.0
            A[j, i] = 1.0  # Symmetric
    
    return A


def build_adjacency_from_coords(coords: np.ndarray, k_nearest: int = 5) -> np.ndarray:
    """
    Build adjacency from coordinates using k-nearest neighbors.
    
    Args:
        coords: Node coordinates (N, 2) - (lat, lon)
        k_nearest: Number of nearest neighbors
        
    Returns:
        Adjacency matrix (N, N)
    """
    N = coords.shape[0]
    
    # Compute pairwise distances
    distances = pairwise_distances(coords)
    
    # Connect k nearest neighbors
    A = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        # Get k nearest (excluding self)
        nearest = np.argsort(distances[i])[1:k_nearest+1]
        A[i, nearest] = 1.0
        A[nearest, i] = 1.0  # Symmetric
    
    return A


class SpatialAttention(nn.Module):
    """Spatial attention mechanism for graph nodes."""
    
    def __init__(self, num_nodes: int, in_channels: int, time_steps: int):
        """
        Args:
            num_nodes: Number of graph nodes
            in_channels: Number of input channels/features
            time_steps: Number of time steps
        """
        super().__init__()
        self.num_nodes = num_nodes
        
        # Linear transformations
        self.W1 = nn.Linear(in_channels * time_steps, num_nodes, bias=False)
        self.W2 = nn.Linear(in_channels * time_steps, num_nodes, bias=False)
        
        # Learnable spatial matrix
        self.Vs = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.1)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute spatial attention weights.
        
        Args:
            X: Input tensor (batch, num_nodes, channels, time_steps)
            
        Returns:
            Spatial attention matrix (batch, num_nodes, num_nodes)
        """
        B, N, C, T = X.shape
        
        # Flatten channels and time
        x_flat = X.reshape(B, N, C * T)  # (B, N, C*T)
        
        # Compute attention scores
        lhs = self.W1(x_flat)  # (B, N, N)
        rhs = self.W2(x_flat)  # (B, N, N)
        
        # S = sigmoid(W1(X) + W2(X)^T + Vs)
        S = torch.sigmoid(lhs.unsqueeze(2) + rhs.unsqueeze(1))
        S = S.sum(dim=-1)  # (B, N, N)
        
        # Add learnable matrix
        S = S + self.Vs.unsqueeze(0)
        
        # Normalize
        S = F.softmax(S, dim=-1)
        
        return S


class TemporalAttention(nn.Module):
    """Temporal attention mechanism for time steps."""
    
    def __init__(self, num_nodes: int, in_channels: int, time_steps: int):
        """
        Args:
            num_nodes: Number of graph nodes
            in_channels: Number of input channels/features
            time_steps: Number of time steps
        """
        super().__init__()
        self.time_steps = time_steps
        
        # Linear transformations
        self.W1 = nn.Linear(num_nodes * in_channels, time_steps, bias=False)
        self.W2 = nn.Linear(num_nodes * in_channels, time_steps, bias=False)
        
        # Learnable temporal matrix
        self.Ve = nn.Parameter(torch.randn(time_steps, time_steps) * 0.1)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal attention weights.
        
        Args:
            X: Input tensor (batch, num_nodes, channels, time_steps)
            
        Returns:
            Temporal attention matrix (batch, time_steps, time_steps)
        """
        B, N, C, T = X.shape
        
        # Permute to (B, T, N, C)
        X_t = X.permute(0, 3, 1, 2).contiguous()
        
        # Flatten nodes and channels
        x_flat = X_t.reshape(B, T, N * C)  # (B, T, N*C)
        
        # Compute attention scores
        lhs = self.W1(x_flat)  # (B, T, T)
        rhs = self.W2(x_flat)  # (B, T, T)
        
        # E = sigmoid(W1(X) + W2(X)^T + Ve)
        E = torch.sigmoid(lhs.unsqueeze(2) + rhs.unsqueeze(1))
        E = E.sum(dim=-1)  # (B, T, T)
        
        # Add learnable matrix
        E = E + self.Ve.unsqueeze(0)
        
        # Normalize
        E = F.softmax(E, dim=-1)
        
        return E


class ChebConv(nn.Module):
    """Chebyshev graph convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int, K: int, L_tilde: np.ndarray):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            K: Order of Chebyshev polynomials
            L_tilde: Scaled Laplacian matrix
        """
        super().__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Register Laplacian as buffer (not trainable)
        self.register_buffer('Ltilde', torch.from_numpy(L_tilde.astype(np.float32)))
        
        # Learnable weights for each polynomial order
        self.Theta = nn.Parameter(
            torch.randn(K, in_channels, out_channels) * (2.0 / math.sqrt(max(1, in_channels)))
        )
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply Chebyshev graph convolution.
        
        Args:
            X: Input tensor (batch, num_nodes, in_channels)
            
        Returns:
            Output tensor (batch, num_nodes, out_channels)
        """
        B, N, C_in = X.shape
        L = self.Ltilde
        
        # Compute Chebyshev polynomials
        T_k = []
        
        # T0 = X
        T0 = X
        T_k.append(T0)
        
        if self.K > 1:
            # T1 = L * X
            T1 = torch.einsum('ij,bjk->bik', L, X)
            T_k.append(T1)
            
            # T_k = 2*L*T_(k-1) - T_(k-2)
            for k in range(2, self.K):
                T_new = 2 * torch.einsum('ij,bjk->bik', L, T_k[-1]) - T_k[-2]
                T_k.append(T_new)
        
        # Aggregate: sum_k Theta_k * T_k
        output = torch.zeros(B, N, self.out_channels, device=X.device)
        for k in range(self.K):
            # T_k: (B, N, C_in), Theta[k]: (C_in, C_out)
            output += torch.einsum('bni,io->bno', T_k[k], self.Theta[k])
        
        return output


class SpatialTemporalBlock(nn.Module):
    """Single spatial-temporal block with attention mechanisms."""
    
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        out_channels: int,
        time_steps: int,
        K_cheb: int,
        L_tilde: np.ndarray
    ):
        """
        Args:
            num_nodes: Number of graph nodes
            in_channels: Number of input channels
            out_channels: Number of output channels
            time_steps: Number of time steps
            K_cheb: Order of Chebyshev polynomials
            L_tilde: Scaled Laplacian matrix
        """
        super().__init__()
        
        # Attention mechanisms
        self.spatial_attention = SpatialAttention(num_nodes, in_channels, time_steps)
        self.temporal_attention = TemporalAttention(num_nodes, in_channels, time_steps)
        
        # Graph convolution
        self.cheb_conv = ChebConv(in_channels, out_channels, K_cheb, L_tilde)
        
        # Temporal convolution (1D conv over time)
        self.temporal_conv = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(1, 3),
            padding=(0, 1)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm([out_channels])
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: Input tensor (batch, num_nodes, in_channels, time_steps)
            
        Returns:
            Output tensor (batch, num_nodes, out_channels, time_steps)
        """
        B, N, C, T = X.shape
        
        # 1. Compute attention weights
        S_att = self.spatial_attention(X)  # (B, N, N)
        T_att = self.temporal_attention(X)  # (B, T, T)
        
        # 2. Apply spatial attention to each time step
        X_spatial = []
        for t in range(T):
            x_t = X[:, :, :, t]  # (B, N, C)
            # Apply graph convolution with spatial attention
            x_t = self.cheb_conv(x_t)  # (B, N, out_channels)
            # Apply spatial attention
            x_t = torch.einsum('bnm,bmc->bnc', S_att, x_t)
            X_spatial.append(x_t)
        
        X_spatial = torch.stack(X_spatial, dim=3)  # (B, N, out_channels, T)
        
        # 3. Apply temporal attention
        X_temporal = torch.einsum('btt,bnct->bnct', T_att, X_spatial)
        
        # 4. Temporal convolution
        # Reshape for Conv2d: (B, C, N, T)
        X_temp = X_temporal.permute(0, 2, 1, 3)  # (B, out_channels, N, T)
        X_temp = self.temporal_conv(X_temp)  # (B, out_channels, N, T)
        X_temp = X_temp.permute(0, 2, 1, 3)  # (B, N, out_channels, T)
        
        # 5. Residual connection + normalization
        if C == X_temp.shape[2]:
            X_temp = X_temp + X
        
        # Layer norm over channels
        X_out = self.layer_norm(X_temp.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        
        return F.relu(X_out)


class ASTGCN(nn.Module):
    """
    Attention-based Spatial-Temporal Graph Convolutional Network.
    
    Multi-component architecture:
    - Recent component: Short-term patterns
    - Daily component: Daily patterns (optional)
    - Weekly component: Weekly patterns (optional)
    """
    
    def __init__(
        self,
        num_nodes: int,
        in_features: int,
        time_steps_in: int,
        time_steps_out: int,
        K_cheb: int,
        L_tilde: np.ndarray,
        hidden_channels: int = 64,
        num_blocks: int = 2
    ):
        """
        Args:
            num_nodes: Number of graph nodes
            in_features: Number of input features
            time_steps_in: Input sequence length
            time_steps_out: Output sequence length
            K_cheb: Order of Chebyshev polynomials
            L_tilde: Scaled Laplacian matrix
            hidden_channels: Hidden layer channels
            num_blocks: Number of ST blocks per component
        """
        super().__init__()
        
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.time_steps_in = time_steps_in
        self.time_steps_out = time_steps_out
        
        # Recent component (main temporal component)
        self.recent_blocks = nn.ModuleList([
            SpatialTemporalBlock(
                num_nodes=num_nodes,
                in_channels=in_features if i == 0 else hidden_channels,
                out_channels=hidden_channels,
                time_steps=time_steps_in,
                K_cheb=K_cheb,
                L_tilde=L_tilde
            )
            for i in range(num_blocks)
        ])
        
        # Output projection
        # Transform (B, N, hidden, T_in) → (B, N, features, T_out)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels * time_steps_in, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, in_features * time_steps_out)
        )
        
    def forward(
        self,
        X_recent: torch.Tensor,
        X_daily: Optional[torch.Tensor] = None,
        X_weekly: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            X_recent: Recent history (batch, num_nodes, features, time_steps_in)
            X_daily: Daily history (optional, same shape)
            X_weekly: Weekly history (optional, same shape)
            
        Returns:
            Predictions (batch, num_nodes, features, time_steps_out)
        """
        B, N, F, T = X_recent.shape
        
        # Process recent component
        X = X_recent
        for block in self.recent_blocks:
            X = block(X)  # (B, N, hidden, T)
        
        # Flatten temporal dimension
        X_flat = X.reshape(B, N, -1)  # (B, N, hidden*T)
        
        # Project to output
        out = self.output_proj(X_flat)  # (B, N, F*T_out)
        
        # Reshape to output format
        out = out.reshape(B, N, self.in_features, self.time_steps_out)
        
        return out


def create_astgcn_model(
    num_nodes: int,
    in_features: int,
    time_steps_in: int,
    time_steps_out: int,
    adjacency: np.ndarray,
    K_cheb: int = 3,
    hidden_channels: int = 64,
    num_blocks: int = 2,
    device: str = 'cpu'
) -> ASTGCN:
    """
    Factory function to create ASTGCN model.
    
    Args:
        num_nodes: Number of graph nodes
        in_features: Number of input features
        time_steps_in: Input sequence length
        time_steps_out: Output sequence length  
        adjacency: Adjacency matrix (N, N)
        K_cheb: Chebyshev polynomial order
        hidden_channels: Hidden layer size
        num_blocks: Number of ST blocks
        device: Device to place model on
        
    Returns:
        ASTGCN model instance
    """
    # Compute scaled Laplacian
    L_tilde = compute_scaled_laplacian(adjacency)
    
    # Create model
    model = ASTGCN(
        num_nodes=num_nodes,
        in_features=in_features,
        time_steps_in=time_steps_in,
        time_steps_out=time_steps_out,
        K_cheb=K_cheb,
        L_tilde=L_tilde,
        hidden_channels=hidden_channels,
        num_blocks=num_blocks
    )
    
    return model.to(device)
