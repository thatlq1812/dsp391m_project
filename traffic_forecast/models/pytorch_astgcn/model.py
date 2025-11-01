"""PyTorch implementation of the ASTGCN architecture."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "SpatialAttention",
    "TemporalAttention",
    "ChebyshevConv",
    "STBlock",
    "ASTGCNComponent",
    "PyTorchASTGCN",
]


def _einsum(mat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Helper for multiplying Laplacian-like matrices with batched data."""

    return torch.einsum("ij,bjk->bik", mat, x)


class SpatialAttention(nn.Module):
    """Spatial attention mechanism."""

    def __init__(self, num_nodes: int, channels: int, timesteps: int) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.channels = channels
        self.timesteps = timesteps
        self.W1 = nn.Linear(channels * timesteps, num_nodes, bias=False)
        self.W2 = nn.Linear(channels * timesteps, num_nodes, bias=False)
        self.Vs = nn.Parameter(torch.randn(num_nodes, num_nodes) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, nodes, channels, timesteps = x.shape
        flat = x.reshape(batch, nodes, channels * timesteps)
        lhs = self.W1(flat)
        rhs = self.W2(flat)
        scores = torch.sigmoid(lhs.unsqueeze(2) + rhs.unsqueeze(1)).sum(dim=-1)
        scores = scores + self.Vs.unsqueeze(0).to(scores.device)
        return F.softmax(scores, dim=-1)


class TemporalAttention(nn.Module):
    """Temporal attention mechanism."""

    def __init__(self, num_nodes: int, channels: int, timesteps: int) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.channels = channels
        self.timesteps = timesteps
        self.W1 = nn.Linear(num_nodes * channels, timesteps, bias=False)
        self.W2 = nn.Linear(num_nodes * channels, timesteps, bias=False)
        self.Ve = nn.Parameter(torch.randn(timesteps, timesteps) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, nodes, channels, timesteps = x.shape
        flat = x.permute(0, 3, 1, 2).contiguous().view(batch, timesteps, nodes * channels)
        lhs = self.W1(flat)
        rhs = self.W2(flat)
        scores = torch.sigmoid(lhs.unsqueeze(2) + rhs.unsqueeze(1)).sum(dim=-1)
        scores = scores + self.Ve.unsqueeze(0).to(scores.device)
        return F.softmax(scores, dim=-1)


class ChebyshevConv(nn.Module):
    """Chebyshev graph convolution."""

    def __init__(self, in_channels: int, out_channels: int, order: int, laplacian: torch.Tensor) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.order = max(order, 1)
        self.register_buffer("laplacian", laplacian.float())
        self.coefficients = nn.Parameter(
            torch.randn(self.order, in_channels, out_channels) * (2.0 / math.sqrt(max(1, in_channels)))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, nodes, in_channels = x.shape
        laplacian = self.laplacian

        chebyshev_terms = []
        chebyshev_terms.append(x)
        if self.order > 1:
            chebyshev_terms.append(_einsum(laplacian, x))
        for k in range(2, self.order):
            t_k_minus_1 = chebyshev_terms[-1]
            t_k_minus_2 = chebyshev_terms[-2]
            t_k = 2 * _einsum(laplacian, t_k_minus_1) - t_k_minus_2
            chebyshev_terms.append(t_k)

        output = torch.zeros(batch, nodes, self.out_channels, device=x.device)
        for k, term in enumerate(chebyshev_terms):
            output = output + torch.einsum("bni,io->bno", term, self.coefficients[k])
        return output


class STBlock(nn.Module):
    """Spatial-temporal processing block comprised of attention + graph conv."""

    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        out_channels: int,
        order: int,
        laplacian: torch.Tensor,
        timesteps: int,
    ) -> None:
        super().__init__()
        self.spatial_attention = SpatialAttention(num_nodes, in_channels, timesteps)
        self.temporal_attention = TemporalAttention(num_nodes, in_channels, timesteps)
        self.graph_conv = ChebyshevConv(in_channels, out_channels, order, laplacian)
        self.temporal_conv = nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1))
        self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, nodes, channels, timesteps = x.shape
        spatial_scores = self.spatial_attention(x)
        temporal_scores = self.temporal_attention(x)

        # Apply temporal attention
        x_time = torch.einsum("bnct,bst->bncs", x, temporal_scores)

        # Graph convolution per timestep
        outputs = []
        for t in range(x_time.shape[3]):
            x_slice = x_time[:, :, :, t]
            x_masked = torch.einsum("bij,bjk->bik", spatial_scores, x_slice)
            outputs.append(self.graph_conv(x_masked).unsqueeze(-1))
        g_output = torch.cat(outputs, dim=3)

        temporal_input = g_output.permute(0, 2, 1, 3)
        temporal_output = F.relu(self.temporal_conv(temporal_input))
        residual = self.residual_conv(x.permute(0, 2, 1, 3))
        output = F.relu(temporal_output + residual)
        return output.permute(0, 2, 1, 3)



class ASTGCNComponent(nn.Module):
    """Single component that processes one temporal slice."""

    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        timesteps: int,
        order: int,
        spatial_channels: int,
        blocks_per_component: int,
        laplacian: torch.Tensor,
    ) -> None:
        super().__init__()
        blocks = []
        in_channels = num_features
        for _ in range(blocks_per_component):
            blocks.append(
                STBlock(
                    num_nodes=num_nodes,
                    in_channels=in_channels,
                    out_channels=spatial_channels,
                    order=order,
                    laplacian=laplacian,
                    timesteps=timesteps,
                )
            )
            in_channels = spatial_channels
        self.blocks = nn.ModuleList(blocks)
        self.projection = nn.Conv2d(spatial_channels, 1, kernel_size=(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = x
        for block in self.blocks:
            output = block(output)
        output = output.permute(0, 2, 1, 3)
        output = self.projection(output)
        output = output.squeeze(1).permute(0, 1, 2)
        return output


class PyTorchASTGCN(nn.Module):
    """Multi-output ASTGCN that predicts multiple features simultaneously."""

    def __init__(
        self,
        num_nodes: int,
        num_features: int,
        timesteps: int,
        horizon: int,
        order: int,
        spatial_channels: int,
        blocks_per_component: int,
        laplacian: torch.Tensor,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.timesteps = timesteps
        self.horizon = horizon
        self.components = nn.ModuleList(
            [
                ASTGCNComponent(
                    num_nodes=num_nodes,
                    num_features=num_features,
                    timesteps=timesteps,
                    order=order,
                    spatial_channels=spatial_channels,
                    blocks_per_component=blocks_per_component,
                    laplacian=laplacian,
                )
                for _ in range(num_features)
            ]
        )
        self.weight_recent = nn.Parameter(torch.ones(1, num_nodes, 1))
        self.weight_daily = nn.Parameter(torch.ones(1, num_nodes, 1))
        self.weight_weekly = nn.Parameter(torch.ones(1, num_nodes, 1))

    def forward(self, x_recent: torch.Tensor, x_daily: torch.Tensor, x_weekly: torch.Tensor) -> torch.Tensor:
        outputs = []
        for component in self.components:
            y_recent = component(x_recent)
            y_daily = component(x_daily)
            y_weekly = component(x_weekly)

            y_recent = _match_horizon(y_recent, self.horizon)
            y_daily = _match_horizon(y_daily, self.horizon)
            y_weekly = _match_horizon(y_weekly, self.horizon)

            y_combined = (
                self.weight_recent.to(y_recent.device) * y_recent
                + self.weight_daily.to(y_daily.device) * y_daily
                + self.weight_weekly.to(y_weekly.device) * y_weekly
            )
            outputs.append(y_combined.unsqueeze(2))
        return torch.cat(outputs, dim=2)


def _match_horizon(tensor: torch.Tensor, horizon: int) -> torch.Tensor:
    if tensor.shape[-1] >= horizon:
        return tensor[..., :horizon]
    padding = horizon - tensor.shape[-1]
    return F.pad(tensor, (0, padding))
