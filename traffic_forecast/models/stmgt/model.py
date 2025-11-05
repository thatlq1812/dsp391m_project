"""STMGT model definition and building blocks."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class TemporalEncoder(nn.Module):
    """Hierarchical temporal encoding with cyclical features and embeddings."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.dow_embedding = nn.Embedding(7, d_model // 2)
        self.weekend_embedding = nn.Embedding(2, d_model // 4)
        self.proj = nn.Linear(d_model // 2 + d_model // 4 + 2, d_model)

    def forward(self, temporal_features: dict[str, torch.Tensor]) -> torch.Tensor:
        hour = temporal_features["hour"]
        dow = temporal_features["dow"]
        is_weekend = temporal_features["is_weekend"]

        hour_rad = hour * (2 * np.pi / 24)
        hour_sin = torch.sin(hour_rad).unsqueeze(-1)
        hour_cos = torch.cos(hour_rad).unsqueeze(-1)

        dow_emb = self.dow_embedding(dow)
        weekend_emb = self.weekend_embedding(is_weekend)

        temporal_emb = torch.cat([hour_sin, hour_cos, dow_emb, weekend_emb], dim=-1)
        return self.proj(temporal_emb)


class WeatherCrossAttention(nn.Module):
    """Cross-attention where traffic embeddings attend to weather features."""

    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.2) -> None:
        super().__init__()
        self.weather_proj = nn.Linear(3, d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        traffic_features: torch.Tensor,
        weather_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weather_proj = self.weather_proj(weather_features)
        attn_output, attn_weights = self.cross_attn(
            query=traffic_features,
            key=weather_proj,
            value=weather_proj,
        )
        output = self.norm(traffic_features + self.dropout(attn_output))
        return output, attn_weights


class ParallelSTBlock(nn.Module):
    """Parallel spatial-temporal processing block with GAT and Transformer branches."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.2,
        drop_edge_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.drop_edge_rate = drop_edge_rate

        self.gat = GATv2Conv(hidden_dim, hidden_dim, heads=num_heads, concat=False, dropout=dropout)

        self.temporal_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.temporal_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.fusion_gate = nn.Linear(hidden_dim * 2, hidden_dim)

        self.ln_spatial = nn.LayerNorm(hidden_dim)
        self.ln_temporal1 = nn.LayerNorm(hidden_dim)
        self.ln_temporal2 = nn.LayerNorm(hidden_dim)
        self.ln_fusion = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, num_steps, _ = x.shape
        residual = x

        spatial_outputs: list[torch.Tensor] = []
        for step in range(num_steps):
            x_t = x[:, :, step, :]
            if self.training and self.drop_edge_rate > 0:
                mask = torch.rand(edge_index.size(1), device=edge_index.device) > self.drop_edge_rate
                edge_index_dropped = edge_index[:, mask]
            else:
                edge_index_dropped = edge_index

            x_t_flat = x_t.reshape(batch_size * num_nodes, -1)
            edge_index_batch = edge_index_dropped.unsqueeze(-1).repeat(1, 1, batch_size)
            offsets = torch.arange(batch_size, device=edge_index.device) * num_nodes
            edge_index_batch = edge_index_batch + offsets.view(1, 1, batch_size)
            edge_index_batch = edge_index_batch.reshape(2, -1)

            x_t_out = self.gat(x_t_flat, edge_index_batch).reshape(batch_size, num_nodes, -1)
            spatial_outputs.append(x_t_out)

        x_spatial = torch.stack(spatial_outputs, dim=2)
        x_spatial = self.ln_spatial(x_spatial)

        temporal_outputs: list[torch.Tensor] = []
        for node_idx in range(num_nodes):
            x_n = x[:, node_idx, :, :]
            attn_out, _ = self.temporal_attn(x_n, x_n, x_n)
            x_n = self.ln_temporal1(x_n + self.dropout(attn_out))
            ffn_out = self.temporal_ffn(x_n)
            x_n = self.ln_temporal2(x_n + self.dropout(ffn_out))
            temporal_outputs.append(x_n)

        x_temporal = torch.stack(temporal_outputs, dim=1)

        x_cat = torch.cat([x_spatial, x_temporal], dim=-1)
        alpha = torch.sigmoid(self.fusion_gate(x_cat))
        beta = 1 - alpha
        x_fused = alpha * x_spatial + beta * x_temporal + residual
        return self.ln_fusion(x_fused)


class GaussianMixtureHead(nn.Module):
    """Gaussian Mixture head that emits mean, std, and logits for each component."""

    def __init__(self, hidden_dim: int, num_components: int = 3, pred_len: int = 12) -> None:
        super().__init__()
        self.num_components = num_components
        self.pred_len = pred_len
        self.mu_head = nn.Linear(hidden_dim, pred_len * num_components)
        self.sigma_head = nn.Linear(hidden_dim, pred_len * num_components)
        self.pi_head = nn.Linear(hidden_dim, pred_len * num_components)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size, num_nodes, _ = x.shape
        mu = self.mu_head(x).reshape(batch_size, num_nodes, self.pred_len, self.num_components)
        log_sigma = self.sigma_head(x).reshape(batch_size, num_nodes, self.pred_len, self.num_components)
        logits_pi = self.pi_head(x).reshape(batch_size, num_nodes, self.pred_len, self.num_components)
        sigma = torch.exp(log_sigma).clamp(min=0.1, max=10.0)
        return {"means": mu, "stds": sigma, "logits": logits_pi}


class STMGT(nn.Module):
    """Spatial-Temporal Multi-Modal Graph Transformer."""

    def __init__(
        self,
        num_nodes: int = 62,
        in_dim: int = 1,
        hidden_dim: int = 64,
        num_blocks: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        drop_edge_rate: float = 0.2,
        mixture_components: int = 3,
        seq_len: int = 48,
        pred_len: int = 12,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.traffic_encoder = nn.Linear(in_dim, hidden_dim)
        self.weather_encoder = nn.Linear(3, hidden_dim)
        self.temporal_encoder = TemporalEncoder(hidden_dim)

        self.st_blocks = nn.ModuleList(
            [
                ParallelSTBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    drop_edge_rate=drop_edge_rate,
                )
                for _ in range(num_blocks)
            ]
        )

        self.weather_cross_attn = WeatherCrossAttention(hidden_dim, num_heads, dropout)
        self.output_head = GaussianMixtureHead(
            hidden_dim=hidden_dim,
            num_components=mixture_components,
            pred_len=pred_len,
        )

    def forward(
        self,
        x_traffic: torch.Tensor,
        edge_index: torch.Tensor,
        x_weather: torch.Tensor,
        temporal_features: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        traffic_emb = self.traffic_encoder(x_traffic)
        temporal_emb = self.temporal_encoder(temporal_features)
        x = traffic_emb + temporal_emb.unsqueeze(1)

        for block in self.st_blocks:
            x = block(x, edge_index)

        x = x.mean(dim=2)
        x, _ = self.weather_cross_attn(x, x_weather)
        return self.output_head(x)
