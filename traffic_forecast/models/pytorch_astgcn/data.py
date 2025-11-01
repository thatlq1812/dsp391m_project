"""Data preparation helpers for the PyTorch ASTGCN model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "default_feature_columns",
    "load_processed_dataset",
    "create_dataloaders",
    "ASTGCNTrafficDataset",
    "DatasetMetadata",
]

# ---------------------------------------------------------------------------
# Constants and dataclasses
# ---------------------------------------------------------------------------

default_feature_columns: List[str] = [
    "speed_kmh",
    "temperature_c",
    "wind_speed_kmh",
    "precipitation_mm",
]


@dataclass
class DatasetMetadata:
    """Metadata returned together with dataloaders."""

    nodes: List[str]
    adjacency: np.ndarray
    laplacian: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    coords: Dict[str, Tuple[float, float]]


class ASTGCNTrafficDataset(Dataset):
    """Torch dataset wrapper for ASTGCN tensors."""

    def __init__(self, Xh: np.ndarray, Xd: np.ndarray, Xw: np.ndarray, y: np.ndarray):
        self.Xh = torch.from_numpy(Xh).float()
        self.Xd = torch.from_numpy(Xd).float()
        self.Xw = torch.from_numpy(Xw).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return self.Xh.shape[0]

    def __getitem__(self, index: int):
        return self.Xh[index], self.Xd[index], self.Xw[index], self.y[index]


# ---------------------------------------------------------------------------
# Data preparation pipeline
# ---------------------------------------------------------------------------

def _validate_features(df: pd.DataFrame, features: Iterable[str]) -> List[str]:
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required feature columns in dataset: " + ", ".join(missing)
        )
    return list(features)


def _expand_to_node_table(df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Convert edge-based observations into node-centric table."""

    records: List[Dict[str, float]] = []

    for _, row in df.iterrows():
        timestamp = row["timestamp"]
        for prefix in ("a", "b"):
            node_col = f"node_{prefix}_id"
            if node_col not in row or pd.isna(row[node_col]):
                continue

            lat = row.get(f"lat_{prefix}", np.nan)
            lon = row.get(f"lon_{prefix}", np.nan)

            record = {
                "node_id": row[node_col],
                "timestamp": timestamp,
                "lat": float(lat) if not pd.isna(lat) else 0.0,
                "lon": float(lon) if not pd.isna(lon) else 0.0,
            }

            for feat in features:
                value = row.get(feat, 0.0)
                record[feat] = float(value) if not pd.isna(value) else 0.0

            records.append(record)

    node_df = pd.DataFrame.from_records(records)
    if node_df.empty:
        raise RuntimeError("No node-centric records could be derived from dataset")

    node_df = node_df.sort_values(["timestamp", "node_id"]).reset_index(drop=True)

    # Edge list for adjacency
    if {"node_a_id", "node_b_id"}.issubset(df.columns):
        edges = df[["node_a_id", "node_b_id"]].dropna().drop_duplicates()
        edges = edges.rename(columns={"node_a_id": "u", "node_b_id": "v"})
        edge_df = edges.reset_index(drop=True)
    else:
        edge_df = pd.DataFrame(columns=["u", "v"])

    return node_df, edge_df


def _build_adjacency(edge_df: pd.DataFrame, nodes: List[str], coords: Dict[str, Tuple[float, float]]) -> np.ndarray:
    """Build symmetric adjacency matrix."""

    node_index = {node: idx for idx, node in enumerate(nodes)}
    adjacency = np.zeros((len(nodes), len(nodes)), dtype=float)

    if not edge_df.empty:
        for _, row in edge_df.iterrows():
            u = row["u"]
            v = row["v"]
            if u in node_index and v in node_index:
                i, j = node_index[u], node_index[v]
                adjacency[i, j] = 1.0
                adjacency[j, i] = 1.0
    else:
        # Fallback: connect nearest neighbours using coordinates
        coord_array = np.array([coords[node] for node in nodes])
        if np.all(coord_array == 0.0):
            return np.eye(len(nodes), dtype=float)
        distance_matrix = pairwise_distances(coord_array)
        threshold = np.percentile(distance_matrix, 10)
        adjacency = (distance_matrix <= threshold).astype(float)
        np.fill_diagonal(adjacency, 0.0)

    return adjacency


def _compute_scaled_laplacian(adjacency: np.ndarray) -> np.ndarray:
    degree = adjacency.sum(axis=1)
    with np.errstate(divide="ignore"):
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
    d_mat_inv = np.diag(degree_inv_sqrt)
    identity = np.eye(adjacency.shape[0], dtype=float)
    laplacian = identity - d_mat_inv @ adjacency @ d_mat_inv

    eigenvalues = np.linalg.eigvals(laplacian)
    lambda_max = np.max(eigenvalues.real)
    if lambda_max <= 0:
        lambda_max = 2.0
    scaled = (2.0 / lambda_max) * laplacian - identity
    return scaled.astype(np.float32)


def _build_tensors(
    node_df: pd.DataFrame,
    features: List[str],
    input_window: int,
    forecast_horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Dict[str, Tuple[float, float]], np.ndarray, np.ndarray]:
    timestamps = sorted(node_df["timestamp"].unique())
    nodes = sorted(node_df["node_id"].unique())
    node_index = {node: idx for idx, node in enumerate(nodes)}
    time_index = {ts: idx for idx, ts in enumerate(timestamps)}

    tensor = np.zeros((len(timestamps), len(nodes), len(features)), dtype=float)
    coords: Dict[str, Tuple[float, float]] = {node: (0.0, 0.0) for node in nodes}

    for _, row in node_df.iterrows():
        ti = time_index[row["timestamp"]]
        ni = node_index[row["node_id"]]
        for fi, feature in enumerate(features):
            value = row.get(feature, 0.0)
            tensor[ti, ni, fi] = float(0.0 if pd.isna(value) else value)
        coords[row["node_id"]] = (
            float(row.get("lat", 0.0) or 0.0),
            float(row.get("lon", 0.0) or 0.0),
        )

    mean = tensor.mean(axis=0, keepdims=True)
    std = tensor.std(axis=0, keepdims=True) + 1e-6
    tensor = (tensor - mean) / std

    samples_Xh: List[np.ndarray] = []
    samples_Xd: List[np.ndarray] = []
    samples_Xw: List[np.ndarray] = []
    samples_y: List[np.ndarray] = []

    total_steps = tensor.shape[0]
    if total_steps < input_window + forecast_horizon:
        raise RuntimeError("Not enough timestamps for requested window and horizon")

    for end in range(input_window, total_steps - forecast_horizon + 1):
        start = end - input_window
        X_recent = tensor[start:end]
        Y_target = tensor[end:end + forecast_horizon]
        Xr = np.transpose(X_recent, (1, 2, 0)).astype(np.float32)
        Xd = Xr.copy()
        Xw = Xr.copy()
        Y = np.transpose(Y_target, (1, 2, 0)).astype(np.float32)
        samples_Xh.append(Xr)
        samples_Xd.append(Xd)
        samples_Xw.append(Xw)
        samples_y.append(Y)

    Xh = np.stack(samples_Xh, axis=0)
    Xd = np.stack(samples_Xd, axis=0)
    Xw = np.stack(samples_Xw, axis=0)
    y = np.stack(samples_y, axis=0)

    return Xh, Xd, Xw, y, nodes, coords, mean.squeeze(0), std.squeeze(0)


def load_processed_dataset(
    parquet_path: Path,
    features: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed parquet data and convert to node-centric table."""

    if not parquet_path.exists():
        raise FileNotFoundError(f"Dataset not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if "timestamp" not in df.columns:
        raise RuntimeError("Dataset must include a 'timestamp' column")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    feature_cols = _validate_features(df, features or default_feature_columns)
    node_df, edge_df = _expand_to_node_table(df, feature_cols)
    return node_df, edge_df


def create_dataloaders(
    parquet_path: Path,
    features: Optional[Iterable[str]],
    input_window: int,
    forecast_horizon: int,
    batch_size: int,
    random_seed: int = 42,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader, DatasetMetadata]:
    """Create PyTorch dataloaders for training/validation/testing."""

    node_df, edge_df = load_processed_dataset(parquet_path, features)
    feature_cols = _validate_features(node_df, features or default_feature_columns)

    Xh, Xd, Xw, y, nodes, coords, mean, std = _build_tensors(
        node_df,
        feature_cols,
        input_window,
        forecast_horizon,
    )

    adjacency = _build_adjacency(edge_df, nodes, coords)
    laplacian = _compute_scaled_laplacian(adjacency)

    indices = np.arange(Xh.shape[0])
    train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=random_seed)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_size, random_state=random_seed)

    train_ds = ASTGCNTrafficDataset(Xh[train_idx], Xd[train_idx], Xw[train_idx], y[train_idx])
    val_ds = ASTGCNTrafficDataset(Xh[val_idx], Xd[val_idx], Xw[val_idx], y[val_idx])
    test_ds = ASTGCNTrafficDataset(Xh[test_idx], Xd[test_idx], Xw[test_idx], y[test_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    metadata = DatasetMetadata(
        nodes=nodes,
        adjacency=adjacency,
        laplacian=laplacian,
        mean=mean,
        std=std,
        coords=coords,
    )

    return train_loader, val_loader, test_loader, metadata
