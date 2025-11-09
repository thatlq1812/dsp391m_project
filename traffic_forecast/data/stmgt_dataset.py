"""
STMGT Dataset and DataLoader

Purpose:
    - Load combined parquet data
    - Create sliding windows for time series
    - Prepare inputs for STMGT model
    - Handle graph structure (edge_index)

Author: DSP391m Team
Date: October 31, 2025
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from functools import partial


class STMGTDataset(Dataset):
    """
    Dataset for STMGT model
    
    Creates sliding windows from traffic data with:
    - Traffic history (12 timesteps = 3 hours)
    - Weather forecast (12 timesteps = 3 hours)
    - Temporal features (hour, dow, weekend)
    - Target speed (12 timesteps = 3 hours ahead)
    """
    
    def __init__(
        self,
        data_path='data/processed/all_runs_combined.parquet',
        graph_path='cache/overpass_topology.json',
        seq_len=12,  # History length (3 hours @ 15min)
        pred_len=12,  # Prediction length (3 hours @ 15min)
        split='train',
        train_ratio=0.7,
        val_ratio=0.15
    ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.split = split
        
        # Load data
        print(f"Loading data from {data_path}...")
        df = pd.read_parquet(data_path)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Data preprocessing and validation
        print("Preprocessing data...")
        
        # Fix temporal features (recompute from timestamp to avoid NaN)
        df['hour'] = df['timestamp'].dt.hour
        df['dow'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['dow'] >= 5).astype(int)
        
        # Fix missing weather data
        weather_cols = ['temperature_c', 'wind_speed_kmh', 'precipitation_mm']
        for col in weather_cols:
            if col in df.columns:
                # Fill NaN with column mean
                mean_val = df[col].mean()
                if pd.isna(mean_val):
                    mean_val = 0.0
                df[col] = df[col].fillna(mean_val)
                print(f"  {col}: filled {df[col].isna().sum()} missing values")
            else:
                # Create column with zeros if not exists
                df[col] = 0.0
                print(f"  {col}: created with zeros")
        
        # Validate data
        print("Validating data...")
        assert df['speed_kmh'].notna().all(), "Speed contains NaN"
        assert (df['speed_kmh'] >= 0).all(), "Speed contains negative values"
        assert (df['hour'] >= 0).all() and (df['hour'] <= 23).all(), "Invalid hour values"
        assert (df['dow'] >= 0).all() and (df['dow'] <= 6).all(), "Invalid day of week"
        print("  Data validation passed!")
        
        # Compute normalization statistics (will be used by model)
        self.speed_mean = df['speed_kmh'].mean()
        self.speed_std = df['speed_kmh'].std()
        self.weather_mean = df[weather_cols].mean().values
        self.weather_std = df[weather_cols].std().values
        
        print(f"  Speed: mean={self.speed_mean:.2f}, std={self.speed_std:.2f}")
        print(f"  Weather means: {self.weather_mean}")
        print(f"  Weather stds: {self.weather_std}")
        
        # Load graph structure
        # Use actual edges from data instead of topology file
        print("Building graph from traffic data...")
        
        # Get unique nodes from data
        unique_node_ids = set()
        unique_node_ids.update(df['node_a_id'].unique())
        unique_node_ids.update(df['node_b_id'].unique())
        
        self.node_list = sorted(list(unique_node_ids))
        self.node_to_idx = {node: idx for idx, node in enumerate(self.node_list)}
        self.num_nodes = len(self.node_list)
        
        print(f"Number of nodes: {self.num_nodes}")
        
        # Create edge_index from actual traffic edges
        edge_set = set()
        for _, row in df[['node_a_id', 'node_b_id']].drop_duplicates().iterrows():
            node_a = row['node_a_id']
            node_b = row['node_b_id']
            if node_a in self.node_to_idx and node_b in self.node_to_idx:
                edge_set.add((self.node_to_idx[node_a], self.node_to_idx[node_b]))
        
        edge_list = list(edge_set)
        self.edge_index = torch.tensor(edge_list, dtype=torch.long).t()
        
        # OPTIMIZATION: Pre-group data by run_id to avoid repeated filtering in __getitem__
        print("Pre-grouping data by run_id for faster loading...")
        self.run_data_cache = {}
        for run_id in df['run_id'].unique():
            self.run_data_cache[run_id] = df[df['run_id'] == run_id].copy()
        print(f"  Cached {len(self.run_data_cache)} runs")
        print(f"Number of edges: {self.edge_index.size(1)}")
        
        # Store dataframe for later access
        self.df = df
        
        # Create edge mapping for data lookup
        df['edge_key'] = df['node_a_id'].astype(str) + '--' + df['node_b_id'].astype(str)
        
        # Get unique runs
        runs = df['run_id'].unique()
        
        # Split by runs (blocked time split)
        n_train = int(len(runs) * train_ratio)
        n_val = int(len(runs) * val_ratio)
        
        if split == 'train':
            selected_runs = runs[:n_train]
        elif split == 'val':
            selected_runs = runs[n_train:n_train+n_val]
        else:  # test
            selected_runs = runs[n_train+n_val:]
        
        df_split = df[df['run_id'].isin(selected_runs)].reset_index(drop=True)
        
        print(f"Split: {split}")
        print(f"  Runs: {len(selected_runs)}")
        print(f"  Records: {len(df_split)}")
        
        # Create samples
        # Each run is a complete graph snapshot (144 edges)
        # We need sequential runs to form time series
        selected_runs = sorted(selected_runs)
        
        self.samples = []
        
        # Create sliding windows across runs (each run = 1 timestep)
        for i in range(len(selected_runs) - seq_len - pred_len + 1):
            window_runs = selected_runs[i:i+seq_len+pred_len]
            
            sample = {
                'runs': window_runs,
                'input_runs': window_runs[:seq_len],
                'target_runs': window_runs[seq_len:seq_len+pred_len]
            }
            self.samples.append(sample)
        
        print(f"Total samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get one sample
        
        Returns:
            Dictionary with graph-level tensors
        """
        sample = self.samples[idx]
        
        # Load data for all runs in window (use pre-cached data)
        input_data_list = []
        target_data_list = []
        
        for run_id in sample['input_runs']:
            run_data = self.run_data_cache[run_id]
            input_data_list.append(run_data)
        
        for run_id in sample['target_runs']:
            run_data = self.run_data_cache[run_id]
            target_data_list.append(run_data)
        
        # Prepare graph tensors
        # x_traffic: [num_nodes, seq_len, 1] - speed for each edge
        # x_weather: [seq_len, 3] - global weather
        # y_target: [num_nodes, pred_len] - target speed
        
        x_traffic = torch.zeros(self.num_nodes, self.seq_len, 1)
        y_target = torch.zeros(self.num_nodes, self.pred_len)
        
        # Fill input data (vectorized - much faster than iterrows)
        for t, run_data in enumerate(input_data_list):
            node_ids = run_data['node_a_id'].values
            speeds = run_data['speed_kmh'].values
            for node_id, speed in zip(node_ids, speeds):
                node_a_idx = self.node_to_idx.get(node_id)
                if node_a_idx is not None:
                    x_traffic[node_a_idx, t, 0] = speed
        
        # Fill target data (vectorized)
        for t, run_data in enumerate(target_data_list):
            node_ids = run_data['node_a_id'].values
            speeds = run_data['speed_kmh'].values
            for node_id, speed in zip(node_ids, speeds):
                node_a_idx = self.node_to_idx.get(node_id)
                if node_a_idx is not None:
                    y_target[node_a_idx, t] = speed
        
        # Weather data (use from first run - assumed same for all nodes)
        x_weather = torch.zeros(self.seq_len, 3)
        
        for t, run_data in enumerate(input_data_list):
            row = run_data.iloc[0]  # Any row, weather is same
            x_weather[t, 0] = row['temperature_c'] if not pd.isna(row['temperature_c']) else 0.0
            x_weather[t, 1] = row['wind_speed_kmh'] if not pd.isna(row['wind_speed_kmh']) else 0.0
            x_weather[t, 2] = row['precipitation_mm'] if not pd.isna(row['precipitation_mm']) else 0.0
        
        # Temporal features
        timestamps = [pd.to_datetime(run_data.iloc[0]['timestamp']) for run_data in input_data_list]
        hours = torch.tensor([ts.hour for ts in timestamps], dtype=torch.long)
        dows = torch.tensor([ts.dayofweek for ts in timestamps], dtype=torch.long)
        is_weekends = torch.tensor([1 if ts.dayofweek >= 5 else 0 for ts in timestamps], dtype=torch.long)
        
        return {
            'x_traffic': x_traffic,  # [num_nodes, seq_len, 1]
            'x_weather': x_weather,  # [seq_len, 3]
            'hour': hours,  # [seq_len]
            'dow': dows,  # [seq_len]
            'is_weekend': is_weekends,  # [seq_len]
            'y_target': y_target  # [num_nodes, pred_len]
        }


def collate_fn_stmgt(batch, num_nodes=None, edge_index=None):
    """
    Custom collate function for STMGT
    
    Each sample is already graph-level, just stack into batch
    
    Args:
        batch: List of samples from __getitem__
        num_nodes: Number of nodes in graph
        edge_index: Edge connectivity tensor
    """
    batch_size = len(batch)
    seq_len = batch[0]['x_traffic'].size(1)
    pred_len = batch[0]['y_target'].size(1)
    
    # Stack all samples
    x_traffic = torch.stack([s['x_traffic'] for s in batch])  # [B, N, T, 1]
    x_weather = torch.stack([s['x_weather'] for s in batch])  # [B, T, 3]
    y_target = torch.stack([s['y_target'] for s in batch])  # [B, N, P]
    
    temporal_features = {
        'hour': torch.stack([s['hour'] for s in batch]),  # [B, T]
        'dow': torch.stack([s['dow'] for s in batch]),  # [B, T]
        'is_weekend': torch.stack([s['is_weekend'] for s in batch])  # [B, T]
    }
    
    return {
        'x_traffic': x_traffic,
        'x_weather': x_weather,
        'edge_index': edge_index,
        'temporal_features': temporal_features,
        'y_target': y_target
    }


def create_stmgt_dataloaders(
    data_path='data/processed/all_runs_combined.parquet',
    graph_path='cache/overpass_topology.json',
    batch_size=16,
    num_workers=0,
    seq_len=12,
    pred_len=12,
    pin_memory=False,
    persistent_workers=False,
    prefetch_factor=None,
):
    """
    Create train/val/test dataloaders for STMGT
    """
    
    # Create datasets
    train_dataset = STMGTDataset(
        data_path=data_path,
        graph_path=graph_path,
        seq_len=seq_len,
        pred_len=pred_len,
        split='train'
    )
    
    val_dataset = STMGTDataset(
        data_path=data_path,
        graph_path=graph_path,
        seq_len=seq_len,
        pred_len=pred_len,
        split='val'
    )
    
    test_dataset = STMGTDataset(
        data_path=data_path,
        graph_path=graph_path,
        seq_len=seq_len,
        pred_len=pred_len,
        split='test'
    )
    
    # Get shared attributes
    num_nodes = train_dataset.num_nodes
    edge_index = train_dataset.edge_index
    
    # Create collate function (use partial instead of lambda for Windows multiprocessing)
    collate_fn = partial(collate_fn_stmgt, num_nodes=num_nodes, edge_index=edge_index)
    
    # Common loader kwargs
    def _loader_kwargs(shuffle: bool) -> dict:
        kwargs = {
            'batch_size': batch_size,
            'shuffle': shuffle,
            'num_workers': num_workers,
            'collate_fn': collate_fn,
        }

        if pin_memory:
            kwargs['pin_memory'] = True

        if num_workers > 0:
            if persistent_workers:
                kwargs['persistent_workers'] = True
            if prefetch_factor is not None:
                kwargs['prefetch_factor'] = prefetch_factor

        return kwargs

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        **_loader_kwargs(shuffle=True)
    )
    
    val_loader = DataLoader(
        val_dataset,
        **_loader_kwargs(shuffle=False)
    )
    
    test_loader = DataLoader(
        test_dataset,
        **_loader_kwargs(shuffle=False)
    )
    
    return train_loader, val_loader, test_loader, num_nodes, edge_index


if __name__ == "__main__":
    # Test dataset
    print("Testing STMGT Dataset...")
    
    train_loader, val_loader, test_loader, num_nodes, edge_index = create_stmgt_dataloaders(
        batch_size=4
    )
    
    print(f"\nDataloaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    print(f"  Num nodes: {num_nodes}")
    print(f"  Num edges: {edge_index.size(1)}")
    
    # Test one batch
    print("\nTesting one batch...")
    batch = next(iter(train_loader))
    
    print(f"Batch shapes:")
    print(f"  x_traffic: {batch['x_traffic'].shape}")
    print(f"  x_weather: {batch['x_weather'].shape}")
    print(f"  edge_index: {batch['edge_index'].shape}")
    print(f"  y_target: {batch['y_target'].shape}")
    print(f"  hour: {batch['temporal_features']['hour'].shape}")
    
    print("\nDataset test successful!")
