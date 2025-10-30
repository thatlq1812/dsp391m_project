"""
Data management for traffic forecasting pipeline.

Handles:
- Loading data from JSON runs
- Preprocessing and feature engineering
- Train/test splitting
- Data caching

Author: thatlq1812
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from traffic_forecast.core.config import PipelineConfig

logger = logging.getLogger(__name__)


class DataManager:
    """Manage data loading, preprocessing, and feature engineering."""
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize data manager.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.data_raw = None
        self.data_processed = None
        self.data_features = None
        
        # Train/test splits
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Metadata
        self.feature_names = []
        self.scaler = None
        self.encoders = {}
    
    def load_data(self, max_runs: Optional[int] = None) -> pd.DataFrame:
        """
        Load traffic data from JSON files.
        
        Args:
            max_runs: Maximum number of runs to load (None = all)
            
        Returns:
            Raw DataFrame
        """
        logger.info("Loading traffic data...")
        
        data_dir = Path(self.config.data_dir)
        run_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()], reverse=True)
        
        if max_runs:
            run_dirs = run_dirs[:max_runs]
        
        logger.info(f"Found {len(run_dirs)} runs to load")
        
        all_data = []
        for run_dir in run_dirs:
            try:
                # Load traffic edges
                traffic_file = run_dir / 'traffic_edges.json'
                if not traffic_file.exists():
                    logger.warning(f"No traffic_edges.json in {run_dir.name}")
                    continue
                
                with open(traffic_file, 'r') as f:
                    traffic_data = json.load(f)
                
                df = pd.DataFrame(traffic_data)
                df['run_name'] = run_dir.name
                all_data.append(df)
                
                if self.config.verbose:
                    logger.info(f"✓ Loaded {len(df)} records from {run_dir.name}")
            
            except Exception as e:
                logger.error(f"Error loading {run_dir.name}: {e}")
                continue
        
        if not all_data:
            raise ValueError("No data loaded! Check your data directory.")
        
        self.data_raw = pd.concat(all_data, ignore_index=True)
        
        # Parse timestamp
        self.data_raw['timestamp'] = pd.to_datetime(self.data_raw['timestamp'])
        
        logger.info(f"✓ Total records loaded: {len(self.data_raw):,}")
        logger.info(f"Date range: {self.data_raw['timestamp'].min()} to {self.data_raw['timestamp'].max()}")
        
        return self.data_raw
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess raw data: clean, add basic features.
        
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing data...")
        
        if self.data_raw is None:
            raise ValueError("No raw data loaded. Call load_data() first.")
        
        df = self.data_raw.copy()
        
        # Add time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_name'] = df['timestamp'].dt.day_name()
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Add congestion levels
        df['congestion_level'] = df['speed_kmh'].apply(self._categorize_congestion)
        
        # Add speed categories
        df['speed_category'] = pd.cut(
            df['speed_kmh'],
            bins=[0, 10, 20, 30, 40, 100],
            labels=['very_slow', 'slow', 'moderate', 'fast', 'very_fast']
        )
        
        self.data_processed = df
        
        logger.info(f"✓ Preprocessed {len(df)} records")
        logger.info(f"Added columns: hour, day_of_week, is_weekend, congestion_level, speed_category")
        
        return self.data_processed
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Create advanced features for deep learning models.
        
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features...")
        
        if self.data_processed is None:
            raise ValueError("No preprocessed data. Call preprocess_data() first.")
        
        df = self.data_processed.copy()
        
        # Create edge_id if not exists
        if 'edge_id' not in df.columns:
            if 'node_a_id' in df.columns and 'node_b_id' in df.columns:
                df['edge_id'] = df['node_a_id'] + '_to_' + df['node_b_id']
            else:
                # Fallback: use start/end location
                df['edge_id'] = df.index.astype(str)
        
        # Sort for lag features
        df = df.sort_values(['edge_id', 'timestamp']).reset_index(drop=True)
        
        # 1. Lag features
        for lag in [1, 2, 3]:
            df[f'speed_lag_{lag}'] = df.groupby('edge_id')['speed_kmh'].shift(lag)
        
        # 2. Rolling statistics
        df['speed_rolling_mean_3'] = df.groupby('edge_id')['speed_kmh'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        df['speed_rolling_std_3'] = df.groupby('edge_id')['speed_kmh'].transform(
            lambda x: x.rolling(window=3, min_periods=1).std()
        )
        
        # 3. Cyclical time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # 4. Rush hour indicators
        df['is_morning_rush'] = df['hour'].isin([7, 8, 9]).astype(int)
        df['is_evening_rush'] = df['hour'].isin([17, 18, 19]).astype(int)
        df['is_rush_hour'] = (df['is_morning_rush'] | df['is_evening_rush']).astype(int)
        
        # 5. Encode categorical
        if 'congestion_level' in df.columns:
            le = LabelEncoder()
            df['congestion_level_encoded'] = le.fit_transform(df['congestion_level'])
            self.encoders['congestion_level'] = le
        
        # Drop NaN from lag features
        df = df.dropna()
        
        self.data_features = df
        
        logger.info(f"✓ Feature engineering complete: {df.shape[1]} columns, {len(df)} rows")
        logger.info(f"Dropped {len(self.data_processed) - len(df)} rows with NaN values")
        
        return self.data_features
    
    def prepare_train_test(
        self,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare train/test split.
        
        Args:
            test_size: Fraction for test set (default from config)
            random_state: Random seed (default from config)
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info("Preparing train/test split...")
        
        if self.data_features is None:
            raise ValueError("No feature data. Call engineer_features() first.")
        
        test_size = test_size or self.config.test_size
        random_state = random_state or self.config.random_state
        
        # Select features
        feature_cols = [col for col in self.config.feature_columns if col in self.data_features.columns]
        target_col = self.config.target_column
        
        X = self.data_features[feature_cols].copy()
        y = self.data_features[target_col].copy()
        
        # Time-based split (last 20% as test)
        split_idx = int(len(X) * (1 - test_size))
        
        self.X_train = X.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_train = y.iloc[:split_idx]
        self.y_test = y.iloc[split_idx:]
        
        self.feature_names = feature_cols
        
        logger.info(f"✓ Train: {len(self.X_train):,} samples ({len(self.X_train)/len(X)*100:.1f}%)")
        logger.info(f"✓ Test: {len(self.X_test):,} samples ({len(self.X_test)/len(X)*100:.1f}%)")
        logger.info(f"✓ Features: {len(feature_cols)}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def create_sequences(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time-series prediction.
        
        Args:
            X: Features DataFrame
            y: Target Series
            sequence_length: Number of past time steps
            
        Returns:
            X_sequences (samples, timesteps, features), y_targets
        """
        X_seq, y_seq = [], []
        
        X_values = X.values
        y_values = y.values
        
        for i in range(len(X) - sequence_length):
            X_seq.append(X_values[i:i+sequence_length])
            y_seq.append(y_values[i+sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def save_processed_data(self, output_path: Optional[Path] = None) -> Path:
        """
        Save processed data to Parquet.
        
        Args:
            output_path: Output file path (default: processed_dir/all_runs_combined.parquet)
            
        Returns:
            Path to saved file
        """
        if self.data_features is None:
            raise ValueError("No processed data to save.")
        
        output_path = output_path or (self.config.processed_dir / 'all_runs_combined.parquet')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.data_features.to_parquet(output_path, index=False)
        
        logger.info(f"✓ Saved processed data to {output_path}")
        return output_path
    
    def load_processed_data(self, input_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load processed data from Parquet.
        
        Args:
            input_path: Input file path (default: processed_dir/all_runs_combined.parquet)
            
        Returns:
            Loaded DataFrame
        """
        input_path = input_path or (self.config.processed_dir / 'all_runs_combined.parquet')
        
        if not input_path.exists():
            raise FileNotFoundError(f"Processed data not found: {input_path}")
        
        self.data_features = pd.read_parquet(input_path)
        
        logger.info(f"✓ Loaded processed data from {input_path}")
        logger.info(f"  Shape: {self.data_features.shape}")
        
        return self.data_features
    
    @staticmethod
    def _categorize_congestion(speed: float) -> str:
        """Categorize congestion level based on speed."""
        if speed < 15:
            return 'heavy'
        elif speed < 25:
            return 'moderate'
        elif speed < 35:
            return 'light'
        else:
            return 'free_flow'
    
    def get_summary(self) -> Dict:
        """Get summary statistics."""
        summary = {}
        
        if self.data_raw is not None:
            summary['raw_data'] = {
                'rows': len(self.data_raw),
                'columns': len(self.data_raw.columns),
                'date_range': (
                    str(self.data_raw['timestamp'].min()),
                    str(self.data_raw['timestamp'].max())
                )
            }
        
        if self.data_features is not None:
            summary['features'] = {
                'rows': len(self.data_features),
                'columns': len(self.data_features.columns),
                'feature_names': self.feature_names
            }
        
        if self.X_train is not None:
            summary['train_test'] = {
                'train_samples': len(self.X_train),
                'test_samples': len(self.X_test),
                'num_features': len(self.feature_names)
            }
        
        return summary
