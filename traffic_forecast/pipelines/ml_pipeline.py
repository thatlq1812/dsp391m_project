"""
Maintainer Profile

Name: THAT Le Quang
- Role: AI & DS Major Student
- GitHub: [thatlq1812]
- Primary email: fxlqthat@gmail.com
- Academic email: thatlqse183256@fpt.edu.com
- Alternate email: thatlq1812@gmail.com
- Phone (VN): +84 33 863 6369 / +84 39 730 6450

---

End-to-end ML pipeline for traffic forecasting.
Automates: data loading -> feature engineering -> preprocessing -> training -> evaluation
"""

import json
import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from traffic_forecast import PROJECT_ROOT

warnings.filterwarnings('ignore', category=FutureWarning)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for ML pipeline."""
    
    # Data paths
    data_runs_dir: Path = PROJECT_ROOT / "data_runs"
    output_dir: Path = PROJECT_ROOT / "data" / "ml_ready"
    
    # Feature engineering
    temporal_features: bool = True
    weather_features: bool = True
    spatial_features: bool = True
    historical_features: bool = True
    
    # Historical window (for lag features)
    lag_windows: List[int] = field(default_factory=lambda: [1, 2, 3, 6, 12])  # hours
    rolling_windows: List[int] = field(default_factory=lambda: [3, 6, 12, 24])  # hours
    
    # Train/test split
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    
    # Target variable
    target_column: str = "duration_in_traffic_s"
    prediction_horizons: List[int] = field(default_factory=lambda: [5, 15, 30, 60])  # minutes
    
    # Preprocessing
    handle_missing: str = "interpolate"  # Options: drop, interpolate, ffill
    normalize: bool = True
    remove_outliers: bool = True
    outlier_std_threshold: float = 3.0
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> 'PipelineConfig':
        """Load config from YAML file."""
        with config_path.open(encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        
        ml_cfg = cfg.get('ml_pipeline', {})
        return cls(**ml_cfg)


@dataclass
class PipelineArtifacts:
    """Artifacts produced by the pipeline."""
    
    train_path: Path
    val_path: Path
    test_path: Path
    scaler_path: Path
    metadata_path: Path
    feature_importance_path: Optional[Path] = None
    summary_path: Optional[Path] = None


class DataLoader:
    """Load and consolidate data from collection runs."""
    
    def __init__(self, data_runs_dir: Path):
        self.data_runs_dir = data_runs_dir
    
    def find_latest_runs(self, n_runs: int = 10) -> List[Path]:
        """Find N most recent collection runs."""
        if not self.data_runs_dir.exists():
            raise FileNotFoundError(f"Data runs directory not found: {self.data_runs_dir}")
        
        run_dirs = sorted(
            [d for d in self.data_runs_dir.iterdir() if d.is_dir()],
            key=lambda x: x.name,
            reverse=True
        )
        
        logger.info(f"Found {len(run_dirs)} collection runs")
        return run_dirs[:n_runs]
    
    def load_run_data(self, run_dir: Path) -> Dict[str, pd.DataFrame]:
        """Load all data from a single run."""
        data = {}
        
        # Load Google traffic data
        google_file = run_dir / "collectors" / "google" / "traffic_snapshot.json"
        if google_file.exists():
            with google_file.open(encoding='utf-8') as f:
                google_data = json.load(f)
            
            if google_data:
                df_google = pd.DataFrame(google_data)
                df_google['timestamp'] = pd.to_datetime(df_google['timestamp'])
                data['google'] = df_google
        
        # Load weather data
        weather_file = run_dir / "collectors" / "open_meteo" / "weather_snapshot.json"
        if weather_file.exists():
            with weather_file.open(encoding='utf-8') as f:
                weather_data = json.load(f)
            
            if weather_data:
                df_weather = pd.DataFrame(weather_data)
                df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'])
                data['weather'] = df_weather
        
        # Load nodes data
        nodes_file = run_dir / "collectors" / "overpass" / "nodes.json"
        if nodes_file.exists():
            with nodes_file.open(encoding='utf-8') as f:
                nodes_data = json.load(f)
            
            if nodes_data:
                data['nodes'] = pd.DataFrame(nodes_data)
        
        return data
    
    def consolidate_runs(self, run_dirs: List[Path]) -> pd.DataFrame:
        """Consolidate multiple runs into single DataFrame."""
        all_records = []
        
        for run_dir in run_dirs:
            logger.info(f"Loading run: {run_dir.name}")
            
            try:
                run_data = self.load_run_data(run_dir)
                
                if 'google' not in run_data:
                    logger.warning(f"No Google data in {run_dir.name}, skipping")
                    continue
                
                df_google = run_data['google']
                df_weather = run_data.get('weather')
                df_nodes = run_data.get('nodes')
                
                # Merge Google + Weather by node_id and timestamp
                if df_weather is not None:
                    df_merged = df_google.merge(
                        df_weather,
                        on=['node_id', 'timestamp'],
                        how='left',
                        suffixes=('', '_weather')
                    )
                else:
                    df_merged = df_google.copy()
                
                # Add node metadata (lat, lon, street names)
                if df_nodes is not None:
                    df_merged = df_merged.merge(
                        df_nodes[['node_id', 'lat', 'lon']],
                        on='node_id',
                        how='left',
                        suffixes=('', '_node')
                    )
                
                all_records.append(df_merged)
            
            except Exception as e:
                logger.error(f"Error loading {run_dir.name}: {e}")
                continue
        
        if not all_records:
            raise ValueError("No valid data found in any runs")
        
        df_consolidated = pd.concat(all_records, ignore_index=True)
        logger.info(f"Consolidated {len(df_consolidated)} records from {len(all_records)} runs")
        
        return df_consolidated


class FeatureEngineer:
    """Engineer features from raw traffic data."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df = df.copy()
        
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Peak hours (Vietnam traffic patterns)
        df['is_morning_peak'] = ((df['hour'] >= 6) & (df['hour'] <= 9)).astype(int)
        df['is_evening_peak'] = ((df['hour'] >= 16) & (df['hour'] <= 19)).astype(int)
        df['is_lunch_peak'] = ((df['hour'] >= 11) & (df['hour'] <= 13)).astype(int)
        df['is_peak'] = (df['is_morning_peak'] | df['is_evening_peak'] | df['is_lunch_peak']).astype(int)
        
        # Cyclical encoding for hour and day
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        logger.info("Added temporal features")
        return df
    
    def add_spatial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add location-based features."""
        df = df.copy()
        
        # Distance from city center (HCMC coordinates)
        center_lat, center_lon = 10.772465, 106.697794
        
        if 'lat' in df.columns and 'lon' in df.columns:
            df['dist_from_center_km'] = self._haversine_distance(
                df['lat'], df['lon'],
                center_lat, center_lon
            )
        
        logger.info("Added spatial features")
        return df
    
    def add_lag_features(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """Add lagged values for time series."""
        df = df.copy()
        df = df.sort_values(['node_id', 'timestamp'])
        
        for lag_hours in self.config.lag_windows:
            lag_name = f'{value_col}_lag_{lag_hours}h'
            df[lag_name] = df.groupby('node_id')[value_col].shift(lag_hours)
        
        logger.info(f"Added lag features for {value_col}")
        return df
    
    def add_rolling_features(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """Add rolling statistics."""
        df = df.copy()
        df = df.sort_values(['node_id', 'timestamp'])
        
        for window_hours in self.config.rolling_windows:
            # Rolling mean
            df[f'{value_col}_rolling_mean_{window_hours}h'] = (
                df.groupby('node_id')[value_col]
                .rolling(window=window_hours, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            
            # Rolling std
            df[f'{value_col}_rolling_std_{window_hours}h'] = (
                df.groupby('node_id')[value_col]
                .rolling(window=window_hours, min_periods=1)
                .std()
                .reset_index(level=0, drop=True)
            )
        
        logger.info(f"Added rolling features for {value_col}")
        return df
    
    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps."""
        logger.info("Starting feature engineering...")
        
        if self.config.temporal_features:
            df = self.add_temporal_features(df)
        
        if self.config.spatial_features:
            df = self.add_spatial_features(df)
        
        # Historical features only if we have enough data
        if self.config.historical_features and len(df) > 24:
            df = self.add_lag_features(df, 'duration_in_traffic_s')
            df = self.add_rolling_features(df, 'duration_in_traffic_s')
        
        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        return df
    
    @staticmethod
    def _haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate haversine distance in km."""
        R = 6371  # Earth radius in km
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        a = (np.sin(delta_lat / 2) ** 2 +
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2) ** 2)
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        
        return R * c


class DataPreprocessor:
    """Preprocess engineered features."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.scaler = StandardScaler()
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on config."""
        df = df.copy()
        
        if self.config.handle_missing == "drop":
            df = df.dropna()
        elif self.config.handle_missing == "interpolate":
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear', limit_direction='both')
            df = df.fillna(0)  # Fill remaining with 0
        elif self.config.handle_missing == "ffill":
            df = df.fillna(method='ffill').fillna(method='bfill')
        
        logger.info(f"Handled missing values using: {self.config.handle_missing}")
        return df
    
    def remove_outliers(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Remove outliers using z-score method."""
        if not self.config.remove_outliers:
            return df
        
        df = df.copy()
        mean = df[column].mean()
        std = df[column].std()
        
        z_scores = np.abs((df[column] - mean) / std)
        df_filtered = df[z_scores < self.config.outlier_std_threshold]
        
        removed = len(df) - len(df_filtered)
        logger.info(f"Removed {removed} outliers from {column}")
        
        return df_filtered
    
    def normalize_features(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Normalize features using StandardScaler."""
        if not self.config.normalize:
            return X_train, X_val, X_test
        
        # Fit on training data only
        self.scaler.fit(X_train)
        
        # Transform all sets
        X_train_scaled = pd.DataFrame(
            self.scaler.transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        logger.info("Normalized features using StandardScaler")
        return X_train_scaled, X_val_scaled, X_test_scaled


class MLPipeline:
    """Complete ML pipeline orchestrator."""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.data_loader = DataLoader(self.config.data_runs_dir)
        self.feature_engineer = FeatureEngineer(self.config)
        self.preprocessor = DataPreprocessor(self.config)
        self.artifacts: Optional[PipelineArtifacts] = None
    
    def run(
        self,
        n_runs: int = 10,
        save_artifacts: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Run complete pipeline.
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        logger.info("=" * 80)
        logger.info("STARTING ML PIPELINE")
        logger.info("=" * 80)
        
        # Step 1: Load data
        logger.info("\n[1/6] Loading data from collection runs...")
        run_dirs = self.data_loader.find_latest_runs(n_runs)
        df_raw = self.data_loader.consolidate_runs(run_dirs)
        
        # Step 2: Feature engineering
        logger.info("\n[2/6] Engineering features...")
        df_features = self.feature_engineer.engineer_all_features(df_raw)
        
        # Step 3: Handle missing values
        logger.info("\n[3/6] Preprocessing data...")
        df_clean = self.preprocessor.handle_missing_values(df_features)
        
        # Step 4: Remove outliers from target
        df_clean = self.preprocessor.remove_outliers(df_clean, self.config.target_column)
        
        # Step 5: Prepare features and target
        logger.info("\n[4/6] Preparing features and target...")
        
        # Select feature columns (exclude metadata and target)
        exclude_cols = [
            'timestamp', 'node_id', 'origin_node_id', 'destination_node_id',
            self.config.target_column, 'distance_m', 'lat_x', 'lon_x',
            'lat_y', 'lon_y', 'lat', 'lon'
        ]
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        
        X = df_clean[feature_cols]
        y = df_clean[self.config.target_column]
        
        logger.info(f"Features: {len(feature_cols)} columns")
        logger.info(f"Target: {self.config.target_column}")
        logger.info(f"Samples: {len(X)}")
        
        # Step 6: Train/val/test split
        logger.info("\n[5/6] Splitting data...")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state
        )
        
        # Second split: train vs val
        val_ratio = self.config.validation_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=self.config.random_state
        )
        
        logger.info(f"Train: {len(X_train)} samples")
        logger.info(f"Val: {len(X_val)} samples")
        logger.info(f"Test: {len(X_test)} samples")
        
        # Step 7: Normalize
        X_train, X_val, X_test = self.preprocessor.normalize_features(
            X_train, X_val, X_test
        )
        
        # Step 8: Save artifacts
        if save_artifacts:
            logger.info("\n[6/6] Saving artifacts...")
            self.artifacts = self._save_artifacts(
                X_train, X_val, X_test,
                y_train, y_val, y_test,
                feature_cols
            )
            logger.info(f"Artifacts saved to: {self.config.output_dir}")
        
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETE!")
        logger.info("=" * 80)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def _save_artifacts(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        feature_cols: List[str]
    ) -> PipelineArtifacts:
        """Save all pipeline artifacts."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save datasets
        train_path = self.config.output_dir / f"train_{timestamp}.parquet"
        val_path = self.config.output_dir / f"val_{timestamp}.parquet"
        test_path = self.config.output_dir / f"test_{timestamp}.parquet"
        
        train_df = X_train.copy()
        train_df[self.config.target_column] = y_train
        train_df.to_parquet(train_path, index=False)
        
        val_df = X_val.copy()
        val_df[self.config.target_column] = y_val
        val_df.to_parquet(val_path, index=False)
        
        test_df = X_test.copy()
        test_df[self.config.target_column] = y_test
        test_df.to_parquet(test_path, index=False)
        
        # Save scaler
        scaler_path = self.config.output_dir / f"scaler_{timestamp}.pkl"
        joblib.dump(self.preprocessor.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'feature_columns': feature_cols,
            'target_column': self.config.target_column,
            'n_features': len(feature_cols),
            'n_train': len(X_train),
            'n_val': len(X_val),
            'n_test': len(X_test),
            'config': {
                'temporal_features': self.config.temporal_features,
                'weather_features': self.config.weather_features,
                'spatial_features': self.config.spatial_features,
                'historical_features': self.config.historical_features,
                'normalize': self.config.normalize,
                'test_size': self.config.test_size,
                'validation_size': self.config.validation_size
            }
        }
        
        metadata_path = self.config.output_dir / f"metadata_{timestamp}.json"
        with metadata_path.open('w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return PipelineArtifacts(
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            scaler_path=scaler_path,
            metadata_path=metadata_path
        )


def run_pipeline_from_cli():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ML pipeline')
    parser.add_argument('--n-runs', type=int, default=10, help='Number of collection runs to use')
    parser.add_argument('--no-save', action='store_true', help='Do not save artifacts')
    args = parser.parse_args()
    
    config = PipelineConfig()
    pipeline = MLPipeline(config)
    
    X_train, X_val, X_test, y_train, y_val, y_test = pipeline.run(
        n_runs=args.n_runs,
        save_artifacts=not args.no_save
    )
    
    print(f"\nPipeline complete!")
    print(f"Train shape: {X_train.shape}")
    print(f"Val shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")


if __name__ == "__main__":
    run_pipeline_from_cli()
