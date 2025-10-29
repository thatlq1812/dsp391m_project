"""
Preprocessing module for ML pipeline.
Handles data cleaning, imputation, scaling, and train/val/test splits.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
import warnings


class DataPreprocessor:
    """Preprocess data for machine learning."""

    def __init__(
        self,
        target_column: str = 'speed_kmh',
        scaler_type: str = 'standard',
        handle_outliers: bool = True,
        outlier_std: float = 3.0
    ):
        """
        Initialize preprocessor.

        Args:
            target_column: Name of target variable column
            scaler_type: Type of scaler ('standard', 'robust', 'none')
            handle_outliers: Whether to handle outliers
            outlier_std: Number of std deviations for outlier detection
        """
        self.target_column = target_column
        self.scaler_type = scaler_type
        self.handle_outliers = handle_outliers
        self.outlier_std = outlier_std

        self.scaler = None
        self.imputer = None
        self.feature_columns = None
        self.stats = {}

    def fit(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None):
        """
        Fit preprocessor on training data.

        Args:
            df: Training DataFrame
            feature_columns: List of feature column names. If None, auto-detect.
        """
        if feature_columns is None:
            # Auto-detect numeric features
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if self.target_column in feature_columns:
                feature_columns.remove(self.target_column)

        self.feature_columns = feature_columns
        X = df[feature_columns].copy()

        # Fit imputer
        self.imputer = SimpleImputer(strategy='median')
        self.imputer.fit(X)

        # Fit scaler
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif self.scaler_type == 'none':
            self.scaler = None
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

        if self.scaler:
            X_imputed = self.imputer.transform(X)
            self.scaler.fit(X_imputed)

        # Compute statistics
        self.stats = {
            'n_features': len(feature_columns),
            'missing_counts': df[feature_columns].isnull().sum().to_dict(),
            'feature_means': df[feature_columns].mean().to_dict(),
            'feature_stds': df[feature_columns].std().to_dict()
        }

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor.

        Args:
            df: DataFrame to transform

        Returns:
            Transformed DataFrame
        """
        if self.feature_columns is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        result = df.copy()
        X = result[self.feature_columns].copy()

        # Impute missing values
        X_imputed = self.imputer.transform(X)

        # Scale features
        if self.scaler:
            X_scaled = self.scaler.transform(X_imputed)
            result[self.feature_columns] = X_scaled
        else:
            result[self.feature_columns] = X_imputed

        return result

    def fit_transform(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(df, feature_columns)
        return self.transform(df)

    @staticmethod
    def remove_outliers(
        df: pd.DataFrame,
        columns: Optional[List[str]] = None,
        std_threshold: float = 3.0
    ) -> pd.DataFrame:
        """
        Remove outliers based on standard deviation method.

        Args:
            df: DataFrame to clean
            columns: Columns to check for outliers. If None, check all numeric.
            std_threshold: Number of standard deviations for outlier threshold

        Returns:
            DataFrame with outliers removed
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        result = df.copy()
        for col in columns:
            if col in result.columns:
                mean = result[col].mean()
                std = result[col].std()
                lower = mean - std_threshold * std
                upper = mean + std_threshold * std

                n_before = len(result)
                result = result[(result[col] >= lower) & (result[col] <= upper)]
                n_removed = n_before - len(result)

                if n_removed > 0:
                    warnings.warn(f"Removed {n_removed} outliers from column '{col}'")

        return result

    @staticmethod
    def handle_missing_values(
        df: pd.DataFrame,
        strategy: str = 'median',
        fill_value: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Handle missing values in DataFrame.

        Args:
            df: DataFrame with potential missing values
            strategy: Imputation strategy ('mean', 'median', 'constant', 'drop')
            fill_value: Value to use when strategy='constant'

        Returns:
            DataFrame with missing values handled
        """
        result = df.copy()

        if strategy == 'drop':
            return result.dropna()

        numeric_cols = result.select_dtypes(include=[np.number]).columns

        if strategy == 'mean':
            for col in numeric_cols:
                result[col].fillna(result[col].mean(), inplace=True)
        elif strategy == 'median':
            for col in numeric_cols:
                result[col].fillna(result[col].median(), inplace=True)
        elif strategy == 'constant':
            if fill_value is None:
                raise ValueError("fill_value must be provided when strategy='constant'")
            for col in numeric_cols:
                result[col].fillna(fill_value, inplace=True)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return result

    @staticmethod
    def clip_values(
        df: pd.DataFrame,
        column: str,
        lower: Optional[float] = None,
        upper: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Clip values in a column to specified range.

        Args:
            df: DataFrame to modify
            column: Column name to clip
            lower: Lower bound (None = no lower bound)
            upper: Upper bound (None = no upper bound)

        Returns:
            DataFrame with clipped values
        """
        result = df.copy()
        if column in result.columns:
            result[column] = result[column].clip(lower=lower, upper=upper)
        return result


def split_data(
    df: pd.DataFrame,
    target_column: str = 'speed_kmh',
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
    time_based: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.

    Args:
        df: DataFrame to split
        target_column: Name of target column
        test_size: Proportion of data for test set
        val_size: Proportion of remaining data for validation set
        random_state: Random seed for reproducibility
        time_based: If True, split by time (oldest=train, newest=test)

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    if time_based and 'timestamp' in df.columns:
        # Sort by timestamp
        df_sorted = df.sort_values('timestamp').reset_index(drop=True)

        n = len(df_sorted)
        test_start = int(n * (1 - test_size))
        val_start = int(test_start * (1 - val_size))

        train_df = df_sorted.iloc[:val_start].copy()
        val_df = df_sorted.iloc[val_start:test_start].copy()
        test_df = df_sorted.iloc[test_start:].copy()
    else:
        # Random split
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state
        )

        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            random_state=random_state
        )

    return train_df, val_df, test_df


def prepare_features_target(
    df: pd.DataFrame,
    target_column: str = 'speed_kmh',
    drop_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target variable.

    Args:
        df: DataFrame with features and target
        target_column: Name of target column
        drop_columns: Additional columns to drop from features

    Returns:
        Tuple of (X, y) where X is features DataFrame and y is target Series
    """
    if drop_columns is None:
        drop_columns = []

    # Automatically drop non-numeric and identifier columns
    auto_drop = ['node_a_id', 'node_b_id', 'node_id', 'timestamp', 'run_name', 'api_type']
    drop_columns = list(set(drop_columns + auto_drop))

    # Drop columns that exist
    cols_to_drop = [col for col in drop_columns if col in df.columns]
    if target_column in df.columns:
        cols_to_drop.append(target_column)

    X = df.drop(columns=cols_to_drop)

    # Keep only numeric columns
    X = X.select_dtypes(include=[np.number])

    y = df[target_column] if target_column in df.columns else None

    return X, y


def get_preprocessing_pipeline(config: Optional[Dict] = None) -> DataPreprocessor:
    """
    Create preprocessing pipeline from configuration.

    Args:
        config: Configuration dictionary with preprocessing parameters

    Returns:
        Configured DataPreprocessor instance
    """
    if config is None:
        config = {}

    return DataPreprocessor(
        target_column=config.get('target_column', 'speed_kmh'),
        scaler_type=config.get('scaler_type', 'standard'),
        handle_outliers=config.get('handle_outliers', True),
        outlier_std=config.get('outlier_std', 3.0)
    )
