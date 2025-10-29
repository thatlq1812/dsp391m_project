"""
Automated data quality validation for collected traffic data.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DataQualityValidator:
    """Validate quality of collected traffic data."""

    def __init__(self, thresholds: Optional[Dict] = None):
        """
        Initialize validator with quality thresholds.

        Args:
            thresholds: Dictionary of validation thresholds
        """
        self.thresholds = thresholds or {
            'min_speed_kmh': 0,
            'max_speed_kmh': 120,
            'min_duration_sec': 0,
            'max_duration_sec': 3600,
            'min_distance_km': 0,
            'max_distance_km': 50,
            'completeness_threshold': 0.8,
            'max_missing_rate': 0.2
        }

    def validate_traffic_data(self, df: pd.DataFrame) -> Dict:
        """
        Validate traffic data DataFrame.

        Args:
            df: DataFrame with traffic data

        Returns:
            Dictionary with validation results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_records': len(df),
            'passed': True,
            'errors': [],
            'warnings': [],
            'metrics': {}
        }

        # Check for required columns
        required_cols = ['speed_kmh', 'duration_sec', 'distance_km']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            results['errors'].append(f"Missing required columns: {missing_cols}")
            results['passed'] = False
            return results

        # Validate speed
        speed_issues = self._validate_speed(df)
        results['metrics']['speed'] = speed_issues
        if speed_issues['invalid_count'] > 0:
            results['warnings'].append(
                f"{speed_issues['invalid_count']} records with invalid speed"
            )

        # Validate duration
        duration_issues = self._validate_duration(df)
        results['metrics']['duration'] = duration_issues
        if duration_issues['invalid_count'] > 0:
            results['warnings'].append(
                f"{duration_issues['invalid_count']} records with invalid duration"
            )

        # Validate distance
        distance_issues = self._validate_distance(df)
        results['metrics']['distance'] = distance_issues
        if distance_issues['invalid_count'] > 0:
            results['warnings'].append(
                f"{distance_issues['invalid_count']} records with invalid distance"
            )

        # Check completeness
        completeness = self._check_completeness(df)
        results['metrics']['completeness'] = completeness
        if completeness['rate'] < self.thresholds['completeness_threshold']:
            results['errors'].append(
                f"Data completeness {completeness['rate']:.2%} below threshold "
                f"{self.thresholds['completeness_threshold']:.2%}"
            )
            results['passed'] = False

        # Check for duplicates
        duplicates = self._check_duplicates(df)
        results['metrics']['duplicates'] = duplicates
        if duplicates['count'] > 0:
            results['warnings'].append(
                f"{duplicates['count']} duplicate records found"
            )

        # Statistical summary
        results['metrics']['statistics'] = self._calculate_statistics(df)

        return results

    def _validate_speed(self, df: pd.DataFrame) -> Dict:
        """Validate speed values."""
        min_speed = self.thresholds['min_speed_kmh']
        max_speed = self.thresholds['max_speed_kmh']

        invalid = (
            (df['speed_kmh'] < min_speed) |
            (df['speed_kmh'] > max_speed) |
            df['speed_kmh'].isna()
        )

        return {
            'invalid_count': invalid.sum(),
            'invalid_rate': invalid.mean(),
            'min': df['speed_kmh'].min(),
            'max': df['speed_kmh'].max(),
            'mean': df['speed_kmh'].mean(),
            'median': df['speed_kmh'].median()
        }

    def _validate_duration(self, df: pd.DataFrame) -> Dict:
        """Validate duration values."""
        min_dur = self.thresholds['min_duration_sec']
        max_dur = self.thresholds['max_duration_sec']

        invalid = (
            (df['duration_sec'] < min_dur) |
            (df['duration_sec'] > max_dur) |
            df['duration_sec'].isna()
        )

        return {
            'invalid_count': invalid.sum(),
            'invalid_rate': invalid.mean(),
            'min': df['duration_sec'].min(),
            'max': df['duration_sec'].max(),
            'mean': df['duration_sec'].mean()
        }

    def _validate_distance(self, df: pd.DataFrame) -> Dict:
        """Validate distance values."""
        min_dist = self.thresholds['min_distance_km']
        max_dist = self.thresholds['max_distance_km']

        invalid = (
            (df['distance_km'] < min_dist) |
            (df['distance_km'] > max_dist) |
            df['distance_km'].isna()
        )

        return {
            'invalid_count': invalid.sum(),
            'invalid_rate': invalid.mean(),
            'min': df['distance_km'].min(),
            'max': df['distance_km'].max(),
            'mean': df['distance_km'].mean()
        }

    def _check_completeness(self, df: pd.DataFrame) -> Dict:
        """Check data completeness."""
        total_cells = df.size
        missing_cells = df.isna().sum().sum()

        return {
            'total_cells': total_cells,
            'missing_cells': missing_cells,
            'rate': 1 - (missing_cells / total_cells) if total_cells > 0 else 0
        }

    def _check_duplicates(self, df: pd.DataFrame) -> Dict:
        """Check for duplicate records."""
        if 'node_a_id' in df.columns and 'node_b_id' in df.columns:
            duplicates = df.duplicated(subset=['node_a_id', 'node_b_id'], keep='first')
            return {
                'count': duplicates.sum(),
                'rate': duplicates.mean()
            }
        return {'count': 0, 'rate': 0}

    def _calculate_statistics(self, df: pd.DataFrame) -> Dict:
        """Calculate statistical summary."""
        stats = {}

        for col in ['speed_kmh', 'duration_sec', 'distance_km']:
            if col in df.columns:
                stats[col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'q25': float(df[col].quantile(0.25)),
                    'median': float(df[col].median()),
                    'q75': float(df[col].quantile(0.75)),
                    'max': float(df[col].max())
                }

        return stats

    def save_report(self, results: Dict, output_path: Path):
        """Save validation report to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Validation report saved to {output_path}")


def validate_collection_run(run_dir: Path) -> Dict:
    """
    Validate a complete collection run.

    Args:
        run_dir: Path to collection run directory

    Returns:
        Validation results dictionary
    """
    validator = DataQualityValidator()

    # Find traffic data file
    traffic_file = run_dir / 'collectors' / 'google' / 'traffic_data.json'

    if not traffic_file.exists():
        return {
            'passed': False,
            'error': f"Traffic data file not found: {traffic_file}"
        }

    # Load and validate data
    try:
        with traffic_file.open() as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        results = validator.validate_traffic_data(df)

        # Save validation report
        report_path = run_dir / 'quality_validation.json'
        validator.save_report(results, report_path)

        return results

    except Exception as e:
        logger.error(f"Error validating data: {e}")
        return {
            'passed': False,
            'error': str(e)
        }


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_quality_validator.py <run_directory>")
        sys.exit(1)

    run_dir = Path(sys.argv[1])
    results = validate_collection_run(run_dir)

    print(json.dumps(results, indent=2, default=str))
    sys.exit(0 if results.get('passed') else 1)
