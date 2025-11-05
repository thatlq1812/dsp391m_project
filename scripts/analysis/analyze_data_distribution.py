"""
Data Distribution Analysis for STMGT Model Validation

Purpose:
    1. Validate multi-modal speed distribution (Gaussian Mixture K=3)
    2. Check weather-traffic correlations
    3. Verify data quality for training

Expected outputs:
    - Speed histogram and Q-Q plot
    - BIC/AIC scores for K=1..5 Gaussian mixtures
    - Weather correlation matrix
    - Distribution statistics

Author: DSP391m Team
Date: October 31, 2025
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.mixture import GaussianMixture

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze processed dataset distributions for STMGT.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/processed/all_runs_combined.parquet"),
        help="Path to the processed dataset (default: data/processed/all_runs_combined.parquet)",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory to store generated analysis charts (default: outputs)",
    )
    return parser.parse_args(argv)


def load_data(dataset_path: Path) -> pd.DataFrame:
    """Load processed traffic data."""

    print("Loading data...")
    df = pd.read_parquet(dataset_path)
    print(f"Loaded {len(df):,} records")
    print(f"Unique runs: {df['run_id'].nunique() if 'run_id' in df.columns else 'N/A'}")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    return df


def analyze_speed_distribution(df: pd.DataFrame, outputs_dir: Path) -> pd.Series:
    """Analyze speed distribution and test for multi-modality."""
    print("\n" + "="*70)
    print("SPEED DISTRIBUTION ANALYSIS")
    print("="*70)
    
    speeds = df['speed_kmh'].dropna().clip(lower=0)
    
    print(f"\nSpeed Statistics:")
    print(f"  Count: {len(speeds):,}")
    print(f"  Mean: {speeds.mean():.2f} km/h")
    print(f"  Std: {speeds.std():.2f} km/h")
    print(f"  Min: {speeds.min():.2f} km/h")
    print(f"  Q1: {speeds.quantile(0.25):.2f} km/h")
    print(f"  Median: {speeds.median():.2f} km/h")
    print(f"  Q3: {speeds.quantile(0.75):.2f} km/h")
    print(f"  Max: {speeds.max():.2f} km/h")
    
    # Test normality
    sample_speeds = speeds.sample(min(5000, len(speeds)), random_state=42)
    shapiro_stat, shapiro_p = stats.shapiro(sample_speeds)
    print(f"\nShapiro-Wilk Test:")
    print(f"  Statistic: {shapiro_stat:.6f}")
    print(f"  P-value: {shapiro_p:.6e}")
    print(f"  Normal? {'No (multi-modal likely)' if shapiro_p < 0.05 else 'Yes'}")
    
    # Plot histogram and Q-Q plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(speeds, bins=80, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].set_xlabel('Speed (km/h)', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title('Speed Distribution Histogram', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Q-Q plot
    stats.probplot(sample_speeds, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot (Normal Distribution)', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = outputs_dir / 'speed_distribution.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()
    
    return speeds


def fit_gaussian_mixtures(speeds: pd.Series, outputs_dir: Path) -> tuple[pd.DataFrame, GaussianMixture]:
    """Fit Gaussian Mixture Models with K=1..5 and compare BIC/AIC."""
    print("\n" + "="*70)
    print("GAUSSIAN MIXTURE MODEL SELECTION")
    print("="*70)
    
    X = speeds.values.reshape(-1, 1)
    
    results = []
    print("\nFitting GMMs for K=1 to 5...")
    for k in range(1, 6):
        gmm = GaussianMixture(n_components=k, covariance_type='full', random_state=42, max_iter=200)
        gmm.fit(X)
        
        bic = gmm.bic(X)
        aic = gmm.aic(X)
        
        results.append({
            'K': k,
            'BIC': bic,
            'AIC': aic
        })
        
        print(f"  K={k}: BIC={bic:,.0f}, AIC={aic:,.0f}")
    
    results_df = pd.DataFrame(results)
    
    # Find optimal K
    optimal_bic = results_df.loc[results_df['BIC'].idxmin(), 'K']
    optimal_aic = results_df.loc[results_df['AIC'].idxmin(), 'K']
    
    print(f"\nOptimal K (BIC): {optimal_bic}")
    print(f"Optimal K (AIC): {optimal_aic}")
    
    # Plot BIC/AIC curves
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    ax.plot(results_df['K'], results_df['BIC'], marker='o', linewidth=2, markersize=8, label='BIC')
    ax.plot(results_df['K'], results_df['AIC'], marker='s', linewidth=2, markersize=8, label='AIC')
    ax.set_xlabel('Number of Components (K)', fontsize=11)
    ax.set_ylabel('Information Criterion', fontsize=11)
    ax.set_title('Gaussian Mixture Model Selection', fontsize=12, fontweight='bold')
    ax.set_xticks(range(1, 6))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = outputs_dir / 'gmm_model_selection.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_path}")
    plt.close()
    
    # Fit best model and show components
    best_k = int(optimal_bic)
    print(f"\nFitting GMM with K={best_k} components...")
    
    gmm_best = GaussianMixture(n_components=best_k, covariance_type='full', random_state=42, max_iter=200)
    gmm_best.fit(X)
    
    print("\nGMM Components:")
    for i in range(best_k):
        mean = gmm_best.means_[i][0]
        std = np.sqrt(gmm_best.covariances_[i][0][0])
        weight = gmm_best.weights_[i]
        print(f"  Component {i+1}: mean={mean:.2f} km/h, std={std:.2f}, weight={weight:.3f}")
    
    return results_df, gmm_best


def analyze_weather_correlations(df: pd.DataFrame, outputs_dir: Path) -> None:
    """Analyze correlations between weather and traffic speed."""
    print("\n" + "="*70)
    print("WEATHER-TRAFFIC CORRELATION ANALYSIS")
    print("="*70)
    
    cols = ['speed_kmh', 'temperature_c', 'wind_speed_kmh', 'precipitation_mm']
    
    # Check if columns exist
    available_cols = [col for col in cols if col in df.columns]
    
    if len(available_cols) < 2:
        print("\nWarning: Insufficient weather columns for correlation analysis")
        print(f"Available columns: {available_cols}")
        return
    
    corr_data = df[available_cols].dropna()
    
    print(f"\nSamples for correlation: {len(corr_data):,}")
    
    # Pearson correlation
    print("\nPearson Correlation:")
    pearson = corr_data.corr(method='pearson')
    print(pearson.round(3))
    
    # Spearman correlation
    print("\nSpearman Correlation (rank-based):")
    spearman = corr_data.corr(method='spearman')
    print(spearman.round(3))
    
    # Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pearson
    sns.heatmap(pearson, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=axes[0])
    axes[0].set_title('Pearson Correlation Matrix', fontsize=12, fontweight='bold')
    
    # Spearman
    sns.heatmap(spearman, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=axes[1])
    axes[1].set_title('Spearman Correlation Matrix', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    corr_path = outputs_dir / 'weather_correlation.png'
    plt.savefig(corr_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {corr_path}")
    plt.close()
    
    # Scatter plots
    if 'temperature_c' in df.columns and 'precipitation_mm' in df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        # Sample for visualization
        sample = corr_data.sample(min(5000, len(corr_data)), random_state=42)

        # Speed vs Temperature
        axes[0].scatter(sample['temperature_c'], sample['speed_kmh'], alpha=0.3, s=10)
        axes[0].set_xlabel('Temperature (C)', fontsize=11)
        axes[0].set_ylabel('Speed (km/h)', fontsize=11)
        axes[0].set_title('Speed vs Temperature', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Speed vs Wind
        if 'wind_speed_kmh' in sample.columns:
            axes[1].scatter(sample['wind_speed_kmh'], sample['speed_kmh'], alpha=0.3, s=10)
            axes[1].set_xlabel('Wind Speed (km/h)', fontsize=11)
            axes[1].set_ylabel('Speed (km/h)', fontsize=11)
            axes[1].set_title('Speed vs Wind Speed', fontsize=12, fontweight='bold')
            axes[1].grid(True, alpha=0.3)

        # Speed vs Precipitation
        if 'precipitation_mm' in sample.columns:
            axes[2].scatter(sample['precipitation_mm'], sample['speed_kmh'], alpha=0.3, s=10)
            axes[2].set_xlabel('Precipitation (mm)', fontsize=11)
            axes[2].set_ylabel('Speed (km/h)', fontsize=11)
            axes[2].set_title('Speed vs Precipitation', fontsize=12, fontweight='bold')
            axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        scatter_path = outputs_dir / 'weather_scatter_plots.png'
        plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved: {scatter_path}")
        plt.close()


def analyze_temporal_patterns(df: pd.DataFrame, outputs_dir: Path) -> None:
    """Analyze temporal patterns in traffic speed."""
    print("\n" + "="*70)
    print("TEMPORAL PATTERN ANALYSIS")
    print("="*70)
    
    if 'timestamp' not in df.columns:
        print("\nWarning: No timestamp column found")
        return
    
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['dow'] = df['timestamp'].dt.dayofweek
    
    # Hour of day pattern
    hourly_stats = df.groupby('hour')['speed_kmh'].agg(['mean', 'std', 'count']).reset_index()
    
    print("\nHourly Traffic Pattern:")
    print(hourly_stats.to_string(index=False))
    
    # Day of week pattern
    dow_stats = df.groupby('dow')['speed_kmh'].agg(['mean', 'std', 'count']).reset_index()
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    dow_stats['day_name'] = dow_stats['dow'].apply(lambda x: dow_names[x])
    
    print("\nDay of Week Pattern:")
    print(dow_stats[['day_name', 'mean', 'std', 'count']].to_string(index=False))
    
    # Plot temporal patterns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Hour of day
    axes[0].plot(hourly_stats['hour'], hourly_stats['mean'], marker='o', linewidth=2, markersize=6)
    axes[0].fill_between(hourly_stats['hour'], 
                         hourly_stats['mean'] - hourly_stats['std'],
                         hourly_stats['mean'] + hourly_stats['std'],
                         alpha=0.3)
    axes[0].set_xlabel('Hour of Day', fontsize=11)
    axes[0].set_ylabel('Average Speed (km/h)', fontsize=11)
    axes[0].set_title('Speed by Hour of Day', fontsize=12, fontweight='bold')
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].grid(True, alpha=0.3)
    
    # Day of week
    num_days = len(dow_stats)
    axes[1].bar(range(num_days), dow_stats['mean'], yerr=dow_stats['std'], 
                alpha=0.7, capsize=5, color='steelblue')
    axes[1].set_xlabel('Day of Week', fontsize=11)
    axes[1].set_ylabel('Average Speed (km/h)', fontsize=11)
    axes[1].set_title('Speed by Day of Week', fontsize=12, fontweight='bold')
    axes[1].set_xticks(range(num_days))
    axes[1].set_xticklabels([dow_names[d] for d in dow_stats['dow']])
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    temporal_path = outputs_dir / 'temporal_patterns.png'
    plt.savefig(temporal_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {temporal_path}")
    plt.close()


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Main analysis pipeline."""

    args = parse_args(argv)
    dataset_path = args.dataset if args.dataset.is_absolute() else (PROJECT_ROOT / args.dataset)
    outputs_dir = args.outputs_dir if args.outputs_dir.is_absolute() else (PROJECT_ROOT / args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("DATA DISTRIBUTION ANALYSIS FOR STMGT MODEL")
    print("=" * 70)

    df = load_data(dataset_path)

    speeds = analyze_speed_distribution(df, outputs_dir)
    fit_gaussian_mixtures(speeds, outputs_dir)
    analyze_weather_correlations(df, outputs_dir)
    analyze_temporal_patterns(df, outputs_dir)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nKey Findings:")
    print(f"  1. Speed distribution multi-modality: {outputs_dir / 'speed_distribution.png'}")
    print(f"  2. Optimal GMM components: {outputs_dir / 'gmm_model_selection.png'}")
    print(f"  3. Weather correlations: {outputs_dir / 'weather_correlation.png'}")
    print(f"  4. Temporal patterns: {outputs_dir / 'temporal_patterns.png'}")

    print("\nNext steps:")
    print("  - If K=3 is optimal -> Gaussian Mixture output validated")
    print("  - If weather correlation > 0.1 -> Weather features useful")
    print("  - If temporal patterns clear -> Hierarchical encoding needed")


if __name__ == "__main__":
    main()
