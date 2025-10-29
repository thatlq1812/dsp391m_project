"""
Feature Importance Analysis and Visualization
Analyzes XGBoost model feature importance and creates visualizations
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

PROJECT_ROOT = Path(__file__).parent
PLOTS_DIR = PROJECT_ROOT / 'data' / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model_and_data():
    """Load trained model and feature data"""
    # Load model
    model_path = PROJECT_ROOT / 'models' / 'xgboost_traffic_v1.pkl'
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return None, None, None
    
    model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")
    
    # Load latest data
    data_dirs = sorted((PROJECT_ROOT / 'data' / 'downloads').glob('download_*'))
    if not data_dirs:
        print("No data directories found")
        return None, None, None
    
    latest_dir = data_dirs[-1]
    normalized_file = latest_dir / 'normalized_traffic.json'
    
    if not normalized_file.exists():
        print(f"Normalized data not found: {normalized_file}")
        return None, None, None
    
    with open(normalized_file, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    print(f"Loaded data: {len(df)} records from {latest_dir.name}")
    
    return model, df, latest_dir


def create_feature_importance_plot(model, feature_names, top_n=20):
    """Create feature importance bar chart"""
    # Get feature importance
    importance = model.feature_importances_
    
    # Create dataframe
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot top N features
    plt.figure(figsize=(12, 8))
    top_features = feature_importance_df.head(top_n)
    
    sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
    plt.title(f'Top {top_n} Feature Importance - XGBoost Traffic Model', fontsize=16, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    output_file = PLOTS_DIR / 'feature_importance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()
    
    return feature_importance_df


def create_correlation_heatmap(df, features):
    """Create correlation heatmap for numerical features"""
    # Select numerical columns
    numerical_cols = df[features].select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) < 2:
        print("Not enough numerical features for correlation matrix")
        return
    
    # Calculate correlation
    corr_matrix = df[numerical_cols].corr()
    
    # Plot
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                center=0, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8})
    
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = PLOTS_DIR / 'correlation_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def create_distribution_plots(df, target='speed_kmh'):
    """Create distribution plots for key features"""
    if target not in df.columns:
        print(f"Target {target} not in dataframe")
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    # Plot target distribution
    sns.histplot(data=df, x=target, kde=True, ax=axes[0], color='steelblue')
    axes[0].set_title(f'{target} Distribution', fontweight='bold')
    axes[0].set_xlabel(target)
    
    # Plot other important features
    features_to_plot = ['duration_sec', 'distance_km', 'temperature_c', 
                        'precipitation_mm', 'wind_speed_kmh']
    
    for i, feature in enumerate(features_to_plot, 1):
        if feature in df.columns:
            sns.histplot(data=df, x=feature, kde=True, ax=axes[i], color='coral')
            axes[i].set_title(f'{feature} Distribution', fontweight='bold')
            axes[i].set_xlabel(feature)
        else:
            axes[i].text(0.5, 0.5, f'{feature}\nNot Available', 
                        ha='center', va='center', fontsize=12)
            axes[i].set_title(feature, fontweight='bold')
    
    plt.tight_layout()
    
    output_file = PLOTS_DIR / 'feature_distributions.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def create_time_series_plot(df):
    """Create time series plot of traffic speed"""
    if 'timestamp' not in df.columns or 'speed_kmh' not in df.columns:
        print("Missing timestamp or speed_kmh columns")
        return
    
    # Convert timestamp
    df['datetime'] = pd.to_datetime(df['timestamp'])
    df_sorted = df.sort_values('datetime')
    
    # Aggregate by hour
    df_hourly = df_sorted.set_index('datetime').resample('H')['speed_kmh'].agg(['mean', 'std', 'count'])
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Mean speed over time
    ax1.plot(df_hourly.index, df_hourly['mean'], color='steelblue', linewidth=2)
    ax1.fill_between(df_hourly.index, 
                      df_hourly['mean'] - df_hourly['std'],
                      df_hourly['mean'] + df_hourly['std'],
                      alpha=0.3, color='steelblue')
    ax1.set_title('Traffic Speed Over Time (Hourly Average)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Speed (km/h)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Sample count
    ax2.bar(df_hourly.index, df_hourly['count'], color='coral', alpha=0.7)
    ax2.set_title('Number of Samples per Hour', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_file = PLOTS_DIR / 'time_series_traffic.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def save_feature_importance_table(feature_importance_df):
    """Save feature importance to markdown table"""
    output_file = PLOTS_DIR / 'feature_importance.md'
    
    with open(output_file, 'w') as f:
        f.write("# Feature Importance Analysis\n\n")
        f.write("## XGBoost Model - Top 20 Features\n\n")
        f.write("| Rank | Feature | Importance Score |\n")
        f.write("|------|---------|------------------|\n")
        
        for i, row in feature_importance_df.head(20).iterrows():
            f.write(f"| {i+1} | {row['feature']} | {row['importance']:.4f} |\n")
        
        f.write(f"\n**Total Features:** {len(feature_importance_df)}\n")
        f.write(f"**Date Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Saved: {output_file}")


def main():
    """Main execution"""
    print("=" * 60)
    print("FEATURE IMPORTANCE & VISUALIZATION ANALYSIS")
    print("=" * 60)
    
    # Load model and data
    model, df, data_dir = load_model_and_data()
    
    if model is None:
        print("\nERROR: Could not load model or data")
        print("Please train model first using ML_TRAINING.ipynb")
        return
    
    # Get feature names
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
    else:
        print("WARNING: Model does not have feature_names_in_, using column names")
        feature_names = [col for col in df.columns if col not in ['speed_kmh', 'timestamp', 'node_id']]
    
    print(f"\nTotal features: {len(feature_names)}")
    
    # Create visualizations
    print("\n" + "=" * 60)
    print("Creating Visualizations...")
    print("=" * 60)
    
    # 1. Feature Importance
    print("\n1. Feature Importance Plot...")
    feature_importance_df = create_feature_importance_plot(model, feature_names)
    save_feature_importance_table(feature_importance_df)
    
    # 2. Correlation Heatmap
    print("\n2. Correlation Heatmap...")
    create_correlation_heatmap(df, feature_names)
    
    # 3. Distribution Plots
    print("\n3. Feature Distribution Plots...")
    create_distribution_plots(df)
    
    # 4. Time Series
    print("\n4. Time Series Plot...")
    create_time_series_plot(df)
    
    print("\n" + "=" * 60)
    print(f"ALL VISUALIZATIONS SAVED TO: {PLOTS_DIR}")
    print("=" * 60)
    
    # Print top 10 features
    print("\nTop 10 Most Important Features:")
    for i, row in feature_importance_df.head(10).iterrows():
        print(f"  {i+1:2d}. {row['feature']:30s} {row['importance']:.4f}")


if __name__ == "__main__":
    main()
