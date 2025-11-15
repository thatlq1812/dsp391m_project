"""
Generate EDA figures (Figures 5-10)

Section 7: Exploratory Data Analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))
from utils import save_figure, load_parquet_data, FIGURE_DIR

def generate_fig5_eda_speed_hist():
    """Figure 5: Speed Distribution with Gaussian Mixture"""
    print("Generating Figure 5: Speed Histogram with GMM...")
    
    from sklearn.mixture import GaussianMixture
    
    df = load_parquet_data()
    speeds = df['speed_kmh'].values.reshape(-1, 1)
    
    # Fit GMM with K=3 components
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(speeds)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    ax.hist(speeds, bins=60, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='Data')
    
    # GMM components
    x_range = np.linspace(speeds.min(), speeds.max(), 1000).reshape(-1, 1)
    log_prob = gmm.score_samples(x_range)
    pdf = np.exp(log_prob)
    
    ax.plot(x_range, pdf, 'r-', linewidth=2, label='GMM (K=3)')
    
    # Individual components
    for i, (mean, covar, weight) in enumerate(zip(gmm.means_, gmm.covariances_, gmm.weights_)):
        from scipy.stats import norm
        component_pdf = weight * norm.pdf(x_range, mean[0], np.sqrt(covar[0][0]))
        ax.plot(x_range, component_pdf, '--', alpha=0.7, label=f'Component {i+1}')
    
    ax.set_xlabel('Speed (km/h)')
    ax.set_ylabel('Density')
    ax.set_title('Multi-Modal Speed Distribution with Gaussian Mixture (K=3)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    save_figure(fig, 'fig05_eda_speed_hist')

def generate_fig6_hourly_pattern():
    """Figure 6: Average Speed by Hour"""
    print("Generating Figure 6: Hourly Speed Pattern...")
    
    df = load_parquet_data()
    
    # Extract hour
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    
    # Group by hour
    hourly_stats = df.groupby('hour')['speed_kmh'].agg(['mean', 'std']).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Line plot with confidence interval
    ax.plot(hourly_stats['hour'], hourly_stats['mean'], 'o-', linewidth=2, markersize=6, color='steelblue')
    ax.fill_between(hourly_stats['hour'], 
                     hourly_stats['mean'] - hourly_stats['std'],
                     hourly_stats['mean'] + hourly_stats['std'],
                     alpha=0.3, color='steelblue', label='±1 std')
    
    # Highlight rush hours
    ax.axvspan(7, 9, alpha=0.1, color='red', label='Morning Rush')
    ax.axvspan(17, 19, alpha=0.1, color='orange', label='Evening Rush')
    
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Average Speed (km/h)')
    ax.set_title('Traffic Speed by Hour of Day')
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(alpha=0.3)
    
    save_figure(fig, 'fig06_hourly_pattern')

def generate_fig7_weekly_pattern():
    """Figure 7: Speed by Day of Week"""
    print("Generating Figure 7: Weekly Speed Pattern...")
    
    df = load_parquet_data()
    
    # Extract day of week
    df['dow'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    df['dow_name'] = pd.to_datetime(df['timestamp']).dt.day_name()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Box plot
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    sns.boxplot(data=df, x='dow_name', y='speed_kmh', order=day_order, ax=ax, palette='Set2')
    
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Speed (km/h)')
    ax.set_title('Traffic Speed Distribution by Day of Week')
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45)
    
    save_figure(fig, 'fig07_weekly_pattern')

def generate_fig8_spatial_correlation():
    """Figure 8: Traffic Flow Heatmap Over Time"""
    print("Generating Figure 8: Traffic Flow Heatmap...")
    
    df = load_parquet_data()
    
    # Create edge identifier
    df['edge_id'] = df['node_a_id'].astype(str) + '-' + df['node_b_id'].astype(str)
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    
    # Select top 20 busiest edges by average speed variance
    edge_variance = df.groupby('edge_id')['speed_kmh'].std().sort_values(ascending=False)
    top_edges = edge_variance.head(20).index
    
    df_top = df[df['edge_id'].isin(top_edges)]
    
    # Aggregate by hour and edge
    heatmap_data = df_top.pivot_table(index='edge_id', columns='hour', 
                                       values='speed_kmh', aggfunc='mean')
    
    # Sort by average speed
    heatmap_data = heatmap_data.loc[heatmap_data.mean(axis=1).sort_values(ascending=False).index]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Heatmap with better color scheme
    sns.heatmap(heatmap_data, cmap='RdYlGn', center=20, vmin=5, vmax=35,
                linewidths=0.5, cbar_kws={'label': 'Speed (km/h)'}, ax=ax)
    
    ax.set_title('Traffic Speed Patterns: Top 20 Dynamic Edges by Hour', fontsize=14, fontweight='bold')
    ax.set_xlabel('Hour of Day', fontsize=11)
    ax.set_ylabel('Edge ID', fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=8, rotation=0)
    
    # Add rush hour annotations
    ax.axvline(x=7, color='blue', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.axvline(x=17, color='blue', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.text(7, -1.5, 'Morning\nRush', ha='center', fontsize=9, color='blue')
    ax.text(17, -1.5, 'Evening\nRush', ha='center', fontsize=9, color='blue')
    
    save_figure(fig, 'fig08_spatial_corr')

def generate_fig9_temp_speed():
    """Figure 9: Temperature vs Speed Scatter"""
    print("Generating Figure 9: Temperature vs Speed...")
    
    df = load_parquet_data()
    
    # Sample data for plotting (too many points)
    df_sample = df.sample(n=min(10000, len(df)), random_state=42)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    ax.scatter(df_sample['temperature_c'], df_sample['speed_kmh'], alpha=0.3, s=10)
    
    # Regression line
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(df['temperature_c'], df['speed_kmh'])
    x_range = np.linspace(df['temperature_c'].min(), df['temperature_c'].max(), 100)
    ax.plot(x_range, slope * x_range + intercept, 'r-', linewidth=2, 
            label=f'Linear fit: R²={r_value**2:.3f}')
    
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Speed (km/h)')
    ax.set_title('Traffic Speed vs Temperature')
    ax.legend()
    ax.grid(alpha=0.3)
    
    save_figure(fig, 'fig09_temp_speed')

def generate_fig10_weather_box():
    """Figure 10: Speed by Weather Condition"""
    print("Generating Figure 10: Weather Box Plots...")
    
    df = load_parquet_data()
    
    # Categorize weather
    def categorize_weather(precip):
        if precip == 0:
            return 'Clear'
        elif precip <= 5:
            return 'Light Rain'
        else:
            return 'Heavy Rain'
    
    df['weather_category'] = df['precipitation_mm'].apply(categorize_weather)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Box plot
    weather_order = ['Clear', 'Light Rain', 'Heavy Rain']
    sns.boxplot(data=df, x='weather_category', y='speed_kmh', order=weather_order, ax=ax, palette='Blues')
    
    ax.set_xlabel('Weather Condition')
    ax.set_ylabel('Speed (km/h)')
    ax.set_title('Traffic Speed Under Different Weather Conditions')
    ax.grid(axis='y', alpha=0.3)
    
    save_figure(fig, 'fig10_weather_box')

def main():
    """Generate all EDA figures"""
    print(f"Output directory: {FIGURE_DIR}\n")
    
    generate_fig5_eda_speed_hist()
    generate_fig6_hourly_pattern()
    generate_fig7_weekly_pattern()
    generate_fig8_spatial_correlation()
    generate_fig9_temp_speed()
    generate_fig10_weather_box()
    
    print(f"\nAll EDA figures generated in: {FIGURE_DIR}")

if __name__ == "__main__":
    main()
