"""
Common utilities for visualization scripts

Functions for loading data, styling plots, and saving figures.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path

# Set publication-quality styling
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

FIGURE_DIR = Path(__file__).parents[2] / "docs" / "final_report" / "figures"
FIGURE_DIR.mkdir(exist_ok=True, parents=True)

# Consistent figure settings
FIGURE_CONFIG = {
    'figsize': (10, 6),  # Used in plt.subplots(), not savefig()
    'dpi': 300,
    'format': 'png',  # Changed from PDF to PNG for better compatibility
    'bbox_inches': 'tight',
    'transparent': False,
}

def save_figure(fig, filename, **kwargs):
    """
    Save figure with consistent settings
    
    Args:
        fig: Matplotlib figure object
        filename: Output filename (without extension)
        **kwargs: Override default FIGURE_CONFIG settings
    """
    config = FIGURE_CONFIG.copy()
    config.update(kwargs)
    
    # Extract savefig parameters (exclude 'figsize' which is for plt.subplots)
    savefig_params = {
        'dpi': config['dpi'],
        'bbox_inches': config['bbox_inches'],
        'transparent': config['transparent']
    }
    
    output_path = FIGURE_DIR / f"{filename}.{config['format']}"
    fig.savefig(output_path, **savefig_params)
    print(f"Saved: {output_path}")
    plt.close(fig)

def load_parquet_data(filename="baseline_1month.parquet"):
    """Load processed traffic data"""
    data_path = Path(__file__).parents[2] / "data" / "processed" / filename
    if not data_path.exists():
        # Fallback to augmented if baseline not found
        fallback = Path(__file__).parents[2] / "data" / "processed" / "augmented_1year.parquet"
        if fallback.exists():
            print(f"Warning: {filename} not found, using augmented_1year.parquet")
            data_path = fallback
    return pd.read_parquet(data_path)

def load_model_results(model_dir):
    """Load model training results"""
    import json
    results_path = Path(__file__).parents[2] / "outputs" / model_dir / "test_results.json"
    with open(results_path) as f:
        return json.load(f)

def set_plot_style():
    """Set consistent plot styling"""
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'font.family': 'serif',
    })

set_plot_style()
