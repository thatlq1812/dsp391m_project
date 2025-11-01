"""
Page 2: Data Augmentation
Control and monitor data augmentation pipeline
"""

import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import json
import numpy as np
import os

from dashboard.utils.command_blocks import show_command_block
from traffic_forecast.utils.conda import resolve_conda_executable

st.set_page_config(page_title="Data Augmentation", page_icon="", layout="wide")

st.title("Data Augmentation")
st.markdown("Configure and run data augmentation strategies")

PROJECT_ROOT = Path(__file__).parent.parent.parent


def _conda_run_args(script: str) -> list[str]:
    """Build the conda run invocation for the dsp environment."""
    env_name = os.environ.get("CONDA_ENV", "dsp")
    return [
        resolve_conda_executable(),
        "run",
        "-n",
        env_name,
        "python",
        script,
    ]
CONFIG_PATH = PROJECT_ROOT / "configs" / "augmentation_config.json"

# Load or create default config
DEFAULT_CONFIG = {
    "basic": {
        "noise_std_speed": 2.0,
        "noise_std_weather": 0.1,
        "interpolation_steps": 2,
        "gmm_samples_per_run": 10,
        "target_multiplier": 20
    },
    "extreme": {
        "noise_std_speed": 3.0,
        "noise_std_weather": 0.2,
        "interpolation_steps": 3,
        "gmm_samples_per_run": 20,
        "target_multiplier": 45
    }
}

if CONFIG_PATH.exists():
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
else:
    config = DEFAULT_CONFIG

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["Configuration", "Strategy Comparison", "Quality Validation", "Run Augmentation"])

with tab1:
    st.markdown("### Augmentation Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Basic Augmentation")
        
        config['basic']['noise_std_speed'] = st.slider(
            "Speed Noise (km/h)",
            min_value=0.5,
            max_value=5.0,
            value=float(config['basic']['noise_std_speed']),
            step=0.1,
            key="basic_noise_speed"
        )
        
        config['basic']['noise_std_weather'] = st.slider(
            "Weather Noise",
            min_value=0.05,
            max_value=0.5,
            value=float(config['basic']['noise_std_weather']),
            step=0.05,
            key="basic_noise_weather"
        )
        
        config['basic']['interpolation_steps'] = st.slider(
            "Interpolation Steps",
            min_value=1,
            max_value=5,
            value=int(config['basic']['interpolation_steps']),
            key="basic_interp"
        )
        
        config['basic']['gmm_samples_per_run'] = st.slider(
            "GMM Samples per Run",
            min_value=5,
            max_value=30,
            value=int(config['basic']['gmm_samples_per_run']),
            key="basic_gmm"
        )
        
        config['basic']['target_multiplier'] = st.slider(
            "Target Multiplication Factor",
            min_value=10,
            max_value=30,
            value=int(config['basic']['target_multiplier']),
            key="basic_mult"
        )
        
        st.info(f"Expected output: ~{38 * config['basic']['target_multiplier']} runs")
    
    with col2:
        st.markdown("#### Extreme Augmentation")
        
        config['extreme']['noise_std_speed'] = st.slider(
            "Speed Noise (km/h)",
            min_value=1.0,
            max_value=10.0,
            value=float(config['extreme']['noise_std_speed']),
            step=0.5,
            key="extreme_noise_speed"
        )
        
        config['extreme']['noise_std_weather'] = st.slider(
            "Weather Noise",
            min_value=0.1,
            max_value=1.0,
            value=float(config['extreme']['noise_std_weather']),
            step=0.1,
            key="extreme_noise_weather"
        )
        
        config['extreme']['interpolation_steps'] = st.slider(
            "Interpolation Steps",
            min_value=2,
            max_value=10,
            value=int(config['extreme']['interpolation_steps']),
            key="extreme_interp"
        )
        
        config['extreme']['gmm_samples_per_run'] = st.slider(
            "GMM Samples per Run",
            min_value=10,
            max_value=50,
            value=int(config['extreme']['gmm_samples_per_run']),
            key="extreme_gmm"
        )
        
        config['extreme']['target_multiplier'] = st.slider(
            "Target Multiplication Factor",
            min_value=30,
            max_value=60,
            value=int(config['extreme']['target_multiplier']),
            key="extreme_mult"
        )
        
        st.info(f"Expected output: ~{38 * config['extreme']['target_multiplier']} runs")
    
    st.divider()
    
    if st.button("Save Configuration", width='stretch'):
        CONFIG_PATH.parent.mkdir(exist_ok=True)
        with open(CONFIG_PATH, 'w') as f:
            json.dump(config, f, indent=2)
        st.success(f"Configuration saved to `{CONFIG_PATH}`")

with tab2:
    st.markdown("### Strategy Comparison")
    
    basic_parquet = PROJECT_ROOT / "data" / "processed" / "all_runs_augmented.parquet"
    extreme_parquet = PROJECT_ROOT / "data" / "processed" / "all_runs_extreme_augmented.parquet"
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Basic Augmentation (23.4x)")
        if basic_parquet.exists():
            df_basic = pd.read_parquet(basic_parquet)
            
            st.metric("Total Runs", df_basic['run_id'].nunique() if 'run_id' in df_basic.columns else len(df_basic))
            st.metric("Total Records", f"{len(df_basic):,}")
            st.metric("File Size", f"{basic_parquet.stat().st_size / 1024 / 1024:.1f} MB")
            
            if 'speed_kmh' in df_basic.columns:
                st.markdown("**Speed Statistics:**")
                st.write(f"- Mean: {df_basic['speed_kmh'].mean():.2f} km/h")
                st.write(f"- Std: {df_basic['speed_kmh'].std():.2f} km/h")
                st.write(f"- Min: {df_basic['speed_kmh'].min():.2f} km/h")
                st.write(f"- Max: {df_basic['speed_kmh'].max():.2f} km/h")
        else:
            st.warning("Basic augmented data not found")
    
    with col2:
        st.markdown("#### Extreme Augmentation (48.4x)")
        if extreme_parquet.exists():
            df_extreme = pd.read_parquet(extreme_parquet)
            
            st.metric("Total Runs", df_extreme['run_id'].nunique() if 'run_id' in df_extreme.columns else len(df_extreme))
            st.metric("Total Records", f"{len(df_extreme):,}")
            st.metric("File Size", f"{extreme_parquet.stat().st_size / 1024 / 1024:.1f} MB")
            
            if 'speed_kmh' in df_extreme.columns:
                st.markdown("**Speed Statistics:**")
                st.write(f"- Mean: {df_extreme['speed_kmh'].mean():.2f} km/h")
                st.write(f"- Std: {df_extreme['speed_kmh'].std():.2f} km/h")
                st.write(f"- Min: {df_extreme['speed_kmh'].min():.2f} km/h")
                st.write(f"- Max: {df_extreme['speed_kmh'].max():.2f} km/h")
        else:
            st.warning("Extreme augmented data not found")
    
    st.divider()
    
    # Distribution comparison
    if basic_parquet.exists() and extreme_parquet.exists():
        st.markdown("#### Speed Distribution Comparison")
        
        df_basic_sample = df_basic.sample(min(10000, len(df_basic)))
        df_extreme_sample = df_extreme.sample(min(10000, len(df_extreme)))
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=df_basic_sample['speed_kmh'],
            name="Basic (23.4x)",
            opacity=0.6,
            nbinsx=50,
            marker_color='#1f77b4'
        ))
        
        fig.add_trace(go.Histogram(
            x=df_extreme_sample['speed_kmh'],
            name="Extreme (48.4x)",
            opacity=0.6,
            nbinsx=50,
            marker_color='#ff7f0e'
        ))
        
        fig.update_layout(
            barmode='overlay',
            xaxis_title="Speed (km/h)",
            yaxis_title="Frequency",
            height=400
        )
        
        st.plotly_chart(fig, width='stretch')

with tab3:
    st.markdown("### Quality Validation")
    
    st.info("""
    Quality metrics ensure augmented data maintains statistical properties of original data:
    - **KS Test**: Kolmogorov-Smirnov test for distribution similarity (p > 0.05 is good)
    - **Correlation Preservation**: Maintains temporal and spatial correlations
    - **Statistical Moments**: Mean, variance, skewness, kurtosis preservation
    """)
    
    original_parquet = PROJECT_ROOT / "data" / "processed" / "all_runs_combined.parquet"
    
    if original_parquet.exists() and (basic_parquet.exists() or extreme_parquet.exists()):
        df_original = pd.read_parquet(original_parquet)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original Data Stats")
            if 'speed_kmh' in df_original.columns:
                st.write(f"Mean: {df_original['speed_kmh'].mean():.2f} km/h")
                st.write(f"Std: {df_original['speed_kmh'].std():.2f} km/h")
                st.write(f"Skewness: {df_original['speed_kmh'].skew():.3f}")
                st.write(f"Kurtosis: {df_original['speed_kmh'].kurtosis():.3f}")
        
        with col2:
            dataset_choice = st.radio("Compare with:", ["Basic Augmented", "Extreme Augmented"])
            
            if dataset_choice == "Basic Augmented" and basic_parquet.exists():
                df_aug = pd.read_parquet(basic_parquet)
            elif dataset_choice == "Extreme Augmented" and extreme_parquet.exists():
                df_aug = pd.read_parquet(extreme_parquet)
            else:
                df_aug = None
            
            if df_aug is not None and 'speed_kmh' in df_aug.columns:
                st.write(f"Mean: {df_aug['speed_kmh'].mean():.2f} km/h")
                st.write(f"Std: {df_aug['speed_kmh'].std():.2f} km/h")
                st.write(f"Skewness: {df_aug['speed_kmh'].skew():.3f}")
                st.write(f"Kurtosis: {df_aug['speed_kmh'].kurtosis():.3f}")
        
        st.divider()
        
        # KS Test simulation
        st.markdown("#### Distribution Similarity Test")
        
        from scipy import stats
        
        if df_aug is not None:
            ks_stat, p_value = stats.ks_2samp(
                df_original['speed_kmh'].sample(min(1000, len(df_original))),
                df_aug['speed_kmh'].sample(min(1000, len(df_aug)))
            )
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("KS Statistic", f"{ks_stat:.4f}")
            
            with col2:
                st.metric("P-Value", f"{p_value:.4f}")
            
            with col3:
                if p_value > 0.05:
                    st.success("PASS (p > 0.05)")
                else:
                    st.warning("WARNING MARGINAL")
    
    else:
        st.warning("Need both original and augmented data for validation")

with tab4:
    st.markdown("### Run Augmentation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Basic Augmentation (23.4x)")
        st.info("Applies moderate noise and interpolation for balanced augmentation")
        
        if st.button("Prepare Basic Augmentation Command", width='stretch', type="primary"):
            st.warning("Estimated runtime: 5-10 minutes. Execute the command from your terminal.")
            show_command_block(
                _conda_run_args("scripts/data/augment_data_advanced.py"),
                cwd=PROJECT_ROOT,
                description="Run the command below to launch basic augmentation:",
                success_hint="Progress logs will stream in the terminal window.",
            )
            st.success("Command prepared. Copy it into a terminal to start augmentation.")
    
    with col2:
        st.markdown("#### Extreme Augmentation (48.4x)")
        st.info("Applies aggressive augmentation for maximum data diversity")
        
        if st.button("Prepare Extreme Augmentation Command", width='stretch', type="primary"):
            st.warning("Estimated runtime: 10-20 minutes. Execute the command from your terminal.")
            show_command_block(
                _conda_run_args("scripts/data/augment_extreme.py"),
                cwd=PROJECT_ROOT,
                description="Run the command below to launch extreme augmentation:",
                success_hint="Keep the terminal open until augmentation finishes.",
            )
            st.success("Command prepared. Copy it into a terminal to start augmentation.")
    
    st.divider()
    
    st.markdown("#### Rebuild Parquet Files")
    st.info("Parquet files are created automatically during augmentation")
    
    if st.button("Rebuild All Parquet Files", width='stretch', disabled=True):
        st.info("This feature is not needed - parquet files are built automatically")

# Footer
st.divider()
st.caption("Tip: Use extreme augmentation for best training results (48.4x multiplication)")
