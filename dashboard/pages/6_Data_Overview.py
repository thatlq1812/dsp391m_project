"""
Page 1: Data Overview
Monitor data collections and augmentation status
"""

import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
from dashboard.utils.command_blocks import show_command_block, show_command_list

st.set_page_config(page_title="Data Overview", page_icon="", layout="wide")

st.title("Data Overview")
st.markdown("Monitor data collections and augmentation pipeline")

PROJECT_ROOT = Path(__file__).parent.parent.parent

# Tabs
tab1, tab2, tab3 = st.tabs(["Data Collections", "Statistics", "Data Management"])

with tab1:
    st.markdown("### Data Collection Status")
    
    col1, col2, col3 = st.columns(3)
    
    # Check original runs
    raw_runs_dir = PROJECT_ROOT / "data" / "runs"
    original_parquet = PROJECT_ROOT / "data" / "processed" / "all_runs_combined.parquet"
    augmented_parquet = PROJECT_ROOT / "data" / "processed" / "all_runs_augmented.parquet"
    extreme_parquet = PROJECT_ROOT / "data" / "processed" / "all_runs_extreme_augmented.parquet"
    
    with col1:
        st.markdown("#### Original Collections")
        if raw_runs_dir.exists():
            run_dirs = [d for d in raw_runs_dir.iterdir() if d.is_dir()]
            st.metric("Total Runs", len(run_dirs))
            
            if run_dirs:
                latest_run = max(run_dirs, key=lambda x: x.stat().st_mtime)
                st.caption(f"Latest: {latest_run.name}")
        else:
            st.metric("Total Runs", 0)
            st.warning("No raw data directory found")
    
    with col2:
        st.markdown("#### Basic Augmentation")
        if augmented_parquet.exists():
            df = pd.read_parquet(augmented_parquet)
            unique_runs = df['run_id'].nunique() if 'run_id' in df.columns else len(df)
            st.metric("Augmented Runs", unique_runs, delta="23.4x")
            st.caption(f"Size: {augmented_parquet.stat().st_size / 1024 / 1024:.1f} MB")
        else:
            st.metric("Augmented Runs", 0)
            st.warning("Run augmentation first")
    
    with col3:
        st.markdown("#### Extreme Augmentation")
        if extreme_parquet.exists():
            df = pd.read_parquet(extreme_parquet)
            unique_runs = df['run_id'].nunique() if 'run_id' in df.columns else len(df)
            st.metric("Extreme Runs", unique_runs, delta="48.4x")
            st.caption(f"Size: {extreme_parquet.stat().st_size / 1024 / 1024:.1f} MB")
        else:
            st.metric("Extreme Runs", 0)
            st.warning("Run extreme augmentation")
    
    st.divider()
    
    # Data Summary Table
    st.markdown("### Data Files Summary")
    
    data_files = [
        ("Original Combined", original_parquet),
        ("Basic Augmented", augmented_parquet),
        ("Extreme Augmented", extreme_parquet)
    ]
    
    summary_data = []
    for name, path in data_files:
        if path.exists():
            df = pd.read_parquet(path)
            summary_data.append({
                "Dataset": name,
                "Status": "Available",
                "Rows": len(df),
                "Columns": len(df.columns),
                "Size (MB)": round(path.stat().st_size / 1024 / 1024, 1),
                "Modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
            })
        else:
            summary_data.append({
                "Dataset": name,
                "Status": "WARNING Not Found",
                "Rows": None,
                "Columns": None,
                "Size (MB)": None,
                "Modified": "-"
            })
    
    # Convert to DataFrame with proper dtypes
    summary_df = pd.DataFrame(summary_data)
    summary_df["Rows"] = summary_df["Rows"].astype("Int64")  # Nullable integer
    summary_df["Columns"] = summary_df["Columns"].astype("Int64")  # Nullable integer
    summary_df["Size (MB)"] = summary_df["Size (MB)"].astype("Float64")  # Nullable float
    
    st.dataframe(summary_df, width='stretch', hide_index=True)

with tab2:
    st.markdown("### Data Statistics")
    
    # Select dataset
    dataset_option = st.selectbox(
        "Select Dataset",
        ["Original Combined", "Basic Augmented", "Extreme Augmented"],
        index=2
    )
    
    dataset_map = {
        "Original Combined": original_parquet,
        "Basic Augmented": augmented_parquet,
        "Extreme Augmented": extreme_parquet
    }
    
    selected_path = dataset_map[dataset_option]
    
    if selected_path.exists():
        df = pd.read_parquet(selected_path)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        
        with col2:
            if 'run_id' in df.columns:
                st.metric("Unique Runs", df['run_id'].nunique())
            else:
                st.metric("Unique Runs", "-")
        
        with col3:
            if 'edge_id' in df.columns:
                st.metric("Unique Edges", df['edge_id'].nunique())
            else:
                st.metric("Unique Edges", "-")
        
        with col4:
            if 'timestamp' in df.columns:
                time_span = (df['timestamp'].max() - df['timestamp'].min()).days
                st.metric("Time Span (days)", time_span)
            else:
                st.metric("Time Span", "-")
        
        st.divider()
        
        # Speed distribution
        if 'speed_kmh' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Speed Distribution")
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=df['speed_kmh'],
                    nbinsx=50,
                    name="Speed",
                    marker_color='#1f77b4'
                ))
                fig.update_layout(
                    xaxis_title="Speed (km/h)",
                    yaxis_title="Frequency",
                    height=400,
                    showlegend=False
                )
                st.plotly_chart(fig, width='stretch')
            
            with col2:
                st.markdown("#### Speed Statistics")
                speed_stats = df['speed_kmh'].describe()
                st.dataframe({
                    "Metric": ["Count", "Mean", "Std", "Min", "25%", "50%", "75%", "Max"],
                    "Value": [
                        f"{speed_stats['count']:.0f}",
                        f"{speed_stats['mean']:.2f} km/h",
                        f"{speed_stats['std']:.2f} km/h",
                        f"{speed_stats['min']:.2f} km/h",
                        f"{speed_stats['25%']:.2f} km/h",
                        f"{speed_stats['50%']:.2f} km/h",
                        f"{speed_stats['75%']:.2f} km/h",
                        f"{speed_stats['max']:.2f} km/h"
                    ]
                }, hide_index=True, width='stretch')
        
        # Temporal patterns
        if 'timestamp' in df.columns and 'speed_kmh' in df.columns:
            st.markdown("#### Hourly Traffic Patterns")
            df_temp = df.copy()
            df_temp['hour'] = pd.to_datetime(df_temp['timestamp']).dt.hour
            hourly_avg = df_temp.groupby('hour')['speed_kmh'].mean().reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hourly_avg['hour'],
                y=hourly_avg['speed_kmh'],
                mode='lines+markers',
                name='Avg Speed',
                line=dict(color='#ff7f0e', width=3),
                marker=dict(size=8)
            ))
            fig.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Average Speed (km/h)",
                height=400,
                xaxis=dict(tickmode='linear', tick0=0, dtick=2)
            )
            st.plotly_chart(fig, width='stretch')
    
    else:
        st.warning(f"Dataset '{dataset_option}' not found. Please run data collection/augmentation first.")

with tab3:
    st.markdown("### Data Management Tools")
    
    st.markdown("#### Data Processing")
    
    st.info("""
    **Combine Runs**: Merge multiple collection runs into single parquet file
    
    **Rebuild Cache**: Regenerate adjacency matrix and topology cache
    """)
    
    if st.button("Combine All Runs", width='stretch', type="primary"):
        show_command_list(
            [
                [
                    "conda",
                    "run",
                    "-n",
                    "dsp",
                    "--no-capture-output",
                    "python",
                    "scripts/data/combine_runs.py",
                ],
                [
                    "python",
                    "scripts/data/combine_runs.py",
                ],
            ],
            description=(
                "Run one of the commands below to merge all collection runs into a parquet file. "
                "Prefer the Conda variant; the direct Python command is provided as a fallback."
            ),
            cwd=PROJECT_ROOT,
        )
        st.success("Command prepared. Execute it manually to start the combination job.")
    
    st.divider()
    
    # Check if cache rebuild script exists
    cache_rebuild_script = PROJECT_ROOT / "scripts" / "data" / "rebuild_cache.py"
    cache_exists = cache_rebuild_script.exists()
    
    if st.button("Rebuild Cache", width='stretch', disabled=not cache_exists):
        if cache_exists:
            show_command_block(
                [
                    "conda",
                    "run",
                    "-n",
                    "dsp",
                    "--no-capture-output",
                    "python",
                    str(cache_rebuild_script),
                ],
                cwd=PROJECT_ROOT,
                description="Run the command below to regenerate the cached adjacency matrix and topology data:",
            )
            st.success("Command prepared. Execute it manually to rebuild the cache.")
        else:
            st.warning("Cache rebuild script not found at: scripts/data/rebuild_cache.py")

# Footer
st.divider()
st.caption("Tip: Use extreme augmented data (1,839 runs) for best training results")
