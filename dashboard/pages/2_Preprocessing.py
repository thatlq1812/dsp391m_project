"""
Data Preprocessing Control Page

Control data preprocessing pipeline: JSON → Parquet conversion + feature engineering.

Author: thatlq1812
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import sys

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from traffic_forecast.core import PipelineConfig, DataManager

st.set_page_config(page_title="Preprocessing", page_icon="⚙️", layout="wide")

st.title("Data Preprocessing Control")
st.markdown("Convert raw JSON data to processed Parquet files with features")

# Initialize session state
if 'preprocessing_done' not in st.session_state:
    st.session_state.preprocessing_done = False
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = None

# Configuration sidebar
with st.sidebar:
    st.header("Preprocessing Config")
    
    max_runs = st.number_input(
        "Max Runs to Process",
        min_value=1,
        max_value=100,
        value=10,
        help="Limit number of runs (None = all)"
    )
    
    enable_feature_engineering = st.checkbox(
        "Enable Feature Engineering",
        value=True,
        help="Create lag features, rolling stats, etc."
    )
    
    save_output = st.checkbox(
        "Save Processed Data",
        value=True,
        help="Save to data/processed/"
    )

# Main content
tab1, tab2, tab3 = st.tabs(["Run Preprocessing", "Data Preview", "Feature Analysis"])

with tab1:
    st.header("Run Preprocessing Pipeline")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Pipeline Steps:
        
        1. **Load Data** - Read JSON files from `data/runs/`
        2. **Parse Timestamps** - Convert to datetime objects
        3. **Add Time Features** - Hour, day of week, weekend flag
        4. **Add Congestion Levels** - Based on speed thresholds
        5. **Engineer Features** - Lag features, rolling stats, cyclical encoding
        6. **Save Output** - Export to Parquet format
        """)
    
    with col2:
        st.info(f"""
        **Current Settings:**
        - Max Runs: {max_runs if max_runs else 'All'}
        - Feature Engineering: {'Yes' if enable_feature_engineering else 'No'}
        - Save Output: {'Yes' if save_output else 'No'}
        """)
    
    if st.button("Start Preprocessing", type="primary", use_container_width=True):
        # Create config
        config = PipelineConfig()
        config.max_runs = max_runs
        config.enable_feature_engineering = enable_feature_engineering
        config.enable_save_models = save_output
        
        # Create data manager
        data_manager = DataManager(config)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Load data
            status_text.text("Loading raw data...")
            progress_bar.progress(20)
            data_manager.load_data(max_runs=max_runs)
            st.success(f"Loaded {len(data_manager.data_raw):,} records")
            
            # Step 2: Preprocess
            status_text.text("Preprocessing data...")
            progress_bar.progress(40)
            data_manager.preprocess_data()
            st.success(f"Preprocessed {len(data_manager.data_processed):,} records")
            
            # Step 3: Feature engineering
            if enable_feature_engineering:
                status_text.text("Engineering features...")
                progress_bar.progress(60)
                data_manager.engineer_features()
                st.success(f"Created {data_manager.data_features.shape[1]} features")
            
            # Step 4: Save
            if save_output:
                status_text.text("Saving processed data...")
                progress_bar.progress(80)
                output_path = data_manager.save_processed_data()
                st.success(f"Saved to {output_path}")
            
            progress_bar.progress(100)
            status_text.text("Preprocessing complete!")
            
            # Store in session
            st.session_state.data_manager = data_manager
            st.session_state.preprocessing_done = True
            
            st.balloons()
            
        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)

with tab2:
    st.header("Data Preview")
    
    if st.session_state.preprocessing_done and st.session_state.data_manager:
        dm = st.session_state.data_manager
        
        # Show raw data
        st.subheader("Raw Data")
        st.dataframe(dm.data_raw.head(10), use_container_width=True)
        
        st.metric("Total Raw Records", f"{len(dm.data_raw):,}")
        
        # Show processed data
        if dm.data_processed is not None:
            st.subheader("Processed Data")
            st.dataframe(dm.data_processed.head(10), use_container_width=True)
            
            st.metric("Total Processed Records", f"{len(dm.data_processed):,}")
        
        # Show feature data
        if dm.data_features is not None:
            st.subheader("Feature-Engineered Data")
            st.dataframe(dm.data_features.head(10), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Features", dm.data_features.shape[1])
            with col2:
                st.metric("Total Records", f"{len(dm.data_features):,}")
            with col3:
                st.metric("Memory Usage", f"{dm.data_features.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    else:
        st.info("Run preprocessing first to preview data")

with tab3:
    st.header("Feature Analysis")
    
    if st.session_state.preprocessing_done and st.session_state.data_manager:
        dm = st.session_state.data_manager
        
        if dm.data_features is not None:
            df = dm.data_features
            
            # Feature groups
            st.subheader("Feature Groups")
            
            feature_groups = {
                'Time Features': ['hour', 'minute', 'day_of_week', 'is_weekend', 
                                 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos'],
                'Traffic Features': ['speed_kmh', 'duration_sec', 'distance_km', 'congestion_level'],
                'Lag Features': ['speed_lag_1', 'speed_lag_2', 'speed_lag_3'],
                'Rolling Features': ['speed_rolling_mean_3', 'speed_rolling_std_3'],
                'Rush Hour Features': ['is_morning_rush', 'is_evening_rush', 'is_rush_hour']
            }
            
            for group_name, features in feature_groups.items():
                with st.expander(f"{group_name}", expanded=False):
                    available = [f for f in features if f in df.columns]
                    if available:
                        st.write(f"**Available:** {', '.join(available)}")
                        
                        # Show stats
                        numeric_features = [f for f in available if df[f].dtype in ['int64', 'float64']]
                        if numeric_features:
                            st.dataframe(df[numeric_features].describe(), use_container_width=True)
                    else:
                        st.warning("No features from this group available")
            
            # Correlation heatmap
            st.subheader("Feature Correlations")
            
            # Select numeric columns
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != 'timestamp'][:15]  # Limit to 15
            
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    title='Feature Correlation Matrix'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Speed distribution
            st.subheader("Speed Distribution")
            
            fig = px.histogram(
                df,
                x='speed_kmh',
                nbins=50,
                title='Traffic Speed Distribution',
                labels={'speed_kmh': 'Speed (km/h)'},
                color_discrete_sequence=['#1f77b4']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("Run preprocessing first to analyze features")

# Status footer
st.markdown("---")
if st.session_state.preprocessing_done:
    st.success("Preprocessing completed successfully!")
else:
    st.info("Waiting for preprocessing to run...")
