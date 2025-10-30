"""
Traffic Forecasting Control Dashboard

Streamlit app for managing the complete traffic forecasting pipeline:
- Data collection monitoring
- Preprocessing control
- Model training & comparison
- Real-time predictions
- Performance analytics

Author: thatlq1812
Run: streamlit run dashboard/app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Page configuration
st.set_page_config(
    page_title="Traffic Forecasting Control Panel",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">Traffic Forecasting Control Panel</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deep Learning Pipeline Management | LSTM & ASTGCN</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/traffic-jam.png", width=100)
    st.title("Navigation")
    st.markdown("---")
    
    st.markdown("""
    ### Pages
    
    1. **Home** - Dashboard overview
    2. **Data Collection** - Monitor data collection
    3. **Preprocessing** - Data processing control
    4. **Model Training** - Train LSTM & ASTGCN
    5. **Predictions** - Real-time forecasting
    
    ### Quick Info
    
    - **Project:** DSP391m Traffic Forecasting
    - **Author:** thatlq1812
    - **Models:** LSTM, ASTGCN
    - **Location:** Ho Chi Minh City
    """)
    
    st.markdown("---")
    st.info("Navigate using the sidebar pages")

# Main content
st.markdown("## System Overview")

# Status cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>Data</h3>
        <h2>10</h2>
        <p>Collection Runs</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>Pipeline</h3>
        <h2>Ready</h2>
        <p>Status</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>Models</h3>
        <h2>2</h2>
        <p>DL Models</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="metric-card">
        <h3>Status</h3>
        <h2>Online</h2>
        <p>System Health</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Quick Actions
st.markdown("## Quick Actions")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Data Pipeline")
    if st.button("Download Latest Data", key="download"):
        st.info("Downloading latest data from VM...")
        # TODO: Implement download
    
    if st.button("Run Preprocessing", key="preprocess"):
        st.info("Starting preprocessing pipeline...")
        # TODO: Implement preprocessing

with col2:
    st.markdown("### Model Training")
    if st.button("Train LSTM Model", key="train_lstm"):
        st.info("Starting LSTM training...")
        # TODO: Implement LSTM training
    
    if st.button("Train ASTGCN Model", key="train_astgcn"):
        st.info("Starting ASTGCN training...")
        # TODO: Implement ASTGCN training

with col3:
    st.markdown("### Predictions")
    if st.button("View Latest Predictions", key="predictions"):
        st.info("Loading predictions...")
        # TODO: Implement predictions view
    
    if st.button("Model Comparison", key="compare"):
        st.info("Comparing models...")
        # TODO: Implement comparison

st.markdown("---")

# System Information
st.markdown("## System Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Data Status")
    st.json({
        "data_directory": "data/runs/",
        "total_runs": 10,
        "latest_run": "run_20251030_130114",
        "total_records": "1,440+",
        "date_range": "Oct 30, 2025"
    })

with col2:
    st.markdown("### Configuration")
    st.json({
        "models": ["LSTM", "ASTGCN"],
        "sequence_length": 12,
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 50
    })

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>Traffic Forecasting System v2.0 | Powered by Deep Learning (LSTM & ASTGCN)</p>
    <p>Built by thatlq1812 | DSP391m Project</p>
</div>
""", unsafe_allow_html=True)
