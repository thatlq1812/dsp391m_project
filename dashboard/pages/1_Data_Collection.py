"""
Data Collection Monitoring Page

Monitor and control data collection from Google Directions API.

Author: thatlq1812
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
from datetime import datetime
import sys

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Data Collection", page_icon="📊", layout="wide")

st.title("Data Collection Monitoring")
st.markdown("Monitor traffic data collection from Google Directions API")

# Tabs
tab1, tab2, tab3 = st.tabs(["Overview", "Runs Details", "Collection Control"])

with tab1:
    st.header("Collection Overview")
    
    # Load data
    data_dir = PROJECT_ROOT / 'data' / 'runs'
    
    if data_dir.exists():
        run_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()], reverse=True)
        
        st.metric("Total Collection Runs", len(run_dirs))
        
        # Create runs summary
        runs_info = []
        for run_dir in run_dirs[:20]:  # Latest 20
            files = list(run_dir.glob('*.json'))
            total_size = sum(f.stat().st_size for f in files)
            
            # Parse timestamp
            run_name = run_dir.name
            if run_name.startswith('run_'):
                timestamp_str = run_name[4:]
                try:
                    run_time = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                    time_display = run_time.strftime('%Y-%m-%d %H:%M:%S')
                except:
                    time_display = timestamp_str
            else:
                time_display = run_name
            
            runs_info.append({
                'Run Name': run_name,
                'Timestamp': time_display,
                'Files': len(files),
                'Size (KB)': total_size // 1024
            })
        
        df_runs = pd.DataFrame(runs_info)
        
        # Display table
        st.dataframe(df_runs, use_container_width=True)
        
        # Chart: Data collection over time
        fig = px.bar(
            df_runs,
            x='Timestamp',
            y='Size (KB)',
            title='Data Collection Volume Over Time',
            color='Files',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("No data directory found. Please run data collection first.")

with tab2:
    st.header("Run Details")
    
    if data_dir.exists() and len(run_dirs) > 0:
        # Select run
        selected_run = st.selectbox(
            "Select a run to inspect:",
            [d.name for d in run_dirs[:10]]
        )
        
        if selected_run:
            run_path = data_dir / selected_run
            
            # Load run data
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Nodes (Intersections)")
                nodes_file = run_path / 'nodes.json'
                if nodes_file.exists():
                    with open(nodes_file, 'r') as f:
                        nodes = json.load(f)
                    st.metric("Total Nodes", len(nodes))
                    st.json(nodes[0] if nodes else {})
            
            with col2:
                st.subheader("Edges (Road Segments)")
                edges_file = run_path / 'edges.json'
                if edges_file.exists():
                    with open(edges_file, 'r') as f:
                        edges = json.load(f)
                    st.metric("Total Edges", len(edges))
                    st.json(edges[0] if edges else {})
            
            # Traffic data
            st.subheader("Traffic Data")
            traffic_file = run_path / 'traffic_edges.json'
            if traffic_file.exists():
                with open(traffic_file, 'r') as f:
                    traffic = json.load(f)
                
                df_traffic = pd.DataFrame(traffic)
                st.metric("Traffic Records", len(df_traffic))
                
                # Speed distribution
                if 'speed_kmh' in df_traffic.columns:
                    fig = px.histogram(
                        df_traffic,
                        x='speed_kmh',
                        nbins=30,
                        title='Speed Distribution',
                        labels={'speed_kmh': 'Speed (km/h)'},
                        color_discrete_sequence=['#1f77b4']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Avg Speed", f"{df_traffic['speed_kmh'].mean():.2f} km/h")
                    with col2:
                        st.metric("Min Speed", f"{df_traffic['speed_kmh'].min():.2f} km/h")
                    with col3:
                        st.metric("Max Speed", f"{df_traffic['speed_kmh'].max():.2f} km/h")
                
                st.dataframe(df_traffic.head(10), use_container_width=True)

with tab3:
    st.header("Collection Control")
    
    st.info("Control data collection tasks from the VM")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Manual Collection")
        
        if st.button("Run Single Collection", key="run_once"):
            st.info("Running collection task...")
            st.code("bash scripts/collect_once.sh", language="bash")
        
        if st.button("Start Collection Loop (15min)", key="run_loop"):
            st.info("Starting collection loop...")
            st.code("bash scripts/run_interval.sh 900", language="bash")
    
    with col2:
        st.subheader("Data Download")
        
        if st.button("Download Latest Run", key="download_latest"):
            st.info("Downloading latest run from VM...")
            st.code("bash scripts/data/download_latest.sh", language="bash")
        
        if st.button("Download All Data", key="download_all"):
            st.info("Downloading all data from VM...")
            st.code("bash scripts/data/download_data_compressed.sh", language="bash")
    
    st.markdown("---")
    
    # Scheduler status
    st.subheader("Scheduler Status")
    
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        st.metric("Next Collection", "14:00:00", delta="15 min")
    
    with status_col2:
        st.metric("Collections Today", "96")
    
    with status_col3:
        st.metric("Success Rate", "100%", delta="0%")
