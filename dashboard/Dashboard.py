"""
STMGT Traffic Forecasting - Dashboard V4 (Central Control Hub)
Complete project management and ML operations dashboard
"""

import streamlit as st
from pathlib import Path
import json
import subprocess
import psutil
import platform
import numpy as np
import pandas as pd
import sys
import os
from datetime import datetime

# Add dashboard to path for imports
sys.path.append(str(Path(__file__).parent))
from realtime_stats import get_training_stats

PROJECT_ROOT = Path(__file__).parent.parent

# Page config
st.set_page_config(
    page_title="STMGT Central Hub",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .status-good {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">STMGT Central Control Hub</h1>', unsafe_allow_html=True)
st.markdown("**Complete Project Management & ML Operations Dashboard**")

# Sidebar info
with st.sidebar:
    st.markdown("### Dashboard V4")
    st.markdown("**Status:** Production Ready ")
    st.markdown("**Last Updated:** November 1, 2025")
    st.divider()
    
    st.markdown("### Navigation Guide")
    
    with st.expander("Infrastructure & DevOps", expanded=False):
        st.markdown("""
        - **1. System Overview** - Dashboard tổng quan
        - **2. VM Management** - GCP VM control
        - **3. Deployment** - Deploy & version control
        - **4. Monitoring & Logs** - System health & logs
        """)
    
    with st.expander("Data Pipeline", expanded=False):
        st.markdown("""
        - **5. Data Collection** - API collection control
        - **6. Data Overview** - Existing data stats
        - **7. Data Augmentation** - Augmentation pipeline
        """)
    
    with st.expander("ML Workflow", expanded=False):
        st.markdown("""
        - **8. Data Visualization** - Patterns & analysis
        - **9. Training Control** - Train & monitor
        - **10. Model Registry** - Version management
        """)
    
    with st.expander("Production", expanded=False):
        st.markdown("""
        - **11. Predictions** - Real-time inference
        - **12. API & Integration** - Endpoints & webhooks
        """)
    
    st.divider()
    
    # System info
    st.markdown("### System Info")
    st.caption(f"OS: {platform.system()}")
    st.caption(f"CPU: {psutil.cpu_count()} cores")
    st.caption(f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Architecture", "Navigation", "Quick Stats"])

with tab1:
    st.markdown("## System Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        # Count actual nodes
        node_count = 0
        nodes_file = PROJECT_ROOT / "data" / "nodes.json"
        topology_file = PROJECT_ROOT / "cache" / "overpass_topology.json"
        try:
            if nodes_file.exists():
                with open(nodes_file, "r", encoding="utf-8") as f:
                    nodes = json.load(f)
                    node_count = len(nodes)
            elif topology_file.exists():
                with open(topology_file, "r", encoding="utf-8") as f:
                    topology = json.load(f)
                    if isinstance(topology, dict):
                        if "nodes" in topology and isinstance(topology["nodes"], list):
                            node_count = len(topology["nodes"])
                        elif "metadata" in topology:
                            node_count = int(topology["metadata"].get("total_nodes", 0))
        except Exception:
            node_count = 0
        st.metric("Total Nodes", f"{node_count:,}", help="Traffic network nodes")
    
    with col2:
        # Count edges from adjacency matrix
        adj_file = PROJECT_ROOT / "cache" / "adjacency_matrix.npy"
        try:
            if adj_file.exists():
                adj_matrix = np.load(adj_file)
                edge_count = int(np.sum(adj_matrix > 0) / 2)  # Undirected edges
            else:
                edge_count = 0
        except:
            edge_count = 0
        st.metric("Total Edges", f"{edge_count:,}", help="Directional road segments")
    
    with col3:
        # Count data runs from processed files
        processed_dir = PROJECT_ROOT / "data" / "processed"
        try:
            if processed_dir.exists():
                parquet_files = list(processed_dir.glob("*.parquet"))
                total_runs = 0
                for pq_file in parquet_files:
                    try:
                        df = pd.read_parquet(pq_file)
                        if 'run_id' in df.columns:
                            total_runs += df['run_id'].nunique()
                    except:
                        pass
                run_count = total_runs
            else:
                run_count = 0
        except:
            run_count = 0
        st.metric("Data Runs", f"{run_count:,}", help="Total data collection runs")
    
    with col4:
        # Get realtime training stats
        training_stats = get_training_stats()
        st.metric("Best MAE", training_stats["best_mae"], delta="-0.15", delta_color="inverse", help="Mean Absolute Error from latest training")
    
    with col5:
        # Check if VM is accessible
        try:
            result = subprocess.run(["ping", "-n", "1", "34.125.123.45"], capture_output=True, timeout=2)
            vm_status = "Online" if result.returncode == 0 else "Offline"
        except:
            vm_status = "Unknown"
        st.metric("VM Status", vm_status, help="Google Cloud VM")
    
    st.divider()
    
    # Infrastructure status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Infrastructure")
        PROJECT_ROOT = Path(__file__).parent.parent
        
        st.markdown("**Local Environment:**")
        st.write(f"- Python: {platform.python_version()}")
        st.write(f"- OS: {platform.system()} {platform.release()}")
        st.write(f"- CPU Usage: {psutil.cpu_percent()}%")
        st.write(f"- RAM Usage: {psutil.virtual_memory().percent}%")
        
        if psutil.disk_usage('/').percent > 80:
            st.warning(f"WARNING Disk usage: {psutil.disk_usage('/').percent}%")
        else:
            st.success(f"Disk usage: {psutil.disk_usage('/').percent}%")
    
    with col2:
        st.markdown("### Data Status")
        
        # Check data files
        original_data = PROJECT_ROOT / "data" / "processed" / "all_runs_combined.parquet"
        augmented_data = PROJECT_ROOT / "data" / "processed" / "all_runs_extreme_augmented.parquet"
        
        if original_data.exists():
            st.markdown("**Original Data:** Available (38 runs)")
        else:
            st.markdown("WARNING **Original Data:** Not found")
        
        if augmented_data.exists():
            st.markdown("**Augmented Data:** Available (1,839 runs)")
        else:
            st.markdown("WARNING **Augmented Data:** Not found")
        
        # Check cache
        cache_dir = PROJECT_ROOT / "cache"
        if cache_dir.exists():
            st.markdown("**Cache:** Ready")
        else:
            st.markdown("WARNING **Cache:** Missing")
    
    with col2:
        st.markdown("### Model Status")
        
        # Check for trained models
        outputs_dir = PROJECT_ROOT / "outputs"
        if outputs_dir.exists():
            model_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("stmgt")]
            st.markdown(f"**Trained Models:** {len(model_dirs)} available")
            
            # Show latest model
            if model_dirs:
                latest_model = max(model_dirs, key=lambda x: x.stat().st_mtime)
                st.markdown(f"**Latest:** `{latest_model.name}`")
                
                # Check best model file
                best_model_path = latest_model / "best_model.pt"
                if best_model_path.exists():
                    st.markdown(f"**Size:** {best_model_path.stat().st_size / 1024 / 1024:.1f} MB")
        else:
            st.markdown("WARNING **No trained models found**")
    
    # Quick actions
    st.divider()
    st.markdown("### Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("View Data", width='stretch'):
            st.switch_page("pages/6_Data_Overview.py")
    
    with col2:
        if st.button("Train Model", width='stretch'):
            st.switch_page("pages/9_Training_Control.py")
    
    with col3:
        if st.button("Predict", width='stretch'):
            st.switch_page("pages/11_Predictions.py")
    
    with col4:
        if st.button("Manage VM", width='stretch'):
            st.switch_page("pages/2_VM_Management.py")

with tab2:
    st.markdown("## STMGT Architecture")
    
    st.markdown("""
    ### Novel Hybrid Design
    
    STMGT combines 4 state-of-the-art architectures:
    
    1. **ASTGCN** - Graph convolution for spatial dependencies
    2. **Transformer** - Self-attention for temporal patterns
    3. **Multi-Modal Fusion** - Weather/temporal integration via cross-attention
    4. **Probabilistic Output** - GMM (K=3) for uncertainty quantification
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Key Features
        - Parallel spatial-temporal encoding
        - Weather-aware cross-attention
        - Hierarchical temporal embeddings
        - Gaussian Mixture Model output (K=3)
        - 212,140 trainable parameters
        """)
    
    with col2:
        st.markdown("""
        #### Performance Metrics
        - **MAE:** 3.05 km/h 
        - **R²:** 0.769 
        - **MAPE:** 22.98% 
        - **CRPS:** 2.84 
        - **Coverage (80%):** 81.2% 
        """)
    
    st.info("**Learn more:** See `docs/STMGT_ARCHITECTURE.md` for detailed architecture documentation")

with tab3:
    st.markdown("## Navigation Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Infrastructure & DevOps")
        st.markdown("""
        **Structure control & system management**
        
        1. **System Overview** (Home)
           - System health at a glance
           - Quick actions & shortcuts
           
        2. **VM Management**NEW
           - Google Cloud VM control
           - Start/stop/restart instances
           - SSH connection management
           - Resource monitoring
           
        3. **Deployment**NEW
           - Git-based deployment
           - Version control
           - Rollback capabilities
           - Deployment history
           
        4. **Monitoring & Logs**NEW
           - System health checks
           - Real-time log streaming
           - Error tracking
           - Performance metrics
        """)
        
        st.markdown("### Data Pipeline")
        st.markdown("""
        **Management of data collection & processing**
        
        5. **Data Collection**NEW
           - Google Maps API control
           - Collection scheduling
           - Progress monitoring
           - Data download from VM
           
        6. **Data Overview**
           - Existing data statistics
           - Quality metrics
           - Data management tools
           
        7. **Data Augmentation**
           - Configuration & execution
           - Strategy comparison
           - Quality validation
        """)
    
    with col2:
        st.markdown("### ML Workflow")
        st.markdown("""
        **Machine Learning pipeline hoàn chỉnh**
        
        8. **Data Visualization**
           - Traffic patterns analysis
           - GMM distributions
           - Network topology
           - Feature correlations
           
        9. **Training Control**
           - Start/stop training
           - Hyperparameter tuning
           - Real-time monitoring
           - Export reports
           
        10. **Model Registry**NEW
            - Version management
            - Model comparison
            - Artifact storage
            - Performance tracking
        """)
        
        st.markdown("### Production")
        st.markdown("""
        **Deployment & inference management**
        
        11. **Predictions**
            - Real-time inference
            - Scenario simulation
            - Uncertainty quantification
            - Alert system
            
        12. **API & Integration**NEW
            - FastAPI endpoints
            - Webhook management
            - API documentation
            - Integration testing
        """)
    
    st.divider()
    
    st.markdown("### Common Workflows")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Development Workflow:**
        1. Page 2: Manage VM → Start instance
        2. Page 5: Collect data from API
        3. Page 6: Verify data quality
        4. Page 7: Run augmentation
        5. Page 9: Train model
        6. Page 10: Register model version
        7. Page 11: Test predictions
        """)
    
    with col2:
        st.markdown("""
        **Production Deployment:**
        1. Page 3: Deploy latest code to VM
        2. Page 4: Check system health
        3. Page 5: Schedule data collection
        4. Page 12: Setup API endpoints
        5. Page 11: Monitor predictions
        6. Page 4: Track logs & metrics
        """)

with tab4:
    st.markdown("## Quick Statistics")
    
    PROJECT_ROOT = Path(__file__).parent.parent
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Data Statistics")
        
        # Count data files
        data_processed = PROJECT_ROOT / "data" / "processed"
        if data_processed.exists():
            parquet_files = list(data_processed.glob("*.parquet"))
            st.metric("Parquet Files", len(parquet_files))
            
            total_size = sum(f.stat().st_size for f in parquet_files) / (1024**3)
            st.metric("Total Size", f"{total_size:.2f} GB")
        
        # Count raw runs
        data_runs = PROJECT_ROOT / "data" / "runs"
        if data_runs.exists():
            run_dirs = [d for d in data_runs.iterdir() if d.is_dir()]
            st.metric("Raw Runs", len(run_dirs))
    
    with col2:
        st.markdown("### Model Statistics")
        
        outputs_dir = PROJECT_ROOT / "outputs"
        if outputs_dir.exists():
            model_dirs = [d for d in outputs_dir.iterdir() if d.is_dir() and d.name.startswith("stmgt")]
            st.metric("Trained Models", len(model_dirs))
            
            if model_dirs:
                total_size = sum(
                    sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                    for d in model_dirs
                ) / (1024**3)
                st.metric("Models Size", f"{total_size:.2f} GB")
        
        # Count predictions
        pred_dir = PROJECT_ROOT / "data" / "predictions"
        if pred_dir.exists():
            pred_files = list(pred_dir.glob("predictions_*.*"))
            st.metric("Prediction Exports", len(pred_files))
    
    with col3:
        st.markdown("### Documentation")
        
        docs_dir = PROJECT_ROOT / "docs"
        if docs_dir.exists():
            md_files = list(docs_dir.glob("*.md"))
            st.metric("Documentation Files", len(md_files))
        
        report_dir = PROJECT_ROOT / "docs" / "report"
        if report_dir.exists():
            reports = list(report_dir.glob("*.html"))
            st.metric("Training Reports", len(reports))
        
        # Count scripts
        scripts_dir = PROJECT_ROOT / "scripts"
        if scripts_dir.exists():
            py_files = list(scripts_dir.rglob("*.py"))
            st.metric("Python Scripts", len(py_files))

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9rem;'>
    <strong>STMGT Dashboard V3</strong> | DSP391m Project | 
    <a href='https://github.com/thatlq1812/dsp391m_project' target='_blank'>GitHub</a> | 
    Built with Streamlit
</div>
""", unsafe_allow_html=True)
